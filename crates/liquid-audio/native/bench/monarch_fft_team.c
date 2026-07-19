// monarch_fft_team.c — the fused Monarch FFT running ON the substrate: the
// flashkern fixed lane team, driven by the real kcoro kc_port doorbell (not a
// re-invented scheduler). A batch of signals is fanned across the lanes by
// atomic tile-claim; each lane runs the register-resident bf16 Monarch butterfly
// (the parity-gated leaf from monarch_fft_probe.c) on its claimed signals.
//
// The point: the decode GEMV was bandwidth-bound and plateaued at ~3.7x on 8
// lanes (pipeline_probe). The Monarch FFT is compute-bound and L1-resident per
// signal (the [N1,N2] plane + the shared constant DFT tables), so it should scale
// near-linearly — the workload the fixed team was built for.
//
// Build (from this directory):
//   KA=../../../kcoro-sys/vendor/kcoro_arena
//   clang -O3 -ffp-contract=off -march=armv8.6-a+bf16 monarch_fft_team.c \
//     "$KA/port/posix.c" -I"$KA/include" -I"$KA/port" -lm -o /tmp/mft && /tmp/mft
#include "kc_port.h"
#include <arm_neon.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#define N1 32
#define N2 32
#define NT (N1*N2)
static const double PI = 3.14159265358979323846;
static uint16_t f2b(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t l=(x>>16)&1u; x+=0x7fffu+l; return (uint16_t)(x>>16); }
static void* am(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)){perror("alloc");exit(1);} return p; }

// ---- shared constant DFT tables (built once, read-only by every lane) --------
static float twr[N1][N2], twi[N1][N2];
static uint16_t bd1r[N1][N1], bd1i[N1][N1], bd2r[N2][N2], bd2i[N2][N2];
static void build_tables(void){
  for(int j=0;j<N1;j++) for(int k=0;k<N1;k++){ double a=2*PI*k*j/N1; bd1r[j][k]=f2b((float)cos(a)); bd1i[j][k]=f2b((float)-sin(a)); }
  for(int j=0;j<N2;j++) for(int k=0;k<N2;k++){ double a=2*PI*k*j/N2; bd2r[j][k]=f2b((float)cos(a)); bd2i[j][k]=f2b((float)-sin(a)); }
  for(int j=0;j<N1;j++) for(int k=0;k<N2;k++){ double a=2*PI*j*k/NT; twr[j][k]=(float)cos(a); twi[j][k]=(float)-sin(a); }
}
static inline float bfdot32(const uint16_t* a,const uint16_t* b,int n){
  float32x4_t acc=vdupq_n_f32(0); int k=0;
  for(;k+8<=n;k+=8) acc=vbfdotq_f32(acc,vreinterpretq_bf16_u16(vld1q_u16(a+k)),vreinterpretq_bf16_u16(vld1q_u16(b+k)));
  return vaddvq_f32(acc);
}
// one register-resident fused Monarch FFT (verified in monarch_fft_probe.c)
static void monarch_bf16(const float* xflat, float* Xr, float* Xi){
  uint16_t x2[N1][N2];
  for(int k1=0;k1<N1;k1++) for(int k2=0;k2<N2;k2++) x2[k1][k2]=f2b(xflat[k2*N1+k1]);
  uint16_t ZTr[N2][N1], ZTi[N2][N1];
  for(int k1=0;k1<N1;k1++) for(int j2=0;j2<N2;j2++){
    float sr=bfdot32(x2[k1],bd2r[j2],N2), si=bfdot32(x2[k1],bd2i[j2],N2);
    ZTr[j2][k1]=f2b(sr*twr[k1][j2]-si*twi[k1][j2]); ZTi[j2][k1]=f2b(sr*twi[k1][j2]+si*twr[k1][j2]); }
  for(int j2=0;j2<N2;j2++) for(int j1=0;j1<N1;j1++){
    float rr=bfdot32(bd1r[j1],ZTr[j2],N1), ii=bfdot32(bd1i[j1],ZTi[j2],N1);
    float ri=bfdot32(bd1r[j1],ZTi[j2],N1), ir=bfdot32(bd1i[j1],ZTr[j2],N1);
    Xr[j1*N2+j2]=rr-ii; Xi[j1*N2+j2]=ri+ir; }
}

// ---- flashkern lane team on the real kc_port doorbell (cache-line-isolated) --
typedef struct {
  int workers;
  const float *xin; float *Xr, *Xi; int nsig, sig_tile;   // work: nsig signals, xin[nsig*NT]
  _Alignas(128) atomic_int tile_next;
  _Alignas(128) atomic_int completed;
  _Alignas(128) uint32_t dispatch_val;
  _Alignas(128) uint32_t done_val;
  kc_port_wait_word *dispatch, *done;
  _Alignas(128) atomic_int stop;
} Engine;

static void run_tiles(Engine*e){
  int nt=(e->nsig+e->sig_tile-1)/e->sig_tile;
  for(;;){
    int t=atomic_fetch_add_explicit(&e->tile_next,1,memory_order_relaxed);
    if(t>=nt) break;
    int lo=t*e->sig_tile, hi=lo+e->sig_tile; if(hi>e->nsig)hi=e->nsig;
    for(int s=lo;s<hi;s++) monarch_bf16(e->xin+(size_t)s*NT, e->Xr+(size_t)s*NT, e->Xi+(size_t)s*NT);
  }
}
static void* worker_main(void*arg){
  Engine*e=(Engine*)arg;
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
  uint32_t last=0;
  for(;;){
    uint32_t cur=__atomic_load_n(&e->dispatch_val,__ATOMIC_ACQUIRE);
    if(atomic_load_explicit(&e->stop,memory_order_acquire)) break;
    if(cur==last){ kc_port_wait_u32(e->dispatch,cur,0); continue; }
    last=cur; run_tiles(e);
    if(atomic_fetch_add_explicit(&e->completed,1,memory_order_acq_rel)+1==e->workers){
      __atomic_add_fetch(&e->done_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e->done); }
  }
  return NULL;
}
static void dispatch_gen(Engine*e){
  atomic_store_explicit(&e->tile_next,0,memory_order_relaxed);
  atomic_store_explicit(&e->completed,0,memory_order_relaxed);
  uint32_t dobs=__atomic_load_n(&e->done_val,__ATOMIC_ACQUIRE);
  __atomic_add_fetch(&e->dispatch_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e->dispatch);
  while(__atomic_load_n(&e->done_val,__ATOMIC_ACQUIRE)==dobs) kc_port_wait_u32(e->done,dobs,0);
}
static double bench(Engine*e,int gens){
  dispatch_gen(e); dispatch_gen(e); double best=1e30;
  for(int r=0;r<5;r++){ uint64_t t0=kc_port_monotonic_ns();
    for(int g=0;g<gens;g++) dispatch_gen(e);
    double dt=(kc_port_monotonic_ns()-t0)/1e9/gens; if(dt<best)best=dt; }
  return best;
}

int main(void){
  build_tables();
  const int S=512;
  float *xin=am((size_t)S*NT*4), *Xr=am((size_t)S*NT*4), *Xi=am((size_t)S*NT*4);
  for(size_t i=0;i<(size_t)S*NT;i++) xin[i]=((int)((i*2654435761u>>13))%2000-1000)/1024.f;
  double flops_per_fft=2.0*(2.0*N1*N2*N2 + 4.0*N1*N2*N1);   // 6 real GEMMs
  printf("# Monarch FFT on the kc_port lane team — %d signals x %d-pt FFT/generation\n",S,NT);
  printf("# compute-bound, L1-resident per signal: expect near-linear scaling\n\n");
  printf("  workers | ms/gen | FFTs/s (M) | GFLOP/s | speedup\n");
  printf("  --------+--------+------------+---------+--------\n");
  double one=0;
  for(int Wk=1;Wk<=8;Wk*=2){
    Engine e; memset(&e,0,sizeof e);
    e.workers=Wk; e.xin=xin; e.Xr=Xr; e.Xi=Xi; e.nsig=S; e.sig_tile=8;
    atomic_init(&e.tile_next,0); atomic_init(&e.completed,0); atomic_init(&e.stop,0);
    kc_port_wait_u32_prepare(&e.dispatch_val,&e.dispatch); kc_port_wait_u32_prepare(&e.done_val,&e.done);
    kc_port_thread *th[8]; for(int i=0;i<Wk;i++) kc_port_thread_create(&th[i],worker_main,&e);
    double spg=bench(&e,400);
    if(Wk==1) one=spg;
    printf("  %4d    | %5.3f  |   %6.2f   | %7.1f | %.2fx\n",
      Wk, spg*1e3, S/spg/1e6, flops_per_fft*S/spg/1e9, one/spg);
    atomic_store_explicit(&e.stop,1,memory_order_release);
    __atomic_add_fetch(&e.dispatch_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e.dispatch);
    for(int i=0;i<Wk;i++) kc_port_thread_join(th[i]);
    kc_port_wait_u32_release(e.dispatch); kc_port_wait_u32_release(e.done);
  }
  free(xin);free(Xr);free(Xi);
  return 0;
}
