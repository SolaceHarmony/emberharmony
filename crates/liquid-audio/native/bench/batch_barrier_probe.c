// batch_barrier_probe.c — the lever pipeline_probe pointed at: does batching M
// tokens at the barrier escape the weight-bandwidth ceiling? Same kc_port
// doorbell fixed team, but each output tile computes M rows (tokens) against a
// weight row loaded ONCE and reused across all M. If the design holds, weight
// GB/s stays pinned near the single-token ceiling while effective GFLOP/s climbs
// ~M× — i.e. the same bandwidth serves M concurrent conversations.
//
// Build (from this directory):
//   KA=../../../kcoro-sys/vendor/kcoro_arena
//   clang -O3 -ffp-contract=off -march=armv8.6-a+bf16 \
//     batch_barrier_probe.c "$KA/port/posix.c" -I"$KA/include" -I"$KA/port" -o /tmp/batch_barrier_probe
//   /tmp/batch_barrier_probe
#include "kc_port.h"
#include <arm_neon.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#define MMAX 16
static uint16_t f2b(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t l=(x>>16)&1u; x+=0x7fffu+l; return (uint16_t)(x>>16); }
static void* am(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)){perror("alloc");exit(1);} return p; }

// batched tile: for output row n, C[m,n] = A[m,:] . W[n,:] for all m in [0,M).
// W[n] chunk is loaded once per k-step and reused across the M activation rows.
static void row_batch(const uint16_t*wn,const uint16_t*A,float*C,int m0n_stride_unused,
                      int M,int N,int K,int n){
  (void)m0n_stride_unused;
  float32x4_t acc[MMAX]; for(int m=0;m<M;m++) acc[m]=vdupq_n_f32(0);
  int k=0;
  for(;k+8<=K;k+=8){
    uint16x8_t w=vld1q_u16(wn+k);                 // <-- loaded ONCE, reused across M
    bfloat16x8_t wb=vreinterpretq_bf16_u16(w);
    for(int m=0;m<M;m++)
      acc[m]=vbfdotq_f32(acc[m],wb,vreinterpretq_bf16_u16(vld1q_u16(A+(size_t)m*K+k)));
  }
  for(int m=0;m<M;m++) C[(size_t)m*N+n]=vaddvq_f32(acc[m]);
}

typedef struct {                       // contended words on separate 128-byte lines
  int workers,M,N,K,tile_rows;
  const uint16_t *W,*A; float *C;
  _Alignas(128) atomic_int tile_next;
  _Alignas(128) atomic_int completed;
  _Alignas(128) uint32_t dispatch_val;
  _Alignas(128) uint32_t done_val;
  kc_port_wait_word *dispatch,*done;
  _Alignas(128) atomic_int stop;
} Engine;

static void run_tiles(Engine*e){
  int nt=(e->N+e->tile_rows-1)/e->tile_rows;
  for(;;){
    int t=atomic_fetch_add_explicit(&e->tile_next,1,memory_order_relaxed);
    if(t>=nt) break;
    int lo=t*e->tile_rows, hi=lo+e->tile_rows; if(hi>e->N)hi=e->N;
    for(int n=lo;n<hi;n++) row_batch(e->W+(size_t)n*e->K,e->A,e->C,0,e->M,e->N,e->K,n);
  }
}
static void* worker_main(void*arg){
  Engine*e=(Engine*)arg;
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);   // bias to P-cores (macOS-only)
#endif
  uint32_t last=0;
  for(;;){
    uint32_t cur=__atomic_load_n(&e->dispatch_val,__ATOMIC_ACQUIRE);
    if(atomic_load_explicit(&e->stop,memory_order_acquire)) break;
    if(cur==last){ kc_port_wait_u32(e->dispatch,cur,0); continue; }
    last=cur; run_tiles(e);
    if(atomic_fetch_add_explicit(&e->completed,1,memory_order_acq_rel)+1==e->workers){
      __atomic_add_fetch(&e->done_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e->done);
    }
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
  const int N=2048,K=2048,TR=64,WK=8;
  size_t wb=(size_t)N*K*2;
  uint16_t*W=am(wb),*A=am((size_t)MMAX*K*2); float*C=am((size_t)MMAX*N*4);
  for(size_t i=0;i<(size_t)N*K;i++) W[i]=f2b(((int)(i*2654435761u>>13)%1000-500)/512.f);
  for(size_t i=0;i<(size_t)MMAX*K;i++) A[i]=f2b(((int)(i*40503u+7)%1000-500)/512.f);
  printf("# batch-at-the-barrier: %d-worker kc_port team, decode N=%d K=%d, weight=%.1f MiB BF16\n",WK,N,K,wb/1048576.0);
  printf("# W read once per generation regardless of M -> weight GB/s should stay ~flat while GFLOP/s climbs\n\n");
  printf("  M(batch) | ms/gen | weight GB/s | GFLOP/s | GFLOP/s vs M=1\n");
  printf("  ---------+--------+-------------+---------+--------------\n");
  double gbs1=0, gfl1=0;
  for(int M=1;M<=MMAX;M*=2){
    Engine e; memset(&e,0,sizeof e);
    e.workers=WK; e.M=M; e.N=N; e.K=K; e.tile_rows=TR; e.W=W; e.A=A; e.C=C;
    atomic_init(&e.tile_next,0); atomic_init(&e.completed,0); atomic_init(&e.stop,0);
    kc_port_wait_u32_prepare(&e.dispatch_val,&e.dispatch); kc_port_wait_u32_prepare(&e.done_val,&e.done);
    kc_port_thread *th[8]; for(int i=0;i<WK;i++) kc_port_thread_create(&th[i],worker_main,&e);
    double spg=bench(&e,400);
    double gbs=wb/spg/1e9, gfl=2.0*M*N*K/spg/1e9;
    if(M==1){ gbs1=gbs; gfl1=gfl; }
    printf("  %6d   | %5.3f  |   %7.1f   | %7.1f | %5.2fx\n",
           M,spg*1e3,gbs, gfl, gfl/gfl1);
    atomic_store_explicit(&e.stop,1,memory_order_release);
    __atomic_add_fetch(&e.dispatch_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e.dispatch);
    for(int i=0;i<WK;i++) kc_port_thread_join(th[i]);
    kc_port_wait_u32_release(e.dispatch); kc_port_wait_u32_release(e.done);
  }
  printf("\n# if weight GB/s stays ~%.0f while GFLOP/s scales with M, batching beats the bandwidth ceiling.\n",gbs1);
  return 0;
}
