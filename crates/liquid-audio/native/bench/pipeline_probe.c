// pipeline_probe.c — the scheduling spine, for real: a flashkern-style fixed
// lane team parked on the actual in-repo kc_port doorbell (zero-spin), fanning a
// decode projection out by atomic tile-claim, running BFDOT leaves. It links the
// real kc_port (not kc_team, which is mid-migration) so the wake path is honest.
// Measures:
//   (1) lane scaling 1->8 on a real decode layer (linear, or bandwidth-capped?)
//   (2) per-generation orchestration overhead (dispatch + wake + barrier) — the
//       cost the doorbell round-trip adds on top of raw compute.
//
// Build (from this directory):
//   KA=../../../kcoro-sys/vendor/kcoro_arena
//   clang -O3 -ffp-contract=off -march=armv8.6-a+bf16 \
//     pipeline_probe.c "$KA/port/posix.c" -I"$KA/include" -I"$KA/port" -o /tmp/pipeline_probe
//   /tmp/pipeline_probe            # measured results on this M2 Max in bench/README.md
#include "kc_port.h"
#include <arm_neon.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread/qos.h>

static uint16_t f2b(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t l=(x>>16)&1u; x+=0x7fffu+l; return (uint16_t)(x>>16); }
static void* am(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)){perror("alloc");exit(1);} return p; }

// decode BFDOT leaf: y[n] = sum_k W[n,k]*x[k]
static float dot_bfdot(const uint16_t*w,const uint16_t*x,int K){
  float32x4_t c0=vdupq_n_f32(0),c1=c0,c2=c0,c3=c0; int k=0;
  for(;k+32<=K;k+=32){
    c0=vbfdotq_f32(c0,vreinterpretq_bf16_u16(vld1q_u16(w+k)),   vreinterpretq_bf16_u16(vld1q_u16(x+k)));
    c1=vbfdotq_f32(c1,vreinterpretq_bf16_u16(vld1q_u16(w+k+8)), vreinterpretq_bf16_u16(vld1q_u16(x+k+8)));
    c2=vbfdotq_f32(c2,vreinterpretq_bf16_u16(vld1q_u16(w+k+16)),vreinterpretq_bf16_u16(vld1q_u16(x+k+16)));
    c3=vbfdotq_f32(c3,vreinterpretq_bf16_u16(vld1q_u16(w+k+24)),vreinterpretq_bf16_u16(vld1q_u16(x+k+24)));
  }
  for(;k+8<=K;k+=8) c0=vbfdotq_f32(c0,vreinterpretq_bf16_u16(vld1q_u16(w+k)),vreinterpretq_bf16_u16(vld1q_u16(x+k)));
  return vaddvq_f32(vaddq_f32(vaddq_f32(c0,c1),vaddq_f32(c2,c3)));
}

typedef struct {
  int workers;
  const uint16_t *W,*x; float *y; int N,K,tile_rows;
  atomic_int tile_next, completed;
  uint32_t dispatch_val, done_val;      // doorbell addresses (raw u32, acquire/release accessed)
  kc_port_wait_word *dispatch, *done;
  atomic_int stop;
} Engine;

static void run_tiles(Engine*e){
  int nt=(e->N+e->tile_rows-1)/e->tile_rows;
  for(;;){
    int t=atomic_fetch_add_explicit(&e->tile_next,1,memory_order_relaxed);
    if(t>=nt) break;
    int lo=t*e->tile_rows, hi=lo+e->tile_rows; if(hi>e->N)hi=e->N;
    for(int n=lo;n<hi;n++) e->y[n]=dot_bfdot(e->W+(size_t)n*e->K,e->x,e->K);
  }
}

static void* worker_main(void*arg){
  Engine*e=(Engine*)arg;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);   // bias to P-cores
  uint32_t last=0;
  for(;;){
    uint32_t cur=__atomic_load_n(&e->dispatch_val,__ATOMIC_ACQUIRE);
    if(atomic_load_explicit(&e->stop,memory_order_acquire)) break;
    if(cur==last){ kc_port_wait_u32(e->dispatch,cur,0); continue; }  // no new gen: park (zero-spin)
    last=cur;
    run_tiles(e);
    if(atomic_fetch_add_explicit(&e->completed,1,memory_order_acq_rel)+1==e->workers){
      __atomic_add_fetch(&e->done_val,1,__ATOMIC_RELEASE);
      kc_port_wake_u32_all(e->done);
    }
  }
  return NULL;
}

// one generation: dispatch -> team claims+computes -> barrier -> resume
static void dispatch_gen(Engine*e){
  atomic_store_explicit(&e->tile_next,0,memory_order_relaxed);
  atomic_store_explicit(&e->completed,0,memory_order_relaxed);
  uint32_t dobs=__atomic_load_n(&e->done_val,__ATOMIC_ACQUIRE);
  __atomic_add_fetch(&e->dispatch_val,1,__ATOMIC_RELEASE);
  kc_port_wake_u32_all(e->dispatch);
  while(__atomic_load_n(&e->done_val,__ATOMIC_ACQUIRE)==dobs)
    kc_port_wait_u32(e->done,dobs,0);
}

static double bench(Engine*e,int gens){
  dispatch_gen(e); dispatch_gen(e);                 // warm
  double best=1e30;
  for(int r=0;r<5;r++){
    uint64_t t0=kc_port_monotonic_ns();
    for(int g=0;g<gens;g++) dispatch_gen(e);
    double dt=(kc_port_monotonic_ns()-t0)/1e9/gens;
    if(dt<best)best=dt;
  }
  return best;                                       // seconds per generation
}

int main(void){
  printf("# scheduling spine: kc_port doorbell fixed team + atomic tile-claim + BFDOT\n");
  printf("# cpus reported: %u\n\n",kc_port_cpu_count());
  const int N=2048,K=2048,TR=64;
  size_t wb=(size_t)N*K*2;
  uint16_t*W=am(wb),*x=am((size_t)K*2); float*y=am((size_t)N*4);
  for(size_t i=0;i<(size_t)N*K;i++) W[i]=f2b(((int)(i*2654435761u>>13)%1000-500)/512.f);
  for(int k=0;k<K;k++) x[k]=f2b(((k*40503+7)%1000-500)/512.f);

  printf("== decode layer  N=%d K=%d  W=%.1f MiB (BF16), tile=%d rows ==\n",N,K,wb/1048576.0,TR);
  double one=0;
  for(int Wk=1;Wk<=8;Wk*=2){
    Engine e; memset(&e,0,sizeof e);
    e.workers=Wk; e.W=W; e.x=x; e.y=y; e.N=N; e.K=K; e.tile_rows=TR;
    atomic_init(&e.tile_next,0); atomic_init(&e.completed,0); atomic_init(&e.stop,0);
    e.dispatch_val=0; e.done_val=0;
    if(kc_port_wait_u32_prepare(&e.dispatch_val,&e.dispatch)||kc_port_wait_u32_prepare(&e.done_val,&e.done)){printf("prepare failed\n");return 1;}
    kc_port_thread *th[8];
    for(int i=0;i<Wk;i++) kc_port_thread_create(&th[i],worker_main,&e);
    double spg=bench(&e,400);
    if(Wk==1) one=spg;
    double gbs=wb/spg/1e9;
    printf("  workers=%d  %7.3f ms/gen  %6.1f GB/s  speedup=%.2fx (of %d)\n",Wk,spg*1e3,gbs,one/spg,Wk);
    // teardown
    atomic_store_explicit(&e.stop,1,memory_order_release);
    __atomic_add_fetch(&e.dispatch_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e.dispatch);
    for(int i=0;i<Wk;i++) kc_port_thread_join(th[i]);
    kc_port_wait_u32_release(e.dispatch); kc_port_wait_u32_release(e.done);
  }

  // orchestration overhead: tiny work isolates dispatch+wake+barrier round-trip
  printf("\n== orchestration overhead (tiny 1-tile work) — the doorbell round-trip ==\n");
  for(int Wk=1;Wk<=8;Wk*=2){
    Engine e; memset(&e,0,sizeof e);
    e.workers=Wk; e.W=W; e.x=x; e.y=y; e.N=8; e.K=8; e.tile_rows=8;   // ~nothing to compute
    atomic_init(&e.tile_next,0); atomic_init(&e.completed,0); atomic_init(&e.stop,0);
    kc_port_wait_u32_prepare(&e.dispatch_val,&e.dispatch); kc_port_wait_u32_prepare(&e.done_val,&e.done);
    kc_port_thread *th[8];
    for(int i=0;i<Wk;i++) kc_port_thread_create(&th[i],worker_main,&e);
    double spg=bench(&e,20000);
    printf("  workers=%d  %6.2f us/gen  (dispatch %d + barrier + completion wake)\n",Wk,spg*1e6,Wk);
    atomic_store_explicit(&e.stop,1,memory_order_release);
    __atomic_add_fetch(&e.dispatch_val,1,__ATOMIC_RELEASE); kc_port_wake_u32_all(e.dispatch);
    for(int i=0;i<Wk;i++) kc_port_thread_join(th[i]);
    kc_port_wait_u32_release(e.dispatch); kc_port_wait_u32_release(e.done);
  }
  return 0;
}
