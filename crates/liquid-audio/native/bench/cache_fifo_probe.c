// cache_fifo_probe.c — the register -> L1 -> L2 -> DRAM FIFO test we never ran.
// One identical K-stage arithmetic chain, two residency forms, swept across the
// cache hierarchy:
//   FUSED        : read a tile, run all K stages in registers, write once.
//                  Memory traffic = 2*S (read input + write output), independent of K.
//   MATERIALIZED : one pass per stage, each round-tripping the whole S-plane.
//                  Memory traffic ~= 2*K*S (the "convenience planes" anti-pattern).
// Both compute the SAME result (K affine FMA stages; -ffp-contract=off forbids
// the compiler from reassociating them into one, so all K are really emitted).
// The question: where does keeping the chain register-resident start to matter?
//
// Build and run (from this directory):
//   clang -O3 -ffp-contract=off -march=armv8.6-a cache_fifo_probe.c -o /tmp/cache_fifo_probe
//   /tmp/cache_fifo_probe
// Verify no vector spills in the fused hot loop:
//   clang -O3 -ffp-contract=off -march=armv8.6-a -S cache_fifo_probe.c -o - | \
//     sed -n '/_fused_chain:/,/ret/p' | grep -iE 'str .*\[sp|ldr .*\[sp'   # want: empty
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define K 8
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void* am(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)){perror("alloc");exit(1);} return p; }

static float C[K], D[K];   // stage coefficients (runtime, non-constant-foldable)
static volatile float g_sink = 0.f;   // observed at exit so the stores can't be DCE'd
// Leaves assume S % 4 == 0 (all swept sizes are powers of two >= 4096).

// FUSED: each 4-lane chunk flows through all K stages in registers, one store.
__attribute__((noinline))
static void fused_chain(const float* in, float* out, size_t S){
  for(size_t i=0;i<S;i+=4){
    float32x4_t x=vld1q_f32(in+i);
    for(int s=0;s<K;s++) x=vfmaq_n_f32(vdupq_n_f32(D[s]),x,C[s]); // x = x*C[s]+D[s]
    vst1q_f32(out+i,x);
  }
}
// MATERIALIZED: stage 0 reads in -> out; stages 1..K-1 read+write out in place.
__attribute__((noinline))
static void materialized_chain(const float* in, float* out, size_t S){
  for(size_t i=0;i<S;i+=4)
    vst1q_f32(out+i, vfmaq_n_f32(vdupq_n_f32(D[0]),vld1q_f32(in+i),C[0]));
  for(int s=1;s<K;s++)
    for(size_t i=0;i<S;i+=4)
      vst1q_f32(out+i, vfmaq_n_f32(vdupq_n_f32(D[s]),vld1q_f32(out+i),C[s]));
}

static double bench(void(*fn)(const float*,float*,size_t),const float*in,float*out,size_t S,int it){
  fn(in,out,S); fn(in,out,S); double best=1e30;
  for(int r=0;r<7;r++){ double t=now_s(); for(int i=0;i<it;i++) fn(in,out,S); t=(now_s()-t)/it; if(t<best)best=t; }
  return best;
}

int main(void){
  for(int s=0;s<K;s++){ C[s]=0.99f+0.001f*s; D[s]=0.0007f*(s+1); }   // bounded chain
  printf("# register/L1/L2/DRAM FIFO: %d-stage chain, FUSED (regs, 2S traffic) vs\n",K);
  printf("# MATERIALIZED (%d plane round-trips, ~2*%d*S traffic). single thread.\n",K,K);
  printf("# M2 Max: L1d 128 KiB/core, L2 16 MiB/cluster.  useful GB/s = 2*S*4 / time.\n\n");
  printf("  working set |   level  | fused ms | matzd ms | matzd/fused | fused useful GB/s\n");
  printf("  ------------+----------+----------+----------+-------------+------------------\n");
  size_t elems[]={4096,16384,32768,131072,1u<<20,4u<<20,16u<<20,64u<<20};
  const char* lvl[]={"L1","L1","L1~edge","L2","L2","L2~edge","DRAM","DRAM"};
  for(int e=0;e<8;e++){
    size_t S=elems[e]; size_t bytes=S*4;
    float*in=am(bytes),*out=am(bytes);
    for(size_t i=0;i<S;i++) in[i]=((int)(i*2654435761u>>15)%2000-1000)/1024.f;
    int it = bytes< (1u<<20) ? 2000 : (bytes<(16u<<20)?200:20);
    double tf=bench(fused_chain,in,out,S,it);
    double tm=bench(materialized_chain,in,out,S,it);
    double gbs=2.0*bytes/tf/1e9;
    printf("  %6.2f %-4s | %-8s | %8.4f | %8.4f |   %5.2fx    |  %7.1f\n",
      bytes>=1048576? bytes/1048576.0:bytes/1024.0, bytes>=1048576?"MiB":"KiB",
      lvl[e], tf*1e3, tm*1e3, tm/tf, gbs);
    g_sink += out[0] + out[S/2] + out[S-1];   // force the writes to be observed
    free(in);free(out);
  }
  printf("\n# read: where matzd/fused ~= 1, the chain is compute-bound and fusion is moot;\n");
  printf("# where it climbs toward %d, materialized is paying K-times the memory traffic --\n",K);
  printf("# that boundary is where the register/cache FIFO stops being optional.\n");
  return 0;
}
