// bfdot_vs_widen.c — fills the missing cell in Sol's 2x2 {faithful,fast}x{NEON,AMX}:
// the FAST NEON path (BFDOT, FEAT_BF16) vs the faithful widen->f32-FMA leaf, on the
// same C[M,N] = A[M,K] * W[N,K]^T contract, same shapes, same parity + cancellation gate.
// Question: does BFDOT get AMX fast32's ~2x WITHOUT leaving the NEON register file?
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static void* am(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)){perror("alloc");exit(1);} return p; }
static uint16_t f2b(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t l=(x>>16)&1u; x+=0x7fffu+l; return (uint16_t)(x>>16); }
static float b2f(uint16_t b){ uint32_t x=(uint32_t)b<<16; float f; memcpy(&f,&x,4); return f; }

static inline float32x4_t wlo(uint16x8_t b){ return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b),16)); }
static inline float32x4_t whi(uint16x8_t b){ return vreinterpretq_f32_u32(vshll_high_n_u16(b,16)); }

// FAITHFUL: widen bf16->f32 (shift), f32 FMA, 4 independent accumulators (decode-fair).
static float dot_widen(const uint16_t*a,const uint16_t*w,int K){
  float32x4_t c0=vdupq_n_f32(0),c1=c0,c2=c0,c3=c0; int k=0;
  for(;k+16<=K;k+=16){
    uint16x8_t w0=vld1q_u16(w+k), a0=vld1q_u16(a+k);
    uint16x8_t w1=vld1q_u16(w+k+8), a1=vld1q_u16(a+k+8);
    c0=vfmaq_f32(c0,wlo(w0),wlo(a0)); c1=vfmaq_f32(c1,whi(w0),whi(a0));
    c2=vfmaq_f32(c2,wlo(w1),wlo(a1)); c3=vfmaq_f32(c3,whi(w1),whi(a1));
  }
  float s=vaddvq_f32(vaddq_f32(vaddq_f32(c0,c1),vaddq_f32(c2,c3)));
  for(;k<K;k++) s+=b2f(a[k])*b2f(w[k]);
  return s;
}
// FAST: native BFDOT (2 bf16 MACs per lane), no widen, 4 accumulators. Its pairwise
// reduction order differs from the widen path -> expected fast-but-unfaithful.
static float dot_bfdot(const uint16_t*a,const uint16_t*w,int K){
  float32x4_t c0=vdupq_n_f32(0),c1=c0,c2=c0,c3=c0; int k=0;
  for(;k+32<=K;k+=32){
    c0=vbfdotq_f32(c0, vreinterpretq_bf16_u16(vld1q_u16(w+k)),    vreinterpretq_bf16_u16(vld1q_u16(a+k)));
    c1=vbfdotq_f32(c1, vreinterpretq_bf16_u16(vld1q_u16(w+k+8)),  vreinterpretq_bf16_u16(vld1q_u16(a+k+8)));
    c2=vbfdotq_f32(c2, vreinterpretq_bf16_u16(vld1q_u16(w+k+16)), vreinterpretq_bf16_u16(vld1q_u16(a+k+16)));
    c3=vbfdotq_f32(c3, vreinterpretq_bf16_u16(vld1q_u16(w+k+24)), vreinterpretq_bf16_u16(vld1q_u16(a+k+24)));
  }
  for(;k+8<=K;k+=8) c0=vbfdotq_f32(c0, vreinterpretq_bf16_u16(vld1q_u16(w+k)), vreinterpretq_bf16_u16(vld1q_u16(a+k)));
  float s=vaddvq_f32(vaddq_f32(vaddq_f32(c0,c1),vaddq_f32(c2,c3)));
  for(;k<K;k++) s+=b2f(a[k])*b2f(w[k]);
  return s;
}
typedef float(*dotfn)(const uint16_t*,const uint16_t*,int);
static void gemm(dotfn f,const uint16_t*A,const uint16_t*W,float*C,int M,int N,int K){
  for(int n=0;n<N;n++){ const uint16_t*w=W+(size_t)n*K;
    for(int m=0;m<M;m++) C[(size_t)m*N+n]=f(A+(size_t)m*K,w,K); }
}

int main(void){
  printf("# BFDOT (fast NEON) vs widen->f32-FMA (faithful NEON) — C[M,N]=A[M,K]*W[N,K]^T\n");
  printf("# GB/s counts 2*N*K checkpoint bytes; best-of-5\n\n");
  // cancellation probe (same idea as Sol's reduction-order gate)
  { int K=2048; uint16_t*a=am(K*2),*w=am(K*2);
    for(int k=0;k<K;k++){ float v=(k%2? -1.f:1.f)*(1.f+k*1e-3f); a[k]=f2b(v); w[k]=f2b(1.f); }
    float rw=dot_widen(a,w,K), rb=dot_bfdot(a,w,K);
    printf("# cancellation probe: widen=%.9g (bf16=%04x)  bfdot=%.9g (bf16=%04x)  %s\n\n",
      rw,f2b(rw),rb,f2b(rb), f2b(rw)==f2b(rb)?"bf16-MATCH":"bf16-DIFFER");
    free(a);free(w); }
  struct{const char*t;int M,N,K;}sh[]={{"decode M=1",1,2048,2048},{"up M=4",4,8192,2048},{"down M=4",4,2048,8192},{"adapter M=7",7,2048,512}};
  for(int s=0;s<4;s++){ int M=sh[s].M,N=sh[s].N,K=sh[s].K;
    uint16_t*A=am((size_t)M*K*2),*W=am((size_t)N*K*2); float*Cw=am((size_t)M*N*4),*Cb=am((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=f2b(((int)(i*2654435761u>>13)%1000-500)/512.f);
    for(size_t i=0;i<(size_t)N*K;i++) W[i]=f2b(((int)(i*40503u+7)%1000-500)/512.f);
    double wb=2.0*N*K; int it=wb>16e6?20:80;
    // parity: bfdot vs widen (the faithful reference)
    gemm(dot_widen,A,W,Cw,M,N,K); gemm(dot_bfdot,A,W,Cb,M,N,K);
    int bfmis=0; double maxabs=0; for(int i=0;i<M*N;i++){ if(f2b(Cw[i])!=f2b(Cb[i]))bfmis++; double d=fabs(Cw[i]-Cb[i]); if(d>maxabs)maxabs=d; }
    double bw=1e30,bb=1e30;
    for(int r=0;r<5;r++){ double t=now_s(); for(int i=0;i<it;i++) gemm(dot_widen,A,W,Cw,M,N,K); t=now_s()-t; if(t/it<bw)bw=t/it; }
    for(int r=0;r<5;r++){ double t=now_s(); for(int i=0;i<it;i++) gemm(dot_bfdot,A,W,Cb,M,N,K); t=now_s()-t; if(t/it<bb)bb=t/it; }
    printf("== %s N=%d K=%d ==\n",sh[s].t,N,K);
    printf("  NEON widen (faithful)  median=%7.3f ms  %6.1f GB/s\n",bw*1e3,wb/bw/1e9);
    printf("  NEON BFDOT (fast)      median=%7.3f ms  %6.1f GB/s   (%.2fx)\n",bb*1e3,wb/bb/1e9,bw/bb);
    printf("  parity bfdot vs widen: bf16-diff=%d/%d  max-abs=%.3e\n\n",bfmis,M*N,maxabs);
    free(A);free(W);free(Cw);free(Cb);
  }
  return 0;
}
