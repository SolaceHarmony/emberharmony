// amx_vs_neon.c — microbenchmark: register-resident NEON vs Accelerate/AMX for
// the flashkern GEMV (decode, M=1) and GEMM (prefill, M>1) regimes on Apple M2.
//
// Measures the four facts the register-resident kernel design rests on:
//   1. GEMV throughput is memory-bound. Once the weights exceed the ~16 MB L2,
//      the NEON path and Accelerate/AMX converge on the single-core DRAM ceiling,
//      so AMX's compute advantage only pays when weights fit L2 (small) or the
//      batch is large (prefill) — the seam is L2-residency, not M=1 vs M>1.
//   2. Independent accumulators (x4) break the RAW FMA chain and expose ILP.
//   3. A fused BF16 round-to-nearest-even epilogue costs nothing over the plain
//      GEMV (one write); the AMX path pays a separate epilogue pass (round-trip).
//   4. Reordering the reduction for ILP perturbs the F32 result — keep the
//      rounding point and reduction order pinned where the numerics are validated.
//
// Build: clang -O3 -ffp-contract=off amx_vs_neon.c -framework Accelerate -o amx_vs_neon
// Run:   ./amx_vs_neon        (measured results for M2 Max in bench/README.md)
#define ACCELERATE_NEW_LAPACK 1   // non-deprecated cblas interface (32-bit ints)
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static void* amalloc(size_t n){ void*p=NULL; if(posix_memalign(&p,128,n)) { perror("posix_memalign"); exit(1);} return p; }

// f32 -> bf16 round-to-nearest-even (the LFM_RNE_WIDEN / w14 macro, in C)
static inline uint16_t f32_to_bf16_rne(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t lsb=(x>>16)&1u, bias=0x7fffu+lsb;
    x+=bias; return (uint16_t)(x>>16);
}

// ---- GEMV kernels: y[N] = W[N x K] * x[K], row-major -----------------------
static void gemv_scalar(const float*W,const float*x,float*y,int N,int K){
  for(int n=0;n<N;n++){ const float*w=W+(size_t)n*K; float s=0.f;
    for(int k=0;k<K;k++) s+=w[k]*x[k]; y[n]=s; }
}
// one accumulator: serial FMA chain (RAW-latency-bound if not memory-bound)
static void gemv_neon1(const float*W,const float*x,float*y,int N,int K){
  for(int n=0;n<N;n++){ const float*w=W+(size_t)n*K; float32x4_t a=vdupq_n_f32(0);
    int k=0; for(;k+4<=K;k+=4) a=vfmaq_f32(a,vld1q_f32(w+k),vld1q_f32(x+k));
    float s=vaddvq_f32(a); for(;k<K;k++) s+=w[k]*x[k]; y[n]=s; }
}
// four independent accumulators: breaks the RAW chain, exposes ILP
static inline float dot4acc(const float*w,const float*x,int K){
  float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0; int k=0;
  for(;k+16<=K;k+=16){
    a0=vfmaq_f32(a0,vld1q_f32(w+k),   vld1q_f32(x+k));
    a1=vfmaq_f32(a1,vld1q_f32(w+k+4), vld1q_f32(x+k+4));
    a2=vfmaq_f32(a2,vld1q_f32(w+k+8), vld1q_f32(x+k+8));
    a3=vfmaq_f32(a3,vld1q_f32(w+k+12),vld1q_f32(x+k+12));
  }
  float s=vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3)));
  for(;k<K;k++) s+=w[k]*x[k]; return s;
}
static void gemv_neon4(const float*W,const float*x,float*y,int N,int K){
  for(int n=0;n<N;n++) y[n]=dot4acc(W+(size_t)n*K,x,K);
}
// register-resident 4-acc + FUSED BF16 RNE epilogue: rounds in-register, one write
static void gemv_neon4_bf16(const float*W,const float*x,uint16_t*yb,int N,int K){
  for(int n=0;n<N;n++) yb[n]=f32_to_bf16_rne(dot4acc(W+(size_t)n*K,x,K));
}
// Accelerate (AMX) + SEPARATE bf16 epilogue = the round-trip model
static void gemv_accel_bf16(const float*W,const float*x,float*ytmp,uint16_t*yb,int N,int K){
  cblas_sgemv(CblasRowMajor,CblasNoTrans,N,K,1.0f,W,K,x,1,0.0f,ytmp,1); // AMX writes plane
  for(int n=0;n<N;n++) yb[n]=f32_to_bf16_rne(ytmp[n]);                   // epilogue reads plane
}

static double bench(void(*fn)(void*),void*ctx,int iters){
  fn(ctx); fn(ctx); // warmup
  double best=1e30;
  for(int r=0;r<5;r++){ double t=now_s(); for(int i=0;i<iters;i++) fn(ctx); t=now_s()-t; if(t<best)best=t; }
  return best/iters;
}
// thunks
typedef struct{const float*W,*x;float*y;uint16_t*yb;float*ytmp;int N,K;} Ctx;
static void t_scalar(void*c){Ctx*x=c;gemv_scalar(x->W,x->x,x->y,x->N,x->K);}
static void t_neon1 (void*c){Ctx*x=c;gemv_neon1 (x->W,x->x,x->y,x->N,x->K);}
static void t_neon4 (void*c){Ctx*x=c;gemv_neon4 (x->W,x->x,x->y,x->N,x->K);}
static void t_n4bf16(void*c){Ctx*x=c;gemv_neon4_bf16(x->W,x->x,x->yb,x->N,x->K);}
static void t_accel (void*c){Ctx*x=c;gemv_accel_bf16(x->W,x->x,x->ytmp,x->yb,x->N,x->K);}

int main(void){
  printf("# GEMV (decode, M=1): y = W[N x K] * x  — best-of-5, aligned-128\n");
  printf("# metric: GB/s = weight bytes (N*K*4) read per call; GFLOP/s = 2*N*K\n\n");
  int shapes[][2]={{2048,2048},{4096,4096},{2048,8192},{8192,2048}};
  for(int s=0;s<4;s++){
    int N=shapes[s][0],K=shapes[s][1];
    size_t wb=(size_t)N*K*4;
    float*W=amalloc(wb),*x=amalloc((size_t)K*4),*y=amalloc((size_t)N*4),*yr=amalloc((size_t)N*4),*ytmp=amalloc((size_t)N*4);
    uint16_t*yb=amalloc((size_t)N*2);
    for(size_t i=0;i<(size_t)N*K;i++) W[i]=((int)(i*2654435761u>>13)%1000-500)/512.f;
    for(int k=0;k<K;k++) x[k]=((k*40503)%1000-500)/512.f;
    Ctx c={W,x,y,yb,ytmp,N,K};
    int iters = wb>32u*1024*1024?20:80;
    gemv_scalar(W,x,yr,N,K); // reference (pinned sequential order)
    struct{const char*name;void(*fn)(void*);}k[]={{"scalar",t_scalar},{"neon x1",t_neon1},{"neon x4",t_neon4},{"neon x4 + fused bf16",t_n4bf16},{"Accelerate/AMX + bf16",t_accel}};
    printf("== N=%d K=%d  (W=%.1f MB, %s L1d)\n",N,K,wb/1048576.0, wb>131072?">":"fits");
    for(int i=0;i<5;i++){
      double dt=bench(k[i].fn,&c,iters);
      double gbs=wb/dt/1e9, gflops=2.0*N*K/dt/1e9;
      printf("  %-22s  %7.1f GB/s   %7.1f GFLOP/s   %6.3f ms\n",k[i].name,gbs,gflops,dt*1e3);
    }
    // numerics: does reordering the reduction change the result vs pinned scalar?
    gemv_neon4(W,x,y,N,K);
    double maxrel=0; for(int n=0;n<N;n++){ double d=fabs(y[n]-yr[n]); double r=d/(fabs(yr[n])+1e-9); if(r>maxrel)maxrel=r; }
    gemv_neon4_bf16(W,x,yb,N,K);
    int bfmismatch=0; for(int n=0;n<N;n++) if(yb[n]!=f32_to_bf16_rne(yr[n])) bfmismatch++;
    printf("  numerics: neon-x4 vs pinned-scalar max rel err = %.2e ; bf16(neon4) != bf16(pinned) on %d/%d outputs\n\n",maxrel,bfmismatch,N);
    free(W);free(x);free(y);free(yr);free(ytmp);free(yb);
  }
  // Prefill: Accelerate GEMM GFLOP/s rising with batch M => compute-bound, AMX wins
  printf("# GEMM (prefill): Y[M x N] = X[M x K] * W[K x N], Accelerate/AMX, N=K=2048\n");
  printf("# watch GFLOP/s climb with M as the matmul leaves the bandwidth ceiling\n\n");
  int N=2048,K=2048;
  float*W=amalloc((size_t)K*N*4);
  for(size_t i=0;i<(size_t)K*N;i++) W[i]=((int)(i*2654435761u>>13)%1000-500)/512.f;
  int Ms[]={1,4,16,64,256};
  for(int mi=0;mi<5;mi++){ int M=Ms[mi];
    float*X=amalloc((size_t)M*K*4),*Y=amalloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++) X[i]=((i*40503)%1000-500)/512.f;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,X,K,W,N,0,Y,N); // warm
    double best=1e30; int it=M<16?60:20;
    for(int r=0;r<5;r++){ double t=now_s(); for(int i=0;i<it;i++) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,X,K,W,N,0,Y,N); t=now_s()-t; if(t<best)best=t;}
    best/=it;
    printf("  M=%-4d  %7.1f GFLOP/s   %7.1f GB/s(weights)   %6.3f ms\n",M,2.0*M*N*K/best/1e9,(double)K*N*4/best/1e9,best*1e3);
    free(X);free(Y);
  }
  free(W);
  return 0;
}
