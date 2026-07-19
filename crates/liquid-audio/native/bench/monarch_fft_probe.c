// monarch_fft_probe.c — fused Monarch FFT (single signal), the register/cache-FIFO
// concept applied to the long-conv FFT, following the recovered bf16 fused design.
// N = N1*N2. Two small matmuls against CONSTANT DFT matrices + a twiddle, the whole
// [N1,N2] complex plane held in L1, no memory pass between stages:
//   step 1  A = D_N1 @ xT      (x real -> 2 real GEMMs)      [N1,N1]@[N1,N2]
//   step 2  Z = A (.) twiddle  (elementwise complex, in place)
//   step 3  B = Z @ D_N2       (complex -> 4 real GEMMs)     [N1,N2]@[N2,N2]
//   out     X[j1*N2 + j2] = B[j1,j2]
// Bailey 4-step convention: input x[k1,k2]=x_flat[k2*N1+k1] (col-major), output row-major.
// Two kernels, both parity-gated against a naive O(N^2) DFT:
//   - scalar fp32  (pins the index mapping)
//   - BFDOT bf16, fp32 accumulate  (the fast tier; DFT matrices are constants,
//     so their transposes are a one-time init, not a per-call repack)
//
// Build:  clang -O3 -ffp-contract=off -march=armv8.6-a+bf16 monarch_fft_probe.c -lm -o /tmp/mf
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define N1 32
#define N2 32
#define NT (N1*N2)
static const double PI = 3.14159265358979323846;
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint16_t f2b(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t l=(x>>16)&1u; x+=0x7fffu+l; return (uint16_t)(x>>16); }

// ---- constant DFT matrices (built once) --------------------------------------
static float d1r[N1][N1], d1i[N1][N1];      // D_N1[j1][k1] = exp(-2pi i k1 j1 / N1), contract k1 (row contiguous)
static float d2r[N2][N2], d2i[N2][N2];      // D_N2^T[j2][k2] = exp(-2pi i k2 j2 / N2), contract k2 (row contiguous)
static float twr[N1][N2], twi[N1][N2];      // twiddle W_NT^{j1 k2}
static uint16_t bd1r[N1][N1], bd1i[N1][N1], bd2r[N2][N2], bd2i[N2][N2];  // bf16 copies

static void build_tables(void){
  for(int j=0;j<N1;j++) for(int k=0;k<N1;k++){ double a=2*PI*k*j/N1; d1r[j][k]=(float)cos(a); d1i[j][k]=(float)-sin(a);
    bd1r[j][k]=f2b(d1r[j][k]); bd1i[j][k]=f2b(d1i[j][k]); }
  for(int j=0;j<N2;j++) for(int k=0;k<N2;k++){ double a=2*PI*k*j/N2; d2r[j][k]=(float)cos(a); d2i[j][k]=(float)-sin(a);
    bd2r[j][k]=f2b(d2r[j][k]); bd2i[j][k]=f2b(d2i[j][k]); }
  for(int j=0;j<N1;j++) for(int k=0;k<N2;k++){ double a=2*PI*j*k/NT; twr[j][k]=(float)cos(a); twi[j][k]=(float)-sin(a); }
}

// ---- reference: naive O(N^2) DFT in double ----------------------------------
static void dft_ref(const float* x, double* Xr, double* Xi){
  for(int K=0;K<NT;K++){ double sr=0,si=0;
    for(int n=0;n<NT;n++){ double a=2*PI*n*K/NT; sr+=x[n]*cos(a); si+=x[n]*(-sin(a)); }
    Xr[K]=sr; Xi[K]=si; }
}

// ---- Monarch, scalar fp32 (pins the mapping) --------------------------------
// x[k1,k2]=xflat[k2*N1+k1]; step1: DFT over k2; twiddle W_N^{k1 j2}; transpose;
// step3: DFT over k1; out X[j1*N2+j2].
static void monarch_f32(const float* xflat, float* Xr, float* Xi){
  float x2[N1][N2];
  for(int k1=0;k1<N1;k1++) for(int k2=0;k2<N2;k2++) x2[k1][k2]=xflat[k2*N1+k1];
  float ZTr[N2][N1], ZTi[N2][N1];                     // transposed stage-1+twiddle: ZT[j2][k1]
  for(int k1=0;k1<N1;k1++) for(int j2=0;j2<N2;j2++){ float sr=0,si=0;
    for(int k2=0;k2<N2;k2++){ sr+=x2[k1][k2]*d2r[j2][k2]; si+=x2[k1][k2]*d2i[j2][k2]; }
    ZTr[j2][k1]=sr*twr[k1][j2]-si*twi[k1][j2];
    ZTi[j2][k1]=sr*twi[k1][j2]+si*twr[k1][j2]; }
  for(int j2=0;j2<N2;j2++) for(int j1=0;j1<N1;j1++){ float br=0,bi=0;
    for(int k1=0;k1<N1;k1++){ br+=d1r[j1][k1]*ZTr[j2][k1]-d1i[j1][k1]*ZTi[j2][k1];
                              bi+=d1r[j1][k1]*ZTi[j2][k1]+d1i[j1][k1]*ZTr[j2][k1]; }
    Xr[j1*N2+j2]=br; Xi[j1*N2+j2]=bi; }
}

// ---- Monarch, BFDOT bf16 inputs, fp32 accumulate ----------------------------
static inline float bfdot32(const uint16_t* a,const uint16_t* b,int n){   // sum a[k]*b[k], k=0..n
  float32x4_t acc=vdupq_n_f32(0); int k=0;
  for(;k+8<=n;k+=8) acc=vbfdotq_f32(acc,vreinterpretq_bf16_u16(vld1q_u16(a+k)),vreinterpretq_bf16_u16(vld1q_u16(b+k)));
  return vaddvq_f32(acc);
}
static void monarch_bf16(const float* xflat, float* Xr, float* Xi){
  uint16_t x2[N1][N2];
  for(int k1=0;k1<N1;k1++) for(int k2=0;k2<N2;k2++) x2[k1][k2]=f2b(xflat[k2*N1+k1]);
  uint16_t ZTr[N2][N1], ZTi[N2][N1];                  // transposed stage-1+twiddle, bf16
  for(int k1=0;k1<N1;k1++) for(int j2=0;j2<N2;j2++){
    float sr=bfdot32(x2[k1],bd2r[j2],N2), si=bfdot32(x2[k1],bd2i[j2],N2);
    ZTr[j2][k1]=f2b(sr*twr[k1][j2]-si*twi[k1][j2]);
    ZTi[j2][k1]=f2b(sr*twi[k1][j2]+si*twr[k1][j2]); }
  for(int j2=0;j2<N2;j2++) for(int j1=0;j1<N1;j1++){
    float rr=bfdot32(bd1r[j1],ZTr[j2],N1), ii=bfdot32(bd1i[j1],ZTi[j2],N1);
    float ri=bfdot32(bd1r[j1],ZTi[j2],N1), ir=bfdot32(bd1i[j1],ZTr[j2],N1);
    Xr[j1*N2+j2]=rr-ii; Xi[j1*N2+j2]=ri+ir; }
}

static double maxerr(const float* Xr,const float* Xi,const double* Rr,const double* Ri){
  double m=0,scale=0; for(int K=0;K<NT;K++) scale+=Rr[K]*Rr[K]+Ri[K]*Ri[K]; scale=sqrt(scale/NT)+1e-9;
  for(int K=0;K<NT;K++){ double dr=Xr[K]-Rr[K], di=Xi[K]-Ri[K]; double e=sqrt(dr*dr+di*di)/scale; if(e>m)m=e; }
  return m;
}

int main(void){
  build_tables();
  float x[NT]; for(int n=0;n<NT;n++) x[n]=((int)((n*2654435761u>>13))%2000-1000)/1024.f;
  double Rr[NT],Ri[NT]; dft_ref(x,Rr,Ri);
  float Xr[NT],Xi[NT];

  printf("# fused Monarch FFT, N=%d = %dx%d, single signal\n\n",NT,N1,N2);
  monarch_f32(x,Xr,Xi);
  printf("  parity Monarch-f32  vs naive DFT: max rel err = %.2e\n", maxerr(Xr,Xi,Rr,Ri));
  monarch_bf16(x,Xr,Xi);
  printf("  parity Monarch-bf16 vs naive DFT: max rel err = %.2e   (bf16 in, fp32 accumulate)\n\n", maxerr(Xr,Xi,Rr,Ri));

  int it=200000;
  float Yr[NT],Yi[NT];
  monarch_f32(x,Yr,Yi); monarch_bf16(x,Yr,Yi);       // warm
  double t=now_s(); for(int i=0;i<it;i++) monarch_f32(x,Yr,Yi); double tf=(now_s()-t)/it;
  t=now_s(); for(int i=0;i<it;i++) monarch_bf16(x,Yr,Yi); double tb=(now_s()-t)/it;
  double flops=2.0*(2.0*N1*N1*N2 + 4.0*N1*N2*N2);    // 6 real GEMMs
  printf("  Monarch f32   : %.3f us/FFT   %6.1f GFLOP/s\n", tf*1e6, flops/tf/1e9);
  printf("  Monarch bf16  : %.3f us/FFT   %6.1f GFLOP/s   (%.2fx)\n", tb*1e6, flops/tb/1e9, tf/tb);
  return 0;
}
