// ai_bench.cu — uniform prints, fallback chain: FP4(native?)->INT8->FP4(emul) + FP16 baseline
// Build: nvcc -O3 -std=c++17 ai_bench.cu -lcublasLt -lcublas -o ai_bench

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#if __has_include(<cuda_fp4.h>)
  #include <cuda_fp4.h>
  #define HAVE_CUDA_FP4 1
#else
  #define HAVE_CUDA_FP4 0
#endif

#define CHECK_CUDA(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

static inline void timer_begin(cudaEvent_t& a, cudaEvent_t& b){ CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b)); CHECK_CUDA(cudaEventRecord(a)); }
static inline float timer_end(cudaEvent_t a, cudaEvent_t b, int iters){ CHECK_CUDA(cudaEventRecord(b)); CHECK_CUDA(cudaEventSynchronize(b)); float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,a,b)); CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b)); return ms/iters; }

// ---------------- init kernels ----------------
__global__ void init_half(__half* a, size_t n, unsigned seed){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  uint32_t x = seed + 1664525u*(uint32_t)i + 1013904223u;
  float v = ((x&0xFFFF)/32768.0f)-1.0f; a[i]=__float2half(v);
}
__global__ void init_bf16(nv_bfloat16* a, size_t n, unsigned seed){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  uint32_t x = seed + 22695477u*(uint32_t)i + 1u;
  float v = ((x&0xFFFF)/32768.0f)-1.0f; a[i]=__float2bfloat16(v);
}
__global__ void bf16_to_i8(const nv_bfloat16* in, int8_t* out, size_t n, float s){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float v = __bfloat162float(in[i]) * s; int q=(int)lrintf(v);
  if(q<-127) q=-127; if(q>127) q=127; out[i]=(int8_t)q;
}

// ---------------- FP16 baseline ----------------
struct GemmRes { float ms; float tflops; };
static GemmRes bench_fp16_tc(int M,int N,int K,int iters){
  GemmRes r{0,0};
  size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
  __half *A,*B,*C; CHECK_CUDA(cudaMalloc(&A,nA*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&B,nB*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&C,nC*sizeof(__half)));
  int bs=256; int gsA=(int)((nA+bs-1)/bs), gsB=(int)((nB+bs-1)/bs);
  init_half<<<gsA,bs>>>(A,nA,123); init_half<<<gsB,bs>>>(B,nB,456); CHECK_CUDA(cudaMemset(C,0,nC*sizeof(__half)));

  cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
  CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
  float alpha=1.f, beta=0.f;

  cudaEvent_t s,e; timer_begin(s,e);
  for(int i=0;i<iters;i++){
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
      N,M,K, &alpha,
      B, CUDA_R_16F, N,
      A, CUDA_R_16F, K,
      &beta,
      C, CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  r.ms = timer_end(s,e,iters);
  double ops = 2.0*(double)M*N*K;
  r.tflops = (float)(ops / ((r.ms/1000.0)*1e12));
  cublasDestroy(h); cudaFree(A); cudaFree(B); cudaFree(C); return r;
}

// ---------------- INT8 via cuBLASLt ----------------
static GemmRes bench_int8_lt(int M,int N,int K,int iters, int* ok, float* avg_ms_out){
  *ok=0; GemmRes r{0,0}; if(avg_ms_out) *avg_ms_out=0.f;
  size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
  nv_bfloat16 *Ah,*Bh; int8_t *A8,*B8; float *C;
  CHECK_CUDA(cudaMalloc(&Ah,nA*sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&Bh,nB*sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&A8,nA*sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&B8,nB*sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&C, nC*sizeof(float)));
  int bs=256; int gsA=(int)((nA+bs-1)/bs), gsB=(int)((nB+bs-1)/bs);
  init_bf16<<<gsA,bs>>>(Ah,nA,111); init_bf16<<<gsB,bs>>>(Bh,nB,222);
  bf16_to_i8<<<gsA,bs>>>(Ah,A8,nA,127.f);
  bf16_to_i8<<<gsB,bs>>>(Bh,B8,nB,127.f);
  CHECK_CUDA(cudaMemset(C,0,nC*sizeof(float)));

  cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8I, K, M, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8I, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, N, M, N));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, N, M, N));

  float alpha=1.f, beta=0.f;
  cudaEvent_t s,e; timer_begin(s,e);
  for(int i=0;i<iters;i++){
    CHECK_CUBLAS(cublasLtMatmul(lt, op,
      &alpha, B8, Bd, Ah, Ad, &beta, C, Cd, C, Dd,
      NULL, NULL, 0, 0));
  }
  float ms = timer_end(s,e,iters);
  if(avg_ms_out) *avg_ms_out = ms;
  double ops=2.0*(double)M*N*K;
  r.ms = ms; r.tflops = (float)(ops / ((ms/1000.0)*1e12));
  cublasLtDestroy(lt);
  cudaFree(Ah); cudaFree(Bh); cudaFree(A8); cudaFree(B8); cudaFree(C);
  *ok=1; return r;
}

// ---------------- FP4 emulation (packed 4-bit + fused dequant) ----------------
__device__ __forceinline__ int8_t nibble_s4(uint8_t v){ v&=0xF; return (v&0x8)? (int8_t)v-16 : (int8_t)v; }

template<int BM,int BN,int BK>
__global__ void gemm_fp4_fused(const __half* __restrict__ A, const uint8_t* __restrict__ B4,
                               const float* __restrict__ scale, float* __restrict__ C,
                               int M,int N,int K){
  __shared__ __half As[BM][BK];
  __shared__ __half Bs[BK][BN];
  int row0=blockIdx.y*BM, col0=blockIdx.x*BN;
  int tx=threadIdx.x&15, ty=threadIdx.x>>4;
  const int RM=BM/16, RN=BN/16;
  float acc[RM][RN]; #pragma unroll
  for(int i=0;i<RM;i++) for(int j=0;j<RN;j++) acc[i][j]=0.f;
  for(int k0=0;k0<K;k0+=BK){
    #pragma unroll
    for(int i=ty;i<BM;i+=16) for(int j=tx;j<BK;j+=16){
      int r=row0+i, c=k0+j;
      As[i][j]=(r<M&&c<K)? A[r*K+c]:__float2half(0.f);
    }
    #pragma unroll
    for(int i=ty;i<BK;i+=16) for(int j=tx;j<BN;j+=16){
      int r=k0+i, c=col0+j; __half h=__float2half(0.f);
      if(r<K&&c<N){
        size_t byte_idx=((size_t)r>>1)*N + c;
        uint8_t pk=B4[byte_idx];
        int8_t q = ( (r&1)==0 ) ? nibble_s4(pk) : nibble_s4(pk>>4);
        float w = (float)q * scale[c];
        h = __float2half(w);
      }
      Bs[i][j]=h;
    }
    __syncthreads();
    #pragma unroll
    for(int kk=0; kk<BK; ++kk){
      __half aR[RM], bR[RN];
      #pragma unroll
      for(int i=0;i<RM;i++) aR[i]=As[ty*RM+i][kk];
      #pragma unroll
      for(int j=0;j<RN;j++) bR[j]=Bs[kk][tx*RN+j];
      #pragma unroll
      for(int i=0;i<RM;i++){
        float av=__half2float(aR[i]);
        #pragma unroll
        for(int j=0;j<RN;j++) acc[i][j]+= av * __half2float(bR[j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<RM;i++){
    int r=row0+ty*RM+i; if(r>=M) continue;
    #pragma unroll
    for(int j=0;j<RN;j++){
      int c=col0+tx*RN+j; if(c<N) C[r*N+c]=acc[i][j];
    }
  }
}

static void host_quantize_fp4_col(const __half* dB, int K,int N, uint8_t** dB4, float** dScale){
  size_t elems=(size_t)K*N;
  __half* hB=0; CHECK_CUDA(cudaMallocHost(&hB, elems*sizeof(__half)));
  CHECK_CUDA(cudaMemcpy(hB, dB, elems*sizeof(__half), cudaMemcpyDeviceToHost));
  uint8_t* hB4=0; CHECK_CUDA(cudaMallocHost(&hB4, ((elems+1)/2)));
  float* hS=0; CHECK_CUDA(cudaMallocHost(&hS, N*sizeof(float)));

  for(int n=0;n<N;n++){
    float m=0.f;
    for(int k=0;k<K;k++){ float v=fabsf(__half2float(hB[(size_t)k*N+n])); if(v>m)m=v; }
    hS[n] = m>0 ? m/7.f : 1.f;
    for(int k=0;k<K;k++){
      float x=__half2float(hB[(size_t)k*N+n]);
      int q=(int)lrintf(x/hS[n]); if(q<-8) q=-8; if(q>7) q=7;
      uint8_t u=(uint8_t)(q & 0xF);
      size_t byte_idx=((size_t)k>>1)*N + n;
      if((k&1)==0) hB4[byte_idx]=(hB4[byte_idx]&0xF0)|u;
      else         hB4[byte_idx]=(hB4[byte_idx]&0x0F)|(u<<4);
    }
  }
  CHECK_CUDA(cudaMalloc(dB4, ((elems+1)/2)));
  CHECK_CUDA(cudaMemcpy(*dB4, hB4, ((elems+1)/2), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(dScale, N*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(*dScale, hS, N*sizeof(float), cudaMemcpyHostToDevice));
  cudaFreeHost(hB); cudaFreeHost(hB4); cudaFreeHost(hS);
}

static GemmRes bench_fp4_emulated(int M,int N,int K,int iters, float* avg_ms_out){
  if(avg_ms_out) *avg_ms_out=0.f;
  GemmRes r{0,0};
  size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
  __half *A,*B; float *C;
  CHECK_CUDA(cudaMalloc(&A,nA*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&B,nB*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&C,nC*sizeof(float)));
  int bs=256; int gsA=(int)((nA+bs-1)/bs), gsB=(int)((nB+bs-1)/bs);
  init_half<<<gsA,bs>>>(A,nA,777); init_half<<<gsB,bs>>>(B,nB,888);

  uint8_t* B4; float* S; host_quantize_fp4_col(B,K,N,&B4,&S);
  dim3 block(256); const int BM=128,BN=128,BK=64; dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);

  cudaEvent_t s,e; timer_begin(s,e);
  for(int i=0;i<iters;i++){ gemm_fp4_fused<BM,BN,BK><<<grid,block>>>(A,B4,S,C,M,N,K); }
  float ms = timer_end(s,e,iters);
  if(avg_ms_out) *avg_ms_out = ms;
  double ops=2.0*(double)M*N*K;
  r.ms = ms; r.tflops=(float)(ops/((ms/1000.0)*1e12));

  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(B4); cudaFree(S);
  return r;
}

// ---------------- KV bandwidth ----------------
__global__ void kv_stream(const float4* K,const float4* V,float4* O,size_t n){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x, s=gridDim.x*blockDim.x;
  for(; i<n; i+=s){ float4 a=K[i], b=V[i]; O[i]={a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w}; }
}
static float bench_kv(float seconds, float* GBps, float* avg_ms){
  size_t freeB=0,totalB=0; CHECK_CUDA(cudaMemGetInfo(&freeB,&totalB));
  size_t n_vec = (size_t)(freeB*0.6) / (3*sizeof(float4)); if(n_vec < (1u<<20)){ *GBps=0; if(avg_ms) *avg_ms=NAN; return NAN; }
  float4 *K,*V,*O; CHECK_CUDA(cudaMalloc(&K,n_vec*sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&V,n_vec*sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&O,n_vec*sizeof(float4)));
  CHECK_CUDA(cudaMemset(K,0,n_vec*sizeof(float4))); CHECK_CUDA(cudaMemset(V,0,n_vec*sizeof(float4))); CHECK_CUDA(cudaMemset(O,0,n_vec*sizeof(float4)));
  int bs=256; int gs=(int)min((size_t)65535, (n_vec+bs-1)/bs);

  int iters = 10;
  cudaEvent_t s,e; timer_begin(s,e);
  for(int i=0;i<iters;i++){ kv_stream<<<gs,bs>>>(K,V,O,n_vec); }
  float ms = timer_end(s,e,iters);
  double moved = 3.0 * sizeof(float4) * (double)n_vec;
  *GBps = (float)((moved / (ms/1000.0)) / 1e9);
  if(avg_ms) *avg_ms = ms;
  cudaFree(K); cudaFree(V); cudaFree(O); return ms;
}

// ---------------- inference estimates ----------------
static void print_infer_estimates(double tflops_mix, double gbps, const char* low_name){
  // params
  const int d=4096, L=32, r=4, B=1;
  const double s_kv=2.0; // FP16 KV
  const double s_w_fp16=2.0;
  double s_w_low = 0.5;  // default for fp4_emulated
  if(low_name && low_name[0]=='i') s_w_low = 1.0; // int8

  auto toks = [&](int t, double s_w)->double{
    const double F_layer = (8.0+4.0*r)*(double)d*(double)d + 4.0*t*(double)d;
    const double F_total = (double)L*F_layer*B;
    const double Tcomp = F_total / (tflops_mix*1e12);
    const double Q_kv = (double)L*2.0*B*(double)d*s_kv*t;
    const double beta_w = 1.2;
    const double Q_w  = (double)L*beta_w*(double)d*(double)d*s_w;
    const double Tmem = (Q_kv + Q_w) / (gbps*1e9);
    const double T = (Tcomp > Tmem) ? Tcomp : Tmem;
    return 1.0/T;
  };

  printf("\n\n=== Inference-oriented estimates (tokens/s) ===\n");
  printf("Using: d=%d, L=%d, r=%d, batch=%d, KV=FP16\n", d,L,r,B);
  printf("Measured: mixed_GEMM=%.2f TFLOPS, BW=%.1f GB/s\n", tflops_mix, gbps);
  const int tlist[3]={512,2048,8192};
  for(int i=0;i<3;i++){
    int t=tlist[i];
    double a=toks(t,s_w_fp16);
    double b=toks(t,s_w_low);
    const char* label = (low_name && low_name[0]=='i') ? "INT8-weights" : "FP4-weights";
    printf("t=%4d: FP16-weights %.1f tok/s, %s %.1f tok/s\n", t, a, label, b);
  }
}

// ---------------- main with old-style prints ----------------
int main(){
  cudaDeviceProp p{}; CHECK_CUDA(cudaGetDeviceProperties(&p,0));
  int M=8192,N=8192,K=8192;
  printf("GPU: %s, CC %d.%d, SMs %d, globalMem %.1f GB\n", p.name,p.major,p.minor,p.multiProcessorCount, p.totalGlobalMem/1e9);
  printf("Problem sizes: M=%d N=%d K=%d\n", M,N,K);

  // "Budgets" just for format consistency
  const double BUDGET_FP16_S=7.0, BUDGET_FP4_S=7.0, BUDGET_KV_S=6.0;

  printf("\n=== AI bench (FP16 + FP4 mix) ===\n");
  printf("Budgets: FP16 %.1fs, FP4 %.1fs, KV %.1fs\n", BUDGET_FP16_S, BUDGET_FP4_S, BUDGET_KV_S);

  // FP16
  GemmRes r16 = bench_fp16_tc(M,N,K,10);
  printf("FP16 TC GEMM: %.2f TFLOPS (avg %.2f ms)\n", r16.tflops, r16.ms);

  // Low-precision chain: try INT8, else FP4 emu. (Native FP4 gated by future toolchains.)
  const char* low_name = "int8";
  GemmRes rlow{0,0}; int ok=0; float low_ms=0.f;
  rlow = bench_int8_lt(M,N,K,10,&ok,&low_ms);
  if(!ok){
    rlow = bench_fp4_emulated(M,N,K,5,&low_ms);
    low_name="fp4_emulated";
  }
  // Report with legacy wording when emulated
  if(low_name[0]=='i')
    printf("Low-precision GEMM (int8): %.2f TFLOPS (avg %.2f ms)\n", rlow.tflops, low_ms);
  else
    printf("FP4 fused-dequant GEMM (eff): %.2f TFLOPS (avg %.2f ms)\n", rlow.tflops, low_ms);

  // KV bandwidth
  float gbps=0, kv_ms=0; bench_kv(5.0,&gbps,&kv_ms);
  if(std::isnan(kv_ms)) printf("KV bandwidth: skipped (insufficient memory)\n");
  else                  printf("KV bandwidth: %.1f GB/s (avg %.2f ms)\n", gbps, kv_ms);

  // Anchors and norms
  const double C0=20.0, M0=400.0;
  double compute_norm = r16.tflops>0 ? r16.tflops/C0 : 1e-6;
  double mem_norm     = gbps>0 ? gbps/M0 : 1e-6;
  printf("Anchors: C0=%.1f TFLOPS, M0=%.0f GB/s\n", C0, M0);
  printf("Pretest norms: compute=%.3f, memory=%.3f\n", compute_norm, mem_norm);

  // Dynamic mix and score
  double frac_low = 0.5 + 0.35 * (compute_norm/(compute_norm+mem_norm)); // 50–85%
  double F = 2.0*(double)M*N*K;
  double t16 = F/(r16.tflops*1e12);
  double tl  = rlow.tflops>0 ? F/(rlow.tflops*1e12) : 1e9;
  double tmix = frac_low*tl + (1.0-frac_low)*t16;
  double mix_tflops = F/(tmix*1e12);

  double pwt = 0.2 + 0.6/(1.0+compute_norm);
  double score = 1.0 / ( (pwt/ (mix_tflops/C0) ) + ((1.0-pwt)/(gbps/M0)) );

  // Selection line for integer/low path
  if(low_name[0]=='i') printf("Selected low-precision mode: INT8\n");
  else                 printf("Selected low-precision mode: FP4(emulated)\n");
  printf("Selected low-precision fraction: %.2f\n", frac_low);
  printf("Mixed GEMM throughput: %.2f TFLOPS\n", mix_tflops);
  printf("Dynamic weight p=%.3f (higher compute -> lower p -> memory weighs more)\n", pwt);
  printf("Overall score: %.2f\n", 100.0*score);

  // Inference estimates
  print_infer_estimates(mix_tflops, gbps, low_name);
  return 0;
}

