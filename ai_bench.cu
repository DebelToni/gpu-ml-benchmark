// ai_bench.cu — feature-checked low-precision chain + FP16 baseline
// Order: FP4(native) -> FP6(native? none) -> FP8 -> INT8 -> FP4(emulated)
// Build: nvcc -O3 -std=c++17 ai_bench.cu -lcublasLt -lcublas -o ai_bench
// Notes: FP4 in cuBLASLt requires CUDA >= 12.9/13 with CUDA_R_4F_E2M1; FP6 not in cuBLASLt as of 12.9/13.
// Docs: cuBLAS 13.0 data types inc. CUDA_R_4F_E2M1; FP4/FP6/FP8 intrinsics in CUDA Math API. :contentReference[oaicite:0]{index=0}

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#if __has_include(<cuda_fp8.h>)
  #include <cuda_fp8.h>
  #define HAVE_CUDA_FP8 1
#else
  #define HAVE_CUDA_FP8 0
#endif

#if __has_include(<cuda_fp4.h>)
  #include <cuda_fp4.h>
  #define HAVE_CUDA_FP4 1
#else
  #define HAVE_CUDA_FP4 0
#endif

// ---------- error checks ----------
#define CHECK_CUDA(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

// ---------- timing ----------
template <class F>
float avg_ms(F launch, int iters=10, int warmup=3){
  cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  for(int i=0;i<warmup;i++) launch();
  CHECK_CUDA(cudaEventRecord(s));
  for(int i=0;i<iters;i++) launch();
  CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
  return ms/iters;
}

// ---------- utils ----------
template <typename T>
T* dalloc(size_t n){ T* p=nullptr; CHECK_CUDA(cudaMalloc(&p, n*sizeof(T))); return p; }

template <typename T>
__global__ void init_uniform(T* a, size_t n, unsigned seed){
  size_t i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=n) return;
  // simple LCG
  uint32_t x = seed + 1664525u*(uint32_t)i + 1013904223u;
  float v = (float)(x & 0xFFFF)/32768.f - 1.f; // [-1,1)
  if constexpr (std::is_same<T,__half>::value) a[i]=__float2half(v);
  else if constexpr (std::is_same<T,nv_bfloat16>::value) a[i]=__float2bfloat16(v);
  else if constexpr (std::is_same<T,float>::value) a[i]=v;
  else if constexpr (std::is_same<T,int8_t>::value) a[i]=(int8_t)max(-127,min(127,(int)lrintf(v*127.f)));
}

template <typename T>
void fill_uniform(T* dptr, size_t n, unsigned seed=123){
  int bs=256; int gs=(int)((n+bs-1)/bs);
  init_uniform<<<gs,bs>>>(dptr, n, seed);
  CHECK_CUDA(cudaGetLastError());
}

// ================= FP16 Tensor Core (baseline) =================
struct GemmRes { float ms=0, tflops=0; };
GemmRes bench_fp16_tc(int M,int N,int K){
  GemmRes r;
  __half *A=dalloc<__half>((size_t)M*K);
  __half *B=dalloc<__half>((size_t)K*N);
  __half *C=dalloc<__half>((size_t)M*N);
  fill_uniform(A,(size_t)M*K);
  fill_uniform(B,(size_t)K*N);
  CHECK_CUDA(cudaMemset(C,0,(size_t)M*N*sizeof(__half)));
  cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
  CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
  float alpha=1.f, beta=0.f;
  auto once=[&](){
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
      N,M,K, &alpha,
      B, CUDA_R_16F, N,
      A, CUDA_R_16F, K,
      &beta,
      C, CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  r.ms = avg_ms(once, 10, 3);
  double ops = 2.0*(double)M*N*K;
  r.tflops = (float)(ops/((r.ms/1000.0)*1e12));
  cublasDestroy(h);
  cudaFree(A); cudaFree(B); cudaFree(C);
  return r;
}

// ================= FP8 via cuBLASLt (E4M3) =================
#if HAVE_CUDA_FP8
__global__ void bf16_to_fp8_e4m3(const nv_bfloat16* __restrict__ in,
                                 __nv_fp8_e4m3* __restrict__ out, size_t n, float scale){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float v = __bfloat162float(in[i]) * scale;
  out[i] = __nv_cvt_float_to_fp8_e4m3(v, __NV_SATFINITE);
}
#endif

GemmRes bench_fp8_lt(int M,int N,int K, bool& ok){
  ok=false; GemmRes r{};
#if HAVE_CUDA_FP8
  // A,B in FP8 E4M3, compute in FP32, output BF16
  size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
  nv_bfloat16 *Ah=dalloc<nv_bfloat16>(nA);
  nv_bfloat16 *Bh=dalloc<nv_bfloat16>(nB);
  nv_bfloat16 *C =dalloc<nv_bfloat16>(nC);
  fill_uniform(Ah,nA,111); fill_uniform(Bh,nB,222);
  __nv_fp8_e4m3 *A8=dalloc<__nv_fp8_e4m3>(nA);
  __nv_fp8_e4m3 *B8=dalloc<__nv_fp8_e4m3>(nB);
  // simple per-tensor scale = 1.0 (inputs already small), adjust if you want dynamic range
  float *dSa=dalloc<float>(1), *dSb=dalloc<float>(1);
  float s=1.0f; CHECK_CUDA(cudaMemcpy(dSa,&s,sizeof(float),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSb,&s,sizeof(float),cudaMemcpyHostToDevice));
  { int bs=256; int gsA=(int)((nA+bs-1)/bs), gsB=(int)((nB+bs-1)/bs);
    bf16_to_fp8_e4m3<<<gsA,bs>>>(Ah,A8,nA,1.0f);
    bf16_to_fp8_e4m3<<<gsB,bs>>>(Bh,B8,nB,1.0f);
    CHECK_CUDA(cudaGetLastError());
  }

  cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  // Optional per-tensor scales (API present in headers). If not supported at runtime, Lt will ignore or error; we guard with status.
#ifdef CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSa, sizeof(dSa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSb, sizeof(dSb)));
#endif

  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, K, M, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF,   N, M, N));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF,   N, M, N));

  float alpha=1.f, beta=0.f;
  auto once=[&](){
    CHECK_CUBLAS(cublasLtMatmul(lt, op,
      &alpha, B8, Bd, Ah, Ad, &beta, C, Cd, C, Dd,
      nullptr, nullptr, 0, 0));
  };
  r.ms = avg_ms(once, 10, 3);
  double ops=2.0*(double)M*N*K;
  r.tflops = (float)(ops/((r.ms/1000.0)*1e12));
  cublasLtDestroy(lt);
  cudaFree(Ah); cudaFree(Bh); cudaFree(C); cudaFree(A8); cudaFree(B8); cudaFree(dSa); cudaFree(dSb);
  ok=true;
#endif
  return r;
}

// ================= INT8 via cuBLASLt =================
__global__ void bf16_to_i8(const nv_bfloat16* in, int8_t* out, size_t n, float s){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float v = __bfloat162float(in[i]) * s;
  int q = (int)lrintf(v);
  out[i] = (int8_t)max(-127,min(127,q));
}

GemmRes bench_int8_lt(int M,int N,int K, bool& ok){
  ok=false; GemmRes r{};
  size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
  nv_bfloat16 *Ah=dalloc<nv_bfloat16>(nA);
  nv_bfloat16 *Bh=dalloc<nv_bfloat16>(nB);
  float *C = dalloc<float>(nC); // accumulate to FP32
  fill_uniform(Ah,nA,333); fill_uniform(Bh,nB,444);
  int8_t *A8 = dalloc<int8_t>(nA);
  int8_t *B8 = dalloc<int8_t>(nB);
  { int bs=256; int gsA=(int)((nA+bs-1)/bs), gsB=(int)((nB+bs-1)/bs);
    bf16_to_i8<<<gsA,bs>>>(Ah,A8,nA,127.f);  // simple scale
    bf16_to_i8<<<gsB,bs>>>(Bh,B8,nB,127.f);
    CHECK_CUDA(cudaGetLastError());
  }
  cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8I, K, M, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8I, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, N, M, N));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, N, M, N));
  float alpha=1.f, beta=0.f;
  auto once=[&](){
    CHECK_CUBLAS(cublasLtMatmul(lt, op,
      &alpha, B8, Bd, Ah, Ad, &beta, C, Cd, C, Dd,
      nullptr, nullptr, 0, 0));
  };
  r.ms = avg_ms(once, 10, 3);
  double ops=2.0*(double)M*N*K;
  r.tflops = (float)(ops/((r.ms/1000.0)*1e12));
  cublasLtDestroy(lt);
  cudaFree(Ah); cudaFree(Bh); cudaFree(C); cudaFree(A8); cudaFree(B8);
  ok=true; return r;
}

// ================= FP4 emulation (packed 4-bit + fused dequant) =================
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

void host_quantize_fp4_col(const __half* dB, int K,int N, uint8_t** dB4, float** dScale){
  // copy to host once to compute per-col scales; ok for a bench
  std::vector<__half> hB((size_t)K*N);
  CHECK_CUDA(cudaMemcpy(hB.data(), dB, hB.size()*sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<uint8_t> hB4(((size_t)K*N+1)/2, 0);
  std::vector<float> hS(N);
  for(int n=0;n<N;n++){
    float m=0.f;
    for(int k=0;k<K;k++){ float v=fabsf(__half2float(hB[(size_t)k*N+n])); if(v>m)m=v;}
    hS[n] = m>0 ? m/7.f : 1.f;
    for(int k=0;k<K;k++){
      float x=__half2float(hB[(size_t)k*N+n]);
      int q=(int)lrintf(x/hS[n]); if(q<-8)q=-8; if(q>7)q=7;
      uint8_t u=(uint8_t)(q & 0xF);
      size_t byte_idx=((size_t)k>>1)*N + n;
      if((k&1)==0) hB4[byte_idx]=(hB4[byte_idx]&0xF0)|u;
      else         hB4[byte_idx]=(hB4[byte_idx]&0x0F)|(u<<4);
    }
  }
  CHECK_CUDA(cudaMalloc(dB4, hB4.size())); CHECK_CUDA(cudaMemcpy(*dB4, hB4.data(), hB4.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(dScale, (size_t)N*sizeof(float))); CHECK_CUDA(cudaMemcpy(*dScale, hS.data(), (size_t)N*sizeof(float), cudaMemcpyHostToDevice));
}

GemmRes bench_fp4_emulated(int M,int N,int K){
  GemmRes r{};
  __half *A=dalloc<__half>((size_t)M*K);
  __half *B=dalloc<__half>((size_t)K*N);
  float  *C=dalloc<float>((size_t)M*N);
  fill_uniform(A,(size_t)M*K,555); fill_uniform(B,(size_t)K*N,666);
  uint8_t* B4=nullptr; float* S=nullptr; host_quantize_fp4_col(B,K,N,&B4,&S);
  dim3 block(256); const int BM=128,BN=128,BK=64; dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
  auto once=[&](){ gemm_fp4_fused<BM,BN,BK><<<grid,block>>>(A,B4,S,C,M,N,K); };
  r.ms = avg_ms(once, 5, 2);
  double ops=2.0*(double)M*N*K;
  r.tflops = (float)(ops/((r.ms/1000.0)*1e12));
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(B4); cudaFree(S);
  return r;
}

// ================= KV bandwidth (streaming) =================
__global__ void kv_stream(const float4* __restrict__ K, const float4* __restrict__ V, float4* __restrict__ O, size_t n){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t s=gridDim.x*blockDim.x;
  for(; i<n; i+=s){ float4 a=K[i], b=V[i]; O[i]={a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w}; }
}
float bench_kv_gbps(double seconds, float& GBps){
  size_t freeB=0,totalB=0; CHECK_CUDA(cudaMemGetInfo(&freeB,&totalB));
  size_t n_vec = (size_t)(freeB*0.6) / (3*sizeof(float4)); if(n_vec< (1u<<20)) {GBps=0; return NAN;}
  float4 *K=dalloc<float4>(n_vec), *V=dalloc<float4>(n_vec), *O=dalloc<float4>(n_vec);
  CHECK_CUDA(cudaMemset(K,0,n_vec*sizeof(float4))); CHECK_CUDA(cudaMemset(V,0,n_vec*sizeof(float4))); CHECK_CUDA(cudaMemset(O,0,n_vec*sizeof(float4)));
  int bs=256; int gs=(int)std::min<size_t>((n_vec+bs-1)/bs, 65535);
  auto once=[&](){ kv_stream<<<gs,bs>>>(K,V,O,n_vec); };
  // run to ~seconds
  float ms=0; { float chunk = avg_ms(once, 10, 2); int reps = (int)ceil((seconds*1000)/chunk); ms = avg_ms(once, reps>10?10:reps, 2); }
  double moved = 3.0 * sizeof(float4) * (double)n_vec;
  GBps = (float)((moved/ (ms/1000.0)) / 1e9);
  cudaFree(K); cudaFree(V); cudaFree(O); return ms;
}

// ================= main =================
int main(){
  cudaDeviceProp p{}; CHECK_CUDA(cudaGetDeviceProperties(&p,0));
  int M=8192,N=8192,K=8192;
  printf("GPU: %s, CC %d.%d, SMs %d, globalMem %.1f GB\n", p.name, p.major,p.minor,p.multiProcessorCount, p.totalGlobalMem/1e9);
  printf("Problem sizes: M=%d N=%d K=%d\n", M,N,K);

  // 1) FP16 baseline
  GemmRes fp16 = bench_fp16_tc(M,N,K);
  printf("\nFP16 TC GEMM: %.2f TFLOPS (avg %.2f ms)\n", fp16.tflops, fp16.ms);

  // 2) Lowest quantization path available
  const char* low_name = "fp4_emulated";
  GemmRes low = {}; bool ok=false;
#if HAVE_CUDA_FP4
  if (p.major>=10){
    // Native FP4 path via cuBLASLt would go here; many toolkits expose type but not full examples yet.
    // If your headers/runtime support it, implement like FP8 with CUDA_R_4F_E2M1.
    // For now, skip to FP8 and mark for rebuild.
  }
#endif

  if(!ok){
    // FP6 native: not in cuBLASLt as of CUDA 12.9/13.0 -> skip to FP8. :contentReference[oaicite:1]{index=1}
  }
  if(!ok){
    low = bench_fp8_lt(M,N,K, ok);
    if(ok){ low_name="fp8_e4m3"; }
  }
  if(!ok){
    low = bench_int8_lt(M,N,K, ok);
    if(ok){ low_name="int8"; }
  }
  if(!ok){
    low = bench_fp4_emulated(M,N,K);
    low_name="fp4_emulated";
  }

  printf("\nLow-precision GEMM (%s): %.2f TFLOPS (avg %.2f ms)\n", low_name, low.tflops, low.ms);

  // 3) KV bandwidth
  float gbps=0; bench_kv_gbps(5.0, gbps); // ~5s
  printf("KV bandwidth: %.1f GB/s\n", gbps);

  // 4) Dynamic mix: choose fraction of low-precision vs FP16
  const double C0=20.0, M0=400.0;
  double compute_norm = fmax(1e-6, fp16.tflops/C0);
  double mem_norm     = fmax(1e-6, gbps/M0);
  double frac_low = 0.5 + 0.35 * (compute_norm/(compute_norm+mem_norm)); // 50–85%
  double t16 = (2.0*(double)M*N*K) / (fp16.tflops*1e12);
  double tlow = (low.tflops>0)? (2.0*(double)M*N*K) / (low.tflops*1e12) : 1e9;
  double tmix = frac_low*tlow + (1.0-frac_low)*t16;
  double mix_tflops = (2.0*(double)M*N*K) / (tmix*1e12);

  double pwt = 0.2 + 0.6/(1.0+compute_norm);
  double score = 1.0 / ( (pwt/fmax(1e-6, mix_tflops/C0)) + ((1.0-pwt)/fmax(1e-6, gbps/M0)) );
  printf("\nSelected low-precision: %s, fraction=%.2f\n", low_name, frac_low);
  printf("Mixed GEMM throughput: %.2f TFLOPS\n", mix_tflops);
  printf("Dynamic weight p=%.3f\n", pwt);
  printf("Overall score: %.2f\n", 100.0*score);

  // Hints if FP4 native was skipped
#if HAVE_CUDA_FP4
  if (p.major>=10 && (std::string(low_name)!="fp4_e2m1")){
    printf("Note: build with CUDA >=13 and cuBLASLt FP4 to enable native FP4 (CUDA_R_4F_E2M1).\n");
  }
#else
  if (p.major>=10){
    printf("Note: CUDA headers lack FP4 types here; upgrade toolkit to enable native FP4.\n");
  }
#endif
  return 0;
}

