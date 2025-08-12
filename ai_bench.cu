// ai_bench_sustained.cu  (long-run, rotating weights)
// CUDA 12+. Builds with: nvcc -O3 -std=c++17 ai_bench_fp4.cu -lcublas -o ai_bench_fp4

#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cmath>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// #define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CUDA_CHECK(x) do {                                      \
  cudaError_t _err = (x);                                       \
  if (_err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
            cudaGetErrorString(_err));                             \
    exit(1);                                                       \
  }                                                                \
} while (0)
#define CHECK_CUBLAS(x) do{auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

// ---------- KV streaming kernel with offset ----------
__global__ void kv_stream_kernel(const float4* __restrict__ K,
                                 const float4* __restrict__ V,
                                 float4* __restrict__ O,
                                 size_t n_vec, size_t start_off){
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = i0; i < n_vec; i += stride){
        size_t idx = (i + start_off);
        if (idx >= n_vec) idx -= n_vec; // wrap once
        float4 a = K[idx];
        float4 b = V[idx];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        O[idx] = c;
    }
}

// ---------- timing helper: run in chunks until budget seconds ----------
template <typename F>
float avg_ms_budget(F f, double seconds_budget, int chunk_iters=5, int warmup=3){
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    for(int i=0;i<warmup;++i){ f(); }
    double total_ms = 0.0; long total_iters = 0;
    while (total_ms < seconds_budget*1000.0){
        CUDA_CHECK(cudaEventRecord(s));
        for(int i=0;i<chunk_iters;++i) f();
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
        total_ms += ms; total_iters += chunk_iters;
    }
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return float(total_ms / double(total_iters));
}

// ---------- FP4 utils ----------
__device__ __forceinline__ int8_t nibble_to_s4(uint8_t v){
    v &= 0xF; return (v&0x8)? int8_t(v)-16 : int8_t(v);
}

template<int BM, int BN, int BK>
__global__ void gemm_fp4w_fused_kernel(const __half* __restrict__ A,
                                       const uint8_t* __restrict__ B4,
                                       const float* __restrict__ scale,
                                       float* __restrict__ C,
                                       int M,int N,int K){
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int RM=BM/16, RN=BN/16;
    float acc[RM][RN]; 
    #pragma unroll
    for(int i=0;i<RM;i++) for(int j=0;j<RN;j++) acc[i][j]=0.0f;

    for(int k0=0;k0<K;k0+=BK){
        #pragma unroll
        for(int i=ty;i<BM;i+=16) for(int j=tx;j<BK;j+=16){
            int r=row0+i, c=k0+j;
            As[i][j] = (r<M&&c<K)? A[r*K+c] : __float2half(0.f);
        }
        #pragma unroll
        for(int i=ty;i<BK;i+=16) for(int j=tx;j<BN;j+=16){
            int r=k0+i, c=col0+j;
            __half h=__float2half(0.f);
            if(r<K && c<N){
                size_t byte_idx = (size_t(r)>>1)*N + c;
                uint8_t packed = B4[byte_idx];
                int8_t q = ( (r&1)==0 ) ? nibble_to_s4(packed) : nibble_to_s4(packed>>4);
                float w = float(q) * scale[c];
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

void quantize_fp4_per_col(const std::vector<__half>& B_half,int K,int N,
                          std::vector<uint8_t>& B4,std::vector<float>& scale){
    scale.resize(N);
    for(int n=0;n<N;n++){
        float m=0.f;
        for(int k=0;k<K;k++){ float v=fabsf(__half2float(B_half[k*N+n])); if(v>m)m=v; }
        scale[n] = m>0 ? m/7.0f : 1.0f;
    }
    B4.assign((K*N+1)/2,0);
    for(int n=0;n<N;n++){
        float s=scale[n];
        for(int k=0;k<K;k++){
            float x=__half2float(B_half[k*N+n]);
            int q=int(lrintf(x/s)); if(q<-8)q=-8; if(q>7)q=7;
            uint8_t u=uint8_t(q & 0xF);
            size_t byte_idx=(size_t(k)>>1)*N + n;
            if((k&1)==0) B4[byte_idx]=(B4[byte_idx]&0xF0)|u;
            else         B4[byte_idx]=(B4[byte_idx]&0x0F)|(u<<4);
        }
    }
}

int main(){
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    printf("GPU: %s, CC %d.%d, SMs %d, globalMem %.1f GB\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,prop.totalGlobalMem/1e9);

    // --- target runtimes (~20 s total) ---
    const double BUDGET_FP16_S = 7.0;
    const double BUDGET_FP4_S  = 7.0;
    const double BUDGET_KV_S   = 6.0;

    // problem sizes
    int M=8192,N=8192,K=8192;

    size_t freeB=0,totalB=0; CUDA_CHECK(cudaMemGetInfo(&freeB,&totalB));
    auto need_bytes = [&](int m,int n,int k){
        size_t A = size_t(m)*k*sizeof(__half);
        size_t B = size_t(k)*n*sizeof(__half);
        size_t C16= size_t(m)*n*sizeof(__half);
        size_t C32= size_t(m)*n*sizeof(float);
        size_t B_alt = B;                         // rotate weights
        size_t B4a = ((size_t)k*n + 1)/2;
        size_t B4b = B4a;                         // second 4-bit set
        size_t Scales = size_t(n)*sizeof(float)*2;
        return A + B + B_alt + C16 + C32 + B4a + B4b + Scales + (size_t)(0.25*totalB);
    };
    while (need_bytes(M,N,K) > freeB && M>=2048 && N>=2048 && K>=2048){ M/=2; N/=2; K/=2; }
    printf("Problem sizes: M=%d N=%d K=%d\n",M,N,K);

    // --- allocate and init A, B0, B1 ---
    __half *A=nullptr,*B0=nullptr,*B1=nullptr,*C16=nullptr;
    CUDA_CHECK(cudaMalloc(&A,  size_t(M)*K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B0, size_t(K)*N*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B1, size_t(K)*N*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&C16,size_t(M)*N*sizeof(__half)));

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    {
        std::vector<__half> hA(size_t(M)*K), hB0(size_t(K)*N), hB1(size_t(K)*N);
        for(auto& x:hA)  x=__float2half(dist(rng));
        for(auto& x:hB0) x=__float2half(dist(rng));
        for(auto& x:hB1) x=__float2half(dist(rng));
        CUDA_CHECK(cudaMemcpy(A,  hA.data(),  hA.size()*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B0, hB0.data(), hB0.size()*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B1, hB1.data(), hB1.size()*sizeof(__half), cudaMemcpyHostToDevice));
    }

    // cuBLAS
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    float alpha=1.f, beta=0.f;

    // FP16 GEMM alternating B0/B1
    bool toggle=false;
    auto gemm_fp16_once = [&](){
        const __half* B = toggle ? B1 : B0; toggle=!toggle;
        CHECK_CUBLAS(
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         B, CUDA_R_16F, N,
                         A, CUDA_R_16F, K,
                         &beta,
                         C16, CUDA_R_16F, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    float gemm16_ms = avg_ms_budget(gemm_fp16_once, BUDGET_FP16_S, /*chunk*/3, /*warmup*/3);
    CUDA_CHECK(cudaDeviceSynchronize());
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops16 = (flops / (gemm16_ms/1000.0)) / 1e12;

    // --- FP4 path: quantize two distinct B sets ---
    std::vector<__half> hB0(size_t(K)*N), hB1(size_t(K)*N);
    CUDA_CHECK(cudaMemcpy(hB0.data(), B0, hB0.size()*sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hB1.data(), B1, hB1.size()*sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> hB4a,hB4b; std::vector<float> hSa,hSb;
    quantize_fp4_per_col(hB0,K,N,hB4a,hSa);
    quantize_fp4_per_col(hB1,K,N,hB4b,hSb);
    uint8_t *B4a=nullptr,*B4b=nullptr; float *dSa=nullptr,*dSb=nullptr; float* C32=nullptr;
    CUDA_CHECK(cudaMalloc(&B4a, hB4a.size()));
    CUDA_CHECK(cudaMalloc(&B4b, hB4b.size()));
    CUDA_CHECK(cudaMalloc(&dSa, size_t(N)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSb, size_t(N)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C32, size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(B4a, hB4a.data(), hB4a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B4b, hB4b.data(), hB4b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSa, hSa.data(), size_t(N)*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSb, hSb.data(), size_t(N)*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C32, 0, size_t(M)*N*sizeof(float)));

    dim3 block(256);
    const int BM=128, BN=128, BK=64;
    dim3 grid( (N+BN-1)/BN, (M+BM-1)/BM );

    bool toggle4=false;
    auto gemm_fp4_once = [&](){
        const uint8_t* B4 = toggle4 ? B4b : B4a;
        const float*   Sc = toggle4 ? dSb : dSa; toggle4=!toggle4;
        gemm_fp4w_fused_kernel<BM,BN,BK><<<grid,block>>>(A,B4,Sc,C32,M,N,K);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    float gemm4_ms = avg_ms_budget(gemm_fp4_once, BUDGET_FP4_S, /*chunk*/2, /*warmup*/2);
    CUDA_CHECK(cudaDeviceSynchronize());
    double tflops4_eff = (flops / (gemm4_ms/1000.0)) / 1e12;

    // --- KV bandwidth for longer, with offset advance ---
    CUDA_CHECK(cudaMemGetInfo(&freeB,&totalB));
    size_t target = (size_t)(freeB * 0.65);
    size_t n_vec = target / (3ull * sizeof(float4));
    float4 *dK=nullptr,*dV=nullptr,*dO=nullptr;
    double gbps=0.0; float kv_ms=0.0f;
    if (n_vec >= (1<<20)){
        CUDA_CHECK(cudaMalloc(&dK, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&dV, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&dO, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dK,0,n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dV,0,n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dO,0,n_vec*sizeof(float4)));
        int threads=256;
        int blocks=int(std::min<size_t>((n_vec+threads-1)/threads,65535));
        size_t off=0, step = (1ull<<18); // ~1M elements stride
        auto kv_once = [&](){
            kv_stream_kernel<<<blocks,threads>>>(dK,dV,dO,n_vec,off);
            off += step; if (off >= n_vec) off -= n_vec;
        };
        CUDA_CHECK(cudaDeviceSynchronize());
        kv_ms = avg_ms_budget(kv_once, BUDGET_KV_S, /*chunk*/5, /*warmup*/3);
        CUDA_CHECK(cudaDeviceSynchronize());
        double moved = 3.0 * sizeof(float4) * double(n_vec);
        gbps = (moved / (kv_ms/1000.0)) / 1e9;
    }

    // --- scoring (unchanged) ---
    const double C0=20.0, M0=400.0;
    double compute_norm = fmax(1e-6, tflops16/C0);
    double mem_norm     = fmax(1e-6, gbps/M0);
    double fp4_frac = 0.5 + 0.35 * (compute_norm / (compute_norm + mem_norm));
    double t_mix = fp4_frac*(gemm4_ms/1000.0) + (1.0-fp4_frac)*(gemm16_ms/1000.0);
    double tflops_mix = flops / t_mix / 1e12;
    double p = 0.2 + 0.6 / (1.0 + compute_norm);
    double score = 1.0 / ( (p/fmax(1e-6,tflops_mix/C0)) + ((1.0-p)/fmax(1e-6,gbps/M0)) );
    double score100 = 100.0 * score;

    // --- output ---
    printf("\n=== AI bench (FP16 + FP4 mix) ===\n");
    printf("Budgets: FP16 %.1fs, FP4 %.1fs, KV %.1fs\n", BUDGET_FP16_S, BUDGET_FP4_S, BUDGET_KV_S);
    printf("FP16 TC GEMM: %.2f TFLOPS (avg %.2f ms)\n", tflops16, gemm16_ms);
    printf("FP4 fused-dequant GEMM (eff): %.2f TFLOPS (avg %.2f ms)\n", tflops4_eff, gemm4_ms);
    if (gbps>0) printf("KV bandwidth: %.1f GB/s (avg %.2f ms)\n", gbps, kv_ms);
    else        printf("KV bandwidth: skipped (insufficient memory)\n");
    printf("Anchors: C0=%.1f TFLOPS, M0=%.0f GB/s\n", C0, M0);
    printf("Pretest norms: compute=%.3f, memory=%.3f\n", compute_norm, mem_norm);
    printf("Selected FP4 fraction: %.2f\n", fp4_frac);
    printf("Mixed GEMM throughput: %.2f TFLOPS\n", tflops_mix);
    printf("Dynamic weight p=%.3f (higher compute -> lower p -> memory weighs more)\n", p);
    printf("Overall score: %.2f\n", score100);

    // --- inference estimates (unchanged) ---
    {
        printf("\n\n=== Inference-oriented estimates (tokens/s) ===\n");
        const int d=4096, Llayers=32, r=4, Bbatch=1;
        const double s_kv=2.0, s_w_fp16=2.0, s_w_fp4=0.5, beta_w=1.2;
        if (gbps<=0){ printf("Bandwidth test missing. Skipping inference estimates.\n"); }
        else{
            auto toks_per_s = [&](double t, double s_w)->double{
                const double F_layer = (8.0+4.0*r)*(double)d*(double)d + 4.0*t*(double)d;
                const double F_total = (double)Llayers*F_layer*Bbatch;
                const double Tcomp = F_total / (tflops_mix*1e12);
                const double Q_kv = (double)Llayers*2.0*Bbatch*(double)d*s_kv*t;
                const double Q_w  = (double)Llayers*beta_w*(double)d*(double)d*s_w;
                const double Tmem = (Q_kv + Q_w) / (gbps*1e9);
                const double T = fmax(Tcomp,Tmem);
                return 1.0/T;
            };
            const int ts[3]={512,2048,8192};
            printf("Using: d=%d, L=%d, r=%d, batch=%d, KV=FP16\n", d,Llayers,r,Bbatch);
            printf("Measured: mixed_GEMM=%.2f TFLOPS, BW=%.1f GB/s\n", tflops_mix, gbps);
            for(int i=0;i<3;i++){
                int t=ts[i];
                double a=toks_per_s(t,s_w_fp16), b=toks_per_s(t,s_w_fp4);
                printf("t=%4d: FP16-weights %.1f tok/s, FP4-weights %.1f tok/s\n", t,a,b);
            }
        }
    }

    if (A) cudaFree(A); if (B0) cudaFree(B0); if (B1) cudaFree(B1); if (C16) cudaFree(C16);
    cudaFree(B4a); cudaFree(B4b); cudaFree(dSa); cudaFree(dSb); cudaFree(C32);
    if (dK) cudaFree(dK); if (dV) cudaFree(dV); if (dO) cudaFree(dO);
    cublasDestroy(h);
    return 0;
}

