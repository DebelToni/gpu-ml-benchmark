// ai_bench_fp4.cu
// CUDA 12+. Single file. No CUTLASS.
// Subtests: FP16 TC GEMM (cuBLAS), fused FP4(weight-only)+FP16 accumulate GEMM kernel, KV-stream bandwidth.
// Dynamic mix: at least 50% of GEMMs use FP4; increases toward FP4 on compute-strong GPUs.
// Notes: RTX 3060 has no native FP4. We emulate weight-only 4-bit with per-column scales and fused dequant.

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

// #define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
//   fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CUDA_CHECK(x) do {                                      \
  cudaError_t _err = (x);                                       \
  if (_err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
            cudaGetErrorString(_err));                             \
    exit(1);                                                       \
  }                                                                \
} while (0)

#define CHECK_CUBLAS(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)s); exit(1);} } while(0)

template <typename F>
float time_ms(F f, int iters, int warmup=3) {
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    for(int i=0;i<warmup;++i) f();
    CUDA_CHECK(cudaEventRecord(s));
    for(int i=0;i<iters;++i) f();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return ms/iters;
}

// =======================
// KV-like bandwidth kernel
// =======================
__global__ void kv_stream_kernel(const float4* __restrict__ K,
                                 const float4* __restrict__ V,
                                 float4* __restrict__ O,
                                 size_t n_vec) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = gridDim.x * blockDim.x;
    for (; i < n_vec; i += s) {
        float4 a = K[i];
        float4 b = V[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        O[i] = c;
    }
}

// =======================
// FP4 fused-dequant GEMM
// C[M,N] = A[M,K] (half) * dequant4(B4[K,N], scale[N])
// B4 packs two signed 4-bit values per byte: q in [-8,7], stored as (uint8)(q & 0xF)
// Per-column scale: w ≈ q * scale[col]
// Tiled, shared-memory, half FMA accumulate to float.
// =======================
__device__ __forceinline__ int8_t nibble_to_s4(uint8_t v) {
    // interpret low 4 bits as signed [-8,7]
    v &= 0xF;
    return (v & 0x8) ? int8_t(v) - 16 : int8_t(v);
}

template<int BM, int BN, int BK>
__global__ void gemm_fp4w_fused_kernel(const __half* __restrict__ A,  // MxK, row-major
                                       const uint8_t* __restrict__ B4, // KxN, row-major, 2 elems/byte
                                       const float* __restrict__ scale, // N scales
                                       float* __restrict__ C,          // MxN, row-major (fp32 out)
                                       int M, int N, int K)
{
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];

    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    // per-thread accumulator
    // map threads as 16x16 = 256 threads per block; each does a  (BM/16)x(BN/16) micro-tile
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;

    const int RM = BM / 16; // rows per thread
    const int RN = BN / 16; // cols per thread
    float acc[RM][RN];
    #pragma unroll
    for (int i=0;i<RM;i++) for (int j=0;j<RN;j++) acc[i][j]=0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // load A tile [BM x BK]
        #pragma unroll
        for (int i=ty; i<BM; i+=16) {
            #pragma unroll
            for (int j=tx; j<BK; j+=16) {
                int r = row0 + i;
                int c = k0 + j;
                As[i][j] = (r<M && c<K) ? A[r*K + c] : __float2half(0.0f);
            }
        }
        // load and dequant B4 tile [BK x BN] -> Bs as half
        #pragma unroll
        for (int i=ty; i<BK; i+=16) {
            #pragma unroll
            for (int j=tx; j<BN; j+=16) {
                int r = k0 + i; // K index
                int c = col0 + j; // N index
                __half h = __float2half(0.0f);
                if (r<K && c<N) {
                    // B4 is row-major KxN: index (r,c) -> packed byte idx = r*N + c, two per byte over K dim? We pack over K fastest.
                    // Packing scheme: along K, two consecutive k elements for same col are in one byte.
                    size_t linear = size_t(r)*N + c; // element index in KxN
                    // byte index over K: floor(k/2)*N + c
                    size_t byte_idx = (size_t(r)>>1)*N + c;
                    uint8_t packed = B4[byte_idx];
                    int8_t q = ( (r & 1)==0 ) ? nibble_to_s4(packed) : nibble_to_s4(packed>>4);
                    float w = float(q) * scale[c];
                    h = __float2half(w);
                }
                Bs[i][j] = h;
            }
        }
        __syncthreads();

        // compute micro tile
        #pragma unroll
        for (int kk=0; kk<BK; ++kk) {
            __half a_reg[RM];
            __half b_reg[RN];
            // each thread loads its row/col elements
            #pragma unroll
            for (int i=0;i<RM;i++) a_reg[i] = As[ty*RM + i][kk];
            #pragma unroll
            for (int j=0;j<RN;j++) b_reg[j] = Bs[kk][tx*RN + j];
            #pragma unroll
            for (int i=0;i<RM;i++) {
                float av = __half2float(a_reg[i]);
                #pragma unroll
                for (int j=0;j<RN;j++) {
                    acc[i][j] += av * __half2float(b_reg[j]);
                }
            }
        }
        __syncthreads();
    }

    // store
    #pragma unroll
    for (int i=0;i<RM;i++) {
        int r = row0 + ty*RM + i;
        if (r>=M) continue;
        #pragma unroll
        for (int j=0;j<RN;j++) {
            int c = col0 + tx*RN + j;
            if (c<N) C[r*N + c] = acc[i][j];
        }
    }
}

// Host-side quantize B (half) to signed 4-bit with per-column scales.
void quantize_fp4_per_col(const std::vector<__half>& B_half, int K, int N,
                          std::vector<uint8_t>& B4, std::vector<float>& scale) {
    scale.resize(N);
    // find max abs per column
    for (int n=0;n<N;n++){
        float m = 0.0f;
        for (int k=0;k<K;k++){
            float v = fabsf(__half2float(B_half[k*N + n]));
            if (v>m) m=v;
        }
        scale[n] = m > 0 ? m / 7.0f : 1.0f; // map to [-8,7] -> approx 7*scale
    }
    B4.assign((K*N + 1)/2, 0);
    for (int n=0;n<N;n++){
        float s = scale[n];
        for (int k=0;k<K;k++){
            float x = __half2float(B_half[k*N + n]);
            int q = int(lrintf(x / s));
            if (q < -8) q = -8; if (q > 7) q = 7;
            uint8_t u = uint8_t(q & 0xF);
            size_t byte_idx = (size_t(k)>>1)*N + n;
            if ((k & 1)==0) B4[byte_idx] = (B4[byte_idx] & 0xF0) | u;
            else            B4[byte_idx] = (B4[byte_idx] & 0x0F) | (u<<4);
        }
    }
}

int main(){
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, CC %d.%d, SMs %d, globalMem %.1f GB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem/1e9);

    // Sizes (reduce if VRAM tight). Keep multiples of 128 for TC GEMM.
    int M = 8192, N = 8192, K = 8192;

    size_t freeB=0,totalB=0; CUDA_CHECK(cudaMemGetInfo(&freeB,&totalB));
    auto need_bytes = [&](int m,int n,int k){
        size_t A = size_t(m)*k*sizeof(__half);
        size_t B = size_t(k)*n*sizeof(__half);
        size_t C = size_t(m)*n*sizeof(__half);
        size_t B4 = ((size_t)k*n + 1)/2; // bytes
        size_t S  = size_t(n)*sizeof(float);
        // for both FP16 and FP4 paths plus temp
        return A + B + C + B4 + S + (size_t)(0.3*totalB); // headroom
    };
    while (need_bytes(M,N,K) > freeB && M>=2048 && N>=2048 && K>=2048) { M/=2; N/=2; K/=2; }

    printf("Problem sizes: M=%d N=%d K=%d\n", M,N,K);

    // Allocate A,B,C for FP16 GEMM
    __half *A=nullptr,*B=nullptr,*C16=nullptr;
    CUDA_CHECK(cudaMalloc(&A, size_t(M)*K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B, size_t(K)*N*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&C16, size_t(M)*N*sizeof(__half)));
    // Host init
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    {
        std::vector<__half> hA(size_t(M)*K), hB(size_t(K)*N);
        for (auto& x: hA) x = __float2half(dist(rng));
        for (auto& x: hB) x = __float2half(dist(rng));
        CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size()*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, hB.data(), hB.size()*sizeof(__half), cudaMemcpyHostToDevice));
    }

    // cuBLAS FP16 TC GEMM
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    float alpha=1.0f, beta=0.0f;
    auto gemm_fp16_once = [&](){
        CHECK_CUBLAS(
            cublasGemmEx(h,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, CUDA_R_16F, N,
                A, CUDA_R_16F, K,
                &beta,
                C16, CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
        );
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    float gemm16_ms = time_ms(gemm_fp16_once, 10, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops16 = (flops / (gemm16_ms/1000.0)) / 1e12;

    // Prepare FP4 weights (host quantize, device upload)
    std::vector<__half> hB(size_t(K)*N);
    CUDA_CHECK(cudaMemcpy(hB.data(), B, hB.size()*sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> hB4; std::vector<float> hScale;
    quantize_fp4_per_col(hB, K, N, hB4, hScale);
    uint8_t *B4=nullptr; float* dScale=nullptr; float* C32=nullptr;
    CUDA_CHECK(cudaMalloc(&B4, hB4.size()));
    CUDA_CHECK(cudaMalloc(&dScale, size_t(N)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C32, size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(B4, hB4.data(), hB4.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dScale, hScale.data(), size_t(N)*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C32, 0, size_t(M)*N*sizeof(float)));

    // Launch FP4 fused kernel, tile params
    dim3 block(256);
    const int BM=128, BN=128, BK=64;
    dim3 grid( (N+BN-1)/BN, (M+BM-1)/BM );

    auto gemm_fp4_once = [&](){
        gemm_fp4w_fused_kernel<BM,BN,BK><<<grid, block>>>(A, B4, dScale, C32, M,N,K);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    float gemm4_ms = time_ms(gemm_fp4_once, 5, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    double tflops4_eff = (flops / (gemm4_ms/1000.0)) / 1e12; // effective, counts 2MNK

    // KV stream bandwidth
    CUDA_CHECK(cudaMemGetInfo(&freeB,&totalB));
    size_t target = (size_t)(freeB * 0.65);
    size_t n_vec = target / (3ull * sizeof(float4));
    float4 *dK=nullptr,*dV=nullptr,*dO=nullptr;
    double gbps=0.0; float kv_ms=0.0f;
    if (n_vec >= (1<<20)) {
        CUDA_CHECK(cudaMalloc(&dK, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&dV, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&dO, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dK, 0, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dV, 0, n_vec*sizeof(float4)));
        CUDA_CHECK(cudaMemset(dO, 0, n_vec*sizeof(float4)));
        int threads = 256;
        int blocks = int(std::min<size_t>((n_vec + threads - 1)/threads, 65535));
        auto kv_once = [&](){ kv_stream_kernel<<<blocks,threads>>>(dK,dV,dO,n_vec); };
        CUDA_CHECK(cudaDeviceSynchronize());
        kv_ms = time_ms(kv_once, 10, 3);
        CUDA_CHECK(cudaDeviceSynchronize());
        double moved = 3.0 * sizeof(float4) * double(n_vec);
        gbps = (moved / (kv_ms/1000.0)) / 1e9;
    } else {
        gbps = 0.0;
        kv_ms = NAN;
    }

    // Dynamic FP4 share. More FP4 when compute strong relative to memory.
    // compute_norm from FP16 TC, mem_norm from bandwidth.
    const double C0 = 20.0;   // TFLOPS anchor
    const double M0 = 400.0;  // GB/s anchor
    double compute_norm = fmax(1e-6, tflops16 / C0);
    double mem_norm     = fmax(1e-6, gbps     / M0);
    // fp4_frac in [0.5, 0.85]
    double fp4_frac = 0.5 + 0.35 * (compute_norm / (compute_norm + mem_norm));

    // Mixed GEMM effective TFLOPS via time-weighted harmonic mean of per-op times
    // t = f*(F/t4)^{-1} + (1-f)*(F/t16)^{-1} => throughput = F / t
    double t_fp4 = gemm4_ms/1000.0;
    double t_fp16 = gemm16_ms/1000.0;
    double t_mix = fp4_frac * t_fp4 + (1.0 - fp4_frac) * t_fp16;
    double tflops_mix = flops / t_mix / 1e12;

    // Overall score with memory-weight shift as compute rises
    double p = 0.2 + 0.6 / (1.0 + compute_norm);
    double mem_norm2 = fmax(1e-6, gbps / M0);
    double comp_norm2 = fmax(1e-6, tflops_mix / C0);
    double score = 1.0 / ( (p/comp_norm2) + ((1.0-p)/mem_norm2) );
    double score100 = 100.0 * score;

    // Print
    printf("\n=== AI bench (FP16 + FP4 mix) ===\n");
    printf("FP16 TC GEMM: %.2f TFLOPS (avg %.2f ms)\n", tflops16, gemm16_ms);
    printf("FP4 fused-dequant GEMM (eff): %.2f TFLOPS (avg %.2f ms)\n", tflops4_eff, gemm4_ms);
    if (gbps>0) printf("KV bandwidth: %.1f GB/s (avg %.2f ms)\n", gbps, kv_ms);
    else        printf("KV bandwidth: skipped (insufficient memory)\n");

    printf("Anchors: C0=%.1f TFLOPS, M0=%.0f GB/s\n", C0, M0);
    printf("Pretest norms: compute=%.3f, memory=%.3f\n", compute_norm, mem_norm);
    printf("Selected FP4 fraction: %.2f\n", fp4_frac);
    printf("Mixed GEMM throughput: %.2f TFLOPS\n", tflops_mix);

    double p_show = p;
    printf("Dynamic weight p=%.3f (higher compute -> lower p -> memory weighs more)\n", p_show);
    printf("Overall score: %.2f\n", score100);
// ------------------------------------------------------------
// Inference-oriented estimates (tokens/s) using measured C & BW
// Append this block after printing "Overall score"
// ------------------------------------------------------------

{
    printf("\n\n=== Inference-oriented estimates (tokens/s) ===\n");
    // Model knobs (edit to taste)
    const int d = 4096;       // model dim
    const int Llayers = 32;   // transformer layers
    const int r = 4;          // MLP expansion
    const int Bbatch = 1;     // decode batch size
    const double s_kv = 2.0;  // bytes per KV element (FP16 cache)
    const double s_w_fp16 = 2.0; // bytes per weight (FP16)
    const double s_w_fp4  = 0.5; // bytes per weight (4-bit packed, scales amortized)
    const double beta_w = 1.2;   // weight reload factor (tiling/overhead)

    if (gbps <= 0) {
        printf("Bandwidth test missing. Skipping inference estimates.\n");
    } else {
        auto toks_per_s = [&](double t, double s_w)->double {
            // Per-token FLOPs per *layer* (forward decode)
            // F_layer ≈ (8 + 4r) d^2 + 4 t d
            const double F_layer = (8.0 + 4.0*r) * (double)d * (double)d + 4.0 * t * (double)d;
            const double F_total = (double)Llayers * F_layer * Bbatch;
            // Compute time from mixed GEMM throughput
            const double Tcomp = F_total / (tflops_mix * 1e12);
            // Memory bytes: KV read+write across all layers + weight streaming
            const double Q_kv = (double)Llayers * 2.0 * Bbatch * (double)d * s_kv * t;
            const double Q_w  = (double)Llayers * beta_w * (double)d * (double)d * s_w;
            const double Tmem = (Q_kv + Q_w) / (gbps * 1e9);
            const double T = fmax(Tcomp, Tmem);
            return 1.0 / T;
        };

        const int t_list[3] = {512, 2048, 8192};
        printf("Using: d=%d, L=%d, r=%d, batch=%d, KV=FP16\n", d, Llayers, r, Bbatch);
        printf("Measured: mixed_GEMM=%.2f TFLOPS, BW=%.1f GB/s\n", tflops_mix, gbps);
        for (int i=0;i<3;i++){
            int t = t_list[i];
            double ts_fp16 = toks_per_s(t, s_w_fp16);
            double ts_fp4  = toks_per_s(t, s_w_fp4);
            printf("t=%4d: FP16-weights %.1f tok/s, FP4-weights %.1f tok/s\n", t, ts_fp16, ts_fp4);
        }
    }
}


    // Cleanup
    if (dK) cudaFree(dK); if (dV) cudaFree(dV); if (dO) cudaFree(dO);
    cudaFree(A); cudaFree(B); cudaFree(C16);
    cudaFree(B4); cudaFree(dScale); cudaFree(C32);
    cublasDestroy(h);
    return 0;
}

