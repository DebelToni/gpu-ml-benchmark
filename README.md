# AI Mini Bench (FP16 + FP4 mix)

Single-binary CUDA benchmark for NVIDIA GPUs. Measures:
- FP16 Tensor Core GEMM TFLOPS (cuBLAS)
- FP4 weight-only GEMM (fused dequant) effective TFLOPS
- KV-like VRAM bandwidth (GB/s)
- Training-style score + inference token/s estimates


All the params for the benchmark are displayed and can be adusted to match your usecase best.


Here are your options for running this benchmark:

## Run from Docker Hub
Requires NVIDIA Container Toolkit.
```bash
docker run --rm --gpus all -v "$(pwd)/logs:/logs" bonanc/gpu-ml-benchmark:latest
```

---

## Requirements
- NVIDIA GPU + driver
- CUDA 12.9+ toolchain (or use Docker)
- File in repo root: `ai_bench_fp4.cu` (from chat above)

## Native build
```bash
nvcc -O3 -std=c++17 ai_bench_fp4.cu -lcublas -o ai_bench_fp4
./ai_bench_fp4
```

or used the execuatble if you trust random source, note you will still need to install the libraries
