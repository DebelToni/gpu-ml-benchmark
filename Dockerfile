# CUDA 13.0 has FP4/FP6/FP8 intrinsics and cuBLASLt FP4/FP8 types.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai_bench.cu /app/

# Build for Ampere, Ada, Hopper, Blackwell
RUN nvcc -O3 -std=c++17 /app/ai_bench.cu -o /usr/local/bin/ai_bench \
    -lcublasLt -lcublas \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90a \
    -gencode arch=compute_100,code=sm_100 \
    -gencode arch=compute_110,code=sm_110 \
    -gencode arch=compute_120,code=sm_120

CMD ["/usr/local/bin/ai_bench"]

