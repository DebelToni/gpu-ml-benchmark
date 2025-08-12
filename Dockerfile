# Minimal image that builds and runs the benchmark.
# Uses CUDA 12.9 devel (cuBLASLt FP4 is in 12.9+/13.x). Change tag if you have 13.x locally.

FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai_bench.cu /app/ai_bench.cu

# Build for Ampere/Hopper/Blackwell. Add more gencodes if needed.
RUN nvcc -O3 -std=c++17 ai_bench.cu -o /usr/local/bin/ai_bench \
    -lcublasLt -lcublas \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90a \
    -gencode arch=compute_100,code=sm_100

# Default command: run and print
CMD ["/usr/local/bin/ai_bench"]

