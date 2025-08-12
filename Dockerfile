FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai_bench.cu /app/

RUN nvcc -O3 -std=c++17 /app/ai_bench.cu -o /usr/local/bin/ai_bench \
    -lcublasLt -lcublas \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90a \
    -gencode arch=compute_100,code=sm_100

CMD ["/usr/local/bin/ai_bench"]

