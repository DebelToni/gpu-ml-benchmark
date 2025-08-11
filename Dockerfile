
### Dockerfile
```dockerfile
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai_bench_fp4.cu /app/
RUN nvcc -O3 -std=c++17 /app/ai_bench_fp4.cu -lcublas -o /usr/local/bin/ai_bench_fp4

RUN mkdir -p /logs
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

