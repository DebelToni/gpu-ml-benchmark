#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs
ts="$(date -Iseconds | tr ':' '-')"
log="/logs/ai_bench_${ts}.txt"
echo "Writing log to $log"
nvidia-smi || true
/usr/local/bin/ai_bench | tee "$log"

