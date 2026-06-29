#!/usr/bin/env bash
set -euo pipefail
vllm serve google/gemma-4-12B-it \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
