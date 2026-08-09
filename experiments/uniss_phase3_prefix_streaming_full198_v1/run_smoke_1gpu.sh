#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
env \
  RUN_NAME="uniss_phase3_prefix_streaming_smoke1_${RUN_ID}" \
  CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MASTER_PORT=29666 \
  TRAIN_ITERS=2 MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=2 \
  WARMUP_ITERS=1 NUM_WORKERS=2 SAVE_INTERVAL=2 EVAL_INTERVAL=1 EVAL_ITERS=1 \
  VALID_LIMIT=16 SMOKE=1 \
  bash "${ROOT}/experiments/uniss_phase3_prefix_streaming_full198_v1/run_megatron.sh"

