#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1.env"
STAGE_C_MARKER="${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v1/stage_c_source_proxy_15shard_v1/STAGE_C_SOURCE_PROXY_COMPLETE.json"
STAGE_D_DATA="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1"
LOG="${REPO_ROOT}/logs/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1/pipeline.log"
mkdir -p "$(dirname "${LOG}")"

while [[ ! -f "${STAGE_C_MARKER}" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for Stage C completion" | tee -a "${LOG}"
  sleep 30
done
while [[ ! -f "${STAGE_D_DATA}/train/STAGE_D_PROXY_DATA_READY.json" || ! -f "${STAGE_D_DATA}/valid/STAGE_D_PROXY_DATA_READY.json" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for Stage D train/valid data" | tee -a "${LOG}"
  sleep 30
done

echo "[$(date -u +%FT%TZ)] running Stage D one-GPU/two-iteration smoke" | tee -a "${LOG}"
STAGE_D_SAVE_ROOT="${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1_smoke" \
STAGE_D_TENSORBOARD_DIR="${REPO_ROOT}/runs/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1_smoke/tensorboard" \
scripts/simul_uniss/train_qwen_stage.sh \
  --stage interleaved --smoke --config "${CONFIG}" 2>&1 | tee -a "${LOG}"
touch "${STAGE_D_DATA}/STAGE_D_GPU_SMOKE_COMPLETE"

tmux has-session -t uniss_stage_d_proxy_tb 2>/dev/null || \
  tmux new-session -d -s uniss_stage_d_proxy_tb \
    "cd '${REPO_ROOT}' && scripts/simul_uniss_subsecond_v1/start_stage_d_tensorboard.sh"

echo "[$(date -u +%FT%TZ)] starting Stage D formal eight-GPU training" | tee -a "${LOG}"
exec scripts/simul_uniss/train_qwen_stage.sh --stage interleaved --config "${CONFIG}" \
  2>&1 | tee -a "${LOG}"
