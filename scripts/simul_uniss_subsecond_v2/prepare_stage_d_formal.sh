#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_D_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_d_formal_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
mkdir -p "${STAGE_D_TRAIN_ROOT}" "${STAGE_D_VALID_ROOT}" "${LOG_DIR}"

train_replay=()
for index in $(seq -w 0 14); do
  train_replay+=("${REPO_ROOT}/data/processed/phase3_unist198_sharded/train-000${index}.jsonl")
done
for path in "${FORMAL_STAGE_A_ROOT}/formal_train_manifest.jsonl" "${FORMAL_STAGE_A_ROOT}/formal_valid_manifest.jsonl" "${train_replay[@]}" "${REPO_ROOT}/data/processed/validation_unist_dev/phase3_dev.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing required Stage-D input: ${path}" >&2; exit 1; }
done

python -m training.simul_uniss.subsecond_v2.prepare_stage_d \
  --input-manifest "${FORMAL_STAGE_A_ROOT}/formal_train_manifest.jsonl" \
  --phase3-replay "${train_replay[@]}" \
  --output-dir "${STAGE_D_TRAIN_ROOT}" \
  --tokenizer "${REPO_ROOT}/pretrained_models/UniSS" \
  --tick-ms 160 --seq-length 18000 --replay-ratio 0.30 --progress-interval 1000 \
  2>&1 | tee -a "${LOG_DIR}/prepare_train.log"

python -m training.simul_uniss.subsecond_v2.prepare_stage_d \
  --input-manifest "${FORMAL_STAGE_A_ROOT}/formal_valid_manifest.jsonl" \
  --phase3-replay "${REPO_ROOT}/data/processed/validation_unist_dev/phase3_dev.jsonl" \
  --output-dir "${STAGE_D_VALID_ROOT}" \
  --tokenizer "${REPO_ROOT}/pretrained_models/UniSS" \
  --tick-ms 160 --seq-length 18000 --replay-ratio 0.30 --progress-interval 100 \
  2>&1 | tee -a "${LOG_DIR}/prepare_valid.log"
