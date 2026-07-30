#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-/opt/dlami/nvme/jasonleeeli/env_recovery/uniss-train-20260721/activate_uniss.sh}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
ROOT="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1"
LOG_ROOT="${REPO_ROOT}/logs/simul_uniss_subsecond_v1/stage_d_micro_write_proxy_15shard_v1"
mkdir -p "${ROOT}" "${LOG_ROOT}"

TRAIN_REPLAY=()
for index in $(seq -w 0 14); do
  TRAIN_REPLAY+=("${REPO_ROOT}/data/processed/phase3_unist198_sharded/train-000${index}.jsonl")
done

python -m training.simul_uniss.subsecond_v1.prepare_stage_d \
  --schedules "${REPO_ROOT}/data/processed/simul_uniss_v1/bootstrap_15shard/schedules.jsonl" \
  --phase3-replay "${TRAIN_REPLAY[@]}" \
  --output-dir "${ROOT}/train" \
  --seq-length 18000 \
  --replay-ratio 0.30 \
  --progress-interval 1000 \
  2>&1 | tee -a "${LOG_ROOT}/prepare_train.log"

python -m training.simul_uniss.subsecond_v1.prepare_stage_d \
  --schedules "${REPO_ROOT}/data/processed/simul_uniss_v1/validation_dev/schedules.jsonl" \
  --phase3-replay "${REPO_ROOT}/data/processed/validation_unist_dev/phase3_dev.jsonl" \
  --output-dir "${ROOT}/valid" \
  --seq-length 18000 \
  --replay-ratio 0.30 \
  --progress-interval 100 \
  2>&1 | tee -a "${LOG_ROOT}/prepare_valid.log"
