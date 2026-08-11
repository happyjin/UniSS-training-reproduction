#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SMOKE_PACKED="${DATA_ROOT}/smoke128_v2/packed_dense_v3.jsonl"
SMOKE_ROOT="${PACKED_ROOT}/smoke8_v5"
SMOKE_MANIFEST="${SMOKE_ROOT}/training_manifest.json"
SMOKE_SAVE="${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}_smoke8_v5"
SMOKE_TB="${REPO_ROOT}/runs/${EXPERIMENT_NAME}_smoke8_v5/tensorboard"
SMOKE_LOG="${REPO_ROOT}/logs/${EXPERIMENT_NAME}_smoke8_v5.log"

"${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_training_manifest \
  --trajectory-packed "${SMOKE_PACKED}" \
  --replay-packed "${PHASE3_REPLAY_PACKED}" \
  --replay-offsets "${PHASE3_REPLAY_OFFSETS}" \
  --output-root "${SMOKE_ROOT}" \
  --coverage-epochs "${COVERAGE_EPOCHS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --data-parallel-size "${NPROC_PER_NODE}" \
  --replay-fraction "${REPLAY_FRACTION}" \
  --seed "${SEED}"

read -r TRAIN_ITERS WARMUP_ITERS REPLAY_SUBSET < <(
  "${PYTHON}" - "${SMOKE_MANIFEST}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["train_iters"], value["warmup_iters"], value["replay_subset_offsets"])
PY
)

TRACKER="${SMOKE_SAVE}/latest_checkpointed_iteration.txt"
if [[ -s "${TRACKER}" ]] && [[ "$(tr -d '[:space:]' < "${TRACKER}")" == "${TRAIN_ITERS}" ]]; then
  echo "8-GPU smoke already complete at ${TRAIN_ITERS}"
  exit 0
fi
if [[ -e "${SMOKE_SAVE}" && -n "$(find "${SMOKE_SAVE}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite an incomplete smoke directory: ${SMOKE_SAVE}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
RUN_NAME="${EXPERIMENT_NAME}_smoke8_v1" RUN_SAVE_DIR="${SMOKE_SAVE}" \
RUN_TB_DIR="${SMOKE_TB}" RUN_LOG="${SMOKE_LOG}" \
RUN_TRAINING_MANIFEST="${SMOKE_MANIFEST}" \
RUN_TRAJECTORY_PACKED="${SMOKE_PACKED}" \
RUN_TRAJECTORY_OFFSETS="${SMOKE_PACKED}.offsets.bin" \
RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" \
RUN_REPLAY_OFFSETS="${REPLAY_SUBSET}" \
RUN_VALID_TRAJECTORY_PACKED="${VALID_TRAJECTORY_PACKED}" \
RUN_VALID_TRAJECTORY_OFFSETS="${VALID_TRAJECTORY_OFFSETS}" \
RUN_VALID_REPLAY_PACKED="${VALID_REPLAY_PACKED}" \
RUN_VALID_REPLAY_OFFSETS="${VALID_REPLAY_OFFSETS}" \
RUN_FULL_VALIDATION=1 RUN_EVAL_INTERVAL=3 EVAL_ITERS=1 \
RUN_TRAIN_ITERS="${TRAIN_ITERS}" RUN_WARMUP_ITERS="${WARMUP_ITERS}" \
RUN_NPROC="${NPROC_PER_NODE}" RUN_MBS="${MICRO_BATCH_SIZE}" \
RUN_GBS="${GLOBAL_BATCH_SIZE}" RUN_MASTER_PORT=29732 \
RUN_LOAD="${PHASE3_NATIVE_CHECKPOINT}" RUN_FINETUNE=1 \
RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 RUN_STRICTNESS=log_all \
RUN_SMOKE=1 RUN_AUDIT_GRADIENTS=1 RUN_SAVE_INTERVAL=1 \
bash "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh" "$@"

[[ "${DRY_RUN}" == "1" ]] && exit 0

actual="$(tr -d '[:space:]' < "${TRACKER}")"
[[ "${actual}" == "${TRAIN_ITERS}" ]] || {
  echo "8-GPU smoke stopped at ${actual}, expected ${TRAIN_ITERS}" >&2
  exit 1
}
