#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
cd "${REPO_ROOT}"

for value in "${TRAJECTORY_MANIFEST}" "${VALID_TRAJECTORY_MANIFEST}" \
  "${DATA_AUDIT}" "${TRAINING_MANIFEST}" "${REPLAY_SUBSET_OFFSETS}" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${value}" ]] || { echo "missing required input: ${value}" >&2; exit 2; }
done
[[ "$(tr -d '[:space:]' < "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt")" == 9075 ]] || {
  echo "Phase-3 root is not pinned to iter_0009075" >&2; exit 2;
}

read -r TRAIN_ITERS WARMUP_ITERS MANIFEST_EPOCHS < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
assert value['status']=='complete'
assert value['strict_global_shuffle']
assert value['multi_file_prefix_sum_namespace']
print(value['train_iters'], value['warmup_iters'], value['coverage_epochs'])
PY
)
[[ "${MANIFEST_EPOCHS}" == "${COVERAGE_EPOCHS}" ]] || {
  echo "COVERAGE_EPOCHS differs from frozen manifest" >&2; exit 2;
}
[[ "${COVERAGE_EPOCHS}" == 1 ]] || { echo "formal v1 is exactly one full epoch" >&2; exit 2; }

TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
if [[ -s "${TRACKER}" ]]; then
  CURRENT="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( CURRENT == TRAIN_ITERS )); then echo "complete at ${CURRENT}"; exit 0; fi
  (( CURRENT < TRAIN_ITERS )) || { echo "checkpoint exceeds target" >&2; exit 2; }
  LOAD="${SAVE_DIR}"; FINETUNE=0; LOAD_OPTIM=1; LOAD_RNG=1; STRICTNESS=raise_all
else
  [[ ! -e "${SAVE_DIR}" || -z "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "refusing a fresh run in nonempty ${SAVE_DIR}" >&2; exit 3;
  }
  LOAD="${PHASE3_NATIVE_CHECKPOINT}"; FINETUNE=1; LOAD_OPTIM=0; LOAD_RNG=0; STRICTNESS=log_all
fi

mkdir -p "${RUN_DIR}" "${REPORT_ROOT}"
cat > "${RUN_DIR}/manifest.txt" <<EOF
experiment=${EXPERIMENT_NAME}
created_at=$(date -u +%FT%TZ)
repo_commit=$(git rev-parse HEAD)
base_checkpoint=${PHASE3_NATIVE_CHECKPOINT}/iter_0009075
scope=fixed_unist_shards_00000_00014
phrase_minimum_tokens=4
coverage_epochs=${COVERAGE_EPOCHS}
train_iters=${TRAIN_ITERS}
shuffle=full_randperm_over_global_multifile_complete_pack_ids
tensorboard_dir=${TB_DIR}
EOF

export CUDA_VISIBLE_DEVICES COVERAGE_EPOCHS REPLAY_FRACTION
export LR_QWEN_LORA LR_FRONTEND LR_NEW_HEADS MIN_LR WEIGHT_DECAY NUM_WORKERS NPZ_LRU_CAPACITY SEED
FIRST_TRAIN_PACK="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parts"][0]["packed"])' "${TRAJECTORY_MANIFEST}")"
FIRST_VALID_PACK="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parts"][0]["packed"])' "${VALID_TRAJECTORY_MANIFEST}")"
export RUN_NAME="${EXPERIMENT_NAME}" RUN_SAVE_DIR="${SAVE_DIR}" RUN_TB_DIR="${TB_DIR}"
export RUN_LOG="${LOG_PATH}" RUN_TRAINING_MANIFEST="${TRAINING_MANIFEST}"
export RUN_TRAJECTORY_PACKED="${FIRST_TRAIN_PACK}" RUN_TRAJECTORY_OFFSETS="${FIRST_TRAIN_PACK}.offsets.bin"
export RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" RUN_REPLAY_OFFSETS="${REPLAY_SUBSET_OFFSETS}"
export RUN_VALID_TRAJECTORY_PACKED="${FIRST_VALID_PACK}" RUN_VALID_TRAJECTORY_OFFSETS="${FIRST_VALID_PACK}.offsets.bin"
export RUN_VALID_REPLAY_PACKED="" RUN_VALID_REPLAY_OFFSETS=""
export RUN_ENTRYPOINT="${HERE}/training/pretrain_content_first_megatron.py"
export RUN_LORA_DROPOUT=0 RUN_ATTENTION_DROPOUT=0 RUN_HIDDEN_DROPOUT=0
export RUN_FULL_VALIDATION=1 RUN_EVAL_INTERVAL="${EVAL_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}"
export RUN_TRAIN_ITERS="${TRAIN_ITERS}" RUN_WARMUP_ITERS="${WARMUP_ITERS}"
export RUN_NPROC=8 RUN_MBS="${MICRO_BATCH_SIZE}" RUN_GBS="${GLOBAL_BATCH_SIZE}"
export RUN_SEQ_LENGTH="${SEQ_LENGTH}" RUN_MASTER_PORT="${MASTER_PORT}" RUN_LOAD="${LOAD}"
export RUN_FINETUNE="${FINETUNE}" RUN_LOAD_OPTIM="${LOAD_OPTIM}" RUN_LOAD_RNG="${LOAD_RNG}"
export RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE="${RUN_SMOKE:-0}" RUN_AUDIT_GRADIENTS=1
export RUN_SAVE_INTERVAL="${SAVE_INTERVAL}" RUN_LOG_INTERVAL="${LOG_INTERVAL}"
INNER=()
[[ "${DRY_RUN}" == 1 ]] && INNER+=(--dry-run)
bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh "${INNER[@]}" \
  --pilot15-trajectory-manifest "${TRAJECTORY_MANIFEST}" \
  --pilot15-valid-trajectory-manifest "${VALID_TRAJECTORY_MANIFEST}" "$@"
