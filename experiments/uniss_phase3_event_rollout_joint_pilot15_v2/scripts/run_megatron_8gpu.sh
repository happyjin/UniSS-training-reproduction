#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.env"
cd "${REPO_ROOT}"

for value in "${TRAJECTORY_MANIFEST}" "${VALID_TRAJECTORY_MANIFEST}" \
  "${DATA_AUDIT}" "${TRAINING_MANIFEST}" "${REPLAY_SUBSET_OFFSETS}" \
  "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${value}" ]] || { echo "Missing pilot15 v2 artifact: ${value}" >&2; exit 1; }
done
[[ "$(tr -d '[:space:]' < "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt")" == "9075" ]] || {
  echo "Phase3 root is not pinned to iter_0009075" >&2; exit 1;
}

read -r TRAIN_ITERS WARMUP_ITERS MANIFEST_EPOCHS TRAJECTORY_COUNT REPLAY_COUNT < <(
  "${PYTHON}" - "${TRAINING_MANIFEST}" "${DATA_AUDIT}" <<'PY'
import json, os, sys
manifest=json.load(open(sys.argv[1], encoding='utf-8'))
audit=json.load(open(sys.argv[2], encoding='utf-8'))
if manifest.get('schema_version') != 'uniss_event_rollout_pilot15_training_manifest_v1':
    raise SystemExit('wrong pilot15 training manifest schema')
if not manifest.get('strict_global_shuffle') or not manifest.get('multi_file_prefix_sum_namespace'):
    raise SystemExit('strict multi-file global shuffle is not frozen')
if audit.get('status') != 'pass' and not bool(int(os.environ.get('ALLOW_SAMPLED_AUDIT','0'))):
    raise SystemExit('formal training requires full data audit pass')
print(manifest['train_iters'], manifest['warmup_iters'], manifest['coverage_epochs'],
      manifest['trajectory_count'], manifest['replay_selected'])
PY
)
[[ "${MANIFEST_EPOCHS}" == "${COVERAGE_EPOCHS}" ]] || {
  echo "Coverage epochs differ from frozen manifest" >&2; exit 1;
}

TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
if [[ -s "${TRACKER}" ]]; then
  current="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( current == TRAIN_ITERS )); then echo "pilot15 v2 complete at ${TRAIN_ITERS}"; exit 0; fi
  (( current < TRAIN_ITERS )) || { echo "checkpoint exceeds target" >&2; exit 1; }
  LOAD="${SAVE_DIR}"; FINETUNE=0; LOAD_OPTIM=1; LOAD_RNG=1; STRICTNESS=raise_all
else
  if [[ -e "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing fresh run in non-empty checkpoint directory: ${SAVE_DIR}" >&2; exit 1
  fi
  LOAD="${PHASE3_NATIVE_CHECKPOINT}"; FINETUNE=1; LOAD_OPTIM=0; LOAD_RNG=0; STRICTNESS=log_all
fi

FIRST_TRAIN_PACK="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parts"][0]["packed"])' "${TRAJECTORY_MANIFEST}")"
FIRST_VALID_PACK="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parts"][0]["packed"])' "${VALID_TRAJECTORY_MANIFEST}")"
mkdir -p "${RUN_DIR}" "${REPORT_ROOT}"
cat > "${RUN_DIR}/manifest.txt" <<EOF
experiment=${EXPERIMENT_NAME}
updated_at=$(date -u +%FT%TZ)
repo_commit=$(git rev-parse HEAD)
base_checkpoint=${PHASE3_NATIVE_CHECKPOINT}/iter_0009075
repair=trainable_causal_frontend
scope=fixed_unist_shards_00000_00014
trajectory_manifest=${TRAJECTORY_MANIFEST}
trajectory_count=${TRAJECTORY_COUNT}
replay_count=${REPLAY_COUNT}
replay_fraction=${REPLAY_FRACTION}
coverage_epochs=${COVERAGE_EPOCHS}
train_iters=${TRAIN_ITERS}
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
sequence_length=${SEQ_LENGTH}
shuffle=full_randperm_over_global_multifile_complete_pack_ids
session_internal_event_order=preserved
data_audit=${DATA_AUDIT}
tensorboard_dir=${TB_DIR}
tensorboard_port=${TENSORBOARD_PORT}
EOF

export CUDA_VISIBLE_DEVICES COVERAGE_EPOCHS REPLAY_FRACTION
export LR_QWEN_LORA LR_FRONTEND LR_NEW_HEADS MIN_LR WEIGHT_DECAY NUM_WORKERS NPZ_LRU_CAPACITY SEED
INNER=()
[[ "${DRY_RUN}" == 1 ]] && INNER+=(--dry-run)
RUN_NAME="${EXPERIMENT_NAME}" RUN_SAVE_DIR="${SAVE_DIR}" RUN_TB_DIR="${TB_DIR}" \
RUN_LOG="${LOG_PATH}" RUN_TRAINING_MANIFEST="${TRAINING_MANIFEST}" \
RUN_TRAJECTORY_PACKED="${FIRST_TRAIN_PACK}" RUN_TRAJECTORY_OFFSETS="${FIRST_TRAIN_PACK}.offsets.bin" \
RUN_REPLAY_PACKED="${PHASE3_REPLAY_PACKED}" RUN_REPLAY_OFFSETS="${REPLAY_SUBSET_OFFSETS}" \
RUN_VALID_TRAJECTORY_PACKED="${FIRST_VALID_PACK}" RUN_VALID_TRAJECTORY_OFFSETS="${FIRST_VALID_PACK}.offsets.bin" \
RUN_VALID_REPLAY_PACKED="" RUN_VALID_REPLAY_OFFSETS="" \
RUN_ENTRYPOINT="${HERE}/training/pretrain_event_rollout_megatron.py" \
RUN_LORA_DROPOUT=0 RUN_ATTENTION_DROPOUT=0 RUN_HIDDEN_DROPOUT=0 \
RUN_FULL_VALIDATION=1 RUN_EVAL_INTERVAL="${EVAL_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}" \
RUN_TRAIN_ITERS="${TRAIN_ITERS}" RUN_WARMUP_ITERS="${WARMUP_ITERS}" \
RUN_NPROC=8 RUN_MBS="${MICRO_BATCH_SIZE}" RUN_GBS="${GLOBAL_BATCH_SIZE}" \
RUN_SEQ_LENGTH="${SEQ_LENGTH}" RUN_MASTER_PORT="${MASTER_PORT}" RUN_LOAD="${LOAD}" \
RUN_FINETUNE="${FINETUNE}" RUN_LOAD_OPTIM="${LOAD_OPTIM}" RUN_LOAD_RNG="${LOAD_RNG}" \
RUN_STRICTNESS="${STRICTNESS}" RUN_SMOKE="${RUN_SMOKE:-0}" RUN_AUDIT_GRADIENTS=1 \
RUN_SAVE_INTERVAL="${SAVE_INTERVAL}" RUN_LOG_INTERVAL="${LOG_INTERVAL}" \
RUN_EXIT_INTERVAL="${RUN_EXIT_INTERVAL:-}" \
bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_megatron_training.sh \
  "${INNER[@]}" --pilot15-trajectory-manifest "${TRAJECTORY_MANIFEST}" \
  --pilot15-valid-trajectory-manifest "${VALID_TRAJECTORY_MANIFEST}" "$@"
