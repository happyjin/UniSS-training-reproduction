#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

require_file "${FORMAL_STAGE_A_ROOT}/STAGE_A_SOURCE_COMPLETE.json"
require_file "${FORMAL_JOINT_ROOT}/manifest_summary.json"
require_file "${FULL_REPLAY_OFFSETS}.json"
"${PYTHON}" - "${FULL_REPLAY_OFFSETS}.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if not value.get("complete"):
    raise SystemExit("full replay index is not complete")
PY

export RUN_NAME="${RUN_NAME:-phase3_whisper_streamspeech_joint_full198_v1}"
export TRAIN_MANIFEST="${FORMAL_JOINT_ROOT}/joint_train.jsonl"
export VALID_MANIFEST="${FORMAL_JOINT_ROOT}/joint_valid.jsonl"
export TOKENIZER_MAP_DIR="${FORMAL_JOINT_ROOT}/tokenizer_maps"
export DIRECTION_INDEX_DIR="${FORMAL_JOINT_ROOT}/direction_indices"
export REPLAY_OFFSETS="${FULL_REPLAY_OFFSETS}"
export ALLOW_PARTIAL_REPLAY_INDEX=0
export BALANCE_VALIDATION=1
export TRAIN_ITERS="${TRAIN_ITERS:-9075}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-4000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-8}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"

exec bash "${SCRIPT_ROOT}/run_megatron_8gpu.sh"
