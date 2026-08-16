#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:-stage_a_formal8_$(date -u +%Y%m%dT%H%M%SZ)}
TRAIN_BUILD="${STAGE_A_TRAIN_PACKS}.build.json"
VALID_BUILD="${STAGE_A_VALID_PACKS}.build.json"
TRAINING_GATE="${REPORT_ROOT}/stage_a_causal_whisper_asr/STAGE_A_TRAINING_GATE_PASSED.json"
MANIFEST="${REPORT_ROOT}/stage_a_formal/${RUN_ID}/RUN_MANIFEST.json"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.formal_geometry \
  --train-build "${TRAIN_BUILD}" \
  --valid-build "${VALID_BUILD}" \
  --training-gate "${TRAINING_GATE}" \
  --pcm-glm-geometry-gate "${STAGE_A_FORMAL_PCM_GLM_GATE}" \
  --output "${MANIFEST}" \
  --run-id "${RUN_ID}" \
  --git-head "$(git rev-parse HEAD)"

readarray -t geometry < <(
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["geometry"]
for key in ("train_iters", "eval_iters", "warmup_iters"):
    print(value[key])
PY
)
[[ "${geometry[0]}" == "381" ]] || { echo "unexpected Stage A train iters" >&2; exit 4; }

export RUN_ID
export RUN_TRAIN_PACKS="${STAGE_A_TRAIN_PACKS}"
export RUN_VALID_PACKS="${STAGE_A_VALID_PACKS}"
export RUN_LOAD="${PHASE3_NATIVE_ROOT}"
export RUN_SAVE_DIR="${CHECKPOINT_ROOT}/stage_a_formal/${RUN_ID}"
export RUN_TENSORBOARD_DIR="${RUN_ROOT}/stage_a_formal/${RUN_ID}/tensorboard"
export RUN_LOG="${LOG_ROOT}/stage_a_formal/${RUN_ID}/train.log"
export RUN_SEQ_LENGTH=18000
export RUN_MBS=1
export RUN_GBS=128
export RUN_COVERAGE_EPOCHS=3
export RUN_TRAIN_ITERS="${geometry[0]}"
export RUN_MAX_ACOUSTICS=2
export RUN_NUM_WORKERS=4
export RUN_MASTER_PORT=29672
export RUN_SAVE_INTERVAL=100
export RUN_EVAL_INTERVAL=50
export RUN_EVAL_ITERS="${geometry[1]}"
export RUN_LOG_INTERVAL=1
export RUN_WARMUP_ITERS="${geometry[2]}"
export RUN_STRICTNESS=log_all
export RUN_SMOKE=0
export RUN_AUDIT_GRADIENTS=0
export RUN_FINETUNE=1
export RUN_LOAD_OPTIM=0
export RUN_LOAD_RNG=0

exec bash "${SCRIPT_DIR}/run_stage_a_megatron.sh" "$@"
