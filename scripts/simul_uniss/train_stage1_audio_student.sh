#!/usr/bin/env bash
set -euo pipefail

SMOKE=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

cmd=(python -m training.simul_uniss.train_audio_student
  --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl"
  --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
  --output-dir "${STAGE1_AUDIO_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE1_AUDIO_TENSORBOARD_DIR}"
  --device "${STAGE1_AUDIO_DEVICE}"
  --batch-size "${STAGE1_AUDIO_BATCH_SIZE}"
  --max-steps "${STAGE1_AUDIO_MAX_STEPS}"
)
if [[ "${SMOKE}" == "1" ]]; then
  cmd=(python -m training.simul_uniss.train_audio_student
    --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl"
    --policy-tokenizer "${POLICY_TOKENIZER_MODEL}"
    --output-dir "${STAGE1_AUDIO_OUTPUT_DIR}/smoke"
    --tensorboard-dir "${STAGE1_AUDIO_TENSORBOARD_DIR}"
    --device "${STAGE1_AUDIO_DEVICE}"
    --batch-size 1
    --max-steps 2
    --hidden-size 32
    --num-layers 1
    --num-heads 4
    --max-audio-seconds 2
    --log-interval 1
    --save-interval 1
  )
fi
mkdir -p "${STAGE1_AUDIO_OUTPUT_DIR}" "${STAGE1_AUDIO_TENSORBOARD_DIR}" "${LOG_DIR}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage1_audio_student.log"
