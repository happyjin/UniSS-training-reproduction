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

cmd=(python -m training.simul_uniss.train_bicodec_refinement
  --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl"
  --bicodec-checkpoint "${BICODEC_MODEL_DIR}/BiCodec"
  --output-dir "${STAGE5_REFINEMENT_OUTPUT_DIR}"
  --tensorboard-dir "${STAGE5_REFINEMENT_TENSORBOARD_DIR}"
  --device "${BICODEC_DEVICE}"
  --batch-size "${STAGE5_REFINEMENT_BATCH_SIZE}"
  --max-steps "${STAGE5_REFINEMENT_MAX_STEPS}"
)
if [[ "${SMOKE}" == "1" ]]; then
  cmd+=(--batch-size 1 --max-steps 2 --chunk-tokens 20 --log-interval 1 --save-interval 1)
fi
mkdir -p "${STAGE5_REFINEMENT_OUTPUT_DIR}" "${STAGE5_REFINEMENT_TENSORBOARD_DIR}" "${LOG_DIR}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage5_bicodec_refinement.log"
