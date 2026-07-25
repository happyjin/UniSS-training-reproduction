#!/usr/bin/env bash
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
manifest="${STAGE0_AUDIO_DIR}/audio_manifest.jsonl"
cmd=(torchrun --nproc_per_node "${SIMUL_NPROC_PER_NODE}"
  --master_addr "${SIMUL_MASTER_ADDR}" --master_port "${STAGE5_REFINEMENT_MASTER_PORT}"
  -m training.simul_uniss.train_bicodec_refinement --manifest "${manifest}"
  --bicodec-checkpoint "${BICODEC_MODEL_DIR}/BiCodec"
  --output-dir "${STAGE5_REFINEMENT_OUTPUT_DIR}" --tensorboard-dir "${STAGE5_REFINEMENT_TENSORBOARD_DIR}"
  --device cuda --batch-size "${STAGE5_REFINEMENT_BATCH_SIZE}"
  --max-steps "${STAGE5_REFINEMENT_MAX_STEPS}" --validation-records "${STAGE1_VALIDATION_RECORDS}"
  --eval-interval "${STAGE1_EVAL_INTERVAL}" --save-interval "${STAGE1_SAVE_INTERVAL}"
  --seed "${SIMUL_DATA_SEED}")
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 1; }
[[ ! -e "${STAGE5_REFINEMENT_OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${STAGE5_REFINEMENT_OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}" "${STAGE5_REFINEMENT_TENSORBOARD_DIR}"
export CUDA_VISIBLE_DEVICES="${SIMUL_CUDA_VISIBLE_DEVICES}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage05_bicodec_refinement.log"
