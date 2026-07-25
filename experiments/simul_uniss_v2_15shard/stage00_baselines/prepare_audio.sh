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

cmd=(python -m training.simul_uniss.reconstruct_unist_audio
  --input "${UNIST_ROOT}/train-00000.parquet"
  --output-dir "${STAGE0_AUDIO_DIR}"
  --bicodec-model-dir "${BICODEC_MODEL_DIR}"
  --device cuda:0
  --limit-records "${STAGE0_RECONSTRUCT_RECORDS}"
  --side both)
if [[ "${DRY_RUN}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ ! -e "${STAGE0_AUDIO_DIR}" ]] || { echo "Refusing to overwrite ${STAGE0_AUDIO_DIR}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage00_audio_reconstruction.log"

