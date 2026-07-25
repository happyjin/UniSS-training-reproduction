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
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 1; }
  [[ ! -e "${STAGE0_PREFIX_DIR}" ]] || { echo "Refusing to overwrite ${STAGE0_PREFIX_DIR}" >&2; exit 1; }
  mkdir -p "${STAGE0_PREFIX_DIR}" "${STAGE0_PREFIX_TENSORBOARD_DIR}" "${LOG_DIR}"
fi
for ((index=0; index<STAGE0_PREFIX_RECORDS; index++)); do
  cmd=(python -m training.simul_uniss.prefix_reencode_baseline
    --manifest "${manifest}"
    --glm-tokenizer "${GLM_TOKENIZER_DIR}"
    --output "${STAGE0_PREFIX_DIR}/record_${index}.json"
    --tensorboard-dir "${STAGE0_PREFIX_TENSORBOARD_DIR}"
    --device cuda:0
    --record-index "${index}"
    --chunk-ms "${CHUNK_MS}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${index}" == "0" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; fi
  else
    CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage00_prefix_baseline.log"
  fi
done

