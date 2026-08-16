#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

if [[ -e "${STAGE_A_TRAIN_PACKS}" || -e "${STAGE_A_VALID_PACKS}" ]]; then
  echo "refusing to overwrite formal Stage A packs" >&2
  exit 2
fi
mkdir -p "${STAGE_A_PACK_ROOT}" "${TMPDIR}" "${LOG_ROOT}/stage_a_pack_full"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.build_training_packs \
  --manifest "${STAGE_A_SOURCE_TRAIN}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --output "${STAGE_A_TRAIN_PACKS}" \
  --workers 26 \
  --seq-length 18000 \
  > "${LOG_ROOT}/stage_a_pack_full/train.log" 2>&1 &
train_pid=$!

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.build_training_packs \
  --manifest "${STAGE_A_SOURCE_VALID}" \
  --model "${PHASE3_HF_CHECKPOINT}" \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --output "${STAGE_A_VALID_PACKS}" \
  --workers 4 \
  --seq-length 18000 \
  > "${LOG_ROOT}/stage_a_pack_full/valid.log" 2>&1 &
valid_pid=$!

status=0
wait "${train_pid}" || status=$?
wait "${valid_pid}" || status=$?
if (( status != 0 )); then
  echo "Stage A formal pack build failed; inspect ${LOG_ROOT}/stage_a_pack_full" >&2
  exit "${status}"
fi
echo "train_packs=${STAGE_A_TRAIN_PACKS}"
echo "valid_packs=${STAGE_A_VALID_PACKS}"
