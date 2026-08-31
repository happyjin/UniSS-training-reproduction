#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

COMPONENTS=${COMPONENTS:-8}
DEVICE=${DEVICE:-cuda:0}
GPU=${GPU:-0}
RUN_ID=${RUN_ID:-teacher_forced_ceiling_v1}
OUTPUT=${OUTPUT:-${REPORT_ROOT}/TEACHER_FORCED_CEILING.json}
OUTPUT_AUDIO=${OUTPUT_AUDIO:-${REPO_ROOT}/eval_outputs/${EXPERIMENT_NAME}/${RUN_ID}}
LOG=${LOG:-${LOG_ROOT}/teacher_forced_ceiling.log}

for path in "${EPISODES}" "${WHISPERVQ_MODEL}/config.json" "${BASE_HF}/config.json" \
  "${CONTENT_FIRST_SFT}/.metadata" "${CONTENT_FIRST_RUNTIME_EXPORT}/manifest.json" \
  "${SOURCE_SNAPSHOT}"; do
  [[ -e "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_AUDIO}" ]] || { echo "refusing to overwrite ${OUTPUT_AUDIO}" >&2; exit 3; }
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")" "${OUTPUT_AUDIO}"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_THREADS:-2}
export UNISS_CONTENT_FIRST_RUNTIME_EXPORT=${CONTENT_FIRST_RUNTIME_EXPORT}

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" -u \
  -m experiments.uniss_phase3_content_first_diagnostics_v1.diagnostics.teacher_forced_ceiling \
  --episodes "${EPISODES}" \
  --base-hf "${BASE_HF}" \
  --adapter-checkpoint "${CONTENT_FIRST_SFT}" \
  --whispervq-model "${WHISPERVQ_MODEL}" \
  --bicodec-model "${BICODEC_MODEL}" \
  --source-snapshot "${SOURCE_SNAPSHOT}" \
  --components "${COMPONENTS}" \
  --device "${DEVICE}" \
  --output-audio "${OUTPUT_AUDIO}" \
  --output "${OUTPUT}" 2>&1 | tee "${LOG}"

echo "OUTPUT=${OUTPUT}"
echo "AUDIO=${OUTPUT_AUDIO}"
