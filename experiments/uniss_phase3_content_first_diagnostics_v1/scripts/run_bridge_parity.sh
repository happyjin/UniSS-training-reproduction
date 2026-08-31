#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

COMPONENTS=${COMPONENTS:-8}
DEVICE=${DEVICE:-cuda:0}
GPU=${GPU:-0}
OUTPUT=${OUTPUT:-${REPORT_ROOT}/BRIDGE_PARITY.json}
LOG=${LOG:-${LOG_ROOT}/bridge_parity.log}

for path in "${EPISODES}" "${WHISPERVQ_MODEL}/config.json"; do
  [[ -e "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_THREADS:-2}

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" -u \
  -m experiments.uniss_phase3_content_first_diagnostics_v1.diagnostics.bridge_parity \
  --episodes "${EPISODES}" \
  --whispervq-model "${WHISPERVQ_MODEL}" \
  --components "${COMPONENTS}" \
  --device "${DEVICE}" \
  --output "${OUTPUT}" 2>&1 | tee "${LOG}"

echo "OUTPUT=${OUTPUT}"
