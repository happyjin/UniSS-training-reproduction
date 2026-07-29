#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 LABEL RUN_ROOT GPU_ID" >&2
  exit 2
fi

LABEL="$1"
RUN_ROOT="$2"
GPU_ID="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
OUTPUT_DIR="${RUN_ROOT}/metrics_whisper_attention_mask_v2"
LOG_DIR="${REPO_ROOT}/logs/evaluation/whisper_attention_mask_v2"
COMPLETE="${OUTPUT_DIR}/COMPLETE"

cd "${REPO_ROOT}"
[[ -f "${RUN_ROOT}/results.jsonl" ]] || {
  echo "Missing results.jsonl for ${LABEL}: ${RUN_ROOT}" >&2
  exit 1
}
[[ -x "${ENV_ROOT}/bin/python" ]] || {
  echo "Missing evaluation environment: ${ENV_ROOT}" >&2
  exit 1
}
if [[ -f "${COMPLETE}" ]]; then
  echo "${LABEL}: already complete"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec > >(tee -a "${LOG_DIR}/${LABEL}.log") 2>&1
echo "[$(date -u +%FT%TZ)] start label=${LABEL} gpu=${GPU_ID} run=${RUN_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ENV_ROOT}/bin/python" -m evaluation.asr_transcribe \
  --input "${RUN_ROOT}/results.jsonl" \
  --output "${OUTPUT_DIR}/asr_results_eng.jsonl" \
  --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
  --device cuda:0 \
  --batch-size "${WHISPER_BATCH_SIZE:-8}" \
  --target-language eng \
  --resume

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_DIR}/asr_results_eng.jsonl" \
  --output "${OUTPUT_DIR}/speech_bleu_eng.json" \
  --hypothesis-field asr_text \
  --reference-field translation_ref \
  --score-empty-hypotheses

"${ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/verify.py" \
  --input "${RUN_ROOT}/results.jsonl" \
  --asr "${OUTPUT_DIR}/asr_results_eng.jsonl" \
  --output "${OUTPUT_DIR}/verification.json"

touch "${COMPLETE}"
echo "[$(date -u +%FT%TZ)] complete label=${LABEL}"
