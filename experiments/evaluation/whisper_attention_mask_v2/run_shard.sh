#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 LABEL RUN_ROOT GPU_ID SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

LABEL="$1"
RUN_ROOT="$2"
GPU_ID="$3"
SHARD_INDEX="$4"
NUM_SHARDS="$5"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-metrics_whisper_attention_mask_v2}"
OUTPUT_DIR="${RUN_ROOT}/${OUTPUT_SUBDIR}"
SHARD_DIR="${OUTPUT_DIR}/shards"
LOG_DIR="${REPO_ROOT}/logs/evaluation/whisper_attention_mask_v2"
printf -v SHARD_TAG '%03d' "${SHARD_INDEX}"
PART="${SHARD_DIR}/asr_results_eng.part_${SHARD_TAG}.jsonl"
MARKER="${SHARD_DIR}/shard_${SHARD_TAG}.COMPLETE"

cd "${REPO_ROOT}"
[[ -f "${RUN_ROOT}/results.jsonl" ]] || {
  echo "Missing results.jsonl for ${LABEL}: ${RUN_ROOT}" >&2
  exit 1
}
[[ -x "${ENV_ROOT}/bin/python" ]] || {
  echo "Missing evaluation environment: ${ENV_ROOT}" >&2
  exit 1
}
if [[ -f "${OUTPUT_DIR}/COMPLETE" ]]; then
  echo "${LABEL}: already complete"
  exit 0
fi

mkdir -p "${SHARD_DIR}" "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "${LOG_DIR}/${LABEL}.shard_${SHARD_TAG}.log") 2>&1
echo "[$(date -u +%FT%TZ)] start label=${LABEL} shard=${SHARD_INDEX}/${NUM_SHARDS} gpu=${GPU_ID}"

if [[ ! -f "${MARKER}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ENV_ROOT}/bin/python" -m evaluation.asr_transcribe \
    --input "${RUN_ROOT}/results.jsonl" \
    --output "${PART}" \
    --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
    --device cuda:0 \
    --batch-size "${WHISPER_BATCH_SIZE:-8}" \
    --target-language eng \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --completed-input "${OUTPUT_DIR}/asr_results_eng.jsonl" \
    --resume
  touch "${MARKER}"
fi

"${ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/finalize_shards.py" \
  --run-root "${RUN_ROOT}" \
  --num-shards "${NUM_SHARDS}" \
  --output-subdir "${OUTPUT_SUBDIR}"
echo "[$(date -u +%FT%TZ)] shard complete label=${LABEL} shard=${SHARD_INDEX}/${NUM_SHARDS}"
