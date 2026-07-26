#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
MODEL_ROOT="${MODEL_ROOT:-/opt/dlami/nvme/jasonleeeli/evaluation_models}"
AUTOPCP_COMPARATOR="${AUTOPCP_COMPARATOR:-${MODEL_ROOT}/AutoPCP-multilingual-v2}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUTPUT_ROOT}/metrics"

"${ENV_ROOT}/bin/python" -m evaluation.asr_transcribe \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/asr_results.jsonl" \
  --device "${DEVICE}" \
  --batch-size "${ASR_BATCH_SIZE:-8}" \
  --resume

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/metrics/asr_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/speech_bleu.json" \
  --hypothesis-field asr_text \
  --reference-field translation_ref

TORCH_HOME="${TORCH_HOME:-/opt/dlami/nvme/jasonleeeli/cache/torch}" \
"${ENV_ROOT}/bin/python" -m evaluation.utmos_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics" \
  --device "${DEVICE}" \
  --resume

"${ENV_ROOT}/bin/python" -m evaluation.autopcp_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics" \
  --comparator-path "${AUTOPCP_COMPARATOR}" \
  --device "${DEVICE}" \
  --pick-layer 9 \
  --symmetrize \
  --batch-size "${AUTOPCP_BATCH_SIZE:-16}" \
  --chunk-size "${AUTOPCP_CHUNK_SIZE:-1024}" \
  --num-process "${AUTOPCP_NUM_PROCESS:-4}" \
  --resume
