#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 STAGE HF_CHECKPOINT MANIFEST OUTPUT_DIR" >&2
  exit 2
fi

STAGE="$1"
HF_CHECKPOINT="$2"
MANIFEST="$3"
OUTPUT_DIR="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite evaluation output: ${OUTPUT_DIR}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
"${ENV_ROOT}/bin/python" "${REPO_ROOT}/training/generate_unist_eval_audio.py" \
  --manifest "${MANIFEST}" \
  --model "${HF_CHECKPOINT}" \
  --speech-tokenizer "${SPEECH_TOKENIZER}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode quality performance \
  --limit-records 0 \
  --max-new-tokens "${MAX_NEW_TOKENS:-1500}" \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k -1 \
  --repetition-penalty 1.1 \
  --seed "${SEED:-20260726}" \
  --dtype bfloat16 \
  --device cuda:0 \
  --local-files-only \
  --save-source-audio \
  --save-reference-audio

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_audio_output.py" \
  --manifest "${MANIFEST}" \
  --results "${OUTPUT_DIR}/results.jsonl" \
  --summary "${OUTPUT_DIR}/summary.json" \
  --expected-modes quality performance

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_DIR}/results.jsonl" \
  --output "${OUTPUT_DIR}/metrics/text_bleu.json"

"${ENV_ROOT}/bin/python" -m evaluation.slc_metrics \
  --input "${OUTPUT_DIR}/results.jsonl" \
  --output-dir "${OUTPUT_DIR}/metrics"

echo "${STAGE} output: ${OUTPUT_DIR}"
