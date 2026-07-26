#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 STAGE HF_CHECKPOINT MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

STAGE="$1"
HF_CHECKPOINT="$2"
MANIFEST="$3"
OUTPUT_ROOT="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# vLLM 0.8.5 defaults to the V1 engine, which rejects per-request logits
# processors. UniSS must mask Megatron's 73 padded vocabulary rows, so use the
# compatible V0 engine until V1 supports this sampling hook.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

if [[ -e "${OUTPUT_ROOT}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/metrics"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${ENV_ROOT}/bin/python" -m evaluation.vllm_generate \
  --manifest "${MANIFEST}" \
  --model "${HF_CHECKPOINT}" \
  --output-dir "${OUTPUT_ROOT}/vllm" \
  --mode quality performance \
  --limit-records 0 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k -1 \
  --repetition-penalty 1.1 \
  --max-new-tokens "${MAX_NEW_TOKENS:-1500}" \
  --seed "${SEED:-20260726}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
  --request-batch-size "${REQUEST_BATCH_SIZE:-256}" \
  --dtype bfloat16 \
  "${RESUME_ARGS[@]}"

"${ENV_ROOT}/bin/python" -m evaluation.decode_audio \
  --input "${OUTPUT_ROOT}/vllm/generation_results.jsonl" \
  --manifest "${MANIFEST}" \
  --speech-tokenizer "${SPEECH_TOKENIZER}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --save-source-audio \
  --save-reference-audio \
  --resume

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_audio_output.py" \
  --manifest "${MANIFEST}" \
  --results "${OUTPUT_ROOT}/results.jsonl" \
  --summary "${OUTPUT_ROOT}/summary.json" \
  --expected-modes quality performance

"${ENV_ROOT}/bin/python" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/vllm/generation_results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/text_bleu.json"

"${ENV_ROOT}/bin/python" -m evaluation.slc_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics"

echo "${STAGE} vLLM output: ${OUTPUT_ROOT}"
