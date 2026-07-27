#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_DIR EXPECTED_RECORDS" >&2
  exit 2
fi
RUN_DIR="$1"
EXPECTED="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

IFS=',' read -r -a GPU_IDS <<<"${DEV_GPUS}"
mkdir -p "${RUN_DIR}/audio" "${RUN_DIR}/logs"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${rank}]}" "${TRAIN_ENV}/bin/python" \
    -m evaluation.simultaneous_streaming.stage4_streaming_decode \
    --input "${RUN_DIR}/generation_results.jsonl" \
    --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${RUN_DIR}/audio" \
    --rank "${rank}" --world-size 4 --device cuda:0 \
    --batch-size "${DECODE_BATCH_SIZE}" \
    --left-context-tokens "${LEFT_CONTEXT_TOKENS}" \
    --holdback-tokens "${HOLDBACK_TOKENS}" \
    --overlap-ms "${OVERLAP_MS}" \
    --resume \
    >"${RUN_DIR}/logs/decode_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
if [[ "${status}" -ne 0 ]]; then
  tail -n 100 "${RUN_DIR}/logs/decode_rank"*.log >&2 || true
  exit "${status}"
fi
if [[ ! -f "${RUN_DIR}/results.jsonl" ]]; then
  "${EVAL_ENV}/bin/python" -m evaluation.simultaneous_streaming.stage4_aggregate merge \
    --input-dir "${RUN_DIR}/audio" \
    --pattern 'results.rank*.jsonl' \
    --output "${RUN_DIR}/results.jsonl" \
    --expected-records "${EXPECTED}" \
    --expected-ranks 4
fi
