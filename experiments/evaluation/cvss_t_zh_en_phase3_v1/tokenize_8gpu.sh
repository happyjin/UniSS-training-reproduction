#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${CVSS_ROOT}/canonical_16k/cvss_t_zh_en_test/manifests/cvss_t_zh_en_test_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CVSS_ROOT}/tokenized/cvss_t_zh_en_v1}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "${NUM_SHARDS}" -ne 8 ]]; then
  echo "CVSS-T formal tokenization expects 8 GPUs; got EVAL_GPU_LIST=${GPU_LIST_VALUE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
pids=()
for ((index = 0; index < NUM_SHARDS; index++)); do
  gpu="${GPU_IDS[${index}]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_ROOT}/bin/python" -m evaluation.cvss_t.tokenize \
    --pair-manifest "${PAIR_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --device cuda:0 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${index}" \
    >"${OUTPUT_DIR}/logs/tokenize_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  tail -n 80 "${OUTPUT_DIR}/logs/tokenize_"*.log >&2 || true
  exit "${status}"
fi

"${ENV_ROOT}/bin/python" -m evaluation.cvss_t.merge_tokenized \
  --part-root "${OUTPUT_DIR}/parts" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  --expected-pairs 4897 \
  --smoke-count 10 \
  --listen-count 50
