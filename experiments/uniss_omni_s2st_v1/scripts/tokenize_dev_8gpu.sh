#!/usr/bin/env bash
# BiCodec-tokenise the CVSS-T *dev* split, for training-time loss curves.
#
# A sibling of experiments/evaluation/cvss_t_zh_en_phase3_v1/tokenize_8gpu.sh
# rather than a parameterisation of it: that script is referenced by the
# completed formal evaluation and is not being touched. The only real
# difference is the pair count the merge step asserts -- dev has 4,843 where
# test has 4,897 -- plus the split name, which keeps the two apart in both
# the filenames and the records' own split field.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${CVSS_ROOT}/canonical_16k/cvss_t_zh_en_dev/manifests/cvss_t_zh_en_dev_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CVSS_ROOT}/tokenized/cvss_t_zh_en_dev_v1}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
EXPECTED_PAIRS="${EXPECTED_PAIRS:-4843}"
SPLIT_NAME="${SPLIT_NAME:-dev}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST_VALUE}"
NUM_SHARDS="${#GPU_IDS[@]}"

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
    --split-name "${SPLIT_NAME}" \
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
  --expected-pairs "${EXPECTED_PAIRS}" \
  --split-name "${SPLIT_NAME}" \
  --smoke-count 10 \
  --listen-count 50
echo "DEV TOKENIZE DONE"
