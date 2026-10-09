#!/usr/bin/env bash
# Stage 1 acceptance: generate codes, decode them, score the audio.
#
# Generation and decoding shard across 8 GPUs; scoring reuses the project's
# own run_objective_metrics.sh so the numbers are comparable to every
# published figure rather than produced by a parallel implementation.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}" PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
PY_EVAL=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
DEV=/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_dev_v1
SUBSET=/opt/dlami/nvme/jasonleeeli/CVSS/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1/talker_last.pt}"
OUT="${OUT:-${REPO_ROOT}/eval_outputs/uniss_omni_s2st_v1/stage1_acceptance}"
LIMIT="${LIMIT:-200}"

mkdir -p "${OUT}/logs"

echo "[1/3] generating"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i "${PY}" -m experiments.uniss_omni_s2st_v1.evaluation.stage1_generate \
    --checkpoint "${CKPT}" \
    --dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
    --subset "${SUBSET}" --limit "${LIMIT}" \
    --shard $i --num-shards 8 --output "${OUT}" \
    > "${OUT}/logs/generate_${i}.log" 2>&1 &
done
wait

echo "[2/3] decoding"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i "${PY_EVAL}" -m experiments.uniss_omni_s2st_v1.evaluation.stage1_decode \
    --generated "${OUT}"/generated_*.jsonl \
    --shard $i --num-shards 8 --decode-gold --output "${OUT}" \
    > "${OUT}/logs/decode_${i}.log" 2>&1 &
done
wait
cat "${OUT}"/results_*.jsonl > "${OUT}/results.jsonl"
echo "  $(wc -l < "${OUT}/results.jsonl") rows in results.jsonl"

echo "[3/3] objective metrics"
EXPECTED_PAIRS="$(wc -l < "${OUT}/results.jsonl")" \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/run_objective_metrics.sh "${OUT}" cmn-\>eng \
  2>&1 | tail -40
echo "STAGE1 ACCEPTANCE DONE"
