#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

RUN_ID=${1:?usage: run_fresh_coverage_rollout_8gpu.sh RUN_ID ADAPTER_CHECKPOINT ROUND}
ADAPTER=${2:?missing adapter checkpoint}
ROUND=${3:?missing round index}
WORKERS=${ROLLOUT_WORKERS:-64}
GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-4}
EPISODES=${EPISODES:-${REPO_ROOT}/data/processed/uniss_phasea_event_constrained_grpo_long_v2/protocol64_v1/episodes.jsonl}
OUTPUT=${REPO_ROOT}/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/${RUN_ID}
BASE_HF=${BASE_HF:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}
FIXED_SFT_CHECKPOINT=${FIXED_SFT_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1_formal1e_v1/iter_0000717}
WHISPERVQ_MODEL=${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer
BICODEC_MODEL=${REPO_ROOT}/pretrained_models/UniSS/bicodec
SOURCE_SNAPSHOT=${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json
BASELINE_ROLLOUT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/ROLLOUT_MERGED.json

[[ ${WORKERS} -eq 64 && ${GROUP_SIZE} -eq 4 ]] || {
  echo "formal geometry is fixed at 64 workers and group size four" >&2; exit 2;
}
for path in "${ADAPTER}/.metadata" "${FIXED_SFT_CHECKPOINT}/.metadata" "${EPISODES}" \
  "${BASE_HF}/config.json" "${BASELINE_ROLLOUT}" "${SOURCE_SNAPSHOT}"; do
  [[ -e "${path}" ]] || { echo "missing rollout input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}/workers" "${OUTPUT}/logs"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${ROLLOUT_OMP_THREADS:-2}

pids=()
for worker in $(seq 0 63); do
  gpu=$((worker % 8))
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" -u \
    -m experiments.uniss_phasea_coverage_constrained_grpo_v3.training.rollout \
    --episodes "${EPISODES}" --baseline-rollout "${BASELINE_ROLLOUT}" \
    --output "${OUTPUT}/workers/worker_${worker}" \
    --worker-index "${worker}" --num-workers 64 --group-size 4 \
    --decision-chunk-ms 320 --acoustic-rollover-ms 24000 \
    --base-hf "${BASE_HF}" --adapter-checkpoint "${ADAPTER}" \
    --v1-checkpoint "${FIXED_SFT_CHECKPOINT}" --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" --source-snapshot "${SOURCE_SNAPSHOT}" \
    --device cuda:0 --policy-temperature 0.70 --policy-top-p 0.90 \
    --action-temperature 0.80 --round-index "${ROUND}" \
    >"${OUTPUT}/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "one or more rollout workers failed" >&2; exit 4; }
"${PYTHON}" -m experiments.uniss_phasea_coverage_constrained_grpo_v3.training.merge_rollouts \
  --worker-root "${OUTPUT}/workers" --output "${OUTPUT}/ROLLOUT_MERGED.json" \
  --expected-workers 64 --expected-episodes 64
echo "OUTPUT=${OUTPUT}"
