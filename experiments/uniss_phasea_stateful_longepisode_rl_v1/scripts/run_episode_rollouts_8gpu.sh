#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

RUN_ID=${1:?usage: run_episode_rollouts_8gpu.sh RUN_ID train|valid EPISODES_PER_WORKER}
SPLIT=${2:?missing train or valid split}
PER_WORKER=${3:?missing episodes per worker}
[[ "${SPLIT}" == train || "${SPLIT}" == valid ]] || {
  echo "split must be train or valid" >&2
  exit 2
}
[[ "${PER_WORKER}" =~ ^[1-9][0-9]*$ ]] || {
  echo "episodes per worker must be positive" >&2
  exit 2
}

EPISODES=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/${SPLIT}/episodes.jsonl
OUTPUT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${RUN_ID}
REPORT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/rollouts/${RUN_ID}/REPORT.zh-CN.md
[[ -f "${EPISODES}" ]] || { echo "missing ${EPISODES}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" && ! -e "${REPORT}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2
  exit 3
}
mkdir -p "${OUTPUT}/workers" "${OUTPUT}/logs"

export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON}"):${PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

pids=()
for worker in $(seq 0 7); do
  worker_output=${OUTPUT}/workers/worker_${worker}
  log=${OUTPUT}/logs/worker_${worker}.log
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON}" -u \
    "${EXPERIMENT_ROOT}/training/rollout.py" \
    --episodes "${EPISODES}" \
    --output "${worker_output}" \
    --worker-index "${worker}" \
    --num-workers 8 \
    --maximum-episodes "${PER_WORKER}" \
    --group-size 4 \
    --decision-chunk-ms "${DECISION_CHUNK_MS}" \
    --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
    --base-hf "${PHASE_A_HF}" \
    --v1-checkpoint "${PHASE_A_CHECKPOINT}" \
    --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
    --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
    --source-snapshot "${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json" \
    --device cuda:0 \
    --policy-temperature 0.7 \
    --policy-top-p 0.9 \
    >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "one or more rollout workers failed; inspect ${OUTPUT}/logs" >&2
  exit 4
fi

"${PYTHON}" "${EXPERIMENT_ROOT}/training/merge_rollout_workers.py" \
  --workers-root "${OUTPUT}/workers" \
  --expected-workers 8 \
  --output "${OUTPUT}/ROLLOUT_MERGED.json"
"${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/write_rollout_report.py" \
  --stage-name "${RUN_ID}" \
  --rollout "${OUTPUT}/ROLLOUT_MERGED.json" \
  --output "${REPORT}"

echo "ROLLOUT=${OUTPUT}/ROLLOUT_MERGED.json"
echo "REPORT=${REPORT}"
