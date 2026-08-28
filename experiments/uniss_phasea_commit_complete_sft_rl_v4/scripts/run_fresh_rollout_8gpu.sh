#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=${1:?usage: run_fresh_rollout_8gpu.sh RUN_ID ADAPTER_CHECKPOINT ROUND}
ADAPTER=${2:?missing adapter checkpoint}
ROUND=${3:?missing round index}
WORKERS=${ROLLOUT_WORKERS:-32}
GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-4}
if (( WORKERS < 8 || WORKERS > 64 )); then
  echo "ROLLOUT_WORKERS must be in [8, 64]" >&2
  exit 2
fi
if (( GROUP_SIZE < 2 || GROUP_SIZE > 4 )); then
  echo "ROLLOUT_GROUP_SIZE must be in [2, 4]" >&2
  exit 2
fi
EPISODES=${IMMUTABLE_PROTOCOL64}
OUTPUT=${REPO_ROOT}/eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/${RUN_ID}
[[ -f "${ADAPTER}/.metadata" ]] || { echo "missing adapter ${ADAPTER}" >&2; exit 2; }
[[ -f "${EPISODES}" ]] || { echo "missing selected episodes" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}/workers" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${ROLLOUT_OMP_THREADS:-4}

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  gpu=$((worker % 8))
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" -u "${EXPERIMENT_ROOT}/training/rollout.py" \
    --episodes "${EPISODES}" --baseline-rollout "${BASELINE_ROLLOUT}" \
    --output "${OUTPUT}/workers/worker_${worker}" \
    --worker-index "${worker}" --num-workers "${WORKERS}" --group-size "${GROUP_SIZE}" \
    --decision-chunk-ms "${DECISION_CHUNK_MS}" \
    --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
    --base-hf "${PHASE_A_HF}" --adapter-checkpoint "${ADAPTER}" \
    --v1-checkpoint "${PHASE_A_CHECKPOINT}" --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" --source-snapshot "${SOURCE_SNAPSHOT}" \
    --device cuda:0 --policy-temperature 0.70 --policy-top-p 0.90 \
    --action-temperature 0.80 --round-index "${ROUND}" \
    >"${OUTPUT}/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "one or more rollout workers failed" >&2; exit 4; }
"${PYTHON}" "${EXPERIMENT_ROOT}/training/merge_rollouts.py" \
  --worker-root "${OUTPUT}/workers" --output "${OUTPUT}/ROLLOUT_MERGED.json" \
  --expected-workers "${WORKERS}" --expected-episodes 64
echo "OUTPUT=${OUTPUT}"
