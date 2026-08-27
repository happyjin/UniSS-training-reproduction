#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=${1:?usage: run_rollouts_8gpu.sh RUN_ID ADAPTER_CHECKPOINT}
ADAPTER=${2:?missing adapter checkpoint}
OUTPUT=${REPO_ROOT}/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}
[[ -f "${ADAPTER}/.metadata" ]] || { echo "missing ${ADAPTER}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}/workers" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON}" -u "${EXPERIMENT_ROOT}/training/rollout.py" \
    --episodes "${TRAIN_EPISODES}" --protocol "${DEMO_PROTOCOL}" \
    --baseline-rollout "${BASELINE_ROLLOUT}" \
    --output "${OUTPUT}/workers/worker_${worker}" \
    --worker-index "${worker}" --num-workers 8 --group-size 8 \
    --decision-chunk-ms "${DECISION_CHUNK_MS}" \
    --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
    --base-hf "${PHASE_A_HF}" --adapter-checkpoint "${ADAPTER}" \
    --v1-checkpoint "${PHASE_A_CHECKPOINT}" --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" --source-snapshot "${SOURCE_SNAPSHOT}" \
    --device cuda:0 --asr-temperature 0.30 --policy-temperature 0.70 \
    --policy-top-p 0.90 --retention 0.98 \
    >"${OUTPUT}/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "one or more rollout workers failed" >&2; exit 4; }
"${PYTHON}" "${REPO_ROOT}/experiments/uniss_phasea_stateful_longepisode_rl_v1/training/merge_rollout_workers.py" \
  --workers-root "${OUTPUT}/workers" --expected-workers 8 \
  --output "${OUTPUT}/ROLLOUT_MERGED.json"
echo "OUTPUT=${OUTPUT}"

