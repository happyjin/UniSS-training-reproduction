#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

RUN_ID=${1:-reference_attribution_valid16_v1}
OUTPUT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${RUN_ID}
EPISODES=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/valid/episodes.jsonl
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}/workers" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${worker} "${PYTHON}" -u \
    "${EXPERIMENT_ROOT}/evaluation/reference_attribution.py" \
    --episodes "${EPISODES}" \
    --output "${OUTPUT}/workers/worker_${worker}" \
    --worker-index "${worker}" \
    --num-workers 8 \
    --maximum-episodes 2 \
    --base-hf "${PHASE_A_HF}" \
    --v1-checkpoint "${PHASE_A_CHECKPOINT}" \
    --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
    --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
    --strict-runtime "${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/run_strict_causal_cascade.py" \
    --device cuda:0 \
    >"${OUTPUT}/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "reference attribution worker failed" >&2; exit 4; }
echo "WORKERS=${OUTPUT}/workers"
