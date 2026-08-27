#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 RUN_ID OUTPUT_DIR ADAPTER_OR_NONE" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

RUN_ID=$1
OUTPUT=$2
ADAPTER=$3
PROTOCOL=${EXPERIMENT_ROOT}/evaluation/protocol_train_seen_long8.json
mkdir -p "$(dirname "${OUTPUT}")"
exec 9>"${OUTPUT}.lock"
flock 9
if [[ -f "${OUTPUT}/SCORED.json" ]]; then
  echo "SCORED=${OUTPUT}/SCORED.json"
  exit 0
fi
[[ -f "${PROTOCOL}" ]] || { echo "missing protocol ${PROTOCOL}" >&2; exit 3; }
[[ -f "${PHASE_A_CHECKPOINT}/.metadata" ]] || { echo "missing Phase A checkpoint" >&2; exit 3; }
if [[ "${ADAPTER}" != NONE ]]; then
  [[ -f "${ADAPTER}/.metadata" ]] || { echo "missing adapter ${ADAPTER}" >&2; exit 3; }
fi
gpu_count=$(nvidia-smi -L 2>/dev/null | rg -c '^GPU [0-9]+:' || true)
gpu_count=${gpu_count:-0}
[[ ${gpu_count} -ge 8 ]] || { echo "eight visible GPUs are required; found ${gpu_count}" >&2; exit 4; }

mkdir -p "${OUTPUT}/parts" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}

mapfile -t IDS < <("${PYTHON}" -c \
  'import json,sys; print("\n".join(str(x["sample_id"]) for x in json.load(open(sys.argv[1]))["records"]))' \
  "${PROTOCOL}")
[[ ${#IDS[@]} -eq 8 ]] || { echo "protocol must contain exactly eight records" >&2; exit 5; }
adapter_args=()
[[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")

pids=()
for index in 0 1 2 3 4 5 6 7; do
  sample_id=${IDS[$index]}
  (
    exec 8>"${OUTPUT}/parts/${sample_id}.lock"
    flock 8
    [[ -f "${OUTPUT}/parts/${sample_id}/results.json" ]] && exit 0
    CUDA_VISIBLE_DEVICES=${index} "${PYTHON}" -u \
      "${REPO_ROOT}/experiments/uniss_phasea_stateful_longepisode_rl_v1/evaluation/stateful_longform.py" \
      --run-id "${RUN_ID}" --audio-protocol "${PROTOCOL}" \
      --sample-id "${sample_id}" --decision-chunk-ms "${DECISION_CHUNK_MS}" \
      --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
      --output "${OUTPUT}/parts/${sample_id}" --base-hf "${PHASE_A_HF}" \
      "${adapter_args[@]}" --v1-checkpoint "${PHASE_A_CHECKPOINT}" \
      --whispervq-model "${WHISPERVQ_MODEL}" --bicodec-model "${BICODEC_MODEL}" \
      --source-snapshot "${SOURCE_SNAPSHOT}" --strict-runtime "${STRICT_RUNTIME}" \
      --device cuda:0 >"${OUTPUT}/logs/${sample_id}.log" 2>&1
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ ${status} -eq 0 ]] || { echo "one or more Runtime-v2 workers failed" >&2; exit 6; }

if [[ ! -f "${OUTPUT}/results.json" ]]; then
  "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/merge_parts.py" \
    --run-id "${RUN_ID}" --protocol "${PROTOCOL}" \
    --parts-root "${OUTPUT}/parts" --output "${OUTPUT}/results.json"
fi
"${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/score_results.py" \
  --run-id "${RUN_ID}" --protocol "${PROTOCOL}" \
  --results "${OUTPUT}/results.json" --output "${OUTPUT}/SCORED.json"
echo "SCORED=${OUTPUT}/SCORED.json"
