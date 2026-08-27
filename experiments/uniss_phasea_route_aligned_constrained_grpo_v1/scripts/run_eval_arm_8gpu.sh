#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then echo "usage: $0 RUN_ID OUTPUT ADAPTER_OR_NONE" >&2; exit 2; fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=$1; OUTPUT=$2; ADAPTER=$3
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}/parts" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
mapfile -t IDS < <("${PYTHON}" -c 'import json,sys; print("\n".join(x["sample_id"] for x in json.load(open(sys.argv[1]))["records"]))' "${DEMO_PROTOCOL}")
adapter_args=(); [[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")
pids=()
for index in $(seq 0 7); do
  sample=${IDS[$index]}
  CUDA_VISIBLE_DEVICES=${index} "${PYTHON}" -u "${EXPERIMENT_ROOT}/evaluation/stateful_longform.py" \
    --run-id "${RUN_ID}" --audio-protocol "${DEMO_PROTOCOL}" --sample-id "${sample}" \
    --decision-chunk-ms "${DECISION_CHUNK_MS}" --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
    --output "${OUTPUT}/parts/${sample}" --base-hf "${PHASE_A_HF}" "${adapter_args[@]}" \
    --v1-checkpoint "${PHASE_A_CHECKPOINT}" --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" --source-snapshot "${SOURCE_SNAPSHOT}" \
    --strict-runtime "${STRICT_RUNTIME}" --device cuda:0 >"${OUTPUT}/logs/${sample}.log" 2>&1 &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "evaluation worker failure" >&2; exit 4; }
"${PYTHON}" "${REPO_ROOT}/experiments/uniss_phasea_rl_trainlong_eval_v1/evaluation/merge_parts.py" \
  --run-id "${RUN_ID}" --protocol "${DEMO_PROTOCOL}" --parts-root "${OUTPUT}/parts" \
  --output "${OUTPUT}/results.json"
"${PYTHON}" "${REPO_ROOT}/experiments/uniss_phasea_rl_trainlong_eval_v1/evaluation/score_results.py" \
  --run-id "${RUN_ID}" --protocol "${DEMO_PROTOCOL}" --results "${OUTPUT}/results.json" \
  --output "${OUTPUT}/SCORED.json"

