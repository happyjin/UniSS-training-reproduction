#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "Usage: $0 RUN_ID OUTPUT_DIR ADAPTER_OR_NONE [GPU0 GPU1 GPU2 GPU3]" >&2
  echo "Use SKIP for a sample that should be deferred to a later resume." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

RUN_ID=$1
OUTPUT=$2
ADAPTER=$3
GPUS=("${4:-0}" "${5:-1}" "${6:-2}" "${7:-3}")
mkdir -p "$(dirname "${OUTPUT}")"
exec 9>"${OUTPUT}.lock"
flock 9
if [[ -f "${OUTPUT}/results.json" ]]; then
  echo "RESULTS=${OUTPUT}/results.json"
  exit 0
fi
if [[ -e "${OUTPUT}" && ( ! -d "${OUTPUT}/parts" || ! -d "${OUTPUT}/logs" ) ]]; then
  echo "refusing malformed partial output ${OUTPUT}" >&2
  exit 3
fi
if [[ "${ADAPTER}" != NONE ]]; then
  [[ -f "${ADAPTER}/.metadata" ]] || { echo "missing adapter checkpoint: ${ADAPTER}" >&2; exit 3; }
fi

mkdir -p "${OUTPUT}/parts" "${OUTPUT}/logs"
export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}

adapter_args=()
[[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")
IDS=(
  long_zh_singapore_vietnam_full
  long_zh_zhangheqiao_full
  long_en_helen_keller_full
  long_en_shimon_peres_full
)

pids=()
for index in 0 1 2 3; do
  sample_id=${IDS[$index]}
  [[ "${GPUS[$index]}" == SKIP ]] && continue
  (
    exec 8>"${OUTPUT}/parts/${sample_id}.lock"
    flock 8
    [[ -f "${OUTPUT}/parts/${sample_id}/results.json" ]] && exit 0
    CUDA_VISIBLE_DEVICES=${GPUS[$index]} "${PYTHON}" -u \
      "${EXPERIMENT_ROOT}/evaluation/stateful_longform.py" \
      --run-id "${RUN_ID}" \
      --audio-protocol "${LONG_AUDIO_PROTOCOL}" \
      --sample-id "${sample_id}" \
      --decision-chunk-ms "${DECISION_CHUNK_MS}" \
      --acoustic-rollover-ms "${ACOUSTIC_ROLLOVER_MS}" \
      --output "${OUTPUT}/parts/${sample_id}" \
      --base-hf "${PHASE_A_HF}" \
      "${adapter_args[@]}" \
      --v1-checkpoint "${PHASE_A_CHECKPOINT}" \
      --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
      --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
      --source-snapshot "${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json" \
      --strict-runtime "${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/run_strict_causal_cascade.py" \
      --device cuda:0 > "${OUTPUT}/logs/${sample_id}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} -eq 0 ]] || { echo "one or more stateful workers failed" >&2; exit 1; }

completed=$(find "${OUTPUT}/parts" -mindepth 2 -maxdepth 2 -name results.json | wc -l)
if [[ ${completed} -ne 4 ]]; then
  echo "PARTIAL=${completed}/4 OUTPUT=${OUTPUT}"
  exit 0
fi

"${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/merge_stateful_parts.py" \
  --run-id "${RUN_ID}" --parts-root "${OUTPUT}/parts" --output "${OUTPUT}/results.json"
