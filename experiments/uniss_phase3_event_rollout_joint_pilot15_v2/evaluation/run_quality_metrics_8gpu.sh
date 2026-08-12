#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CHECKPOINT_EVALUATION_ROOT" >&2
  exit 2
fi

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ROOT="$(realpath "$1")"
RESULTS="${RESULTS:-${ROOT}/valid_aggregate/results.jsonl}"
METRICS="${METRICS:-${ROOT}/valid_aggregate/metrics}"
EVAL_ENV="${EVAL_ENV:-${USER_ROOT}/conda_envs/uniss-eval}"
PYTHON="${PYTHON:-${EVAL_ENV}/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
AUTOPCP_COMPARATOR="${AUTOPCP_COMPARATOR:-${USER_ROOT}/evaluation_models/AutoPCP-multilingual-v2}"
AUTOPCP_ENCODER="${AUTOPCP_ENCODER:-${USER_ROOT}/evaluation_models/wav2vec2-large-xlsr-53}"
WAVLM_SPEAKER_MODEL="${WAVLM_SPEAKER_MODEL:-${USER_ROOT}/evaluation_models/wavlm-base-plus-sv}"

for required in "${RESULTS}" "${PYTHON}" "${AUTOPCP_COMPARATOR}" "${AUTOPCP_ENCODER}"; do
  [[ -e "${required}" ]] || { echo "Missing quality-metric input: ${required}" >&2; exit 1; }
done
for required in config.json preprocessor_config.json pytorch_model.bin; do
  [[ -f "${AUTOPCP_ENCODER}/${required}" ]] || {
    echo "Incomplete local AutoPCP encoder: ${AUTOPCP_ENCODER}/${required}" >&2
    exit 1
  }
done
[[ -f "${WAVLM_SPEAKER_MODEL}/config.json" ]] || {
  echo "Missing fixed-speaker WavLM model: ${WAVLM_SPEAKER_MODEL}" >&2
  exit 1
}
[[ ! -e "${METRICS}/complete.json" ]] || {
  echo "Quality metrics already complete at ${METRICS}" >&2
  exit 1
}

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
  echo "GPU_LIST must contain exactly 8 GPU IDs" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu}" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || { echo "Duplicate GPU ID: ${gpu}" >&2; exit 2; }
  SEEN_GPUS["${gpu}"]=1
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${USER_ROOT}/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${USER_ROOT}/cache/torch}"
export LD_LIBRARY_PATH="${EVAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${METRICS}/logs" "${METRICS}/shards"

wait_workers() {
  local phase="$1"
  shift
  local status=0
  local pid
  for pid in "$@"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${phase} failed; inspect ${METRICS}/logs/${phase}_*.log" >&2
    return "${status}"
  fi
}

run_asr() {
  local input="$1"
  local root="$2"
  local phase="$3"
  local pids=()
  local index gpu output
  mkdir -p "${root}/shards/asr" "${root}/logs"
  for ((index = 0; index < 8; index++)); do
    gpu="${GPU_IDS[${index}]}"
    output="${root}/shards/asr/part_$(printf '%03d' "${index}").jsonl"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m evaluation.asr_transcribe \
      --input "${input}" --output "${output}" --device cuda:0 \
      --batch-size "${ASR_BATCH_SIZE:-8}" --num-shards 8 --shard-index "${index}" \
      --resume >"${root}/logs/${phase}_${index}.log" 2>&1 &
    pids+=("$!")
  done
  wait_workers "${phase}" "${pids[@]}"
  "${PYTHON}" -m evaluation.merge_metric_shards \
    --metric asr --input "${input}" --metric-dir "${root}" \
    --shard-root "${root}/shards" --num-shards 8
}

run_asr "${RESULTS}" "${METRICS}" asr
"${PYTHON}" -m evaluation.text_metrics \
  --input "${RESULTS}" --output "${METRICS}/text_bleu.json" \
  --hypothesis-field generated_translation --reference-field translation_ref \
  --score-empty-hypotheses
"${PYTHON}" -m evaluation.text_metrics \
  --input "${METRICS}/asr_results.jsonl" --output "${METRICS}/speech_bleu.json" \
  --hypothesis-field asr_text --reference-field translation_ref \
  --score-empty-hypotheses
"${PYTHON}" -m evaluation.slc_metrics \
  --input "${RESULTS}" --output-dir "${METRICS}"

utmos_pids=()
mkdir -p "${METRICS}/shards/utmos"
for ((index = 0; index < 8; index++)); do
  gpu="${GPU_IDS[${index}]}"
  part="${METRICS}/shards/utmos/part_$(printf '%03d' "${index}")"
  mkdir -p "${part}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m evaluation.utmos_metrics \
    --input "${RESULTS}" --output-dir "${part}" --device cuda:0 \
    --num-shards 8 --shard-index "${index}" --resume \
    >"${METRICS}/logs/utmos_${index}.log" 2>&1 &
  utmos_pids+=("$!")
done
wait_workers utmos "${utmos_pids[@]}"
"${PYTHON}" -m evaluation.merge_metric_shards \
  --metric utmos --input "${RESULTS}" --metric-dir "${METRICS}" \
  --shard-root "${METRICS}/shards" --num-shards 8

autopcp_pids=()
mkdir -p "${METRICS}/shards/autopcp"
for ((index = 0; index < 8; index++)); do
  gpu="${GPU_IDS[${index}]}"
  part="${METRICS}/shards/autopcp/part_$(printf '%03d' "${index}")"
  mkdir -p "${part}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m evaluation.autopcp_metrics \
    --input "${RESULTS}" --output-dir "${part}" \
    --comparator-path "${AUTOPCP_COMPARATOR}" --device cuda:0 \
    --encoder-model "${AUTOPCP_ENCODER}" \
    --pick-layer 9 --symmetrize --batch-size "${AUTOPCP_BATCH_SIZE:-16}" \
    --chunk-size "${AUTOPCP_CHUNK_SIZE:-1024}" --num-process 1 \
    --num-shards 8 --shard-index "${index}" --resume \
    >"${METRICS}/logs/autopcp_${index}.log" 2>&1 &
  autopcp_pids+=("$!")
done
wait_workers autopcp "${autopcp_pids[@]}"
"${PYTHON}" -m evaluation.merge_metric_shards \
  --metric autopcp --input "${RESULTS}" --metric-dir "${METRICS}" \
  --shard-root "${METRICS}/shards" --num-shards 8

speaker_pids=()
mkdir -p "${METRICS}/shards/speaker"
for ((index = 0; index < 8; index++)); do
  gpu="${GPU_IDS[${index}]}"
  part="${METRICS}/shards/speaker/part_$(printf '%03d' "${index}")"
  mkdir -p "${part}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.speaker_similarity \
    --input "${RESULTS}" --output-dir "${part}" --model "${WAVLM_SPEAKER_MODEL}" \
    --device cuda:0 --batch-size "${SPEAKER_BATCH_SIZE:-8}" \
    --num-shards 8 --shard-index "${index}" --resume --local-files-only \
    >"${METRICS}/logs/speaker_${index}.log" 2>&1 &
  speaker_pids+=("$!")
done
wait_workers speaker "${speaker_pids[@]}"
speaker_parts=()
for ((index = 0; index < 8; index++)); do
  speaker_parts+=(--part "${METRICS}/shards/speaker/part_$(printf '%03d' "${index}")/per_sample_speaker_similarity.jsonl")
done
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.merge_speaker_similarity \
  --input "${RESULTS}" "${speaker_parts[@]}" --output-dir "${METRICS}"

PREFIX_ROOT="${METRICS}/prefix_asr"
if [[ ! -e "${PREFIX_ROOT}" ]]; then
  "${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.build_prefix_asr_manifest \
    --results "${RESULTS}" --output-root "${PREFIX_ROOT}"
fi
run_asr "${PREFIX_ROOT}/prefix_asr_manifest.jsonl" "${PREFIX_ROOT}/metrics" prefix_asr
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.score_prefix_asr \
  --asr-results "${PREFIX_ROOT}/metrics/asr_results.jsonl" \
  --runtime-results "${RESULTS}" \
  --output "${PREFIX_ROOT}/useful_audio.json" \
  --minimum-similarity 0.50 --minimum-content-units 2

"${PYTHON}" - "${RESULTS}" "${METRICS}/complete.json" <<'PY'
import json, sys
from pathlib import Path

results = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = sum(1 for line in results.open(encoding="utf-8") if line.strip())
required = [
    "asr_results.jsonl", "text_bleu.json", "speech_bleu.json", "slc.json",
    "utmos.json", "autopcp.json", "speaker_similarity.json",
    "prefix_asr/useful_audio.json",
]
missing = [name for name in required if not (output.parent / name).is_file()]
if missing:
    raise SystemExit(f"quality metric completion is missing: {missing}")
output.write_text(json.dumps({"status": "complete", "result_rows": rows, "artifacts": required}, indent=2) + "\n")
PY

printf '%s\n' "${METRICS}"
