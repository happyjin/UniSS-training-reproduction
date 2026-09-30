#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:?RUN_ID must name the Stage 00 baseline run}
OUTPUT_ROOT="${EVAL_ROOT}/stage00_phase3_offline_${RUN_ID}"
VALIDATION_ROOT="${DATA_ROOT}/stage00_fixed_validation_v1"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite offline baseline: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${TMPDIR}"

run_workers() {
  local kind=$1
  shift
  local modes=("$@")
  local pids=()
  local rank manifest output
  for rank in $(seq 0 7); do
    printf -v manifest '%s/pilot15_%s.part%02d.jsonl' \
      "${VALIDATION_ROOT}" "${kind}" "${rank}"
    output="${OUTPUT_ROOT}/${kind}/part$(printf '%02d' "${rank}")"
    mkdir -p "$(dirname "${output}")"
    extra=()
    if [[ "${kind}" == "text_256" ]]; then
      extra+=(--skip-audio-decode)
    else
      extra+=(--save-source-audio --save-reference-audio)
    fi
    CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON_BIN}" \
      "${REPO_ROOT}/training/generate_unist_eval_audio.py" \
      --manifest "${manifest}" \
      --model "${PHASE3_HF_CHECKPOINT}" \
      --speech-tokenizer "${REPO_ROOT}/pretrained_models/UniSS" \
      --output-dir "${output}" \
      --mode "${modes[@]}" \
      --limit-records 0 \
      --max-new-tokens 1500 \
      --temperature 0 \
      --top-p 0.8 \
      --top-k -1 \
      --repetition-penalty 1.1 \
      --seed 20260816 \
      --dtype bfloat16 \
      --device cuda:0 \
      --local-files-only \
      "${extra[@]}" \
      >"${OUTPUT_ROOT}/logs/${kind}_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    tail -n 80 "${OUTPUT_ROOT}/logs/${kind}_rank"*.log >&2 || true
    return "${status}"
  fi
}

# Names match the immutable manifest file prefixes produced by Stage 00.
run_workers text_256 quality performance
run_workers audio_64 quality performance direct_s2st tts

# Normalize directory names expected by the deterministic aggregator.
mv "${OUTPUT_ROOT}/text_256" "${OUTPUT_ROOT}/text"
mv "${OUTPUT_ROOT}/audio_64" "${OUTPUT_ROOT}/audio"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.aggregate_offline_baseline \
  --root "${OUTPUT_ROOT}" \
  --parts 8 \
  --expected-text-records 256 \
  --expected-audio-records 64 \
  | tee "${OUTPUT_ROOT}/aggregate.log"

echo "offline_baseline=${OUTPUT_ROOT}"
