#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON_BIN=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
MODEL="${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"
MANIFEST_ROOT="${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage00_matching_offline"
RUN_ID=${RUN_ID:?RUN_ID is required}
OUTPUT_ROOT="${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage00_matching_offline/${RUN_ID}"

cd "${REPO_ROOT}"
[[ -f "${MANIFEST_ROOT}/matching_stage_a_334.jsonl" ]] || {
  echo "missing matching offline manifest" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to overwrite matching offline run: ${OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}/parts" "${OUTPUT_ROOT}/logs"

pids=()
for rank in $(seq 0 7); do
  printf -v tag '%02d' "${rank}"
  CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage00_matching_offline.evaluate_offline_asr \
    --manifest "${MANIFEST_ROOT}/parts/part_${tag}.jsonl" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_ROOT}/parts/part_${tag}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --max-new-tokens 1500 \
    --repetition-penalty 1.1 \
    >"${OUTPUT_ROOT}/logs/part_${tag}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || {
  tail -n 80 "${OUTPUT_ROOT}"/logs/part_??.log >&2 || true
  exit 1
}

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage00_matching_offline.merge_offline_asr \
  --manifest "${MANIFEST_ROOT}/matching_stage_a_334.jsonl" \
  --root "${OUTPUT_ROOT}" \
  --parts 8 \
  --expected-records 334 \
  | tee "${OUTPUT_ROOT}/merge.log"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
