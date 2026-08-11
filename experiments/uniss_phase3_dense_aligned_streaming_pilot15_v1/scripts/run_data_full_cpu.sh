#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"
cd "${REPO_ROOT}"

build_split() {
  local split="$1"
  local source="$2"
  local parts_root="$3"
  local output="$4"
  local workers="$5"
  mkdir -p "${parts_root}"
  export split source parts_root workers
  export REPO_ROOT PYTHON LOW_WATERMARK_MS TARGET_BUFFER_MS SEMANTIC_HISTORY_TOKENS
  seq 0 "$((workers - 1))" | xargs -P "${workers}" -I{} bash -c '
    part="$(printf "%03d" "$1")"
    out_dir="${parts_root}/part-${part}"
    mkdir -p "${out_dir}"
    "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions \
      --input-manifest "${source}" \
      --output "${out_dir}/dense.jsonl" \
      --marker "${out_dir}/PART_COMPLETE.json" \
      --split "${split}" --num-parts "${workers}" --part-index "$1" \
      --low-watermark-ms "${LOW_WATERMARK_MS}" \
      --target-buffer-ms "${TARGET_BUFFER_MS}" \
      --semantic-history-tokens "${SEMANTIC_HISTORY_TOKENS}" \
      --progress-interval 10000
  ' _ {}
  "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.assemble_dense_sessions \
    --parts-root "${parts_root}" --output "${output}" \
    --marker "${output}.complete.json" --expected-parts "${workers}"
  "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.audit \
    --manifest "${output}" \
    --output "${REPORT_ROOT}/dense_${split}_audit.json"
}

mkdir -p "${DATA_ROOT}" "${REPORT_ROOT}"
build_split train "${FORMAL_TRAIN}" "${TRAIN_PARTS_ROOT}" "${DENSE_TRAIN}" "${TRAIN_DATA_WORKERS}"
build_split valid "${FORMAL_VALID}" "${VALID_PARTS_ROOT}" "${DENSE_VALID}" "${VALID_DATA_WORKERS}"

