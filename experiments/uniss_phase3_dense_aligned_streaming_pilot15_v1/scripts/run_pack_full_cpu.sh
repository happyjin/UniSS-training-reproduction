#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/config.env"
cd "${REPO_ROOT}"

pack_parts() {
  local dense_parts_root="$1"
  local pack_parts_root="$2"
  local output="$3"
  local workers="$4"
  mkdir -p "${pack_parts_root}" "${PACKED_ROOT}"
  export dense_parts_root pack_parts_root workers REPO_ROOT PYTHON PHASE3_MODEL SEQ_LENGTH
  seq 0 "$((workers - 1))" | xargs -P "${workers}" -I{} bash -c '
    part="$(printf "%03d" "$1")"
    dense="${dense_parts_root}/part-${part}/dense.jsonl"
    out_dir="${pack_parts_root}/part-${part}"
    mkdir -p "${out_dir}"
    "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.pack_dense_sessions \
      --dense-manifest "${dense}" \
      --tokenizer "${PHASE3_MODEL}" \
      --output "${out_dir}/packed.jsonl" \
      --marker "${out_dir}/PACK_COMPLETE.json" \
      --seq-length "${SEQ_LENGTH}" --progress-interval 10000
  ' _ {}
  "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.assemble_dense_packs \
    --parts-root "${pack_parts_root}" --output "${output}" \
    --marker "${output}.complete.json" --expected-parts "${workers}" \
    --seq-length "${SEQ_LENGTH}"
}

if [[ ! -f "${DENSE_TRAIN}.complete.json" || ! -f "${DENSE_VALID}.complete.json" ]]; then
  echo "Dense train/valid assembly is incomplete; refusing to pack." >&2
  exit 1
fi
pack_parts "${TRAIN_PARTS_ROOT}" "${PACK_PARTS_ROOT}" "${TRAJECTORY_PACKED}" "${TRAIN_DATA_WORKERS}"
pack_parts "${VALID_PARTS_ROOT}" "${VALID_PACK_PARTS_ROOT}" "${VALID_TRAJECTORY_PACKED}" "${VALID_DATA_WORKERS}"
"${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_training_manifest \
  --trajectory-packed "${TRAJECTORY_PACKED}" \
  --replay-packed "${PHASE3_REPLAY_PACKED}" \
  --replay-offsets "${PHASE3_REPLAY_OFFSETS}" \
  --output-root "${PACKED_ROOT}" \
  --coverage-epochs "${COVERAGE_EPOCHS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --data-parallel-size "${NPROC_PER_NODE}" \
  --replay-fraction "${REPLAY_FRACTION}" \
  --seed "${SEED}"
