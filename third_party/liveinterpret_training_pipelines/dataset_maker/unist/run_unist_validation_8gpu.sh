#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python}"
WORLD_SIZE="${WORLD_SIZE:-8}"
REPORT_DIR="${REPORT_DIR:-/opt/dlami/nvme/zeyangsong/data/unist/validation/unist_qwen3_s2s_v2_20260730}"
SAMPLES="${SAMPLES:-${REPORT_DIR}/roundtrip_samples.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

mkdir -p "${REPORT_DIR}/gpu/logs"
pids=()
cleanup_children() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM

for ((rank=0; rank<WORLD_SIZE; rank++)); do
  log_path="${REPORT_DIR}/gpu/logs/rank$(printf '%02d' "${rank}").log"
  echo "Starting validation rank ${rank}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${rank}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_unist_codec_roundtrip.py" \
      --samples "${SAMPLES}" --report-dir "${REPORT_DIR}" \
      --rank "${rank}" --world-size "${WORLD_SIZE}" --device cuda:0 \
      --max-samples "${MAX_SAMPLES}" >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one GPU validation rank failed." >&2
  exit "${status}"
fi

if [[ "${MAX_SAMPLES}" -eq 0 ]]; then
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_unist_codec_roundtrip.py" \
      --samples "${SAMPLES}" --report-dir "${REPORT_DIR}" --finalize-only
else
  echo "Smoke validation complete; finalization intentionally skipped."
fi
