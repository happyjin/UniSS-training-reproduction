#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/config.env"
EVAL_FAMILY="${EVAL_FAMILY:-overfit2_v1}"
EXPORT_FAMILY="${EXPORT_FAMILY:-overfit2_v1}"

while true; do
  latest=0
  if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    latest="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
  fi
  if (( latest >= COVERAGE_EPOCHS )) && \
     ! pgrep -f 'overfit2/pretrain_overfit2.py' >/dev/null; then
    break
  fi
  sleep 10
done

ITERATION="${COVERAGE_EPOCHS}" TAG=natural_eos_v1 \
EVAL_FAMILY="${EVAL_FAMILY}" EXPORT_FAMILY="${EXPORT_FAMILY}" \
  bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit2/evaluate_checkpoint.sh"

"${PYTHON}" - "${REPO_ROOT}/reports/uniss_phase3_runtime_parity_streaming_v2/${EVAL_FAMILY}_natural_eos_v1/iter_$(printf '%07d' "${COVERAGE_EPOCHS}")/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
sample = value["samples"][0]
print(json.dumps({
    "quality_passed": value["quality_passed"],
    "first_write_source_ms": sample["first_write_source_ms"],
    "first_audio_source_ms": sample["first_audio_source_ms"],
    "natural_eos": sample["natural_eos"],
    "text_similarity": sample["text_similarity"],
    "semantic_tokens": sample["semantic_tokens"],
    "rtf": sample["rtf"],
    "quality_failures": sample["quality_failures"],
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if value["quality_passed"] else 2)
PY
