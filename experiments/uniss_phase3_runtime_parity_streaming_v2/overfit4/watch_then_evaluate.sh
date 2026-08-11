#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit4/config.env"

while true; do
  latest=0
  if [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    latest="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
  fi
  if (( latest >= COVERAGE_EPOCHS )) && \
     ! pgrep -f 'overfit4/pretrain_overfit4.py' >/dev/null; then
    break
  fi
  sleep 10
done

ITERATION="${COVERAGE_EPOCHS}" TAG=content_consolidation_v1 \
  bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit4/evaluate_checkpoint.sh"

summary="${REPO_ROOT}/reports/uniss_phase3_runtime_parity_streaming_v2/overfit4_v1_content_consolidation_v1/iter_$(printf '%07d' "${COVERAGE_EPOCHS}")/summary.json"
"${PYTHON}" - "${summary}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
sample = value["samples"][0]
print(json.dumps({
    "quality_passed": value["quality_passed"],
    "generated_text": sample["generated_text"],
    "text_similarity": sample["text_similarity"],
    "natural_eos": sample["natural_eos"],
    "first_audio_source_ms": sample["first_audio_source_ms"],
    "first_audio_wall_ms": sample["first_audio_wall_ms"],
    "rtf": sample["rtf"],
    "quality_failures": sample["quality_failures"],
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if value["quality_passed"] else 2)
PY
