#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit6/config.env"
while true; do
  latest=0
  [[ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]] && \
    latest="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
  if (( latest >= COVERAGE_EPOCHS )) && ! pgrep -f 'overfit6/pretrain_overfit6.py' >/dev/null; then
    break
  fi
  sleep 10
done
ITERATION="${COVERAGE_EPOCHS}" TAG=untied_parallel_v1 \
  bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/overfit6/evaluate_checkpoint.sh"
summary="${REPO_ROOT}/reports/uniss_phase3_runtime_parity_streaming_v2/overfit6_v1_untied_parallel_v1/iter_$(printf '%07d' "${COVERAGE_EPOCHS}")/summary.json"
"${PYTHON}" - "${summary}" <<'PY'
import json, sys
v=json.load(open(sys.argv[1], encoding='utf-8')); s=v['samples'][0]
print(json.dumps({'strict_gate_passed':v['quality_passed'], 'generated_text':s['generated_text'],
 'text_similarity':s['text_similarity'], 'natural_writes':s['natural_writes'],
 'natural_eos':s['natural_eos'], 'first_audio_source_ms':s['first_audio_source_ms'],
 'first_audio_wall_ms':s['first_audio_wall_ms'], 'rtf':s['rtf'],
 'forced_writes':s['forced_writes'], 'revision_violations':s['committed_revision_violations'],
 'quality_failures':s['quality_failures']}, ensure_ascii=False, indent=2))
raise SystemExit(0 if v['quality_passed'] else 2)
PY

