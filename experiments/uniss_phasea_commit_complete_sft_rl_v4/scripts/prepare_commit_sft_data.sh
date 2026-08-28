#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
DATA_ID=${1:-commit_sft_min2_v1}
OUTPUT=${REPO_ROOT}/data/processed/uniss_phasea_commit_complete_sft_rl_v4/${DATA_ID}
EVENT_OUTPUT=${OUTPUT}/commit_events.jsonl
PACK_OUTPUT=${OUTPUT}/packs
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}"
"${PYTHON}" "${EXPERIMENT_ROOT}/data/build_commit_events.py" \
  --input "${EVENTS}" --output "${EVENT_OUTPUT}" --minimum-delta-tokens 2
"${PYTHON}" "${REPO_ROOT}/experiments/uniss_phasea_coverage_constrained_grpo_v3/data/build_action_packs.py" \
  --events "${EVENT_OUTPUT}" --phase3-replay "${PHASE3_REPLAY}" \
  --output "${PACK_OUTPUT}" --seq-length 18000
"${PYTHON}" - "${EVENT_OUTPUT}.audit.json" "${PACK_OUTPUT}/AUDIT.json" <<'PY'
import json, sys
events=json.load(open(sys.argv[1]))
packs=json.load(open(sys.argv[2]))
assert events['status']=='passed' and packs['status']=='passed'
for split in ('train','valid'):
    assert events['splits'][split]['WAIT'] > 0
    assert events['splits'][split]['COMMIT'] > 0
assert packs['train']['action_tokens'] > 0 and packs['train']['response_tokens'] > 0
print(json.dumps({'status':'passed','events':events['events'],'packs':packs['train']['records']},sort_keys=True))
PY
echo "OUTPUT=${OUTPUT}"
