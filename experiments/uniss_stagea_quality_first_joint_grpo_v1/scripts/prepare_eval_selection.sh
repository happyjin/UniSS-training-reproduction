#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"

OUTPUT=${1:-${EXPERIMENT_ROOT}/evaluation/protocols/validation64_e2e16_seed20260825.json}
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing to overwrite frozen evaluation selection: ${OUTPUT}" >&2
  exit 3
}
mkdir -p "$(dirname -- "${OUTPUT}")"

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
"${PYTHON_BIN}" \
  "${REPO_ROOT}/experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/evaluation/selection.py" \
  --input "${REPO_ROOT}/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl" \
  --output "${OUTPUT}" \
  --samples 64 \
  --e-s2s-samples 16 \
  --seed 20260825

