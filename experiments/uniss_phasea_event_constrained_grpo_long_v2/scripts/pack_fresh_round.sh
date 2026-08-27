#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
ROLLOUT_ID=${1:?usage: pack_fresh_round.sh ROLLOUT_ID PACK_ID}
PACK_ID=${2:?missing pack ID}
ROLLOUT_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/${ROLLOUT_ID}
OUTPUT=${REPO_ROOT}/data/processed/uniss_phasea_event_constrained_grpo_long_v2/${PACK_ID}
mapfile -t TRAJECTORIES < <(find "${ROLLOUT_ROOT}/workers" -mindepth 2 -maxdepth 2 -name trajectories.jsonl -print | sort)
[[ ${#TRAJECTORIES[@]} -eq 8 ]] || { echo "expected eight trajectory files" >&2; exit 2; }
ARGS=()
for path in "${TRAJECTORIES[@]}"; do ARGS+=(--trajectory "${path}"); done
"${PYTHON}" "${EXPERIMENT_ROOT}/training/pack_rollouts.py" \
  "${ARGS[@]}" --phase3-replay "${PHASE3_REPLAY}" --output "${OUTPUT}" --seq-length 18000
