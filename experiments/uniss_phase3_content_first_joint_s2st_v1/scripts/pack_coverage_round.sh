#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
ROLLOUT_ID=${1:?usage: pack_coverage_round.sh ROLLOUT_ID PACK_ID}
PACK_ID=${2:?missing pack ID}
ROLLOUT_ROOT=${REPO_ROOT}/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/${ROLLOUT_ID}
OUTPUT=${REPO_ROOT}/data/processed/uniss_phase3_content_first_joint_s2st_v1/${PACK_ID}
PHASE3_REPLAY=${REPO_ROOT}/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/task_pools/task_pool_formal_p4_20260820T154500Z_train/packed/train_phase3_performance_replay.jsonl
mapfile -t TRAJECTORIES < <(find "${ROLLOUT_ROOT}/workers" -mindepth 2 -maxdepth 2 -name trajectories.jsonl -print | sort)
[[ ${#TRAJECTORIES[@]} -eq 64 ]] || { echo "expected 64 trajectory files" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
ARGS=()
for path in "${TRAJECTORIES[@]}"; do ARGS+=(--trajectory "${path}"); done
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
"${PYTHON}" -m experiments.uniss_phasea_coverage_constrained_grpo_v3.training.pack_rollouts \
  "${ARGS[@]}" --phase3-replay "${PHASE3_REPLAY}" --output "${OUTPUT}" --seq-length 18000
