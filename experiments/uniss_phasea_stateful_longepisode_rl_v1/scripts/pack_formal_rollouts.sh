#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

TRAIN_RUN=${1:-formal_train64_g4_v1}
VALID_RUN=${2:-formal_valid16_g4_v1}
PACK_ID=${3:-formal_64x4_train_16x4_valid_v1}
TRAIN_MERGED=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${TRAIN_RUN}/ROLLOUT_MERGED.json
VALID_MERGED=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${VALID_RUN}/ROLLOUT_MERGED.json
OUTPUT=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/${PACK_ID}
TRAIN_REPLAY=${REPO_ROOT}/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/task_pools/task_pool_formal_p4_20260820T154500Z_train/packed/train_phase3_performance_replay.jsonl
VALID_REPLAY=${REPO_ROOT}/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/task_pools/task_pool_formal_p4_20260820T154500Z_valid/packed/valid_phase3_performance_replay.jsonl
[[ -f "${TRAIN_MERGED}" && -f "${VALID_MERGED}" ]] || {
  echo "formal rollout merge files are incomplete" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}"

mapfile -t TRAIN_TRAJECTORIES < <("${PYTHON}" - "${TRAIN_MERGED}" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["trajectory_paths"]:
    print(value)
PY
)
mapfile -t VALID_TRAJECTORIES < <("${PYTHON}" - "${VALID_MERGED}" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["trajectory_paths"]:
    print(value)
PY
)

train_args=()
for path in "${TRAIN_TRAJECTORIES[@]}"; do train_args+=(--trajectory "${path}"); done
valid_args=()
for path in "${VALID_TRAJECTORIES[@]}"; do valid_args+=(--trajectory "${path}"); done

"${PYTHON}" "${EXPERIMENT_ROOT}/training/pack_trajectories.py" \
  "${train_args[@]}" \
  --phase3-replay "${TRAIN_REPLAY}" \
  --match-replay-to-rl \
  --output "${OUTPUT}/train_packs.jsonl" \
  --seq-length 18000
"${PYTHON}" "${EXPERIMENT_ROOT}/training/pack_trajectories.py" \
  "${valid_args[@]}" \
  --phase3-replay "${VALID_REPLAY}" \
  --match-replay-to-rl \
  --output "${OUTPUT}/valid_packs.jsonl" \
  --seq-length 18000

echo "TRAIN=${OUTPUT}/train_packs.jsonl"
echo "VALID=${OUTPUT}/valid_packs.jsonl"
