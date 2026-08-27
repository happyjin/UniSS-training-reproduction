#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=${1:?usage: pack_rollouts.sh ROLLOUT_ID PACK_ID}
PACK_ID=${2:?missing pack ID}
MERGED=${REPO_ROOT}/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}/ROLLOUT_MERGED.json
OUTPUT=${REPO_ROOT}/data/processed/uniss_phasea_route_aligned_constrained_grpo_v1/${PACK_ID}
[[ -f "${MERGED}" ]] || { echo "missing ${MERGED}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
mkdir -p "${OUTPUT}"
mapfile -t TRAJECTORIES < <("${PYTHON}" - "${MERGED}" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["trajectory_paths"]:
    print(value)
PY
)
args=(); for path in "${TRAJECTORIES[@]}"; do args+=(--trajectory "${path}"); done
for split in train valid; do
  "${PYTHON}" "${EXPERIMENT_ROOT}/training/pack_trajectories.py" \
    "${args[@]}" --phase3-replay "${PHASE3_REPLAY}" \
    --output "${OUTPUT}/${split}_packs.jsonl" --seq-length 18000
done
echo "OUTPUT=${OUTPUT}"

