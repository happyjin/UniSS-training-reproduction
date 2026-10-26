#!/usr/bin/env bash
set -euo pipefail
# Multi-node entry point for the existing single-node training scripts.
#
# It does not rewrite them: torchrun reads PET_NNODES / PET_NODE_RANK /
# PET_MASTER_ADDR as defaults for any flag the inner script does not pass on the
# command line, so scripts/train_phase*.sh go multi-node unmodified.
#
# Per-node form (run the same command on every node, NODE_RANK differs):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.1.6.203 \
#     scripts/multinode/launch_multinode.sh scripts/train_phase1.sh --train-iters 20
#
# Fan-out form (run once on any node that can ssh to all of them; rank order
# follows NODELIST, master is the first entry):
#   NODELIST="10.1.6.203 10.1.6.77" \
#     scripts/multinode/launch_multinode.sh scripts/train_phase1.sh --train-iters 20

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${1:?usage: launch_multinode.sh <train_script> [args...]}"
shift || true

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"
NODELIST="${NODELIST:-}"

if [[ -n "${NODELIST}" && -z "${MULTINODE_CHILD:-}" ]]; then
  read -r -a nodes <<<"${NODELIST}"
  (( ${#nodes[@]} >= 2 )) || { echo "NODELIST needs at least 2 nodes" >&2; exit 2; }
  master="${MASTER_ADDR:-${nodes[0]}}"
  run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  logdir="${REPO_ROOT}/logs/multinode/${run_id}"
  [[ ! -e "${logdir}" ]] || { echo "refusing to overwrite ${logdir}" >&2; exit 3; }
  mkdir -p "${logdir}"
  printf 'run_id=%s master=%s nodes=%s logs=%s\n' "${run_id}" "${master}" "${#nodes[@]}" "${logdir}"
  pids=()
  for i in "${!nodes[@]}"; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${nodes[$i]}" \
      "cd '${REPO_ROOT}' && MULTINODE_CHILD=1 RUN_ID='${run_id}' NNODES=${#nodes[@]} NODE_RANK=${i} \
       MASTER_ADDR='${master}' MASTER_PORT=${MASTER_PORT} NPROC_PER_NODE=${NPROC_PER_NODE} \
       bash scripts/multinode/launch_multinode.sh '${TRAIN_SCRIPT}' $*" \
      >"${logdir}/node${i}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  echo "exit status ${status}; per-node logs under ${logdir}"
  exit "${status}"
fi

NNODES="${NNODES:?set NNODES (or use NODELIST)}"
NODE_RANK="${NODE_RANK:?set NODE_RANK}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR}"

# shellcheck source=scripts/multinode/env_multinode.sh
source "${REPO_ROOT}/scripts/multinode/env_multinode.sh"

# torchrun defaults picked up by any flag the inner script omits.
export PET_NNODES="${NNODES}"
export PET_NODE_RANK="${NODE_RANK}"
export PET_MASTER_ADDR="${MASTER_ADDR}"
export PET_MASTER_PORT="${MASTER_PORT}"
export PET_NPROC_PER_NODE="${NPROC_PER_NODE}"
# Vars the repo's own scripts read directly.
export NPROC_PER_NODE MASTER_PORT MASTER_ADDR

script_path="${TRAIN_SCRIPT}"
[[ -f "${script_path}" ]] || script_path="${REPO_ROOT}/${TRAIN_SCRIPT}"
[[ -f "${script_path}" ]] || { echo "no such train script: ${TRAIN_SCRIPT}" >&2; exit 2; }

printf '[node %s/%s] master=%s:%s gpus=%s -> %s\n' \
  "${NODE_RANK}" "${NNODES}" "${MASTER_ADDR}" "${MASTER_PORT}" "${NPROC_PER_NODE}" "${script_path}"
exec bash "${script_path}" "$@"
