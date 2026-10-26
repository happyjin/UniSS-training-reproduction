#!/usr/bin/env bash
set -euo pipefail
# Cross-node EFA/NCCL check — run this before the first real multi-node job.
#
#   NODELIST="10.1.6.203 10.1.6.77" scripts/multinode/preflight_nccl.sh
#
# Pass criteria:
#   1. the log says  NET/OFI Selected provider is efa   (not "NET/Socket")
#   2. large-message busbw is in the hundreds of GB/s, not single digits

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NODELIST="${NODELIST:?set NODELIST=\"ip1 ip2 ...\"}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/usr/local/cuda/efa/test-cuda-12.8}"
MPIRUN="${MPIRUN:-/opt/amazon/openmpi/bin/mpirun}"

source "${REPO_ROOT}/scripts/multinode/env_multinode.sh"
export NCCL_DEBUG=INFO

read -r -a nodes <<<"${NODELIST}"
hosts="$(IFS=,; printf '%s' "${nodes[*]/%/:${GPUS_PER_NODE}}")"
nranks=$(( ${#nodes[@]} * GPUS_PER_NODE ))

[[ -x "${NCCL_TESTS_DIR}/all_reduce_perf" ]] || {
  echo "missing ${NCCL_TESTS_DIR}/all_reduce_perf" >&2; exit 2; }

exec "${MPIRUN}" -n "${nranks}" -N "${GPUS_PER_NODE}" --host "${hosts}" \
  --mca plm_rsh_no_tree_spawn 1 --bind-to none \
  -x LD_LIBRARY_PATH -x PATH \
  -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA -x FI_EFA_FORK_SAFE \
  -x NCCL_SOCKET_IFNAME -x NCCL_CROSS_NIC -x NCCL_BUFFSIZE -x NCCL_P2P_NET_CHUNKSIZE \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  "${NCCL_TESTS_DIR}/all_reduce_perf" -b 8 -e 8G -f 2 -g 1
