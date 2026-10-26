#!/usr/bin/env bash
# Sourced by the multi-node launchers. EFA / NCCL / library environment for the
# p5e.48xlarge nodes (8x H200 NVLink, 32x EFA) this cluster is built from.
#
#   source scripts/multinode/env_multinode.sh
#
# Every variable is overridable from the caller: this file only fills blanks.

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
PYTHON="${PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- CUDA + pip-installed NVIDIA libs, same resolution the single-node scripts use ---
SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBRARY_PATH="$(find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | sort | paste -sd: -)"

# --- EFA / libfabric / aws-ofi-nccl: this is what makes inter-node NCCL use RDMA ---
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib64:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FI_PROVIDER="${FI_PROVIDER:-efa}"
export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"

# --- NCCL: bootstrap over the node's primary ENA interface, data plane over EFA ---
DEFAULT_IFACE="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${DEFAULT_IFACE:-enp107s0}}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE:-524288}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export PATH="$(dirname "${PYTHON}"):${PATH}"
