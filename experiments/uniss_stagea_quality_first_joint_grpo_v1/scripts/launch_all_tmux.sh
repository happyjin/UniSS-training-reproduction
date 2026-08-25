#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-}
[[ -z "${MODE}" || "${MODE}" == "--smoke" ]] || { echo "Usage: $0 [--smoke]" >&2; exit 2; }

RUN_VARIANT=${RUN_VARIANT:-}
DATA_WORKERS=${DATA_WORKERS:-0}

declare -A SESSIONS=(
  [a1_sft]=stagea_grpo_a1
  [a2_g4]=stagea_grpo_a2
  [a3_g8]=stagea_grpo_a3
  [a4_g8_seed2]=stagea_grpo_a4
)
for arm in a1_sft a2_g4 a3_g8 a4_g8_seed2; do
  session=${SESSIONS[$arm]}
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
done
for arm in a1_sft a2_g4 a3_g8 a4_g8_seed2; do
  session=${SESSIONS[$arm]}
  tmux new-session -d -s "${session}" "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && RUN_VARIANT=${RUN_VARIANT} DATA_WORKERS=${DATA_WORKERS} exec ${SCRIPT_DIR}/run_arm_2gpu.sh ${arm} ${MODE}"
done
tmux list-sessions | rg 'stagea_grpo_a[1-4]'
