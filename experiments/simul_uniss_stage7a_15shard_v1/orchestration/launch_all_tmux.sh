#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A COMMANDS=(
  [simul_stage7a_e0_baselines]="${ROOT}/e0_baselines/run_2gpu.sh"
  [simul_stage7a_e1_sft]="${ROOT}/e1_continued_sft/run_2gpu.sh"
  [simul_stage7a_e2_grpo_g4]="${ROOT}/e2_grpo_g4/run_2gpu.sh"
  [simul_stage7a_e3_grpo_g8]="${ROOT}/e3_grpo_g8/run_2gpu.sh"
)
for session in "${!COMMANDS[@]}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to reuse tmux session: ${session}" >&2
    exit 1
  fi
done
for session in simul_stage7a_e0_baselines simul_stage7a_e1_sft simul_stage7a_e2_grpo_g4 simul_stage7a_e3_grpo_g8; do
  command="${COMMANDS[${session}]}"
  tmux new-session -d -s "${session}" "cd '${ROOT}/../..' && '${command}'"
  echo "started ${session}: ${command}"
done
