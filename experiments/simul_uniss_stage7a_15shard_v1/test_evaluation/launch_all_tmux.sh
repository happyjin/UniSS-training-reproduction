#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="${1:-}"
if [[ -n "${SMOKE}" && "${SMOKE}" != "--smoke" ]]; then
  echo "Usage: $0 [--smoke]" >&2
  exit 2
fi
declare -A COMMANDS=(
  [simul_stage7a_test_e0]="${SCRIPT_DIR}/run_one_2gpu.sh e0_stage6 ${SMOKE}"
  [simul_stage7a_test_e1]="${SCRIPT_DIR}/run_one_2gpu.sh e1_continued_sft ${SMOKE}"
  [simul_stage7a_test_e2]="${SCRIPT_DIR}/run_one_2gpu.sh e2_grpo_g4 ${SMOKE}"
  [simul_stage7a_test_e3]="${SCRIPT_DIR}/run_one_2gpu.sh e3_grpo_g8 ${SMOKE}"
)
for session in simul_stage7a_test_e0 simul_stage7a_test_e1 simul_stage7a_test_e2 simul_stage7a_test_e3; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to reuse tmux session ${session}" >&2
    exit 1
  fi
done
for session in simul_stage7a_test_e0 simul_stage7a_test_e1 simul_stage7a_test_e2 simul_stage7a_test_e3; do
  tmux new-session -d -s "${session}" "cd '${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../" && pwd)}' && ${COMMANDS[${session}]}"
  echo "started ${session}: ${COMMANDS[${session}]}"
done
