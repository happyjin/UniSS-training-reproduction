#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}

OUTPUT_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_rl_trainlong_eval_v1
REPORT=${REPO_ROOT}/reports/uniss_phasea_rl_trainlong_eval_v1/REPORT.zh-CN.md
PROTOCOL=${EXPERIMENT_ROOT}/evaluation/protocol_train_seen_long8.json
SESSION_HOLDER=uniss_gpu_load_60
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${REPORT}")" "${REPO_ROOT}/logs/uniss_phasea_rl_trainlong_eval_v1"
exec 9>"${OUTPUT_ROOT}/formal.lock"
flock 9
if [[ -f "${REPORT}" ]]; then
  echo "REPORT=${REPORT}"
  exit 0
fi

temporary_dir=$(mktemp -d /tmp/uniss_trainlong_protocol.XXXXXX)
temporary=${temporary_dir}/protocol.json
restore_holder=0
cleanup() {
  [[ ! -f "${temporary}" ]] || unlink "${temporary}"
  rmdir "${temporary_dir}" 2>/dev/null || true
  if (( restore_holder )) && ! tmux has-session -t "${SESSION_HOLDER}" 2>/dev/null; then
    bash "${REPO_ROOT}/experiments/uniss_phasea_stateful_longepisode_rl_v1/scripts/start_gpu_holder_after_completion.sh" || true
  fi
}
trap cleanup EXIT
"${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/build_protocol.py" \
  --train-episodes "${TRAIN_EPISODES}" --valid-episodes "${VALID_EPISODES}" \
  --formal-rollout "${FORMAL_TRAIN_ROLLOUT}" --per-direction 4 --output "${temporary}"
cmp --silent "${temporary}" "${PROTOCOL}" || {
  echo "committed train-seen protocol differs from fresh deterministic selection" >&2
  exit 3
}

gpu_count=$(nvidia-smi -L 2>/dev/null | rg -c '^GPU [0-9]+:' || true)
gpu_count=${gpu_count:-0}
[[ ${gpu_count} -ge 8 ]] || { echo "eight visible GPUs are required; found ${gpu_count}" >&2; exit 4; }
if tmux has-session -t "${SESSION_HOLDER}" 2>/dev/null; then
  restore_holder=1
  tmux kill-session -t "${SESSION_HOLDER}"
  for _ in 1 2 3 4 5; do
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l || true)
    [[ ${remaining} -eq 0 ]] && break
    sleep 2
  done
fi
remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l || true)
[[ ${remaining} -eq 0 ]] || {
  echo "refusing to start while ${remaining} non-holder GPU processes remain" >&2
  exit 5
}

run_arm() {
  local run_id=$1 adapter=$2
  bash "${SCRIPT_DIR}/run_arm_8gpu.sh" "${run_id}" "${OUTPUT_ROOT}/${run_id}" "${adapter}"
}
run_arm phasea_iter381_runtime_v2 NONE
run_arm rl_iter15_runtime_v2 "${RL_CHECKPOINT_ROOT}/iter_0000015"
run_arm rl_iter30_runtime_v2 "${RL_CHECKPOINT_ROOT}/iter_0000030"
run_arm rl_iter45_runtime_v2 "${RL_CHECKPOINT_ROOT}/iter_0000045"

"${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/write_report.py" \
  --score "${OUTPUT_ROOT}/phasea_iter381_runtime_v2/SCORED.json" \
  --score "${OUTPUT_ROOT}/rl_iter15_runtime_v2/SCORED.json" \
  --score "${OUTPUT_ROOT}/rl_iter30_runtime_v2/SCORED.json" \
  --score "${OUTPUT_ROOT}/rl_iter45_runtime_v2/SCORED.json" \
  --external "${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/results.json" \
  --external "${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/results.json" \
  --external "${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/results.json" \
  --external "${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/results.json" \
  --output "${REPORT}"

echo "REPORT=${REPORT}"
