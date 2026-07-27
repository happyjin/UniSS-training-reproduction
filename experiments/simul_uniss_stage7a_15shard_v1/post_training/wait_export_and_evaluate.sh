#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
RUN_DIR="${EVAL_ROOT}/post_training_action_eval_full"
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to overwrite ${RUN_DIR}" >&2; exit 1; }

declare -A LABELS=(
  [e1_continued_sft]="${CHECKPOINT_ROOT}/e1_continued_sft_full"
  [e2_grpo_g4]="${CHECKPOINT_ROOT}/e2_grpo_g4_full"
  [e3_grpo_g8]="${CHECKPOINT_ROOT}/e3_grpo_g8_full"
)

echo "Waiting for E1/E2/E3 TRAINING_COMPLETE markers..."
while true; do
  ready=1
  for label in e1_continued_sft e2_grpo_g4 e3_grpo_g8; do
    [[ -f "${LABELS[${label}]}/TRAINING_COMPLETE.json" ]] || ready=0
  done
  [[ "${ready}" == 1 ]] && break
  sleep 30
done

mkdir -p "${RUN_DIR}/logs"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits -l 2 > "${RUN_DIR}/gpu_monitor.csv" &
MONITOR_PID="$!"
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

index=0
for label in e1_continued_sft e2_grpo_g4 e3_grpo_g8; do
  checkpoint="${LABELS[${label}]}/best.pt"
  export_dir="${EXPORT_ROOT}/${label}_best_hf"
  if [[ ! -d "${export_dir}" ]]; then
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${TRAIN_ENV}/bin/python" \
      -m training.simul_uniss.stage7a.export_policy_model \
      --checkpoint "${checkpoint}" --output-dir "${export_dir}" \
      > "${RUN_DIR}/logs/export_${label}.log" 2>&1
  fi
  "${ROOT}/post_training/evaluate_checkpoint_2gpu.sh" \
    "${checkpoint}" "${DEV_SAMPLES}" "${RUN_DIR}/${label}/dev.json" \
    "${E0_GPUS}" "$((30530 + index * 2))" 0 \
    > "${RUN_DIR}/logs/eval_${label}_dev.log" 2>&1
  "${ROOT}/post_training/evaluate_checkpoint_2gpu.sh" \
    "${checkpoint}" "${TEST_SAMPLES}" "${RUN_DIR}/${label}/test.json" \
    "${E0_GPUS}" "$((30531 + index * 2))" 0 \
    > "${RUN_DIR}/logs/eval_${label}_test.log" 2>&1
  index=$((index + 1))
done

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${TRAIN_ENV}/bin/python" \
  "${ROOT}/post_training/report.py" \
  --run-dir "${RUN_DIR}" \
  --e0-dir "${EVAL_ROOT}/e0_baselines_full" \
  --output-json "${RUN_DIR}/comparison.json" \
  --report "${RUN_DIR}/stage7a_action_policy_report.md"
touch "${RUN_DIR}/COMPLETE"
cleanup
trap - EXIT
echo "RUN_DIR=${RUN_DIR}"
