#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"
: "${TASK_POOL_RUN_ID:?set the immutable formal task-pool run ID}"

TASK_POOL_WORKERS=${TASK_POOL_WORKERS:-64}
if (( TASK_POOL_WORKERS < 1 || TASK_POOL_WORKERS > $(nproc) )); then
  echo "TASK_POOL_WORKERS must be in [1,nproc]" >&2
  exit 2
fi

export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

SUMMARY_ROOT=${REPORT_ROOT}/task_pools/${TASK_POOL_RUN_ID}
LOG_DIR=${LOG_ROOT}/task_pools/${TASK_POOL_RUN_ID}
if [[ -e "${SUMMARY_ROOT}" || -e "${LOG_DIR}" ]]; then
  echo "refusing to overwrite formal task-pool run: ${TASK_POOL_RUN_ID}" >&2
  exit 2
fi
mkdir -p "${SUMMARY_ROOT}" "${LOG_DIR}"

for split in train valid; do
  gold=${PROCESSED_ROOT}/source_events/${split}_gold_trajectories.jsonl
  rollout_root=${PROCESSED_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_${split}
  rollout=${rollout_root}/${split}_v1_rollouts.jsonl
  rollout_audit=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_${split}/AUDIT.json
  strata=${rollout_root}/${split}_quality_strata.jsonl
  quality_gate=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_${split}/QUALITY_GATE.json
  output=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_${split}
  log=${LOG_DIR}/${split}.log
  command_log=${LOG_DIR}/${split}.command

  for path in "${gold}" "${rollout}" "${rollout}.offsets.bin" "${rollout_audit}" \
    "${strata}" "${strata}.offsets.bin" "${quality_gate}"; do
    [[ -f "${path}" ]] || { echo "missing formal task-pool input: ${path}" >&2; exit 3; }
  done
  jq -e '.status == "passed"' "${rollout_audit}" >/dev/null || {
    echo "V1 rollout audit did not pass: ${rollout_audit}" >&2
    exit 3
  }
  jq -e '.status == "passed"' "${quality_gate}" >/dev/null || {
    echo "V1 rollout quality gate did not pass: ${quality_gate}" >&2
    exit 3
  }
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite formal task-pool output: ${output}" >&2
    exit 3
  }

  cmd=(
    "${PYTHON_BIN}"
    -m experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools
    --gold "${gold}"
    --rollouts "${rollout}"
    --strata-manifest "${strata}"
    --quality-gate "${quality_gate}"
    --tokenizer "${V1_HF_MODEL}"
    --output-root "${output}"
    --split "${split}"
    --workers "${TASK_POOL_WORKERS}"
    --seq-length 18000
  )
  printf '%q ' "${cmd[@]}" > "${command_log}"
  printf '\n' >> "${command_log}"
  "${cmd[@]}" > "${log}" 2>&1

  build_report=${output}/BUILD_COMPLETE.json
  jq -e --arg split "${split}" \
    '.status == "passed" and .split == $split and .seq_length == 18000' \
    "${build_report}" >/dev/null || {
      echo "formal task-pool build did not pass: ${build_report}" >&2
      exit 4
    }
  cp -- "${build_report}" "${SUMMARY_ROOT}/${split}_BUILD_COMPLETE.json"
done

echo "task_pool_run_id=${TASK_POOL_RUN_ID}"
echo "train_report=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_train/BUILD_COMPLETE.json"
echo "valid_report=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_valid/BUILD_COMPLETE.json"
echo "summary_root=${SUMMARY_ROOT}"
