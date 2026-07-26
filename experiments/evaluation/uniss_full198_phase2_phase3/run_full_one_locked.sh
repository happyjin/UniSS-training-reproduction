#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 STAGE HF_CHECKPOINT MANIFEST OUTPUT_ROOT GPU_LIST" >&2
  exit 2
fi

STAGE="$1"
HF_CHECKPOINT="$2"
MANIFEST="$3"
OUTPUT_ROOT="$4"
GPU_LIST="$5"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
LOCK_FILE="${OUTPUT_ROOT}/.full_evaluation.lock"
COMPLETE_FILE="${OUTPUT_ROOT}/FULL_EVALUATION_COMPLETE"
exec 9>"${LOCK_FILE}"
flock 9

if [[ -f "${COMPLETE_FILE}" ]]; then
  echo "${STAGE} full evaluation already complete: ${OUTPUT_ROOT}"
  exit 0
fi

EVAL_GPU_LIST="${GPU_LIST}" \
ENV_ROOT="${ENV_ROOT}" \
RESUME=1 \
ALLOW_GENERATED_FAILURES="${ALLOW_GENERATED_FAILURES:-1}" \
  "${EVAL_ROOT}/run_vllm_eval.sh" "${STAGE}" "${HF_CHECKPOINT}" "${MANIFEST}" "${OUTPUT_ROOT}"

EVAL_GPU_LIST="${GPU_LIST}" \
ENV_ROOT="${ENV_ROOT}" \
  "${EVAL_ROOT}/run_objective_metrics.sh" "${OUTPUT_ROOT}"

touch "${COMPLETE_FILE}"
echo "${STAGE} full generation and objective metrics complete: ${OUTPUT_ROOT}"
