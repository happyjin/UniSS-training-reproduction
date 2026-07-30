#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
REPO_ROOT="${REPO_ROOT:-${USER_ROOT}/projects/UniSS}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh}"
MODE="${E2_MODE:-quick}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29712}"

CHECKPOINT="${E2_CHECKPOINT:-${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v1/stage_b_pilot_15shard_vectorized_v2/best.pt}"
MANIFEST="${E2_MANIFEST:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_e2_v1/unist_dev/stage_a_source/manifest.jsonl}"

case "${MODE}" in
  quick)
    LIMIT_RECORDS="${LIMIT_RECORDS:-512}"
    CONFIGS="${CONFIGS:-160:80}"
    DEFAULT_RUN="unist_dev_quick_512_native_v1"
    ;;
  full)
    LIMIT_RECORDS="${LIMIT_RECORDS:-}"
    CONFIGS="${CONFIGS:-160:80,320:80,160:0,320:0}"
    DEFAULT_RUN="unist_dev_full_7965_geometry_scan_v1"
    ;;
  *)
    echo "E2_MODE must be quick or full" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/simul_uniss_subsecond_e2_v1/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/simul_uniss_subsecond_e2_v1}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

[[ -f "${ACTIVATE_SCRIPT}" ]] || { echo "Missing activation script: ${ACTIVATE_SCRIPT}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --configs "${CONFIGS}"
  --wait-k "2,3"
  --device cuda
)
if [[ -n "${LIMIT_RECORDS}" ]]; then
  ARGS+=(--limit-records "${LIMIT_RECORDS}")
fi

{
  echo "E2_MODE=${MODE}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "CONFIGS=${CONFIGS}"
  echo "LIMIT_RECORDS=${LIMIT_RECORDS:-all}"
  "${CONDA_PREFIX}/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    --master-port="${MASTER_PORT}" \
    -m training.simul_uniss.subsecond_v1.evaluate_e2 \
    "${ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"

echo "REPORT=${OUTPUT_DIR}/REPORT.md"
