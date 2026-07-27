#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

RUN_ID="${1:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || { echo "Run exists: ${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}"

"${SCRIPT_DIR}/prepare_manifests.sh"
"${SCRIPT_DIR}/export_exact_stage3.sh"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 "${ENV_ROOT}/bin/python" \
  -m evaluation.simultaneous_streaming.stage3_action_eval \
  --model "${HF_EXPORT}" \
  --samples "${DEV_SAMPLES}" \
  --schedules "${DEV_SCHEDULES}" \
  --output-dir "${RUN_DIR}/dev" \
  --split dev \
  --rank 0 --world-size 1 --local-rank 0 \
  --dtype "${DTYPE}" \
  --attention-implementation "${ATTENTION_IMPLEMENTATION}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS}" \
  --max-batch-size "${MAX_BATCH_SIZE}" \
  --logit-event-batch "${LOGIT_EVENT_BATCH}" \
  --limit-records 128 \
  --progress-interval 1

echo "SMOKE_RUN_DIR=${RUN_DIR}"

