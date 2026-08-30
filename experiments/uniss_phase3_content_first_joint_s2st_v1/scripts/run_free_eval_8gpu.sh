#!/usr/bin/env bash
set -euo pipefail

# Isolated wrapper around the established fixed-15 free-running evaluator.  It
# only consumes the completed v1 checkpoint and writes a new output root.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

ITERATION="${ITERATION:-717}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/${RUN_NAME}/iter_$(printf '%07d' "$((10#${ITERATION}))")}" 
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}"
EVAL_TAG="${EVAL_TAG:-free_running_content_first_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/checkpoint_evaluation/${EVAL_TAG}/iter_$(printf '%07d' "$((10#${ITERATION}))")}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/runtime_exports/iter_$(printf '%07d' "$((10#${ITERATION}))")}"

[[ -f "${CHECKPOINT}/.metadata" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${EXPORT_ROOT}/manifest.json" ]] || { echo "missing runtime export: ${EXPORT_ROOT}" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite: ${OUTPUT_ROOT}" >&2; exit 3; }

export ITERATION RUN_NAME CHECKPOINT BASE_MODEL EVAL_TAG OUTPUT_ROOT EXPORT_ROOT
exec bash "${REPO_ROOT}/experiments/uniss_phase3_event_rollout_joint_pilot15_v2/evaluation/run_checkpoint_evaluation_8gpu.sh"
