#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

ITERATION="${ITERATION:-717}"
printf -v ITER_TAG 'iter_%07d' "$((10#${ITERATION}))"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}/${ITER_TAG}}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/runtime_exports/${ITER_TAG}}"

[[ -f "${CHECKPOINT}/.metadata" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "missing base HF model: ${BASE_MODEL}" >&2; exit 2; }

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${USER_ROOT}/conda_envs/uniss-offline-demo/bin/python" \
  -m web_demo.true_subsecond_pilot15_streaming_v1.checkpoint_export \
  --checkpoint "${CHECKPOINT}" --base-model "${BASE_MODEL}" --output "${OUTPUT}"
