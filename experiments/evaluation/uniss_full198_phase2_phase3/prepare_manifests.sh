#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/manifests}"
SEED="${SEED:-20260726}"

mkdir -p "${OUTPUT_DIR}"

"${ENV_ROOT}/bin/python" -m evaluation.unist_manifest \
  --input "${REPO_ROOT}/data/raw/UniST/dev-00000.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --split-name dev \
  --seed "${SEED}" \
  --smoke-count 3 \
  --listen-count 50 \
  --repo-root "${REPO_ROOT}"

"${ENV_ROOT}/bin/python" -m evaluation.unist_manifest \
  --input "${REPO_ROOT}/data/raw/UniST/test-00000.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --split-name test \
  --seed "${SEED}" \
  --smoke-count 3 \
  --listen-count 50 \
  --repo-root "${REPO_ROOT}"
