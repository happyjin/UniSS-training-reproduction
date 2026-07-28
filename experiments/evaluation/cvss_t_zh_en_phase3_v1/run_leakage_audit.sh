#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${CVSS_ROOT}/canonical_16k/cvss_t_zh_en_test/manifests/cvss_t_zh_en_test_pairs.jsonl}"
TRAIN_GLOB="${TRAIN_GLOB:-data/raw/UniST/train-*.parquet}"
OUTPUT="${OUTPUT:-${CVSS_ROOT}/audits/cvss_t_zh_en_vs_unist198_text_leakage.json}"
WORKERS="${WORKERS:-32}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}"
"${ENV_ROOT}/bin/python" -m evaluation.cvss_t.leakage_audit \
  --pair-manifest "${PAIR_MANIFEST}" \
  --train-glob "${TRAIN_GLOB}" \
  --output "${OUTPUT}" \
  --workers "${WORKERS}"
