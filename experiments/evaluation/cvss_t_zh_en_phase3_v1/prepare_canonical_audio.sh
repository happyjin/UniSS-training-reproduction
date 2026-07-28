#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${CVSS_ROOT}/manifests/cvss_t_zh_en_v1/cvss_t_zh_en_test_pending.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CVSS_ROOT}/canonical_16k/cvss_t_zh_en_test}"
WORKERS="${WORKERS:-32}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

args=(
  --input-manifest "${INPUT_MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --workers "${WORKERS}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

"${ENV_ROOT}/bin/python" -m evaluation.cvss_t.canonicalize "${args[@]}"
