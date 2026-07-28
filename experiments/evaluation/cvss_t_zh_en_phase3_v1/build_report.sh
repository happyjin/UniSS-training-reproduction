#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ZH_EN_OUTPUT EN_ZH_OUTPUT REPORT_DIR" >&2
  exit 2
fi

ZH_EN_OUTPUT="$1"
EN_ZH_OUTPUT="$2"
REPORT_DIR="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval}"
LEAKAGE_AUDIT="${LEAKAGE_AUDIT:-/opt/dlami/nvme/jasonleeeli/CVSS/audits/cvss_t_zh_en_vs_unist198_text_leakage.json}"
EXPECTED_PAIRS="${EXPECTED_PAIRS:-4897}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

"${ENV_ROOT}/bin/python" -m evaluation.cvss_t.report \
  --run "${ZH_EN_OUTPUT}" "${EN_ZH_OUTPUT}" \
  --output-dir "${REPORT_DIR}" \
  --expected-pairs "${EXPECTED_PAIRS}" \
  --leakage-audit "${LEAKAGE_AUDIT}"

echo "CVSS-T report: ${REPORT_DIR}/cvss_t_phase3_table1_report.md"
