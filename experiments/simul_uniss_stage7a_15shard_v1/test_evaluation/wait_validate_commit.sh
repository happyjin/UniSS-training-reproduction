#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

REPORT="${TEST_EVAL_ROOT}/stage7a_four_way_full_test_report.md"
COMPARISON="${TEST_EVAL_ROOT}/comparison.json"

while [[ ! -f "${TEST_EVAL_ROOT}/COMPLETE" ]]; do
  sleep 30
done

for label in e0_stage6 e1_continued_sft e2_grpo_g4 e3_grpo_g8; do
  run_dir="${TEST_EVAL_ROOT}/${label}/${FULL_RUN_ID}"
  for required in \
    "${run_dir}/COMPLETE" \
    "${run_dir}/aggregate_metrics.json" \
    "${run_dir}/latency_batch1/COMPLETE" \
    "${run_dir}/latency_batch1/aggregate_metrics.json"; do
    [[ -s "${required}" || -f "${required}" ]] || {
      echo "Missing required result: ${required}" >&2
      exit 1
    }
  done
done
[[ -s "${REPORT}" && -s "${COMPARISON}" ]] || {
  echo "Final report or comparison JSON is missing" >&2
  exit 1
}

"${EVAL_ENV}/bin/python" - "${COMPARISON}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {"e0_stage6", "e1_continued_sft", "e2_grpo_g4", "e3_grpo_g8"}
actual = set(payload.get("results", {}))
if actual != expected:
    raise SystemExit(f"comparison labels mismatch: expected={expected}, actual={actual}")
if not payload.get("analysis", {}).get("decision"):
    raise SystemExit("comparison analysis decision is missing")
PY

report_relative="${REPORT#"${REPO_ROOT}/"}"
comparison_relative="${COMPARISON#"${REPO_ROOT}/"}"
if [[ "${report_relative}" == "${REPORT}" || "${comparison_relative}" == "${COMPARISON}" ]]; then
  echo "Result files are outside the repository" >&2
  exit 1
fi

git -C "${REPO_ROOT}" add -f -- "${report_relative}" "${comparison_relative}"
if git -C "${REPO_ROOT}" diff --cached --quiet -- "${report_relative}" "${comparison_relative}"; then
  echo "Final report is already committed"
  exit 0
fi
git -C "${REPO_ROOT}" commit -m "Add Stage7A four-way full test results" -- \
  "${report_relative}" "${comparison_relative}"
git -C "${REPO_ROOT}" push private master:main
echo "REPORT=${REPORT}"
echo "COMPARISON=${COMPARISON}"

