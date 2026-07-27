#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

if [[ -f "${TEST_SCHEDULES}" && -f "${TEST_PROCESSED_DIR}/samples.jsonl" && -f "${TEST_PROCESSED_DIR}/manifest.json" ]]; then
  exec "${SCRIPT_DIR}/verify_test_inputs.sh"
fi
if [[ -e "${TEST_PROCESSED_DIR}" ]]; then
  echo "Refusing to overwrite incomplete test schedule directory: ${TEST_PROCESSED_DIR}" >&2
  exit 1
fi

PARTIAL="${TEST_PROCESSED_DIR}.partial.$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "${TEST_PROCESSED_DIR}")"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${TRAIN_ENV}/bin/python" \
  -m training.simul_uniss.prepare_data \
  --input "${TEST_PARQUET}" \
  --output-dir "${PARTIAL}" \
  --tokenizer "${SPEECH_TOKENIZER}" \
  --chunk-ms 640 \
  --wait-k-chunks 2 \
  --max-phrase-tokens 16 \
  --progress-interval 2000
mv "${PARTIAL}" "${TEST_PROCESSED_DIR}"
exec "${SCRIPT_DIR}/verify_test_inputs.sh"
