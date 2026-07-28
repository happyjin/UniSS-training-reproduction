#!/usr/bin/env bash
set -uo pipefail

CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
SOURCE_REPO="DynamicSuperb/SpeechTranslation_CoVoST2-zh-CN_en"
SOURCE_REVISION="d70e7b6be35124779d34e57974fa8a3a480d96fe"
SOURCE_FILE="data/test-00000-of-00001.parquet"
EXPECTED_SHA256="8c510c417bf45e91b52702f6cf0bca0a5fbe4f9293406f6bfb5808268ca4683b"
SOURCE_URL="https://huggingface.co/datasets/${SOURCE_REPO}/resolve/${SOURCE_REVISION}/${SOURCE_FILE}?download=true"
OUTPUT_DIR="${CVSS_ROOT}/source/common_voice_v4_zh-CN_test_mirror"
OUTPUT_FILE="${OUTPUT_DIR}/covost2_zh-CN_en_test.parquet"
DONE_FILE="${OUTPUT_DIR}/.download_complete"

mkdir -p "${OUTPUT_DIR}" "${CVSS_ROOT}/logs"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

while [[ ! -f "${DONE_FILE}" ]]; do
  echo "[$(timestamp)] downloading ${SOURCE_REPO}@${SOURCE_REVISION}/${SOURCE_FILE}"
  if curl --fail --location --show-error \
      --connect-timeout 30 \
      --speed-time 30 \
      --speed-limit 1024 \
      --retry 0 \
      --continue-at - \
      --output "${OUTPUT_FILE}" \
      "${SOURCE_URL}"; then
    actual_sha256="$(sha256sum "${OUTPUT_FILE}" | awk '{print $1}')"
    if [[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]]; then
      printf '%s  %s\n' "${actual_sha256}" "$(basename "${OUTPUT_FILE}")" >"${OUTPUT_FILE}.sha256"
      {
        printf 'source_repo=%s\n' "${SOURCE_REPO}"
        printf 'source_revision=%s\n' "${SOURCE_REVISION}"
        printf 'source_file=%s\n' "${SOURCE_FILE}"
        printf 'source_url=%s\n' "${SOURCE_URL}"
        printf 'sha256=%s\n' "${actual_sha256}"
        printf 'completed_at=%s\n' "$(timestamp)"
      } >"${OUTPUT_DIR}/SOURCE.txt"
      touch "${DONE_FILE}"
      echo "[$(timestamp)] source test parquet download and SHA256 verification completed"
      break
    fi
    echo "[$(timestamp)] SHA256 mismatch: expected=${EXPECTED_SHA256} actual=${actual_sha256}"
  else
    status=$?
    echo "[$(timestamp)] curl failed with status ${status}"
  fi
  echo "[$(timestamp)] retrying in ${RETRY_SECONDS} seconds"
  sleep "${RETRY_SECONDS}"
done
