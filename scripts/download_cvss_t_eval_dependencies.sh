#!/usr/bin/env bash
set -uo pipefail

CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
COVOST_URL="${COVOST_URL:-https://dl.fbaipublicfiles.com/covost/covost_v2.zh-CN_en.tsv.tar.gz}"
COVOST_DIR="${CVSS_ROOT}/metadata/covost_v2.zh-CN_en"
COVOST_ARCHIVE="${COVOST_DIR}/covost_v2.zh-CN_en.tsv.tar.gz"
COVOST_DONE="${COVOST_DIR}/.download_complete"
LOG_DIR="${CVSS_ROOT}/logs"

mkdir -p "${COVOST_DIR}" "${LOG_DIR}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

while [[ ! -f "${COVOST_DONE}" ]]; do
  echo "[$(timestamp)] downloading CoVoST 2 zh-CN->en metadata"
  if curl --fail --location --show-error \
      --connect-timeout 30 \
      --retry 0 \
      --continue-at - \
      --output "${COVOST_ARCHIVE}" \
      "${COVOST_URL}"; then
    if tar -tzf "${COVOST_ARCHIVE}" >/dev/null; then
      tar -xzf "${COVOST_ARCHIVE}" -C "${COVOST_DIR}"
      sha256sum "${COVOST_ARCHIVE}" >"${COVOST_ARCHIVE}.sha256"
      touch "${COVOST_DONE}"
      echo "[$(timestamp)] CoVoST 2 metadata download and archive verification completed"
      break
    fi
    echo "[$(timestamp)] archive verification failed"
  else
    status=$?
    echo "[$(timestamp)] curl failed with status ${status}"
  fi
  echo "[$(timestamp)] retrying in ${RETRY_SECONDS} seconds"
  sleep "${RETRY_SECONDS}"
done
