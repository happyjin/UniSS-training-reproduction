#!/usr/bin/env bash
set -uo pipefail

CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
SOURCE_REPO="fixie-ai/covost2"
SOURCE_REVISION="17c8c81e331e7a6929118121771a58c7ef7331d8"
OUTPUT_DIR="${CVSS_ROOT}/source/common_voice_v4_zh-CN_test_fixie_parquet"

FILES=(
  "zh-CN_en/test-00000-of-00008.parquet"
  "zh-CN_en/test-00001-of-00008.parquet"
  "zh-CN_en/test-00002-of-00008.parquet"
  "zh-CN_en/test-00003-of-00008.parquet"
  "zh-CN_en/test-00004-of-00008.parquet"
  "zh-CN_en/test-00005-of-00008.parquet"
  "zh-CN_en/test-00006-of-00008.parquet"
  "zh-CN_en/test-00007-of-00008.parquet"
)

SHA256S=(
  "cdd3dc27f69cb903bd8da44db456387c14190bc830fe408487c34a1916cbe701"
  "855a8a4fa87b6e21c31a58f70a9df9fca83772aebe1bf287990ffb19a75a8d79"
  "186cfe887fa85cdb4a7195c6ee226b7d4872e07c6744532878c6015610197c16"
  "d4892b84dae79626a6455ab03a551cf46ca68a34e6a0e1ef69734b5f56bc0e39"
  "f43de411271b55392e8073cfe06433540df5c75c6844063246914d9f3c318bb6"
  "7bafa5b921a4eabeb6da0e40926cc6a8c20e77d369c167f9be41832ebc1ce963"
  "16f9fe803f48e05b3db93819457a403f7c5796779e36ca3e3354afefb1d7cc1c"
  "3224b06b667ecbc71ab5e8cbb071b255004c405e55f67e912d56b4ae58598f89"
)

mkdir -p "${OUTPUT_DIR}" "${CVSS_ROOT}/logs"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

download_one() {
  local index="$1"
  local source_file="${FILES[${index}]}"
  local expected_sha256="${SHA256S[${index}]}"
  local basename
  local output_file
  local done_file
  local source_url
  basename="$(basename "${source_file}")"
  output_file="${OUTPUT_DIR}/${basename}"
  done_file="${output_file}.complete"
  source_url="https://huggingface.co/datasets/${SOURCE_REPO}/resolve/${SOURCE_REVISION}/${source_file}?download=true"

  while [[ ! -f "${done_file}" ]]; do
    echo "[$(timestamp)] shard=${index} downloading ${source_file}"
    if curl --fail --location --show-error \
        --connect-timeout 30 \
        --speed-time 30 \
        --speed-limit 1024 \
        --retry 0 \
        --continue-at - \
        --output "${output_file}" \
        "${source_url}"; then
      actual_sha256="$(sha256sum "${output_file}" | awk '{print $1}')"
      if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
        printf '%s  %s\n' "${actual_sha256}" "${basename}" >"${output_file}.sha256"
        touch "${done_file}"
        echo "[$(timestamp)] shard=${index} verified ${basename}"
        break
      fi
      echo "[$(timestamp)] shard=${index} SHA256 mismatch expected=${expected_sha256} actual=${actual_sha256}"
    else
      status=$?
      echo "[$(timestamp)] shard=${index} curl failed status=${status}"
    fi
    echo "[$(timestamp)] shard=${index} retrying in ${RETRY_SECONDS} seconds"
    sleep "${RETRY_SECONDS}"
  done
}

pids=()
for index in "${!FILES[@]}"; do
  download_one "${index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

{
  printf 'source_repo=%s\n' "${SOURCE_REPO}"
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'config=zh-CN_en\n'
  printf 'split=test\n'
  printf 'completed_at=%s\n' "$(timestamp)"
} >"${OUTPUT_DIR}/SOURCE.txt"
touch "${OUTPUT_DIR}/.download_complete"
echo "[$(timestamp)] all fixie-ai CoVoST2 zh-CN_en test shards completed"
