#!/usr/bin/env bash
# CoVoST2 zh-CN_en *validation* shards, from the same pinned mirror revision
# the test split was taken from.
#
# Why this split: training needs a dev curve, and the only CVSS-T material on
# this box is the test split -- which is also the number the project reports,
# so watching it during training would contaminate it. The validation split is
# disjoint from test and comes from the same source, so a dev curve on it is
# comparable to the test number without being the test number.
#
# Deliberately a sibling of download_cvss_t_zh_source_test_fixie.sh rather than
# a parameterisation of it: that script is referenced by completed runs and is
# not being touched.
set -uo pipefail

CVSS_ROOT="${CVSS_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
SOURCE_REPO="fixie-ai/covost2"
SOURCE_REVISION="17c8c81e331e7a6929118121771a58c7ef7331d8"
OUTPUT_DIR="${CVSS_ROOT}/source/common_voice_v4_zh-CN_dev_fixie_parquet"

SHARDS=(
  "2f2b150948a01423f8d90f5b7732dce5997c99285be6a965fbc5b2c92cd84c57"
  "fe5dc4c0f7859d7c009d6b04000501e467b4d2c986e6771d5179707fc4e3d774"
  "6181c63d6f1d2b9038e939356ded8803b17846441838ab51b509a81a154ce4ab"
  "9f3622d27f538f3edca4487226ece1ba1e59397f77a6cb984d8bc7d0989ad1bb"
  "aa4c00cef6bb4c1894f2bff12c402affbd1d805aebbb2f5d8b92ba3467090f33"
  "7ba6805a55deca4976971925f2721509e5e49eaa014661db21f04504caadc8ff"
  "fe0d50f9c12467b485265af9b05c9e840bc109b5ed750d502cc6639510eabe6f"
  "422404d5cc7d7537cb444c65a4cef455b50af56c50dc544ac0b401526ae515e1"
)

mkdir -p "${OUTPUT_DIR}"
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

download_one() {
  local index="$1"
  local name
  name="$(printf 'validation-%05d-of-00008.parquet' "${index}")"
  local expected="${SHARDS[${index}]}"
  local output="${OUTPUT_DIR}/${name}"
  local done_file="${output}.complete"
  local url="https://huggingface.co/datasets/${SOURCE_REPO}/resolve/${SOURCE_REVISION}/zh-CN_en/${name}?download=true"

  while [[ ! -f "${done_file}" ]]; do
    echo "[$(timestamp)] shard=${index} downloading ${name}"
    if curl --fail --location --show-error \
        --connect-timeout 30 --speed-time 30 --speed-limit 1024 \
        --retry 0 --continue-at - --output "${output}" "${url}"; then
      local actual
      actual="$(sha256sum "${output}" | awk '{print $1}')"
      if [[ "${actual}" == "${expected}" ]]; then
        echo "${actual}" > "${output}.sha256"
        touch "${done_file}"
        echo "[$(timestamp)] shard=${index} ok"
      else
        # A digest mismatch means a corrupt or resumed-wrong body; the partial
        # file has to go or --continue-at will append to the damage.
        echo "[$(timestamp)] shard=${index} sha256 mismatch, refetching"
        rm -f "${output}"
      fi
    else
      echo "[$(timestamp)] shard=${index} transfer failed, retrying in ${RETRY_SECONDS}s"
      sleep "${RETRY_SECONDS}"
    fi
  done
}

for i in 0 1 2 3 4 5 6 7; do download_one "${i}" & done
wait

cat > "${OUTPUT_DIR}/SOURCE.txt" <<SRC
source_repo=${SOURCE_REPO}
source_revision=${SOURCE_REVISION}
source_config=zh-CN_en
source_split=validation
note=Dev split for training-time loss curves; disjoint from the reported test split.
SRC
echo "[$(timestamp)] all shards complete -> ${OUTPUT_DIR}"
