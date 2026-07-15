#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-cmots/UniSS}"
REVISION="${REVISION:-main}"
LOCAL_DIR="${LOCAL_DIR:-${REPO_ROOT}/pretrained_models/UniSS}"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/${REVISION}"

FILES=(
  ".gitattributes"
  "README.md"
  "bicodec/.gitattributes"
  "bicodec/BiCodec/config.yaml"
  "bicodec/BiCodec/model.safetensors"
  "bicodec/README.md"
  "bicodec/config.yaml"
  "bicodec/wav2vec2-large-xlsr-53/README.md"
  "bicodec/wav2vec2-large-xlsr-53/config.json"
  "bicodec/wav2vec2-large-xlsr-53/preprocessor_config.json"
  "bicodec/wav2vec2-large-xlsr-53/pytorch_model.bin"
  "config.json"
  "generation_config.json"
  "glm4_tokenizer/.gitattributes"
  "glm4_tokenizer/LICENSE"
  "glm4_tokenizer/README.md"
  "glm4_tokenizer/config.json"
  "glm4_tokenizer/model.safetensors"
  "glm4_tokenizer/preprocessor_config.json"
  "merges.txt"
  "model.safetensors"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

mkdir -p "${LOCAL_DIR}"

for file in "${FILES[@]}"; do
  out="${LOCAL_DIR}/${file}"
  part="${out}.part"
  url="${BASE_URL}/${file}"
  mkdir -p "$(dirname "${out}")"
  if [[ -s "${out}" && ! -e "${part}" ]]; then
    echo "SKIP ${file}"
    continue
  fi
  echo "DOWNLOAD ${file}"
  run_cmd curl -L --fail --retry 20 --retry-all-errors --connect-timeout 30 \
    --speed-limit 1 --speed-time 120 -C - -o "${part}" "${url}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mv "${part}" "${out}"
  fi
done
