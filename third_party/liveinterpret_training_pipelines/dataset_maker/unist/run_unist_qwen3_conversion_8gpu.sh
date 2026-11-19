#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python}"

INPUT_DIR="${INPUT_DIR:-${WORKSPACE_ROOT}/data/unist/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/data/unist/converted/unist_qwen3_s2s_v2}"
SPARK_REPO="${SPARK_REPO:-${WORKSPACE_ROOT}/projects/Spark-TTS}"
BICODEC_MODEL="${BICODEC_MODEL:-${WORKSPACE_ROOT}/projects/Spark-TTS-0.5B/BiCodec}"
QWEN_TOKENIZER="${QWEN_TOKENIZER:-${REPO_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz}"
PROCESS_VERSION="${PROCESS_VERSION:-unist_bicodec_qwen3_v2}"
WORLD_SIZE="${WORLD_SIZE:-8}"
MAX_BATCH_ROWS="${MAX_BATCH_ROWS:-64}"
MAX_BICODEC_PADDING_TOKENS="${MAX_BICODEC_PADDING_TOKENS:-0}"
BICODEC_GROUP_SCOPE="${BICODEC_GROUP_SCOPE:-block}"
BICODEC_GROUP_ROWS="${BICODEC_GROUP_ROWS:-64}"
QWEN_ENCODE_PATH="${QWEN_ENCODE_PATH:-direct}"
MAX_PADDED_SECONDS="${MAX_PADDED_SECONDS:-512}"
READ_BATCH_ROWS="${READ_BATCH_ROWS:-8192}"
MAX_TAR_BYTES="${MAX_TAR_BYTES:-4000000000}"
AUDIO_FORMAT="${AUDIO_FORMAT:-flac}"
LIMIT="${LIMIT:-0}"
MAX_FILES="${MAX_FILES:-0}"

mkdir -p "${OUTPUT_ROOT}/logs"

common_args=(
  --input-dir "${INPUT_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --spark-repo "${SPARK_REPO}"
  --bicodec-model "${BICODEC_MODEL}"
  --qwen-tokenizer "${QWEN_TOKENIZER}"
  --process-version "${PROCESS_VERSION}"
  --world-size "${WORLD_SIZE}"
  --max-batch-rows "${MAX_BATCH_ROWS}"
  --max-bicodec-padding-tokens "${MAX_BICODEC_PADDING_TOKENS}"
  --bicodec-group-scope "${BICODEC_GROUP_SCOPE}"
  --bicodec-group-rows "${BICODEC_GROUP_ROWS}"
  --qwen-encode-path "${QWEN_ENCODE_PATH}"
  --max-padded-seconds "${MAX_PADDED_SECONDS}"
  --read-batch-rows "${READ_BATCH_ROWS}"
  --max-tar-bytes "${MAX_TAR_BYTES}"
  --audio-format "${AUDIO_FORMAT}"
  --limit "${LIMIT}"
  --max-files "${MAX_FILES}"
)

pids=()
cleanup_children() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM

for ((rank=0; rank<WORLD_SIZE; rank++)); do
  log_path="${OUTPUT_ROOT}/logs/rank$(printf '%02d' "${rank}").log"
  echo "Starting rank ${rank}; log=${log_path}"
  printf '\n=== resume/start %s rank=%s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rank}" \
    >>"${log_path}"
  CUDA_VISIBLE_DEVICES="${rank}" PYTHONPATH="${SPARK_REPO}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_unist_bicodec_to_qwen3.py" \
      "${common_args[@]}" --rank "${rank}" --device cuda:0 \
      >>"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one conversion rank failed; outputs remain resumable." >&2
  exit "${status}"
fi

if [[ "${LIMIT}" -eq 0 && "${MAX_FILES}" -eq 0 && "${WORLD_SIZE}" -eq 8 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_unist_bicodec_to_qwen3.py" \
    "${common_args[@]}" --rank 0 --finalize-only
else
  echo "Smoke/partial run complete; finalization intentionally skipped."
fi
