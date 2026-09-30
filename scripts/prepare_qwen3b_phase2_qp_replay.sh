#!/usr/bin/env bash
# Build 3B Phase2 train data: Quality+Performance (existing Phase3 JSONL)
# mixed 2:1 with Phase1 replay generated from the local UniST parquets.
# Direct S2ST is not included. Does not touch GPUs or running Phase1 training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/neuhao}"
PYTHON="${PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
RAW_DIR="${RAW_DIR:-${REPO_ROOT}/data/raw/UniST}"
QP_DIR="${QP_DIR:-${REPO_ROOT}/data/processed/phase3_unist198_sharded}"
PHASE1_DIR="${PHASE1_DIR:-${REPO_ROOT}/data/processed/phase1_unist198_sharded}"
MIX_DIR="${MIX_DIR:-${REPO_ROOT}/data/processed/phase2_qwen3b_qp_replay_sharded}"
PACKED="${PACKED:-${REPO_ROOT}/data/megatron/phase2_qwen3b_qp_replay/packed_train.jsonl}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/runs/qwen3b_phase2_qp_replay_preprocess/logs}"
WORKERS="${WORKERS:-8}"
PACK_WORKERS="${PACK_WORKERS:-8}"
KEEP_PHASE1_JSONL="${KEEP_PHASE1_JSONL:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PHASE1_DIR}" "${MIX_DIR}" "${LOG_DIR}" "$(dirname "${PACKED}")"

[[ -x "${PYTHON}" ]] || { echo "Missing python: ${PYTHON}" >&2; exit 1; }
[[ -d "${TOKENIZER}" ]] || { echo "Missing tokenizer: ${TOKENIZER}" >&2; exit 1; }

shard_rows() {
  "${PYTHON}" -c 'import pyarrow.parquet as pq,sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "$1"
}

build_one() {
  local index_raw="$1"
  local index
  index="$(printf '%05d' "${index_raw}")"
  local raw="${RAW_DIR}/train-${index}.parquet"
  local qp="${QP_DIR}/train-${index}.jsonl"
  local phase1="${PHASE1_DIR}/train-${index}.jsonl"
  local mix="${MIX_DIR}/train-${index}.jsonl"
  local log="${LOG_DIR}/train-${index}.log"
  [[ -f "${raw}" ]] || { echo "Missing parquet: ${raw}" >&2; exit 1; }
  [[ -f "${qp}" ]] || { echo "Missing QP jsonl: ${qp}" >&2; exit 1; }

  local rows qp_expected phase1_expected mix_expected
  rows="$(shard_rows "${raw}")"
  qp_expected=$((rows * 2))
  phase1_expected=$((rows * 4))
  mix_expected=$((rows * 3))

  {
    echo "[$(date -u +%FT%TZ)] shard ${index} rows=${rows}"
    local qp_lines
    qp_lines="$(wc -l < "${qp}")"
    [[ "${qp_lines}" == "${qp_expected}" ]] || {
      echo "QP count mismatch ${qp}: expected=${qp_expected} actual=${qp_lines}" >&2
      exit 1
    }

    if [[ ! -s "${phase1}" ]] || [[ "$(wc -l < "${phase1}")" != "${phase1_expected}" ]]; then
      local phase1_tmp="${phase1}.tmp.$$"
      rm -f "${phase1_tmp}"
      "${PYTHON}" "${REPO_ROOT}/training/prepare_phase1_alignment.py" \
        --input "${raw}" \
        --tokenizer "${TOKENIZER}" \
        --tasks asr s2tt tts \
        --include-mt-proxy \
        --output "${phase1_tmp}"
      mv "${phase1_tmp}" "${phase1}"
    else
      echo "reuse ${phase1}"
    fi

    if [[ ! -s "${mix}" ]] || [[ "$(wc -l < "${mix}")" != "${mix_expected}" ]]; then
      "${PYTHON}" "${REPO_ROOT}/training/mix_qp_phase1_replay.py" \
        --qp "${qp}" \
        --phase1 "${phase1}" \
        --output "${mix}" \
        --expected-total "${mix_expected}"
    else
      echo "reuse ${mix}"
    fi

    if [[ "${KEEP_PHASE1_JSONL}" != "1" ]]; then
      rm -f "${phase1}"
    fi
    echo "[$(date -u +%FT%TZ)] shard ${index} mixed ${mix_expected}"
  } >"${log}" 2>&1
}

export -f build_one shard_rows
export REPO_ROOT PYTHON TOKENIZER RAW_DIR QP_DIR PHASE1_DIR MIX_DIR LOG_DIR KEEP_PHASE1_JSONL

echo "[$(date -u +%FT%TZ)] building phase1 replay and 2:1 mix with ${WORKERS} workers"
seq 0 197 | xargs -P "${WORKERS}" -I '{}' bash -c 'build_one "$1"' _ '{}'

mix_files=()
for index_raw in $(seq 0 197); do
  index="$(printf '%05d' "${index_raw}")"
  mix_files+=("${MIX_DIR}/train-${index}.jsonl")
  [[ -s "${mix_files[-1]}" ]] || { echo "Missing mix shard ${mix_files[-1]}" >&2; exit 1; }
done

if [[ -s "${PACKED}" && -s "${PACKED}.count" ]]; then
  echo "[$(date -u +%FT%TZ)] packed file already exists: ${PACKED} count=$(<"${PACKED}.count")"
else
  echo "[$(date -u +%FT%TZ)] packing ${#mix_files[@]} mix shards"
  rm -rf "${PACKED}.parts" "${PACKED}.tmp"
  "${PYTHON}" "${REPO_ROOT}/training/pack_sequences_parallel.py" \
    --input "${mix_files[@]}" \
    --output "${PACKED}" \
    --seq-length 18000 \
    --workers "${PACK_WORKERS}" \
    | tee "${LOG_DIR}/pack_report.json"
  wc -l < "${PACKED}" > "${PACKED}.count"
fi

echo "[$(date -u +%FT%TZ)] phase2 data ready packed=$(<"${PACKED}.count") path=${PACKED}"
