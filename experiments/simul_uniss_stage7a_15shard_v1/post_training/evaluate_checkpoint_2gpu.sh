#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 CHECKPOINT SAMPLES OUTPUT GPU_LIST MASTER_PORT LIMIT_RECORDS_OR_0" >&2
  exit 2
fi
CHECKPOINT="$1"
SAMPLES="$2"
OUTPUT="$3"
GPU_LIST="$4"
MASTER_PORT="$5"
LIMIT_RECORDS="$6"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"

[[ -f "${CHECKPOINT}" && -f "${SAMPLES}" ]] || { echo "Missing checkpoint or samples" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --samples "${SAMPLES}"
  --output "${OUTPUT}"
  --dtype bf16
  --attention-implementation flash_attention_2
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"
  --max-batch-tokens "${EVAL_MAX_BATCH_TOKENS_PER_GPU}"
  --max-batch-size "${EVAL_MAX_BATCH_SIZE_PER_GPU}"
)
if [[ "${LIMIT_RECORDS}" != 0 ]]; then ARGS+=(--limit-records "${LIMIT_RECORDS}"); fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${TRAIN_ENV}/bin/torchrun" \
  --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
  --master-port "${MASTER_PORT}" --nproc-per-node 2 \
  -m training.simul_uniss.stage7a.evaluate "${ARGS[@]}"
