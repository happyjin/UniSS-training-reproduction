#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/miniconda/envs/liveinterpret/bin/python}"
CONVERTER="${CONVERTER:-${SCRIPT_DIR}/convert_minimind_mimi_to_qwen3.py}"
INPUT_PARQUET="${INPUT_PARQUET:-${REPO_ROOT}/data/sft_t2a_translate_merged.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/sft_t2a_translate_converted}"
MIMI_MODEL="${MIMI_MODEL:-${REPO_ROOT}/projects/minimind-o/model/mimi}"
QWEN3_TOKENIZER="${QWEN3_TOKENIZER:-${REPO_ROOT}/projects/liveinterpret_training_pipelines/models/Qwen3-TTS-Tokenizer-12Hz}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_BATCH_FRAMES="${MAX_BATCH_FRAMES:-68000}"
START_ROW_GROUP="${START_ROW_GROUP:-0}"
END_ROW_GROUP="${END_ROW_GROUP:-105}"
GPU_IDS="${GPU_IDS:-}"

if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "No GPUs found. Set GPU_IDS explicitly, for example GPU_IDS=0,1." >&2
    exit 1
fi

TOTAL_GROUPS=$((END_ROW_GROUP - START_ROW_GROUP))
if [[ "${TOTAL_GROUPS}" -le 0 ]]; then
    echo "Invalid row-group range [${START_ROW_GROUP}, ${END_ROW_GROUP})." >&2
    exit 1
fi
GROUPS_PER_WORKER=$(((TOTAL_GROUPS + ${#GPU_ARRAY[@]} - 1) / ${#GPU_ARRAY[@]}))
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

pids=()
worker_names=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

for worker_index in "${!GPU_ARRAY[@]}"; do
    worker_start=$((START_ROW_GROUP + worker_index * GROUPS_PER_WORKER))
    if [[ "${worker_start}" -ge "${END_ROW_GROUP}" ]]; then
        break
    fi
    worker_end=$((worker_start + GROUPS_PER_WORKER))
    if [[ "${worker_end}" -gt "${END_ROW_GROUP}" ]]; then
        worker_end="${END_ROW_GROUP}"
    fi
    gpu_id="${GPU_ARRAY[worker_index]}"
    worker_name="gpu${gpu_id}-rg${worker_start}-${worker_end}"
    log_path="${LOG_DIR}/${worker_name}.log"
    echo "Starting ${worker_name}; log=${log_path}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${CONVERTER}" \
        --input-parquet "${INPUT_PARQUET}" \
        --output-root "${OUTPUT_ROOT}" \
        --mimi-model "${MIMI_MODEL}" \
        --qwen3-tokenizer "${QWEN3_TOKENIZER}" \
        --device cuda:0 \
        --batch-size "${BATCH_SIZE}" \
        --max-batch-frames "${MAX_BATCH_FRAMES}" \
        --start-row-group "${worker_start}" \
        --end-row-group "${worker_end}" \
        >"${log_path}" 2>&1 &
    pids+=("$!")
    worker_names+=("${worker_name}")
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
        echo "Completed ${worker_names[index]}"
    else
        echo "Failed ${worker_names[index]} (see ${LOG_DIR}/${worker_names[index]}.log)" >&2
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

"${PYTHON_BIN}" "${CONVERTER}" \
    --output-root "${OUTPUT_ROOT}" \
    --finalize-only

echo "All workers and finalization completed: ${OUTPUT_ROOT}"
