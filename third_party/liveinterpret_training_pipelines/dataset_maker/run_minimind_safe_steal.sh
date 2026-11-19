#!/usr/bin/env bash
set -euo pipefail

# Safely let a GPU whose original worker finishes early process a distant,
# not-yet-started row-group range owned by another worker. The original worker
# is never stopped. The steal worker aborts before the owner can approach its
# lowest unfinished row group, preventing concurrent writes to the same part.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${GPU_ID:?set GPU_ID}"
: "${FREE_RANGE_START:?set FREE_RANGE_START}"
: "${FREE_RANGE_END:?set FREE_RANGE_END}"
: "${OWNER_LOG:?set OWNER_LOG}"
: "${STEAL_START:?set STEAL_START}"
: "${STEAL_END:?set STEAL_END}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/miniconda/envs/liveinterpret/bin/python}"
CONVERTER="${CONVERTER:-${SCRIPT_DIR}/convert_minimind_mimi_to_qwen3.py}"
INPUT_PARQUET="${INPUT_PARQUET:-${REPO_ROOT}/data/sft_t2a_translate_merged.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/sft_t2a_translate_converted}"
MIMI_MODEL="${MIMI_MODEL:-${REPO_ROOT}/projects/minimind-o/model/mimi}"
QWEN3_TOKENIZER="${QWEN3_TOKENIZER:-${REPO_ROOT}/projects/liveinterpret_training_pipelines/models/Qwen3-TTS-Tokenizer-12Hz}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_BATCH_FRAMES="${MAX_BATCH_FRAMES:-68000}"
GUARD_DISTANCE="${GUARD_DISTANCE:-2}"
POLL_SECONDS="${POLL_SECONDS:-5}"
LOG_DIR="${OUTPUT_ROOT}/logs"
STATE_DIR="${OUTPUT_ROOT}/state"
WORKER_LOG="${LOG_DIR}/gpu${GPU_ID}-steal-rg${STEAL_START}-${STEAL_END}.log"
LOCK_FILE="${LOG_DIR}/gpu${GPU_ID}-steal-rg${STEAL_START}-${STEAL_END}.lock"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[safe-steal] another guard already owns ${LOCK_FILE}" >&2
    exit 1
fi

if (( STEAL_START < 0 || STEAL_END <= STEAL_START || GUARD_DISTANCE < 1 )); then
    echo "[safe-steal] invalid steal range or guard distance" >&2
    exit 1
fi

state_path() {
    printf '%s/part-rg%05d.json' "${STATE_DIR}" "$1"
}

range_complete() {
    local start="$1" end="$2" rg
    for ((rg=start; rg<end; rg++)); do
        [[ -s "$(state_path "${rg}")" ]] || return 1
    done
}

lowest_unfinished() {
    local rg
    for ((rg=STEAL_START; rg<STEAL_END; rg++)); do
        if [[ ! -s "$(state_path "${rg}")" ]]; then
            printf '%d\n' "${rg}"
            return 0
        fi
    done
    printf '%d\n' "${STEAL_END}"
}

owner_current_rg() {
    local match digits
    match="$(rg -o 'Converting rg[0-9]{5}' "${OWNER_LOG}" 2>/dev/null | tail -n 1 || true)"
    [[ -n "${match}" ]] || return 1
    digits="${match##*rg}"
    printf '%d\n' "$((10#${digits}))"
}

# Worker processes may live in a different PID namespace from this guard, so
# state files are the cross-session completion barrier. If the original worker
# fails, this deliberately waits instead of reusing a GPU that may still be in
# an unknown state.
echo "[safe-steal] waiting for original GPU${GPU_ID} range " \
     "[${FREE_RANGE_START}, ${FREE_RANGE_END})"
while ! range_complete "${FREE_RANGE_START}" "${FREE_RANGE_END}"; do
    sleep "${POLL_SECONDS}"
done

# Wait until the original CUDA context has actually released its memory.
while true; do
    used_mib="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib < 10000 )); then
        break
    fi
    sleep "${POLL_SECONDS}"
done

owner_rg="$(owner_current_rg)"
first_unfinished="$(lowest_unfinished)"
if (( first_unfinished >= STEAL_END )); then
    echo "[safe-steal] target range is already complete"
    exit 0
fi
if (( owner_rg >= first_unfinished - GUARD_DISTANCE )); then
    echo "[safe-steal] owner is already too close: owner_rg=${owner_rg}, " \
         "first_unfinished=${first_unfinished}; refusing to start" >&2
    exit 1
fi

echo "[safe-steal] starting GPU${GPU_ID} range [${STEAL_START}, ${STEAL_END}); log=${WORKER_LOG}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${CONVERTER}" \
    --input-parquet "${INPUT_PARQUET}" \
    --output-root "${OUTPUT_ROOT}" \
    --mimi-model "${MIMI_MODEL}" \
    --qwen3-tokenizer "${QWEN3_TOKENIZER}" \
    --device cuda:0 \
    --batch-size "${BATCH_SIZE}" \
    --max-batch-frames "${MAX_BATCH_FRAMES}" \
    --start-row-group "${STEAL_START}" \
    --end-row-group "${STEAL_END}" \
    >"${WORKER_LOG}" 2>&1 &
steal_pid="$!"

abort_steal() {
    if kill -0 "${steal_pid}" 2>/dev/null; then
        kill -TERM "${steal_pid}" 2>/dev/null || true
        wait "${steal_pid}" 2>/dev/null || true
    fi
}
trap abort_steal INT TERM

while kill -0 "${steal_pid}" 2>/dev/null; do
    first_unfinished="$(lowest_unfinished)"
    if (( first_unfinished >= STEAL_END )); then
        break
    fi
    owner_rg="$(owner_current_rg)"
    if (( owner_rg >= first_unfinished - GUARD_DISTANCE )); then
        echo "[safe-steal] guard triggered: owner_rg=${owner_rg}, " \
             "first_unfinished=${first_unfinished}; stopping steal pid=${steal_pid}" >&2
        abort_steal
        exit 2
    fi
    sleep "${POLL_SECONDS}"
done

wait "${steal_pid}"
if ! range_complete "${STEAL_START}" "${STEAL_END}"; then
    echo "[safe-steal] steal worker exited without completing the target range" >&2
    exit 1
fi
echo "[safe-steal] completed range [${STEAL_START}, ${STEAL_END}) on GPU${GPU_ID}"
