#!/usr/bin/env bash
set -euo pipefail

# Find the fastest full-add Stage-1 setting for text -> translated speech.
# The benchmark uses all visible GPUs, real forward/backward/optimizer steps,
# paired raw/codec samples, and isolated cold caches for every raw-audio case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

: "${GPU_IDS:=0,1,2,3,4,5,6,7}"
: "${CONDA_ENV:=liveinterpret}"
: "${BENCH_SAMPLES:=8192}"
: "${CAPACITY_STEPS:=2}"
: "${SPEED_STEPS:=8}"
: "${CODEC_CAPACITY_BATCHES:=64 72 80 96}"
: "${RAW_CAPACITY_BATCHES:=32 48 64 72}"
: "${CODEC_WORKERS:=0 4 8}"
: "${MAX_CODEC_FRAMES:=384}"
: "${MAX_LENGTH:=4096}"
: "${BENCH_TIMEOUT:=20m}"
: "${KEEP_WORK_DIR:=false}"
: "${BENCH_CASES:=}"

RAW_SHARD="${RAW_SHARD:-${WORKSPACE_ROOT}/data/sft_t2a_translate_converted/raw_audio/manifests/part-rg00000.jsonl}"
CODEC_SHARD="${CODEC_SHARD:-${WORKSPACE_ROOT}/data/sft_t2a_translate_converted/qwen3_tts_codec/shards/part-rg00000.parquet}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results/t2s_translation_$(date -u +%Y%m%d_%H%M%S)}"
WORK_DIR="$(mktemp -d /tmp/t2s-translation-bench.XXXXXX)"
SUBSET_DIR="${WORK_DIR}/subset"
RUN_ROOT="${WORK_DIR}/runs"
RESULTS_CSV="${RESULT_DIR}/results.csv"
mkdir -p "${RESULT_DIR}/logs" "${RUN_ROOT}"

cleanup() {
    if [[ "${KEEP_WORK_DIR}" == "true" ]]; then
        echo "[bench] keeping work directory: ${WORK_DIR}"
        return
    fi
    case "${WORK_DIR}" in
        /tmp/t2s-translation-bench.*) rm -rf -- "${WORK_DIR}" ;;
        *) echo "[bench] refusing to remove unexpected path: ${WORK_DIR}" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM

if [[ ! -s "${RAW_SHARD}" || ! -s "${CODEC_SHARD}" ]]; then
    echo "[bench] missing raw or codec shard" >&2
    exit 1
fi

set +u
source /opt/dlami/nvme/zeyangsong/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
python "${REPO_ROOT}/dataset_maker/build_t2s_translation_benchmark_subset.py" \
    --raw-manifest "${RAW_SHARD}" \
    --codec-parquet "${CODEC_SHARD}" \
    --output-dir "${SUBSET_DIR}" \
    --num-samples "${BENCH_SAMPLES}" \
    | tee "${RESULT_DIR}/subset_summary.json"

printf '%s\n' \
    'phase,mode,batch_size,workers,steps,status,runtime_s,step_s,global_samples_s,peak_mem_mib,avg_gpu_util_pct,log' \
    >"${RESULTS_CSV}"

IFS=',' read -r -a _gpu_array <<<"${GPU_IDS}"
NUM_GPUS="${#_gpu_array[@]}"
: "${MASTER_PORT_BASE:=$((31000 + RANDOM % 1000))}"
CASE_INDEX=0
LAST_STATUS=""
LAST_RUNTIME=""
LAST_STEP_S=""
LAST_SAMPLES_S=""

run_case() {
    local phase="$1" mode="$2" bs="$3" workers="$4" steps="$5"
    local data log telemetry output cache run_name status rc sampler_pid
    local runtime step_s samples_s peak_mem avg_util
    CASE_INDEX=$((CASE_INDEX + 1))
    run_name="t2s_bench_${phase}_${mode}_bs${bs}_w${workers}"
    log="${RESULT_DIR}/logs/${run_name}.log"
    telemetry="${RESULT_DIR}/logs/${run_name}.gpu.csv"
    output="${RUN_ROOT}/${run_name}"
    cache="${RUN_ROOT}/${run_name}_codec_cache"
    if [[ "${mode}" == "codec" ]]; then
        data="${SUBSET_DIR}/codec.parquet"
    else
        data="${SUBSET_DIR}/raw.jsonl"
    fi
    mkdir -p "${output}" "${cache}"

    echo "[bench] ${phase} mode=${mode} bs=${bs} workers=${workers} steps=${steps}"
    (
        while true; do
            nvidia-smi -i "${GPU_IDS}" --query-gpu=index,memory.used,utilization.gpu \
                --format=csv,noheader,nounits >>"${telemetry}" 2>/dev/null || true
            sleep 0.5
        done
    ) &
    sampler_pid="$!"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    CONDA_ENV="${CONDA_ENV}" \
    BRIDGE_ALIGNMENT_DATA="${data}" \
    BRIDGE_ALIGNMENT_CODEC_CACHE="${cache}" \
    TALKER_METHOD=full_add \
    UNFREEZE_TALKER_LAYERS=28 \
    UNFREEZE_MTP=1 \
    UNFREEZE_CODEC_HEAD=1 \
    QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false \
    PER_DEVICE_TRAIN_BATCH_SIZE="${bs}" \
    PER_DEVICE_EVAL_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=1 \
    DATALOADER_NUM_WORKERS="${workers}" \
    DATALOADER_PERSISTENT_WORKERS="$([[ "${workers}" -gt 0 ]] && echo true || echo false)" \
    DATALOADER_PREFETCH_FACTOR="$([[ "${workers}" -gt 0 ]] && echo 2 || echo '')" \
    DATASET_NUM_PROC=8 \
    SPLIT_DATASET_RATIO=0 \
    MAX_CODEC_FRAMES="${MAX_CODEC_FRAMES}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    REPORT_TO=none \
    SAVE_STEPS=1000000 \
    EVAL_STEPS=1000000 \
    LOGGING_STEPS=1 \
    RUN_NAME="${run_name}" \
    OUTPUT_DIR="${output}" \
    BATCH_TAG=t2s_benchmark \
    MASTER_PORT=$((MASTER_PORT_BASE + CASE_INDEX)) \
    timeout --signal=TERM --kill-after=60s "${BENCH_TIMEOUT}" \
        bash "${SCRIPT_DIR}/train_text_to_speech_translation.sh" \
        --max_steps "${steps}" --save_strategy no \
        2>&1 | tee "${log}"
    rc="${PIPESTATUS[0]}"
    set -e
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true

    status=ok
    if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
        status=timeout
    elif rg -qi 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError' "${log}"; then
        status=oom
    elif [[ "${rc}" -ne 0 ]]; then
        status="error_${rc}"
    fi

    runtime="$(rg -o "'train_runtime': [0-9.]+" "${log}" | tail -n 1 | awk '{print $2}' || true)"
    if [[ -n "${runtime}" ]]; then
        step_s="$(awk -v runtime="${runtime}" -v steps="${steps}" 'BEGIN {printf "%.4f", runtime/steps}')"
        samples_s="$(awk -v bs="${bs}" -v ngpu="${NUM_GPUS}" -v step_s="${step_s}" 'BEGIN {printf "%.3f", bs*ngpu/step_s}')"
    else
        step_s=""
        samples_s=""
    fi
    peak_mem="$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$2); if ($2+0>m)m=$2+0} END{print m}' "${telemetry}" 2>/dev/null || true)"
    avg_util="$(awk -F, '$2+0>10000 {gsub(/ /,"",$3); s+=$3; n++} END{if(n)printf "%.1f",s/n; else print ""}' "${telemetry}" 2>/dev/null || true)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${phase}" "${mode}" "${bs}" "${workers}" "${steps}" "${status}" \
        "${runtime}" "${step_s}" "${samples_s}" "${peak_mem}" "${avg_util}" "${log}" \
        >>"${RESULTS_CSV}"
    LAST_STATUS="${status}"
    LAST_RUNTIME="${runtime}"
    LAST_STEP_S="${step_s}"
    LAST_SAMPLES_S="${samples_s}"
    echo "[bench] result status=${status} step_s=${step_s:-NA} samples/s=${samples_s:-NA} peak_mem=${peak_mem:-NA}MiB"

    # Do not start the next distributed job until all CUDA contexts from this
    # case have gone away. This is especially important after an OOM.
    local attempt max_used
    for attempt in $(seq 1 30); do
        max_used="$(nvidia-smi -i "${GPU_IDS}" --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}')"
        (( max_used < 1000 )) && break
        sleep 2
    done
    if (( max_used >= 1000 )); then
        echo "[bench] warning: GPU memory is still in use (${max_used} MiB max)" >&2
    fi
}

capacity_search() {
    local mode="$1" workers="$2" batches="$3" bs
    local max_fit=0
    for bs in ${batches}; do
        run_case capacity "${mode}" "${bs}" "${workers}" "${CAPACITY_STEPS}"
        if [[ "${LAST_STATUS}" == "ok" ]]; then
            max_fit="${bs}"
        else
            break
        fi
    done
    printf '%s\n' "${max_fit}"
}

if [[ -n "${BENCH_CASES}" ]]; then
    # Fast targeted mode, e.g.
    # BENCH_CASES='codec:32:4:8 codec:48:4:8 raw:32:0:8'
    for case_spec in ${BENCH_CASES}; do
        IFS=: read -r mode bs workers steps <<<"${case_spec}"
        if [[ -z "${mode}" || -z "${bs}" || -z "${workers}" || -z "${steps}" ]]; then
            echo "[bench] invalid BENCH_CASES item: ${case_spec}" >&2
            exit 1
        fi
        run_case targeted "${mode}" "${bs}" "${workers}" "${steps}"
    done
    echo "[bench] targeted cases complete: ${RESULTS_CSV}"
    column -s, -t "${RESULTS_CSV}" 2>/dev/null || cat "${RESULTS_CSV}"
    exit 0
fi

# Command substitution would run the function in a subshell and swallow live
# logs, so write the discovered limits into ordinary globals.
CODEC_MAX=0
for bs in ${CODEC_CAPACITY_BATCHES}; do
    run_case capacity codec "${bs}" 4 "${CAPACITY_STEPS}"
    [[ "${LAST_STATUS}" == "ok" ]] || break
    CODEC_MAX="${bs}"
done

RAW_MAX=0
for bs in ${RAW_CAPACITY_BATCHES}; do
    run_case capacity raw "${bs}" 0 "${CAPACITY_STEPS}"
    [[ "${LAST_STATUS}" == "ok" ]] || break
    RAW_MAX="${bs}"
done

if (( CODEC_MAX > 0 )); then
    # Compare the maximum with two lower-padding alternatives, then sweep
    # workers at the fastest measured batch. Duplicate cases are avoided.
    declare -A seen_codec=()
    for bs in $((CODEC_MAX / 2)) $(((CODEC_MAX * 3) / 4)) "${CODEC_MAX}"; do
        bs=$(( (bs / 4) * 4 ))
        (( bs < 4 )) && bs=4
        key="${bs}:4"
        [[ -n "${seen_codec[${key}]:-}" ]] && continue
        seen_codec["${key}"]=1
        run_case speed codec "${bs}" 4 "${SPEED_STEPS}"
    done
    CODEC_BEST="$(awk -F, '
        $1=="speed" && $2=="codec" && $6=="ok" && $9+0>best {
            best=$9+0; bs=$3
        }
        END {print bs+0}' "${RESULTS_CSV}")"
    (( CODEC_BEST > 0 )) || CODEC_BEST="${CODEC_MAX}"
    for workers in ${CODEC_WORKERS}; do
        [[ "${workers}" == "4" ]] && continue
        run_case workers codec "${CODEC_BEST}" "${workers}" "${SPEED_STEPS}"
    done
fi

if (( RAW_MAX > 0 )); then
    for bs in $((RAW_MAX / 2)) "${RAW_MAX}"; do
        bs=$(( (bs / 4) * 4 ))
        (( bs < 4 )) && bs=4
        run_case speed raw "${bs}" 0 "${SPEED_STEPS}"
    done
fi

echo "[bench] complete: ${RESULTS_CSV}"
column -s, -t "${RESULTS_CSV}" 2>/dev/null || cat "${RESULTS_CSV}"
echo "[bench] fastest successful row per input mode:"
awk -F, '
    NR > 1 && $6 == "ok" && $9 + 0 > best[$2] {
        best[$2] = $9 + 0
        row[$2] = $0
    }
    END {
        for (mode in row) print row[mode]
    }' "${RESULTS_CSV}" | sort
