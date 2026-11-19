#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lightweight launcher for text→speech translation experiments.
#
# Usage:
#   bash pipelines/run_t2s.sh plan              # show config, no launch
#   bash pipelines/run_t2s.sh start              # launch all 4 experiments
#   bash pipelines/run_t2s.sh start full_add     # launch one experiment
#   bash pipelines/run_t2s.sh stop               # kill all experiments
#   bash pipelines/run_t2s.sh status             # show running status
#
# Defaults: wandb ON, 2 GPUs/exp, bs=64, max_frames=384, per-branch cache.
# Override any setting via env vars before calling.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Configurable defaults ------------------------------------------------
: "${DATA_PATH:=/opt/dlami/nvme/zeyangsong/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.le30s.parquet}"
: "${CONDA_SH:=/opt/dlami/nvme/zeyangsong/miniconda/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=liveinterpret}"
: "${BATCH_TAG:=batch5.0}"
: "${WANDB_PROJECT:=qwen_omnispeak_t2s_v2}"
: "${WANDB_RUN_GROUP:=batch5.0_t2s}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=64}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=16}"
: "${MAX_CODEC_FRAMES:=384}"
: "${MAX_LENGTH:=4096}"
: "${NUM_TRAIN_EPOCHS:=5}"
: "${LEARNING_RATE:=1e-5}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${EVAL_STEPS:=2500}"
: "${SAVE_STEPS:=2500}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${DATASET_NUM_PROC:=16}"
: "${DATALOADER_NUM_WORKERS:=4}"
: "${GROUP_BY_LENGTH:=false}"
: "${MONITOR_PYTHON:=/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python}"
: "${ENABLE_HARDWARE_MONITOR:=true}"

# ---- Experiment definitions ------------------------------------------------
# format: "key|gpu_ids|master_port|entry_script|talker_method|unfreeze_thinker"
EXPERIMENTS=(
    "bridge_concat|0,1|29500|pipelines/bridge_alignment/train_text_to_speech_translation.sh|bridge_add|false"
    "full_add|2,3|29501|pipelines/upstream_talker/train_text_to_speech_translation.sh|full_add|false"
    "reply_add|4,5|29502|pipelines/upstream_talker/train_text_to_speech_translation.sh|reply_add|false"
    "bridge_concat_thinker|6,7|29503|pipelines/bridge_alignment/train_text_to_speech_translation.sh|bridge_add|true"
)

LOG_BASE="${REPO_ROOT}/logs/${BATCH_TAG}"
PID_FILE="${LOG_BASE}/launcher.pids"

# ---- Helpers ---------------------------------------------------------------
# Conda activation is handled by the training scripts themselves (via CONDA_SH/CONDA_ENV).
# We only need WANDB env vars here.

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if [[ -f ~/.netrc ]]; then
        WANDB_API_KEY="$(grep -m1 password ~/.netrc 2>/dev/null | awk '{print $NF}')" || true
    fi
fi
export WANDB_API_KEY="${WANDB_API_KEY:-local-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://43.173.190.72:8080}"
export NO_PROXY="${NO_PROXY:-43.173.190.72,localhost,127.0.0.1}"
export WANDB_PROJECT

# ---- Redirect all caches to data disk (avoid filling system disk) ----
DATA_DISK_ROOT="/opt/dlami/nvme/zeyangsong"
export MODELSCOPE_CACHE="${DATA_DISK_ROOT}/.cache/modelscope"
export HF_HOME="${DATA_DISK_ROOT}/.cache/huggingface"
export XDG_CACHE_HOME="${DATA_DISK_ROOT}/.cache/xdg"
export TMPDIR="${DATA_DISK_ROOT}/.tmp"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

show_plan() {
    echo "=== Text→Speech Translation — ${BATCH_TAG} ==="
    echo "  data:       ${DATA_PATH}"
    echo "  wandb:      project=${WANDB_PROJECT} group=${WANDB_RUN_GROUP}"
    echo "  common:     train_bs=${PER_DEVICE_TRAIN_BATCH_SIZE}/gpu, eval_bs=${PER_DEVICE_EVAL_BATCH_SIZE}/gpu, max_frames=${MAX_CODEC_FRAMES}, epochs=${NUM_TRAIN_EPOCHS}, lr=${LEARNING_RATE}"
    echo "  schedule:   val=${SPLIT_DATASET_RATIO}, eval/save=${EVAL_STEPS}/${SAVE_STEPS} steps"
    echo "  sampling:   group_by_length=${GROUP_BY_LENGTH} (false=全局随机采样；true=随机长度分桶)"
    echo "  cache:      per-branch (auto, in .cache/datasets_{method}_{task})"
    echo "  monitor:    ${ENABLE_HARDWARE_MONITOR}"
    echo ""
    echo "  Experiments:"
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r key gpus port script method unfreeze <<< "$exp"
        local thinker_tag="frozen"
        [[ "$unfreeze" == "true" ]] && thinker_tag="trainable"
        echo "    ${key}: GPU ${gpus} | ${method} | thinker=${thinker_tag}"
    done
}

launch_one() {
    local exp="$1"
    IFS='|' read -r key gpu_ids port entry_script talker_method unfreeze <<< "$exp"

    local run_name="${key}-3b-lr${LEARNING_RATE}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}x2-t2s"
    local run_log_dir="${LOG_BASE}/${run_name}"
    local training_log="${run_log_dir}/training.log"
    local hardware_log="${run_log_dir}/hardware.log"
    mkdir -p "${run_log_dir}"

    # Per-branch HF datasets cache on the data disk. CRITICAL: bridge_concat and
    # bridge_concat_thinker both source common_args.sh which defaults
    # HF_DATASETS_CACHE=.cache/datasets_bridge_concat — sharing one cache dir
    # makes concurrent parquet parsing collide. Give every branch its own dir.
    local ds_cache="${DATA_DISK_ROOT}/.cache/datasets_t2s_${key}"
    mkdir -p "${ds_cache}"

    # Common env for all experiments
    local common_env=(
        "CUDA_VISIBLE_DEVICES=${gpu_ids}"
        "NPROC_PER_NODE=2"
        "MASTER_PORT=${port}"
        "CONDA_ENV=${CONDA_ENV}"
        "CONDA_SH=${CONDA_SH}"
        "BATCH_TAG=${BATCH_TAG}"
        "MODEL_TYPE=qwen2_5_3b_omnispeak_a"
        "MODEL_PATH=${REPO_ROOT}/models/qwen2_5_3b_omnispeak_a"
        "BRIDGE_ALIGNMENT_DATA=${DATA_PATH}"
        "RUN_NAME=${run_name}"
        "OUTPUT_DIR=${REPO_ROOT}/output/qwen_omnispeak_output/${BATCH_TAG}/${run_name}"
        "REPORT_TO=wandb"
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
        "WANDB_API_KEY=${WANDB_API_KEY}"
        "WANDB_BASE_URL=${WANDB_BASE_URL}"
        "NO_PROXY=${NO_PROXY}"
        "LEARNING_RATE=${LEARNING_RATE}"
        "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
        "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
        "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
        "GRADIENT_ACCUMULATION_STEPS=1"
        "SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO}"
        "EVAL_STEPS=${EVAL_STEPS}"
        "SAVE_STEPS=${SAVE_STEPS}"
        "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
        "MAX_LENGTH=${MAX_LENGTH}"
        "MAX_CODEC_FRAMES=${MAX_CODEC_FRAMES}"
        "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
        "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
        "RESIDUAL_LOSS_WEIGHT=0.3"
        "UNFREEZE_TALKER_LAYERS=28"
        "QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false"
        "MODELSCOPE_CACHE=${MODELSCOPE_CACHE}"
        "HF_HOME=${HF_HOME}"
        "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
        "TMPDIR=${TMPDIR}"
        "HF_DATASETS_CACHE=${ds_cache}"
        "GROUP_BY_LENGTH=${GROUP_BY_LENGTH}"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    )

    # Method-specific
    if [[ "${entry_script}" == *"bridge_alignment"* ]]; then
        common_env+=("TALKER_TRAINING_SCOPE=layers")
        common_env+=("BRIDGE_EXTRA_TRAINABLE=model.talker.code_predictor model.talker.codec_head")
        [[ "${unfreeze}" == "true" ]] && common_env+=("UNFREEZE_THINKER=1")
    else
        common_env+=("TALKER_METHOD=${talker_method}")
        common_env+=("UNFREEZE_MTP=1")
        common_env+=("UNFREEZE_CODEC_HEAD=1")
    fi

    if [[ "${ENABLE_HARDWARE_MONITOR}" == "true" && -f "${REPO_ROOT}/tools/monitor_training_resources.py" ]]; then
        rm -f "${training_log}"  # ensure monitor can create a fresh log-file
        nohup "${MONITOR_PYTHON}" "${REPO_ROOT}/tools/monitor_training_resources.py" run \
            --gpu-ids "${gpu_ids}" \
            --interval 1 \
            --output-dir "${run_log_dir}" \
            --log-file "${training_log}" \
            --summary-only \
            --summary-refresh-seconds 10 \
            --summary-window-samples 300 \
            -- \
            env "${common_env[@]}" bash "${entry_script}" \
            >> "${run_log_dir}/monitor.log" 2>&1 &
    else
        nohup env "${common_env[@]}" bash "${entry_script}" \
            > "${training_log}" 2>&1 &
    fi
    local pid=$!
    echo "${key}	${gpu_ids}	${pid}	${training_log}	${hardware_log}" >> "${PID_FILE}"
    echo "  [STARTED] ${key} GPU=${gpu_ids} PID=${pid}"
    echo "    log: ${training_log}"
    [[ "${ENABLE_HARDWARE_MONITOR}" == "true" ]] && echo "    hardware: ${hardware_log}"
}

do_start() {
    local filter="${1:-all}"
    mkdir -p "${LOG_BASE}"
    : > "${PID_FILE}"

    # GPU busy check - only check the specific GPUs for this experiment
    if [[ "${ALLOW_BUSY_GPUS:-false}" != "true" ]]; then
        local gpu_busy
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r key gpus rest <<< "$exp"
            if [[ "${filter}" == "all" || "${filter}" == "${key}" ]]; then
                for gid in ${gpus//,/ }; do
                    gpu_busy="$(nvidia-smi -i "${gid}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
                    if [[ -n "${gpu_busy}" ]]; then
                        echo "[WARN] GPU ${gid} has active processes: ${gpu_busy}"
                        echo "       Set ALLOW_BUSY_GPUS=true to override."
                        return 1
                    fi
                done
            fi
        done
    fi

    echo "Starting experiments..."
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r key rest <<< "$exp"
        if [[ "${filter}" == "all" || "${filter}" == "${key}" ]]; then
            launch_one "$exp"
        fi
    done
    echo ""
    echo "PID manifest: ${PID_FILE}"
}

do_stop() {
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "No PID file found at ${PID_FILE}"
        return
    fi
    echo "Stopping experiments..."
    while IFS=$'\t' read -r key gpus pid tlog hlog; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
            echo "  [STOP] ${key} PID=${pid}"
        fi
    done < "${PID_FILE}"
    sleep 3
    # Force kill any remaining
    pkill -f "train_text_to_speech_translation" 2>/dev/null || true
    pkill -f "monitor_training_resources" 2>/dev/null || true
    echo "Done."
}

do_status() {
    echo "=== ${BATCH_TAG} Status ==="
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "  No experiments tracked (PID file missing)."
    else
        while IFS=$'\t' read -r key gpus pid tlog hlog; do
            local status="DEAD"
            kill -0 "${pid}" 2>/dev/null && status="RUNNING"
            echo "  ${key}: ${status} (PID=${pid}, GPU=${gpus})"
            if [[ -f "${tlog}" ]]; then
                local last_line
                last_line="$(tail -1 "${tlog}" 2>/dev/null | sed 's/\r//g' | cut -c1-80)"
                echo "    log: ${last_line}"
            fi
        done < "${PID_FILE}"
    fi
    echo ""
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null
}

# ---- Main ------------------------------------------------------------------
case "${1:-plan}" in
    plan)   show_plan ;;
    start)  shift; do_start "${1:-all}" ;;
    stop)   do_stop ;;
    status) do_status ;;
    *)
        echo "Usage: bash pipelines/run_t2s.sh {plan|start [key]|stop|status}"
        echo "Keys: bridge_concat, full_add, reply_add, bridge_concat_thinker"
        exit 1
        ;;
esac
