#!/usr/bin/env bash
# Batch 5.0 continuation: warm-start the interrupted bridge-concat + thinker
# experiment from checkpoint-15500 and continue on GPUs 4-7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${ACTION:=${1:-plan}}"
: "${INIT_CHECKPOINT:=${REPO_ROOT}/output/qwen_omnispeak_output/batch5.0/bridge_concat_thinker-3b-lr1e-5-bs52x2-t2s/v0-20260721-095451/checkpoint-15500}"
: "${DATA_PATH:=/opt/dlami/nvme/zeyangsong/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.le30s.parquet}"
: "${RUN_NAME:=bridge_concat_thinker-from-bridge15500-3b-lr1e-5-bs26x4-t2s}"
: "${BATCH_TAG:=batch5.0}"
# checkpoint-15500 was at epoch ~=0.975 of a 5-epoch run. This model-only
# continuation covers approximately the remaining four epochs.
: "${NUM_TRAIN_EPOCHS:=4.025}"
: "${LEARNING_RATE:=1e-5}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=26}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${DATALOADER_NUM_WORKERS:=4}"
: "${DATASET_NUM_PROC:=16}"
: "${GROUP_BY_LENGTH:=false}"
: "${MAX_CODEC_FRAMES:=384}"
: "${MAX_LENGTH:=4096}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${EVAL_STEPS:=2500}"
: "${SAVE_STEPS:=2500}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${MASTER_PORT:=29511}"
: "${WANDB_PROJECT:=qwen_omnispeak_t2s_v2}"
: "${WANDB_RUN_GROUP:=batch5.0_t2s}"
: "${MONITOR_PYTHON:=/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python}"

DATA_DISK_ROOT="/opt/dlami/nvme/zeyangsong"
RUN_LOG_DIR="${REPO_ROOT}/logs/${BATCH_TAG}/${RUN_NAME}"
TRAINING_LOG="${RUN_LOG_DIR}/training.log"
OUTPUT_DIR="${REPO_ROOT}/output/qwen_omnispeak_output/${BATCH_TAG}/${RUN_NAME}"
PID_FILE="${RUN_LOG_DIR}/launcher.pid"
DATASET_CACHE="${DATA_DISK_ROOT}/.cache/datasets_t2s_bridge_concat_thinker"
TMUX_SESSION="batch50_bridge_concat_thinker"

show_plan() {
    echo "=== Batch 5.0 bridge-concat + thinker continuation (4 GPU) ==="
    echo "  init:       ${INIT_CHECKPOINT}"
    echo "  data:       ${DATA_PATH}"
    echo "  output:     ${OUTPUT_DIR}"
    echo "  GPUs:       4,5,6,7"
    echo "  trainable:  bridge + talker 28 layers + MTP + codec_head + thinker"
    echo "  batch:      ${PER_DEVICE_TRAIN_BATCH_SIZE} x 4 x ga${GRADIENT_ACCUMULATION_STEPS} = $((PER_DEVICE_TRAIN_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
    echo "  schedule:   lr=${LEARNING_RATE}, remaining_epochs=${NUM_TRAIN_EPOCHS}, eval/save=${EVAL_STEPS}/${SAVE_STEPS}"
    echo "  sampling:   group_by_length=${GROUP_BY_LENGTH}, workers=${DATALOADER_NUM_WORKERS}/rank"
    echo "  resume:     model weights only (ZeRO-2 optimizer checkpoint has 2 shards)"
}

start_run() {
    [[ -d "${INIT_CHECKPOINT}" ]] || { echo "[ERROR] missing checkpoint: ${INIT_CHECKPOINT}" >&2; exit 1; }
    [[ -s "${DATA_PATH}" ]] || { echo "[ERROR] missing dataset: ${DATA_PATH}" >&2; exit 1; }
    [[ ! -e "${TRAINING_LOG}" ]] || { echo "[ERROR] log already exists: ${TRAINING_LOG}" >&2; exit 1; }
    [[ ! -e "${OUTPUT_DIR}" ]] || { echo "[ERROR] output already exists: ${OUTPUT_DIR}" >&2; exit 1; }

    local gid busy
    for gid in 4 5 6 7; do
        busy="$(nvidia-smi -i "${gid}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
        [[ -z "${busy}" ]] || { echo "[ERROR] GPU ${gid} has compute processes: ${busy//$'\n'/, }" >&2; exit 1; }
    done

    mkdir -p "${RUN_LOG_DIR}" "${DATASET_CACHE}" \
        "${DATA_DISK_ROOT}/.cache/huggingface" \
        "${DATA_DISK_ROOT}/.cache/modelscope" \
        "${DATA_DISK_ROOT}/.cache/xdg" \
        "${DATA_DISK_ROOT}/.tmp"

    local common_env=(
        "CUDA_VISIBLE_DEVICES=4,5,6,7"
        "NPROC_PER_NODE=4"
        "NNODES=1"
        "NODE_RANK=0"
        "MASTER_ADDR=127.0.0.1"
        "MASTER_PORT=${MASTER_PORT}"
        "CONDA_ENV=liveinterpret"
        "CONDA_SH=/opt/dlami/nvme/zeyangsong/miniconda/etc/profile.d/conda.sh"
        "MODEL_TYPE=qwen2_5_3b_omnispeak_a"
        "MODEL_PATH=${INIT_CHECKPOINT}"
        "BRIDGE_ALIGNMENT_DATA=${DATA_PATH}"
        "RUN_NAME=${RUN_NAME}"
        "OUTPUT_DIR=${OUTPUT_DIR}"
        "BATCH_TAG=${BATCH_TAG}"
        "TALKER_TRAINING_SCOPE=layers"
        "UNFREEZE_TALKER_LAYERS=28"
        "BRIDGE_EXTRA_TRAINABLE=model.talker.code_predictor model.talker.codec_head"
        "UNFREEZE_THINKER=1"
        "QWEN_OMNISPEAK_TEXT_LOSS=true"
        "QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT=1.0"
        "QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false"
        "RESIDUAL_LOSS=true"
        "RESIDUAL_LOSS_WEIGHT=0.3"
        "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
        "LEARNING_RATE=${LEARNING_RATE}"
        "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
        "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
        "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
        "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
        "DATALOADER_PERSISTENT_WORKERS=true"
        "DATALOADER_PREFETCH_FACTOR=2"
        "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
        "GROUP_BY_LENGTH=${GROUP_BY_LENGTH}"
        "MAX_CODEC_FRAMES=${MAX_CODEC_FRAMES}"
        "MAX_LENGTH=${MAX_LENGTH}"
        "SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO}"
        "EVAL_STEPS=${EVAL_STEPS}"
        "SAVE_STEPS=${SAVE_STEPS}"
        "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
        "REPORT_TO=wandb"
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
        "WANDB_BASE_URL=${WANDB_BASE_URL:-http://43.173.190.72:8080}"
        "NO_PROXY=${NO_PROXY:-43.173.190.72,localhost,127.0.0.1}"
        "HF_HOME=${DATA_DISK_ROOT}/.cache/huggingface"
        "HF_DATASETS_CACHE=${DATASET_CACHE}"
        "MODELSCOPE_CACHE=${DATA_DISK_ROOT}/.cache/modelscope"
        "XDG_CACHE_HOME=${DATA_DISK_ROOT}/.cache/xdg"
        "TMPDIR=${DATA_DISK_ROOT}/.tmp"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    )
    [[ -z "${WANDB_API_KEY:-}" ]] || common_env+=("WANDB_API_KEY=${WANDB_API_KEY}")

    echo "$$" > "${PID_FILE}"
    exec "${MONITOR_PYTHON}" "${REPO_ROOT}/tools/monitor_training_resources.py" run \
        --gpu-ids 4,5,6,7 \
        --interval 1 \
        --output-dir "${RUN_LOG_DIR}" \
        --log-file "${TRAINING_LOG}" \
        --summary-only \
        --summary-refresh-seconds 10 \
        --summary-window-samples 300 \
        -- \
        env -u RESUME_FROM -u PREV_CHECKPOINT "${common_env[@]}" \
        bash "${REPO_ROOT}/pipelines/bridge_alignment/train_text_to_speech_translation.sh" \
        >"${RUN_LOG_DIR}/monitor.log" 2>&1
}

launch_detached() {
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "[ERROR] tmux session already exists: ${TMUX_SESSION}" >&2
        exit 1
    fi
    tmux new-session -d -s "${TMUX_SESSION}" \
        "cd '${REPO_ROOT}' && exec env ACTION=worker bash '${REPO_ROOT}/pipelines/run_bridge_concat_thinker_5_0.sh'"
    echo "[STARTED] tmux=${TMUX_SESSION}"
    echo "  training: ${TRAINING_LOG}"
    echo "  hardware: ${RUN_LOG_DIR}/hardware.log"
}

stop_run() {
    [[ -s "${PID_FILE}" ]] || { echo "[ERROR] missing PID file: ${PID_FILE}" >&2; exit 1; }
    local pid
    pid="$(<"${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}"
        echo "[STOP] PID=${pid}"
    else
        echo "[STOPPED] PID=${pid}"
    fi
    tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
}

case "${ACTION}" in
    plan) show_plan ;;
    start) show_plan; launch_detached ;;
    worker) start_run ;;
    stop) stop_run ;;
    status)
        show_plan
        if [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
            echo "  status: RUNNING pid=$(<"${PID_FILE}")"
        else
            echo "  status: STOPPED"
        fi
        ;;
    *) echo "Usage: bash pipelines/run_bridge_concat_thinker_5_0.sh {plan|start|stop|status}" >&2; exit 1 ;;
esac
