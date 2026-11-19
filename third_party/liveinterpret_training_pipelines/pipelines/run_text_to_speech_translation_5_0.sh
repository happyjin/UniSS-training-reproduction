#!/usr/bin/env bash
set -euo pipefail

# Batch 5.0: text -> translated text + target speech, precomputed Qwen3 codec.
#
# Default is a read-only plan.  After reviewing the experiment definition:
#   ACTION=run bash pipelines/run_text_to_speech_translation_5_0.sh
#
# The four monitored jobs are launched in the background.  This script does
# not apply CPU/NUMA affinity because the affinity benchmark found no material
# gain.  Resource monitoring is external to the trainer and excludes eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${ACTION:=plan}"
: "${DATA_PATH:=/opt/dlami/nvme/zeyangsong/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.parquet}"
: "${CONDA_ENV:=liveinterpret}"
: "${MONITOR_PYTHON:=/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python}"
: "${BATCH_TAG:=batch5.0}"
: "${WANDB_RUN_GROUP:=text-speech-translation-5.0}"
: "${NUM_TRAIN_EPOCHS:=5}"
: "${TALKER_ONLY_LR:=1e-5}"
: "${THINKER_CONTROL_LR:=1e-5}"
: "${THINKER_CONTROL_CE:=true}"
: "${THINKER_LOSS_WEIGHT:=1.0}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=52}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${DATALOADER_NUM_WORKERS:=8}"
: "${DATALOADER_PERSISTENT_WORKERS:=true}"
: "${DATALOADER_PREFETCH_FACTOR:=2}"
: "${DATASET_NUM_PROC:=8}"
: "${GROUP_BY_LENGTH:=false}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${EVAL_STEPS:=2500}"
: "${SAVE_STEPS:=2500}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${MAX_LENGTH:=4096}"
: "${MAX_CODEC_FRAMES:=768}"
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
: "${ALLOW_BUSY_GPUS:=false}"
: "${ALLOW_EXISTING_OUTPUT:=false}"
: "${RUN_KEYS:=all}"
: "${WAIT_FOR_JOBS:=false}"
: "${TRAINING_CACHE_ROOT:=/opt/dlami/nvme/zeyangsong/.cache/liveinterpret_training}"
: "${TRAINING_HF_HOME:=${TRAINING_CACHE_ROOT}/huggingface}"
: "${TRAINING_HUB_CACHE:=${TRAINING_HF_HOME}/hub}"
: "${TRAINING_MODELSCOPE_CACHE:=${TRAINING_CACHE_ROOT}/modelscope}"
: "${TRAINING_TORCH_HOME:=${TRAINING_CACHE_ROOT}/torch}"
: "${TRAINING_TMP_ROOT:=/opt/dlami/nvme/zeyangsong/.tmp/li}"

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_ROOT:-${REPO_ROOT}/output/qwen_omnispeak_output}/${BATCH_TAG}"
LOG_BASE="${REPO_ROOT}/logs/${BATCH_TAG}"
PID_FILE="${LOG_BASE}/launcher_${RUN_STAMP}.pids"
LAUNCHER_LOG="${LOG_BASE}/launcher_${RUN_STAMP}.log"

if [[ ! -s "${DATA_PATH}" ]]; then
    echo "[ERROR] codec dataset does not exist or is empty: ${DATA_PATH}" >&2
    exit 1
fi

case "${ACTION}" in
    plan|run) ;;
    *)
        echo "[ERROR] ACTION must be plan or run, got: ${ACTION}" >&2
        exit 1
        ;;
esac

print_plan() {
    echo "Batch 5.0 text-to-speech translation"
    echo "  data=${DATA_PATH}"
    echo "  common=2 GPUs/job, bs=${PER_DEVICE_TRAIN_BATCH_SIZE}/rank, ga=${GRADIENT_ACCUMULATION_STEPS}, workers=${DATALOADER_NUM_WORKERS}/rank"
    echo "  speaker_prompt=false, talker_scope=whole, max_codec_frames=${MAX_CODEC_FRAMES}, epochs=${NUM_TRAIN_EPOCHS}"
    echo "  eval_split=${SPLIT_DATASET_RATIO}, eval_steps=${EVAL_STEPS}, save_steps=${SAVE_STEPS}, group_by_length=${GROUP_BY_LENGTH}"
    echo "  cache_root=${TRAINING_CACHE_ROOT} (datasets isolated per run), tmp_root=${TRAINING_TMP_ROOT}"
    echo "  bridge-add frozen thinker : GPU 0,1  lr=${TALKER_ONLY_LR}"
    echo "  full-add frozen thinker   : GPU 2,3  lr=${TALKER_ONLY_LR}"
    echo "  reply-add frozen thinker  : GPU 4,5  lr=${TALKER_ONLY_LR}"
    echo "  bridge-add train thinker  : GPU 6,7  lr=${THINKER_CONTROL_LR}, thinker_ce=${THINKER_CONTROL_CE}, weight=${THINKER_LOSS_WEIGHT}"
    echo "  cpu_affinity=off, train-only resource monitor=on"
}

print_plan
if [[ "${ACTION}" == "plan" ]]; then
    echo "[PLAN] No process was started. Run the thinker BS52 smoke test and verify idle GPUs before setting ACTION=run."
    exit 0
fi

if [[ "${ALLOW_BUSY_GPUS}" != "true" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[ERROR] nvidia-smi is unavailable; cannot safely verify GPU ownership." >&2
        exit 1
    fi
    active_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -n "${active_compute}" ]]; then
        echo "[ERROR] GPUs already have compute processes; refusing to overlap by default." >&2
        echo "        PIDs: ${active_compute//$'\n'/, }" >&2
        echo "        Inspect them, or set ALLOW_BUSY_GPUS=true explicitly." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}"
: > "${PID_FILE}"
declare -a LAUNCHED_PIDS=()

launch_one() {
    local run_key="$1"
    local gpu_ids="$2"
    local master_port="$3"
    local entry_script="$4"
    local talker_method="$5"
    local unfreeze_thinker="$6"
    local learning_rate="$7"
    local thinker_ce="$8"
    local run_name="${run_key}-whole_talker-3b-lr${learning_rate}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}x2-t2s-5.0"
    local output_dir="${OUTPUT_BASE}/${run_name}"
    local run_log_dir="${LOG_BASE}/${run_name}"
    local training_log="${run_log_dir}/training.log"
    local hardware_log="${run_log_dir}/hardware.log"
    local run_cache_dir="${TRAINING_CACHE_ROOT}/${BATCH_TAG}/${run_name}"
    local datasets_cache_dir="${run_cache_dir}/datasets"
    # Keep TMPDIR short: torch elastic creates AF_UNIX sockets below it, whose
    # full paths must fit the platform sockaddr_un limit (typically 108 bytes).
    local tmp_dir="${TRAINING_TMP_ROOT}/${BATCH_TAG}/${master_port}"
    local xdg_cache_dir="${run_cache_dir}/xdg"

    if [[ -e "${output_dir}" && "${ALLOW_EXISTING_OUTPUT}" != "true" ]]; then
        echo "[ERROR] output already exists: ${output_dir}" >&2
        echo "        Refusing to mix runs; set ALLOW_EXISTING_OUTPUT=true only if intentional." >&2
        return 1
    fi
    if [[ -e "${training_log}" || -e "${hardware_log}" ]]; then
        echo "[ERROR] experiment logs already exist: ${run_log_dir}" >&2
        return 1
    fi
    mkdir -p \
        "${run_log_dir}" \
        "${datasets_cache_dir}" \
        "${tmp_dir}" \
        "${xdg_cache_dir}" \
        "${TRAINING_HUB_CACHE}" \
        "${TRAINING_MODELSCOPE_CACHE}" \
        "${TRAINING_TORCH_HOME}"

    local extra_config
    extra_config="{\"batch\":\"5.0\",\"task\":\"text_to_speech_translation\",\"branch\":\"${run_key}\",\"talker_training_scope\":\"whole\",\"unfreeze_thinker\":${unfreeze_thinker},\"thinker_ce_loss\":${thinker_ce},\"speaker_prompt\":false,\"input_format\":\"qwen3_codec_parquet\",\"num_gpus\":2,\"effective_batch_size\":$((PER_DEVICE_TRAIN_BATCH_SIZE * 2 * GRADIENT_ACCUMULATION_STEPS))}"

    local command=(
        "${MONITOR_PYTHON}" "${REPO_ROOT}/tools/monitor_training_resources.py" run
        --gpu-ids "${gpu_ids}"
        --interval 1
        --output-dir "${run_log_dir}"
        --log-file "${training_log}"
        --summary-only
        --summary-refresh-seconds 10
        --summary-window-samples 300
        --
        env -u RESUME_FROM -u PREV_CHECKPOINT
        "CUDA_VISIBLE_DEVICES=${gpu_ids}"
        "NPROC_PER_NODE=2"
        "NNODES=1"
        "NODE_RANK=0"
        "MASTER_ADDR=127.0.0.1"
        "MASTER_PORT=${master_port}"
        "CONDA_ENV=${CONDA_ENV}"
        "HF_HOME=${TRAINING_HF_HOME}"
        "HF_DATASETS_CACHE=${datasets_cache_dir}"
        "HUGGINGFACE_HUB_CACHE=${TRAINING_HUB_CACHE}"
        "MODELSCOPE_CACHE=${TRAINING_MODELSCOPE_CACHE}"
        "TORCH_HOME=${TRAINING_TORCH_HOME}"
        "XDG_CACHE_HOME=${xdg_cache_dir}"
        "TMPDIR=${tmp_dir}"
        "MODEL_TYPE=qwen2_5_3b_omnispeak_a"
        "MODEL_PATH=${REPO_ROOT}/models/qwen2_5_3b_omnispeak_a"
        "BRIDGE_ALIGNMENT_DATA=${DATA_PATH}"
        "BATCH_TAG=${BATCH_TAG}"
        "DISABLE_INNER_TEE=true"
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
        "WANDB_EXTRA_CONFIG=${extra_config}"
        "RUN_NAME=${run_name}"
        "OUTPUT_DIR=${output_dir}"
        "TALKER_METHOD=${talker_method}"
        "TALKER_TRAINING_SCOPE=whole"
        "UNFREEZE_TALKER_LAYERS=28"
        "UNFREEZE_MTP=1"
        "UNFREEZE_CODEC_HEAD=1"
        "UNFREEZE_THINKER=${unfreeze_thinker}"
        "QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false"
        "QWEN_OMNISPEAK_TEXT_LOSS=${thinker_ce}"
        "QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT=${THINKER_LOSS_WEIGHT}"
        "LEARNING_RATE=${learning_rate}"
        "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
        "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
        "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
        "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
        "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
        "DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
        "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
        "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
        "GROUP_BY_LENGTH=${GROUP_BY_LENGTH}"
        "SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO}"
        "EVAL_STEPS=${EVAL_STEPS}"
        "SAVE_STEPS=${SAVE_STEPS}"
        "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
        "MAX_LENGTH=${MAX_LENGTH}"
        "MAX_CODEC_FRAMES=${MAX_CODEC_FRAMES}"
        "RESIDUAL_LOSS=true"
        "RESIDUAL_LOSS_WEIGHT=${RESIDUAL_LOSS_WEIGHT}"
        bash "${entry_script}"
    )

    nohup "${command[@]}" >>"${LAUNCHER_LOG}" 2>&1 &
    local launcher_pid=$!
    LAUNCHED_PIDS+=("${launcher_pid}")
    printf '%s\t%s\t%s\t%s\t%s\n' "${run_key}" "${gpu_ids}" "${launcher_pid}" "${training_log}" "${hardware_log}" >> "${PID_FILE}"
    echo "[STARTED] ${run_key} GPUs=${gpu_ids} PID=${launcher_pid} train_log=${training_log} hardware_log=${hardware_log}"
}

should_launch() {
    local key="$1"
    [[ "${RUN_KEYS}" == "all" || ",${RUN_KEYS}," == *",${key},"* ]]
}

if should_launch bridge_add_thinker_frozen; then
    launch_one bridge_add_thinker_frozen 0,1 35001 \
        "${REPO_ROOT}/pipelines/bridge_alignment/train_text_to_speech_translation.sh" bridge_add false "${TALKER_ONLY_LR}" false
fi
if should_launch full_add_thinker_frozen; then
    launch_one full_add_thinker_frozen 2,3 35002 \
        "${REPO_ROOT}/pipelines/upstream_talker/train_text_to_speech_translation.sh" full_add false "${TALKER_ONLY_LR}" false
fi
if should_launch reply_add_thinker_frozen; then
    launch_one reply_add_thinker_frozen 4,5 35003 \
        "${REPO_ROOT}/pipelines/upstream_talker/train_text_to_speech_translation.sh" reply_add false "${TALKER_ONLY_LR}" false
fi
if should_launch bridge_add_thinker_unfrozen; then
    launch_one bridge_add_thinker_unfrozen 6,7 35004 \
        "${REPO_ROOT}/pipelines/bridge_alignment/train_text_to_speech_translation.sh" bridge_add true "${THINKER_CONTROL_LR}" "${THINKER_CONTROL_CE}"
fi

echo "[DONE] Selected background monitor/training parents launched. PID manifest: ${PID_FILE}"

if [[ "${WAIT_FOR_JOBS}" == "true" ]]; then
    echo "[WAIT] Supervising ${#LAUNCHED_PIDS[@]} jobs until completion."
    overall_status=0
    for launcher_pid in "${LAUNCHED_PIDS[@]}"; do
        if ! wait "${launcher_pid}"; then
            echo "[ERROR] Monitored training parent ${launcher_pid} exited non-zero." >&2
            overall_status=1
        fi
    done
    echo "[WAIT] All selected jobs exited; aggregate_status=${overall_status}."
    exit "${overall_status}"
fi
