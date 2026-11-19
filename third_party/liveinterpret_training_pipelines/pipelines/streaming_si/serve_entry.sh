#!/usr/bin/env bash
# streaming_si 实时推理服务的启动脚本（KtrainJob framework: pytorch，常驻）。
#
# 由 ktrain/launch_train.sh 在 Pod 内执行：它已经把 yaml 里的 fsx:// 工件转成
# ${FSX_MOUNT}/<path>（模型直接读 FSX，不做 S3 同步），并准备了 NCCL/NVMe 等环境。
# 本脚本只做三件事：校验模型目录 → 起服务 → 等第一个请求把 CUDA Graph 录好。
#
# 环境变量（yaml spec.env）：
#   MODEL_PATH              S4 完整模型目录（fsx:// 或绝对路径）；必须是 S1-S4 产物，不接受 S5 adapter
#   COSYVOICE_REPO          CosyVoice 官方仓库（镜像内已有 /opt/CosyVoice）
#   COSYVOICE_SPEECH_TOKENIZER  源语音声学 prompt 用的 speech_tokenizer_v3.onnx；缺省时服务只用说话人向量
#   SERVE_PORT              服务端口（默认 8000）
#   SERVE_DEVICE            推理设备（默认 cuda:0）
#   SERVE_EXTRA_ARGS        追加给 server.py 的参数（空格分隔）
#
# 常驻：脚本前台运行 uvicorn，Pod 不退出，直到 `ktrain job stop`。
# 连接方式见 ktrain/ktrainjob_streaming_si_serve.yaml 头部注释。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_DEVICE="${SERVE_DEVICE:-cuda:0}"
COSYVOICE_REPO="${COSYVOICE_REPO:-/opt/CosyVoice}"
: "${MODEL_PATH:?需要 MODEL_PATH（S4 完整模型目录）}"

log() { echo "[serve] $*" >&2; }
die() { echo "[serve][FATAL] $*" >&2; exit 1; }

# launch_train.sh 只转换 MODEL_PATH 等固定变量；这里补上本服务自己的 fsx:// 变量
to_local() {
    case "$1" in
        fsx://*) echo "${FSX_MOUNT:-/mnt/fsx}/${1#fsx://}" ;;
        *)       echo "$1" ;;
    esac
}
MODEL_PATH="$(to_local "${MODEL_PATH}")"
[[ -n "${COSYVOICE_SPEECH_TOKENIZER:-}" ]] && COSYVOICE_SPEECH_TOKENIZER="$(to_local "${COSYVOICE_SPEECH_TOKENIZER}")"

# ---- 模型目录校验（缺一项都会在推理时才炸，提前查清楚）----
[[ -d "${MODEL_PATH}" ]] || die "MODEL_PATH 不存在：${MODEL_PATH}"
for f in config.json model.safetensors.index.json; do
    [[ -f "${MODEL_PATH}/${f}" ]] || die "模型目录缺 ${f}：${MODEL_PATH}"
done
[[ -f "${MODEL_PATH}/speech_decoder/flow.pt" ]] || \
    die "模型目录缺 speech_decoder/flow.pt，无法合成语音；只要译文请改调 --no_vocoder（SERVE_EXTRA_ARGS）"
if [[ -f "${MODEL_PATH}/s4_metadata.json" ]]; then
    die "${MODEL_PATH} 是 S5 adapter 目录；本服务只接 S4 完整模型目录"
fi
[[ -d "${COSYVOICE_REPO}" ]] || die "COSYVOICE_REPO 不存在：${COSYVOICE_REPO}"

if [[ -n "${COSYVOICE_SPEECH_TOKENIZER:-}" ]]; then
    [[ -f "${COSYVOICE_SPEECH_TOKENIZER}" ]] || \
        log "[WARN] COSYVOICE_SPEECH_TOKENIZER 不存在，退回只用说话人向量：${COSYVOICE_SPEECH_TOKENIZER}"
else
    log "[WARN] 未设 COSYVOICE_SPEECH_TOKENIZER，源语音声学 prompt 关闭（音色仍由 CAM++ 保留）"
fi

log "MODEL_PATH=${MODEL_PATH}"
log "COSYVOICE_REPO=${COSYVOICE_REPO}"
log "port=${SERVE_PORT} device=${SERVE_DEVICE}"
gpu_line="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -3 || true)"
[[ -n "${gpu_line}" ]] && log "GPU: ${gpu_line//$'\n'/ | }"

export PYTHONPATH="${REPO_ROOT}/custom${PYTHONPATH:+:${PYTHONPATH}}"
export COSYVOICE_REPO
[[ -n "${COSYVOICE_SPEECH_TOKENIZER:-}" ]] && export COSYVOICE_SPEECH_TOKENIZER
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC2206
EXTRA=( ${SERVE_EXTRA_ARGS:-} )

log "启动服务（模型加载 + CUDA Graph 预热需要一两分钟，health 在此之前一直是 503）"
exec "${PYTHON}" -m streaming_si.serve.server \
    --model_dir "${MODEL_PATH}" \
    --cosyvoice_repo "${COSYVOICE_REPO}" \
    --cosyvoice_dir "${MODEL_PATH}/speech_decoder" \
    ${COSYVOICE_SPEECH_TOKENIZER:+--prompt_tokenizer "${COSYVOICE_SPEECH_TOKENIZER}"} \
    --device "${SERVE_DEVICE}" \
    --host 0.0.0.0 --port "${SERVE_PORT}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
