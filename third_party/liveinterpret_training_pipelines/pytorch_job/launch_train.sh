#!/usr/bin/env bash
# =============================================================================
# PyTorchJob 容器入口（H200 / speech-h200-eks）。纯基础设施接线，不做 aws / s3 sync：
# 代码、模型、tokenizer、manifest、音频都已在节点就绪——
#   * 代码：经 S3 CSI 挂载 /s3all 直接读；TRAIN_SCRIPT 在 yaml 中为 repo 相对路径，
#     由本脚本按 CODE_PREFIX 拼出 /s3all/<CODE_PREFIX>/<TRAIN_SCRIPT> 后执行；
#   * OUTPUT_ROOT 同样落在 /s3all（S3 CSI 挂载），不转换；
#   * 模型/tokenizer/manifest/音频：submit 流程已并行同步到节点 NVMe（hostPath），
#     或直接由 FSX CSI 挂载进容器（fsx:// 形式，无需同步）。
#
# =============================================================================
# 工件 URI scheme（与 derive_sync_jobs.py 的三分支语义严格一一对应，唯一约定）
# =============================================================================
#   s3://<bucket>/<key>  S3 直链。submit 侧用 `aws s3 sync --size-only` 同步到
#                        每个节点的 NVMe，容器内最终路径 = ${NVME_MOUNT}/<key>。
#   s3mnt://<key>        已由 S3 CSI 挂载进容器的对象（挂载点 = yaml 里 volume
#                        `s3all` 的 mountPath，默认 /s3all）。submit 侧不走
#                        aws s3 sync，而是从挂载点做本地文件系统同步到 NVMe，
#                        容器内最终路径同样是 ${NVME_MOUNT}/<key>。
#   fsx://<path>         已由 FSX CSI 挂载进容器（挂载点 = volume `fsx` 的
#                        mountPath，默认 /fsx）。**完全跳过同步**，训练直接
#                        从 FSX 读，容器内最终路径 = ${FSX_MOUNT}/<path>。
#   /abs/path 或其他     原样透传（本地开发 / 已是绝对路径的场景）。
#
# 本脚本只负责「URI -> 容器内真实路径」这一步转换，不做任何数据搬运。
# =============================================================================
set -euo pipefail

NVME_MOUNT="${NVME_MOUNT:-/}"
# FSX CSI 在容器内的挂载点，与 yaml 里 volume 名 `fsx` 的 mountPath 必须一致。
# derive_sync_jobs.py 会校验：出现 fsx:// 工件却没挂 fsx volume 时直接 FATAL。
FSX_MOUNT="${FSX_MOUNT:-/fsx}"
# S3 CSI 在容器内的挂载点，与 yaml 里 volume 名 `s3all` 的 mountPath 一致。
# 仅在 s3mnt:// 语义下作为「同步源」，训练侧最终仍读 NVMe，故此处不参与路径转换。
S3ALL_MOUNT="${S3ALL_MOUNT:-/s3all}"

# 代码 S3 挂载前缀（/s3all 下的路径前缀），与 sync_code.sh 同步目的地严格同源。
# TRAIN_SCRIPT 在 yaml 中配置为 repo 相对路径（如 pipelines/.../train_xxx.sh），
# 此处拼出容器内实际执行路径 /s3all/<CODE_PREFIX>/<TRAIN_SCRIPT>。
CODE_PREFIX="${CODE_PREFIX:-deferzhou/liveinterpret_training_pipelines}"
if [[ "${TRAIN_SCRIPT}" != /* && "${TRAIN_SCRIPT}" != s3://* ]]; then
  TRAIN_SCRIPT="/s3all/${CODE_PREFIX}/${TRAIN_SCRIPT}"
fi

# 路径归一化：折叠重复斜杠（yaml 里写成 s3://bucket//data/x 时不能拼出
# /nvme/data//x —— 与 sync 落地的目标路径对不上会静默读不到文件）。
# 只折叠斜杠，不做 .. 解析（工件路径里不应出现 ..）。
normalize_path() {
  local p="$1"
  # 连续斜杠 -> 单斜杠
  while [[ "${p}" == *"//"* ]]; do p="${p//\/\///}"; done
  # 去掉结尾斜杠（根 "/" 除外）
  [[ "${#p}" -gt 1 ]] && p="${p%/}"
  echo "${p}"
}

# 单个 URI -> 容器内真实路径。三种 scheme + 原样透传，语义见文件头表格。
artifact_to_path() {
  local v="$1"
  case "${v}" in
    s3://*)
      # s3://<bucket>/<key> -> ${NVME_MOUNT}/<key>（两桶共享同一 NVMe hostPath，
      # key 与 NVMe 下的相对路径 1:1，无隐式映射潜规则）
      local rest="${v#s3://}"; rest="${rest#*/}"
      normalize_path "${NVME_MOUNT}/${rest}"
      ;;
    s3mnt://*)
      # s3mnt://<key>：submit 侧已从 ${S3ALL_MOUNT}/<key> 同步到 ${NVME_MOUNT}/<key>，
      # 训练读 NVMe（与 s3:// 落点一致，避免训练期反复走 CSI 网络读）
      normalize_path "${NVME_MOUNT}/${v#s3mnt://}"
      ;;
    fsx://*)
      # fsx://<path> -> ${FSX_MOUNT}/<path>，训练直接读 FSX，无同步
      normalize_path "${FSX_MOUNT}/${v#fsx://}"
      ;;
    *)
      # 绝对路径 / 空值 / 其他形式：原样透传（不归一化，避免改动用户显式写法）
      echo "${v}"
      ;;
  esac
}

# 多值版本：ARTIFACT_AUDIO_SHARDS 等空格分隔的多前缀逐个转换后重新拼回。
# 训练脚本目前不读 ARTIFACT_AUDIO_SHARDS，但保持转换语义一致，避免将来
# 有人开始读它时踩「只有这个变量没转换」的坑。
artifacts_to_paths() {
  local raw="$1" out=() tok
  for tok in ${raw}; do
    out+=("$(artifact_to_path "${tok}")")
  done
  echo "${out[*]:-}"
}

export MODEL_PATH="$(artifact_to_path "${MODEL_PATH:-}")"
export QWEN3_TTS_TOKENIZER="$(artifact_to_path "${QWEN3_TTS_TOKENIZER:-}")"
export BRIDGE_ALIGNMENT_DATA="$(artifact_to_path "${BRIDGE_ALIGNMENT_DATA:-}")"
# 可选变量：未设置时不导出空串（下游用 ${VAR:-default} 兜底，导出空串会顶掉默认值）
if [[ -n "${QWEN3_TTS_TOKENIZER_BASE:-}" ]]; then
  export QWEN3_TTS_TOKENIZER_BASE="$(artifact_to_path "${QWEN3_TTS_TOKENIZER_BASE}")"
fi
if [[ -n "${ARTIFACT_AUDIO_SHARDS:-}" ]]; then
  export ARTIFACT_AUDIO_SHARDS="$(artifacts_to_paths "${ARTIFACT_AUDIO_SHARDS}")"
fi

export FSX_MOUNT S3ALL_MOUNT

# WDS 音频（legacy 布局）：manifest 内 wds://curated_audio/... -> /curated_audio/...
# 老任务依赖该软链把 /curated_audio 指到 ${NVME_MOUNT}/data，必须保留。
# 但新任务（unist）用 yaml volumeMount 把真实目录挂到 wds netloc 对应的路径上，
# 因此这里只在 /curated_audio **完全不存在** 时才建软链：
#   * -e 对已存在的挂载目录/真实目录为真 -> 不动它（原来的 ln -sfn 会在目标是
#     目录时把软链建到目录**内部**，产生 /curated_audio/data 这种错位，很隐蔽）；
#   * -L 覆盖「悬空软链」的情况（指向不存在目标的软链 -e 为假），此时视为可重建。
if [[ ! -e /curated_audio && ! -L /curated_audio ]]; then
  ln -sfn "${NVME_MOUNT}/data" /curated_audio
fi

echo "[launch] NVME_MOUNT=${NVME_MOUNT} FSX_MOUNT=${FSX_MOUNT} S3ALL_MOUNT=${S3ALL_MOUNT}"
echo "[launch] MODEL_PATH=${MODEL_PATH}"
echo "[launch] QWEN3_TTS_TOKENIZER=${QWEN3_TTS_TOKENIZER}"
echo "[launch] BRIDGE_ALIGNMENT_DATA=${BRIDGE_ALIGNMENT_DATA}"

# 分布式 env（infra 层统一导出；多节点由 operator 注入 WORLD_SIZE/RANK）
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash "${TRAIN_SCRIPT}" "$@"
