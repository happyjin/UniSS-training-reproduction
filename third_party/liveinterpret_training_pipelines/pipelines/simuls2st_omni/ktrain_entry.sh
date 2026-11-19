#!/usr/bin/env bash
# SimulS2ST-Omni 的 ktrain 入口（KtrainJob framework: pytorch，yaml 里 TRAIN_SCRIPT 指向本文件）。
#
# ktrain/launch_train.sh 在每个 Pod（每节点一个）上执行本脚本，已导出 NNODES / NODE_RANK / MASTER_* /
# NPROC_PER_NODE，并把 MODEL_PATH 的 s3:// 换成了节点 NVMe 路径。本脚本：
#   1. 按 SIMULS2ST_MAP_* 在节点 NVMe 上用软链拼出与源项目同构的目录树（data/...、models/...、outputs/），
#      令 SIMULS2ST_ROOT 指向它：阶段配置 configs/stage*.yaml 原样使用，不必改写任何路径；
#   2. 交接目录（SIMULS2ST_INIT_TALKER_FROM / SIMULS2ST_MERGE_ADAPTERS_FROM）的 s3:// 换成 NVMe 路径；
#   3. 按 SIMULS2ST_STAGE 调用 train_stage{1,2,3}.sh。
#
# 环境变量（yaml spec.env）：
#   SIMULS2ST_STAGE              stage1 | stage2 | stage3（必填）
#   SIMULS2ST_MAP_<任意名>        "<配置里的相对路径> <s3:// URI | 绝对路径 | fsx:// 路径>"，例如
#                                SIMULS2ST_MAP_EMILIA: "data/emilia/dualcodec s3://<bucket>/<key>/emilia/dualcodec/"
#                                （第二个 token 是 s3:// 时 submit.sh 自动同步到 ${NVME_MOUNT}/<key>）
#   MODEL_PATH                   offline 基座（自动映射为 models/SimulS2ST-Omni/offline）
#   OUTPUT_ROOT                  outputs/ 的落点（缺省 ${NVME_MOUNT}/output/simuls2st_omni）；Stage 1 写
#                                outputs/checkpoint/stage1_talker_warmup，Stage 2 默认从那里接 talker
#   SIMULS2ST_INIT_TALKER_FROM   Stage 2 起点（缺省用配置的 outputs/checkpoint/stage1_talker_warmup）
#   SIMULS2ST_MERGE_ADAPTERS_FROM Stage 3 起点（缺省 outputs/checkpoint/stage2_multitask_full）
#   SIMULS2ST_SHARED_OUTPUT      1 = OUTPUT_ROOT 在多节点共享存储上（不再加 --save_on_each_node）
#   SIMULS2ST_EXTRA_ARGS         追加给 swift 的参数（空格分隔）
#   RESUME                       续训的 checkpoint-N（NVMe 路径或 s3://）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVME_MOUNT="${NVME_MOUNT:-/nvme}"
: "${SIMULS2ST_STAGE:?需要 SIMULS2ST_STAGE（stage1 | stage2 | stage3）}"

log() { echo "[simuls2st/ktrain] $*" >&2; }
die() { echo "[simuls2st/ktrain][FATAL] $*" >&2; exit 1; }

# 与 launch_train.sh / derive_sync_jobs.py 同一规则：s3://<bucket>/<key> -> ${NVME_MOUNT}/<key>；fsx://p -> ${FSX_MOUNT}/p
to_local() {
    local v="$1" rest
    case "${v}" in
        s3://*) rest="${v#s3://}"; rest="${rest#*/}"; v="${NVME_MOUNT%/}/${rest}" ;;
        fsx://*) v="${FSX_MOUNT:-/fsx}/${v#fsx://}" ;;
    esac
    while [[ "${v}" == *"//"* ]]; do v="${v//\/\///}"; done
    [[ ${#v} -gt 1 ]] && v="${v%/}"
    echo "${v}"
}

MULTI_NODE=false
[[ "${NNODES:-1}" -gt 1 ]] && MULTI_NODE=true

# ---- 1. 软链目录树 ----
: "${MODEL_PATH:?需要 MODEL_PATH（offline 基座）}"
MODEL_PATH="$(to_local "${MODEL_PATH}")"
[[ -f "${MODEL_PATH}/config.json" ]] || die "MODEL_PATH 下没有 config.json: ${MODEL_PATH}（同步 Job 未完成或路径不对）"
OUTPUT_ROOT="$(to_local "${OUTPUT_ROOT:-${NVME_MOUNT%/}/output/simuls2st_omni}")"
mkdir -p "${OUTPUT_ROOT}"

maps=("models/SimulS2ST-Omni/offline ${MODEL_PATH}" "outputs ${OUTPUT_ROOT}")
while IFS='=' read -r name _; do
    [[ "${name}" == SIMULS2ST_MAP_* ]] || continue
    maps+=("${!name}")
done < <(env | sort)

tag="$(printf '%s\n' "${maps[@]}" | cksum | cut -d' ' -f1)"
ROOT="${NVME_MOUNT%/}/_simuls2st_root/${tag}"
mkdir -p "${ROOT}"
exec 7>"${ROOT}.lock"
flock 7
for m in "${maps[@]}"; do
    read -r rel target extra <<< "${m}"
    [[ -n "${rel}" && -n "${target}" && -z "${extra:-}" ]] || die "映射格式应为 \"<相对路径> <目标>\"，收到: ${m}"
    [[ "${rel}" != /* && "${rel}" != *..* ]] || die "映射的相对路径不能是绝对路径或含 ..: ${rel}"
    local_target="$(to_local "${target}")"
    [[ -e "${local_target}" ]] || die "映射目标不存在: ${rel} -> ${local_target}（同步 Job 未完成？）"
    mkdir -p "$(dirname "${ROOT}/${rel}")"
    ln -sfn "${local_target}" "${ROOT}/${rel}"
    log "map ${rel} -> ${local_target}"
done
flock -u 7
export SIMULS2ST_ROOT="${ROOT}" MODEL_PATH

# ---- 2. 阶段交接 ----
[[ -n "${SIMULS2ST_INIT_TALKER_FROM:-}" ]] && export SIMULS2ST_INIT_TALKER_FROM="$(to_local "${SIMULS2ST_INIT_TALKER_FROM}")"
[[ -n "${SIMULS2ST_MERGE_ADAPTERS_FROM:-}" ]] && export SIMULS2ST_MERGE_ADAPTERS_FROM="$(to_local "${SIMULS2ST_MERGE_ADAPTERS_FROM}")"
[[ -n "${RESUME:-}" ]] && export RESUME="$(to_local "${RESUME}")"

# ---- 3. 启动 ----
case "${SIMULS2ST_STAGE}" in
    stage1|stage2|stage3) ;;
    *) die "未知 SIMULS2ST_STAGE=${SIMULS2ST_STAGE}（stage1 | stage2 | stage3）" ;;
esac
extra=()
# checkpoint 落在各节点本地 NVMe：每个节点都存一份，续训和下一阶段在同一组节点上才读得到
if [[ "${MULTI_NODE}" == true && "${SIMULS2ST_SHARED_OUTPUT:-0}" != 1 ]]; then
    extra+=(--save_on_each_node true)
fi
# shellcheck disable=SC2206
[[ -n "${SIMULS2ST_EXTRA_ARGS:-}" ]] && extra+=(${SIMULS2ST_EXTRA_ARGS})
log "stage=${SIMULS2ST_STAGE} nodes=${NNODES:-1} node_rank=${NODE_RANK:-0} root=${SIMULS2ST_ROOT}"
exec bash "${SCRIPT_DIR}/train_${SIMULS2ST_STAGE}.sh" ${extra[@]+"${extra[@]}"}
