#!/usr/bin/env bash
# =============================================================================
# KtrainJob 容器入口（H200 / speech-h200-eks）。纯基础设施接线，不做 aws / s3 sync：
# 代码、模型、tokenizer、manifest、音频都已在节点就绪——
#   * 代码：经 S3 CSI 挂载 /s3all 直接读；TRAIN_SCRIPT 在 yaml 中为 repo 相对路径，
#     由本脚本按 CODE_PREFIX 拼出 /s3all/<CODE_PREFIX>/<TRAIN_SCRIPT> 后执行；
#   * OUTPUT_ROOT 落在 /nvme（NVMe hostPath），不转换；
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
REPO_ROOT_CANDIDATE="/s3all/${CODE_PREFIX}"
if [[ ! -d "${REPO_ROOT_CANDIDATE}" ]]; then
  REPO_ROOT_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
# nsys ray-actor 早期 hook：sitecustomize.py 靠 CPython `site` 模块在解释器
# 启动时自动 import，早于 `import swift` / `--external_plugins`，是唯一
# 能抢在 RayHelper.worker() 的 new_init 调用 _create_workers() 之前打
# ray.actor.ActorClass.options() 补丁的时机（完整时序证据见
# custom/ray_nsight_patch.py 头部文档）。未设 NSYS_PROFILE_STEP / USE_RAY
# 时该文件是完全无操作的 no-op，不影响其他任务（含本脚本原生的 WDS 路径）。
export PYTHONPATH="${REPO_ROOT_CANDIDATE}/ktrain/nsys_early_hook:${PYTHONPATH:-}"

# 路径归一化：折叠重复斜杠（yaml 里写成 s3://bucket//data/x 时不能拼出
# /nvme/data//x —— 与 sync 落地的目标路径对不上会静默读不到文件）。
# 只折叠斜杠，不做 .. 解析（工件路径里不应出现 ..）。
normalize_path() {
  local p="$1"
  # 连续斜杠 -> 单斜杠
  while [[ "${p}" == *"//"* ]]; do p="${p//\/\///}"; done
  # 去掉结尾斜杠（根 "/" 除外）
  [[ ${#p} -gt 1 ]] && p="${p%/}"
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
if [[ -n "${QWEN3_TTS_SPEAKER_ENCODER:-}" ]]; then
  export QWEN3_TTS_SPEAKER_ENCODER="$(artifact_to_path "${QWEN3_TTS_SPEAKER_ENCODER}")"
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

# unist 音频软链：原 pytorch_job 版用 initContainer 建，ktrain operator 不支持
# 自定义 init_containers（见 _capabilities.py），故在 launch_train.sh 里建。
# wds://unist/audio/processed/shards/dataset=unist/process_version=unist_bicodec_qwen3_v2
# -> /unist/audio/processed/shards/dataset=unist/process_version=unist_bicodec_qwen3_v2
# -> 软链到真实音频目录。
# 音频来源分两种（与 yaml 里 ARTIFACT_AUDIO_SHARDS 的 scheme 对应）：
#   * s3://  —— 经 sync Job 同步到 NVMe，落盘根 = artifact_to_path(ARTIFACT_AUDIO_SHARDS)
#             （已在上方把 ARTIFACT_AUDIO_SHARDS 改写为该根），真实目录 =
#             <根>/converted/unist_qwen3_s2s_v2/speech/unist_bicodec_qwen3_v2；
#   * fsx:// —— FSX CSI 直读，真实目录 = ${FSX_MOUNT}/zeyangsong/unist/converted/.../unist_bicodec_qwen3_v2。
# 容器内路径始终 = jsonl 中的 wds 路径（/unist/audio/.../process_version=unist_bicodec_qwen3_v2），
# 由软链把该路径指向真实目录实现。幂等（软链已存在则不动）。
UNIST_AUDIO_REL="converted/unist_qwen3_s2s_v2/speech/unist_bicodec_qwen3_v2"
UNIST_LINK_DIR="/unist/audio/processed/shards/dataset=unist"
UNIST_LINK_TARGET="${UNIST_LINK_DIR}/process_version=unist_bicodec_qwen3_v2"
UNIST_AUDIO_SRC=""
if [[ -n "${ARTIFACT_AUDIO_SHARDS:-}" ]]; then
  if [[ "${ARTIFACT_AUDIO_SHARDS}" == "${NVME_MOUNT%/}"/* ]]; then
    # s3:// 同步到 NVMe：ARTIFACT_AUDIO_SHARDS 已是 NVMe 落盘根，补上音频相对子目录
    UNIST_AUDIO_SRC="${ARTIFACT_AUDIO_SHARDS%/}/${UNIST_AUDIO_REL}"
  else
    # fsx:// 等：URI 本身已指向音频目录，直接用作软链目标
    UNIST_AUDIO_SRC="${ARTIFACT_AUDIO_SHARDS%/}"
  fi
elif [[ -n "${FSX_MOUNT:-}" ]]; then
  UNIST_AUDIO_SRC="${FSX_MOUNT%/}/zeyangsong/unist/${UNIST_AUDIO_REL}"
fi
if [[ -n "${UNIST_AUDIO_SRC}" && -d "${UNIST_AUDIO_SRC}" && ! -e "${UNIST_LINK_TARGET}" ]]; then
  mkdir -p "${UNIST_LINK_DIR}"
  ln -sfn "${UNIST_AUDIO_SRC}" "${UNIST_LINK_TARGET}"
  echo "[launch] created unist audio symlink: ${UNIST_LINK_TARGET} -> ${UNIST_AUDIO_SRC}"
fi

# EFA NCCL 插件符号链接（H200 节点）：aws-ofi-nccl 插件名 libnccl-net-ofi.so，
# NCCL 找 libnccl-net.so -> 不建符号链接会回退 TCP Socket。
if [[ -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so && ! -e /opt/amazon/ofi-nccl/lib/libnccl-net.so ]]; then
  ln -sfn libnccl-net-ofi.so /opt/amazon/ofi-nccl/lib/libnccl-net.so
fi

echo "[launch] NVME_MOUNT=${NVME_MOUNT} FSX_MOUNT=${FSX_MOUNT} S3ALL_MOUNT=${S3ALL_MOUNT}"
echo "[launch] MODEL_PATH=${MODEL_PATH}"
echo "[launch] QWEN3_TTS_TOKENIZER=${QWEN3_TTS_TOKENIZER}"
echo "[launch] BRIDGE_ALIGNMENT_DATA=${BRIDGE_ALIGNMENT_DATA}"

# 逐节点 prep 模式（ray_entrypoint.py 在每个 GPU worker 上调用）：
# 只做工件路径转换 + 文件副作用（软链/EFA 插件软链）后退出，不启动训练。
if [[ "${NODE_PREP_ONLY:-false}" == "true" ]]; then
  echo "[launch] NODE_PREP_ONLY done on $(hostname)"
  exit 0
fi

# 分布式 env（infra 层统一导出；多节点由 ktrain operator 注入 WORLD_SIZE/RANK）
# _nproc 同时供 ray 分支做引导计算——ray 模式下 NPROC_PER_NODE 不得 export
# （原因见下方 ray 分支注释），故先捕获为本地变量。
_nproc="${NPROC_PER_NODE:-8}"
export NPROC_PER_NODE="${_nproc}"

# runtimeEnv.pip 经 init container 以 `pip install --target /opt/ktrain-pip` 安装：
# 库经 PYTHONPATH=/opt/ktrain-pip 可 import，但 console scripts（ray CLI）落在
# /opt/ktrain-pip/bin，不在默认 PATH —— 存在即前置（无 runtimeEnv.pip 时目录不存在，不动）。
[[ -d /opt/ktrain-pip/bin ]] && export PATH="/opt/ktrain-pip/bin:${PATH}"

# ---------------------------------------------------------------------------
# 统一训练执行入口（支持 nsys 包裹）
# ---------------------------------------------------------------------------
# 所有最终执行训练脚本的地方（kuberay driver / ray head / torchrun）都走这里，
# 以便在需要时用 nsys profile 包裹训练进程，配合 custom/nsys_step_profile.py
# 的 cudaProfilerStart/Stop 只记录目标 step。
#   NSYS_PROFILE_STEP=<n>  在第 n 个 optimizer step 记录（代码里 Start/Stop）
#   NSYS_OUTPUT_DIR=<path>  报告落盘目录（默认 /nvme/output/nsys）
# 未设置 NSYS_PROFILE_STEP 时与原来完全一致（无 nsys 开销）。
#
# ray 模式（USE_RAY=true）例外：此处执行的是 kuberay driver 进程，训练实际发生
# 在 RayHelper._create_workers 创建的 actor 进程里，driver 从不调用
# cudaProfilerStart/Stop——包裹 driver 抓不到训练 kernel，还会因 ptrace 干扰
# actor fork（ActorUnschedulableError）。ray 模式改为让 custom/ray_nsight_patch.py
# 通过 ray runtime_env={"nsight":...} 把 nsys 包裹在 actor 自己的 worker 进程上，
# 此处对 ray 分支直接跳过 nsys 包裹（driver 侧零开销、零风险）。
_exec_train() {
    local _step="${NSYS_PROFILE_STEP:-0}"
    local _is_ray="false"
    [[ "${USE_RAY:-false}" == "true" || "${USE_RAY:-false}" == "1" ]] && _is_ray="true"
    if [[ "${_step}" -gt 0 && "${_is_ray}" == "false" ]] && command -v nsys >/dev/null 2>&1; then
        local _out="${NSYS_OUTPUT_DIR:-/nvme/output/nsys}"
        mkdir -p "${_out}"
        echo "[launch] wrapping train with nsys (profile step ${_step}, out ${_out})"
        # --capture-range=cudaProfilerApi：仅记录 cudaProfilerStart..Stop 窗口，
        # 训练全程不落 trace，只在目标 step 有采样（避免 nsys 常驻开销污染 MFU）。
        # -o 使用 rank 区分多进程/多节点，避免互相覆盖。
        # 注意 nsys 2024.6.x 的参数语法：
        #   * 进程采样用 --sample=process-tree（不是 --sample-process-tree）
        #   * --cuda-memory-usage 需显式 =true
        exec nsys profile \
            --capture-range=cudaProfilerApi \
            --trace=cuda,nvtx \
            --cuda-memory-usage=true \
            -o "${_out}/rank${RANK:-0}" \
            bash "${TRAIN_SCRIPT}" "$@"
    elif [[ "${_step}" -gt 0 && "${_is_ray}" == "true" ]]; then
        echo "[launch] ray mode: nsys wraps actor worker processes via" \
             "custom/ray_nsight_patch.py (runtime_env nsight plugin)," \
             "not this driver process — skipping driver-level nsys wrap."
    fi
    exec bash "${TRAIN_SCRIPT}" "$@"
}

# ---------------------------------------------------------------------------
# Ray 集群引导（USE_RAY=true，ms-swift `swift sft --use_ray` 多节点 deepspeed）
#   * RANK=0 pod：ray start --head，等集群 GPU 凑齐后跑训练脚本（driver 只在 head）；
#   * 其余 pod：ray start --address=<head> --block 常驻为纯 ray worker，不跑训练脚本。
#   * 反模式防御：USE_RAY 下禁止把 torchrun 分布式 env（NNODES/NODE_RANK/MASTER_*/
#     NPROC_PER_NODE/PET_*）带进 driver 与 actor env——ms-swift 的 use_torchrun()
#     只看 NPROC_PER_NODE/NNODES 是否存在（swift/cli/main.py:30），套 torchrun 会
#     起 8 个 driver 重复建 placement group；torchrun 的 action=env 还会读 operator
#     注入的 PET_NNODES=2 → static rendezvous 永远等不到第二个 agent（oh33 实锤）。
#   * ray 由任务 yaml 的 runtimeEnv.pip 注入（镜像本身不预装 ray）。
# ---------------------------------------------------------------------------
if [[ "${USE_RAY:-false}" == "true" || "${USE_RAY:-false}" == "1" ]]; then
  # ---- KubeRay（framework: ray）模式：集群已由 RayJob operator 建好 ----
  # 本进程是经 ray_entrypoint.py 钉到 GPU worker 的单 driver，职责只剩 env 清理：
  #   * NPROC_PER_NODE 必须 unset（swift/cli/main.py:30 use_torchrun 只看其存在，
  #     套 torchrun 会起 8 个 driver 重复建 placement group）；
  #   * torchrun 系 env 一律剥离（RayHelper._create_workers 会把 driver env 副本
  #     注入每个 actor；KubeRay pod 无 PET_*/MASTER_* 注入，此处为防御）。
  # RAY_DEVICE_GROUPS/RAY_WORLD_SIZE 已由 ray_entrypoint.py 按活集群派生注入。
  if [[ "${RAY_CLUSTER_MODE:-}" == "kuberay" ]]; then
    unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT RANK WORLD_SIZE \
          LOCAL_RANK LOCAL_WORLD_SIZE TORCHELASTIC_RUN_ID || true
    for _v in $(env | sed -n 's/^\(PET_[A-Za-z_0-9]*\)=.*$/\1/p'); do unset "${_v}"; done
    # 把本 driver 进程自己可见的 GPU 限制为 1 张（而不是设 LOCAL_WORLD_SIZE）：
    # 本进程因 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 默认能看到本机全部
    # GPU，且 LOCAL_WORLD_SIZE 未设时 swift/utils/env.py::is_mp() 取默认值 1，
    # 会算出 n_gpu(8)//local_world_size(1)=8>=2 → 误判为 "device_map 多卡模式"，
    # 与 use_liger_kernel 冲突直接 raise ValueError（liger_kernel does not
    # support device_map）——与 USE_LIGER_KERNEL 本身无关，纯粹是该判定式在
    # ray driver 单进程场景下的误判。
    # 注意：不能改用 export LOCAL_WORLD_SIZE=8 来修（曾踩坑）：RayHelper.
    # _create_workers() 用 os.environ.copy() 把 driver 的环境变量整份带给每个
    # actor，只覆盖 CUDA_VISIBLE_DEVICES（改成 actor 自己的单卡），不会覆盖
    # LOCAL_WORLD_SIZE——泄漏后每个 actor 变成 n_gpu=1 但 LOCAL_WORLD_SIZE=8，
    # is_mp() 内部 `assert n_gpu % local_world_size == 0` 直接 AssertionError，
    # 16 个 actor 全部创建失败（ActorDiedError）。
    # 改为限制 driver 自己的 CUDA_VISIBLE_DEVICES=0（只看 1 张卡）：
    # is_mp() 算出 1//1=1<2 → 不误判；且 CUDA_VISIBLE_DEVICES 本就会被
    # _create_workers 逐 actor 覆盖，不会泄漏影响训练 actor 的真实卡分配。
    export CUDA_VISIBLE_DEVICES="0"
    echo "[launch] kuberay mode: single driver on $(hostname)"
    echo "[launch] RAY_DEVICE_GROUPS=${RAY_DEVICE_GROUPS:-<unset>}"
    echo "[launch] driver CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (fix for is_mp() liger/device_map false-positive; actors get their own via RayHelper)"
    _exec_train "$@"
  fi

  RAY_PORT="${RAY_PORT:-6379}"
  _nnodes="${NNODES:-${WORLD_SIZE:-1}}"
  _rank="${NODE_RANK:-${RANK:-0}}"
  _head="${MASTER_ADDR:-127.0.0.1}"
  _world=$(( _nnodes * _nproc ))
  export RAY_ADDRESS="${_head}:${RAY_PORT}"
  export RAY_EXP_NAME="${RAY_EXP_NAME:-${WANDB_EXP_NAME:-bridge_alignment_ray}}"
  # 供训练脚本展示用（ms-swift 不读它）；anyscale dispatch_train.py 白名单同款变量。
  export RAY_WORLD_SIZE="${RAY_WORLD_SIZE:-${_world}}"
  # GPU pinning 由 RayHelper 经 runtime_env 显式注入 CUDA_VISIBLE_DEVICES，
  # 关闭 ray 自身的覆写，避免 fractional-GPU actor 被 ray 指派到错误卡（anyscale 验证）。
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
  # 首跑保留全部 actor 日志（不去重），便于双节点排障。
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
  # device_groups：全集群 _world 个 GPU、每节点 _nproc 个（放置粒度），
  # 单一 default 组覆盖全部 rank（与 anyscale dispatch_train.py 派生格式一致）。
  if [[ -z "${RAY_DEVICE_GROUPS:-}" ]]; then
    _ranks="$(seq -s, 0 $(( _world - 1 )))"
    export RAY_DEVICE_GROUPS="{\"nproc_per_node\":${_nproc},\"default\":{\"device\":\"GPU\",\"ranks\":[${_ranks}],\"workers\":[\"default\"]}}"
  fi
  echo "[launch] USE_RAY=1 nodes=${_nnodes} gpus/node=${_nproc} world=${_world}"
  echo "[launch] RAY_ADDRESS=${RAY_ADDRESS} RAY_EXP_NAME=${RAY_EXP_NAME}"
  echo "[launch] RAY_DEVICE_GROUPS=${RAY_DEVICE_GROUPS}"
  # torchrun 分布式 env 剥离（必须在 ray start 之前，否则 raylet 把它们带进所有 actor）：
  #   * NNODES/NODE_RANK/MASTER_*/RANK/WORLD_SIZE：ms-swift 误判 use_torchrun →
  #     TORCHELASTIC TCPStore hang（anyscale/deadlock_analysis.md §5.2）；
  #   * NPROC_PER_NODE：swift/cli/main.py:30 的 use_torchrun() 只看它是否存在——
  #     ray 模式必须单 driver 进程直跑 sft.py（RayHelper._create_workers 起 16 actors）；
  #     若套 torchrun 起 8 个本地 driver，每个都建 16-GPU placement group，7 个永远
  #     pending（continue_pretrain/common_args.sh:154 同款 guard 的设计依据）；
  #   * PET_*：ktrain operator 注入 PET_NNODES=2/PET_NODE_RANK/PET_MASTER_*，
  #     torchrun 参数 action=env 会读它们 → static rendezvous 等第 2 个 agent
  #     → TCPStore '1/2 clients joined' 900s 超时（oh33 实锤）。
  unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT RANK WORLD_SIZE \
        LOCAL_RANK LOCAL_WORLD_SIZE TORCHELASTIC_RUN_ID || true
  for _v in $(env | sed -n 's/^\(PET_[A-Za-z_0-9]*\)=.*$/\1/p'); do unset "${_v}"; done
  if [[ "${_rank}" == "0" ]]; then
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --port="${RAY_PORT}" --num-gpus="${_nproc}" \
              --disable-usage-stats --include-dashboard=false
    # 等全部 worker 加入（placement group 需要 _world 个 GPU，缺卡会永久 pending）
    python - "${RAY_ADDRESS}" "${_world}" <<'PY'
import sys, time
import ray
addr, want = sys.argv[1], int(sys.argv[2])
ray.init(address=addr, ignore_reinit_error=True)
deadline = time.time() + 1800
while True:
    got = int(ray.cluster_resources().get("GPU", 0))
    if got >= want:
        print(f"[launch] ray cluster ready: {got}/{want} GPUs", flush=True)
        break
    if time.time() > deadline:
        print(f"[launch][FATAL] ray cluster only {got}/{want} GPUs after 1800s",
              file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"[launch] waiting ray cluster GPUs: {got}/{want}", flush=True)
    time.sleep(15)
ray.shutdown()
PY
    _exec_train "$@"
  else
    echo "[launch] ray worker: joining ${RAY_ADDRESS}"
    _attempt=0
    # 注意：--include-dashboard 是 head-only 参数，worker 侧传了会被 ray CLI 直接拒绝
    until ray start --address="${RAY_ADDRESS}" --num-gpus="${_nproc}" \
                    --block --disable-usage-stats; do
        _attempt=$(( _attempt + 1 ))
        if [[ "${_attempt}" -ge 120 ]]; then
            echo "[launch][FATAL] cannot join ray head ${RAY_ADDRESS} after 120 attempts" >&2
            exit 1
        fi
        echo "[launch] ray head not ready, retry ${_attempt} in 5s"
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    done
    # ray start --block 正常常驻不返回；返回即 raylet 退出，pod 随之结束
    exit 0
  fi
fi

export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

_exec_train "$@"
