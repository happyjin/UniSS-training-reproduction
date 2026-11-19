#!/usr/bin/env bash
# =============================================================================
# KtrainJob 容器入口 — Lance 数据源版（新建，不改 launch_train.sh）
#
# 与 launch_train.sh 的差异：
#   * 额外把 LANCE_TRAIN_SOURCE_URI 从 s3:// / s3mnt:// / fsx:// 转成容器内路径
#   * 不再依赖 ARTIFACT_AUDIO_SHARDS / WDS tar / /curated_audio / unist 软链
#   * 训练读：本地 NVMe 上的 processed.lance（source bytes + codes）
#            + Shape A: BRIDGE_ALIGNMENT_DATA → lance:// 指针 JSONL
#            + Shape B: LANCE_TRAIN_SHAPE=B → 启动时从 Lance 生成临时指针 JSONL
#
# 工件 URI scheme（与 ktrain-job-ops/derive_sync_jobs.py 一致）：
#   s3://<bucket>/<key>  → submit 预热到 ${NVME_MOUNT}/<key>
#   s3mnt://<key>        → 从 S3 CSI 增量同步到 ${NVME_MOUNT}/<key>
#   fsx://<path>         → ${FSX_MOUNT}/<path>（不同步）
#   /abs/path            → 原样透传
#
# 提交示例：
#   bash .codebuddy/skills/ktrain-job-ops/scripts/submit.sh \
#     --yaml ktrain/ktrainjob_sts_translation_lance.yaml \
#     --repo <repo> \
#     --launcher ktrain/launch_train_lance.sh
# =============================================================================
set -euo pipefail

NVME_MOUNT="${NVME_MOUNT:-/nvme}"
FSX_MOUNT="${FSX_MOUNT:-/fsx}"
S3ALL_MOUNT="${S3ALL_MOUNT:-/s3all}"

CODE_PREFIX="${CODE_PREFIX:-deferzhou/liveinterpret_training_pipelines}"
if [[ "${TRAIN_SCRIPT}" != /* && "${TRAIN_SCRIPT}" != s3://* ]]; then
  TRAIN_SCRIPT="/s3all/${CODE_PREFIX}/${TRAIN_SCRIPT}"
fi

normalize_path() {
  local p="$1"
  while [[ "${p}" == *"//"* ]]; do p="${p//\/\///}"; done
  [[ ${#p} -gt 1 ]] && p="${p%/}"
  echo "${p}"
}

artifact_to_path() {
  local v="$1"
  case "${v}" in
    s3://*)
      local rest="${v#s3://}"; rest="${rest#*/}"
      normalize_path "${NVME_MOUNT}/${rest}"
      ;;
    s3mnt://*)
      normalize_path "${NVME_MOUNT}/${v#s3mnt://}"
      ;;
    fsx://*)
      normalize_path "${FSX_MOUNT}/${v#fsx://}"
      ;;
    *)
      echo "${v}"
      ;;
  esac
}

# ---- 模型 / tokenizer / manifest（与原版相同转换）----
export MODEL_PATH="$(artifact_to_path "${MODEL_PATH:-}")"
export QWEN3_TTS_TOKENIZER="$(artifact_to_path "${QWEN3_TTS_TOKENIZER:-}")"
export BRIDGE_ALIGNMENT_DATA="$(artifact_to_path "${BRIDGE_ALIGNMENT_DATA:-}")"
if [[ -n "${QWEN3_TTS_TOKENIZER_BASE:-}" ]]; then
  export QWEN3_TTS_TOKENIZER_BASE="$(artifact_to_path "${QWEN3_TTS_TOKENIZER_BASE}")"
fi
if [[ -n "${QWEN3_TTS_SPEAKER_ENCODER:-}" ]]; then
  export QWEN3_TTS_SPEAKER_ENCODER="$(artifact_to_path "${QWEN3_TTS_SPEAKER_ENCODER}")"
fi

# ---- Lance 数据源（本入口新增）----
# yaml 里写成 s3://.../processed.lance → sync Job 预热到 NVMe → 这里改成 /nvme/.../processed.lance
# custom/wds_audio.py 的 resolve_lance_* 读的就是这个 URI。
if [[ -n "${LANCE_TRAIN_SOURCE_URI:-}" ]]; then
  export LANCE_TRAIN_SOURCE_URI="$(artifact_to_path "${LANCE_TRAIN_SOURCE_URI}")"
fi
# 可选：pin 版本（纯数字字符串，不做路径转换）
if [[ -n "${LANCE_TRAIN_SOURCE_VERSION:-}" ]]; then
  export LANCE_TRAIN_SOURCE_VERSION
fi
# 可选：TaskSpec filter（Shape B / make_from_lance 共用；launch 原样透传）
if [[ -n "${LANCE_TRAIN_TASK_FILTER:-}" ]]; then
  export LANCE_TRAIN_TASK_FILTER
fi
if [[ -n "${LANCE_TRAIN_TASK_NAME:-}" ]]; then
  export LANCE_TRAIN_TASK_NAME
fi
# 指针回退根（本份 test_full 数据抽查全内联，一般不需要；保留钩子）
if [[ -n "${LANCE_WDS_DATA_ROOT:-}" ]]; then
  export LANCE_WDS_DATA_ROOT="$(artifact_to_path "${LANCE_WDS_DATA_ROOT}")"
fi

# Shape B：ms-swift 当前会把 ``*.py`` 识别成普通 Python 数据文件，不能可靠地
# 加载 HF loading script。因此启动时从 Lance 生成轻量 pointer JSONL；音频、
# codec 和 mel 仍全部从 Lance 读取，不复制训练 payload。
REPO_ROOT_CANDIDATE="/s3all/${CODE_PREFIX}"
if [[ ! -d "${REPO_ROOT_CANDIDATE}" ]]; then
  REPO_ROOT_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
_shape="$(echo "${LANCE_TRAIN_SHAPE:-A}" | tr '[:lower:]' '[:upper:]')"
export LANCE_TRAIN_REQUESTED_SHAPE="${_shape}"
if [[ "${_shape}" == "B" ]]; then
  export _MATERIALIZE_LANCE_MANIFEST=1
elif [[ "${BRIDGE_ALIGNMENT_DATA:-}" == *.py ]]; then
  export LANCE_TRAIN_REQUESTED_SHAPE=B
  export _MATERIALIZE_LANCE_MANIFEST=1
else
  export LANCE_TRAIN_SHAPE="${LANCE_TRAIN_SHAPE:-A}"
fi
export LANCE_TRAIN_MAX_ROWS="${LANCE_TRAIN_MAX_ROWS:-}"
export LANCE_PREFETCH_AUDIO="${LANCE_PREFETCH_AUDIO:-1}"
export LANCE_USE_PRECOMPUTED_MEL="${LANCE_USE_PRECOMPUTED_MEL:-}"
export PYTHONPATH="${REPO_ROOT_CANDIDATE}/vendor:${REPO_ROOT_CANDIDATE}:${PYTHONPATH:-}"
# nsys ray-actor 早期 hook：sitecustomize.py 靠 CPython `site` 模块在解释器
# 启动时自动 import，早于 `import swift` / `--external_plugins`，是唯一
# 能抢在 RayHelper.worker() 的 new_init 调用 _create_workers() 之前打
# ray.actor.ActorClass.options() 补丁的时机（完整时序证据见
# custom/ray_nsight_patch.py 头部文档）。追加而非替换 PYTHONPATH——多个
# sitecustomize.py 只有排在最前的目录里那份会被 site 模块实际 import
# （同名模块只导入一次），此处放最前确保生效。未设 NSYS_PROFILE_STEP /
# USE_RAY 时该文件是完全无操作的 no-op，不影响任何其他任务。
export PYTHONPATH="${REPO_ROOT_CANDIDATE}/ktrain/nsys_early_hook:${PYTHONPATH}"
# 默认不设 DATALOADER_MP_CONTEXT → DataLoader 用 PyTorch 默认 fork（与 B / m0cc 对齐）。
# forkserver/spawn 会 pickle DeepSpeed hook，F 列 h4a4 已验证不可用。
if [[ -n "${DATALOADER_MP_CONTEXT:-}" ]]; then
  echo "[launch-lance] DATALOADER_MP_CONTEXT=${DATALOADER_MP_CONTEXT}"
fi

# H200 训练镜像默认没有 pylance；离线 wheel 随 CODE_PREFIX 同步（与 g6e launcher 同款）。
if ! python -c 'import lance' 2>/dev/null; then
  echo "[launch-lance] lance 缺失，安装 pylance ..."
  if compgen -G "${REPO_ROOT_CANDIDATE}/wheels/*.whl" >/dev/null; then
    python -m pip install -q --no-cache-dir --no-index \
      --find-links "${REPO_ROOT_CANDIDATE}/wheels" pylance \
      || echo "[launch-lance][WARN] 离线 wheel 安装失败，回退 PyPI"
  fi
  python -c 'import lance' 2>/dev/null \
    || python -m pip install --no-cache-dir pylance 2>&1 | tail -5
fi
if ! python -c 'import lance' 2>/dev/null; then
  echo "[launch-lance][FATAL] pylance 不可用，无法物化 pointer JSONL / 读 Lance" >&2
  exit 1
fi

export FSX_MOUNT S3ALL_MOUNT

# EFA NCCL 插件（与原版相同）
if [[ -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so && ! -e /opt/amazon/ofi-nccl/lib/libnccl-net.so ]]; then
  ln -sfn libnccl-net-ofi.so /opt/amazon/ofi-nccl/lib/libnccl-net.so
fi

# Materialize the lightweight row-pointer JSONL on **every** node (including
# Ray NODE_PREP). Original D/E/F/G ranks each read BRIDGE_ALIGNMENT_DATA from
# a shared FSX glob; after the data-source swap that path is local NVMe, so
# each node must have its own copy. Skip if a previous launch already wrote it.
if [[ "${_MATERIALIZE_LANCE_MANIFEST:-0}" == "1" ]]; then
  MANIFEST_DIR="${NVME_MOUNT}/manifests"
  MANIFEST_OUT="${MANIFEST_DIR}/lance_${LANCE_TRAIN_TASK_NAME:-s2st}_v${LANCE_TRAIN_SOURCE_VERSION:-latest}.jsonl"
  _mat_t0="$(date +%s)"
  if [[ -s "${MANIFEST_OUT}" ]]; then
    echo "[launch-lance] reuse pointer manifest ${MANIFEST_OUT}"
    export LANCE_MANIFEST_REUSED=1
  else
    MANIFEST_TMP="${MANIFEST_OUT}.tmp.$$"
    mkdir -p "${MANIFEST_DIR}"
    trap 'rm -f "${MANIFEST_TMP:-}"' EXIT
    export MANIFEST_TMP
    python - <<'PY'
import os
from custom.lance_manifest import materialize_from_env

materialize_from_env(os.environ['MANIFEST_TMP'])
PY
    mv -f "${MANIFEST_TMP}" "${MANIFEST_OUT}"
    trap - EXIT
    export LANCE_MANIFEST_REUSED=0
  fi
  _mat_elapsed="$(( $(date +%s) - _mat_t0 ))"
  echo "[launch-lance] materialize elapsed_s=${_mat_elapsed} reused=${LANCE_MANIFEST_REUSED} path=${MANIFEST_OUT} bytes=$(wc -c < "${MANIFEST_OUT}")"
  export LANCE_MATERIALIZE_ELAPSED_S="${_mat_elapsed}"
  export BRIDGE_ALIGNMENT_DATA="${MANIFEST_OUT}"
  export LANCE_TRAIN_SHAPE=A
  export WDS_AUDIO_MODE="${WDS_AUDIO_MODE:-auto}"
fi

# 逐节点 prep（ray_entrypoint.py 在每个 GPU worker 上调用）：路径转换 +
# pointer JSONL + EFA 软链，不启动训练。
if [[ "${NODE_PREP_ONLY:-false}" == "true" ]]; then
  echo "[launch-lance] NODE_PREP_ONLY done on $(hostname) uri=${LANCE_TRAIN_SOURCE_URI:-<unset>} manifest=${BRIDGE_ALIGNMENT_DATA:-<unset>}"
  exit 0
fi

echo "[launch-lance] NVME_MOUNT=${NVME_MOUNT} FSX_MOUNT=${FSX_MOUNT} S3ALL_MOUNT=${S3ALL_MOUNT}"
echo "[launch-lance] MODEL_PATH=${MODEL_PATH}"
echo "[launch-lance] QWEN3_TTS_TOKENIZER=${QWEN3_TTS_TOKENIZER}"
echo "[launch-lance] LANCE_TRAIN_REQUESTED_SHAPE=${LANCE_TRAIN_REQUESTED_SHAPE}"
echo "[launch-lance] LANCE_TRAIN_SHAPE=${LANCE_TRAIN_SHAPE}"
echo "[launch-lance] BRIDGE_ALIGNMENT_DATA=${BRIDGE_ALIGNMENT_DATA}"
echo "[launch-lance] LANCE_TRAIN_SOURCE_URI=${LANCE_TRAIN_SOURCE_URI:-<unset>}"
echo "[launch-lance] LANCE_TRAIN_SOURCE_VERSION=${LANCE_TRAIN_SOURCE_VERSION:-<latest>}"
echo "[launch-lance] LANCE_TRAIN_TASK_FILTER=${LANCE_TRAIN_TASK_FILTER:-<default>}"
echo "[launch-lance] LANCE_USE_PRECOMPUTED_MEL=${LANCE_USE_PRECOMPUTED_MEL:-<schema>}"

# 启动前轻量门禁：Lance / manifest(or Shape B loader) 必须已在本地
if [[ -n "${LANCE_TRAIN_SOURCE_URI:-}" ]]; then
  if [[ ! -e "${LANCE_TRAIN_SOURCE_URI}" ]]; then
    echo "[launch-lance][FATAL] LANCE_TRAIN_SOURCE_URI 不存在: ${LANCE_TRAIN_SOURCE_URI}" >&2
    echo "[launch-lance][FATAL] 检查 submit 派生的 sync Job 是否已把 processed.lance 同步到本节点 NVMe" >&2
    exit 1
  fi
fi
if [[ -n "${BRIDGE_ALIGNMENT_DATA:-}" && ! -e "${BRIDGE_ALIGNMENT_DATA}" ]]; then
  echo "[launch-lance][FATAL] BRIDGE_ALIGNMENT_DATA 不存在: ${BRIDGE_ALIGNMENT_DATA}" >&2
  if [[ "${LANCE_TRAIN_SHAPE}" == "B" ]]; then
    echo "[launch-lance][FATAL] Shape B 需要 custom/lance_source_loader.py 随代码同步到本节点" >&2
  else
    echo "[launch-lance][FATAL] 需先 make_from_lance 产出 JSONL 并上传到对应 S3 前缀（再经 sync 落到 NVMe）" >&2
  fi
  exit 1
fi

# 下面分布式 / Ray 接线从 launch_train.sh 原样对齐：只换数据源，不改
# SCENE / batch / warmup。禁止传 --max_steps——原文 B–E 是 2 epochs +
# warmup_ratio=0.03（~2225 warmup），200 step 后人工 stop；传 max_steps=201
# 会把 warmup 压成 ~6 步，配置就不齐了。
[[ -d /opt/ktrain-pip/bin ]] && export PATH="/opt/ktrain-pip/bin:${PATH}"

_nproc="${NPROC_PER_NODE:-8}"
export NPROC_PER_NODE="${_nproc}"

if [[ "${USE_RAY:-false}" == "true" || "${USE_RAY:-false}" == "1" ]]; then
  if [[ "${RAY_CLUSTER_MODE:-}" == "kuberay" ]]; then
    unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT RANK WORLD_SIZE \
          LOCAL_RANK LOCAL_WORLD_SIZE TORCHELASTIC_RUN_ID || true
    for _v in $(env | sed -n 's/^\(PET_[A-Za-z_0-9]*\)=.*$/\1/p'); do unset "${_v}"; done
    echo "[launch-lance] kuberay mode: single driver on $(hostname)"
    echo "[launch-lance] RAY_DEVICE_GROUPS=${RAY_DEVICE_GROUPS:-<unset>}"
    exec bash "${TRAIN_SCRIPT}" "$@"
  fi

  RAY_PORT="${RAY_PORT:-6379}"
  _nnodes="${NNODES:-${WORLD_SIZE:-1}}"
  _rank="${NODE_RANK:-${RANK:-0}}"
  _head="${MASTER_ADDR:-127.0.0.1}"
  _world=$(( _nnodes * _nproc ))
  export RAY_ADDRESS="${_head}:${RAY_PORT}"
  export RAY_EXP_NAME="${RAY_EXP_NAME:-${WANDB_EXP_NAME:-bridge_alignment_ray}}"
  export RAY_WORLD_SIZE="${RAY_WORLD_SIZE:-${_world}}"
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
  if [[ -z "${RAY_DEVICE_GROUPS:-}" ]]; then
    _ranks="$(seq -s, 0 $(( _world - 1 )))"
    export RAY_DEVICE_GROUPS="{\"nproc_per_node\":${_nproc},\"default\":{\"device\":\"GPU\",\"ranks\":[${_ranks}],\"workers\":[\"default\"]}}"
  fi
  echo "[launch-lance] USE_RAY=1 nodes=${_nnodes} gpus/node=${_nproc} world=${_world}"
  unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT RANK WORLD_SIZE \
        LOCAL_RANK LOCAL_WORLD_SIZE TORCHELASTIC_RUN_ID || true
  for _v in $(env | sed -n 's/^\(PET_[A-Za-z_0-9]*\)=.*$/\1/p'); do unset "${_v}"; done
  if [[ "${_rank}" == "0" ]]; then
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --port="${RAY_PORT}" --num-gpus="${_nproc}" \
              --disable-usage-stats --include-dashboard=false
    python - "${RAY_ADDRESS}" "${_world}" <<'PY'
import sys, time
import ray
addr, want = sys.argv[1], int(sys.argv[2])
ray.init(address=addr, ignore_reinit_error=True)
deadline = time.time() + 1800
while True:
    got = int(ray.cluster_resources().get("GPU", 0))
    if got >= want:
        print(f"[launch-lance] ray cluster ready: {got}/{want} GPUs", flush=True)
        break
    if time.time() > deadline:
        print(f"[launch-lance][FATAL] ray cluster only {got}/{want} GPUs after 1800s",
              file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"[launch-lance] waiting ray cluster GPUs: {got}/{want}", flush=True)
    time.sleep(15)
ray.shutdown()
PY
    exec bash "${TRAIN_SCRIPT}" "$@"
  else
    echo "[launch-lance] ray worker: joining ${RAY_ADDRESS}"
    _attempt=0
    until ray start --address="${RAY_ADDRESS}" --num-gpus="${_nproc}" \
                    --block --disable-usage-stats; do
        _attempt=$(( _attempt + 1 ))
        if [[ "${_attempt}" -ge 120 ]]; then
            echo "[launch-lance][FATAL] cannot join ray head ${RAY_ADDRESS} after 120 attempts" >&2
            exit 1
        fi
        echo "[launch-lance] ray head not ready, retry ${_attempt} in 5s"
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    done
    exit 0
  fi
fi

export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 2026-09-16 补充：本文件当年是从 launch_train.sh"原样对齐"复制的分布式/
# Ray 接线（见上方 227 行注释），但复制时间点早于 launch_train.sh 加入 nsys
# 包裹功能（_exec_train()），一直漏了。非 ray（torchrun/pytorch）路径下，
# 这里补上同款逻辑：USE_RAY!=true 且 NSYS_PROFILE_STEP>0 时用
# `nsys profile ... -o ${NSYS_OUTPUT_DIR}/rank${RANK}` 包裹整个 torchrun
# 调用（含其 spawn 出的全部 local rank 子进程），配合
# custom/nsys_step_profile.py 的 cudaProfilerStart/Stop 只记录目标 step。
# 未设 NSYS_PROFILE_STEP 时与原来完全一致（无 nsys 开销）。详见
# launch_train.sh 同名函数的完整注释。
_step="${NSYS_PROFILE_STEP:-0}"
if [[ "${_step}" -gt 0 ]] && command -v nsys >/dev/null 2>&1; then
    _out="${NSYS_OUTPUT_DIR:-/nvme/output/nsys}"
    mkdir -p "${_out}"
    echo "[launch-lance] wrapping train with nsys (profile step ${_step}, out ${_out})"
    exec nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --trace=cuda,nvtx \
        --sample=none \
        --cuda-memory-usage=true \
        -o "${_out}/rank${RANK:-0}" \
        bash "${TRAIN_SCRIPT}" "$@"
fi
exec bash "${TRAIN_SCRIPT}" "$@"
