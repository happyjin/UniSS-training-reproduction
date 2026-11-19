#!/usr/bin/env bash
# =============================================================================
# common_args.sh — 统一的 bridge-alignment 训练参数（deepspeed + megatron）
#
# 通过 TRAIN_BACKEND 环境变量选择后端：
#   TRAIN_BACKEND=deepspeed  → swift sft + ZeRO-2（默认）
#   TRAIN_BACKEND=megatron    → megatron sft + mcore
#
# 两后端共享的参数（CLI 参数名一致）设一次；各自专属的参数（CLI 参数名不同）
# 在对应分支内设置，环境变量名跟随 CLI 参数名，不强行统一。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------
: "${TRAIN_BACKEND:=deepspeed}"
case "${TRAIN_BACKEND}" in
    deepspeed|megatron) ;;
    *) echo "[ERROR] TRAIN_BACKEND must be 'deepspeed' or 'megatron', got: ${TRAIN_BACKEND}" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 路径约定（k8s）：env 注入「去掉 S3 头的 /nvme/<s3-key>」。
# ---------------------------------------------------------------------------
# JSONL manifest 模式（默认，兼容既有任务）
export BRIDGE_ALIGNMENT_DATA="${BRIDGE_ALIGNMENT_DATA:-${NVME_MOUNT:-/}/data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train}"
export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-${NVME_MOUNT:-/}/origin_models/Qwen3-TTS-Tokenizer-12Hz}"
export BRIDGE_ALIGNMENT_CODEC_CACHE="${BRIDGE_ALIGNMENT_CODEC_CACHE:-${NVME_MOUNT:-/}/.cache/bridge_alignment_codec}"
export BRIDGE_ALIGNMENT_APPEND_TARGET_TEXT="${BRIDGE_ALIGNMENT_APPEND_TARGET_TEXT:-true}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${NVME_MOUNT:-/}/.cache/hf_modules_qwen_omnispeak_bridge_fix}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 缓存路径：必须显式落到 NVMe（ms-swift 的 jsonl→Arrow 缓存在 MODELSCOPE_CACHE）
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${NVME_MOUNT:-/}/.cache/hf_datasets}"
export DATASETS_CACHE="${DATASETS_CACHE:-${NVME_MOUNT:-/}/.cache/hf_datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${NVME_MOUNT:-/}/.cache/modelscope}"
export HF_HOME="${HF_HOME:-${NVME_MOUNT:-/}/.cache/huggingface}"

PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
: "${MODEL_PATH:=${NVME_MOUNT:-/}/origin_models/custom_models/qwen3_omnispeak_d}"
: "${MODEL_TYPE:=qwen3_omnispeak_d}"
TEMPLATE="${TEMPLATE:-qwen_omnispeak}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
: "${OUTPUT_ROOT:=${NVME_MOUNT:-/}/output/qwen_omnispeak_output}"
export OUTPUT_ROOT

# ---------------------------------------------------------------------------
# 本地 dev conda 兜底（两后端共享）
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-base}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-true}"
if [[ "${AUTO_ACTIVATE_CONDA}" == "true" ]]; then
    CONDA_SH="${CONDA_SH:-}"
    if [[ -z "${CONDA_SH}" ]]; then
        for candidate in \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/miniconda/etc/profile.d/conda.sh" \
            "/root/miniconda3/etc/profile.d/conda.sh" \
            "/opt/miniconda3/etc/profile.d/conda.sh"; do
            if [[ -f "${candidate}" ]]; then
                CONDA_SH="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
    else
        echo "[WARN] conda.sh not found; continuing with current shell environment." >&2
    fi
fi

# ---------------------------------------------------------------------------
# 容器感知 CPU 核数（cgroup v2/v1 配额）
# ---------------------------------------------------------------------------
_get_container_cpus() {
    local host_cpus quota period effective
    host_cpus="$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)"
    effective="${host_cpus}"
    if [[ -r /sys/fs/cgroup/cpu.max ]]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [[ -n "${quota}" && "${quota}" != "max" && -n "${period}" && "${period}" -gt 0 ]]; then
            effective="$(( (quota + period - 1) / period ))"
        fi
    elif [[ -r /sys/fs/cgroup/cpu/cpu.cpu_quota_us && -r /sys/fs/cgroup/cpu/cpu.cpu_period_us ]]; then
        quota="$(cat /sys/fs/cgroup/cpu/cpu.cpu_quota_us 2>/dev/null)"
        period="$(cat /sys/fs/cgroup/cpu/cpu.cpu_period_us 2>/dev/null)"
        if [[ -n "${quota}" && "${quota}" -gt 0 && -n "${period}" && "${period}" -gt 0 ]]; then
            effective="$(( (quota + period - 1) / period ))"
        fi
    fi
    [[ "${effective}" -gt "${host_cpus}" ]] && effective="${host_cpus}"
    echo "${effective}"
}

# ---------------------------------------------------------------------------
# 容器感知 GPU 卡数（优先 CUDA_VISIBLE_DEVICES，回退 torch / nvidia-smi，失败为 0）
# ---------------------------------------------------------------------------
_get_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
        local IFS=',' n=0 dev
        for dev in ${CUDA_VISIBLE_DEVICES}; do
            [[ -n "${dev}" ]] && (( ++n ))
        done
        if [[ "${n}" -gt 0 ]]; then
            echo "${n}"
            return 0
        fi
    fi
    local count
    count="$(python - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
        echo "${count}"
        return 0
    fi
    count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0)"
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
        echo "${count}"
        return 0
    fi
    echo 0
}

# ---------------------------------------------------------------------------
# WDS / prompt 归一化 / ICL（两后端共享）
# ---------------------------------------------------------------------------
: "${WDS_AUDIO_MODE:=auto}"
: "${WDS_AUDIO_STRICT:=true}"
: "${WDS_AUDIO_MAX_BYTES:=268435456}"
export WDS_AUDIO_MODE WDS_AUDIO_STRICT WDS_AUDIO_MAX_BYTES
: "${BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT:=false}"
export BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT
: "${QWEN_OMNISPEAK_USE_ICL:=false}"
export QWEN_OMNISPEAK_USE_ICL

# ---------------------------------------------------------------------------
# CPU codec tokenizer（两后端共享）
# ---------------------------------------------------------------------------
export QWEN3_TTS_TOKENIZER_DEVICE="${QWEN3_TTS_TOKENIZER_DEVICE:-cpu}"
# DATASET_NUM_PROC / DATALOADER_*（两后端 CLI 参数名一致）
CONTAINER_CPUS="$(_get_container_cpus)"
: "${DATASET_NUM_PROC:=${CONTAINER_CPUS}}"
# DATALOADER_NUM_WORKERS：优先按容器可用 CPU 核数动态设置；无法探测或无足够核数时回退到默认值。
# 动态：DATALOADER_NUM_WORKERS = floor(CONTAINER_CPUS / (2 × LOCAL_WORLD_SIZE))，不设下限。
#   约束：总 worker = DATALOADER_NUM_WORKERS × 2 × LOCAL_WORLD_SIZE（train + eval 两个 DataLoader；
#         --dataloader_persistent_workers true 下 train worker 常驻，eval 时二者并存）必须 <= 可用 CPU 核数，
#         否则 CPU 超配 → dataloader 进程互相争抢 → 吞吐反而下降。
#   不设下限 1：若可用 CPU 不足以支撑 1 个 worker（floor 计算 == 0），则保持 0 —— 即 dataloader 合入
#         训练进程（num_workers=0）、不额外 fork 子进程，这正是预期的默认回退行为。
#   回退默认值 = 0：探测不到 CONTAINER_CPUS / LOCAL_WORLD_SIZE 时，同样合入训练进程；
#         原硬编码 16 过大（在 135 核上本就超配），不作为默认。
#   变量优先级：显式 env DATALOADER_NUM_WORKERS（含 0）> 动态计算 > 回退 0。
#   实测印证（ktrain-8zom, CONTAINER_CPUS=135, LOCAL_WORLD_SIZE=8）：动态 8 → train=64/eval=64/总=128（<=135）。
if [[ -z "${DATALOADER_NUM_WORKERS:-}" ]]; then
    _lws="${LOCAL_WORLD_SIZE:-${NPROC_PER_NODE:-$(_get_gpu_count)}}"
    _lws="${_lws:-1}"
    if [[ "${CONTAINER_CPUS}" =~ ^[0-9]+$ && "${CONTAINER_CPUS}" -gt 0 \
          && "${_lws}" =~ ^[0-9]+$ && "${_lws}" -gt 0 ]]; then
        # 不强制下限：CPU 不够时分得 0，即合入训练进程、不 fork。
        DATALOADER_NUM_WORKERS="$(( CONTAINER_CPUS / (2 * _lws) ))"
    else
        DATALOADER_NUM_WORKERS=0
    fi
fi
: "${DATALOADER_PERSISTENT_WORKERS:=true}"
: "${DATALOADER_PREFETCH_FACTOR:=2}"
# lazy_tokenize：预处理阶段只解码音频并分片增量写 Arrow cache，tokenize 推迟到训练 dataloader 逐 batch 进行。
: "${LAZY_TOKENIZE:=true}"
: "${GROUP_BY_LENGTH:=true}"
# Lance / 19M-row path: ms-swift silently switches to eager AddLengthPreprocessor
# if group_by_length is on and lazy_tokenize is left unset. Force a loud fail
# instead of an OOM after the full-table encode.
if [[ "${GROUP_BY_LENGTH}" == "true" && "${LAZY_TOKENIZE}" != "true" ]]; then
    echo "ERROR: GROUP_BY_LENGTH=true requires LAZY_TOKENIZE=true (dataset.py bakes lengths from source_duration_ms)." >&2
    exit 1
fi


# ---------------------------------------------------------------------------
# 共享 Trainer 超参（两后端 CLI 参数名一致）
# ---------------------------------------------------------------------------
: "${NUM_TRAIN_EPOCHS:=1}"
: "${MAX_LENGTH:=2048}"
: "${EVAL_STEPS:=500}"
: "${SAVE_STEPS:=500}"
: "${LOGGING_STEPS:=10}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${SPLIT_DATASET_RATIO:=0.01}"



# ---------------------------------------------------------------------------
# 残差 loss（两后端共享）
# ---------------------------------------------------------------------------
: "${RESIDUAL_LOSS:=true}"
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
# 已设置的带前缀值优先（SCENE 预设直接写带前缀变量），无前缀仅作兜底，
# 防止无前缀默认把任务脚本已设好的带前缀值静默覆盖（2026-08-24 覆盖链事故）。
export QWEN_OMNISPEAK_RESIDUAL_LOSS="${QWEN_OMNISPEAK_RESIDUAL_LOSS:-${RESIDUAL_LOSS}}"
export QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT="${QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT:-${RESIDUAL_LOSS_WEIGHT}}"

# codec EOS loss / history corruption（两后端共享）
if [[ -n "${CODEC_EOS_LOSS_WEIGHT:-}" ]]; then
    export QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT="${CODEC_EOS_LOSS_WEIGHT}"
fi
if [[ -n "${CODEC_HISTORY_CORRUPTION_PROB:-}" ]]; then
    export QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB="${CODEC_HISTORY_CORRUPTION_PROB}"
fi

# ---------------------------------------------------------------------------
# wandb / report（两后端共享）
# ---------------------------------------------------------------------------
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_bridge_alignment}"
: "${WANDB_DIR:=${OUTPUT_ROOT}/wandb}"
: "${WANDB_WATCH:=false}"
: "${WANDB_LOG_MODEL:=false}"
export WANDB_PROJECT WANDB_DIR WANDB_WATCH WANDB_LOG_MODEL
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------------------------------------------------------------------------
# 分布式 env（两后端共享：torchrun launch）
# ---------------------------------------------------------------------------
if [[ "${USE_RAY:-false}" != "true" && "${USE_RAY:-false}" != "1" ]]; then
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi
USE_RAY="${USE_RAY:-false}"
RAY_EXP_NAME="${RAY_EXP_NAME:-}"
RAY_DEVICE_GROUPS="${RAY_DEVICE_GROUPS:-}"

# ===========================================================================
# DeepSpeed 后端专属参数（TRAIN_BACKEND=deepspeed）
# ===========================================================================
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    DEEPSPEED="${DEEPSPEED:-zero0}"

    # ZeRO-3 回退
    if [[ "${DEEPSPEED}" == "zero3" && "${ALLOW_BRIDGE_ZERO3:-false}" != "true" ]]; then
        echo "[WARN] Bridge alignment can hang under ZeRO-3; falling back to zero2." >&2
        DEEPSPEED="zero2"
    fi

    # ATTN_IMPL（swift sft --attn_impl）探测 FA3→FA2→sdpa
    ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
    if [[ "${ATTN_IMPL}" == "flash_attention_3" || "${ATTN_IMPL}" == "flash_attn_3" ]]; then
        if ! python - <<'PY' >/dev/null 2>&1
import importlib.util as u, sys
sys.exit(0 if (u.find_spec('flash_attn_3') and u.find_spec('flash_attn_interface')) else 1)
PY
        then
            echo "[WARN] flash_attn_3 not importable; trying flash_attention_2." >&2
            ATTN_IMPL="flash_attention_2"
        fi
    fi
    if [[ "${ATTN_IMPL}" == "flash_attention_2" || "${ATTN_IMPL}" == "flash_attn" ]]; then
        if ! python - <<'PY' >/dev/null 2>&1
import importlib.util as u, sys
sys.exit(0 if u.find_spec('flash_attn') else 1)
PY
        then
            echo "[WARN] flash_attn (FA2) unavailable; falling back to sdpa." >&2
            ATTN_IMPL="sdpa"
        fi
    fi

    # NPROC_PER_NODE 自动探测（复用顶部 _get_gpu_count：CUDA_VISIBLE_DEVICES → torch → nvidia-smi，失败为 0）
    # USE_RAY=true 时禁止设置：swift/cli/main.py:30 的 use_torchrun() 只看 NPROC_PER_NODE/NNODES
    # 是否存在，ray 模式必须单 driver 直跑（RayHelper 自建 16 actors），套 torchrun 会起
    # 8 个 driver 重复建 placement group（与 continue_pretrain/common_args.sh:154 同款 guard）。
    if [[ -z "${NPROC_PER_NODE:-}" && -n "${DEEPSPEED}" && "${DEEPSPEED}" != "none" \
          && "${USE_RAY}" != "true" && "${USE_RAY}" != "1" ]]; then
        GPU_COUNT="$(_get_gpu_count)"
        if [[ "${GPU_COUNT}" -gt 1 ]]; then
            export NPROC_PER_NODE="${GPU_COUNT}"
        fi
    fi

    # deepspeed 专属超参
    # effective_bs = PER_DEVICE_TRAIN_BATCH_SIZE × NPROC_PER_NODE × GRADIENT_ACCUMULATION_STEPS
    #              = 64 × 16 × 8 = 4096（对齐 megatron dense GBS=4096，两端有效批次一致）
    : "${PER_DEVICE_TRAIN_BATCH_SIZE:=64}"
    : "${GRADIENT_ACCUMULATION_STEPS:=8}"
    : "${PER_DEVICE_EVAL_BATCH_SIZE:=4}"
    : "${LEARNING_RATE:=1e-4}"
    : "${WARMUP_RATIO:=0.03}"
    : "${TALKER_TRAINING_SCOPE:=layers}"
    : "${UNFREEZE_TALKER_LAYERS:=2}"
    : "${UNFREEZE_THINKER:=false}"

    # 显存/速度优化：
    #   - GRADIENT_CHECKPOINTING：重计算（前向不存激活，反向重算）。省显存、约 +30% 计算耗时。
    #   - USE_LIGER_KERNEL：Triton 算子融合（RMSNorm/RoPE/SwiGLU/CrossEntropy），减显存+加速，无精度损失。
    #     注意：当前训练镜像未预装 liger-kernel 包，启用需先在镜像中 pip install liger-kernel。
    : "${GRADIENT_CHECKPOINTING:=true}"
    : "${USE_LIGER_KERNEL:=false}"
    export GRADIENT_CHECKPOINTING USE_LIGER_KERNEL

fi  # end deepspeed

# ===========================================================================
# Megatron 后端专属参数（TRAIN_BACKEND=megatron）
# ===========================================================================
if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
    # 禁用 torch.compile（MoE grouped-GEMM 兼容）
    export TORCH_COMPILE_DISABLE=1
    export NVTE_TORCH_COMPILE=0

    # SCENE 预设
    SCENE="${SCENE:-dense}"
    case "${SCENE}" in
        moe)  _TP=8;  _EP=1;  _GBS=16;   _MBS=8  ;;
        # MBS=16：adapter 模式只训 thinker_adapter（~16.8M Trainable），thinker/talker 冻结。
        # 实测依据：MBS=32 在 step 0 thinker forward (qwen3 q_proj) 即 OOM（138.71/139.80 GiB，
        # rank1，try alloc 1.49GiB 失败）——thinker 虽冻结但前向 attention 激活仍随 MBS 线性增长，
        # 与可训练参数量无关。MBS=8（58lk）正常。MBS=16 为折中（待实测验证显存余量）。
        # 整除校验：GBS=4096, MBS=16, DP=8 → 4096 % (16×8)=128 = 0 ✓ num_microbatches=32。
        dense) _TP=1;  _EP=1;  _GBS=4096; _MBS=16 ;;
        # batch4_8 场景（adapter + whole talker，thinker 冻结）：对齐 deepspeed 侧
        # SCENE 预设的有效 batch（GBS=512=64×8×1）。MBS 取 8：上方实测注释表明
        # MBS=32 即 OOM、MBS=16 在 adapter-only 下为折中；whole-talker 可训状态
        # 显存更高，选更保守的 MBS=8（代价是 MFU 略低）。
        # 整除校验：GBS=512, MBS=8, DP=8 → 512 % (8×8) = 0 ✓ num_microbatches=8。
        batch4_8_adapter|batch4_8_thinker|batch4_8_thinker_cosine|stage2_5_unist)
            _TP=1;  _EP=1;  _GBS=512;  _MBS=8 ;;
        # batch4_8_adapter_cmp（四方对照，2026-08-27）：adapter + whole talker、thinker 冻结，
        # MAX_LENGTH=576（batch4.8 家族）。对齐基准任务 A 的有效 batch：固定 GBS=512，
        # MBS=16（单/双节点统一）：
        #   DP=8（单机）→ num_microbatches=4；DP=16（双节点）→ 2。
        #   整除校验：512 % (16×8)=0 ✓、512 % (16×16)=0 ✓。
        # 显存：MBS=16 在 adapter-only（dense 场景注释）实测可行；whole-talker 可训状态
        # 显存更高，但本家族 MAX_LENGTH=576 远短于 dense 的 2048，dense attn bias
        # [B,1,T,T] 随 T 平方级缩小（≤576 vs ~9.5k），141 GiB H200 余量充足（本组实跑复核）。
        batch4_8_adapter_cmp)
            _TP=1;  _EP=1;  _GBS=512; _MBS=16 ;;
        *)
            echo "[ERROR] 未知 SCENE=${SCENE}（megatron 支持 dense/moe/batch4_8_adapter/batch4_8_thinker/batch4_8_thinker_cosine/stage2_5_unist/batch4_8_adapter_cmp）" >&2
            exit 1
            ;;
    esac

    # 模型并行
    TP="${TP:-${_TP}}"
    PP="${PP:-1}"
    CP="${CP:-1}"
    EP="${EP:-${_EP}}"
    ETP="${ETP:-1}"
    SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"

    # Talker codec loss
    export QWEN_OMNISPEAK_MCORE_CODEC_LOSS="${QWEN_OMNISPEAK_MCORE_CODEC_LOSS:-true}"
    if [[ -n "${MCORE_HIDDEN_LAYER:-}" ]]; then
        export QWEN_OMNISPEAK_MCORE_HIDDEN_LAYER="${MCORE_HIDDEN_LAYER}"
    fi

    # 冻结策略：与 deepspeed 统一，用 --freeze_parameters_ratio 全冻结 +
    # --trainable_parameters 按名前缀激活。
    # 【归属规则】_trainable_params / _trainable_regex 的值由各任务脚本
    # （train_*.sh / upstream_talker/*.sh）按各自场景生成并导出——不同任务的
    # 冻结配置不集中在 common_args.sh；一个任务有多套配置时在任务脚本内
    # 用 if/case 区分。common_args.sh 只负责消费，不提供任何默认值。
    FREEZE_PARAMETERS_RATIO="${FREEZE_PARAMETERS_RATIO:-1.0}"

    # megatron 专属超参
    # ---- batch 规模：GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE ---------------------
    # 关系式： num_microbatches = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE × DP)
    #          DP = NNODES × 每节点 GPU 数 / (TP × PP × CP)   # 本预设 TP=PP=CP=1 → DP = 总卡数
    # 【硬约束】megatron 启动断言 GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE × DP) == 0，
    #           不整除直接启动失败，改任一侧都要重新校验整除性。
    #
    # GLOBAL_BATCH_SIZE —— 一次参数更新消费的样本数（跨全部 DP rank + 梯度累积之和）
    #   作用：决定收敛行为与总步数；【不影响峰值显存】，它只改变梯度累积次数。
    #   影响：
    #     * 总步数 = 训练样本数 × NUM_TRAIN_EPOCHS / GBS
    #       实证（ktrain-ug04 日志）：18,994,932 × 2 / 1024 = 37,099，与进度条一致。
    #     * GBS 翻倍需同步上调 LR（经验 √2 ~ 2 倍），否则等效学习率下降、收敛变慢。
    #     * GBS 越大 → 累积次数越多、单步墙钟越长，但 all-reduce 被摊薄，通信占比下降。
    #     * 扩节点时固定 GBS 不改收敛：DP 增大只是把累积次数换成并行，数学等价。
    #   取值：先按「目标总步数 / LR 可承受范围」定 GBS，再挑满足整除约束的 MBS。
    #
    # MICRO_BATCH_SIZE —— 单卡单次 forward/backward 的样本数
    #   作用：【唯一决定峰值显存的 batch 旋钮】；同 GBS 下改 MBS 不改变梯度数值，
    #         只改变「并行算多少 / 串行累积多少次」的切分，收敛完全等价。
    #   MBS 越大 → 越省时间、越吃显存：
    #     * 累积次数减少（= GBS/(MBS×DP)），kernel launch 与调度开销被摊薄、GEMM 更饱和，
    #       单位样本吞吐通常上升（通用原理；本项目尚无「同 recompute 档位」的对照实测）。
    #     * 峰值显存急升且【不是线性】：talker 的 attn bias 是 dense [B,1,T,T]
    #       （modeling_qwen_omnispeak.py:build_block_causal_mask），量级 ~ MBS × T²，
    #       T = sem_len + codec_len，codec_len 实测可达 ~9.5k。
    #     * OOM 风险被长样本尖峰放大：GROUP_BY_LENGTH=true 把长样本聚到同一 micro-batch，
    #       尖峰远高于稳态 —— 实测 MBS=16 稳态 125.06 GiB 连跑 9 步平稳，第 10 步撞尖峰
    #       132.85 GiB 即 OOM（单卡可用 139.80 GiB）。故调大后必须实跑 ≥200 步才算验证过。
    #   MBS 越小 → 越省显存、越费时间：
    #     * 峰值显存下降、余量变大
    #     * 累积次数增多，单步内串行的 micro-batch 更多，kernel launch / python 调度开销
    #       占比上升、小 batch GEMM 算不满 SM，单位样本吞吐通常下降。
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${_GBS}}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${_MBS}}"
    # LR 归一：任务脚本 SCENE 预设写 LEARNING_RATE（deepspeed 变量名），
    # megatron 读 LR；显式 LR 优先，其次 LEARNING_RATE，最后默认 1e-4。
    LR="${LR:-${LEARNING_RATE:-1e-4}}"
    MIN_LR="${MIN_LR:-1e-5}"
    LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"

    # 优化器：use_distributed_optimizer = ZeRO-1（优化器状态按 DP 分片，省显存）
    USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-true}"
    export USE_DISTRIBUTED_OPTIMIZER

    # 重计算，none/full/ selective
    RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-none}"
    RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
    RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

    # attention backend（megatron sft --attention_backend）
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

    # 融合 CrossEntropy 的具体实现：native（默认）或 te（TransformerEngine）。
    # --cross_entropy_loss_fusion true 之前一直是硬编码传参、没有配套传
    # --cross_entropy_fusion_impl，实际吃的是 swift CLI 自己的默认值
    # native（见 swift/megatron/arguments/megatron_args.py 的
    # `cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'`）。
    # 这里加上可覆盖开关，方便按需切到 te 做 A/B 对比（未设置时行为不变）。
    CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-native}"

    # overlap
    OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
    OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
    ALIGN_PARAM_GATHER="${ALIGN_PARAM_GATHER:-true}"

    # packing / padding_free：多模态 codec loss 硬约束，必须为 false。
    # megatron-swift 的 padding_free 默认值为 true，不传 CLI 会导致 guard 失败，必须显式传 false。
    PACKING="${PACKING:-false}"
    PADDING_FREE="${PADDING_FREE:-false}"

    # megatron-swift 不读 WANDB_PROJECT env，需命令行 --wandb_project 传参
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-}"

fi  # end megatron

# ===========================================================================
# 训练函数
# ===========================================================================

# ---- DeepSpeed: run_bridge_alignment（swift sft）----
run_bridge_alignment() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2
    local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${task_name}}"

    # 冻结配置归任务脚本所有：未显式导出 _trainable_params 立即失败，
    # 杜绝 common_args.sh 隐式兜底导致"名义配置与实际训练不一致"。
    local trainable_params="${_trainable_params:?[common_args] _trainable_params 未设置：冻结配置归各任务脚本（train_*.sh）所有，请在任务脚本内按场景生成后导出}"
    local trainable_regex="${_trainable_regex:-}"
    local trainable_args=()
    if [[ -n "${trainable_params}" ]]; then
        trainable_args=(--trainable_parameters ${trainable_params})
    fi
    if [[ -n "${trainable_regex}" ]]; then
        trainable_args+=(--trainable_parameters_regex "${trainable_regex}")
    fi

    # 可选优化器 / LR 调度透传（任务脚本 SCENE 预设或 env 注入；
    # grouped_lr 由 plugin.py 注册，thinker 组学习率读 env THINKER_LR）。
    # 未设置时不传，保持 swift 默认，不改变既有任务行为。
    local extra_optim_args=()
    [[ -n "${OPTIMIZER:-}" ]] && extra_optim_args+=(--optimizer "${OPTIMIZER}")
    [[ -n "${LR_SCHEDULER_TYPE:-}" ]] && extra_optim_args+=(--lr_scheduler_type "${LR_SCHEDULER_TYPE}")

    mkdir -p "${output_dir}" "${BRIDGE_ALIGNMENT_CODEC_CACHE}"

    local ray_args=()
    if [[ "${USE_RAY}" == "true" || "${USE_RAY}" == "1" ]]; then
        ray_args+=(--use_ray true)
        [[ -n "${RAY_EXP_NAME}" ]] && ray_args+=(--ray_exp_name "${RAY_EXP_NAME}")
        [[ -n "${RAY_DEVICE_GROUPS}" ]] && ray_args+=(--device_groups "${RAY_DEVICE_GROUPS}")
    fi

    local dataloader_args=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
    if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
        dataloader_args+=(--dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}")
        if [[ -n "${DATALOADER_PREFETCH_FACTOR}" ]]; then
            dataloader_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
        fi
    fi

    local resume_args=()
    if [[ -n "${RESUME_FROM:-}" ]]; then
        resume_args+=(--resume_from_checkpoint "${RESUME_FROM}")
    fi

    # ---- 后端一致性验证预设（BACKEND_PARITY=1）----
    # 依据 swift 官方文档：dataset_shuffle=false 时 "#N" 为顺序采样（取前 N 条）；
    # train_dataloader_shuffle 控制 dataloader 级打乱。两层同关 + 同 seed，
    # 保证 megatron/deepspeed 逐 micro-batch 组成一致。
    local dataset_arg="${dataset_name}"
    local parity_args=()
    if [[ "${BACKEND_PARITY:-0}" == "1" ]]; then
        dataset_arg="${dataset_name}#${PARITY_SAMPLE_SIZE:-1899493}"
        parity_args=(--dataset_shuffle false --train_dataloader_shuffle false
                     --seed "${PARITY_SEED:-42}" --data_seed "${PARITY_SEED:-42}")
        # D′ 实验（2026-08-25）：custom/ 数据管线已整体替换为 zeyang_dev 版
        # （dataset/template/wds_audio/codec_tokenizer 5 文件），zeyang 版
        # template.py 带 MAX_CODEC_FRAMES 截断（默认 384）；B 侧 parity 时
        # 放开为 9999（upstream.sh parity 块同款），此处同值对齐，
        # 保证 D′ 与 B 的监督口径逐帧一致（前 2M 行 max codec_len=425，无 OOM 风险）。
        export MAX_CODEC_FRAMES="${PARITY_MAX_CODEC_FRAMES:-9999}"
    fi

    swift sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${TEMPLATE}" \
        --dataset "${dataset_arg}" \
        --load_from_cache_file "${BRIDGE_LOAD_FROM_CACHE_FILE:-true}" \
        --group_by_length "${GROUP_BY_LENGTH}" \
        --remove_unused_columns false \
        --lazy_tokenize "${LAZY_TOKENIZE}" \
        "${ray_args[@]}" \
        --tuner_type full \
        --freeze_parameters_ratio 1.0 \
        "${trainable_args[@]}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --attn_impl "${ATTN_IMPL}" \
        --deepspeed "${DEEPSPEED}" \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
        --use_liger_kernel "${USE_LIGER_KERNEL:-false}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --max_length "${MAX_LENGTH}" \
        --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --report_to "${REPORT_TO}" \
        --run_name "${RUN_NAME}" \
        "${dataloader_args[@]}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --output_dir "${output_dir}" \
        "${resume_args[@]}" \
        "${parity_args[@]}" \
        "${extra_optim_args[@]}" \
        "$@"
}

# ---- Megatron: run_bridge_alignment_megatron（megatron sft）----
run_bridge_alignment_megatron() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2
    local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${task_name}}"

    mkdir -p "${output_dir}" "${BRIDGE_ALIGNMENT_CODEC_CACHE}"

    local recompute_args=(--recompute_granularity "${RECOMPUTE_GRANULARITY}")
    if [[ "${RECOMPUTE_GRANULARITY}" == "full" ]]; then
        recompute_args+=(--recompute_method "${RECOMPUTE_METHOD}")
        recompute_args+=(--recompute_num_layers "${RECOMPUTE_NUM_LAYERS}")
    fi

    local wandb_args=()
    if [[ "${REPORT_TO}" == *wandb* ]]; then
        wandb_args+=(--wandb_project "${WANDB_PROJECT}")
        [[ -n "${WANDB_EXP_NAME}" ]] && wandb_args+=(--wandb_exp_name "${WANDB_EXP_NAME}")
    fi

    local moe_args=()
    if [[ "${EP}" -gt 1 ]]; then
        moe_args+=(--expert_model_parallel_size "${EP}")
        moe_args+=(--expert_tensor_parallel_size "${ETP}")
        moe_args+=(--moe_grouped_gemm true)
        moe_args+=(--moe_permute_fusion true)
        moe_args+=(--moe_shared_expert_overlap true)
        moe_args+=(--moe_aux_loss_coeff 1e-3)
    fi

    # 统一冻结/激活语义（与 deepspeed 一致）：--freeze_parameters_ratio 全冻结 +
    # --trainable_parameters / --trainable_parameters_regex 按名前缀激活。
    # _trainable_params / _trainable_regex 由各任务脚本按场景生成（前缀 visual.*），
    # 未设置立即失败（与 deepspeed 侧同一 fail-fast 规则）。
    local trainable_args=(--freeze_parameters_ratio "${FREEZE_PARAMETERS_RATIO}")
    : "${_trainable_params:?[common_args] _trainable_params 未设置：冻结配置归各任务脚本（train_*.sh）所有，请在任务脚本内按场景生成后导出}"
    if [[ -n "${_trainable_params:-}" ]]; then
        trainable_args+=(--trainable_parameters ${_trainable_params})
    fi
    if [[ -n "${_trainable_regex:-}" ]]; then
        trainable_args+=(--trainable_parameters_regex "${_trainable_regex}")
    fi

    # Checkpoint 策略（P1-8）：默认 --finetune true + 不保存 optimizer/rng（历史
    # 行为：仅 warm-start，中断后无法等价续训）。RESUME_FROM 非空时切换为等价
    # resume：ms-swift megatron 的 --load 恢复 model/optimizer/rng/iteration，而
    # --finetune true 会清零 iteration 且不加载 optimizer/rng（官方 Megatron-SWIFT
    # 文档），故 resume 时必须去掉 --finetune。续训前提是保存时带 optimizer/rng：
    # 需设置 MEGATRON_SAVE_FULL_STATE=true。
    local no_save_args=(--no_save_optim true --no_save_rng true)
    if [[ "${MEGATRON_SAVE_FULL_STATE:-false}" == "true" ]]; then
        no_save_args=(--no_save_optim false --no_save_rng false)
    fi
    local megatron_ckpt_args=("${no_save_args[@]}" --finetune true)
    if [[ -n "${RESUME_FROM:-}" ]]; then
        megatron_ckpt_args=("${no_save_args[@]}" --load "${RESUME_FROM}")
    fi

    # PyTorch DataLoader 拒绝 num_workers=0 与 persistent/prefetch 组合；
    # 仅在 workers>0 时传（与 deepspeed 分支守卫一致，P2-3）。
    local m_dataloader_args=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
    if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
        m_dataloader_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
                            --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}")
    fi

    # ---- 后端一致性验证预设（BACKEND_PARITY=1，与 deepspeed 分支同参数）----
    local dataset_arg="${dataset_name}"
    local parity_args=()
    if [[ "${BACKEND_PARITY:-0}" == "1" ]]; then
        dataset_arg="${dataset_name}#${PARITY_SAMPLE_SIZE:-1899493}"
        parity_args=(--dataset_shuffle false --train_dataloader_shuffle false
                     --seed "${PARITY_SEED:-42}" --data_seed "${PARITY_SEED:-42}")
    fi

    megatron sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${TEMPLATE}" \
        --dataset "${dataset_arg}" \
        --load_from_cache_file "${BRIDGE_LOAD_FROM_CACHE_FILE:-true}" \
        --remove_unused_columns false \
        --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
        --tuner_type full \
        "${trainable_args[@]}" \
        --tensor_model_parallel_size "${TP}" \
        --pipeline_model_parallel_size "${PP}" \
        --context_parallel_size "${CP}" \
        --sequence_parallel "${SEQUENCE_PARALLEL}" \
        "${moe_args[@]}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --global_batch_size "${GLOBAL_BATCH_SIZE}" \
        "${recompute_args[@]}" \
        "${megatron_ckpt_args[@]}" \
        --cross_entropy_loss_fusion true \
        --cross_entropy_fusion_impl "${CROSS_ENTROPY_FUSION_IMPL}" \
        --lr "${LR}" \
        --min_lr "${MIN_LR}" \
        --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --max_length "${MAX_LENGTH}" \
        --packing "${PACKING}" \
        --padding_free "${PADDING_FREE}" \
        --group_by_length "${GROUP_BY_LENGTH}" \
        --lazy_tokenize "${LAZY_TOKENIZE}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --attention_backend "${ATTENTION_BACKEND}" \
        --overlap_grad_reduce "${OVERLAP_GRAD_REDUCE}" \
        --overlap_param_gather "${OVERLAP_PARAM_GATHER}" \
        --align_param_gather "${ALIGN_PARAM_GATHER}" \
        --use_distributed_optimizer "${USE_DISTRIBUTED_OPTIMIZER}" \
        --save_safetensors true \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --report_to "${REPORT_TO}" \
        "${wandb_args[@]}" \
        "${m_dataloader_args[@]}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --streaming "${BRIDGE_STREAMING:-false}" \
        --output_dir "${output_dir}" \
        "${parity_args[@]}" \
        "$@"
}

# ---- 分发函数：根据 TRAIN_BACKEND 调用对应后端 ----
run_train() {
    if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
        run_bridge_alignment_megatron "$@"
    else
        run_bridge_alignment "$@"
    fi
}
