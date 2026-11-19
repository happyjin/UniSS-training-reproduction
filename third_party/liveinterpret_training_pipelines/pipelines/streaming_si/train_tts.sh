#!/usr/bin/env bash
# streaming_si TTS 类阶段（Emilia 单语：文本 -> FSQ speech token），三个阶段：
#
#   s1    cond_mode=text_only  训整个 talker；thinker hidden 路置零，只学「怎么说」。不依赖 thinker，可与 S2 并行
#   s3-w  cond_mode=full       桥接预热：只训 talker.thinker_proj + talker.fusion，talker 主干 / text_proj 冻结
#   s3    cond_mode=full       正式：训整个 talker（主干 + text_proj + thinker_proj + fusion）
#   （不预热时直接用合并目录跑 s3，即新版源项目的单段 S1-1 写法）
#
# 与源项目编号的对应：s1 = 源 S1-0，s3-w = 源 S1-1w，s3 = 源 S1-1a（新版源项目合并为 S1-1）。
#
# 为什么先预热（旧版源项目 train_omni_talker8_s1.sh 的 S1-1w / S1-1a，docs/model_architecture.md §4）：
#   S1 时 thinker 路恒为零，thinker_proj 从未收到梯度（实测基座 -> S1 只有 weight decay 造成的 7.8% 缩放），
#   fusion 也只见过 cond=0。合并 S2 thinker 后直接切 full，随机桥接的输出量级约为 text 路的 40 倍，
#   实测 S3 起点 speech CE：text_only 3.51 -> full 4.95。此时放开主干，会先拿已收敛的 talker 去适应噪声。
#   fusion 必须和 thinker_proj 一起预热：只放 thinker_proj 时，fusion 的门控还停在「cond 恒零」的解上。
#   预热达标的判据：同一批样本上 full 的 speech CE ≤ text_only（custom/streaming_si/scripts/s3w_bridge_gate.py）。
#
# MODEL_PATH：
#   s1    export_streaming_si.py 由基座导出的目录
#   s3-w  merge_stage_checkpoints.py 合成的目录（S2 的 thinker+audio_proj + S1 的 talker）
#   s3    s3-w 的最终 checkpoint（不预热时为上面的合成目录）
#
# 用法：
#   MODEL_PATH=... SI_S1_DATA='/data/s1_emilia_fsq_v3/*.parquet' bash pipelines/streaming_si/train_tts.sh s1
#
# 数据必须是 v3 代 FSQ（_v3 后缀，CosyVoice3 tokenizer + 正确 mel 前端）：旧代码本内容上无法分辨，
# 训错不会报错。生成后先跑源项目 data_prep/verify_fsq_data.py 验收。
set -euo pipefail
STAGE="${1:?必须指定阶段: s1 | s3-w | s3}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SI_S1_DATA:?需要 SI_S1_DATA（Emilia-FSQ parquet 文件或 glob）}"
export SI_S1_DATA

# 默认训练范围 talker（common_args.sh _scope_args）；s3-w 只训桥接
SCOPE=talker
case "${STAGE}" in
    s1) SI_COND_MODE=text_only; SI_STRIP_PROMPT_TEXT=false; LABEL=s1_tts ;;
    # s3-w 起必须去掉 user turn 中重复的原文：否则 cond 带整句前瞻，到 S5 流式时不可迁移
    s3) SI_COND_MODE=full; SI_STRIP_PROMPT_TEXT=true; LABEL=s3_tts ;;
    s3-w)
        SI_COND_MODE=full; SI_STRIP_PROMPT_TEXT=true; SCOPE=talker_bridge; LABEL=s3w_tts
        # 源 S1-1w：lr 3e-4、1 epoch 但只用约 1/10 的分片。这里用全量数据随机打乱 + MAX_STEPS 截断，
        # 等价且中英比例自然均衡；yaml 里按「1 epoch 步数的约 10%」设 MAX_STEPS
        : "${LEARNING_RATE:=3e-4}"
        ;;
    *) echo "[ERROR] 阶段应为 s1 | s3-w | s3，收到 ${STAGE}" >&2; exit 1 ;;
esac

# ---- 损失与样本口径（源 train_s1_talker.py / train_s1_talker.sh）----
export SI_COND_MODE SI_STRIP_PROMPT_TEXT
export SI_TEXT_LOSS_WEIGHT=0          # 只训 talker；text=0 时跳过 lm_head
export SI_SPEECH_LOSS_WEIGHT=1
export SI_HIDDEN_NOISE="${SI_HIDDEN_NOISE_OVERRIDE:-0.0}"
export SI_MAX_SPEECH_LEN=750          # 25Hz，≈30s
export SI_MAX_PROMPT_SPEECH_LEN=125   # 音色 prompt ≈5s
export SI_REF_DROP_RATIO=0.5          # 当前 Emilia 数据没有 ref_speech_tokens 列，音色 prompt 恒为空（音色由 flow 的 spk_emb 决定）
export SI_TEXT_SCHEDULED_SAMPLING=0.0

# ---- 优化超参（源：H200 实测 BS=32/ACC=1 吞吐约为 BS=8 的 2 倍；按 2 轮规划 cosine）----
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LEARNING_RATE:=1e-4}"
: "${NUM_TRAIN_EPOCHS:=2}"
: "${MAX_LENGTH:=512}"

source "${SCRIPT_DIR}/common_args.sh"
run_streaming_si streaming_si_s1 "${LABEL}" "${SCOPE}" "$@"
