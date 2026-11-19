#!/usr/bin/env bash
# streaming_si 离线 S2ST 阶段（源音频 -> 译文 + 目标 FSQ）。两个阶段：
#
#   s4-w  桥接预热：只训 talker.thinker_proj + talker.fusion（thinker / audio_proj / talker 主干 / text_proj 冻结），
#         只算语音 loss；MODEL_PATH = S3 的最终 checkpoint
#   s4    正式：joint 训练 audio_proj + thinker + talker；MODEL_PATH = s4-w 的最终 checkpoint
#         （不预热时也可直接用 S3 的 checkpoint）
#
# 与源项目编号的对应：s4-w = 源 S3-w，s4 = 源 S3。
#
# 为什么要 s4-w（旧版源项目 train_s3_s2st.sh run_s3_w / train_s3_s2st.py --mode bridge）：
#   S3 的 thinker 路输入是单语 TTS（Emilia，上下文就是这句话本身）；S4 换成 UNISS 跨语（上下文是源语音），
#   同一个译文 token 的 thinker hidden 分布变了。text 路查的是冻结的 talker_text_embed，与上下文无关，
#   且 UNISS 的目标语（cmn / eng）Emilia 都覆盖，所以只需重新对齐 thinker 路的桥接。
#   fusion 必须和 thinker_proj 一起放开（只放 thinker_proj 时门控仍停在旧解上）。
#   text_proj 保持冻结：门 A 的 text_only 参照才不动，也避免 loss 从 text 路这条捷径降下去。
#   thinker 冻结时文本 CE 对可训参数没有梯度，故 text_loss_weight=0（同时跳过 lm_head 全序列投影）。
#
#   预热必须是短程的：源用 8 个等间隔 parquet × 每 rank 8 万行（8 卡 × BS16 ≈ 5000 步，约 64 万样本，lr 3e-4）。
#   这里与 s3-w 相同：全量数据随机打乱 + MAX_STEPS 截断（等价且两个方向自然均衡），
#   默认 2500 步 × 全局 256（16 卡 × 每卡 16）= 64 万样本。
#   验收看门 A，不看训练 loss 曲线（桥接预热期曲线本就近乎平）：
#     python custom/streaming_si/scripts/s4w_bridge_gate.py --model <s4-w ckpt> --data '<P2 parquet glob>' --wav_dir <tar 目录>
#   要求每个方向 full 的 speech CE < text_only。
#
# 用法：
#   MODEL_PATH=.../s3_tts/checkpoint-xxx SI_P2_DATA='/data/unist/machine_1/p2_fsq_v3/*.parquet' \
#       bash pipelines/streaming_si/train_s2st.sh s4-w
#   MODEL_PATH=.../s4w_s2st/checkpoint-xxx SI_P2_DATA=... bash pipelines/streaming_si/train_s2st.sh s4
set -euo pipefail
STAGE="${1:?必须指定阶段: s4-w | s4}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SI_P2_DATA:?需要 SI_P2_DATA（UniST P2 parquet 文件或 glob）}"
export SI_P2_DATA
[[ -n "${SI_P2_WAV_DIR:-}" ]] && export SI_P2_WAV_DIR

case "${STAGE}" in
    s4)
        SCOPE=joint; LABEL=s4_s2st
        # 源默认 1:1。语音 CE（约 3.1）远大于文本 CE（约 0.6），thinker 的梯度主要来自语音：
        # 24 卡 / lr 8e-5 实测文本 loss 4300 步内 0.58 -> 0.82（翻译能力被侵蚀），可在 yaml 调高
        : "${SI_TEXT_LOSS_WEIGHT:=1}"
        export SI_TEXT_LOSS_WEIGHT
        # 噪声负样本（译文为空）保留：文本 CE 学「噪声+指令 -> 立即 eos」，speech 监督为空。
        # S4-w 不设（thinker 冻结 + text_w=0，这类行无梯度）。
        export SI_KEEP_NOISE_NEG=1
        : "${LEARNING_RATE:=5e-5}"
        ;;
    s4-w)
        SCOPE=talker_bridge; LABEL=s4w_s2st
        export SI_TEXT_LOSS_WEIGHT=0
        : "${LEARNING_RATE:=3e-4}"
        : "${MAX_STEPS:=2500}"
        : "${WARMUP_RATIO:=0.03}"
        ;;
    *) echo "[ERROR] 阶段应为 s4-w | s4，收到 ${STAGE}" >&2; exit 1 ;;
esac

# ---- 损失与样本口径（源 train_s3_s2st.py，--mode joint / bridge）----
export SI_SPEECH_LOSS_WEIGHT=1
export SI_COND_MODE=full
export SI_HIDDEN_NOISE=0.0
export SI_ST_TASK=translate
# 与 S2 相同：源音频上限 30s，超长样本丢弃而不截断（另含 target_fsq > SI_MAX_SPEECH_LEN，750 = 30s @25Hz）。
# 设 SI_DROP_OVERLONG=0 回到源口径（20s / 截断）。
: "${SI_MAX_AUDIO_SEC:=30}"
: "${SI_DROP_OVERLONG:=1}"
export SI_MAX_AUDIO_SEC SI_DROP_OVERLONG
export SI_MIN_AUDIO_SEC=0.5
export SI_MAX_TARGET_LEN=256
export SI_MAX_SPEECH_LEN=750

: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${MAX_LENGTH:=768}"                   # 同 S2：30s 375 帧 + 指令 + 256 译文 + eos = 641
export SI_MAX_LENGTH="${MAX_LENGTH}"

source "${SCRIPT_DIR}/common_args.sh"
run_streaming_si streaming_si_s3 "${LABEL}" "${SCOPE}" "$@"
