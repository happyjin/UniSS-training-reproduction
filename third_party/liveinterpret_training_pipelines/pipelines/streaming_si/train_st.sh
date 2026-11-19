#!/usr/bin/env bash
# streaming_si S2：源音频 -> 目标语言译文（S2TT），训 audio_proj + thinker，不跑 talker。
# 与源项目编号相同（源 S2）。
#
# 用法：
#   MODEL_PATH=... SI_P2_DATA='/data/unist/machine_1/p2_fsq_v3/*.parquet' bash pipelines/streaming_si/train_st.sh
#
# P2 源音频 tar 默认取 <parquet 目录>/../p2_source_wav；一个数据集只能对应一个 tar 目录
# （m1/m2 的 tar 同名不同内容），多机数据请分别训练或显式 SI_P2_WAV_DIR。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SI_P2_DATA:?需要 SI_P2_DATA（UniST P2 parquet 文件或 glob）}"
export SI_P2_DATA
[[ -n "${SI_P2_WAV_DIR:-}" ]] && export SI_P2_WAV_DIR

# ---- 损失与样本口径（源 train_s2_st.py / train_s2_st.sh）----
export SI_TEXT_LOSS_WEIGHT=1
export SI_SPEECH_LOSS_WEIGHT=0        # speech=0 时不构造 answer_mask，talker 完全不前向
export SI_COND_MODE=full
export SI_HIDDEN_NOISE=0.0
export SI_ST_TASK=translate
# 与源项目的差异：源音频上限 20s -> 30s（Voxtral 编码器 max_position_embeddings=1500 帧 @50Hz，正好 30s），
# 超长样本（音频 > 30s / 译文 > 256 token / 总长 > MAX_LENGTH）丢弃而不截断（源是截音频、截译文）。
# 设 SI_DROP_OVERLONG=0 回到源口径。
: "${SI_MAX_AUDIO_SEC:=30}"
: "${SI_DROP_OVERLONG:=1}"
export SI_MAX_AUDIO_SEC SI_DROP_OVERLONG
export SI_MIN_AUDIO_SEC=0.5
export SI_MAX_TARGET_LEN=256

: "${PER_DEVICE_TRAIN_BATCH_SIZE:=8}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LEARNING_RATE:=1e-4}"
: "${NUM_TRAIN_EPOCHS:=1}"
# 30s 音频 375 帧 + 指令 9 + 译文 256 + eos = 641；取 768 使总长不成为额外的丢弃条件。
# padding 按 batch 内最长样本动态补齐，调大不增加平均显存，只抬高最坏一批的上限
: "${MAX_LENGTH:=768}"
export SI_MAX_LENGTH="${MAX_LENGTH}"     # 预处理器的超长过滤与 template 用同一个值

source "${SCRIPT_DIR}/common_args.sh"
run_streaming_si streaming_si_s2 s2_st thinker_audio "$@"
