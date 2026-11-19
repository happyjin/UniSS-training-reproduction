#!/usr/bin/env bash
# SimulS2ST-Omni Stage 2：多任务联合训练（thinker LoRA r32，talker 继续全参；
# ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5，按步抽任务，23438 步）。
#
#   bash pipelines/simuls2st_omni/train_stage2.sh
#   SIMULS2ST_INIT_TALKER_FROM=<Stage 1 输出目录或 checkpoint-N> bash pipelines/simuls2st_omni/train_stage2.sh
#
# 起点：配置 model.init_talker_from（缺省 outputs/checkpoint/stage1_talker_warmup）里的 talker 权重。
# 多卡时 ddp_find_unused_parameters=true（文本任务 batch 不经过 talker），见 integration/swift_args.py。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULS2ST_CONFIG="${SIMULS2ST_CONFIG:-${SCRIPT_DIR}/configs/stage2.yaml}"
source "${SCRIPT_DIR}/common_args.sh"
run_simuls2st "$@"
