#!/usr/bin/env bash
# SimulS2ST-Omni Stage 1：talker warmup（thinker 冻结，talker 全参，Emilia DualCodec TTS，2 epoch）。
#
#   bash pipelines/simuls2st_omni/train_stage1.sh
#   RESUME=<output>/checkpoint-N bash pipelines/simuls2st_omni/train_stage1.sh
#
# 产出（每个 checkpoint-N 与最终的 output_dir 根目录）：talker_components.pt，供 Stage 2 的 init_talker_from。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULS2ST_CONFIG="${SIMULS2ST_CONFIG:-${SCRIPT_DIR}/configs/stage1.yaml}"
source "${SCRIPT_DIR}/common_args.sh"
run_simuls2st "$@"
