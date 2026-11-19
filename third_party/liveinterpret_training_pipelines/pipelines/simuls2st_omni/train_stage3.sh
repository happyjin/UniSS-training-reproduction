#!/usr/bin/env bash
# SimulS2ST-Omni Stage 3：流式轨迹微调（先把 Stage 2 的 adapter 合进基座，再给 thinker / talker 各挂一套
# 新的 LoRA r8；talker 的目标含 thinker_to_talker_proj；延迟倍数 m ∈ {1..12}，1 epoch）。
#
#   bash pipelines/simuls2st_omni/train_stage3.sh
#   SIMULS2ST_MERGE_ADAPTERS_FROM=<Stage 2 输出目录或 checkpoint-N> bash pipelines/simuls2st_omni/train_stage3.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULS2ST_CONFIG="${SIMULS2ST_CONFIG:-${SCRIPT_DIR}/configs/stage3.yaml}"
source "${SCRIPT_DIR}/common_args.sh"
run_simuls2st "$@"
