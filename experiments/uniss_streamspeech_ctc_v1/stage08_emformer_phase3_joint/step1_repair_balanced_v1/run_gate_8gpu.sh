#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_NAME=stage08_step1_repair_balanced_gate32_v1 \
CHECKPOINT_ITERS="50 100 150 200 250 300 350 400" \
MEGATRON_ROOT="$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_p3w2_zhen1p25_v1" \
  bash "$ROOT/experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_frozen_qwen/run_step1_gate_8gpu.sh"
