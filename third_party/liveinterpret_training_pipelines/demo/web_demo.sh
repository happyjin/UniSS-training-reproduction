#!/usr/bin/env bash
set -euo pipefail

readonly PYTHON_BIN="/opt/dlami/nvme/zeyangsong/.conda_envs/liveinterpret/bin/python"
readonly DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEMO_ENTRY="${DEMO_DIR}/web_demo.py"
readonly CHECKPOINT="/opt/dlami/nvme/zeyangsong/projects/liveinterpret_training_pipelines/output/qwen_omnispeak_output/batch4.8/upstream_reply_add-adapter-talker-from-base-unist-bs64x8-s2s/v0-20260803-045815/checkpoint-41000"
readonly CODEC_TOKENIZER="/opt/dlami/nvme/zeyangsong/projects/liveinterpret_training_pipelines/models/Qwen3-TTS-Tokenizer-12Hz"
readonly SPEAKER_ENCODER="/opt/dlami/nvme/zeyangsong/projects/liveinterpret_training_pipelines/models/Qwen3-TTS-12Hz-1.7B-Base"
readonly WARMUP_AUDIO="/opt/dlami/nvme/zeyangsong/projects/liveinterpret_training_pipelines/inference_outputs/batch4.7/stage2_5_ckpt47000/ar_smoke/val_11/source.wav"

# 启动时仍可临时覆盖，例如：DEVICE=cuda:1 PORT=7861 ./demo/web_demo.sh
readonly DEVICE="${DEVICE:-cuda:0}"
readonly PORT="${PORT:-7860}"

exec "${PYTHON_BIN}" "${DEMO_ENTRY}" \
  --model-type qwen2_5_3b_omnispeak_a \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --talker-layout reply_add \
  --codec-tokenizer "${CODEC_TOKENIZER}" \
  --codec-tokenizer-device "${DEVICE}" \
  --speaker-encoder "${SPEAKER_ENCODER}" \
  --warmup-audio "${WARMUP_AUDIO}" \
  --warmup-target-lang zh \
  --port "${PORT}"
