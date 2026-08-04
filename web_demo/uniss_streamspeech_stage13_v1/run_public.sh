#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}
DEMO_ENV=${DEMO_ENV:-$USER_ROOT/conda_envs/uniss-offline-demo}
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PORT=${UNISS_STAGE13_PORT:-7865}
MEDIA_BIN=$ROOT/web_demo/streaming_s2st_r2_v1/bin

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export PATH=$MEDIA_BIN:${PATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"
for media_tool in ffmpeg ffprobe; do
  if ! command -v "$media_tool" >/dev/null 2>&1; then
    echo "Missing required media tool: $media_tool (expected under $MEDIA_BIN)" >&2
    exit 1
  fi
done
cd "$ROOT"
exec "$DEMO_ENV/bin/python" -m web_demo.uniss_streamspeech_stage13_v1.app_gradio \
  --host 0.0.0.0 --port "$PORT" --share
