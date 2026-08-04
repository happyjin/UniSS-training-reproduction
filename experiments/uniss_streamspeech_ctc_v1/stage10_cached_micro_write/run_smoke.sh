#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-stage10_cached_micro_write_smoke_eng_cmn_v1}

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
mkdir -p "$HF_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"

"$PYTHON" -m experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write.smoke \
  --direction "${DIRECTION:-eng->cmn}" \
  --max-write-events "${MAX_WRITE_EVENTS:-3}" \
  --max-write-tokens "${MAX_WRITE_TOKENS:-384}" \
  --output-json "$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.json" \
  --output-md "$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.md"
