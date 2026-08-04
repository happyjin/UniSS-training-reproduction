#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-stage09_online_runtime_smoke_eng_cmn_v1}
OUT_JSON=$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.json
OUT_MD=$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.md

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
mkdir -p "$HF_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"
test ! -e "$OUT_JSON"
test ! -e "$OUT_MD"

"$PYTHON" -m experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.smoke \
  --direction "${DIRECTION:-eng->cmn}" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD"
