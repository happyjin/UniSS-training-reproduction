#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step0_rtf_decomposition_v1}

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
mkdir -p "$HF_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"

cd "$ROOT"

"$PYTHON" -m experiments.simul_s2st_route_v1.step0_rtf_decomposition.decompose \
  --run-name "$RUN_NAME" \
  --samples-per-direction "${SAMPLES_PER_DIRECTION:-4}" \
  --min-source-seconds "${MIN_SOURCE_SECONDS:-3.0}" \
  --max-source-seconds "${MAX_SOURCE_SECONDS:-12.0}" \
  --device "${DEVICE:-cuda:0}" \
  --max-write-tokens "${MAX_WRITE_TOKENS:-384}" \
  --output-json "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.json" \
  --output-md "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.md" \
  "$@"
