#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2a_upsample_ratio_v1}

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TMPDIR=$USER_ROOT/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The measurement is pure statistics; keep it off the GPUs the training jobs use.
export CUDA_VISIBLE_DEVICES=""

cd "$ROOT"

"$PYTHON" -m experiments.simul_s2st_route_v1.step2_nar_ctc_head.measure_upsample_ratio \
  --run-name "$RUN_NAME" \
  --sample-rows "${SAMPLE_ROWS:-200000}" \
  --output-json "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.json" \
  --output-md "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.md" \
  "$@"
