#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step3_ar_pareto_smoke8_v1}
OUT_JSON=$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.json
OUT_MD=$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.md
MASTER_PORT=${MASTER_PORT:-29831}
MAX_SAMPLES=${MAX_SAMPLES:-8}

export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_DEVICE_MAX_CONNECTIONS=1

test ! -e "$OUT_JSON"
test ! -e "$OUT_MD"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$MASTER_PORT" \
  "$ROOT/experiments/simul_s2st_route_v1/step3_waitk_pareto/evaluate_ar_pareto.py" \
  --run-name "$RUN_NAME" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  --max-samples "$MAX_SAMPLES" \
  --lagging-k 0 2 4 8 \
  --lambda-window 0 512
