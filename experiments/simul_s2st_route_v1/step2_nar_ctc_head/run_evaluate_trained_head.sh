#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2_trained_nar_decode_v1}
CKPT_ROOT=$ROOT/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64
OUT_JSON=$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.json
OUT_MD=$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.md

export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

test ! -e "$OUT_JSON"
test ! -e "$OUT_MD"

"$PYTHON" "$ROOT/experiments/simul_s2st_route_v1/step2_nar_ctc_head/evaluate_trained_head.py" \
  --run-name "$RUN_NAME" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  --checkpoint "iter1000=$CKPT_ROOT/iter_0001000" \
  --checkpoint "iter2000=$CKPT_ROOT/iter_0002000" \
  --checkpoint "iter3000=$CKPT_ROOT/iter_0003000" \
  --samples-per-direction 16
