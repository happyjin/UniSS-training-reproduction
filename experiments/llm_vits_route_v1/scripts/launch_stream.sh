#!/bin/bash
# Detached streaming fine-tune. Usage: launch_stream.sh <card> <variant> <backbone_ckpt> <align_dir> <run_dir> [extra args]
set -euo pipefail
CARD="${1:?card}"; VAR="${2:?variant}"; CK="${3:?backbone ckpt}"; AL="${4:?align dir}"; RUN="${5:?run dir}"; shift 5
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
mkdir -p "$RUN"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 setsid nohup /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
  scripts/train_stream.py --variant "$VAR" --backbone-config configs/ljs_char_base.json --backbone-ckpt "$CK" \
  --align-dir "$AL" --run-dir "$RUN" "$@" >> "$RUN/train.log" 2>&1 < /dev/null & disown
echo "$VAR on card $CARD -> $RUN"
