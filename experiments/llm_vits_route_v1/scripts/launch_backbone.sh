#!/bin/bash
# Detached E0 backbone training. Usage: launch_backbone.sh <cards e.g. 3,4,5,6,7> <run_dir>
set -euo pipefail
CARDS="${1:?cards}"; RUN="${2:?run dir}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
N=$(echo "$CARDS" | tr ',' '\n' | wc -l)
mkdir -p "$RUN"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$CARDS OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 setsid nohup /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/torchrun \
  --nproc_per_node "$N" --master_port 29631 scripts/train_backbone.py --config configs/ljs_char_base.json --run-dir "$RUN" \
  >> "$RUN/train.log" 2>&1 < /dev/null & disown
echo "launched on $CARDS -> $RUN/train.log"
