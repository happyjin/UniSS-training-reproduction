#!/usr/bin/env bash
# DEV grid calibration for SeamlessStreaming (R1): data/si_spont_v1 dev (GigaSpeech/WenetSpeech-style spontaneous dev,
# not a test set), first 60 utterances per direction, to choose the latency grid registered in
# paper/baselines_run_plan_seamless_v1.md BEFORE any RealSI run. One process per (direction, config group), loaded once.
#   bash pilot_dev.sh <gpu> <direction> <group-name> <thr:wait:chunk,...>
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
source experiments/s2st_baselines_v1/seamless/env.sh
GPU=$1; D=$2; G=$3; CFGS=$4
OUT=eval_outputs/s2st_baselines_v1/seamless/dev_calib/dev_$D
LOG=logs/s2st_baselines_v1/pilot_${D}_${G}.log
[ -e "$LOG" ] && { echo "refuse: $LOG exists"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU exec $PYS experiments/s2st_baselines_v1/seamless/run_seamless.py --direction $D \
  --items data/si_spont_v1/dev_${D}_ref.jsonl --limit 60 --dev-configs "$CFGS" --out $OUT --shard 0 --num-shards 1 \
  > $LOG 2>&1
