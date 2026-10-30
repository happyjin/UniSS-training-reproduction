#!/usr/bin/env bash
# Paper v5 training on the new server (8 x H200), the two runs the 2026-10-30 deadline cut:
#   1. Zh->En MIST-Simul (deadline labels) for 3,000 steps from O2, as the En->Zh S2-dl (8 GPUs, MBS 8, GBS 64: with
#      8 data-parallel ranks MBS 16 would need GBS 128, a different optimisation setting than En->Zh).
#   2. then the mixed-interval En->Zh model (S2-dl-m, rows e2e_gstar_stream_multid_v1), same settings.
# After iteration 30 of each run, writes the per-card memory to logs/gpu_coord/train_mem.txt and touches
# logs/gpu_coord/TRAIN{1,4}_STABLE (the single-speaker branch stacks its jobs only then); TRAIN{1,4}_DONE at the end.
# run_thinker_gstar.sh refuses existing outputs. Launch: setsid nohup bash ... > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v5_train.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; D=data/ami_multispk_si_v1; C=logs/gpu_coord
O2=${EV}/thinker_gstar_o2_20261029b/thinker_merged
mkdir -p $C

stable_watch() {   # $1 tag, $2 training log
  until grep -qE "iteration +30/" "$2" 2>/dev/null; do
    grep -qE "THINKER_GSTAR_(FAILED|DONE)|拒绝覆盖" "$2" 2>/dev/null && return
    sleep 20
  done
  { echo "[$(date -u +%F' '%T) UTC] $1 stable: per-card used MiB (index, used, total)"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    grep -E "iteration +30/" "$2" | grep -oE "elapsed time per iteration \(ms\): [0-9.]+"; } >> $C/train_mem.txt
  touch $C/TRAIN$1_STABLE
}

train() {   # $1 tag, $2 run id, $3 rows, $4 port
  local L=logs/multispk/$2.log
  log "train $2 (rows $3)"
  stable_watch $1 $L &
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 MBS=8 GBS=64 RUN_ID=$2 BASE=${O2} INIT_MERGED=1 ROWS=$3 \
    TRAIN_ITERS=3000 MASTER_PORT=$4 bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > $L 2>&1
  if grep -q THINKER_GSTAR_DONE $L; then log "$2 done"; else log "$2 FAILED: $(tail -3 $L | tr '\n' ' ' | cut -c1-300)"; fi
  touch $C/TRAIN$1_DONE
}

train 1 thinker_gstar_s2dl_zh2en_3k_20261030 ${D}/e2e_gstar_stream_zh2en_v1/rows_train.jsonl 29861
train 4 thinker_gstar_s2dlm_3k_20261030 ${D}/e2e_gstar_stream_multid_v1/rows_train.jsonl 29862
log "V5_TRAIN_DONE"
