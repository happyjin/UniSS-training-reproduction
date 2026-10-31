#!/usr/bin/env bash
# v10 (prereg_v5 section 1): adapter seeds s3 and s4 with s0's recipe (run_thinker_sd.sh, step 200 of an 800-step
# schedule), one after the other on all eight cards; each run merges its step-200 adapter on the CPU afterwards, and
# the next run starts as soon as the previous one's trainer has exited (the merge overlaps it). s2's merged copy, which
# the old server had, is rebuilt from its step-200 LoRA on the CPU at the start. Nothing existing is overwritten
# (run_thinker_sd.sh and merge refuse existing outputs).
#   MBS (default 32) may be set to 16: GBS stays 256 (two micro-batches accumulated).
#   setsid nohup bash reports/qabc_v10/train_seeds.sh > logs/qabc_v10/train_seeds.log 2>&1 &
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/qabc_v10
O=eval_outputs/uniss_qwen3_omni_v1
log() { echo "[$(date -u '+%F %T')] $*"; }
S2=$O/thinker_op_v3s2_20261025T143042Z/thinker_merged
if [ ! -e "$S2" ]; then
  log "merge s2 -> $S2"
  CUDA_VISIBLE_DEVICES="" /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python \
    experiments/uniss_qwen3_omni_v1/training/merge_thinker_sd.py \
    --checkpoint-dir checkpoints/uniss_qwen3_omni_v1/thinker_op_v3s2_20261025T143042Z/iter_0000200 \
    --rank 32 --alpha 64 --out "$S2" > $L/merge_s2.log 2>&1 &
fi
for k in 3 4; do
  RID=thinker_op_v3s${k}_20261031
  log "train s$k ($RID)"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_ID=$RID ROWS=data/si_thinker_oracle_prefix_v1/rows_train.jsonl NPROC=8 \
    MBS=${MBS:-32} GBS=256 NUM_WORKERS=12 TRAIN_ITERS=800 SAVE_INTERVAL=200 EXIT_INTERVAL=200 WARMUP_ITERS=30 \
    TRAIN_SEED=2026110$k SD_SEED=2026110$k MASTER_PORT=2982$k MERGED=$PWD/$O/$RID/thinker_merged_i200 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh > $L/train_s$k.log 2>&1 &
  sp=$!
  # wait for the trainer (the script's torch.distributed child) to exit; the merge then runs on the CPU
  sleep 60
  while kill -0 $sp 2>/dev/null; do
    tr=$(ps -o pid= --ppid $sp | tr -d ' ' | head -1)
    if [ -n "$tr" ] && ps -o args= -p "$tr" | grep -q torch.distributed.run; then sleep 30; continue; fi
    [ -f checkpoints/uniss_qwen3_omni_v1/$RID/latest_checkpointed_iteration.txt ] && break
    sleep 30
  done
  log "s$k trainer finished (script pid $sp; merge continues if it is running)"
  grep -q THINKER_SD_FAILED $L/train_s$k.log && { log "s$k FAILED"; exit 1; }
done
wait
log "ALL_DONE"
