#!/usr/bin/env bash
# sst_hsmm_v1 P1a checkpoint diagnostic (reports/sst_hsmm_v1/prereg.zh-CN.md changes 3 + 4; report only). Not stacked on the
# training any more (2-sample forward: training holds 75-79 GB/card, too little margin): waits for P2 to end (p2/DONE or
# p2/FAILED; stops on p1/FAILED), then p1_diag 8 shards on cards 0-7 for the new run's checkpoints 500..2500; the old
# (no forced wait) run's 500 / 1000 em already exist (computed before the stop), then p1_diag_score (CPU) for both runs ->
# p1/diag/summary.json, p1/diag/summary_nofw.json, p1/diag/DONE. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p1_diag_orch.sh > logs/sst_hsmm_v1/p1_diag_orch.log 2>&1 < /dev/null 8>&- 9>&-
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p1
OLD=eval_outputs/sst_hsmm_v1/p1_nofw_20261122
P2=eval_outputs/sst_hsmm_v1/p2
R=$P1/diag
LOGD=logs/sst_hsmm_v1/p1_diag
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
die() { log "FAILED: $*"; [ -n "$DRY" ] || { mkdir -p "$R"; touch "$R/FAILED"; touch "$STOP"; }; exit 1; }

run_ckpt() {   # root step
  local root=$1 s=$2 ck O ps_=() k tag
  ck=$root/train/$(printf 'ckpt_%05d.pt' "$s"); O=$root/diag/$(printf 'ckpt_%05d' "$s"); tag=$(basename $root)_$s
  [ -n "$DRY" ] || [ -s "$ck" ] || die "no $ck"
  for k in 0 1 2 3 4 5 6 7; do
    ps_+=("$(launch $k diag_${tag}_$k "${QENV[@]}" CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=4 $PYQ -m experiments.sst_hsmm_v1.p1_diag \
      --ckpt $ck --shard $k --num-shards 8 --out $O)")
    [ -n "$DRY" ] || sleep 10
  done
  waitp "${ps_[@]}"
  [ -n "$DRY" ] && return 0
  for k in 0 1 2 3 4 5 6 7; do [ -s $O/shard_$k.json ] || die "diag $tag shard $k"; done
  log "diag $tag done ($(ls $O/em | wc -l) items)"
}

if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD
  log "orch $$: waiting for P2 to end"
  until [ -e $P2/DONE ] || [ -e $P2/FAILED ]; do [ -e $P1/FAILED ] && { log "P1 FAILED"; exit 1; }; sleep 60; done
  mkdir -p $R
  git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
for s in 500 1000 1500 2000 2500; do run_ckpt $P1 $s; done
[ -n "$DRY" ] || [ "$(ls $OLD/diag/ckpt_01000/em | wc -l)" = 1200 ] || die "old ckpt 1000 em incomplete"
[ -n "$DRY" ] && { echo "[dry] p1_diag_score x2" >&2; exit 0; }
touch $STOP
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p1_diag_score --steps 0 500 1000 1500 2000 2500 \
  --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p1_diag_score --steps 0 500 1000 --p1-root $OLD \
  --out $R/summary_nofw.json > $LOGD/score_nofw.log 2>&1 || die "score nofw"
touch $R/DONE
log "diag done"
