#!/usr/bin/env bash
# sst_hsmm_v1 P1a checkpoint diagnostic (reports/sst_hsmm_v1/prereg.zh-CN.md change 3; report only; user OK to stack on the
# training cards). Checkpoints 500..2000: as each appears (training at least 5 steps past it, p1 not DONE), p1_diag 8 shards
# on cards 0-7 stacked on the training (memory-capped per process). Checkpoint 2500 (and any of 500..2000 the training
# outran): after P2 ends (p2/DONE or p2/FAILED), so it never competes with P2's vLLM. Then p1_diag_score (CPU) for
# steps 0 500 .. 2500 -> p1/diag/summary.json + p1/diag/DONE. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p1_diag_orch.sh > logs/sst_hsmm_v1/p1_diag_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p1
P2=eval_outputs/sst_hsmm_v1/p2
R=$P1/diag
LOGD=logs/sst_hsmm_v1/p1_diag
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/unused.stop
source experiments/sst_hsmm_v1/lib.sh
die() { log "FAILED: $*"; [ -n "$DRY" ] || { mkdir -p "$R"; touch "$R/FAILED"; }; exit 1; }
cur_step() { tail -1 $P1/train/metrics_rank0.jsonl 2>/dev/null | $PYQ -c "import json,sys; print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo -1; }

run_ckpt() {   # step
  local s=$1 ck O ps_=() k
  ck=$P1/train/$(printf 'ckpt_%05d.pt' "$s"); O=$R/$(printf 'ckpt_%05d' "$s")
  [ -n "$DRY" ] || [ -s "$ck" ] || die "no $ck"
  for k in 0 1 2 3 4 5 6 7; do
    ps_+=("$(launch $k diag_${s}_$k "${QENV[@]}" CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=4 $PYQ -m experiments.sst_hsmm_v1.p1_diag \
      --ckpt $ck --shard $k --num-shards 8 --out $O)")
    [ -n "$DRY" ] || sleep 20
  done
  waitp "${ps_[@]}"
  [ -n "$DRY" ] && return 0
  for k in 0 1 2 3 4 5 6 7; do [ -s $O/shard_$k.json ] || die "diag $s shard $k"; done
  log "diag $s done ($(ls $O/em | wc -l) items)"
}

if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$"
fi
LATE=""
for s in 500 1000 1500 2000; do
  if [ -n "$DRY" ]; then run_ckpt $s; continue; fi
  until [ "$(cur_step)" -ge $((s + 5)) ] || [ -e $P1/DONE ] || [ -e $P1/FAILED ]; do sleep 60; done
  [ -e $P1/FAILED ] && die "P1 FAILED"
  if [ -e $P1/DONE ]; then LATE="$LATE $s"; continue; fi
  run_ckpt $s
done
if [ -z "$DRY" ]; then
  log "waiting for P2 to end before checkpoint 2500${LATE:+ and$LATE}"
  until [ -e $P2/DONE ] || [ -e $P2/FAILED ]; do [ -e $P1/FAILED ] && die "P1 FAILED"; sleep 60; done
fi
for s in $LATE 2500; do run_ckpt $s; done
[ -n "$DRY" ] && { echo "[dry] p1_diag_score" >&2; exit 0; }
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p1_diag_score --steps 0 500 1000 1500 2000 2500 \
  --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "diag done"
