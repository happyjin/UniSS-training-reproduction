#!/usr/bin/env bash
# prereg change 23 (PC-MBR): prmbr gen on N's test blocks (test_robust/UE) and the dev control blocks (wave17/eval_enc_wise05),
# 2 shards per direction each = 8 jobs on cards 0-7 -> prmbr select (B main, A ablation) -> s/{test,dev}_{b,a}/blocks_<d>_w1.jsonl,
# text scores (s_text.json) -> s/DONE. Placeholders stopped by PID (lib.sh). Refuses outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/s
LOGD=logs/sst_hsmm_v1/s
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
T=eval_outputs/mist_w3po_v1/test_robust
DEVB=eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05
DEVI=eval_outputs/mist_w3po_v1/wave2_S1/data
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: PC-MBR (change 23)"
  start_watcher
fi
ps_=(); g=0
for set_ in test dev; do for d in en2zh zh2en; do for k in 0 1; do
  if [ $set_ = test ]; then B=$T/UE/thinker_$d/blocks_${d}_w1.jsonl; I=$T/data/items_$d.jsonl
  else B=$DEVB/thinker_$d/blocks_${d}_w1.jsonl; I=$DEVI/items_$d.jsonl; fi
  ps_+=("$(launch $g gen_${set_}_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.prmbr gen \
    --blocks $B --items $I --direction $d --shard $k --num-shards 2 --out $R/gen_${set_}_${d}_$k.jsonl)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 20
done; done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for set_ in test dev; do for d in en2zh zh2en; do
  for k in 0 1; do [ -s $R/gen_${set_}_${d}_$k.jsonl ] || die "gen $set_ $d $k"; done
  if [ $set_ = test ]; then B=$T/UE/thinker_$d/blocks_${d}_w1.jsonl; else B=$DEVB/thinker_$d/blocks_${d}_w1.jsonl; fi
  mkdir -p $R/${set_}_b $R/${set_}_a
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.prmbr select --blocks $B --gens "$R/gen_${set_}_${d}_*.jsonl" --direction $d \
    --out-b $R/${set_}_b/blocks_${d}_w1.jsonl --out-a $R/${set_}_a/blocks_${d}_w1.jsonl >> $LOGD/select.log 2>&1 || die "select $set_ $d"
done; done
log "select: $(tr '\n' ' ' < $LOGD/select.log)"
touch $R/DONE
log "s done"
