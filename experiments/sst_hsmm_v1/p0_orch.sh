#!/usr/bin/env bash
# sst_hsmm_v1 P0 (reports/sst_hsmm_v1/prereg.zh-CN.md 3.1), cards 0-7:
#  P0a  plain LA2 of the UE Thinker (combo_si_thinker --second-view none, w1) on the held-out items (clean + noisy),
#       en2zh card 0, zh2en card 1 -> eval_outputs/sst_hsmm_v1/p0/la2/thinker_<d>
#  P0b  prefix-forward emissions + timing, 6 shards on cards 2-7 -> eval_outputs/sst_hsmm_v1/p0/em
#  then p0_analyze (CPU) -> eval_outputs/sst_hsmm_v1/p0/summary.json
#   setsid nohup bash experiments/sst_hsmm_v1/p0_orch.sh > logs/sst_hsmm_v1/p0_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/p0
D=eval_outputs/sst_hsmm_v1/data
LOGD=logs/sst_hsmm_v1/p0
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
TH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  for d in en2zh zh2en; do [ -s $D/items_$d.jsonl ] || { echo "no data $d" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=()
k=0
for d in en2zh zh2en; do
  ps_+=("$(launch $k la2_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$k $PYV -m experiments.mist_w3po_v1.voicecond.combo_si_thinker \
    --second-view none --thinker $TH --direction $d --gpu-mem 0.85 --out $R/la2/thinker_$d --items-file $D/items_$d.jsonl)")
  k=$((k + 1))
  [ -n "$DRY" ] || sleep 20
done
for s in 0 1 2 3 4 5; do
  g=$((2 + s))
  ps_+=("$(launch $g em_$s "${QENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYQ -m experiments.sst_hsmm_v1.p0b --shard $s --num-shards 6 --out $R)")
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $R/la2/thinker_$d/blocks_${d}_w1.jsonl ] || die "la2 $d"; done
for s in 0 1 2 3 4 5; do [ -n "$DRY" ] || [ -s $R/shard_$s.json ] || die "em shard $s"; done
[ -n "$DRY" ] && exit 0
touch $STOP
log "gpu part done"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p0_analyze > $LOGD/analyze.log 2>&1 || die "analyze"
touch $R/DONE
log "p0 done"
