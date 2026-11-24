#!/usr/bin/env bash
# prereg change 22 (diagnostic, report only): prefix-forced full-audio continuation of N's committed prefixes on the test set,
# 8 shards on cards 0-7 (prefix_forced) -> eval_outputs/sst_hsmm_v1/pfx_test/forced_*.jsonl, DONE.
#   setsid nohup bash experiments/sst_hsmm_v1/pfx_orch.sh > logs/sst_hsmm_v1/pfx_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/pfx_test
LOGD=logs/sst_hsmm_v1/pfx_test
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
W17=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
if [ -z "$DRY" ]; then
  [ -e $R/DONE ] && { echo "exists: $R/DONE (refuse to overwrite)" >&2; exit 1; }
  [ -s $R/events.jsonl ] || { echo "missing events" >&2; exit 1; }
  mkdir -p $LOGD
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: prefix-forced test diagnostic (change 22)"
  start_watcher
fi
ps_=()
for k in 0 1 2 3 4 5 6 7; do
  ps_+=("$(launch $k pfx_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$k $PYV -m experiments.sst_hsmm_v1.prefix_forced \
    --events $R/events.jsonl --thinker $W17 --shard $k --num-shards 8 --out $R/forced_$k.jsonl)")
  [ -n "$DRY" ] || sleep 20
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for k in 0 1 2 3 4 5 6 7; do [ -s $R/forced_$k.jsonl ] || die "pfx $k"; done
touch $R/DONE
log "pfx done"
