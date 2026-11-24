#!/usr/bin/env bash
# prereg change 20: offline ceiling of the N translator on the test set: 4 shards per direction on cards 0-7 (offline_ceiling gen),
# then offline_ceiling score -> eval_outputs/sst_hsmm_v1/ceiling/summary.json, DONE. Placeholders stopped by PID per card (lib.sh).
#   setsid nohup bash experiments/sst_hsmm_v1/ceil_orch.sh > logs/sst_hsmm_v1/ceil_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/ceiling
LOGD=logs/sst_hsmm_v1/ceiling
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: offline ceiling (change 20)"
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g gen_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.offline_ceiling gen \
    --direction $d --shard $k --num-shards 4 --out $R/offline_${d}_$k.jsonl)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 20
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $R/offline_${d}_$k.jsonl ] || die "gen $d $k"; done; done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.offline_ceiling score --parts "$R/offline_{d}_*.jsonl" --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "ceiling done"
