#!/usr/bin/env bash
# dev offline translations (dev_offline.py), 2 shards per direction on cards 0-3 -> eval_outputs/sst_hsmm_v1/devoff
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/devoff
LOGD=logs/sst_hsmm_v1/devoff
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then [ -e $R ] && { echo "exists $R" >&2; exit 1; }; mkdir -p $R $LOGD; start_watcher; fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1; do
  ps_+=("$(launch $g off_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.dev_offline gen --direction $d --shard $k --num-shards 2 --gpu-mem 0.85 --out $R/offline_${d}_$k.jsonl)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 30
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP; touch $R/DONE; log "devoff done"
