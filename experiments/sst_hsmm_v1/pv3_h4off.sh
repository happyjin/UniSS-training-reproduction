#!/usr/bin/env bash
# plan v3 (change 56) H4 precheck 2 (report only): after h4tr/TRAIN_DONE, whole-audio wave2_S1 dev decodes of h4tr/merged with the
# mono tag (dev_offline_h4, 4 shards per direction, card pool) -> offdev_score vs wave17 (devoff) -> eval_outputs/sst_hsmm_v1/pv3/h4off.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv3/h4off
LOGD=logs/sst_hsmm_v1/pv3_h4off
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD
  until [ -e $D0/h4tr/TRAIN_DONE ]; do [ -e $D0/h4tr/DIVERGED ] || [ -e $D0/h4tr/FAILED ] && exit 0; sleep 60; done
  mkdir -p $R
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g off_${d}_$k "${VENV[@]}" OFF_MODEL=$D0/h4tr/merged CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.dev_offline_h4 gen \
    --direction $d --shard $k --num-shards 4 --gpu-mem 0.6 --out $R/h4m_${d}_$k.jsonl)")
  g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
$PYE experiments/sst_hsmm_v1/offdev_score.py --out $R/score.json "w17=$D0/devoff/offline_{d}_*.jsonl" "h4m=$R/h4m_{d}_*.jsonl" > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "h4 offline done"
