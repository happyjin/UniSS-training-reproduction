#!/usr/bin/env bash
# en2zh analysis after change 56 (report only): whole-audio wave2_S1 dev decodes of the E1 drafter (p2ft/merged) and the H1 drafter
# (h1tr/merged) with the complete-audio tag (dev_offline_tag, 4 shards per direction, card pool) -> offdev_score vs wave17 (devoff)
# -> eval_outputs/sst_hsmm_v1/pv3/offtag.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv3/offtag
LOGD=logs/sst_hsmm_v1/pv3_offtag
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  start_watcher
fi
ps_=(); g=0
for m in p2ft h1tr; do for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g off_${m}_${d}_$k "${VENV[@]}" OFF_MODEL=$D0/$m/merged CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.dev_offline_tag gen \
    --direction $d --shard $k --num-shards 4 --gpu-mem 0.6 --out $R/${m}_${d}_$k.jsonl)")
  g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
done; done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
$PYE experiments/sst_hsmm_v1/offdev_score.py --out $R/score.json "w17=$D0/devoff/offline_{d}_*.jsonl" "p2ft=$R/p2ft_{d}_*.jsonl" \
  "h1tr=$R/h1tr_{d}_*.jsonl" "h4m=$D0/pv3/h4off/h4m_{d}_*.jsonl" > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "offtag done"
