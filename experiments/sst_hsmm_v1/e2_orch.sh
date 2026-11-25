#!/usr/bin/env bash
# E2 theta (prereg change 27): SDC (--mode sdc, e2/head.pt) on the dev items (wave2_S1, wave15 views) for theta in
# {0.80, 0.85, 0.90, 0.95} x 2 directions = 8 jobs on cards 0-7 -> e2/dev_<theta>_<d>; then pick_theta.py -> e2/theta.json.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/e2
R=$E/devrun
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
VIEWS=eval_outputs/mist_w3po_v1/wave15/views
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/e2dev
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s $E/head.pt ] || { echo "head missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for th in 0.80 0.85 0.90 0.95; do for d in en2zh zh2en; do
  ps_+=("$(launch $g dev_${th}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.sdc_si_thinker --mode sdc \
    --head $E/head.pt --theta $th --verifier $VER --second-view enh --views $VIEWS/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $DATA/items_$d.jsonl --out $E/dev_${th}_$d)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
$PYQ -m experiments.sst_hsmm_v1.pick_theta > $LOGD/pick.log 2>&1 || die "pick"
touch $R/DONE
log "e2 done"
