#!/usr/bin/env bash
# HSC2 theta (prereg change 29): HSC2 (--mode hsc, h3/params.json, hsc2_si_thinker) on the dev items (wave2_S1, wave15 views) for theta in
# {0.80, 0.90, 0.95, 0.98} x 2 directions = 8 jobs on cards 0-7 -> h4/dev_<theta>_<d>; then pick_theta_hsc.py --root h4 -> h4/theta.json.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/h4
PRM=eval_outputs/sst_hsmm_v1/h3/params.json
R=$E/devrun
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
VIEWS=eval_outputs/mist_w3po_v1/wave15/views
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/h4dev
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s $PRM ] || { echo "params missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for th in 0.80 0.90 0.95 0.98; do for d in en2zh zh2en; do
  ps_+=("$(launch $g dev_${th}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc2_si_thinker --mode hsc \
    --params $PRM --theta $th --verifier $VER --second-view enh --views $VIEWS/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $DATA/items_$d.jsonl --out $E/dev_${th}_$d)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
$PYQ -m experiments.sst_hsmm_v1.pick_theta_hsc --root h4 > $LOGD/pick.log 2>&1 || die "pick"
touch $R/DONE
log "h4 done"
