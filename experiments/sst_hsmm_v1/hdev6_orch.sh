#!/usr/bin/env bash
# prereg change 33 dev wave (copy of hdev5_orch.sh): hsc6_si_thinker --variant $VAR (params $PRM) on the dev items (wave2_S1, wave15 views) for theta in
# {0.80, 0.90, 0.95, 0.98} x 2 directions = 8 jobs on cards 0-7 -> eval_outputs/sst_hsmm_v1/$OUTN/dev_<theta>_<d>.
#   VAR=soft PRM=eval_outputs/sst_hsmm_v1/h3/params.json OUTN=h6soft bash experiments/sst_hsmm_v1/hdev_orch.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${VAR:?}" "${PRM:?}" "${OUTN:?}"
E=eval_outputs/sst_hsmm_v1/$OUTN
R=$E/devrun
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
VIEWS=eval_outputs/mist_w3po_v1/wave15/views
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/${OUTN}dev
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
  ps_+=("$(launch $g dev_${th}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc6_si_thinker --mode hsc --variant $VAR \
    --params $PRM --theta $th --verifier $VER --second-view enh --views $VIEWS/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $DATA/items_$d.jsonl --out $E/dev_${th}_$d)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for th in 0.80 0.90 0.95 0.98; do for d in en2zh zh2en; do [ -s $E/dev_${th}_$d/blocks_${d}_w1.jsonl ] || die "dev $th $d"; done; done
touch $STOP; touch $R/DONE; log "$OUTN done"
