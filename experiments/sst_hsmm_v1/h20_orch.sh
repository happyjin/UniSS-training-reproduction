#!/usr/bin/env bash
# prereg change 36: rollouts of the deployed policy (hsc9_si_thinker n2 theta 0.75, params h14) on the open UniST p4 mixtures (h1's shards
# h1/run/data, e1 views), 4 shards per direction on cards 0-7 -> h20/roll_<d>_<k>; then hsmm_onpol fit (h1 + h20 rolls, CPU)
# -> h20/precheck.json (+ h20/params.json on GO; exit 3 on NO-GO). Copy of h1_orch.sh. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/e1
H=eval_outputs/sst_hsmm_v1/h20
R=$H/run
SH=eval_outputs/sst_hsmm_v1/h1/run/data
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/h20run
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s eval_outputs/sst_hsmm_v1/h14/params.json ] || { echo "h14 missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc9_si_thinker --mode hsc --variant n2 \
    --params eval_outputs/sst_hsmm_v1/h14/params.json --theta 0.75 \
    --verifier $VER --second-view enh --views $E/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
touch $STOP
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_onpol fit --roots "eval_outputs/sst_hsmm_v1/h1/roll_*,$H/roll_*" --onpol-root "$H/roll_*" \
  --pre-out $H/precheck.json --out $H/params.json > $LOGD/fit.log 2>&1
rc=$?
[ $rc -eq 3 ] && { log "h20 precheck NO-GO"; touch $R/NOGO; exit 3; }
[ $rc -eq 0 ] || die "fit"
touch $R/DONE
log "h20 done"
