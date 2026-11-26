#!/usr/bin/env bash
# prereg change 38-2: N-policy rollouts with K-sample consistency (hsc10_si_thinker --mode rollout --samples 6 --samp-max 24, old SAP
# verifier) on the e1 items (h1 shards), cards 0-7 -> h26/roll_<d>_<k>; then hsmm_cons fit -> h26/precheck.json + h26/params.json
# (exit 3 on NO-GO). Copy of h23_orch.sh. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/e1
H=eval_outputs/sst_hsmm_v1/h26
R=$H/run
SH=eval_outputs/sst_hsmm_v1/h1/run/data
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/h26run
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s $VER/adapter_config.json ] || { echo "SAP lora missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc10_si_thinker --mode rollout --samples 6 --samp-max 24 \
    --verifier $VER --second-view enh --views $E/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
touch $STOP
OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_cons fit --roots "$H/roll_*" --pre-out $H/precheck.json \
  --out $H/params.json > $LOGD/fit.log 2>&1
rc=$?
[ $rc -eq 3 ] && { log "h26 precheck NO-GO"; touch $R/NOGO; exit 3; }
[ $rc -eq 0 ] || die "fit"
touch $R/DONE
log "h26 done"
