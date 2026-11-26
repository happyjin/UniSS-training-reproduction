#!/usr/bin/env bash
# prereg change 37-4: N-policy rollouts (hsc_si_thinker --mode rollout, verifier SAP2 = v2/lora/a1.0000) on the e1 items (h1's shards),
# cards 0-7 -> h23/roll_<d>_<k>; then hsmm_anchor fit (HSC6 objective, precheck rule as h14) -> h23/precheck.json + h23/params.json
# (exit 3 on NO-GO). Copy of h1_orch.sh. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/e1
H=eval_outputs/sst_hsmm_v1/h23
R=$H/run
SH=eval_outputs/sst_hsmm_v1/h1/run/data
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v2/lora/a1.0000
LOGD=logs/sst_hsmm_v1/h23run
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s $VER/adapter_config.json ] || { echo "SAP2 lora missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc_si_thinker --mode rollout \
    --verifier $VER --second-view enh --views $E/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
touch $STOP
OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_anchor fit --roots "$H/roll_*" --pre-out $H/precheck.json \
  --out $H/params.json > $LOGD/fit.log 2>&1
rc=$?
[ $rc -eq 3 ] && { log "h23 precheck NO-GO"; touch $R/NOGO; exit 3; }
[ $rc -eq 0 ] || die "fit"
touch $R/DONE
log "h23 done"
