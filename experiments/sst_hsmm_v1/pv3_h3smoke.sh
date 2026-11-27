#!/usr/bin/env bash
# plan v3 (change 56): H3 smoke on the card pool -- 3 edev items per direction, hsc22 --syn-gate (E1 drafter, pfe1 params); checks the
# blocks exist and the gate logged reads; then touches eval_outputs/sst_hsmm_v1/pv3/H3_SMOKE_OK (pfh3's WAIT). Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv3/h3smoke
LOGD=logs/sst_hsmm_v1/pv3_h3smoke
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
[ -n "$DRY" ] || until [ -d $D0/h1tr/train ] && [ -d $D0/h4tr/train ]; do sleep 30; done     # never take cards between smoke and train
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $R $LOGD
  for d in en2zh zh2en; do head -n 3 $D0/edev/data/items_$d.jsonl > $R/items_$d.jsonl; done
  start_watcher
fi
ps_=()
for d in en2zh zh2en; do
  m=heal; [ $d = zh2en ] && m=base
  ps_+=("$(launch 0 smoke_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYV -m experiments.sst_hsmm_v1.hsc22_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/pfe1/h/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/edev/views/$d \
    --thinker eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 --direction $d --gpu-mem 0.6 --variants $m@0.70 \
    --draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --syn-gate --items-file $R/items_$d.jsonl --out $R/run_$d)")
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do
  ls $R/run_$d/blocks_*.jsonl > /dev/null 2>&1 || die "smoke $d: no blocks"
  grep -q '"syn"' $R/run_$d/blocks_*.jsonl || die "smoke $d: no syn records"
done
grep -h -o '"syn_[a-z_]*": [0-9]*' $R/run_*/*.json 2>/dev/null | sort | uniq -c | head
touch $D0/pv3/H3_SMOKE_OK
log "h3 smoke ok"
