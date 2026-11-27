#!/usr/bin/env bash
# change 58 C2' (HCG v2, hsc24) smoke on the card pool -- 3 edev items per direction, hsc23 --mid-mbr 4 --mbr-util hcg (p2ft adapter + tags, pfe1
# params); checks blocks + hcg statistics, then touches eval_outputs/sst_hsmm_v1/pv5/HCG2_SMOKE_OK. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv5/hcgsmoke2
LOGD=logs/sst_hsmm_v1/pv5_hcgsmoke2
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $R $LOGD
  for d in en2zh zh2en; do head -n 3 $D0/edev/data/items_$d.jsonl > $R/items_$d.jsonl; done
  start_watcher
fi
ps_=()
for d in en2zh zh2en; do
  m=heal; [ $d = zh2en ] && m=base
  ps_+=("$(launch 0 smoke_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYV -m experiments.sst_hsmm_v1.hsc24_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/pfe1/h/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/edev/views/$d \
    --thinker eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 --direction $d --gpu-mem 0.6 --variants $m@0.70 \
    --draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --mid-mbr 4 --mbr-util hcg --items-file $R/items_$d.jsonl --out $R/run_$d)")
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do
  ls $R/run_$d/blocks_*.jsonl > /dev/null 2>&1 || die "smoke $d: no blocks"
  grep -q '"hcg_base_chosen"\|"mid_mbr_reads"' $R/run_$d/*.json || die "smoke $d: no hcg stats"
done
grep -h -o '"\(hcg_[a-z_]*\|mid_mbr_[a-z_]*\)": [0-9]*' $R/run_*/*.json 2>/dev/null | sort | uniq -c | head
touch $D0/pv5/HCG2_SMOKE_OK
log "hcg smoke ok"
