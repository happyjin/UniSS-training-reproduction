#!/usr/bin/env bash
# change 66 HCG-open (hsc28 --hcg-open, p2ft adapter + tags, pfe1 params) smoke on the card pool -- 20 edev items per direction; checks
# blocks + hcg statistics and the pre-registered degeneracy rule: if the base-open candidate wins >= 90% of the scored non-final reads
# in BOTH directions, HCG-open = the base drafter (EGA head, already measured) -> pv10/OPEN_DEGENERATE (no streaming); else
# pv10/OPEN_SMOKE_OK. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv10/opensmoke
LOGD=logs/sst_hsmm_v1/pv10_opensmoke
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $R $LOGD
  for d in en2zh zh2en; do head -n 20 $D0/edev/data/items_$d.jsonl > $R/items_$d.jsonl; done
  start_watcher
fi
ps_=()
for d in en2zh zh2en; do
  m=heal; [ $d = zh2en ] && m=base
  ps_+=("$(launch 0 smoke_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYV -m experiments.sst_hsmm_v1.hsc28_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/pfe1/h/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/edev/views/$d \
    --thinker eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 --direction $d --gpu-mem 0.6 --variants $m@0.70 \
    --draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --mid-mbr 4 --mbr-util hcg --hcg-open --items-file $R/items_$d.jsonl --out $R/run_$d)")
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do
  ls $R/run_$d/blocks_*.jsonl > /dev/null 2>&1 || die "smoke $d: no blocks"
  grep -q '"mid_mbr_reads"' $R/run_$d/*.json || die "smoke $d: no hcg stats"
done
v=$($PYQ - $R <<'PY'
import glob, json, sys
deg = []
for d in ("en2zh", "zh2en"):
    j = json.load(open(glob.glob(f"{sys.argv[1]}/run_{d}/*.json")[0]))
    rate = j.get("hcg_base_chosen", 0) / max(1, j.get("mid_mbr_reads", 0))
    print(d, "base-open chosen", j.get("hcg_base_chosen", 0), "of", j.get("mid_mbr_reads", 0), f"= {rate:.3f}", "empty", j.get("hcg_open_empty", 0), file=sys.stderr)
    deg.append(rate >= 0.90)
print("DEG" if all(deg) else "OK")
PY
)
log "open smoke verdict $v"
mkdir -p $D0/pv10
if [ "$v" = DEG ]; then touch $D0/pv10/OPEN_DEGENERATE; else touch $D0/pv10/OPEN_SMOKE_OK; fi
