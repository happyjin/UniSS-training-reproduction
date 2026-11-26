#!/usr/bin/env bash
# prereg change 45: hsc16 variants on edev. en2zh (1292) x {base@0.75, heal@0.75/0.70/0.65/0.60, cec@0.75, pandora@0.75@{0.10,0.15,0.20}}
# 5 shards on cards 0-4; zh2en (2000) x {base@0.75, pandora@0.75@{0.10,0.15,0.20}} 3 shards on cards 5-7; then p45_analyze ->
# p45/p45_result.json; p45/DONE. Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
E=$D0/p45
R=$E
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
LOGD=logs/sst_hsmm_v1/p45
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
declare -A NS=([en2zh]=5 [zh2en]=3)
declare -A VS=([en2zh]="base@0.75,heal@0.75,heal@0.70,heal@0.65,heal@0.60,cec@0.75,pandora@0.75@0.10,pandora@0.75@0.15,pandora@0.75@0.20" \
               [zh2en]="base@0.75,pandora@0.75@0.10,pandora@0.75@0.15,pandora@0.75@0.20")
if [ -z "$DRY" ]; then
  [ -e $E ] && { echo "exists: $E" >&2; exit 1; }
  mkdir -p $LOGD $E/data; git rev-parse HEAD > $E/commit.txt
  for d in en2zh zh2en; do
    $PYQ -I - $D0/edev/data/items_$d.jsonl $E/data $d ${NS[$d]} <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
d, K = sys.argv[3], int(sys.argv[4])
open(f"{sys.argv[2]}/items_{d}.jsonl", "x", encoding="utf-8").writelines(rows)
n = len(rows); b = [round(k * n / K) for k in range(K + 1)]
for k in range(K):
    with open(f"{sys.argv[2]}/items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(d, n, b)
PY
  done
  start_watcher
fi
ps_=(); ci=0
for d in en2zh zh2en; do for k in $(seq 0 $((${NS[$d]} - 1))); do
  ps_+=("$(launch $ci p45_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$ci $PYV -m experiments.sst_hsmm_v1.hsc16_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/h14/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/edev/views/$d --thinker $NTH \
    --direction $d --gpu-mem 0.6 --variants "${VS[$d]}" --items-file $E/data/items_${d}_$k.jsonl --out $E/run_${d}_$k)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do for k in $(seq 0 $((${NS[$d]} - 1))); do [ -s $E/run_${d}_$k/blocks_${d}_base_0.75.jsonl ] || die "p45 $d $k"; done; done
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.p45_analyze --root $E --data $E/data --out $E/p45_result.json > $LOGD/analyze.log 2>&1 || die "analyze"
touch $E/DONE; log "p45 done"
