#!/usr/bin/env bash
# prereg change 44 (P0-K): hsc15 --variants base,snap,heal (p3h8 policy) on edev en2zh (all 1292), 8 shards on cards 0-7,
# then p0k_analyze -> p0k/p0k_result.json; p0k/DONE. Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
E=$D0/p0k
R=$E
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
LOGD=logs/sst_hsmm_v1/p0k
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $E ] && { echo "exists: $E" >&2; exit 1; }
  mkdir -p $LOGD $E/data; git rev-parse HEAD > $E/commit.txt
  $PYQ -I - $D0/edev/data/items_en2zh.jsonl $E/data <<'PY' || die "split"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
open(f"{sys.argv[2]}/items_en2zh.jsonl", "x", encoding="utf-8").writelines(rows)
n = len(rows); b = [round(k * n / 8) for k in range(9)]
for k in range(8):
    with open(f"{sys.argv[2]}/items_en2zh_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(n, b)
PY
  start_watcher
fi
ps_=()
for k in 0 1 2 3 4 5 6 7; do
  ps_+=("$(launch $k p0k_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$k $PYV -m experiments.sst_hsmm_v1.hsc15_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/h14/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/edev/views/en2zh --thinker $NTH \
    --direction en2zh --gpu-mem 0.6 --variants base,snap,heal --items-file $E/data/items_en2zh_$k.jsonl --out $E/run_$k)")
  [ -n "$DRY" ] || sleep 45
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for k in 0 1 2 3 4 5 6 7; do [ -s $E/run_$k/blocks_en2zh_heal.jsonl ] || die "p0k shard $k"; done
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.p0k_analyze --root $E --data $E/data --out $E/p0k_result.json > $LOGD/analyze.log 2>&1 || die "analyze"
touch $E/DONE; log "p0k done"
