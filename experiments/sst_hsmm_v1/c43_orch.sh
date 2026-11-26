#!/usr/bin/env bash
# prereg change 43: P0-R (hsc14 forced-commit forks at every non-final read, k in {0,1,2,4}, p3h8 policy, e1 lines 501-1000,
# 3 shards per direction on cards 0-5) and P0-T (si_thinker_stream --policy la2 --windows 1,0.5 on edev first 500, cards 6-7)
# in parallel; then p0r_analyze -> p0r/p0r_result.json and p0t_analyze -> p0t/p0t_result.json; c43/DONE.
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
ER=$D0/p0r
ET=$D0/p0t
R=$D0/c43
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
LOGD=logs/sst_hsmm_v1/c43
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  for d in en2zh zh2en; do
    $PYQ -I - $ER/data/items_$d.jsonl $ER/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 3) for k in range(4)]
for k in range(3):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(sys.argv[3], n, b)
PY
  done
  start_watcher
fi
ps_=(); ci=0
for d in en2zh zh2en; do for k in 0 1 2; do
  ps_+=("$(launch $ci p0r_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$ci $PYV -m experiments.sst_hsmm_v1.hsc14_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/h14/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/e1/views/$d --thinker $NTH \
    --direction $d --gpu-mem 0.6 --fork-reads 99 --force-ks 0,1,2,4 --items-file $ER/data/items_${d}_$k.jsonl --out $ER/run_${d}_$k)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done; done
for d in en2zh zh2en; do
  ps_+=("$(launch $ci p0t_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$ci $PYV -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --policy la2 \
    --windows 1,0.5 --direction $d --thinker $NTH --items-file $ET/data/items_$d.jsonl --out $ET/$d --gpu-mem 0.6)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
ok=1
for d in en2zh zh2en; do for k in 0 1 2; do [ -s $ER/run_${d}_$k/forks_$d.jsonl ] || ok=0; done; done
[ $ok = 1 ] && { PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.p0r_analyze --root $ER --data $ER/data --out $ER/p0r_result.json > $LOGD/p0r_analyze.log 2>&1 || log "p0r analyze FAILED"; } || log "p0r outputs missing"
[ -s $ET/en2zh/blocks_en2zh_w0.5.jsonl ] && [ -s $ET/zh2en/blocks_zh2en_w0.5.jsonl ] && { PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.p0t_analyze --root $ET --data $ET/data --out $ET/p0t_result.json > $LOGD/p0t_analyze.log 2>&1 || log "p0t analyze FAILED"; } || log "p0t outputs missing"
touch $R/DONE; log "c43 done"
