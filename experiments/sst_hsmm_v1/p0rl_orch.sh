#!/usr/bin/env bash
# prereg change 40-P0: shared fork precheck. hsc11_si_thinker (p3h8 policy: n2, h14, theta 0.75, SAP v/lora/a1.0000, enh views,
# wave17) with --fork-reads 2 --fork-k 4 --fork-temp 1.0 --fork-top-p 1.0 on eval_outputs/sst_hsmm_v1/p0rl/data (e1 first 500 /
# direction), 4 shards per direction = 8 jobs on cards $CARDS (default 0-7), then p0rl_analyze -> p0rl/p0_result.json.
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/p0rl
R=$E/orch
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
PRM=eval_outputs/sst_hsmm_v1/h14/params.json
LOGD=logs/sst_hsmm_v1/p0rl
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  for d in en2zh zh2en; do
    $PYQ -I - $E/data/items_$d.jsonl $E/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 4) for k in range(5)]
for k in range(4):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(sys.argv[3], n, b)
PY
  done
  start_watcher
fi
ps_=(); CA=(${CARDS:-0 1 2 3 4 5 6 7}); ci=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  g=${CA[$((ci % ${#CA[@]}))]}
  ps_+=("$(launch $g p0_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc11_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $PRM --theta 0.75 --verifier $VER --second-view enh --views eval_outputs/sst_hsmm_v1/e1/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --fork-reads 2 --fork-k 4 --fork-temp 1.0 --fork-top-p 1.0 --items-file $E/data/items_${d}_$k.jsonl --out $E/run_${d}_$k)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  [ -s $E/run_${d}_$k/blocks_${d}_w1.jsonl ] && [ -s $E/run_${d}_$k/forks_${d}.jsonl ] || die "p0 $d $k"
done; done
touch $STOP
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.p0rl_analyze --root $E --out $E/p0_result.json > $LOGD/analyze.log 2>&1 || die "analyze"
touch $R/DONE; log "p0rl done"
