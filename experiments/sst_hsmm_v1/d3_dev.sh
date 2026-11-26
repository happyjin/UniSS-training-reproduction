#!/usr/bin/env bash
# prereg change 46 (D3), stages 2-3: waits for d3/TRAIN_DONE ->
#  2a. N-policy rollouts of the D3 drafter (hsc_si_thinker --mode rollout, thinker d3/merged, SAP verifier v/lora/a1.0000 unchanged)
#      on h1's e1 shards, cards 0-7 -> d3/h/roll_<d>_<k>
#  2b. hsmm_anchor fit --no-gate (h14 settings) -> d3/h/params.json (+ precheck.json, report only)
#  2c. edev hsc16: D3 en2zh heal@{0.75,0.70,0.65,0.60} (3 shards, cards 0-2), D3 zh2en base@{0.75,0.70,0.65,0.60} (3 shards, cards 3-5),
#      wave17 zh2en base@{0.75,0.70,0.65,0.60} control (h14, 2 shards, cards 6-7) -> d3/dev, d3/dev_w17
#  2d. d3_analyze -> d3/dev/choice.json (frontier verdict + pick_h10 selection), then stage 3: p3d3_report.sh (test regardless).
# Placeholders stopped by PID (lib.sh). Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/d3_dev.sh > logs/sst_hsmm_v1/d3_dev.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/d3
H=$R/h
E1=$D0/e1
SH=$D0/h1/run/data
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
DTH=$R/merged
VER=$D0/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/d3dev
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  for x in $H $R/dev $R/dev_w17; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  mkdir -p $LOGD
  log "d3_dev $$: waiting for $R/TRAIN_DONE"
  until [ -e $R/TRAIN_DONE ]; do
    for f in NOGO DIVERGED FAILED; do [ -e $R/$f ] && { log "d3 stage 0-1 ended with $f; d3_dev stops"; exit 0; }; done
    sleep 60
  done
  mkdir -p $H $R/dev/data $R/dev_w17/data
  git rev-parse HEAD > $H/commit.txt
  start_watcher
fi
# 2a. rollouts
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc_si_thinker --mode rollout \
    --verifier $VER --second-view enh --views $E1/views/$d --thinker $DTH --direction $d --gpu-mem 0.6 \
    --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
  log "rollouts done; fitting (CPU)"
  OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_anchor fit --roots "$H/roll_*" --pre-out $H/precheck.json \
    --out $H/params.json --no-gate > $LOGD/fit.log 2>&1 || die "fit"
  [ -s $H/params.json ] || die "params"
  log "fit done: $(grep -h 'precheck' $LOGD/fit.log | tail -n 1)"
  for d in en2zh zh2en; do
    $PYQ -I - $D0/edev/data/items_$d.jsonl $R/dev/data $R/dev_w17/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
d = sys.argv[4]
for root, K in ((sys.argv[2], 3), (sys.argv[3], 2)):
    if d == "en2zh" and root == sys.argv[3]:
        continue
    open(f"{root}/items_{d}.jsonl", "x", encoding="utf-8").writelines(rows)
    n = len(rows); b = [round(k * n / K) for k in range(K + 1)]
    for k in range(K):
        with open(f"{root}/items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(rows[b[k]:b[k + 1]])
    print(root, d, n, b)
PY
  done
fi
# 2c. edev
VD3E="heal@0.75,heal@0.70,heal@0.65,heal@0.60"
VB="base@0.75,base@0.70,base@0.65,base@0.60"
ps_=(); g=0
for k in 0 1 2; do
  ps_+=("$(launch $g dev_en2zh_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc16_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $H/params.json --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/en2zh --thinker $DTH \
    --direction en2zh --gpu-mem 0.6 --variants "$VD3E" --items-file $R/dev/data/items_en2zh_$k.jsonl --out $R/dev/run_en2zh_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done
for k in 0 1 2; do
  ps_+=("$(launch $g dev_zh2en_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc16_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $H/params.json --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/zh2en --thinker $DTH \
    --direction zh2en --gpu-mem 0.6 --variants "$VB" --items-file $R/dev/data/items_zh2en_$k.jsonl --out $R/dev/run_zh2en_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done
for k in 0 1; do
  ps_+=("$(launch $g w17_zh2en_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc16_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $D0/h14/params.json --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/zh2en --thinker $NTH \
    --direction zh2en --gpu-mem 0.6 --variants "$VB" --items-file $R/dev_w17/data/items_zh2en_$k.jsonl --out $R/dev_w17/run_zh2en_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for k in 0 1 2; do [ -s $R/dev/run_en2zh_$k/blocks_en2zh_heal_0.60.jsonl ] && [ -s $R/dev/run_zh2en_$k/blocks_zh2en_base_0.60.jsonl ] || die "dev $k"; done
for k in 0 1; do [ -s $R/dev_w17/run_zh2en_$k/blocks_zh2en_base_0.60.jsonl ] || die "dev_w17 $k"; done
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.d3_analyze --out $R/dev/choice.json > $LOGD/analyze.log 2>&1 || die "analyze"
log "dev done: $(tail -n 3 $LOGD/analyze.log | tr '\n' ' ' | cut -c1-900)"
touch $R/dev/DONE
# 3. test (regardless of the dev verdict, change 46)
bash experiments/sst_hsmm_v1/p3d3_report.sh > logs/sst_hsmm_v1/p3d3_report.log 2>&1 < /dev/null 8>&- 9>&-
log "p3d3 rc=$?"
