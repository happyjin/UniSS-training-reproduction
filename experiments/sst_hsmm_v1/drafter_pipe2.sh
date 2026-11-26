#!/usr/bin/env bash
# change 49+: drafter_pipe.sh + STOP_AFTER_FIT=1 (stop after rollouts + HSMM fit; touch $R/h/FIT_DONE; no report needed).
# Was: change 47+: generic drafter-scheme pipeline (D3 stages 2-3 for any merged drafter), cards 0-7.
#  env: TAG (root eval_outputs/sst_hsmm_v1/$TAG), DTH (merged drafter dir), optional WAIT (file to wait for; FAILED/NOGO/DIVERGED
#       next to it stop), optional ADAPTERS + SCALE (merge.py --scale-mult) when DTH must be built, optional EXTRA_OFF ("name=dir ...")
#  1. merge (if ADAPTERS)  2. whole-audio dev decodes (wave2_S1 dev, dev_offline2) of DTH (+ EXTRA_OFF), offdev_score -> $R/off/score.json
#  3. N-policy rollouts on h1 shards -> $R/h/roll_*, hsmm_anchor fit --no-gate -> $R/h/params.json
#  4. edev hsc16: en2zh heal@0.75..0.60 (cards 0-2), zh2en base@0.75..0.60 (cards 3-5) -> $R/dev ; drafter_analyze -> $R/dev/choice.json
#  5. test (regardless): experiments/sst_hsmm_v1/p3${TAG}_report.sh -> eval_outputs/sst_hsmm_v1/p3$TAG
# Placeholders stopped by PID (lib.sh). Refuses existing outputs.
#   TAG=.. DTH=.. setsid nohup bash experiments/sst_hsmm_v1/drafter_pipe.sh > logs/sst_hsmm_v1/$TAG.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${TAG:?}" "${DTH:?}"
D0=eval_outputs/sst_hsmm_v1
R=$D0/$TAG
H=$R/h
SH=$D0/h1/run/data
E1=$D0/e1
VER=$D0/v/lora/a1.0000
REP=experiments/sst_hsmm_v1/p3${TAG}_report.sh
LOGD=logs/sst_hsmm_v1/$TAG
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
[ -n "${STOP_AFTER_FIT:-}" ] || [ -f "$REP" ] || { echo "missing $REP" >&2; exit 1; }
if [ -z "$DRY" ]; then
  for x in $R/off $H $R/dev $D0/p3$TAG; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  if [ -n "${WAIT:-}" ]; then
    log "$TAG $$: waiting for $WAIT"
    until [ -e "$WAIT" ]; do
      for f in NOGO DIVERGED FAILED; do [ -e "$(dirname "$WAIT")/$f" ] && { log "upstream ended with $f; $TAG stops"; touch $R/FAILED; exit 0; }; done
      sleep 60
    done
  fi
  git rev-parse HEAD > $R/commit.txt
  if [ -n "${ADAPTERS:-}" ]; then
    [ -e "$DTH" ] && die "exists $DTH"
    PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters "$ADAPTERS" --scale-mult "${SCALE:-1.0}" --out "$DTH" > $LOGD/merge.log 2>&1 || die "merge"
    grep -q '"adapted"' $LOGD/merge.log || die "merge check"
    log "merged: $(tail -n 1 $LOGD/merge.log)"
  fi
  [ -s "$DTH/model.safetensors.index.json" ] || die "no drafter $DTH"
  mkdir -p $R/off $H $R/dev/data
  start_watcher
fi
# 2. whole-audio dev decodes
MODELS=("$TAG=$DTH" ${EXTRA_OFF:-})
NM=${#MODELS[@]}; NSH=$((4 / NM)); [ $NSH -lt 1 ] && NSH=1
ps_=(); g=0
for m in "${MODELS[@]}"; do nm=${m%%=*}; md=${m#*=}
  for d in en2zh zh2en; do for k in $(seq 0 $((NSH - 1))); do
    ps_+=("$(launch $g off_${nm}_${d}_$k "${VENV[@]}" OFF_MODEL=$md CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.dev_offline2 gen \
      --direction $d --shard $k --num-shards $NSH --gpu-mem 0.85 --out $R/off/${nm}_${d}_$k.jsonl)")
    g=$((g + 1)); [ -n "$DRY" ] || sleep 30
  done; done
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  SYS=("w17=$D0/devoff/offline_{d}_*.jsonl")
  for m in "${MODELS[@]}"; do nm=${m%%=*}; SYS+=("$nm=$R/off/${nm}_{d}_*.jsonl"); done
  $PYE experiments/sst_hsmm_v1/offdev_score.py --out $R/off/score.json "${SYS[@]}" > $LOGD/offscore.log 2>&1 || die "offscore"
  log "offline dev: $(grep ' all ' $LOGD/offscore.log | tr '\n' ' ')"
fi
# 3. rollouts + fit
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
  if [ -n "${STOP_AFTER_FIT:-}" ]; then touch $STOP $H/FIT_DONE; log "fit done (STOP_AFTER_FIT)"; exit 0; fi
  for d in en2zh zh2en; do
    $PYQ -I - $D0/edev/data/items_$d.jsonl $R/dev/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
root, d, K = sys.argv[2], sys.argv[3], 3
open(f"{root}/items_{d}.jsonl", "x", encoding="utf-8").writelines(rows)
n = len(rows); b = [round(k * n / K) for k in range(K + 1)]
for k in range(K):
    with open(f"{root}/items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
PY
  done
fi
# 4. edev
VE="heal@0.75,heal@0.70,heal@0.65,heal@0.60"
VB="base@0.75,base@0.70,base@0.65,base@0.60"
ps_=(); g=0
for d in en2zh zh2en; do
  if [ $d = en2zh ]; then V=$VE; else V=$VB; fi
  for k in 0 1 2; do
    ps_+=("$(launch $g dev_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc16_si_thinker --mode hsc --variant n2 --samples 0 \
      --params $H/params.json --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/$d --thinker $DTH \
      --direction $d --gpu-mem 0.6 --variants "$V" --items-file $R/dev/data/items_${d}_$k.jsonl --out $R/dev/run_${d}_$k)")
    g=$((g + 1)); [ -n "$DRY" ] || sleep 40
  done
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for k in 0 1 2; do [ -s $R/dev/run_en2zh_$k/blocks_en2zh_heal_0.60.jsonl ] && [ -s $R/dev/run_zh2en_$k/blocks_zh2en_base_0.60.jsonl ] || die "dev $k"; done
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.drafter_analyze --root $R/dev --out $R/dev/choice.json > $LOGD/analyze.log 2>&1 || die "analyze"
log "dev done: $(python3 -c "import json;c=json.load(open('$R/dev/choice.json'));print({d:(c[d]['spec'],c[d]['effective']) for d in c})")"
touch $R/dev/DONE
# 5. test (regardless of the dev verdict)
bash $REP > logs/sst_hsmm_v1/p3${TAG}_report.log 2>&1 < /dev/null 8>&- 9>&-
log "p3$TAG rc=$?"
