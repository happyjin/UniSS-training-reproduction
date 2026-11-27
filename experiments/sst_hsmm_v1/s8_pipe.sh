#!/usr/bin/env bash
# change 54 (S8, anticipation-agreement certificate), cards 0-7. Base drafter = wave17 everywhere; the P2F adapter ($DL) only
# drafts an extra evidence copy of every non-final read.
#  1. N-policy rollouts (hsc21_roll --p2f-lora) on the h1 shards -> $R/h/roll_*  (stores draft_p2f)
#  2. hsmm_p2f fit (HSC6 + pb emission; precheck reported, not gating) -> $R/h/params.json
#  3. edev hsc21_si_thinker --p2f-lora: en2zh heal@0.75..0.60 (cards 0-2), zh2en base@0.75..0.60 (cards 3-5) -> $R/dev ;
#     drafter_analyze (pick_h10 rule vs wave17) -> $R/dev/choice.json
#  4. test (regardless): experiments/sst_hsmm_v1/p3${TAG}_report.sh -> eval_outputs/sst_hsmm_v1/p3$TAG
# env: TAG DL, optional WAIT (file). Placeholders stopped by PID (lib.sh). Refuses existing outputs.
#   TAG=aac DL=.. setsid nohup bash experiments/sst_hsmm_v1/s8_pipe.sh > logs/sst_hsmm_v1/$TAG.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${TAG:?}" "${DL:?}"
D0=eval_outputs/sst_hsmm_v1
R=$D0/$TAG
H=$R/h
SH=$D0/h1/run/data
E1=$D0/e1
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=$D0/v/lora/a1.0000
REP=experiments/sst_hsmm_v1/p3${TAG}_report.sh
LOGD=logs/sst_hsmm_v1/$TAG
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
[ -f "$REP" ] && [ -s "$DL/adapter_config.json" ] || { echo "missing $REP / $DL" >&2; exit 1; }
if [ -z "$DRY" ]; then
  for x in $R $D0/p3$TAG $LOGD; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  if [ -n "${WAIT:-}" ]; then
    log "$TAG $$: waiting for $WAIT"
    until [ -e "$WAIT" ]; do sleep 60; done
  fi
  git rev-parse HEAD > $R/commit.txt
  mkdir -p $H $R/dev/data
  start_watcher
fi
# 1. rollouts
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc21_roll --mode rollout --p2f-lora $DL \
    --verifier $VER --second-view enh --views $E1/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
  grep -q draft_p2f $H/roll_en2zh_0/blocks_en2zh_w1.jsonl || die "no draft_p2f in rollouts"
  log "rollouts done; fitting (CPU)"
  # 2. fit
  OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_p2f fit --roots "$H/roll_*" --pre-out $H/precheck.json \
    --out $H/params.json > $LOGD/fit.log 2>&1 || die "fit"
  [ -s $H/params.json ] || die "params"
  log "fit: $(grep -E '^(ref_masked_80|p2f_80)' $LOGD/fit.log | python3 -c "import sys,json;[print(l.split()[0], {k:round(v,4) for k,v in json.loads(l.split(' ',1)[1]).items() if k in ('auc','ece')}) for l in sys.stdin]" | tr '\n' ' ') $(grep precheck $LOGD/fit.log)"
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
# 3. edev
VE="heal@0.75,heal@0.70,heal@0.65,heal@0.60"
VB="base@0.75,base@0.70,base@0.65,base@0.60"
ps_=(); g=0
for d in en2zh zh2en; do
  if [ $d = en2zh ]; then V=$VE; else V=$VB; fi
  for k in 0 1 2; do
    ps_+=("$(launch $g dev_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc21_si_thinker --mode hsc --variant n2 --samples 0 \
      --params $H/params.json --p2f-lora $DL --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/$d --thinker $NTH \
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
# 4. test (regardless of the dev verdict)
bash $REP > logs/sst_hsmm_v1/p3${TAG}_report.log 2>&1 < /dev/null 8>&- 9>&-
log "p3$TAG rc=$?"
touch $R/DONE
