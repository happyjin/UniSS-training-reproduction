#!/usr/bin/env bash
# Joint bidirectional run (b), STAGE 2 = the method (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md §3, §5):
# joint SerialCRF v2' on the selected stage-1 model, cards 4-7, open-ended with the pre-registered dev5 early stop.
#   1. wait for logs/mist_w3po_v1/joint/JOINT_S1_SELECTED (joint_stage1.sh; "JOINT_STOP s1" there -> JOINT_STOP s2, exit)
#   2. slotB: stage 1's owner file (ours) is renamed to slotB.released_<UTC> and a new one claimed (noclobber); wait
#      until 4-7 hold only placeholders or paper baseline evals within 135 GiB with the training peak (PEAK_GIB 120 =
#      --mem-cap-gib); takeover_train.sh -> run_crf_v2p.sh (serial_v2p, tau 0.5, --final-read-full-post 1, lr 1e-4 ->
#      1e-5, warm-up 50, seed 20261109, LoRA r32 / alpha 64, TOK 32000, CAP 120, fast-moe, prefetch, per-row features)
#      with STEPS=15000 SAVE_EVERY=500 and EXTRA = --sched crf/sched_v1 --dir zh2en --sched2 crf/sched_en2zh_v1
#      --dir2 en2zh --base <stage-1 merged> --lr-horizon 12000 --stop-file <out>/STOP  ->  JOINT_TRAINING_STARTED s2 <run>
#   3. step 500: optim.pt must exist, load, and hold a non-empty Adam state (exp_avg_sq not all zero); otherwise STOP
#      file (clean stop at a checkpoint), JOINT_STOP
#   4. every checkpoint step_00500, step_01000, ...: one detached joint_proxy.sh (both directions' dev5 text proxies +
#      held-out losses); every minute joint_select.py --stage s2 with the references zh2en = A' s3000's dev5 decode
#      (refs/aprime.ok, started by stage 1), en2zh = (a)'s selected checkpoint's own dev5 proxy (logs/mist_w3po_v1/
#      en2zh/thinker_selection.json chosen_name; EN2ZH_STOP / EN2ZH_PROXY_FAILED before it exists -> STOP file,
#      JOINT_STOP)
#        stop -> <out>/STOP (train_crf_sft saves a checkpoint at the next step and exits: STOP_FILE_SEEN; that
#                checkpoint is recorded only), JOINT_EARLY_STOP s2 <decision step>; after the exit slotB is released,
#                logs/mist_w3po_v1/joint/JOINT_S2_SELECTED and JOINT_SELECTED s2 <step>
#        no_feasible -> STOP file, JOINT_STOP;  ceiling -> JOINT_CEILING + JOINT_STOP (ask the coordinator)
#   RUN_ID=joint_v2p_<UTC> [CARDS=4,5,6,7] [PORT=29995] [PEAK_GIB=120] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_stage2.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/stage2.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
RUN_ID=${RUN_ID:?}; CARDS=${CARDS:-4,5,6,7}; PORT=${PORT:-29995}; PEAK_GIB=${PEAK_GIB:-120}
[ "$CARDS" = 4,5,6,7 ] || { echo "(b) trains on 4,5,6,7"; exit 1; }
OUT=$CRF/train/$RUN_ID
TL=logs/mist_w3po_v1/$RUN_ID.log
R=$JL/$RUN_ID
RO=$JO/$RUN_ID
SEL=$RO/selection.json
M1=$JL/JOINT_S1_SELECTED
MARK=$JL/JOINT_S2_SELECTED
launch_cmd() {   # <stage-1 merged model>
  echo "CARDS=$CARDS RUN_ID=$RUN_ID PORT=$PORT STEPS=15000 SAVE_EVERY=500 EXTRA=\"--sched $CRF/sched_v1 --dir zh2en \
--sched2 $CRF/sched_en2zh_v1 --dir2 en2zh --base $1 --lr-horizon 12000 --stop-file $OUT/STOP\" \
bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
}
ranks() { ps -eo pid=,args= | awk -v s="--out $OUT " '$2 ~ /\/python$/ && $3 ~ /train_crf_sft\.py$/ && index($0, s) {print $1}'; }
exited() { [ -e $TL ] && grep -q CRF_RUN_EXIT $TL; }
release_slot() {
  [ -e $SL/slotB.owner ] && grep -q "$RUN_ID" $SL/slotB.owner && command mv -f $SL/slotB.owner $SL/slotB.released_$(date -u +%Y%m%dT%H%M%SZ) \
    && log "slotB released"
}
stop_clean() {   # STOP file -> a checkpoint at the next step and a clean exit
  [ -e $OUT/STOP ] || ( set -o noclobber; echo "$* utc $(date -u +%FT%TZ)" > $OUT/STOP )
  log "STOP file written ($*); waiting for the training to exit"
  until exited; do sleep 30; done
  log "training exit: $(grep -E 'STOP_FILE_SEEN|CRF_RUN_EXIT' $TL | tail -2 | tr '\n' ' ')"
}
jstop() { log "JOINT_STOP s2: $*"; exit 1; }
optim_ok() {   # <ckpt dir>
  $PY - "$1" <<'EOF'
import sys, torch
from pathlib import Path
f = Path(sys.argv[1], "optim.pt")
assert f.exists() and f.stat().st_size > 0, "optim.pt missing or empty"
d = torch.load(f, map_location="cpu", weights_only=False)
st = d["state"]
assert len(st) > 0, "no Adam state"
sq = sum(float(v["exp_avg_sq"].double().abs().sum()) for v in st.values() if "exp_avg_sq" in v)
assert sq > 0, "exp_avg_sq all zero"
print(f"optim.pt {f.stat().st_size} bytes, {len(st)} parameter states, sum|exp_avg_sq| {sq:.4g}")
EOF
}
if [ -n "${DRY_RUN:-}" ]; then
  MD=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged
  echo "(DRY: stage-1 merged model stands in as $MD)"
  echo "stage-1 marker: $([ -e $M1 ] && cat $M1 || echo 'not yet')"
  echo "cards $CARDS non-ready reasons: $(cards_ready $CARDS $PEAK_GIB | wc -l) lines"
  CARDS=$CARDS LAUNCH="$(launch_cmd $MD)" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh | tail -1 | cut -c1-200
  eval "DRY_RUN=1 $(launch_cmd $MD)"
  echo "optim check on A' step_00500: $(optim_ok $CRF/train/crf_v2p_20261110T075131Z/ckpt/step_00500 2>&1 | tail -1)"
  STAGE=s2 RUN=$RUN_ID STEP=500 BASE=$MD SRC=$OUT/ckpt/step_00500 DRY_RUN=1 bash $SC/joint_proxy.sh
  echo "refs: zh2en $(cat $JL/refs/aprime.ok 2>/dev/null || echo '<refs/aprime.ok pending>'), en2zh $($PY -c "import json; print(json.load(open('$E2Z/thinker_selection.json'))['chosen_name'])" 2>/dev/null || echo '<(a) selection pending>')"
  exit 0
fi
for f in $MARK $SEL $OUT $TL $R $RO; do [ -e $f ] && { echo "拒绝覆盖已有输出: $f"; exit 1; }; done
mkdir -p $R/proxy $RO
log "stage 2 $RUN_ID: waiting for $M1"
until [ -e $M1 ]; do
  grep -q "JOINT_STOP s1" $JL/stage1.log 2>/dev/null && jstop "stage 1 stopped ($(grep 'JOINT_STOP s1' $JL/stage1.log | tail -1 | cut -c1-200))"
  sleep 60
done
S1STEP=$(awk '{print $2}' $M1); MERGED=$(awk '{print $6}' $M1)
[ -e $MERGED/sd_proj.pt ] || jstop "stage-1 merged model missing ($MERGED)"
log "stage-1 selection: step $S1STEP, base $MERGED"
if [ -e $SL/slotB.owner ] && grep -q "joint bidirectional (b)" $SL/slotB.owner; then
  command mv -f $SL/slotB.owner $SL/slotB.released_$(date -u +%Y%m%dT%H%M%SZ) && log "stage-1 slotB owner file released (handover)"
fi
until [ ! -e $SL/slotB.owner ]; do sleep 30; done
( set -o noclobber; printf 'who: joint bidirectional (b) executor (mist_w3po_v1 (b), stage 2)\ntask: joint SerialCRF v2p %s on cards %s (open-ended, dev5 early stop, STOP file); released by renaming to slotB.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$CARDS" "$(date -u +%FT%TZ)" > $SL/slotB.owner ) && log "slotB.owner claimed" || jstop "slotB.owner claim failed"
t0=$(date +%s)
until [ -z "$(cards_ready $CARDS $PEAK_GIB)" ]; do
  [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && [ $(( $(date +%s) - t0 )) -gt 60 ] && log "waiting for $CARDS: $(cards_ready $CARDS $PEAK_GIB | head -3 | tr '\n' ' ')"
  sleep 30
done
CARDS=$CARDS LAUNCH="$(launch_cmd $MERGED)" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 30); do sleep 10; n=$(ranks | wc -l); [ "$n" -ge 4 ] && break; done
log "stage-2 ranks running: $n"
[ "$n" -ge 4 ] || { release_slot; jstop "training ranks missing ($n; $(tail -2 $TL 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; }
log "JOINT_TRAINING_STARTED s2 $RUN_ID"
declare -A LAUNCHED=()
OPTIM_OK=""; LAST=""; CEIL=""
while :; do
  for d in $(ls -d $OUT/ckpt/step_* 2>/dev/null); do
    s=$((10#${d##*step_}))
    [ $((s % 500)) = 0 ] && [ -e $d/DONE ] || continue
    [ -n "${LAUNCHED[$s]:-}" ] && continue
    [ -e $OUT/STOP ] && continue                        # saved after the STOP: recorded only
    if [ -z "$OPTIM_OK" ]; then
      if [ $s = 500 ] && optim_ok $d > $R/optim_check_s500.log 2>&1; then OPTIM_OK=1; log "optimizer state at step 500: $(tail -1 $R/optim_check_s500.log)"
      else stop_clean "optimizer state check failed at $d"; release_slot; jstop "optimizer state missing or empty in $d ($(tail -1 $R/optim_check_s500.log | cut -c1-200))"; fi
    fi
    LAUNCHED[$s]=1
    setsid nohup env STAGE=s2 RUN=$RUN_ID STEP=$s BASE=$MERGED SRC=$d bash $SC/joint_proxy.sh \
      < /dev/null >> $R/proxy/s$(printf %05d $s).log 2>&1 &
    log "step $s: proxy pid $!"
  done
  ZR=$(cat $JL/refs/aprime.ok 2>/dev/null)
  [ -e $JL/refs/aprime.failed ] && { exited || stop_clean "A' reference failed"; release_slot; jstop "reference dev5 decode aprime failed twice"; }
  ER=$($PY -c "import json; print(json.load(open('$E2Z/thinker_selection.json'))['chosen_name'])" 2>/dev/null)
  if [ -z "$ER" ] && [ -n "$(en2zh_failed)" ]; then
    exited || stop_clean "experiment (a) stopped before its selection"; release_slot
    jstop "(a) has no Thinker selection ($(en2zh_failed | cut -c1-200)): no en2zh reference"
  fi
  CANDS=""
  for s in $(printf '%s\n' "${!LAUNCHED[@]}" | sort -n); do
    S5=$(printf %05d $s)
    if [ -e $R/proxy/s$S5.ok ]; then CANDS="$CANDS --cand $s $(awk '{print $2, $3}' $R/proxy/s$S5.ok) ok"
    elif [ -e $R/proxy/s$S5.failed ]; then CANDS="$CANDS --cand $s s2z_dev5_${RUN_ID}_s$S5 s2e_dev5_${RUN_ID}_s$S5 failed"
    else CANDS="$CANDS --cand $s s2z_dev5_${RUN_ID}_s$S5 s2e_dev5_${RUN_ID}_s$S5 pending"; fi
  done
  if [ -n "$ZR" ] && [ -n "$ER" ] && [ -n "$CANDS" ]; then
    D=$($PY $SC/joint_select.py --stage s2 --ref-zh2en $ZR --ref-en2zh $ER $CANDS --out $SEL 2> $R/select.err)
    A=$(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['action'], d.get('prefix_end'), d.get('decision_step'), d.get('chosen'))" 2>/dev/null)
    if [ "$A" != "$LAST" ]; then echo "$D" >> $R/decisions.jsonl; log "rule: $A"; LAST=$A; fi
    case ${A%% *} in
      stop) break;;
      no_feasible) exited || stop_clean "no feasible checkpoint"; release_slot; jstop "no feasible checkpoint at the stop point ($A)";;
      ceiling) log "JOINT_CEILING s2 15000"; exited || stop_clean "ceiling"; release_slot; jstop "ceiling 15000 reached without an early stop: ask the coordinator";;
    esac
  fi
  if exited && [ -z "$CEIL" ]; then
    if grep -q "CRF_RUN_EXIT rc=0" $TL && [ -e $OUT/ckpt/step_15000/DONE ]; then CEIL=1; log "training reached the ceiling 15000; waiting for the last proxies"
    else release_slot; jstop "training exited on its own ($(grep CRF_RUN_EXIT $TL | tail -1)) before a decision"; fi
  fi
  sleep 60
done
DS=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['decision_step'])")
CH=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['chosen'])")
log "JOINT_EARLY_STOP s2 $DS"
exited || stop_clean "early stop: decision step $DS, chosen $CH"
grep -q "CRF_RUN_EXIT rc=0" $TL || log "note: training exit $(grep CRF_RUN_EXIT $TL | tail -1) (checkpoints up to the stop are complete: DONE files)"
release_slot
CK=$OUT/ckpt/step_$(printf %05d $CH)
[ -e $CK/DONE ] || jstop "selected checkpoint $CK missing"
( set -o noclobber; echo "step $CH run $RUN_ID ckpt $CK base $MERGED s1step $S1STEP zname $(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['chosen_names']['zh2en'], 'ename', d['chosen_names']['en2zh'])") utc $(date -u +%FT%TZ)" > $MARK ) \
  || jstop "marker $MARK exists"
log "JOINT_SELECTED s2 $CH ($CK)"
