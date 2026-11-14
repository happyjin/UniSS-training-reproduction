#!/usr/bin/env bash
# PLAN B COPY (2026-11-14, user: "先A后B，8卡"): joint_stage2.sh with (1) run_crf_v2p_lat.sh (the v2' posterior with
# a per-window latency budget, crf/serial_v2p_lat.py, ratio V2P_LAT_RATIO, required), (2) the marker JOINT_S2LAT_SELECTED
# instead of JOINT_S2_SELECTED (the method's marker is kept), (3) ranks() matching the new trainer entry. Use with
# S2_FULL=1 CARDS=0,1,2,3,4,5,6,7 (the stage-2 8-card recipe: 64 rows / step, lr 1e-4 -> 1e-5 over 6000, warm-up 100;
# dev5 proxies from step 3000 stacked on the 8 training cards). Start only after plan A's evaluations have finished
# (the slot claims exclude 0-7 from evaluation orchestrators). Every other line is joint_stage2.sh's.
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
# S2_FULL=1 (prereg 修正 4, user 2026-11-13 ~08:10-08:45 Prague; default empty = everything above, unchanged):
#   - 8 cards (CARDS=0,1,2,3,4,5,6,7): stage 1's slotA + slotB owner files are handed over (renamed, new claims)
#   - the A' recipe, only the data bidirectional: STEPS=15000 (ceiling) SAVE_EVERY=500, EXTRA adds --seed2 20261109
#     --lr-horizon 6000 --warmup 100 (lr flat at 1e-5 after 6000); rows per step 64 (any number of cards)
#   - dev5 proxies only for steps >= 3000, stacked on the 8 training cards in stack_cards.py's physical-memory mode:
#     at the first checkpoint >= 3000 the driver writes <run>/proxy_stack.conf (stack file $SL/stack_cards_s2_<run>,
#     group <run>/stack_group) and starts stack47_watchdog.py --phys --cards 0..7 --file-cards 7 --cap-gib 1000
#     --min-free-gib 1.5 (trips on < 1.5 GiB free or a training-log error; stops only stacked jobs by PID). Trial: the
#     first stacked job runs on card 7 alone; once one has run 10 min there without a trip the file is widened to all
#     8 cards (new file + rename) and the measured peak is logged (JOINT_STACK_TRIAL). Trip, or no stacked start on
#     card 7 within 45 min -> JOINT_STACK_FAILED (LOG.zh-CN.md line; no second try, no batch change; the training
#     goes on; the user decides). The A' s3000 dev5 reference (not finished by the old stage-1 driver) is decoded at the
#     same time under a new name (jref_dev5_aprime_s3000_s2, stack rank first) -> refs/aprime.ok
#   - rule: joint_select.py --stage s2 --first-step 3000 (refs A' s3000 / (a)'s s2000), the original early stop
#     (eps 0.2, 2 points without improvement, ceiling 15000), STOP file, selection and marker as above
#   S2_FULL=1 CARDS=0,1,2,3,4,5,6,7 RUN_ID=joint_v2p_<UTC> [DRY_RUN=1] bash .../joint_stage2.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/stage2.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
S2_FULL=${S2_FULL:-}
V2P_LAT_RATIO=${V2P_LAT_RATIO:?plan B: set V2P_LAT_RATIO in [0, 1]}
RUN_ID=${RUN_ID:?}; CARDS=${CARDS:-4,5,6,7}; PORT=${PORT:-29995}; PEAK_GIB=${PEAK_GIB:-120}
if [ -n "$S2_FULL" ]; then
[ "$CARDS" = 0,1,2,3,4,5,6,7 ] || { echo "S2_FULL trains on 0,1,2,3,4,5,6,7 (user 2026-11-13)"; exit 1; }
else
[ "$CARDS" = 4,5,6,7 ] || { echo "(b) trains on 4,5,6,7"; exit 1; }
fi
OUT=$CRF/train/$RUN_ID
TL=logs/mist_w3po_v1/$RUN_ID.log
R=$JL/$RUN_ID
RO=$JO/$RUN_ID
SEL=$RO/selection.json
M1=$JL/JOINT_S1_SELECTED
MARK=$JL/JOINT_S2LAT_SELECTED
launch_cmd() {   # <stage-1 merged model>
  if [ -n "$S2_FULL" ]; then
    echo "V2P_LAT_RATIO=$V2P_LAT_RATIO CARDS=$CARDS RUN_ID=$RUN_ID PORT=$PORT STEPS=15000 SAVE_EVERY=500 EXTRA=\"--sched $CRF/sched_v1 --dir zh2en \
--sched2 $CRF/sched_en2zh_v1 --dir2 en2zh --seed2 20261109 --base $1 --lr-horizon 6000 --warmup 100 --stop-file $OUT/STOP\" \
bash experiments/mist_w3po_v1/scripts/run_crf_v2p_lat.sh"
    return
  fi
  echo "V2P_LAT_RATIO=$V2P_LAT_RATIO CARDS=$CARDS RUN_ID=$RUN_ID PORT=$PORT STEPS=15000 SAVE_EVERY=500 EXTRA=\"--sched $CRF/sched_v1 --dir zh2en \
--sched2 $CRF/sched_en2zh_v1 --dir2 en2zh --base $1 --lr-horizon 12000 --stop-file $OUT/STOP\" \
bash experiments/mist_w3po_v1/scripts/run_crf_v2p_lat.sh"
}
ranks() { ps -eo pid=,args= | awk -v s="--out $OUT " '$2 ~ /\/python$/ && $3 != "-m" && $0 ~ /train_(crf_sft|v2p_lat)\.py / && index($0, s) {print $1}'; }   # workers: python -u <script>
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
release_full() {   # S2_FULL: our slotA and slotB
  local sl
  for sl in slotA slotB; do
    [ -e $SL/$sl.owner ] && grep -q "$RUN_ID" $SL/$sl.owner && command mv -f $SL/$sl.owner $SL/$sl.released_$(date -u +%Y%m%dT%H%M%SZ) \
      && log "$sl released"
  done
}
SF=$SL/stack_cards_s2_$RUN_ID; SG=$JL/$RUN_ID/stack_group
AR_ARGS="--ckpt $APRIME_CK --win $WZ_DEV5 --talker none --shards 4 --shard-by window --window 1 --cap135 --stack-cards $SF --stack-rank 1000000 --stack-group $SG"
train_pid() { ps -eo pid=,args= | awk -v s="--out $OUT " '$2 ~ /\/python$/ && $3 == "-m" && $4 == "torch.distributed.run" && index($0, s) {print $1}' | head -1; }
stack_trial() {   # background: card 7 alone first; widen to 0-7 after 10 min of a stacked job there without a trip
  local t0 t1 peak u tot
  t0=$(date +%s)
  until grep -q STACK47_FIRST_DECODE $R/stack_watchdog.log 2>/dev/null; do
    [ -e $SF ] || { log "JOINT_STACK_FAILED: stack file withdrawn before the first stacked job ($(grep STACK47_WITHDRAWN $R/stack_watchdog.log | tail -1 | cut -c1-300))"; return 1; }
    if [ $(( $(date +%s) - t0 )) -gt 2700 ]; then
      u=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1 == 7 {printf "%.1f of %.1f GiB", $2 / 1024, $3 / 1024}')
      log "JOINT_STACK_FAILED: no stacked start on card 7 within 45 min (card 7 now $u; need real use + 67 GiB <= total - 1.5)"
      [ -e $SF ] && command mv -f $SF $SF.withdrawn_$(date -u +%Y%m%dT%H%M%SZ)
      printf '\n- %s 布拉格（(b) 第 2 段驱动）：JOINT_STACK_FAILED —— 第 3000 步起的 dev5 速查叠不上训练卡（卡 7 实占 %s，需 实占 + 67 GiB ≤ 总量 − 1.5 GiB）；训练照常，速查未开，等用户决定（不再试第二次、不改 batch）。\n' \
        "$(TZ=Europe/Prague date '+%F %H:%M')" "$u" >> reports/mist_w3po_v1/LOG.zh-CN.md
      return 1
    fi
    sleep 20
  done
  t1=$(date +%s); peak=0
  while [ $(( $(date +%s) - t1 )) -lt 600 ]; do
    [ -e $SF ] || { log "JOINT_STACK_FAILED: watchdog tripped during the card-7 trial ($(grep STACK47_WITHDRAWN $R/stack_watchdog.log | tail -1 | cut -c1-300)); peak $peak MiB"; return 1; }
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 == 7 {print $2}')
    [ -n "$u" ] && [ "$u" -gt "$peak" ] && peak=$u
    sleep 20
  done
  tot=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$1 == 7 {print $2}')
  ( echo "0 1 2 3 4 5 6 7 phys" > $SF.wide ) && [ -e $SF ] && command mv -f $SF.wide $SF \
    && log "JOINT_STACK_TRIAL card 7: training + stacked job peak $peak MiB of $tot MiB over 10 min, no trip -> widened to 0-7" \
    || { log "JOINT_STACK_FAILED: stack file gone at widening"; return 1; }
}
aprime_ref() {   # background: the A' s3000 dev5 reference under a new name (stacked, rank first); one retry -> refs/aprime.ok|failed
  local rd=$JL/refs/aprime_s2 n=jref_dev5_aprime_s3000_s2
  [ -e $JL/refs/aprime.ok ] && return 0
  [ -e $rd ] || $TORCHZ --run $rd --name $n $AR_ARGS > $JL/refs/aprime_s2.launch.log 2>&1
  wait_done $rd
  if [ -n "$(orch_failed $rd)" ]; then
    n=${n}_r2
    [ -e ${rd}_r2 ] || $TORCHZ --run ${rd}_r2 --name $n $AR_ARGS > $JL/refs/aprime_s2.r2.launch.log 2>&1
    wait_done ${rd}_r2
    [ -z "$(orch_failed ${rd}_r2)" ] || { ( set -o noclobber; echo "failed $(date -u +%FT%TZ)" > $JL/refs/aprime.failed ); return 1; }
  fi
  ( set -o noclobber; echo "$n" > $JL/refs/aprime.ok )
}
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
if [ -n "${DRY_RUN:-}" ] && [ -n "$S2_FULL" ]; then
  MD=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged
  echo "(DRY S2_FULL: stage-1 merged model stands in as $MD)"
  echo "stage-1 marker: $([ -e $M1 ] && cat $M1 || echo 'not yet')"
  echo "slot owners now: A $([ -e $SL/slotA.owner ] && sed -n 2p $SL/slotA.owner | cut -c1-90 || echo none) | B $([ -e $SL/slotB.owner ] && sed -n 2p $SL/slotB.owner | cut -c1-90 || echo none)"
  echo "cards $CARDS non-ready reasons now: $(cards_ready $CARDS $PEAK_GIB | wc -l) lines"
  CARDS=$CARDS LAUNCH="$(launch_cmd $MD)" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh | tail -1 | cut -c1-200
  eval "DRY_RUN=1 $(launch_cmd $MD)"
  echo "optim check on A' step_00500: $(optim_ok $CRF/train/crf_v2p_20261110T075131Z/ckpt/step_00500 2>&1 | tail -1)"
  echo "stack conf: STACK_FILE=$SF STACK_GROUP=$SG; watchdog: $PY $SC/stack47_watchdog.py --stack-file $SF --cards 0 1 2 3 4 5 6 7 --file-cards 7 --phys --train-log $TL --train-pid <torchrun> --cap-gib 1000 --min-free-gib 1.5 --tick 20 --status-s 120"
  echo "proxy (step >= 3000, stacked; DRY without the conf file, which the driver writes at step 3000):"
  STAGE=s2 RUN=$RUN_ID STEP=3000 BASE=$MD SRC=$OUT/ckpt/step_03000 EXCLUDE="" DRY_RUN=1 bash $SC/joint_proxy.sh
  echo "A' reference: $TORCHZ --run $JL/refs/aprime_s2 --name jref_dev5_aprime_s3000_s2 $AR_ARGS"
  $TORCHZ --run /nonexistent/ra --name jref_dev5_aprime_s3000_s2 $AR_ARGS --dry 2>&1 | tail -1
  echo "refs: zh2en $(cat $JL/refs/aprime.ok 2>/dev/null || echo '<refs/aprime.ok: decoded at step 3000>'), en2zh $($PY -c "import json; print(json.load(open('$E2Z/thinker_selection.json'))['chosen_name'])" 2>/dev/null || echo '<(a) selection pending>')"
  echo "select: $PY $SC/joint_select.py --stage s2 --first-step 3000 --ref-zh2en <A'> --ref-en2zh <(a)> --cand ... --out $SEL"
  exit 0
fi
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
if [ -n "$S2_FULL" ]; then
# ---------------------------------------------------------------------------------------------- S2_FULL (修正 4)
if [ -e $SL/slotA.owner ] && grep -q "joint bidirectional (b)" $SL/slotA.owner; then
  command mv -f $SL/slotA.owner $SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ) && log "stage-1 slotA owner file released (handover)"
fi
for sl in slotA slotB; do
  until [ ! -e $SL/$sl.owner ]; do sleep 30; done
  ( set -o noclobber; printf 'who: joint bidirectional (b) executor (mist_w3po_v1 (b), stage 2, 8 cards: user 2026-11-13 decision)\ntask: joint SerialCRF v2p %s on cards %s (horizon 6000, ceiling 15000, dev5 from 3000 stacked on these cards, early stop by STOP file); released by renaming to %s.released_<UTC>\nutc: %s\n' \
    "$RUN_ID" "$CARDS" "$sl" "$(date -u +%FT%TZ)" > $SL/$sl.owner ) && log "$sl.owner claimed" || { release_full; jstop "$sl.owner claim failed"; }
done
t0=$(date +%s)
until [ -z "$(cards_ready $CARDS $PEAK_GIB)" ]; do
  [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && [ $(( $(date +%s) - t0 )) -gt 60 ] && log "waiting for $CARDS: $(cards_ready $CARDS $PEAK_GIB | head -3 | tr '\n' ' ')"
  sleep 30
done
CARDS=$CARDS LAUNCH="$(launch_cmd $MERGED)" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 30); do sleep 10; n=$(ranks | wc -l); [ "$n" -ge 8 ] && break; done
log "stage-2 ranks running: $n of 8 (torchrun $(train_pid))"
[ "$n" -ge 8 ] || { release_full; jstop "training ranks missing ($n; $(tail -2 $TL 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; }
log "JOINT_TRAINING_STARTED s2 $RUN_ID (S2_FULL: 8 cards, horizon 6000, warm-up 100, seed2 20261109, ceiling 15000)"
declare -A LAUNCHED=()
OPTIM_OK=""; LAST=""; CEIL=""; STACK_ON=""
while :; do
  for d in $(ls -d $OUT/ckpt/step_* 2>/dev/null); do
    s=$((10#${d##*step_}))
    [ $((s % 500)) = 0 ] && [ -e $d/DONE ] || continue
    [ -n "${LAUNCHED[$s]:-}" ] && continue
    [ -e $OUT/STOP ] && continue                        # saved after the STOP: recorded only
    if [ -z "$OPTIM_OK" ]; then
      if [ $s = 500 ] && optim_ok $d > $R/optim_check_s500.log 2>&1; then OPTIM_OK=1; log "optimizer state at step 500: $(tail -1 $R/optim_check_s500.log)"
      else stop_clean "optimizer state check failed at $d"; release_full; jstop "optimizer state missing or empty in $d ($(tail -1 $R/optim_check_s500.log | cut -c1-200))"; fi
    fi
    [ $s -lt 3000 ] && continue                         # 修正 4: no dev5 before step 3000
    if [ -z "$STACK_ON" ]; then
      STACK_ON=1
      mkdir -p $SG
      ( set -o noclobber; printf 'STACK_FILE=%s\nSTACK_GROUP=%s\n' $SF $SG > $JL/$RUN_ID/proxy_stack.conf ) || log "proxy_stack.conf exists: kept"
      setsid nohup $PY $SC/stack47_watchdog.py --stack-file $SF --cards 0 1 2 3 4 5 6 7 --file-cards 7 --phys --train-log $TL \
        --train-pid $(train_pid) --cap-gib 1000 --min-free-gib 1.5 --tick 20 --status-s 120 > $R/stack_watchdog.log 2>&1 < /dev/null 9>&- &
      log "stack watchdog pid $! (card 7 trial first)"
      sleep 10
      ( stack_trial ) < /dev/null >> $JL/stage2_stack.log 2>&1 &
      ( aprime_ref ) < /dev/null >> $JL/refs/refs.log 2>&1 &
      log "A' s3000 dev5 reference (stacked, first in order) pid $!"
    fi
    LAUNCHED[$s]=1
    setsid nohup env STAGE=s2 RUN=$RUN_ID STEP=$s BASE=$MERGED SRC=$d EXCLUDE="" bash $SC/joint_proxy.sh \
      < /dev/null >> $R/proxy/s$(printf %05d $s).log 2>&1 9>&- &
    log "step $s: proxy pid $!"
  done
  ZR=$(cat $JL/refs/aprime.ok 2>/dev/null)
  [ -e $JL/refs/aprime.failed ] && { exited || stop_clean "A' reference failed"; release_full; jstop "reference dev5 decode aprime failed twice"; }
  ER=$($PY -c "import json; print(json.load(open('$E2Z/thinker_selection.json'))['chosen_name'])" 2>/dev/null)
  [ -z "$ER" ] && { exited || stop_clean "no (a) selection"; release_full; jstop "(a) has no Thinker selection: no en2zh reference"; }
  CANDS=""
  for s in $(printf '%s\n' "${!LAUNCHED[@]}" | sort -n); do
    S5=$(printf %05d $s)
    if [ -e $R/proxy/s$S5.ok ]; then CANDS="$CANDS --cand $s $(awk '{print $2, $3}' $R/proxy/s$S5.ok) ok"
    elif [ -e $R/proxy/s$S5.failed ]; then CANDS="$CANDS --cand $s s2z_dev5_${RUN_ID}_s$S5 s2e_dev5_${RUN_ID}_s$S5 failed"
    else CANDS="$CANDS --cand $s s2z_dev5_${RUN_ID}_s$S5 s2e_dev5_${RUN_ID}_s$S5 pending"; fi
  done
  if [ -n "$ZR" ] && [ -n "$CANDS" ]; then
    D=$($PY $SC/joint_select.py --stage s2 --first-step 3000 --ref-zh2en $ZR --ref-en2zh $ER $CANDS --out $SEL 2> $R/select.err)
    A=$(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['action'], d.get('prefix_end'), d.get('decision_step'), d.get('chosen'))" 2>/dev/null)
    if [ "$A" != "$LAST" ]; then echo "$D" >> $R/decisions.jsonl; log "rule: $A"; LAST=$A; fi
    case ${A%% *} in
      stop) break;;
      no_feasible) exited || stop_clean "no feasible checkpoint"; release_full; jstop "no feasible checkpoint at the stop point ($A)";;
      ceiling) log "JOINT_CEILING s2 15000"; exited || stop_clean "ceiling"; release_full; jstop "ceiling 15000 reached without an early stop: ask the coordinator";;
    esac
  fi
  if exited && [ -z "$CEIL" ]; then
    if grep -q "CRF_RUN_EXIT rc=0" $TL && [ -e $OUT/ckpt/step_15000/DONE ]; then CEIL=1; release_full; log "training reached the ceiling 15000; waiting for the last proxies"
    else release_full; jstop "training exited on its own ($(grep CRF_RUN_EXIT $TL | tail -1)) before a decision"; fi
  fi
  sleep 60
done
DS=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['decision_step'])")
CH=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['chosen'])")
log "JOINT_EARLY_STOP s2 $DS"
exited || stop_clean "early stop: decision step $DS, chosen $CH"
grep -q "CRF_RUN_EXIT rc=0" $TL || log "note: training exit $(grep CRF_RUN_EXIT $TL | tail -1) (checkpoints up to the stop are complete: DONE files)"
release_full
CK=$OUT/ckpt/step_$(printf %05d $CH)
[ -e $CK/DONE ] || jstop "selected checkpoint $CK missing"
( set -o noclobber; echo "step $CH run $RUN_ID ckpt $CK base $MERGED s1step $S1STEP zname $(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['chosen_names']['zh2en'], 'ename', d['chosen_names']['en2zh'])") utc $(date -u +%FT%TZ)" > $MARK ) \
  || jstop "marker $MARK exists"
log "JOINT_SELECTED s2 $CH ($CK)"
exit 0
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
