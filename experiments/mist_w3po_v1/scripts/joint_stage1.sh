#!/usr/bin/env bash
# Joint bidirectional run (b), STAGE 1 = the bidirectional baseline (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md
# §2, §5, §8): joint v8 Simul SFT from O2 on cards 4-7, open-ended with the pre-registered dev5 early stop.
#   1. wait for the start condition: "FT_ALL_DONE ali" and "FT_ALL_DONE ami" in logs/mist_w3po_v1/final_test/chain_v2_*.log
#   2. claim slotB (noclobber owner file; sync_train_exclude.sh then keeps every eval orchestrator off 4-7), build the
#      identity checkpoints of the separate v8 models (CPU) and start the reference dev5 decodes (separate v8 zh2en /
#      en2zh = stage-1 latency references; A' s3000 = the stage-2 zh2en reference) once (a)'s Thinker training on 0-3
#      has exited (all (b) proxies too: 0-3 are used only while they are no training cards)
#   3. wait until 4-7 hold only placeholders, or single-speaker paper baseline evals (inference) whose real use plus the
#      training peak (PEAK_GIB, default 75: 63.6 GiB measured at MBS 8 on 8 cards, logs/multispk/
#      thinker_gstar_s2dl_zh2en_3k_20261030.log) stays <= 135 GiB per card; takeover_train.sh stops the placeholders by
#      PID and starts run_thinker_gstar.sh at once: BASE=O2 INIT_MERGED=1 GSTAR_SEQUENTIAL=1 ROWS=e2e_gstar_stream_bidir_v2
#      NPROC=4 MBS=8 GBS=64 TRAIN_ITERS=15000 LR_DECAY_ITERS=12000 SAVE_INTERVAL=500 SAVE_OPTIM=" " NO_MERGE=1, seed
#      20261026, LoRA r32 / alpha 64, lr 1e-4 -> 1e-5, warm-up 50  ->  JOINT_TRAINING_STARTED s1 <run>
#   4. step 500: joint/check_megatron_optim.py on iter_0000500 (exp_avg / exp_avg_sq for every trainable tensor, non-zero
#      second moment). Not ok -> JOINT_STOP (training stopped by PID at no save, slotB released; no fallback without
#      the optimizer: the user decides)
#   5. every saved iteration (500, 1000, ...): one detached joint_proxy.sh (conversion + both directions' dev5 text
#      proxies, asynchronous); every minute joint_select.py --stage s1 on the finished proxies
#        stop        -> STOP marker, JOINT_EARLY_STOP s1 <decision step>; Megatron stopped by PID when no save is in
#                       progress (the saved iterations are complete; later ones are recorded only)
#        no_feasible -> JOINT_STOP;  ceiling (15000 without a stop) -> JOINT_CEILING + JOINT_STOP (ask the coordinator)
#      training exits on its own before 15000 -> JOINT_STOP
#   6. merge the selected iteration once (merge_thinker_gstar onto O2, 66 GB) + its identity checkpoint ->
#      logs/mist_w3po_v1/joint/JOINT_S1_SELECTED and JOINT_SELECTED s1 <step>. slotB stays claimed for stage 2
#      (joint_stage2.sh takes it over).
#   RUN_ID=joint_v8_<UTC> [CARDS=4,5,6,7] [PORT=29817] [PEAK_GIB=75] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_stage1.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/stage1.log 2>&1 < /dev/null &
# S1_FULL=1 (prereg 修正 4, user 2026-11-13 ~08:20 / ~08:30 Prague; default empty = everything above, unchanged): the strict
#   joint v8 baseline retrained from O2 with the single-direction v8 recipe step for step on 8 cards: CARDS=0,1,2,3,4,5,6,7
#   NPROC=8 MBS=8 GBS=64 (= v8's DP 8 x MBS 8, no accumulation) TRAIN_ITERS=6000 LR_DECAY_ITERS=6000 WARMUP_ITERS=100
#   (each global batch = 32 + 32 samples: lr at joint iteration 2k = v8's lr at k), seed 20261026, the same rows,
#   LoRA r32 / alpha 64, lr 1e-4 -> 1e-5 cosine, SAVE_INTERVAL 500 with the optimizer. slotA AND slotB are claimed
#   (noclobber). No reference decodes, no dev5 proxies, no early stop: the final model is iteration 6000 (as v8 = its
#   last step). Optimizer check at 500 (and at 6000); training exit before 6000 -> JOINT_STOP. Then merge iteration
#   6000 + identity checkpoint -> JOINT_S1_SELECTED (same format; zname / ename "-") and JOINT_SELECTED s1 6000; both
#   slots stay claimed for stage 2 (joint_stage2.sh S2_FULL hands them over).
#   S1_FULL=1 RUN_ID=joint_v8h6k_<UTC> CARDS=0,1,2,3,4,5,6,7 [PORT=29817] [DRY_RUN=1] bash .../joint_stage1.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
S1_FULL=${S1_FULL:-}; S1_ITERS=6000
RUN_ID=${RUN_ID:?}; CARDS=${CARDS:-4,5,6,7}; PORT=${PORT:-29817}; PEAK_GIB=${PEAK_GIB:-75}
if [ -n "$S1_FULL" ]; then
[ "$CARDS" = 0,1,2,3,4,5,6,7 ] || { echo "S1_FULL trains on 0,1,2,3,4,5,6,7 (user 2026-11-13 08:30)"; exit 1; }
else
[ "$CARDS" = 4,5,6,7 ] || { echo "(b) trains on 4,5,6,7"; exit 1; }
fi
REPO=/opt/dlami/nvme/jasonleeeli/projects/UniSS
SAVE=$REPO/checkpoints/mist_w3po_v1/$RUN_ID
ROWS=data/ami_multispk_si_v1/e2e_gstar_stream_bidir_v2/rows_train.jsonl
R=$JL/$RUN_ID
RO=$JO/$RUN_ID
TL=$R/train.log
SEL=$RO/selection.json
MARK=$JL/JOINT_S1_SELECTED
FT=logs/mist_w3po_v1/final_test
LAUNCH="setsid nohup env CUDA_VISIBLE_DEVICES=$CARDS RUN_ID=$RUN_ID ROWS=$ROWS BASE=$O2 INIT_MERGED=1 GSTAR_SEQUENTIAL=1 \
NPROC=4 MBS=8 GBS=64 TRAIN_ITERS=15000 LR_DECAY_ITERS=12000 SAVE_INTERVAL=500 SAVE_OPTIM=' ' NO_MERGE=1 SAVE_DIR=$SAVE \
TB_DIR=$REPO/runs/mist_w3po_v1/$RUN_ID MERGED=$REPO/$RO/unused_final_merge MASTER_PORT=$PORT \
bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > $TL 2>&1 < /dev/null &"
[ -n "$S1_FULL" ] && LAUNCH="setsid nohup env CUDA_VISIBLE_DEVICES=$CARDS RUN_ID=$RUN_ID ROWS=$ROWS BASE=$O2 INIT_MERGED=1 GSTAR_SEQUENTIAL=1 \
NPROC=8 MBS=8 GBS=64 TRAIN_ITERS=$S1_ITERS LR_DECAY_ITERS=$S1_ITERS WARMUP_ITERS=100 SAVE_INTERVAL=500 SAVE_OPTIM=' ' NO_MERGE=1 \
SAVE_DIR=$SAVE TB_DIR=$REPO/runs/mist_w3po_v1/$RUN_ID MERGED=$REPO/$RO/unused_final_merge MASTER_PORT=$PORT \
bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > $TL 2>&1 < /dev/null &"
torchrun_pid() {   # the torch.distributed.run launcher of this run (first argument is the python binary: never a shell)
  ps -eo pid=,args= | awk -v s="--save $SAVE " '$2 ~ /\/python$/ && $3 == "-m" && $4 == "torch.distributed.run" && index($0, s) {print $1}' | head -1
}
workers() {
  ps -eo pid=,args= | awk -v s="--save $SAVE " '$2 ~ /\/python$/ && $3 != "-m" && $0 ~ /pretrain_thinker_gstar\.py / && index($0, s) {print $1}'   # torchrun adds -u
}
tracker() { cat $SAVE/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }
stop_training() {   # by PID, never while a save is in progress (an iter dir newer than the tracker)
  local i t n p left
  for i in $(seq 1 360); do
    t=$(tracker); n=$(ls -d $SAVE/iter_* 2>/dev/null | sed 's/.*iter_0*//' | sort -n | tail -1)
    { [ -z "$n" ] || [ "$n" -le "$t" ]; } && break
    sleep 10
  done
  p=$(torchrun_pid)
  log "stopping the training at tracker $(tracker): torchrun ${p:-gone}, workers $(workers | tr '\n' ' ')"
  [ -n "$p" ] && kill $p
  for i in $(seq 1 30); do left=$(workers); [ -z "$left" ] && [ -z "$(torchrun_pid)" ] && break; sleep 10; done
  for p in $(workers); do kill $p 2>/dev/null && log "worker $p: TERM"; done
  sleep 60
  for p in $(workers); do kill -9 $p 2>/dev/null && log "worker $p: KILL"; done
  log "training stopped; last complete checkpoint iteration $(tracker)"
}
release_slot() {
  [ -e $SL/slotB.owner ] && grep -q "$RUN_ID" $SL/slotB.owner && command mv -f $SL/slotB.owner $SL/slotB.released_$(date -u +%Y%m%dT%H%M%SZ) \
    && log "slotB released"
}
jstop() { log "JOINT_STOP s1: $*"; exit 1; }
ref_job() {   # <tag> <tool> <name> <args...>: one reference dev5 decode, rerun once under <name>_r2 -> refs/<tag>.ok|failed
  local tag=$1 tool=$2 name=$3; shift 3
  local rd=$JL/refs/$tag n=$name
  [ -e $JL/refs/$tag.ok ] && return 0
  until a_thinker_exited; do sleep 60; done   # 0-3 only once (a)'s Thinker training has exited
  if [ ! -e $rd ]; then
    $tool --run $rd --name $n "$@" > $JL/refs/$tag.launch.log 2>&1
  fi
  wait_done $rd
  if [ -n "$(orch_failed $rd)" ]; then
    n=${name}_r2
    [ -e ${rd}_r2 ] || $tool --run ${rd}_r2 --name $n "$@" > $JL/refs/$tag.r2.launch.log 2>&1
    wait_done ${rd}_r2
    [ -z "$(orch_failed ${rd}_r2)" ] || { ( set -o noclobber; echo "failed $(date -u +%FT%TZ)" > $JL/refs/$tag.failed ); return 1; }
  fi
  ( set -o noclobber; echo "$n" > $JL/refs/$tag.ok )
}
IZ=$JO/refs/ident_v8sep_z; IE=$JO/refs/ident_v8sep_e
RZ_ARGS="--ckpt $IZ --win $WZ_DEV5 --talker none --shards 4 --shard-by window --window 1 --exclude 4 5 6 7 --cap135"
RE_ARGS="--set dev5 --ckpt $IE --talker none --shards 4 --shard-by window --window 1 --exclude 4 5 6 7 --cap135"
RA_ARGS="--ckpt $APRIME_CK --win $WZ_DEV5 --talker none --shards 4 --shard-by window --window 1 --exclude 4 5 6 7 --cap135"
if [ -n "${DRY_RUN:-}" ] && [ -n "$S1_FULL" ]; then
  echo "S1_FULL: slotA owner: $([ -e $SL/slotA.owner ] && head -c 120 $SL/slotA.owner || echo none); slotB owner: $([ -e $SL/slotB.owner ] && head -c 120 $SL/slotB.owner || echo none)"
  echo "cards $CARDS now (non-ready reasons):"; cards_ready $CARDS $PEAK_GIB | head -8
  echo "processes on $CARDS by kind: $(others_on $CARDS | awk '{print $4}' | sort | uniq -c | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh | tail -1 | cut -c1-400
  eval "$(echo "$LAUNCH" | sed 's/^setsid nohup env /env DRY_RUN=1 /; s/ > .*$//')" | head -3 | cut -c1-1500
  echo "optimizer checks: check_megatron_optim $SAVE/iter_0000500 and iter_$(printf %07d $S1_ITERS); no proxies, no refs, no early stop"
  echo "merge: merge_thinker_gstar --checkpoint-dir $SAVE/iter_$(printf %07d $S1_ITERS) --rank 32 --alpha 64 --base $O2 --out $RO/merged_s$(printf %05d $S1_ITERS)/thinker_merged ; identity_ckpt -> $MARK"
  exit 0
fi
if [ -n "${DRY_RUN:-}" ]; then
  echo "start condition now: ali $(grep -c 'FT_ALL_DONE ali' $FT/chain_v2_ali.log) ami $(grep -c 'FT_ALL_DONE ami' $FT/chain_v2_ami.log)"
  echo "slotB owner: $([ -e $SL/slotB.owner ] && head -c 120 $SL/slotB.owner || echo none)"
  echo "cards $CARDS now (non-ready reasons):"; cards_ready $CARDS $PEAK_GIB | head -5
  echo "processes on $CARDS by kind: $(others_on $CARDS | awk '{print $4}' | sort | uniq -c | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh | tail -1
  eval "$(echo "$LAUNCH" | sed 's/^setsid nohup env /env DRY_RUN=1 /; s/ > .*$//')" | head -3 | cut -c1-600
  echo "identity: $PY -m experiments.mist_w3po_v1.joint.identity_ckpt --base $V8Z --out $IZ ; --base $V8E --out $IE"
  $TORCHZ --run /nonexistent/rz --name jref_dev5_v8sep_z $RZ_ARGS --dry | tail -1
  $EORCH --run /nonexistent/re --name jref_dev5_v8sep_e $RE_ARGS --dry | tail -1
  $TORCHZ --run /nonexistent/ra --name jref_dev5_aprime_s3000 $RA_ARGS --dry | tail -1
  STAGE=s1 RUN=$RUN_ID STEP=500 BASE=$O2 SRC=$SAVE/iter_0000500 DRY_RUN=1 bash $SC/joint_proxy.sh
  echo "select: $PY $SC/joint_select.py --stage s1 --ref-zh2en <refs/v8z.ok> --ref-en2zh <refs/v8e.ok> --cand ... --out $SEL"
  echo "merge: merge_thinker_gstar --checkpoint-dir $SAVE/iter_<s> --rank 32 --alpha 64 --base $O2 --out $RO/merged_s<s>/thinker_merged"
  exit 0
fi
for f in $MARK $SEL $SAVE $R $RO; do [ -e $f ] && { echo "拒绝覆盖已有输出: $f"; exit 1; }; done
if [ -n "$S1_FULL" ]; then
# ---------------------------------------------------------------------------------------------- S1_FULL (修正 4)
mkdir -p $R $RO
optim_check() {   # <iteration>
  local i=$1 f=$R/optim_check_s$1.json
  $PY -m experiments.mist_w3po_v1.joint.check_megatron_optim $SAVE/iter_$(printf %07d $i) > $f 2>> $R/optim_check.log
}
log "stage 1 (S1_FULL) $RUN_ID: 8 cards, $S1_ITERS iterations, cosine to $S1_ITERS, warm-up 100, final = iteration $S1_ITERS"
for sl in slotA slotB; do
  until [ ! -e $SL/$sl.owner ]; do sleep 30; done
  ( set -o noclobber; printf 'who: joint bidirectional (b) executor (mist_w3po_v1 (b), stage 1 then stage 2)\ntask: joint v8 Simul SFT %s on cards %s (8 cards: user 2026-11-13 08:30 Prague decision; %s iterations, no dev5, final = last iteration; then stage 2 SerialCRF v2p on the same 8 cards); released by renaming to %s.released_<UTC>\nutc: %s\n' \
    "$RUN_ID" "$CARDS" "$S1_ITERS" "$sl" "$(date -u +%FT%TZ)" > $SL/$sl.owner ) && log "$sl.owner claimed" || jstop "$sl.owner claim failed"
done
t0=$(date +%s)
until [ -z "$(cards_ready $CARDS $PEAK_GIB)" ]; do
  [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && [ $(( $(date +%s) - t0 )) -gt 60 ] && log "waiting for $CARDS: $(cards_ready $CARDS $PEAK_GIB | head -3 | tr '\n' ' ')"
  sleep 30
done
log "cards $CARDS ready: $(others_on $CARDS | awk '{print $4}' | sort | uniq -c | tr '\n' ' ')"
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 30); do sleep 10; n=$(workers | wc -l); [ "$n" -ge 8 ] && break; done
log "stage-1 ranks running: $n (torchrun $(torchrun_pid))"
[ "$n" -ge 8 ] || { for sl in slotA slotB; do [ -e $SL/$sl.owner ] && grep -q "$RUN_ID" $SL/$sl.owner && command mv -f $SL/$sl.owner $SL/$sl.released_$(date -u +%Y%m%dT%H%M%SZ); done
  jstop "training ranks missing ($n; $(tail -2 $TL | tr '\n' ' ' | cut -c1-300))"; }
log "JOINT_TRAINING_STARTED s1 $RUN_ID (S1_FULL, 8 cards)"
OPTIM_OK=""; LASTT=0
while :; do
  T=$(tracker)
  if [ -z "$OPTIM_OK" ] && [ "$T" -ge 500 ]; then
    if optim_check 500; then OPTIM_OK=1; log "optimizer state at step 500: ok ($(cut -c1-300 $R/optim_check_s500.json))"
    else
      log "optimizer state at step 500 NOT ok: $(cut -c1-400 $R/optim_check_s500.json) $(tail -1 $R/optim_check.log | cut -c1-200)"
      [ -n "$(torchrun_pid)" ] && stop_training
      jstop "optimizer state missing or empty in iter_0000500 (no fallback without the optimizer: ask the coordinator / user)"
    fi
  fi
  [ "$T" != "$LASTT" ] && { log "checkpoint iteration $T saved"; LASTT=$T; }
  if [ -z "$(torchrun_pid)" ]; then
    sleep 20
    if [ -z "$(torchrun_pid)" ]; then
      [ "$(tracker)" -ge $S1_ITERS ] && grep -q THINKER_GSTAR_DONE_NO_MERGE $TL && break
      jstop "training exited at tracker $(tracker) before $S1_ITERS ($(grep -E 'THINKER_GSTAR' $TL | tail -1))"
    fi
  fi
  sleep 60
done
CH=$S1_ITERS
log "training done: tracker $(tracker) ($(grep THINKER_GSTAR_DONE_NO_MERGE $TL | tail -1))"
optim_check $CH && log "optimizer state at step $CH: ok" || log "note: optimizer check at step $CH not ok ($(tail -1 $R/optim_check.log | cut -c1-200)); the model weights are what is merged"
( set -o noclobber; echo "fixed iteration $CH (prereg 修正 4: no dev5, no early stop; final = last iteration) utc $(date -u +%FT%TZ)" > $RO/STOP )
C5=$(printf %05d $CH); M=$RO/merged_s$C5/thinker_merged; I=$RO/merged_s$C5/identity
log "merging iteration $CH onto O2 -> $M"
CUDA_VISIBLE_DEVICES= $PY experiments/uniss_qwen3_omni_v1/training/merge_thinker_gstar.py --checkpoint-dir $SAVE/iter_$(printf %07d $CH) \
  --rank 32 --alpha 64 --base $O2 --out $M > $R/merge.log 2>&1 || jstop "merge of iteration $CH failed ($(tail -1 $R/merge.log | cut -c1-200))"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.joint.identity_ckpt --base $M --out $I > $R/identity.log 2>&1 \
  || jstop "identity checkpoint of $M failed"
( set -o noclobber; echo "step $CH run $RUN_ID merged $M identity $I zname - ename - utc $(date -u +%FT%TZ)" > $MARK ) || jstop "marker $MARK exists"
log "JOINT_SELECTED s1 $CH ($M)"
exit 0
fi
mkdir -p $R/proxy $RO $JL/refs $JO/refs
log "stage 1 $RUN_ID: waiting for FT_ALL_DONE ali + ami"
until grep -q "FT_ALL_DONE ali" $FT/chain_v2_ali.log 2>/dev/null && grep -q "FT_ALL_DONE ami" $FT/chain_v2_ami.log 2>/dev/null; do sleep 60; done
log "final test done"
until [ ! -e $SL/slotB.owner ]; do sleep 30; done
( set -o noclobber; printf 'who: joint bidirectional (b) executor (mist_w3po_v1 (b), stage 1 then stage 2)\ntask: joint v8 Simul SFT %s on cards %s (open-ended, dev5 early stop; then stage 2 SerialCRF v2p on the same cards); released by renaming to slotB.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$CARDS" "$(date -u +%FT%TZ)" > $SL/slotB.owner ) && log "slotB.owner claimed" || jstop "slotB.owner claim failed"
# ---------------------------------------------------------------------------------------------- references (CPU + waiting)
[ -e $IZ/DONE ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.joint.identity_ckpt --base $V8Z --out $IZ > $JL/refs/ident_z.log 2>&1 \
  || jstop "identity checkpoint of $V8Z ($(tail -1 $JL/refs/ident_z.log | cut -c1-200))"
[ -e $IE/DONE ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.joint.identity_ckpt --base $V8E --out $IE > $JL/refs/ident_e.log 2>&1 \
  || jstop "identity checkpoint of $V8E ($(tail -1 $JL/refs/ident_e.log | cut -c1-200))"
( ref_job v8z "setsid nohup $TORCHZ" jref_dev5_v8sep_z $RZ_ARGS ) < /dev/null >> $JL/refs/refs.log 2>&1 &
sleep 90
( ref_job v8e "setsid nohup $EORCH" jref_dev5_v8sep_e $RE_ARGS ) < /dev/null >> $JL/refs/refs.log 2>&1 &
sleep 90
( ref_job aprime "setsid nohup $TORCHZ" jref_dev5_aprime_s3000 $RA_ARGS ) < /dev/null >> $JL/refs/refs.log 2>&1 &
log "reference dev5 decodes started (refs/v8z, v8e, aprime; they wait for free non-training cards)"
# ---------------------------------------------------------------------------------------------- takeover + training
t0=$(date +%s)
until [ -z "$(cards_ready $CARDS $PEAK_GIB)" ]; do
  [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && [ $(( $(date +%s) - t0 )) -gt 60 ] && log "waiting for $CARDS: $(cards_ready $CARDS $PEAK_GIB | head -3 | tr '\n' ' ')"
  sleep 30
done
log "cards $CARDS ready: $(others_on $CARDS | awk '{print $4}' | sort | uniq -c | tr '\n' ' ')"
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 30); do sleep 10; n=$(workers | wc -l); [ "$n" -ge 4 ] && break; done
log "stage-1 ranks running: $n (torchrun $(torchrun_pid))"
[ "$n" -ge 4 ] || { release_slot; jstop "training ranks missing ($n; $(tail -2 $TL | tr '\n' ' ' | cut -c1-300))"; }
log "JOINT_TRAINING_STARTED s1 $RUN_ID"
# ---------------------------------------------------------------------------------------------- proxies + rule
declare -A LAUNCHED=()
OPTIM_OK=""
LAST=""
STOPPED=""
while :; do
  T=$(tracker)
  alive=$(torchrun_pid)
  if [ -z "$OPTIM_OK" ] && [ "$T" -ge 500 ]; then
    if $PY -m experiments.mist_w3po_v1.joint.check_megatron_optim $SAVE/iter_0000500 > $R/optim_check_s500.json 2> $R/optim_check.log; then
      OPTIM_OK=1; log "optimizer state at step 500: ok ($(cut -c1-300 $R/optim_check_s500.json))"
    else
      log "optimizer state at step 500 NOT ok: $(cut -c1-400 $R/optim_check_s500.json) $(tail -1 $R/optim_check.log | cut -c1-200)"
      [ -n "$alive" ] && stop_training
      release_slot
      jstop "optimizer state missing or empty in iter_0000500 (no fallback without the optimizer: ask the coordinator / user)"
    fi
  fi
  if [ -n "$OPTIM_OK" ]; then
    for ((s = 500; s <= T; s += 500)); do
      [ -n "${LAUNCHED[$s]:-}" ] && continue
      [ -d $SAVE/iter_$(printf %07d $s) ] || continue
      LAUNCHED[$s]=1
      setsid nohup env STAGE=s1 RUN=$RUN_ID STEP=$s BASE=$O2 SRC=$SAVE/iter_$(printf %07d $s) bash $SC/joint_proxy.sh \
        < /dev/null >> $R/proxy/s$(printf %05d $s).log 2>&1 &
      log "step $s: proxy pid $!"
    done
  fi
  ZR=$(cat $JL/refs/v8z.ok 2>/dev/null); ER=$(cat $JL/refs/v8e.ok 2>/dev/null)
  for tg in v8z v8e; do [ -e $JL/refs/$tg.failed ] && { [ -n "$alive" ] && stop_training; release_slot; jstop "reference dev5 decode $tg failed twice"; }; done
  CANDS=""
  for s in $(printf '%s\n' "${!LAUNCHED[@]}" | sort -n); do
    S5=$(printf %05d $s)
    if [ -e $R/proxy/s$S5.ok ]; then CANDS="$CANDS --cand $s $(awk '{print $2, $3}' $R/proxy/s$S5.ok) ok"
    elif [ -e $R/proxy/s$S5.failed ]; then CANDS="$CANDS --cand $s s1z_dev5_${RUN_ID}_s$S5 s1e_dev5_${RUN_ID}_s$S5 failed"
    else CANDS="$CANDS --cand $s s1z_dev5_${RUN_ID}_s$S5 s1e_dev5_${RUN_ID}_s$S5 pending"; fi
  done
  if [ -n "$ZR" ] && [ -n "$ER" ] && [ -n "$CANDS" ]; then
    D=$($PY $SC/joint_select.py --stage s1 --ref-zh2en $ZR --ref-en2zh $ER $CANDS --out $SEL 2> $R/select.err)
    A=$(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['action'], d.get('prefix_end'), d.get('decision_step'), d.get('chosen'))" 2>/dev/null)
    if [ "$A" != "$LAST" ]; then echo "$D" >> $R/decisions.jsonl; log "rule: $A"; LAST=$A; fi
    case ${A%% *} in
      stop) break;;
      no_feasible) [ -n "$alive" ] && stop_training; release_slot; jstop "no feasible checkpoint at the stop point ($A)";;
      ceiling) log "JOINT_CEILING s1 15000"; release_slot; jstop "ceiling 15000 reached without an early stop: ask the coordinator";;
    esac
  fi
  if [ -z "$alive" ] && [ -z "$STOPPED" ]; then
    sleep 20
    if [ -z "$(torchrun_pid)" ]; then
      if [ "$(tracker)" -ge 15000 ]; then STOPPED=1; log "training reached the ceiling 15000; waiting for the last proxies"
      else release_slot; jstop "training exited on its own at tracker $(tracker) ($(grep -E 'THINKER_GSTAR' $TL | tail -1))"; fi
    fi
  fi
  sleep 60
done
DS=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['decision_step'])")
CH=$(echo "$D" | $PY -c "import json,sys; print(json.load(sys.stdin)['chosen'])")
( set -o noclobber; echo "decision step $DS chosen $CH utc $(date -u +%FT%TZ)" > $RO/STOP )
log "JOINT_EARLY_STOP s1 $DS"
[ -n "$(torchrun_pid)" ] && stop_training
# ---------------------------------------------------------------------------------------------- merge the selected step
C5=$(printf %05d $CH); M=$RO/merged_s$C5/thinker_merged; I=$RO/merged_s$C5/identity
log "merging iteration $CH onto O2 -> $M"
CUDA_VISIBLE_DEVICES= $PY experiments/uniss_qwen3_omni_v1/training/merge_thinker_gstar.py --checkpoint-dir $SAVE/iter_$(printf %07d $CH) \
  --rank 32 --alpha 64 --base $O2 --out $M > $R/merge.log 2>&1 || jstop "merge of iteration $CH failed ($(tail -1 $R/merge.log | cut -c1-200))"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.joint.identity_ckpt --base $M --out $I > $R/identity.log 2>&1 \
  || jstop "identity checkpoint of $M failed"
( set -o noclobber; echo "step $CH run $RUN_ID merged $M identity $I zname $(echo "$D" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['chosen_names']['zh2en'], 'ename', d['chosen_names']['en2zh'])") utc $(date -u +%FT%TZ)" > $MARK ) \
  || jstop "marker $MARK exists"
log "JOINT_SELECTED s1 $CH ($M)"
