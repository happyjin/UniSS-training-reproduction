#!/usr/bin/env bash
# talker_crf_prereg_v3 (E2', single-channel PL-EM with the corrected budget): training on the A slot (cards 0-3), from
# start_talker_crf_v2_plemm.sh's train_locked (lock FIRST, then stop the workload, launch, release once the memory shows):
#  0. the A slot must be ours: logs/mist_w3po_v1/slots/slotA.owner names the talker-side executor (created with noclobber
#     07:29 UTC); otherwise stop here;
#  1. wait for the rows (logs/mist_w3po_v1/talker_crf_v3/rows_plemm2.rc = 0; rows/plemm2.pt);
#  2. under gpu_launch.lock (fd 9 held by THIS shell only; the training is launched with 9>&-): wait until no non-workload
#     process is on 0-3, stop the no-checkpoint workload there by PID (worker + supervisor, cmdline checked), wait for
#     the memory to go, launch run_talker_crf.sh MODE=plem ROWS=plemm2.pt (verbatim otherwise), wait until the training's
#     memory shows on all four cards (<= 240 s), release the lock;
#  3. the moment no non-workload process holds memory on 0-3 any more (the training's torchrun is over; the CPU export
#     may still run), release the A slot: slotA.owner -> slotA.released_<UTC> (command mv -f; never rm);
#  4. on TALKER_CRF_DONE: export_plemm2, else train_failed_plemm2.
# Checked with bash -n only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_talker_crf_v3_plemm2.sh >> logs/mist_w3po_v1/talker_crf_v3/train.out 2>&1 < /dev/null 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CARDS=0,1,2,3; PORT=29831; S=plemm2
L=logs/mist_w3po_v1/talker_crf_v3
SLOTS=logs/mist_w3po_v1/slots
ROWS=eval_outputs/mist_w3po_v1/talker_crf/rows/plemm2.pt
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""
log() { echo "$(date -u +%FT%TZ) [e2prime] $*" | tee -a "$L/train.log"; }

ours_on_cards() {   # non-workload compute processes on CARDS
  local n=0 u p c
  for c in $(echo "$CARDS" | tr ',' ' '); do
    u=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$u" '$2==u{print $1}'); do
      grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null || n=$((n + 1))
    done
  done
  echo $n
}

stop_workload_on_cards() {
  local u p pp c used i m
  for c in $(echo "$CARDS" | tr ',' ' '); do
    u=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$u" '$2==u{print $1}'); do
      if grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null; then
        pp=$(ps -o ppid= -p "$p" | tr -d ' ')
        log "card $c: stop workload worker $p (supervisor $pp)"
        [ -n "$pp" ] && [ "$pp" != 1 ] && grep -q train_mist_workload /proc/$pp/cmdline 2>/dev/null && kill -TERM "$pp" 2>/dev/null
        kill -TERM "$p" 2>/dev/null
      fi
    done
  done
  for i in $(seq 1 36); do
    used=0
    for c in $(echo "$CARDS" | tr ',' ' '); do
      m=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "$m" -gt 4000 ] && used=1
    done
    [ "$used" = 0 ] && return 0
    sleep 5
  done
  log "WARNING: memory still in use on cards $CARDS: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}

release_slot() {
  local t
  t=$(date -u +%Y%m%dT%H%M%SZ)
  if [ -e $SLOTS/slotA.owner ] && grep -q "talker-side executor" $SLOTS/slotA.owner; then
    echo "released: $(date -u +%FT%TZ) ($1)" >> $SLOTS/slotA.owner
    command mv -f $SLOTS/slotA.owner $SLOTS/slotA.released_$t
    log "A slot released: $SLOTS/slotA.released_$t ($1)"
  else
    log "WARNING: slotA.owner missing or not ours at release time ($1)"
  fi
}

if ! { [ -e $SLOTS/slotA.owner ] && grep -q "talker-side executor" $SLOTS/slotA.owner; }; then
  log "STOP: the A slot is not ours ($SLOTS/slotA.owner)"; touch $L/train_failed_$S; exit 1
fi
log "start; waiting for the rows ($L/rows_$S.rc)"
until [ -e $L/rows_$S.rc ]; do sleep 20; done
if [ "$(cat $L/rows_$S.rc)" != 0 ] || [ ! -s $ROWS ]; then
  log "ROWS_FAILED $S (see $L/rows_$S.log)"; touch $L/train_failed_$S; release_slot "rows failed"; exit 1
fi
log "rows $S: $(cat $ROWS.json | tr -d '\n ' | cut -c1-700)"
for i in $(seq 1 120); do
  n=$(ours_on_cards); [ "$n" = 0 ] && break
  [ $((i % 10)) = 1 ] && log "waiting for $n non-workload GPU processes to leave cards $CARDS"
  sleep 30
done
exec 9>>"$LOCK"
flock 9
log "gpu_launch.lock held"
stop_workload_on_cards
RUN_ID=talker_${S}_$(date -u +%Y%m%dT%H%M%SZ)
echo "$RUN_ID" > $L/run_id_$S
TLOG=$L/train_$S.$RUN_ID.log
RUN_ID=$RUN_ID MODE=plem ROWS=$ROWS CARDS=$CARDS MASTER_PORT=$PORT setsid nohup bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > $TLOG 2>&1 < /dev/null 9>&- &
TP=$!
echo $TP > $L/train_pid_$S
log "training $S: $RUN_ID pid $TP rows $ROWS log $TLOG"
ok=0
for i in $(seq 1 48); do   # the training's memory on all four cards (> 10 GiB each), <= 240 s
  ok=1
  for c in $(echo "$CARDS" | tr ',' ' '); do
    m=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$m" -lt 10000 ] && ok=0
  done
  [ "$ok" = 1 ] && break
  [ -d /proc/$TP ] || break
  sleep 5
done
flock -u 9; exec 9>&-
log "gpu_launch.lock released (training memory visible: $ok)"
released=0; seen=$ok   # release only after the training's memory has been seen on the cards
while [ -d /proc/$TP ]; do
  n=$(ours_on_cards)
  if [ "$seen" = 0 ] && [ "$n" -gt 0 ]; then seen=1; fi
  if [ "$seen" = 1 ] && [ "$released" = 0 ] && [ "$n" = 0 ]; then
    sleep 10
    if [ "$(ours_on_cards)" = 0 ]; then release_slot "training memory gone; launcher pid $TP still exporting"; released=1; fi
  fi
  sleep 15
done
[ "$released" = 0 ] && release_slot "training process over"
if grep -q TALKER_CRF_DONE "$TLOG"; then
  echo "eval_outputs/mist_w3po_v1/talker_crf/$RUN_ID/export" > $L/export_$S.tmp && command mv -f $L/export_$S.tmp $L/export_$S
  log "trained $S; export $(cat $L/export_$S)"
else
  log "TRAIN_FAILED $S $RUN_ID (tail: $(tail -3 $TLOG | tr '\n' ' ' | cut -c1-300))"
  touch $L/train_failed_$S
fi
log "done; cards $CARDS: the guard keep_cards_busy.sh re-arms the workload on idle cards"
