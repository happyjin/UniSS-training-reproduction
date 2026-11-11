#!/usr/bin/env bash
# Takeover of start_talker_crf_v2.sh after its experiment-1 launch (2026-11-11 06:35 UTC incident: the launcher stopped
# the workload on cards 0-3 and THEN waited ~5 min in flock for gpu_launch.lock -- the eval orchestrator held it in the
# visible-wait of its 1.7 GiB CTC scorers -- so the guard re-armed the workload on the idle cards; it was stopped by PID
# just before the training started). Same steps as the original for what is left, with the lock taken FIRST:
#  1. wait until the experiment-1 training (train_pid_bon) exits; export_bon on TALKER_CRF_DONE, else train_failed_bon;
#  2. wait for experiment 2's rows (rows_plemm.done, built by the original launcher's background job);
#  3. under gpu_launch.lock (fd 9 held by THIS shell only; the training is launched with 9>&-): wait until no job of
#     ours is on 0-3, stop the no-checkpoint workload there by PID (worker + supervisor, cmdline checked), wait for the
#     memory to go, launch run_talker_crf.sh MODE=plem ROWS=plemm.pt (verbatim otherwise), wait until the training's
#     memory shows on all four cards (<= 240 s), release the lock;
#  4. on TALKER_CRF_DONE: export_plemm, else train_failed_plemm.
# Checked with bash -n only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_talker_crf_v2_plemm.sh >> logs/mist_w3po_v1/talker_crf_v2/train.out 2>&1 < /dev/null 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CARDS=0,1,2,3; PORT=29821
L=logs/mist_w3po_v1/talker_crf_v2
RD=eval_outputs/mist_w3po_v1/talker_crf/rows
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""
log() { echo "$(date -u +%FT%TZ) [takeover] $*" | tee -a "$L/train.log"; }

ours_on_cards() {
  local n=0 u p
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

train_locked() {   # $1 scheme, $2 rows
  local s=$1 rows=$2 i n RUN_ID TLOG TP c u ok
  for i in $(seq 1 120); do
    n=$(ours_on_cards); [ "$n" = 0 ] && break
    [ $((i % 10)) = 1 ] && log "waiting for $n non-workload GPU processes to leave cards $CARDS"
    sleep 30
  done
  exec 9>>"$LOCK"
  flock 9
  log "gpu_launch.lock held"
  stop_workload_on_cards
  RUN_ID=talker_${s}_$(date -u +%Y%m%dT%H%M%SZ)
  echo "$RUN_ID" > $L/run_id_$s
  TLOG=$L/train_$s.$RUN_ID.log
  RUN_ID=$RUN_ID MODE=plem ROWS=$rows CARDS=$CARDS MASTER_PORT=$PORT setsid nohup bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > $TLOG 2>&1 < /dev/null 9>&- &
  TP=$!
  echo $TP > $L/train_pid_$s
  log "training $s: $RUN_ID pid $TP rows $rows log $TLOG"
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
  while [ -d /proc/$TP ]; do sleep 60; done
  if grep -q TALKER_CRF_DONE "$TLOG"; then
    echo "eval_outputs/mist_w3po_v1/talker_crf/$RUN_ID/export" > $L/export_$s.tmp && command mv -f $L/export_$s.tmp $L/export_$s
    log "trained $s; export $(cat $L/export_$s)"
  else
    log "TRAIN_FAILED $s $RUN_ID (tail: $(tail -3 $TLOG | tr '\n' ' ' | cut -c1-300))"
    touch $L/train_failed_$s
  fi
}

TPB=$(cat $L/train_pid_bon); TLOGB=$L/train_bon.$(cat $L/run_id_bon).log
log "waiting for the experiment-1 training pid $TPB ($TLOGB)"
while [ -d /proc/$TPB ]; do sleep 60; done
if grep -q TALKER_CRF_DONE "$TLOGB"; then
  echo "eval_outputs/mist_w3po_v1/talker_crf/$(cat $L/run_id_bon)/export" > $L/export_bon.tmp && command mv -f $L/export_bon.tmp $L/export_bon
  log "trained bon; export $(cat $L/export_bon)"
else
  log "TRAIN_FAILED bon (tail: $(tail -3 $TLOGB | tr '\n' ' ' | cut -c1-300))"
  touch $L/train_failed_bon
fi
until [ -e $L/rows_plemm.done ]; do sleep 30; done
if [ "$(cat $L/rows_plemm.done)" = ok ]; then
  log "rows plemm: $(cat $RD/plemm.pt.json | tr -d '\n' | cut -c1-600)"
  train_locked plemm $RD/plemm.pt
else
  log "ROWS_FAILED plemm (see $L/rows_plemm.log)"; touch $L/train_failed_plemm
fi
log "trainings over; cards $CARDS go back to the no-checkpoint workload (guard keep_cards_busy.sh)"
