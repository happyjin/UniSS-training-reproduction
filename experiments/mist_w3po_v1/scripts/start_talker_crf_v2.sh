#!/usr/bin/env bash
# talker_crf_prereg_v2 training on cards 0-3 ONLY (user 2026-11-11 ~05:50 UTC): experiment 1 (BoN-SFT, rows
# talker_rows --mode bon) first, then experiment 2 (single-channel PL-EM, rows --mode plem --scope meeting, built in the
# background meanwhile). Each training = run_talker_crf.sh MODE=plem verbatim (T1c start, LR 3e-5, 1200 steps, GBS 4,
# KL 0.5 to T1c) with only the rows file changed. Before each training: wait until no job of ours holds 0-3, stop the
# no-checkpoint workload there by PID (worker + its supervisor, cmdline checked; never a pattern kill) and wait for its
# memory to go. On TALKER_CRF_DONE the export path is written to $L/export_<s> (the eval orchestrator
# talker_crf_v2_orch.py waits for it); on failure $L/train_failed_<s>. The eval cards 4-7 are never touched here.
# After the last training the cards idle -> keep_cards_busy.sh (guard) re-arms the workload on them.
# Checked with bash -n only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_talker_crf_v2.sh > logs/mist_w3po_v1/talker_crf_v2/train.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CARDS=0,1,2,3; PORT=29821
L=logs/mist_w3po_v1/talker_crf_v2
RD=eval_outputs/mist_w3po_v1/talker_crf/rows
CANDS='eval_outputs/mist_w3po_v1/speech/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl'
WIN=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""
mkdir -p $L $RD
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/train.log"; }

ours_on_cards() {   # GPU processes on CARDS that are not the no-checkpoint workload
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
  local u p pp c used i
  for c in $(echo "$CARDS" | tr ',' ' '); do
    u=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$u" '$2==u{print $1}'); do
      if grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null; then
        pp=$(ps -o ppid= -p "$p" | tr -d ' ')
        log "card $c: stop workload worker $p (supervisor $pp)"
        kill -TERM "$p" 2>/dev/null
        [ -n "$pp" ] && [ "$pp" != 1 ] && grep -q train_mist_workload /proc/$pp/cmdline 2>/dev/null && kill -TERM "$pp" 2>/dev/null
      fi
    done
  done
  for i in $(seq 1 36); do   # <= 3 min for the memory to go
    used=0
    for c in $(echo "$CARDS" | tr ',' ' '); do
      m=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "$m" -gt 4000 ] && used=1
    done
    [ "$used" = 0 ] && return 0
    sleep 5
  done
  log "WARNING: memory still in use on cards $CARDS after the stop: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}

train() {   # $1 = scheme (bon | plemm), $2 = rows file
  local s=$1 rows=$2 i n RUN_ID TLOG TP
  for i in $(seq 1 120); do
    n=$(ours_on_cards); [ "$n" = 0 ] && break
    [ $((i % 10)) = 1 ] && log "waiting for $n non-workload GPU processes to leave cards $CARDS"
    sleep 30
  done
  stop_workload_on_cards
  RUN_ID=talker_${s}_$(date -u +%Y%m%dT%H%M%SZ)
  echo "$RUN_ID" > $L/run_id_$s
  TLOG=$L/train_$s.$RUN_ID.log
  flock -o "$LOCK" -c "RUN_ID=$RUN_ID MODE=plem ROWS=$rows CARDS=$CARDS MASTER_PORT=$PORT setsid nohup bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > $TLOG 2>&1 < /dev/null & echo \$! > $L/train_pid_$s" 9>&-
  TP=$(cat $L/train_pid_$s)
  log "training $s: $RUN_ID pid $TP rows $rows log $TLOG"
  while [ -d /proc/$TP ]; do sleep 60; done
  if grep -q TALKER_CRF_DONE "$TLOG"; then
    echo "eval_outputs/mist_w3po_v1/talker_crf/$RUN_ID/export" > $L/export_$s.tmp && command mv -f $L/export_$s.tmp $L/export_$s
    log "trained $s; export $(cat $L/export_$s)"
  else
    log "TRAIN_FAILED $s $RUN_ID (tail: $(tail -3 $TLOG | tr '\n' ' ' | cut -c1-300))"
    touch $L/train_failed_$s
  fi
}

log "start (cards $CARDS)"
# experiment 2's rows in the background (CPU; the meeting lattice)
if [ ! -e $RD/plemm.pt ]; then
  ( $PY -m experiments.mist_w3po_v1.talker.talker_rows --mode plem --scope meeting --cands "$CANDS" --win "$WIN" \
      --out $RD/plemm.pt > $L/rows_plemm.log 2>&1 && echo ok > $L/rows_plemm.done || echo fail > $L/rows_plemm.done ) &
  RP=$!
else
  echo ok > $L/rows_plemm.done; RP=""
fi
# experiment 1
if [ -e $RD/bon.pt ] || $PY -m experiments.mist_w3po_v1.talker.talker_rows --mode bon --cands "$CANDS" --win "$WIN" \
     --out $RD/bon.pt > $L/rows_bon.log 2>&1; then
  log "rows bon: $(cat $RD/bon.pt.json | tr -d '\n' | cut -c1-400)"
  train bon $RD/bon.pt
else
  log "ROWS_FAILED bon (see $L/rows_bon.log)"; touch $L/train_failed_bon
fi
# experiment 2
[ -n "$RP" ] && wait $RP
if [ "$(cat $L/rows_plemm.done 2>/dev/null)" = ok ]; then
  log "rows plemm: $(cat $RD/plemm.pt.json | tr -d '\n' | cut -c1-600)"
  train plemm $RD/plemm.pt
else
  log "ROWS_FAILED plemm (see $L/rows_plemm.log)"; touch $L/train_failed_plemm
fi
log "trainings over; cards $CARDS go back to the no-checkpoint workload (guard keep_cards_busy.sh)"
