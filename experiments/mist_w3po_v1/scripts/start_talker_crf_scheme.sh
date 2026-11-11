#!/usr/bin/env bash
# One talker scheme end to end after the sampling (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md; user go
# 2026-11-10 22:35 UTC: train and evaluate both schemes tonight whatever the prechecks say):
#  1. wait until the sample stage's 8 train talker shards ended with rc 0;
#  2. training rows (talker_rows --mode SCHEME) from the train dumps (CPU);
#  3. take CARDS: mark them active (logs/.../train_active_SCHEME), write exclude_cards into every talker_crf
#     orchestrator run dir (our other jobs stay off), wait (<= 60 min) until no job of ours runs there, stop the
#     no-checkpoint workload there by PID (train_mist_workload.py, cmdline checked), launch run_talker_crf.sh detached
#     under gpu_launch.lock (flock -o, 9>&-) and wait for its PID;
#  4. on TALKER_CRF_DONE: release the cards (exclude_cards rewritten), launch the eval stage of talker_crf_orch.py for
#     this scheme (other schemes' training cards excluded), wait for it.
# Checked with bash -n only; never test-run.
#   SCHEME=plem CARDS=0,1,2,3 PORT=29811 setsid nohup bash experiments/mist_w3po_v1/scripts/start_talker_crf_scheme.sh \
#     > logs/mist_w3po_v1/talker_crf_v1/scheme_plem.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
SCHEME=${SCHEME:?plem|align}; CARDS=${CARDS:?e.g. 0,1,2,3}; PORT=${PORT:-29811}
L=logs/mist_w3po_v1/talker_crf_v1
SAMPLE=$L/sample
ROWS=eval_outputs/mist_w3po_v1/talker_crf/rows/$SCHEME.pt
CANDS='eval_outputs/mist_w3po_v1/speech/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl'
WIN=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface OMP_NUM_THREADS=16
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/scheme_$SCHEME.log"; }

set_exclude() {   # every orchestrator run dir gets the cards of the trainings now active
  local ex=""
  for f in $L/train_active_*; do [ -e "$f" ] && ex="$ex $(cat "$f" | tr ',' ' ')"; done
  echo "$ex" > "$L/exclude_cards.tmp" && command mv -f "$L/exclude_cards.tmp" "$L/exclude_cards"   # read by new orchestrators
  for d in $L/sample $L/precheck $L/eval_plem $L/eval_align; do
    [ -d "$d" ] && echo "$ex" > "$d/exclude_cards.tmp" && command mv -f "$d/exclude_cards.tmp" "$d/exclude_cards"
  done
}

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
  local u p
  for c in $(echo "$CARDS" | tr ',' ' '); do
    u=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$u" '$2==u{print $1}'); do
      if grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null; then
        # the worker and its parent supervisor (which would otherwise re-arm on an idle card)
        pp=$(ps -o ppid= -p "$p" | tr -d ' ')
        log "card $c: stop workload worker $p (supervisor $pp)"
        kill -TERM "$p" 2>/dev/null
        [ -n "$pp" ] && grep -q train_mist_workload /proc/$pp/cmdline 2>/dev/null && kill -TERM "$pp" 2>/dev/null
      fi
    done
  done
  sleep 20
}

log "scheme $SCHEME cards $CARDS: waiting for the train sampling"
# 1. the 8 train talker shards
until [ "$(cat $SAMPLE/state/train__talk*.rc 2>/dev/null | grep -cx 0)" -ge 8 ]; do
  if ls $SAMPLE/events/alert_* >/dev/null 2>&1 && grep -lq "FAILED train__talk" $SAMPLE/events/alert_* 2>/dev/null; then
    log "a train talker shard FAILED for good; stopping"; exit 1
  fi
  sleep 60
done
log "train sampling done"

# 2. rows
mkdir -p eval_outputs/mist_w3po_v1/talker_crf/rows
if [ ! -e "$ROWS" ]; then
  $PY -m experiments.mist_w3po_v1.talker.talker_rows --mode "$SCHEME" --cands "$CANDS" --win "$WIN" --out "$ROWS" \
    > $L/rows_$SCHEME.log 2>&1 || { log "ROWS_FAILED (see $L/rows_$SCHEME.log)"; exit 1; }
fi
log "rows: $(cat $ROWS.json | tr -d '\n' | cut -c1-400)"

# 3. the cards, then the training
echo "$CARDS" > $L/train_active_$SCHEME
set_exclude
for i in $(seq 1 120); do
  n=$(ours_on_cards); [ "$n" = 0 ] && break
  [ $((i % 10)) = 1 ] && log "waiting for $n job processes of ours to leave cards $CARDS"
  sleep 30
done
stop_workload_on_cards
RUN_ID=talker_${SCHEME}_$(date -u +%Y%m%dT%H%M%SZ)
echo "$RUN_ID" > $L/run_id_$SCHEME
TLOG=$L/train_$SCHEME.$RUN_ID.log
flock -o "$LOCK" -c "RUN_ID=$RUN_ID MODE=$SCHEME ROWS=$ROWS CARDS=$CARDS MASTER_PORT=$PORT setsid nohup bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > $TLOG 2>&1 < /dev/null & echo \$! > $L/train_pid_$SCHEME" 9>&-
TP=$(cat $L/train_pid_$SCHEME)
log "training $RUN_ID pid $TP log $TLOG"
while [ -d /proc/$TP ]; do sleep 60; done
release_cards() { command mv -f $L/train_active_$SCHEME $L/ended_train_active_$SCHEME.$(date -u +%H%M%S); set_exclude; }   # out of the train_active_* glob (2026-11-11 fix)
if ! grep -q TALKER_CRF_DONE "$TLOG"; then
  log "TRAIN_FAILED $RUN_ID (tail: $(tail -3 $TLOG | tr '\n' ' ' | cut -c1-300))"
  release_cards; exit 1
fi
release_cards
EXPORT=eval_outputs/mist_w3po_v1/talker_crf/$RUN_ID/export
log "trained; export $EXPORT"

# 4. the eval stage
KAPPA=$($PY -c "import json; print(json.load(open('$ROWS.json')).get('kappa') or 0)")
RQ=""; [ "$SCHEME" = plem ] && RQ="--ref-quality"
setsid $PY experiments/mist_w3po_v1/scripts/talker_crf_orch.py --stage eval --scheme "$SCHEME" --export "$EXPORT" \
  --kappa "$KAPPA" $RQ --run $L/eval_$SCHEME > $L/eval_$SCHEME.out 2>&1 < /dev/null 9>&- &
EP=$!
sleep 5; set_exclude
log "eval orchestrator pid $EP (kappa $KAPPA)"
while [ -d /proc/$EP ]; do sleep 60; done
log "eval done: $(cat $L/eval_$SCHEME/events/DONE 2>/dev/null | tr -d '\n')"
