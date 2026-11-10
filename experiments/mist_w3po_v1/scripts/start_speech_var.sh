#!/usr/bin/env bash
# Speech-gap plan batch 1 (b1 b3 b2 c1 on C0-fp's dev text, 2 TTS seeds each; user go 2026-11-10 ~11:50 UTC: stop the
# no-checkpoint workload, stack on all 8 cards, as fast as possible).
#  1. stop the v2' eval scheduler (SCHED_PID, its only remaining step = restart the workload on all cards) by PID with a
#     cmdline guard; stop every alive workload supervisor / idle-fill loop (card<k>.pid of logs/mist_w3po_v1/workload_at_*)
#     and their descendants by PID (cmdline guard against PID reuse); the v2' eval orchestrator itself is not touched;
#  2. start NS CTC scorer servers (si_ctc_scorer.py, ports 18765..) on cards SCORER_CARDS, under gpu_launch.lock (flock -o),
#     wait for SCORER_READY;
#  3. launch speech_var_orch.py --spread (new run dir, refuses existing dirs); it holds gpu_launch.lock per launch;
#  4. arm idle_fill_until.sh on all 8 cards, stop file = <run>/events/DONE (no idle card between stages);
#  5. wait for the orchestrator to exit; stop the scorers by PID; restart the full workload (start_workload_at.sh).
# Checked with bash -n only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_speech_var.sh \
#     > logs/mist_w3po_v1/start_speech_var_<UTC>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
SCHED_PID=${SCHED_PID:-1873854}
RUN=${RUN:-logs/mist_w3po_v1/speech_var_b1}
VARS=${VARS:-b1,b3,b2,c1}
SCORER_CARDS=${SCORER_CARDS:-"0 2 4 6"}
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
ENVS="export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false"
D=logs/mist_w3po_v1/speech_var_at_$(date -u +%Y%m%dT%H%M%SZ)
if [ -e "$D" ] || [ -e "$RUN" ]; then echo "refusing: $D or $RUN exists"; exit 1; fi
mkdir -p "$D"; echo $$ > "$D/launcher.pid"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/launcher.log"; }
desc() { ps -eo pid=,ppid= | awk -v r="$1" '{par[$1]=$2} END{keep[r]=1; ch=1; while(ch){ch=0; for(p in par) if(!(p in keep) && (par[p] in keep)){keep[p]=1; ch=1}} for(p in keep) print p}'; }

# 1. the v2' scheduler and the workload
if [ -d /proc/$SCHED_PID ] && tr '\0' ' ' < /proc/$SCHED_PID/cmdline | grep -q start_v2p_eval_after_train; then
  # the scheduler's own PID only: its tree holds the v2p eval orchestrator (2026-11-10 11:50 UTC the tree kill stopped
  # it; its jobs, in their own sessions, survived and were adopted by crf_pair_dev_orch.py --resume)
  log "stop v2p scheduler $SCHED_PID only (its last step would restart the workload)"
  kill -TERM $SCHED_PID 2>/dev/null
fi
for f in logs/mist_w3po_v1/workload_at_*/card*.pid; do
  p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && [ -d /proc/$p ] || continue
  tr '\0' ' ' < /proc/$p/cmdline | grep -qE "run_mist_workload|train_mist_workload|idle_fill_until" || continue
  tree=$(desc "$p" | tr '\n' ' '); log "stop workload $f pid $p tree [$tree]"
  kill -TERM $tree 2>/dev/null
done
for i in $(seq 1 60); do
  n=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null && n=$((n+1)); done
  [ $n = 0 ] && break; sleep 5
done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  grep -q train_mist_workload /proc/$p/cmdline 2>/dev/null && { log "workload GPU pid $p still alive -> KILL"; kill -KILL $p; }
done
log "cards: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# 2. CTC scorers for b1
NS=0; SPIDS=""
if echo ",$VARS," | grep -q ",b1,"; then
  for c in $SCORER_CARDS; do
    port=$((18765 + NS))
    flock -o "$LOCK" -c "$ENVS; CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -m experiments.mist_w3po_v1.evaluation.si_ctc_scorer --port $port > $D/scorer$NS.log 2>&1 < /dev/null & echo \$! > $D/scorer$NS.pid" 9>&-
    SPIDS="$SPIDS $(cat $D/scorer$NS.pid)"; NS=$((NS + 1)); sleep 5
  done
  for i in $(seq 1 60); do
    r=0; for k in $(seq 0 $((NS - 1))); do grep -q SCORER_READY $D/scorer$k.log 2>/dev/null && r=$((r+1)); done
    [ $r = $NS ] && break; sleep 5
  done
  log "scorers ready $r/$NS pids [$SPIDS]"
  [ "$r" = "$NS" ] || { log "SCORER_FAILED; stop scorers, not launching"; kill -TERM $SPIDS 2>/dev/null; exit 1; }
fi

# 3. the orchestrator
setsid "$PY" experiments/mist_w3po_v1/scripts/speech_var_orch.py --run "$RUN" --vars "$VARS" --scorers "$NS" --spread \
  > "$D/orch.out" 2>&1 < /dev/null 9>&- &
OP=$!; echo $OP > "$D/orch.pid"; log "orchestrator pid $OP run $RUN vars $VARS"

# 4. idle fill until DONE
sleep 300
W=logs/mist_w3po_v1/workload_at_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$W"
echo "idle_fill_until.sh during speech-gap batch 1, cap 80000, stop file $RUN/events/DONE" > "$W/README"
for k in 0 1 2 3 4 5 6 7; do
  DIR=$W STOP_FILE=$RUN/events/DONE CAP_MIB=80000 setsid nohup bash experiments/mist_w3po_v1/scripts/idle_fill_until.sh $k \
    > "$W/loop$k.out" 2>&1 < /dev/null 9>&- &
  sleep 1
done
log "idle fill armed in $W"

# 5. end
while [ -d /proc/$OP ]; do sleep 60; done
log "orchestrator exited; DONE=$([ -e "$RUN/events/DONE" ] && cat "$RUN/events/DONE" || echo no)"
[ -n "$SPIDS" ] && { log "stop scorers [$SPIDS]"; kill -TERM $SPIDS 2>/dev/null; }
sleep 90   # idle loops see DONE and exit
AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) setsid nohup bash experiments/mist_w3po_v1/scripts/start_workload_at.sh \
  > logs/mist_w3po_v1/start_workload_at_$(date -u +%Y%m%dT%H%M%SZ).out 2>&1 < /dev/null 9>&- &
log "workload restarted on all 8 cards; launcher done"
