#!/usr/bin/env bash
# Start the SerialCRF v2' dev evaluation (crf_pair_dev_orch.py --cands v2p --spread) right after v2' training ends.
# Events (polls every 30 s; gives up after 8 h):
#  1. training root PID gone and ckpt/step_03000/DONE exists (no DONE -> log TRAIN_FAILED, leave the workload, exit);
#  2. the v2' launcher/watcher (WATCH_PID) has exited (it restarts the user's workload) or 15 min passed;
#  3. stop every workload supervisor (card<k>.pid of every logs/mist_w3po_v1/workload_at_*/ that is alive) by PID, then
#     their descendants by PID from ps fields; wait until no GPU process runs train_mist_workload.py (others are left);
#  4. launch the orchestrator (new run dir, refuses if it exists); it holds gpu_launch.lock per launch itself;
#  5. once 8 decode shards are visible (or 20 min), arm idle_fill_until.sh on all 8 cards, stop file = <run>/events/DONE;
#  6. wait for the orchestrator to exit; then restart the full workload on all 8 cards (start_workload_at.sh, AT = now).
# Checked with bash -n only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_v2p_eval_after_train.sh \
#     > logs/mist_w3po_v1/start_v2p_eval_after_train_<UTC>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
TRAIN_PID=${TRAIN_PID:-1013964}
WATCH_PID=${WATCH_PID:-3880632}
CKPT=${CKPT:-eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000}
RUN=${RUN:-logs/mist_w3po_v1/v2p_dev}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
D=logs/mist_w3po_v1/v2p_eval_at_$(date -u +%Y%m%dT%H%M%SZ)
if [ -e "$D" ] || [ -e "$RUN" ]; then echo "refusing: $D or $RUN exists"; exit 1; fi
mkdir -p "$D"; echo $$ > "$D/scheduler.pid"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/scheduler.log"; }
T0=$(date +%s); late() { [ $(( $(date +%s) - T0 )) -gt 28800 ]; }
log "scheduler $$: wait for training pid $TRAIN_PID + $CKPT/DONE, then v2p eval in $RUN"
while [ -d /proc/$TRAIN_PID ]; do late && { log "GAVE_UP waiting for training"; exit 1; }; sleep 30; done
sleep 30
if [ ! -e "$CKPT/DONE" ]; then log "TRAIN_FAILED: $CKPT/DONE missing; workload left to the v2p watcher"; exit 1; fi
log "training done; waiting for watcher $WATCH_PID (workload restart)"
t=$(date +%s); while [ -d /proc/$WATCH_PID ] && [ $(( $(date +%s) - t )) -lt 900 ]; do sleep 30; done
sleep 60   # let start_workload_at.sh write all card pids (10 s apart)
desc() { ps -eo pid=,ppid= | awk -v r="$1" '{par[$1]=$2} END{keep[r]=1; ch=1; while(ch){ch=0; for(p in par) if(!(p in keep) && (par[p] in keep)){keep[p]=1; ch=1}} for(p in keep) print p}'; }
for f in logs/mist_w3po_v1/workload_at_*/card*.pid; do
  p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && [ -d /proc/$p ] || continue
  # PID reuse guard: only a workload supervisor / idle-fill loop (stale pid files may now name other processes)
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
mkdir -p "$(dirname "$RUN")"
setsid "$PY" experiments/mist_w3po_v1/scripts/crf_pair_dev_orch.py --run "$RUN" --cands v2p --spread \
  > "$D/orch.out" 2>&1 < /dev/null 9>&- &
OP=$!; echo $OP > "$D/orch.pid"; log "orchestrator pid $OP run $RUN"
t=$(date +%s)
while [ -d /proc/$OP ] && [ $(( $(date +%s) - t )) -lt 1200 ]; do
  [ "$(grep -c 'visible .*__dec' "$RUN/orch.log" 2>/dev/null || echo 0)" -ge 8 ] && break; sleep 30
done
W=logs/mist_w3po_v1/workload_at_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$W"
echo "idle_fill_until.sh during v2p dev eval, cap 80000, stop file $RUN/events/DONE" > "$W/README"
for k in 0 1 2 3 4 5 6 7; do
  DIR=$W STOP_FILE=$RUN/events/DONE CAP_MIB=80000 setsid nohup bash experiments/mist_w3po_v1/scripts/idle_fill_until.sh $k \
    > "$W/loop$k.out" 2>&1 < /dev/null 9>&- &
  sleep 1
done
log "idle fill armed in $W"
while [ -d /proc/$OP ]; do sleep 60; done
log "orchestrator exited; DONE=$([ -e "$RUN/events/DONE" ] && echo yes || echo no)"
sleep 90   # idle loops see DONE and exit
AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) setsid nohup bash experiments/mist_w3po_v1/scripts/start_workload_at.sh \
  > logs/mist_w3po_v1/start_workload_at_$(date -u +%Y%m%dT%H%M%SZ).out 2>&1 < /dev/null 9>&- &
log "workload restarted on all 8 cards; scheduler done"
