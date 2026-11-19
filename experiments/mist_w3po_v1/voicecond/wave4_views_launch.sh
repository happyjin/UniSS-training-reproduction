#!/usr/bin/env bash
# Robust wave 4, diagnosis #4 (views): start the placeholder watcher (wave4_views_noplaceholder.sh) and the
# wave4_views.py workers on cards 4-7, PER_CARD per card. Before placing on a card, the no-checkpoint placeholders of that
# card (scripts/gpu_load/{run_mist_workload.sh,train_mist_workload.py} --devices <card>) get SIGTERM by PID (argv
# re-checked); never the guard 2726951, gpu_watch 52045, sync 1222453. The next worker on a card starts only after the
# previous one there has loaded the Thinker (" ready on " in its log). Worker k logs to logs/mist_w3po_v1/wave4/views_worker<k>.log.
#   SMOKE=N   one worker on the first card of CARDS with --smoke N (PIDs views_smoke.pids), nothing else
#   DRY_RUN=1 print, start nothing.   CARDS="4 5 6 7" PER_CARD=2 NEED_MIB=72000
# Detached: setsid nohup bash .../wave4_views_launch.sh > logs/mist_w3po_v1/wave4/views_launch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
L=logs/mist_w3po_v1/wave4
CARDS=${CARDS:-"4 5 6 7"}
PER_CARD=${PER_CARD:-2}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
NEED_MIB=${NEED_MIB:-72000}
DEADLINE=${DEADLINE:-3600}
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 GSTAR_THREADS=6"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
mkdir -p $L
if [ "$SMOKE" != 0 ]; then PIDS=$L/views_smoke.pids; CARDS=${CARDS%% *}; PER_CARD=1; N=1; else PIDS=$L/views_workers.pids; N=$(( $(echo $CARDS | wc -w) * PER_CARD )); fi
[ "$DRY_RUN" != 1 ] && [ -e "$PIDS" ] && { log "refusing: $PIDS exists"; exit 1; }
PROTECT=" 2726951 52045 1222453 $$ "
stop_placeholders() {
  local g=$1
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g";; esac
  done < <(ps -eo pid=,args=)
}
if [ "$DRY_RUN" = 1 ]; then
  log "[dry] watcher PIDS=$PIDS STOP=$L/views_noplaceholder.stop; $N workers on cards $CARDS"
  for c in $CARDS; do log "[dry] card $c: CUDA_VISIBLE_DEVICES=$c $PY -I experiments/mist_w3po_v1/voicecond/wave4_views.py --worker k$([ "$SMOKE" != 0 ] && echo " --smoke $SMOKE")"; done
  exit 0
fi
touch "$PIDS"
PIDS=$PIDS STOP=$L/views_noplaceholder.stop MAX_S=14400 setsid nohup bash experiments/mist_w3po_v1/voicecond/wave4_views_noplaceholder.sh \
  > $L/views_noplaceholder_$(basename $PIDS .pids).log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
declare -A LOADING NMINE
w=0; placed=0; t0=$(date +%s)
while [ $placed -lt $N ] && [ $(( $(date +%s) - t0 )) -lt $DEADLINE ]; do
  for c in $CARDS; do
    [ $placed -ge $N ] && break
    if [ -n "${LOADING[$c]:-}" ]; then
      f=$L/views_worker${LOADING[$c]}.log; [ "$SMOKE" != 0 ] && f=$L/views_smoke_worker${LOADING[$c]}.log
      if grep -q -E " ready on |Traceback" "$f" 2>/dev/null; then LOADING[$c]=""; else continue; fi
    fi
    [ "${NMINE[$c]:-0}" -ge "$PER_CARD" ] && continue
    fr=$(free_mib $c)
    case "$fr" in ''|*[!0-9]*) continue;; esac
    if [ "$fr" -lt "$NEED_MIB" ]; then stop_placeholders $c; continue; fi
    args="--worker $w"; lf=$L/views_worker$w.log
    [ "$SMOKE" != 0 ] && { args="$args --smoke $SMOKE"; lf=$L/views_smoke_worker$w.log; }
    [ -e "$lf" ] && { log "refusing: $lf exists"; exit 1; }
    env $ENVS CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -I experiments/mist_w3po_v1/voicecond/wave4_views.py $args \
      > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
    echo "$c $! views_worker$w" >> "$PIDS"
    log "card $c ($fr MiB free): worker $w pid $!"
    LOADING[$c]=$w; NMINE[$c]=$(( ${NMINE[$c]:-0} + 1 )); w=$((w + 1)); placed=$((placed + 1))
  done
  [ $placed -lt $N ] && sleep 5
done
log "placed $placed of $N workers; launch done"
