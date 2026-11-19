#!/usr/bin/env bash
# Wave 4 diagnosis #2 (routing flips): start the placeholder watcher (wave4_route_noplaceholder.sh) and the
# wave4_route.py workers on cards 0-1, PER_CARD per card, the second one on a card only after the first has loaded.
#   SMOKE=N  one worker on card 0 with --smoke N (smoke output, separate pid file), nothing else
#   DRY_RUN=1 prints only. Detached: setsid nohup bash wave4_route_launch.sh > logs/mist_w3po_v1/wave4/route_launch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
L=logs/mist_w3po_v1/wave4
CARDS="0 1"; PER_CARD=${PER_CARD:-2}; SMOKE=${SMOKE:-0}; DRY_RUN=${DRY_RUN:-0}
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 GSTAR_THREADS=6"
PIDS=$L/route_workers.pids
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
mkdir -p $L
if [ "$SMOKE" != 0 ]; then CARDS=0; PER_CARD=1; NW=1; PIDS=$L/route_smoke.pids; else NW=$(( 2 * PER_CARD )); fi
[ -e "$PIDS" ] && [ "$DRY_RUN" != 1 ] && { log "refusing: $PIDS exists"; exit 1; }
PROTECT=" 2726951 52045 1222453 $$ "
stop_ph() {   # placeholders of card $1, by PID, argv re-checked (locals: do not clobber the caller's args)
  local pid pargs a
  while read -r pid pargs; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$pargs" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $1 "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $1";; esac
  done < <(ps -eo pid=,args=)
}
if [ "$DRY_RUN" != 1 ]; then
  touch "$PIDS"
  PIDS=$PIDS setsid nohup bash experiments/mist_w3po_v1/voicecond/wave4_route_noplaceholder.sh > $L/route_noplaceholder_$(date +%H%M%S).log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher pid $!"
fi
start() {   # card worker
  local c=$1 w=$2 args="--worker $2 --num-workers $NW"
  [ "$SMOKE" != 0 ] && args="--worker 0 --num-workers 1 --smoke $SMOKE"
  local lf=$L/route_worker$w.log; [ "$SMOKE" != 0 ] && lf=$L/route_smoke.log
  [ -e "$lf" ] && { log "refusing: $lf exists"; exit 1; }
  if [ "$DRY_RUN" = 1 ]; then log "[dry] card $c: $PY -I experiments/mist_w3po_v1/voicecond/wave4_route.py $args > $lf"; return; fi
  stop_ph $c; sleep 8
  env $ENVS CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -I experiments/mist_w3po_v1/voicecond/wave4_route.py $args > $lf 2>&1 < /dev/null 8>&- 9>&- &
  echo "$c $! route_w$w" >> "$PIDS"; log "card $c worker $w pid $!"
}
loaded() { grep -q -E "MoE layers|Traceback" "$1" 2>/dev/null; }
w=0
for c in $CARDS; do start $c $w; w=$((w + 1)); done
if [ "$PER_CARD" -ge 2 ] && [ "$SMOKE" = 0 ]; then
  for c in $CARDS; do
    i=$(( c == 0 ? 0 : 1 ))
    [ "$DRY_RUN" = 1 ] || until loaded $L/route_worker$i.log; do sleep 10; done
    start $c $w; w=$((w + 1))
  done
fi
log "launch done"
