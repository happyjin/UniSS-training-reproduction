#!/usr/bin/env bash
# Wave 3 group T: start the cards-6-7 placeholder watcher (wave3_T_noplaceholder.sh), wait until the target
# cards have room, then start the wave3_T_thinker.py workers by free memory, at most PER_CARD per card (the next one on a
# card only after the previous one has loaded the Thinker, so two loads never peak together). Worker k: logs/mist_w3po_v1/wave3/T_worker<k>.log.
#   SMOKE=N   one worker on the first card of CARDS with --smoke N (PID file smoke.pids), nothing else
#   DRY_RUN=1 print what would be started, start nothing
#   CARDS="6 7" PER_CARD=2 W0=0  (worker ids W0, W0+1, ...)
# Detached: setsid nohup bash experiments/mist_w3po_v1/voicecond/wave3_T_launch.sh > logs/mist_w3po_v1/wave3/T_launch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
L=logs/mist_w3po_v1/wave3
CARDS=${CARDS:-"6 7"}
PER_CARD=${PER_CARD:-2}
W0=${W0:-0}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 GSTAR_THREADS=6"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ "$SMOKE" != 0 ]; then PIDS=$L/T_smoke.pids; CARDS=${CARDS%% *}; PER_CARD=1; else PIDS=$L/T_workers.pids; fi
[ -e "$PIDS" ] && { log "refusing: $PIDS exists"; exit 1; }
mkdir -p $L

if [ "$DRY_RUN" = 1 ]; then
  log "[dry] watcher: PIDS=$PIDS bash experiments/mist_w3po_v1/voicecond/wave3_T_noplaceholder.sh"
else
  touch "$PIDS"
  PIDS=$PIDS setsid nohup bash experiments/mist_w3po_v1/voicecond/wave3_T_noplaceholder.sh \
    > $L/T_noplaceholder_$(basename $PIDS .pids)_$(date +%H%M%S).log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher started (pid $!)"
fi

free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
# placement by free memory (cards are shared with other agents' jobs): a worker goes to a card of CARDS with
# >= NEED_MIB free, no worker of ours still loading there and < PER_CARD of ours; until N_WORKERS are placed or DEADLINE s
N_WORKERS=${N_WORKERS:-4}; [ "$SMOKE" != 0 ] && N_WORKERS=1
NEED_MIB=${NEED_MIB:-72000}
DEADLINE=${DEADLINE:-10800}
declare -A LOADING NMINE
w=$W0; placed=0; t0=$(date +%s)
while [ $placed -lt $N_WORKERS ] && [ $(( $(date +%s) - t0 )) -lt $DEADLINE ]; do
  for c in $CARDS; do
    [ $placed -ge $N_WORKERS ] && break
    if [ -n "${LOADING[$c]:-}" ]; then
      f=$L/T_worker${LOADING[$c]}.log; [ "$SMOKE" != 0 ] && f=$L/T_smoke_worker${LOADING[$c]}.log
      if grep -q -E " ready on |Traceback" "$f" 2>/dev/null; then LOADING[$c]=""; else continue; fi
    fi
    [ "${NMINE[$c]:-0}" -ge "$PER_CARD" ] && continue
    fr=$(free_mib $c)
    case "$fr" in ''|*[!0-9]*) continue;; esac
    [ "$fr" -lt "$NEED_MIB" ] && continue
    args="--worker $w"; [ "$SMOKE" != 0 ] && args="$args --smoke $SMOKE"
    lf=$L/T_worker$w.log; [ "$SMOKE" != 0 ] && lf=$L/T_smoke_worker$w.log
    if [ -e "$lf" ]; then log "refusing: $lf exists"; exit 1; fi
    if [ "$DRY_RUN" = 1 ]; then
      log "[dry] card $c ($fr MiB free) worker $w: CUDA_VISIBLE_DEVICES=$c $PY -I experiments/mist_w3po_v1/voicecond/wave3_T_thinker.py $args > $lf"
    else
      env $ENVS CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -I experiments/mist_w3po_v1/voicecond/wave3_T_thinker.py $args \
        > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
      echo "$c $!" >> "$PIDS"
      log "card $c ($fr MiB free): worker $w pid $!"
      LOADING[$c]=$w
    fi
    NMINE[$c]=$(( ${NMINE[$c]:-0} + 1 ))
    w=$((w + 1)); placed=$((placed + 1))
  done
  [ $placed -lt $N_WORKERS ] && sleep 5
done
log "placed $placed of $N_WORKERS workers"
log "launch done"
