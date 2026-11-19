#!/usr/bin/env bash
# Wave 1 Thinker prechecks: start the cards-4-7 placeholder watcher (wave1_noplaceholder.sh), wait until the target
# cards are empty, then start the wave1_thinker.py workers, at most PER_CARD per card (the second one on a card only
# after the first has loaded B, so two loads never peak together). Worker k: logs/mist_w3po_v1/wave1/worker<k>.log.
#   SMOKE=N   one worker on the first card of CARDS with --smoke N (PID file smoke.pids), nothing else
#   DRY_RUN=1 print what would be started, start nothing
#   CARDS="4 5 6 7" PER_CARD=2 W0=0  (worker ids W0, W0+1, ...)
# Detached: setsid nohup bash experiments/mist_w3po_v1/voicecond/wave1_launch.sh > logs/mist_w3po_v1/wave1/launch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
L=logs/mist_w3po_v1/wave1
CARDS=${CARDS:-"4 5 6 7"}
PER_CARD=${PER_CARD:-2}
W0=${W0:-0}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 GSTAR_THREADS=6"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ "$SMOKE" != 0 ]; then PIDS=$L/smoke.pids; CARDS=${CARDS%% *}; PER_CARD=1; else PIDS=$L/workers.pids; fi
[ -e "$PIDS" ] && { log "refusing: $PIDS exists"; exit 1; }
mkdir -p $L

if [ "$DRY_RUN" = 1 ]; then
  log "[dry] watcher: PIDS=$PIDS bash experiments/mist_w3po_v1/voicecond/wave1_noplaceholder.sh"
else
  touch "$PIDS"
  PIDS=$PIDS setsid nohup bash experiments/mist_w3po_v1/voicecond/wave1_noplaceholder.sh \
    > $L/noplaceholder_$(basename $PIDS .pids)_$(date +%H%M%S).log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher started (pid $!)"
fi

used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
for c in $CARDS; do
  [ "$DRY_RUN" = 1 ] && { log "[dry] would wait for card $c < 3000 MiB (now $(used $c))"; continue; }
  for _ in $(seq 1 120); do [ "$(used $c)" -lt 3000 ] && break; sleep 5; done
  log "card $c: $(used $c) MiB used"
done

w=$W0
declare -a FIRST
for round in $(seq 1 $PER_CARD); do
  for c in $CARDS; do
    if [ "$round" -gt 1 ] && [ "$DRY_RUN" != 1 ]; then   # wait for the first worker of this card to report "ready" (<= 15 min)
      f=$L/worker${FIRST[$c]}.log
      for _ in $(seq 1 180); do grep -q " ready on " "$f" 2>/dev/null && break; grep -q "Traceback" "$f" 2>/dev/null && break; sleep 5; done
    fi
    args="--worker $w"; [ "$SMOKE" != 0 ] && args="$args --smoke $SMOKE"
    lf=$L/worker$w.log; [ "$SMOKE" != 0 ] && lf=$L/smoke_worker$w.log
    if [ -e "$lf" ]; then log "refusing: $lf exists"; exit 1; fi
    if [ "$DRY_RUN" = 1 ]; then
      log "[dry] card $c worker $w: CUDA_VISIBLE_DEVICES=$c $PY -I experiments/mist_w3po_v1/voicecond/wave1_thinker.py $args > $lf"
    else
      env $ENVS CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -I experiments/mist_w3po_v1/voicecond/wave1_thinker.py $args \
        > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
      echo "$c $!" >> "$PIDS"
      log "card $c: worker $w pid $!"
    fi
    [ "$round" = 1 ] && FIRST[$c]=$w
    w=$((w + 1))
  done
done
log "launch done"
