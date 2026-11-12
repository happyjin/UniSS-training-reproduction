#!/bin/bash
# Start Omni engines for <ROOT> as GPU room appears (coordinator rule 2026-11-12 ~20:40 Prague): every POLL s (>= 60) read
# the real nvidia-smi use (pick_cards.py); when one card has >= 93 GiB free and another >= 39 GiB (or one card >= 119 GiB),
# start ONE engine there through start_engines.sh (MAX_TRIES=1) under the shared gpu_launch.lock (flock -o: the lock is
# not inherited), after writing pending entries to the shared ledger so the other orchestrators count the memory before
# it shows. At most MAX_ENGINES live engines for this ROOT; >= GAP s between my starts; cards of an engine started less
# than SETTLE s ago are not reused. Exits when every shard of ROOT is claimed or done (AUTO_LAUNCH_DONE).
#
#   ROOT=eval_outputs/s2st_baselines_v1/qabc_omnitalker_realsi MAX_ENGINES=3 \
#       setsid nohup bash experiments/s2st_baselines_v1/qabc_omnitalker/auto_launch.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
ROOT=${ROOT:?ROOT=<work folder with jobs/>}
MAX_ENGINES=${MAX_ENGINES:-3}
POLL=${POLL:-60}
GAP=${GAP:-120}
SETTLE=${SETTLE:-300}
LOCK=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/gpu_launch.lock
D=experiments/s2st_baselines_v1/qabc_omnitalker
ts() { TZ=Europe/Prague date +%H:%M:%S; }
declare -A started   # card -> epoch of my last engine start there
last=0

unclaimed() {   # number of shards with neither claim nor done marker
  local n=0 f k
  for f in "$ROOT"/jobs/shard_*.jsonl; do
    k=$(basename "$f" .jsonl)
    [ -e "$ROOT/claims/$k" ] || [ -e "$ROOT/synth/done_${k#shard_}" ] || n=$((n + 1))
  done
  echo $n
}

my_engines() {   # live synth processes working on ROOT (--root or --then-root; main process: parent 1 after setsid nohup)
  ps -eo pid=,ppid=,args= | awk -v r="-root $ROOT " '$2 == 1 && $0 ~ /qabc_omnitalker\.synth/ && index($0 " ", r) && $0 !~ /awk/' | wc -l
}

echo "[$(ts)] auto_launch ROOT=$ROOT MAX_ENGINES=$MAX_ENGINES POLL=$POLL"
while true; do
  u=$(unclaimed)
  if [ "$u" -eq 0 ]; then echo "[$(ts)] every shard claimed or done; AUTO_LAUNCH_DONE"; exit 0; fi
  n=$(my_engines)
  now=$(date +%s)
  if [ "$n" -lt "$MAX_ENGINES" ] && [ $((now - last)) -ge "$GAP" ]; then
    ex=""
    for c in "${!started[@]}"; do [ $((now - ${started[$c]})) -lt "$SETTLE" ] && ex="$ex $c"; done
    pick=$(python3 "$D/pick_cards.py" --exclude $ex)
    if [ -n "$pick" ]; then
      T=${pick%%:*}; K=${pick##*:}
      echo "[$(ts)] room: engine $T:$K ($u shards unclaimed, $n engines live); $(python3 "$D/pick_cards.py" --verbose --exclude $ex | head -1)"
      t=$(date +%s)
      out=$(ROOT=$ROOT flock -w 600 -o "$LOCK" bash $D/launch_one.sh "$T" "$K" 2>&1)
      echo "$out"
      if echo "$out" | grep -q "engine $T:$K pid"; then
        started[$T]=$t; started[$K]=$t; last=$t
      else
        echo "[$(ts)] engine $T:$K not started"
      fi
    fi
  fi
  sleep "$POLL"
done
