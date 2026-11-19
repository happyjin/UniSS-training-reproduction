#!/usr/bin/env bash
# Wave 1 Thinker prechecks (K2/K3/K7/K8/K9): keep the no-checkpoint placeholders off cards 4-7 ONLY (another agent owns
# cards 0-3) while this precheck's workers run. Modelled on idea6_test_noplaceholder_v2.sh. Every POLL s: every process
# whose argv runs scripts/gpu_load/train_mist_workload.py or scripts/gpu_load/run_mist_workload.sh with "--devices N"
# for N in 4..7 (supervisors and workers; PIDs from ps fields, argv re-checked in /proc right before the kill) gets
# SIGTERM. Never touches the guard (2726951), gpu_watch (52045), sync_train_exclude (1222453), cards 0-3 or anything
# else. Exits once the worker PID file ($PIDS, one "card pid" per line, written by wave1_launch.sh) lists at least one
# PID and none of the listed PIDs is a live wave1_thinker.py process; also exits if no PID was listed after 45 min.
# The guard then refills idle cards.
#   [POLL=5] bash experiments/mist_w3po_v1/voicecond/wave1_noplaceholder.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave1/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
PIDS=${PIDS:-logs/mist_w3po_v1/wave1/workers.pids}
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "watcher $$: placeholders off cards 4-7 while the workers in $PIDS live"
t0=$(date +%s)
alive() {
  local n=0 c p a
  if [ -s "$PIDS" ]; then
    while read -r c p; do
      [ -n "${p:-}" ] || continue
      n=$((n + 1))
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *wave1_thinker.py*) return 0;; esac
    done < "$PIDS"
  fi
  [ $n -eq 0 ] && [ $(( $(date +%s) - t0 )) -lt 2700 ]
}
n=0
while alive; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in
      *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*) ;;
      *) continue;;
    esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in
      *scripts/gpu_load/train_mist_workload.py*"--devices 4 "*|*scripts/gpu_load/train_mist_workload.py*"--devices 5 "*|\
      *scripts/gpu_load/train_mist_workload.py*"--devices 6 "*|*scripts/gpu_load/train_mist_workload.py*"--devices 7 "*|\
      *scripts/gpu_load/run_mist_workload.sh*"--devices 4 "*|*scripts/gpu_load/run_mist_workload.sh*"--devices 5 "*|\
      *scripts/gpu_load/run_mist_workload.sh*"--devices 6 "*|*scripts/gpu_load/run_mist_workload.sh*"--devices 7 "*)
        kill -TERM $pid 2>/dev/null && { n=$((n + 1)); log "SIGTERM placeholder $pid: ${a:0:160}"; };;
    esac
  done < <(ps -eo pid=,args=)
  sleep $POLL
done
log "no wave1 worker alive; exit after $n placeholder stops"
