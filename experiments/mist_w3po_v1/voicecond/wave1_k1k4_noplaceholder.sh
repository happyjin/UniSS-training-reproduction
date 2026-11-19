#!/usr/bin/env bash
# Robust wave 1 K1/K4/K7v (cards 0-3 only; another agent owns 4-7): keep the no-checkpoint placeholders off cards 0-3
# while the prechecks run. Modelled on voicecond/idea6_test_noplaceholder_v2.sh. Every POLL s: every process whose argv
# runs scripts/gpu_load/train_mist_workload.py or scripts/gpu_load/run_mist_workload.sh (supervisors and workers; PIDs
# from ps fields, argv re-checked in /proc right before the kill) AND that sits on card 0-3 gets SIGTERM. The card is
# read from CUDA_VISIBLE_DEVICES in /proc/<pid>/environ (UUID or index) when set, else from the --devices argument.
# Never touches the guard (2726951), gpu_watch (52045), sync_train_exclude (1222453), anything on cards 4-7, or
# anything else. Exits once the STOP file exists (created by the K1/K4 run at its end; nothing is deleted).
#   [POLL=5] [STOP=<file>] bash experiments/mist_w3po_v1/voicecond/wave1_k1k4_noplaceholder.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave1/k1k4_noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
STOP=${STOP:-eval_outputs/mist_w3po_v1/wave1_K1K4/WATCH_STOP}
PROTECT=" 2726951 52045 1222453 $$ "
MINE="GPU-6fa29650-f0ee-f874-75ae-35cd8b373e0d GPU-ab4378e2-fbcf-0cfd-b924-f968b51196eb GPU-1d9124c9-fa34-7feb-87e3-5922a80f0ef6 GPU-c3426029-03b4-5ab7-807d-567070a8fc1e"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
on_mine() {   # $1 pid, $2 argv -> 0 iff the process is on card 0-3 (and on no card outside 0-3)
  local pid=$1 a=$2 cvd dv d u
  dv=$(sed -n 's/.*--devices[ =]\([0-9,]*\).*/\1/p' <<< "$a")
  if [ -n "$dv" ]; then   # the guard's placeholders: --devices = the physical card (workers' CUDA_VISIBLE_DEVICES agrees)
    for d in ${dv//,/ }; do case $d in 0|1|2|3) ;; *) return 1;; esac; done
    return 0
  fi
  cvd=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  case "$cvd" in ""|*,*) return 1;; esac   # unknown or several cards: leave it alone
  for u in $MINE 0 1 2 3; do [ "$cvd" = "$u" ] && return 0; done
  return 1
}
log "watching cards 0-3 until $STOP exists"
n=0
while [ ! -e "$STOP" ]; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in
      *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*) ;;
      *) continue;;
    esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in
      *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*)
        on_mine "$pid" "$a" || continue
        kill -TERM $pid 2>/dev/null && { n=$((n + 1)); log "SIGTERM placeholder $pid: ${a:0:160}"; };;
    esac
  done < <(ps -eo pid=,args=)
  sleep $POLL
done
log "STOP file found; exit after $n placeholder stops"
