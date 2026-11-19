#!/usr/bin/env bash
# Wave 3 group G: keep the no-checkpoint placeholders off card 5 ONLY (cards 0-2 S1, 3-4 V, 6-7 T) while G's jobs run.
# Modelled on wave1_noplaceholder.sh. Every POLL s: every process whose argv runs scripts/gpu_load/train_mist_workload.py
# or scripts/gpu_load/run_mist_workload.sh with "--devices 5 " (PIDs from ps fields, argv re-checked in /proc right
# before the kill) gets SIGTERM. Never touches the guard (2726951), gpu_watch (52045), sync (1222453) or anything else.
# Stays while a G job listed in $PIDS ("card pid name") lives, or while $HOLD exists (between stages); exits otherwise
# (also if nothing was listed after 45 min). The guard then refills the card.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/wave3_g_noplaceholder.sh > logs/mist_w3po_v1/wave3/g_noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
PIDS=${PIDS:-logs/mist_w3po_v1/wave3/g_workers.pids}
HOLD=${HOLD:-logs/mist_w3po_v1/wave3/g_hold}
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "watcher $$: placeholders off card 5 while the jobs in $PIDS live or $HOLD exists"
t0=$(date +%s)
alive() {
  local n=0 c p nm a
  [ -e "$HOLD" ] && return 0
  if [ -s "$PIDS" ]; then
    while read -r c p nm; do
      [ -n "${p:-}" ] || continue
      n=$((n + 1))
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *wave3_g_cfg*|*asr_transcribe*|*speech_only*) return 0;; esac
    done < "$PIDS"
  fi
  [ $n -eq 0 ] && [ $(( $(date +%s) - t0 )) -lt 2700 ]
}
n=0
while alive; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in
      *scripts/gpu_load/train_mist_workload.py*"--devices 5 "*|*scripts/gpu_load/run_mist_workload.sh*"--devices 5 "*)
        kill -TERM $pid 2>/dev/null && { n=$((n + 1)); log "SIGTERM placeholder $pid: ${a:0:160}"; };;
    esac
  done < <(ps -eo pid=,args=)
  sleep $POLL
done
log "no G job alive and no hold; exit after $n placeholder stops"
