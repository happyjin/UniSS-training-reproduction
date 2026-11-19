#!/usr/bin/env bash
# Robust wave 7 (task-driven front-end): keep the no-checkpoint placeholders off the cards that hold a live wave-7 job (wave7_fe_*.py, si_thinker_stream, si_infer, asr_transcribe, speech_only). Modelled on
# wave1_noplaceholder.sh. Every POLL s: for each line "card pid name" of $PIDS (written by wave7_fe_run.sh) whose pid is
# a live wave-7 job, every process whose argv
# runs scripts/gpu_load/train_mist_workload.py or scripts/gpu_load/run_mist_workload.sh with "--devices <card>" gets
# SIGTERM (PIDs from ps fields, argv re-checked in /proc right before the kill). Cards without a live wave-7 job are left
# alone, so the guard refills them. Never touches the guard (2726951), gpu_watch (52045), sync_train_exclude
# (1222453) or anything else. Runs until $STOP exists or MAX_S (default 12 h) has passed; stopped by PID at the end.
#   [POLL=5] bash experiments/mist_w3po_v1/voicecond/wave7_fe_noplaceholder.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave7/fe_noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
PIDS=${PIDS:-logs/mist_w3po_v1/wave7/fe_workers.pids}
STOP=${STOP:-logs/mist_w3po_v1/wave7/fe_noplaceholder.stop}
MAX_S=${MAX_S:-43200}
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "watcher $$: placeholders off the cards of live wave-7 jobs listed in $PIDS"
t0=$(date +%s); n=0
while [ ! -e "$STOP" ] && [ $(( $(date +%s) - t0 )) -lt "$MAX_S" ]; do
  cards=" "
  if [ -s "$PIDS" ]; then
    while read -r c p _; do
      [ -n "${p:-}" ] || continue
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *wave7_fe_*|*si_thinker_stream*|*si_infer*|*asr_transcribe*|*speech_only*) cards="$cards${c//,/ } ";; esac
    done < "$PIDS"
  fi
  if [ "$cards" != " " ]; then
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      for g in $cards; do
        case "$a" in
          *scripts/gpu_load/train_mist_workload.py*"--devices $g "*|*scripts/gpu_load/run_mist_workload.sh*"--devices $g "*)
            kill -TERM "$pid" 2>/dev/null && { n=$((n + 1)); log "SIGTERM placeholder $pid (card $g): ${a:0:160}"; }; break;;
        esac
      done
    done < <(ps -eo pid=,args=)
  fi
  sleep "$POLL"
done
log "watcher exit after $n placeholder stops"
