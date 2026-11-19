#!/usr/bin/env bash
# Robust wave 5 main line M: keep the no-checkpoint placeholders off the cards of live M jobs. Modelled on
# wave2_s1_noplaceholder.sh. Every POLL s: for each line "cards pid name" of $PIDS (cards comma-separated) whose pid is
# live and runs wave5_m_train / torchrun / si_thinker_stream / wave5_m_merge, every process whose argv runs
# scripts/gpu_load/train_mist_workload.py or run_mist_workload.sh with "--devices <card>" gets SIGTERM (PIDs from ps
# fields, argv re-checked in /proc right before the kill). Never touches guard 2726951, gpu_watch 52045, sync 1222453.
# Runs until $STOP exists or MAX_S (default 48 h). Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
PIDS=${PIDS:-logs/mist_w3po_v1/wave5/m_workers.pids}
STOP=${STOP:-logs/mist_w3po_v1/wave5/m_noplaceholder.stop}
MAX_S=${MAX_S:-172800}
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
log "watcher $$: placeholders off the cards of live M jobs listed in $PIDS"
t0=$(date +%s); n=0
while [ ! -e "$STOP" ] && [ $(( $(date +%s) - t0 )) -lt "$MAX_S" ]; do
  cards=" "
  if [ -s "$PIDS" ]; then
    while read -r c p _; do
      [ -n "${p:-}" ] || continue
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *wave5_m_train*|*torch.distributed.run*|*si_thinker_stream*|*wave5_m_merge*) cards="$cards${c//,/ } ";; esac
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
