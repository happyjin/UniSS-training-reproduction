#!/usr/bin/env bash
# T4 launch on cards 4-7 (prereg §T4, rule C). With STOP_T3=1 (rule C triggered): stop t3_eval_wait.sh (PID 1628573,
# cmdline checked) first so it cannot release / relaunch anything, then the T3 training tree by PID (torchrun TERM, which
# forwards to its 4 workers; workers / torchrun / bash wrapper KILLed if still alive after 45 s; every cmdline checked
# to belong to the T3 run), rename slotB.owner -> slotB.released_<UTC>, claim slotB.owner for T4 (noclobber), and launch
# T4 at once through takeover_train.sh (stops any of our placeholders on the cards, supervisor first) + run_crf_v2p.sh.
# Without STOP_T3 only the slot claim + takeover + launch. Afterwards checks for 150 s that every card has a T4 rank.
#   SCHED=<dir> DART=<manifest dir> RUN_ID=<id> [STOP_T3=1] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t4.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SCHED=${SCHED:?}
DART=${DART:?}
RUN_ID=${RUN_ID:?}
CARDS=4,5,6,7
T3=t3_joint_t0_20261111T095841Z
W3=3835909
WAIT3=1628573
SL=logs/mist_w3po_v1/slots
log() { echo "$(date -u +%FT%TZ) $*"; }
[ -e "$SCHED/summary.json" ] && [ -e "$DART/summary.json" ] || { log "missing $SCHED or $DART summary"; exit 1; }
[ -e eval_outputs/mist_w3po_v1/crf/train/$RUN_ID ] && { log "拒绝覆盖已有输出: $RUN_ID"; exit 1; }
EXTRA="--sched $SCHED --objective latent_write --dart $DART"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=29991 EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
cmd_of() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
if [ -n "${STOP_T3:-}" ]; then
  wc=$(cmd_of $W3)
  case "$wc" in *"torch.distributed.run"*"--out eval_outputs/mist_w3po_v1/crf/train/$T3"*) ;;
    *) log "PID $W3 is not the T3 wrapper: '$wc'"; exit 1;; esac
  TR_PIDS=$(ps -eo pid=,ppid= | awk -v p=$W3 '$2==p {print $1}')
  WK=()
  for t in $TR_PIDS; do
    case "$(cmd_of $t)" in *"torch.distributed.run"*"$T3"*) ;; *) log "unexpected child $t of $W3: $(cmd_of $t)"; exit 1;; esac
    for k in $(ps -eo pid=,ppid= | awk -v p=$t '$2==p {print $1}'); do
      case "$(cmd_of $k)" in *"train_crf_sft.py"*"$T3"*) WK+=("$k");; *) log "unexpected worker $k: $(cmd_of $k)"; exit 1;; esac
    done
  done
  log "T3 tree: wrapper $W3 (pgid $(ps -o pgid= -p $W3 | tr -d ' ')), torchrun $TR_PIDS, workers ${WK[*]}"
  ps -eo pid=,pgid=,cmd= | awk -v g=$W3 '$2==g' | cut -c1-160
  case "$(cmd_of $WAIT3)" in *"t3_eval_wait.sh $T3"*) ;; *) log "PID $WAIT3 is not t3_eval_wait.sh: $(cmd_of $WAIT3)"; WAIT3="";; esac
  if [ -n "${DRY_RUN:-}" ]; then echo "would stop $WAIT3, then $TR_PIDS (+ ${WK[*]}, $W3); LAUNCH: $LAUNCH"; exit 0; fi
  if [ -n "$WAIT3" ]; then
    SLP=$(ps -eo pid=,ppid= | awk -v p=$WAIT3 '$2==p {print $1}')
    kill $WAIT3 && log "stopped t3_eval_wait.sh $WAIT3 (T3 will not finish)"
    for s in $SLP; do kill $s 2>/dev/null; done
  fi
  kill -TERM $TR_PIDS && log "SIGTERM torchrun $TR_PIDS"
  for i in $(seq 1 45); do
    alive=0
    for p in $TR_PIDS "${WK[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
    sleep 1
  done
  for p in "${WK[@]}" $TR_PIDS $W3; do kill -0 $p 2>/dev/null && kill -KILL $p && log "SIGKILL $p"; done
  log "T3 stopped: $(tail -n 1 logs/mist_w3po_v1/$T3.log | cut -c1-120)"
fi
[ -n "${DRY_RUN:-}" ] && { echo "LAUNCH: $LAUNCH"; exit 0; }
if [ -e $SL/slotB.owner ]; then
  command mv -f $SL/slotB.owner $SL/slotB.released_$(date -u +%Y%m%dT%H%M%SZ) && log "slotB.owner released"
fi
( set -o noclobber; printf 'who: thinker-side executor (mist_w3po_v1 T4/T5)\ntask: T4 training %s (T2 + DART, %s) on cards 4-7, ~5-6 h; released by renaming to slotB.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$SCHED" "$(date -u +%FT%TZ)" > $SL/slotB.owner ) && log "slotB.owner claimed for T4" || log "slotB.owner claim FAILED"
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "T4 ranks running: $n"
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | head -20
