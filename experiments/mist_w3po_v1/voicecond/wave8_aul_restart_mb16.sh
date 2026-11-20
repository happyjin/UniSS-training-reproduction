#!/usr/bin/env bash
# Robust wave 8 AUL, one-shot speed-up restart (prereg "实现修正 2"): once run_aul has written resume_00500.pt and logged
# step 500, stop the post-training chain CHAIN_PID and the trainer (torchrun parent TPID, then its workers, PIDs taken
# from ps fields), and relaunch the SAME run dir with --mb 16 (one CE microbatch per rank; resumes from step 500), then a
# new wave8_aul_chain.sh on the new trainer. Never touches guard 2726951, gpu_watch 52045, sync 1222453.
#   TPID=<torchrun pid> CHAIN_PID=<chain pid> bash wave8_aul_restart_mb16.sh   (detached; checked with bash -n only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}" "${CHAIN_PID:?}"
M=eval_outputs/mist_w3po_v1/wave8/AUL
R=$M/run_aul
LOGD=logs/mist_w3po_v1/wave8
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
until [ -s $R/resume_00500.pt ] && grep -q '"step": 500,' $R/metrics_rank0.jsonl; do
  kill -0 "$TPID" 2>/dev/null || { log "trainer gone before step 500; nothing done"; exit 1; }
  sleep 10
done
log "resume_00500.pt written and step 500 logged"
a=$(tr '\0' ' ' < /proc/$CHAIN_PID/cmdline 2>/dev/null)
case "$a" in *wave8_aul_chain.sh*) kill -TERM "$CHAIN_PID" && log "SIGTERM chain $CHAIN_PID";; *) log "chain $CHAIN_PID not running ($a)";; esac
W=$(ps -eo pid=,ppid= | awk -v t="$TPID" '$2 == t {print $1}')
WW=$(for w in $W; do ps -eo pid=,ppid= | awk -v t="$w" '$2 == t {print $1}'; done)
a=$(tr '\0' ' ' < /proc/$TPID/cmdline 2>/dev/null)
case "$a" in *torch.distributed.run*wave8_aul_train.py*) kill -TERM "$TPID" && log "SIGTERM torchrun $TPID (workers $(echo $W))";;
  *) log "TPID $TPID is not the AUL torchrun: $a"; exit 1;; esac
for i in $(seq 60); do
  alive=""; for p in $TPID $W $WW; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
  [ -z "$alive" ] && break; sleep 2
done
for p in $W $WW; do
  a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$a" in *wave8_aul_train.py*) kill -KILL "$p" 2>/dev/null && log "SIGKILL leftover $p";; esac
done
sleep 20
log "old trainer stopped; relaunch with --mb 16 (resume)"
GPUS=4,5,6,7 STEPS=2300 OUT=$R NAME=run_aul_mb16 MB=16 bash experiments/mist_w3po_v1/voicecond/wave8_aul_run.sh || { log "relaunch failed"; exit 1; }
NT=$(tail -1 $LOGD/aul_workers.pids | awk '$3 == "run_aul_mb16" {print $2}')
[ -n "$NT" ] || { log "no new trainer pid"; exit 1; }
setsid nohup env TPID=$NT bash experiments/mist_w3po_v1/voicecond/wave8_aul_chain.sh > $LOGD/aul_chain_mb16.log 2>&1 < /dev/null 8>&- 9>&- &
NC=$!
echo "4,5,6,7 $NC wave8_aul_chain_mb16" >> $LOGD/aul_workers.pids
log "new trainer $NT, new chain $NC"
