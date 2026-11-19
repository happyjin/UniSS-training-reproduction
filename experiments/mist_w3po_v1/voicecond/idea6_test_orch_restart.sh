#!/usr/bin/env bash
# Idea 6 full-system test (2026-11-19 ~16:10 Prague, coordinator / user: more jobs per card): restart the 18 test
# orchestrators ONE AT A TIME with --resume (thinker_t_eval_orch --resume -> finalpost_dev_orch.Orch.adopt: running jobs
# are setsid -f sessions and are adopted, finished ones read from state/<id>.a<n>.rc, nothing is relaunched or killed)
# and the throughput overrides of voicecond/idea6_test_orch.py (I6_LAUNCH_GAP_TALK=2 I6_LAUNCH_GAP_ASR=2 I6_NEED_TALK=21
# I6_OTHER_WINDOW_S=60). Per orchestrator PID (from logs/mist_w3po_v1/voicecond/idea6/test_run.log "orchestrator pid N"):
# its argv and environment are copied from /proc (relaunch copies the original env verbatim), SIGTERM to that PID only,
# wait for it to exit, relaunch the same argv + --resume under the copied env + overrides (stdout -> <run>.resume.log),
# wait until its orch.log says "resumed:" (<= 300 s), then the next one. An orchestrator already finished (events/DONE)
# or not alive is skipped. Writes the new PIDs to logs/mist_w3po_v1/voicecond/idea6/test_orch_pids_v2.txt.
#   bash experiments/mist_w3po_v1/voicecond/idea6_test_orch_restart.sh      (checked with bash -n only)
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea6/test_orch_restart.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
L=logs/mist_w3po_v1/voicecond/idea6
OUTP=$L/test_orch_pids_v2.txt
[ -e $OUTP ] && { log "refuse: $OUTP exists"; exit 1; }
OV="I6_LAUNCH_GAP_TALK=2 I6_LAUNCH_GAP_ASR=2 I6_NEED_TALK=21 I6_OTHER_WINDOW_S=60"
for p in $(sed -n 's/.* orchestrator pid \([0-9][0-9]*\)$/\1/p' $L/test_run.log); do
  [ -r /proc/$p/cmdline ] || { log "pid $p not alive, skipped"; continue; }
  mapfile -d '' ARGV < /proc/$p/cmdline
  case "${ARGV[*]}" in *voicecond/idea6_test_orch.py*) ;; *) log "pid $p is not a test orchestrator: ${ARGV[*]:0:3}"; continue;; esac
  R=""; for ((i = 0; i < ${#ARGV[@]}; i++)); do [ "${ARGV[$i]}" = "--run" ] && R=${ARGV[$((i + 1))]}; done
  [ -n "$R" ] || { log "pid $p: no --run"; continue; }
  [ -e $R/events/DONE ] && { log "$R done already, skipped"; continue; }
  mapfile -d '' ENVV < /proc/$p/environ
  CWD=$(readlink /proc/$p/cwd)
  n0=$(grep -c "resumed:" $R/orch.log 2>/dev/null || true)
  log "restart $(basename $R): SIGTERM $p"
  kill -TERM $p
  for i in $(seq 1 60); do kill -0 $p 2>/dev/null || break; sleep 1; done
  kill -0 $p 2>/dev/null && { log "pid $p still alive after 60 s; NOT relaunching $(basename $R)"; continue; }
  ( cd "$CWD" && env -i "${ENVV[@]}" $OV setsid nohup "${ARGV[@]}" --resume < /dev/null >> $R.resume.log 2>&1 8>&- 9>&- & echo $! > $R.resume.pid )
  np=$(cat $R.resume.pid)
  echo "$np $R" >> $OUTP
  for i in $(seq 1 60); do [ "$(grep -c "resumed:" $R/orch.log 2>/dev/null || true)" -gt "$n0" ] && break; sleep 5; done
  log "$(basename $R): new pid $np; $(grep "resumed:" $R/orch.log | tail -1 | cut -c1-160)"
done
log "restart done"
