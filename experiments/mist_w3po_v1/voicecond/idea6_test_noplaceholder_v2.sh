#!/usr/bin/env bash
# v2 (16:10 Prague): exits once all 18 run dirs have events/DONE (the orchestrators are restarted with --resume, so
# their PIDs change); otherwise identical to v1.
# Idea 6 full-system test (coordinator 2026-11-19 ~15:55 Prague, user: "尽可能每个gpu叠加多个测试任务"): keep the
# no-checkpoint placeholders off ALL cards while the test's orchestrators are alive. Every POLL s: every process whose
# argv runs scripts/gpu_load/train_mist_workload.py or scripts/gpu_load/run_mist_workload.sh (supervisors and workers;
# PIDs from ps fields, argv re-checked in /proc right before the kill) gets SIGTERM. Never touches the guard (2726951),
# gpu_watch (52045), sync_train_exclude (1222453) or anything else. Exits once none of the orchestrator PIDs listed in
# logs/mist_w3po_v1/voicecond/idea6/test_run.log ("orchestrator pid N") is alive; the guard then refills idle cards.
#   [POLL=5] bash experiments/mist_w3po_v1/voicecond/idea6_test_noplaceholder.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea6/test_noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-5}
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
GL=logs/mist_w3po_v1/voicecond/idea6/test
log "watching the 18 run dirs in $GL until each has events/DONE"
alive() { local d n=0; for d in $GL/obsi_i6* $GL/idea2_ood3_i6*; do [ -d $d ] || continue; n=$((n + 1)); [ -e $d/events/DONE ] || return 0; done; [ $n -lt 18 ]; }
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
      *scripts/gpu_load/train_mist_workload.py*|*scripts/gpu_load/run_mist_workload.sh*)
        kill -TERM $pid 2>/dev/null && { n=$((n + 1)); log "SIGTERM placeholder $pid: ${a:0:160}"; };;
    esac
  done < <(ps -eo pid=,args=)
  sleep $POLL
done
log "all 18 test orchestrators DONE; exit after $n placeholder stops"
