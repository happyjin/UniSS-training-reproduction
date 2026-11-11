#!/usr/bin/env bash
# D1 training on cards 4-7 behind the evals that go first (coordinator for the user, 2026-11-11 ~23:00 Prague):
#   1. logs/mist_w3po_v1/d1/schedule.json (j1_decide --wait) says run_D1; otherwise exit (no D1);
#   2. the roll-in manifest of the base (crf/d1_<base>_w10_v1/summary.json, start_d1_rollin.sh) passed R4; otherwise exit;
#   3. every GO candidate's dev full eval is DONE (T4 / T5 go.json GO -> t_eval/<t>/events/DONE): GO-candidate evals
#      first (T4's is named by the coordinator; T5's may use 4-7 too);
#   4. then the cards are cleared for D1: marker logs/mist_w3po_v1/d1/claiming_4567 (start_t2rl_eval_v3.sh stops placing
#      new t2rl jobs on 4-7; the running ones finish), and the archive-only T1 eval, if its orchestrator is live, is
#      paused exactly as start_t5_v2.sh does (orchestrator by PID, then the processes of every running T1 job's session
#      by PID; record t_eval/t1/paused_<UTC>.txt; it resumes later with --resume on non-training cards).
# Then start_d1.sh unchanged (RUN_ID=d1_<base>_<UTC>, CARDS=4,5,6,7): it waits until slotB is free and only our
# placeholders are on 4-7, claims slotB, takes over, checks 4 ranks. Polls every 60 s. bash -n / DRY_RUN only.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1_after.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
TE=eval_outputs/mist_w3po_v1/crf/t_eval
L=logs/mist_w3po_v1/d1
SCH=$L/schedule.json
DEC=$L/base_decision.json
MARK=$L/claiming_4567
T1R=$E/t1
T1N=dev_t1_piece_t0_s3000_w05
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
is_go() { [ -e $TE/dev5_$1_s3000/go.json ] && grep -q '"GO": true' $TE/dev5_$1_s3000/go.json; }
go_evals_pending() { local t; for t in t4 t5; do is_go $t && ! [ -e $E/$t/events/DONE ] && echo -n "$t "; done; }
orch_pids() {
  ps -ww -eo pid=,args= | awk -v r="$1" '$2 ~ /\/python[0-9.]*$/ && $3 == "experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py" \
    && $4 == "--run" && $5 == r {print $1}'
}
t1_running() {
  local pf sid lead
  for pf in $T1R/state/*.pid; do
    [ -e "$pf" ] || continue
    [ -e "${pf%.pid}.rc" ] && continue
    sid=$(tr -dc 0-9 < "$pf")
    [ -n "$sid" ] || continue
    lead=$(tr '\0' ' ' 2>/dev/null < /proc/$sid/cmdline) || continue
    case "$lead" in *"$T1R/"*|*"$T1N"*) ;; *) continue;; esac
    echo "$(basename "${pf%.pid}") $sid $(cat "${pf%.pid}.card" 2>/dev/null || echo -)"
  done
}
session_pids() { ps -eo pid=,sid= | awk -v s="$1" '$2 == s {print $1}'; }
pause_t1() {
  local ts P p job sid card pids left="" i
  ts=$(date -u +%Y%m%dT%H%M%SZ); P=$T1R/paused_$ts.txt
  ( set -o noclobber; { echo "utc $ts: T1 dev eval paused for D1 on 4-7 (coordinator 2026-11-11 ~23:00 Prague);" \
      "resume later with --resume on non-training cards"; echo "status before:"; cat $T1R/status.json 2>/dev/null; echo; } > $P ) \
    || { log "cannot create $P"; return 1; }
  for p in $(orch_pids $T1R); do
    echo "orchestrator $p: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-400)" >> $P
    kill "$p" 2>/dev/null && log "T1 orchestrator $p stopped (SIGTERM)"
  done
  for i in $(seq 1 30); do [ -z "$(orch_pids $T1R)" ] && break; sleep 1; done
  [ -z "$(orch_pids $T1R)" ] || { log "T1 orchestrator still alive after 30 s: not touching its jobs"; return 1; }
  local all=()
  while read -r job sid card; do
    [ -n "$job" ] || continue
    pids=$(session_pids "$sid")
    echo "job $job card $card sid $sid pids $(echo $pids)" >> $P
    ps -o pid=,args= -p "$(echo $pids | tr ' ' ',')" 2>/dev/null | cut -c1-300 | sed 's/^/  /' >> $P
    for p in $pids; do kill "$p" 2>/dev/null; done
    all+=("$sid")
    log "T1 job $job (card $card, sid $sid) stopped: $(echo $pids)"
  done < <(t1_running)
  for i in $(seq 1 60); do
    left=""; for sid in "${all[@]}"; do left="$left $(session_pids $sid)"; done
    [ -z "${left// /}" ] && break
    sleep 1
  done
  if [ -n "${left// /}" ]; then
    for p in $left; do kill -9 "$p" 2>/dev/null && log "SIGKILL $p (still alive after 60 s)"; done
  fi
  echo "after: $(date -u +%FT%TZ); left in the stopped sessions: ${left:-none}" >> $P
  log "T1 paused: $P"
}
if [ -n "${DRY_RUN:-}" ]; then
  echo "schedule: $([ -s $SCH ] && field $SCH "d['run_D1']" || echo 'not written yet'); base: $([ -s $DEC ] && field $DEC "d['base']" || echo 'not yet')"
  echo "GO full evals pending: '$(go_evals_pending)'; marker $MARK: $([ -e $MARK ] && echo exists || echo absent)"
  echo "T1 orchestrator pids: $(orch_pids $T1R | tr '\n' ' '); running T1 jobs:"; t1_running
  [ -s $DEC ] && RUN_ID=d1_DRY CARDS=4,5,6,7 DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/start_d1.sh
  exit 0
fi
log "waiting for $SCH"
until [ -s $SCH ]; do sleep 60; done
[ "$(field $SCH "d['run_D1']")" = True ] || { log "schedule: run_D1 false -> no D1"; exit 0; }
until [ -s $DEC ]; do sleep 60; done
BASE=$(field $DEC "d['base']")
[ "$BASE" != None ] || { log "no D1 base"; exit 1; }
MAN=eval_outputs/mist_w3po_v1/crf/d1_${BASE}_w10_v1
log "run_D1 true, base $BASE; waiting for $MAN/summary.json (roll-in)"
until [ -s $MAN/summary.json ]; do sleep 60; done
[ "$(field $MAN/summary.json "d['R4_pass']")" = True ] || { log "R4 not passed: no D1 (prereg §4.2)"; exit 0; }
log "R4 passed"
P=$(go_evals_pending); last=""
while [ -n "$P" ]; do
  [ "$P" != "$last" ] && log "waiting for the GO full eval(s): $P" && last=$P
  sleep 60; P=$(go_evals_pending)
done
( set -o noclobber; echo "D1 $BASE claiming 4-7 (start_d1_after.sh), utc $(date -u +%FT%TZ)" > $MARK ) && log "marker $MARK written"
if [ -n "$(orch_pids $T1R)" ] || [ -n "$(t1_running)" ]; then
  [ -e $T1R/events/DONE ] || pause_t1 || log "T1 pause failed (D1 still waits for 4-7 to hold only placeholders)"
fi
RID=d1_${BASE}_$(date -u +%Y%m%dT%H%M%SZ)
log "start_d1.sh RUN_ID=$RID CARDS=4,5,6,7"
RUN_ID=$RID CARDS=4,5,6,7 bash experiments/mist_w3po_v1/scripts/start_d1.sh
