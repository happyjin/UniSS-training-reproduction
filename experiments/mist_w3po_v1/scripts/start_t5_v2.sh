#!/usr/bin/env bash
# T5 on cards 0-3, v2 (coordinator 2026-11-11 19:30 Prague / 18:30 UTC): start_t5.sh with two changes, the rest verbatim.
#   1. the wait is for t1ctrl, t2, t2plem only (each: events/DONE and no live orchestrator). T1's dev eval is archive-only
#      and goes last (user's order T1-ctrl > T2 > T1), so it no longer holds T5 back.
#   2. right before slotA is claimed, the T1 eval is paused: its orchestrator (python argv "... thinker_t_eval_orch.py
#      --run logs/mist_w3po_v1/t_eval/t1 ...") is stopped by PID, then every T1 job attempt still running (state/<id>.a<n>.pid
#      without .rc; the pid file holds the job's session id; the session leader's cmdline must name the T1 run dir or the
#      T1 decode dir) has the processes of its session stopped by PID (SIGTERM, SIGKILL after 60 s). Before / after
#      state goes to logs/mist_w3po_v1/t_eval/t1/paused_<UTC>.txt (new file). The jobs come back on --resume as retries
#      (decodes resume from their partial hyp files): t1_resume_after_t4.sh, on 4-7 after T4's training.
# Then exactly start_t5.sh: noclobber slotA claim, takeover_train.sh (our placeholders on 0-3 stopped by PID), T5 launch
# (run_crf_v2p.sh --objective latent_write --lw-mode t5 --dart <T5 manifest>), rank count. bash -n / DRY_RUN only.
#   RUN_ID=<id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t5_v2.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
C=eval_outputs/mist_w3po_v1/crf
E=logs/mist_w3po_v1/t_eval
SL=logs/mist_w3po_v1/slots
RP=logs/mist_w3po_v1/t4/read_period.txt
T1R=$E/t1
T1N=dev_t1_piece_t0_s3000_w05
log() { echo "$(date -u +%FT%TZ) $*"; }
# python processes whose argv is exactly <python> experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run <dir> ...
# (a shell whose command line merely mentions the script is not matched: its argv[0] is not python)
orch_pids() {
  ps -ww -eo pid=,args= | awk -v r="$1" '$2 ~ /\/python[0-9.]*$/ && $3 == "experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py" \
    && $4 == "--run" && $5 == r {print $1}'
}
evals_done() {
  for r in t1ctrl t2 t2plem; do
    [ -e $E/$r/events/DONE ] || return 1
    [ -z "$(orch_pids $E/$r)" ] || return 1
  done
  return 0
}
# running T1 job attempts: "<job attempt> <sid> <card>" (pid file present, rc file absent, session leader is a T1 job)
t1_running() {
  local pf sid lead
  for pf in $T1R/state/*.pid; do
    [ -e "$pf" ] || continue
    [ -e "${pf%.pid}.rc" ] && continue
    sid=$(tr -dc 0-9 < "$pf")
    [ -n "$sid" ] || continue
    lead=$(tr '\0' ' ' < /proc/$sid/cmdline 2>/dev/null) || continue
    case "$lead" in *"$T1R/"*|*"$T1N"*) ;; *) continue;; esac
    echo "$(basename "${pf%.pid}") $sid $(cat "${pf%.pid}.card" 2>/dev/null || echo -)"
  done
}
session_pids() { ps -eo pid=,sid= | awk -v s="$1" '$2 == s {print $1}'; }
pause_t1() {
  local ts P p job sid card pids left i
  ts=$(date -u +%Y%m%dT%H%M%SZ); P=$T1R/paused_$ts.txt
  ( set -o noclobber; { echo "utc $ts: T1 dev eval paused for T5 on 0-3 (coordinator 2026-11-11 18:30 UTC);" \
      "resume: t1_resume_after_t4.sh (--resume, on 4-7 after T4's training)"; echo "status before:";
      cat $T1R/status.json 2>/dev/null; echo; } > $P ) || { log "cannot create $P"; return 1; }
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
  { echo "after: $(date -u +%FT%TZ); left in the stopped sessions: ${left:-none}"; } >> $P
  log "T1 paused: $P"
}
if [ -z "${DRY_RUN:-}" ]; then
  until [ -s $RP ]; do sleep 10; done
fi
G=$(cat $RP 2>/dev/null || echo 500)
case "$G" in 500) S=$C/sched_t5_pieces4_w05_v1; D=$C/t5_dart_pieces4_w05_v1;; 1000) S=$C/sched_t5_pieces4_w10_v1; D=$C/t5_dart_pieces4_w10_v1;;
  *) log "bad read period '$G'"; exit 1;; esac
[ -e $S/summary.json ] && [ -e $D/summary.json ] || { log "missing $S or $D"; exit 1; }
EXTRA="--sched $S --objective latent_write --lw-mode t5 --dart $D"
LAUNCH="CARDS=0,1,2,3 RUN_ID=$RUN_ID PORT=29971 EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then
  echo "read period $G; LAUNCH: $LAUNCH"
  for r in t1ctrl t2 t2plem t1; do echo "$r: DONE $([ -e $E/$r/events/DONE ] && echo yes || echo no); orchestrator pids: $(orch_pids $E/$r | tr '\n' ' ')"; done
  evals_done && echo "evals done" || echo "evals running"
  echo "T1 jobs that would be stopped (job sid card):"; t1_running
  exit 0
fi
log "v2: read period $G ms: $S, $D; waiting for t1ctrl, t2, t2plem on 0-3 (T1 is paused, not waited for)"
until evals_done; do sleep 10; done
log "t1ctrl, t2, t2plem done"
pause_t1 || { log "T1 pause failed: T5 NOT launched (needs a decision)"; exit 1; }
( set -o noclobber; printf 'who: thinker-side executor (mist_w3po_v1 T4/T5)\ntask: T5 training %s (evidence-time labels, pieces4, DART, t5 lattice; %s) on cards 0-3, ~6-7 h; released by renaming to slotA.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$S" "$(date -u +%FT%TZ)" > $SL/slotA.owner ) && log "slotA.owner claimed for T5" || log "slotA.owner claim FAILED (someone holds slotA): launching anyway is NOT done"
[ -e $SL/slotA.owner ] && grep -q "$RUN_ID" $SL/slotA.owner || exit 1
CARDS=0,1,2,3 LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "T5 ranks running: $n"
