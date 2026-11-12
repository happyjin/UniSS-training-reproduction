#!/usr/bin/env bash
# Move A'rl's remaining work off 4-7 so that D1 can start there now (aprime_rl_prereg_v1 §卡; coordinator 2026-11-12
# ~01:35 Prague: if D1 is ready and A'rl still needs more than ~40 min, its remaining jobs go to 0-3 after VE1):
#   1. both A'rl runs (t_eval/aprl, t_eval/aprlplem; skipped if DONE): exclude_cards <- all 8 cards (new file + command
#      mv -f; the orchestrators stay alive and re-read it every tick)
#   2. every running job attempt of theirs on cards 4-7 (state/<id>.a<n>.pid without .rc, .card in 4..7, session leader's
#      cmdline naming the run dir) has the processes of its session stopped by PID (SIGTERM, SIGKILL after 60 s); the
#      orchestrator retries them later (talk retries 5, ASR parts / scorers 3); record t_eval/<run>/moved_<UTC>.txt
#   3. wait (60 s polls) until VE1's training exited and, if rc=0, its dev5 quick check has started all 4 decodes
#      (t_eval/ve1quick) or written DONE, so that A'rl does not take memory ahead of VE1's quick check
#   4. exclude_cards <- "4 5 6 7" (0-3 open again; the shared GLOBAL_EXCLUDE still keeps any owned slot's cards out)
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/aprl_move_to_0123.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=logs/mist_w3po_v1/t_eval
VE1LOG=logs/mist_w3po_v1/ve1_20261111T214454Z.log
log() { echo "$(date -u +%FT%TZ) $*"; }
set_excl() {   # <run> <cards>
  local f=$E/$1/exclude_cards T
  T=$f.new_$(date -u +%Y%m%dT%H%M%SZ)_$$
  ( set -o noclobber; echo "$2" > $T ) && command mv -f $T $f && log "$1 exclude_cards <- '$2'"
}
running_on_4567() {   # <run>: "<job attempt> <sid> <card>"
  local pf sid lead c
  for pf in $E/$1/state/*.pid; do
    [ -e "$pf" ] || continue
    [ -e "${pf%.pid}.rc" ] && continue
    c=$(cat "${pf%.pid}.card" 2>/dev/null) || continue
    case "$c" in 4|5|6|7) ;; *) continue;; esac
    sid=$(tr -dc 0-9 < "$pf"); [ -n "$sid" ] || continue
    lead=$(tr '\0' ' ' 2>/dev/null < /proc/$sid/cmdline) || continue
    case "$lead" in *"$E/$1/"*) ;; *) continue;; esac
    echo "$(basename "${pf%.pid}") $sid $c"
  done
}
session_pids() { ps -eo pid=,sid= | awk -v s="$1" '$2 == s {print $1}'; }
ve1_quick_started() {
  [ -e $VE1LOG ] && grep -q CRF_RUN_EXIT $VE1LOG || return 1
  grep -q 'CRF_RUN_EXIT rc=0' $VE1LOG || return 0
  [ -e $E/ve1quick/events/DONE ] && return 0
  local k
  for k in 0 1 2 3; do ls $E/ve1quick/state/dev5_ve1_s3000__dec$k.a*.pid > /dev/null 2>&1 || return 1; done
  return 0
}
RUNS=""
for r in aprl aprlplem; do [ -d $E/$r ] && [ ! -e $E/$r/events/DONE ] && RUNS="$RUNS $r"; done
if [ -n "${DRY_RUN:-}" ]; then
  echo "runs to move:$RUNS"
  for r in $RUNS; do echo "$r exclude now '$(cat $E/$r/exclude_cards)'; running on 4-7:"; running_on_4567 $r; done
  echo "VE1 quick decodes started: $(ve1_quick_started && echo yes || echo no)"
  exit 0
fi
[ -n "$RUNS" ] || { log "no A'rl run left to move"; exit 0; }
for r in $RUNS; do set_excl $r "0 1 2 3 4 5 6 7"; done
sleep 5
for r in $RUNS; do
  P=$E/$r/moved_$(date -u +%Y%m%dT%H%M%SZ).txt
  ( set -o noclobber; echo "utc $(date -u +%FT%TZ): $r jobs on 4-7 stopped for D1 (aprl_move_to_0123.sh)" > $P )
  all=()
  while read -r job sid card; do
    [ -n "$job" ] || continue
    pids=$(session_pids "$sid")
    echo "job $job card $card sid $sid pids $(echo $pids)" >> $P
    for p in $pids; do kill "$p" 2>/dev/null; done
    all+=("$sid")
    log "$r job $job (card $card, sid $sid) stopped: $(echo $pids)"
  done < <(running_on_4567 $r)
  for i in $(seq 1 60); do
    left=""; for sid in "${all[@]}"; do left="$left $(session_pids $sid)"; done
    [ -z "${left// /}" ] && break
    sleep 1
  done
  [ -n "${left// /}" ] && for p in $left; do kill -9 "$p" 2>/dev/null && log "SIGKILL $p"; done
  echo "after: $(date -u +%FT%TZ)" >> $P
done
log "waiting for VE1's exit and its quick-check decodes"
until ve1_quick_started; do sleep 60; done
for r in $RUNS; do [ -e $E/$r/events/DONE ] || set_excl $r "4 5 6 7"; done
log "A'rl may use 0-3 now"
