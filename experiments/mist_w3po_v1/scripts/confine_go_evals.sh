#!/usr/bin/env bash
# Keep each GO candidate's dev full eval on its own card group (coordinator 2026-11-11 ~23:15 Prague: T5's on 0-3, T4's
# on 4-7, so that VE1 (0-3) and D1 (4-7) are not both held up). t_after_train.sh starts the full eval orchestrator with
# --exclude = the cards of the slots owned at that moment; this adds the other group: as soon as
# logs/mist_w3po_v1/t_eval/<t>/exclude_cards exists (t4 -> + "0 1 2 3", t5 -> + "4 5 6 7"), a new file with the union is
# written and moved into place (command mv -f; the orchestrator re-reads it every tick; jobs already running elsewhere are
# not touched; cards never sit idle: the guard's placeholders and the other evals take the other group). Polls every 5 s;
# ends when, for both t4 and t5, the quick check is resolved (go.json, or the training exited rc != 0) and a GO full
# eval is DONE (or after 24 h). bash -n / DRY_RUN only.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/confine_go_evals.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=logs/mist_w3po_v1/t_eval
TE=eval_outputs/mist_w3po_v1/crf/t_eval
declare -A LOGF=([t4]=logs/mist_w3po_v1/t4_dart_t0_w10_20261111T173134Z.log [t5]=logs/mist_w3po_v1/t5_evid_p4_20261111T154210Z.log)
declare -A OTHER=([t4]="0 1 2 3" [t5]="4 5 6 7")
log() { echo "$(date -u +%FT%TZ) $*"; }
union() { echo "$1 $2" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ' | sed 's/ $//'; }
settled() {   # t's quick check resolved and no GO full eval left to confine
  local t=$1 g=$TE/dev5_$1_s3000/go.json
  if [ -e $g ]; then
    grep -q '"GO": true' $g || return 0
    [ -e $E/$t/events/DONE ] && return 0
    return 1
  fi
  [ -e ${LOGF[$t]} ] && grep -q CRF_RUN_EXIT ${LOGF[$t]} && ! grep -q 'CRF_RUN_EXIT rc=0' ${LOGF[$t]} && return 0
  return 1
}
if [ -n "${DRY_RUN:-}" ]; then
  for t in t4 t5; do
    f=$E/$t/exclude_cards
    echo "$t: settled $(settled $t && echo yes || echo no); exclude_cards $([ -e $f ] && echo "'$(cat $f)' -> '$(union "$(cat $f)" "${OTHER[$t]}")'" || echo 'not there yet')"
  done
  exit 0
fi
T0=$(date +%s)
log "start"
while ! { settled t4 && settled t5; }; do
  for t in t4 t5; do
    f=$E/$t/exclude_cards
    [ -e $f ] && [ ! -e $E/$t/events/DONE ] || continue
    cur=$(cat $f 2>/dev/null); want=$(union "$cur" "${OTHER[$t]}")
    if [ "$want" != "$(union "$cur" "")" ]; then
      T=$f.new_$(date -u +%Y%m%dT%H%M%SZ)
      ( set -o noclobber; echo "$want" > $T ) && command mv -f $T $f && log "$t full eval exclude_cards '$cur' -> '$want'"
    fi
  done
  [ $(( $(date +%s) - T0 )) -lt 86400 ] || { log "24 h: stop"; exit 0; }
  sleep 5
done
log "t4 and t5 settled"
