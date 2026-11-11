#!/usr/bin/env bash
# VE1 on cards 0-3 behind the evals that go first (coordinator for the user, 2026-11-11 ~23:00 Prague): VE1 runs
# whatever run_D1 says (FORCE=1: the scheduling rule was changed by the coordinator to "run unconditionally"; the
# scientific GO lines are unchanged) and takes 0-3 only after
#   1. T5's dev5 quick check is over (T5 exited rc=0 and go.json written; a failed T5 keeps slotA -> start_ve1.sh waits),
#   2. every GO candidate's dev full eval is DONE (T4 / T5: crf/t_eval/dev5_<t>_s3000/go.json GO -> t_eval/<t>/events/DONE),
#   3. the t2rl eval is DONE (t_eval/t2rl/events/DONE).
# Polls every 60 s; then runs start_ve1.sh unchanged (RUN_ID, CARDS=0,1,2,3, FORCE=1), which itself waits until slotA is
# free and only our placeholders are on 0-3, claims slotA (noclobber), takes over and checks 4 ranks.
#   RUN_ID=ve1_<UTC> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_ve1_after.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
E=logs/mist_w3po_v1/t_eval
TE=eval_outputs/mist_w3po_v1/crf/t_eval
T5LOG=logs/mist_w3po_v1/t5_evid_p4_20261111T154210Z.log
log() { echo "$(date -u +%FT%TZ) $*"; }
is_go() { [ -e $TE/dev5_$1_s3000/go.json ] && grep -q '"GO": true' $TE/dev5_$1_s3000/go.json; }
why_wait() {
  if ! [ -e $T5LOG ] || ! grep -q CRF_RUN_EXIT $T5LOG; then echo "T5 training"; return; fi
  grep -q 'CRF_RUN_EXIT rc=0' $T5LOG && ! [ -e $TE/dev5_t5_s3000/go.json ] && { echo "T5 quick check"; return; }
  local t
  for t in t4 t5; do is_go $t && ! [ -e $E/$t/events/DONE ] && { echo "$t GO: its dev full eval"; return; }; done
  [ -e $E/t2rl/events/DONE ] || { echo "t2rl eval"; return; }
  echo ""
}
if [ -n "${DRY_RUN:-}" ]; then
  echo "waiting for: '$(why_wait)'"
  RUN_ID=$RUN_ID CARDS=0,1,2,3 FORCE=1 DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/start_ve1.sh
  exit 0
fi
log "VE1 $RUN_ID: waiting (T5 quick check, GO full evals, t2rl) before start_ve1.sh"
W=$(why_wait); last=""
while [ -n "$W" ]; do
  [ "$W" != "$last" ] && log "waiting for: $W" && last=$W
  sleep 60; W=$(why_wait)
done
log "prerequisites done -> start_ve1.sh (FORCE=1, CARDS=0,1,2,3)"
RUN_ID=$RUN_ID CARDS=0,1,2,3 FORCE=1 bash experiments/mist_w3po_v1/scripts/start_ve1.sh
