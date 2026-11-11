#!/usr/bin/env bash
# Keeps the shared orchestrator exclusion file logs/mist_w3po_v1/talker_crf_v1/exclude_cards (talker_crf_orch.GLOBAL_EXCLUDE,
# read every tick by every YieldOrch-based eval orchestrator: thinker_t_eval_orch, talker_crf_orch) equal to the cards of
# the owned training slots (logs/mist_w3po_v1/slots/slotA.owner -> 0 1 2 3, slotB.owner -> 4 5 6 7). Then no eval
# orchestrator, whatever --exclude it was started with, places a NEW job on a training card (coordinator 2026-11-11
# ~23:00 Prague: nothing stacks on a training card). Running jobs are never touched (an excluded card only gets no new
# jobs; the orchestrators' headroom stop looks at real memory, not at the exclusion). Polls every 10 s; a change is
# written as a new file + command mv -f and logged. Stops when logs/mist_w3po_v1/slots/sync_train_exclude.STOP exists.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/sync_train_exclude.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SL=logs/mist_w3po_v1/slots
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
log() { echo "$(date -u +%FT%TZ) $*"; }
want() {
  local x=""
  [ -e $SL/slotA.owner ] && x="0 1 2 3"
  [ -e $SL/slotB.owner ] && x="$x 4 5 6 7"
  echo "${x# }"
}
if [ -n "${DRY_RUN:-}" ]; then echo "now: '$(cat $G 2>/dev/null)'; would be: '$(want)'"; exit 0; fi
log "start: $G now '$(cat $G 2>/dev/null)'"
while [ ! -e $SL/sync_train_exclude.STOP ]; do
  W=$(want)
  if [ "$W" != "$(cat $G 2>/dev/null)" ]; then
    T=$G.new_$(date -u +%Y%m%dT%H%M%SZ)_$$
    ( set -o noclobber; echo "$W" > $T ) && command mv -f $T $G && log "$G -> '$W'"
  fi
  sleep 10
done
log "STOP file found"
