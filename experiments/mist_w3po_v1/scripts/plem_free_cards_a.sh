#!/usr/bin/env bash
# Unblock (a)'s En->Zh PL-EM start under STACK_PLEM (coordinator decision 2026-11-14 ~05:45 Prague, user-delegated).
# en2zh_after_select.sh step 5 claims slotA (cards 0-3), then with STACK_PLEM=1 waits until every card has
# used + 59 GiB <= 135 GiB (plem_blockers counts ALL memory, our no-checkpoint placeholders included); only after that
# wait does takeover_train.sh stop the placeholders. Placeholders of ~130 GB on 0-3 would therefore block it forever.
# This watcher waits for the line "STACK_PLEM: start when" in logs/mist_w3po_v1/en2zh/after_select.log (written right
# after the slotA claim) with slotA.owner present and naming the En->Zh PL-EM task, and the guard's training-slot
# exclusion (talker_crf_v1/exclude_cards, kept by sync_train_exclude.sh) covering 0 1 2 3 (so the guard does not
# refill them); then it runs takeover_train.sh with CARDS=0,1,2,3 and LAUNCH=":" -> only our placeholder supervisors /
# workers / keepers on 0-3 are stopped by PID (takeover's own cmdline checks); the PL-EM wait then passes by itself.
# Exits without action on EN2ZH_STOP or once PL-EM training is launched by other means.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/plem_free_cards_a.sh > logs/mist_w3po_v1/en2zh/plem_free_cards_a.log 2>&1 < /dev/null &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/plem_free_cards_a.sh   (prints the state and takeover's DRY_RUN)
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=logs/mist_w3po_v1/en2zh; A=$E/after_select.log
SL=logs/mist_w3po_v1/slots; X=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
log() { echo "$(date -u +%FT%TZ) $*"; }
owner_ok() { [ -e $SL/slotA.owner ] && grep -q "En->Zh PL-EM" $SL/slotA.owner; }
excl_ok() { local c; for c in 0 1 2 3; do tr ' ,' '\n\n' < $X 2>/dev/null | grep -qx $c || return 1; done; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "stack line: $(grep -c 'STACK_PLEM: start when' $A) ; owner_ok: $(owner_ok && echo y || echo n) ; excl_ok: $(excl_ok && echo y || echo n) ($(cat $X))"
  CARDS=0,1,2,3 LAUNCH=":" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  exit 0
fi
log "plem_free_cards_a: waiting for (a)'s STACK_PLEM wait on 0-3"
until grep -q "STACK_PLEM: start when" $A && owner_ok && excl_ok; do
  grep -q EN2ZH_STOP $A && { log "EN2ZH_STOP in $A: nothing done"; exit 1; }
  grep -q "PL-EM training .* launched" $A && { log "PL-EM already launched: nothing done"; exit 0; }
  sleep 20
done
log "slotA claimed by (a), exclusion '$(cat $X)'; stopping our placeholders on 0-3"
CARDS=0,1,2,3 LAUNCH=":" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
log "done; memory now: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
