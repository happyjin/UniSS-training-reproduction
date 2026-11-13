#!/usr/bin/env bash
# Night priority for (a)'s En->Zh chain (coordinator decision 2026-11-14 00:35 Prague, user-delegated): the method row's
# En->Zh speech (dev + test) needs (a)'s PL-EM talker, the longest chain (train16 decode -> sampling -> rows -> PL-EM
# training on 0-3 -> export proxies). (a)'s orchestrators already run on 0-3 only; the joint dev / test orchestrators
# are moved to 4-7: every 120 s, each joint dev/test run dir (logs/mist_w3po_v1/joint/{dev,test}/*/, not *.superseded*)
# without events/DONE gets exclude_cards "0 1 2 3" (the previous contents kept once as exclude_cards.before_prio_a).
# Running jobs are not touched; orchestrators read exclude_cards every tick. Stops (and restores nothing: the joint
# work is then free to use 0-3 again, exclude_cards emptied) once logs/mist_w3po_v1/en2zh/talker_selection.json exists
# or EN2ZH_STOP appears in after_select.log.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/prio_a_cards.sh > logs/mist_w3po_v1/joint/prio_a_cards.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=logs/mist_w3po_v1/joint; E=logs/mist_w3po_v1/en2zh
log() { echo "$(date -u +%FT%TZ) $*"; }
dirs() { for d in $J/dev/*/ $J/test/*/; do d=${d%/}; case $d in *superseded*) continue;; esac; [ -e $d/orch.log ] && [ ! -e $d/events/DONE ] && echo $d; done; }
setx() {   # <dir> <cards text>
  local f=$1/exclude_cards
  [ "$(cat $f 2>/dev/null)" = "$2" ] && return 0
  if [ -n "${DRY_RUN:-}" ]; then echo "would set $f: '$(cat $f 2>/dev/null)' -> '$2'"; return 0; fi
  [ -e $f.before_prio_a ] || cp $f $f.before_prio_a 2>/dev/null || : > $f.before_prio_a
  echo "$2" > $f.tmp && command mv -f $f.tmp $f && log "$f -> '$2'"
}
if [ -n "${DRY_RUN:-}" ]; then for d in $(dirs); do setx $d "0 1 2 3"; done; exit 0; fi
log "prio_a: joint dev/test orchestrators off cards 0-3 until (a)'s talker selection"
until [ -e $E/talker_selection.json ] || grep -q EN2ZH_STOP $E/after_select.log 2>/dev/null; do
  for d in $(dirs); do setx $d "0 1 2 3"; done
  sleep 120
done
for d in $(dirs); do setx $d ""; done
log "prio_a: done ((a) talker selected or (a) stopped); joint orchestrators may use all cards again"
