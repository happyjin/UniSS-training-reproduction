#!/usr/bin/env bash
# Final test (coordinator 2026-11-12 ~20:30 Prague): once BOTH A' test decodes are over, the VE1 / D1v decode
# orchestrators may also use cards 0-3 (their exclude_cards "0 1 2 3" -> empty; new file + mv -f; the orchestrators
# re-read it every tick). Priority stays group 1 speech > VE1 / D1v decodes, so the files are emptied only when, in
# addition, every talker job of the group-1 chains of both sets has started (state/*__talk*.pid) or 25 min after the
# last group-1 chain of both sets was launched, whichever comes first. A run that has not been created is skipped.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/final_test_release_g2.sh
# Run detached: setsid nohup bash .../final_test_release_g2.sh > logs/mist_w3po_v1/final_test/release_g2.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=logs/mist_w3po_v1/final_test
log() { echo "$(date -u +%FT%TZ) $*"; }
G1="ami_aprlplem ami_applem ami_apc1b1s2 ali_aprlplem ali_applem ali_apc1b1s2"
G2="ali_ve1 ali_d1v ami_ve1 ami_d1v"
talks_started() { ls $1/state/ 2>/dev/null | grep -E '__talk[0-9]+\.a[0-9]+\.pid$' | sed -E 's/\.a[0-9]+\.pid$//' | sort -u | wc -l; }
if [ -n "${DRY_RUN:-}" ]; then
  for r in ami_aprime ali_aprime; do echo "$r DONE: $([ -e $E/$r/events/DONE ] && echo yes || echo no)"; done
  for r in $G1; do echo "$r: $([ -e $E/$r/orch.log ] && echo "launched, talkers started $(talks_started $E/$r)/16" || echo 'not launched')"; done
  for r in $G2; do echo "$r exclude_cards: $(cat $E/$r/exclude_cards 2>/dev/null || echo '(no run)')"; done
  exit 0
fi
until [ -e $E/ami_aprime/events/DONE ] && [ -e $E/ali_aprime/events/DONE ]; do sleep 30; done
log "both A' decode orchestrators DONE"
t0=$(date +%s)
while true; do
  n=0; launched=0; last=0
  for r in $G1; do
    if [ -e $E/$r/orch.log ]; then
      launched=$((launched + 1))
      n=$((n + $(talks_started $E/$r)))
      ts=$(head -n 1 $E/$r/orch.log | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z' | head -n 1)
      c=$(date -d "$ts" +%s 2>/dev/null || echo 0); [ "$c" -gt "$last" ] && last=$c
    fi
  done
  [ $launched = 6 ] && [ $n -ge 96 ] && { log "all 96 group-1 talker jobs started"; break; }
  [ $launched = 6 ] && [ $(( $(date +%s) - last )) -ge 1500 ] && { log "25 min since the last group-1 launch ($n / 96 talkers started)"; break; }
  [ $(( $(date +%s) - t0 )) -ge 5400 ] && { log "90 min after the A' decodes ($launched group-1 chains launched, $n talkers started)"; break; }
  sleep 30
done
for r in $G2; do
  [ -e $E/$r/exclude_cards ] || { log "$r: no run, skipped"; continue; }
  printf '\n' > $E/$r/exclude_cards.new && command mv -f $E/$r/exclude_cards.new $E/$r/exclude_cards \
    && log "$r exclude_cards emptied (was 0 1 2 3)"
done
log "FT_G2_RELEASED"
