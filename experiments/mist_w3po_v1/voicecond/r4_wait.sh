#!/usr/bin/env bash
# Robust R4 waiter (reports/mist_w3po_v1/robust_r4_route_prereg.zh-CN.md, change 2), detached: wait until the R2 chain
# (cards 4,5; marker eval_outputs/mist_w3po_v1/r2/DONE or FAILED) or the R3 chain (cards 0,6; marker r3/CHAIN_DONE or
# CHAIN_FAILED) has finished, AND none of its r2_* / r3_* processes is alive, AND both cards hold no compute process except
# scripts/gpu_load/ placeholders (checked 3 times, 30 s apart) -> exec r4_orch.sh on that pair (first pair wins).
# Never cards 1, 2, 3, 7 (R1). Never touches guard 2726951, gpu_watch 52045, sync 1222453; stops nothing itself.
#   setsid nohup bash .../r4_wait.sh > logs/mist_w3po_v1/r4/wait.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/mist_w3po_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
alive() {   # $1 = pattern (r2_ | r3_): any live process whose argv runs voicecond/<pattern>*
  ps -eo args= | grep -E "voicecond/$1[a-z_]+\.(sh|py)" | grep -vq "grep" && return 0 || return 1
}
card_clear() {   # $1 = card: only gpu_load placeholders (or nothing) on it
  local p mb a
  while IFS=', ' read -r p mb; do
    [ -n "$p" ] || continue
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*|'') ;; *) return 1;; esac
  done < <(nvidia-smi -i "$1" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  return 0
}
ready() {   # $1 = group (r2|r3): marker present, no live process, both cards clear 3x
  local g=$1 cards
  if [ "$g" = r2 ]; then { [ -e $E/r2/DONE ] || [ -e $E/r2/FAILED ]; } || return 1; cards="4 5"
  else { [ -e $E/r3/CHAIN_DONE ] || [ -e $E/r3/CHAIN_FAILED ]; } || return 1; cards="0 6"; fi
  alive "${g}_" && return 1
  for k in 1 2 3; do
    for c in $cards; do card_clear $c || return 1; done
    [ $k = 3 ] || sleep 30
  done
  return 0
}
[ -e $E/r4/DONE ] || [ -e $E/r4/FAILED ] && { log "r4 already finished"; exit 1; }
log "r4 wait $$: for R2 (cards 4,5) or R3 (cards 0,6)"
while :; do
  if ready r2; then log "R2 finished ($(ls $E/r2 | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')); cards 4,5 clear -> r4_orch"
    CA=4 CB=5 exec bash experiments/mist_w3po_v1/voicecond/r4_orch.sh
  fi
  if ready r3; then log "R3 finished ($(ls $E/r3 | grep -E '^CHAIN_(DONE|FAILED)$' | tr '\n' ' ')); cards 0,6 clear -> r4_orch"
    CA=0 CB=6 exec bash experiments/mist_w3po_v1/voicecond/r4_orch.sh
  fi
  sleep 60
done
