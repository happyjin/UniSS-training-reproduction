#!/usr/bin/env bash
# Robust wave 5 change 4 follow-up: evaluate fishr_wise05 with the wave-5 chain (wave5_m_eval_orch.sh: merge -> LA2
# Thinker -> T1c Talker -> ASR) on the first card pair that is free of other waves' work, judged by PID only:
#   cards 0,1: wave 9's chain pid W9CHAIN (default 1865257) has exited AND no pid listed for card 0 or 1 in
#              logs/mist_w3po_v1/wave9/w9_workers.pids is alive;
#   cards 2,3: no pid listed for card 2 or 3 in logs/mist_w3po_v1/wave10/w10_workers.pids is alive AND, for 15 minutes
#              in a row, every compute process on cards 2 and 3 (nvidia-smi) is a no-checkpoint placeholder
#              (argv scripts/gpu_load/) -- wave 10 has no chain pid to wait for.
# Cards 4-7 (wave 8 AUL) are never used. Polls every 60 s; placeholders on the chosen cards are stopped by the eval
# launcher (PID, argv re-checked) and kept off by wave5_m_noplaceholder.sh. Checked with bash -n only.
#   setsid nohup bash wave5_m_wise_wait.sh > logs/mist_w3po_v1/wave5/m_wise_wait.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
W9CHAIN=${W9CHAIN:-1865257}
ARM=${ARM:-fishr_wise05}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
alive_on() {   # pids file, card list: any listed pid alive whose card field mentions one of the cards
  local f=$1; shift
  [ -s "$f" ] || return 1
  while read -r c p _; do
    [ -n "${p:-}" ] || continue
    [ -e "/proc/$p" ] || continue
    for g in "$@"; do case ",$c," in *",$g,"*) return 0;; esac; done
  done < "$f"
  return 1
}
only_placeholders() {   # cards...: every compute process on them is a placeholder
  local map uuid pid idx a
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ')
  while IFS=, read -r uuid pid; do
    uuid=$(echo "$uuid" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
    idx=$(echo "$map" | awk -F, -v u="$uuid" '$2==u{print $1}')
    for g in "$@"; do
      if [ "$idx" = "$g" ]; then
        a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
        case "$a" in *scripts/gpu_load/*) ;; *) return 1;; esac
      fi
    done
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  return 0
}
log "waiter $$ for $ARM: cards 0,1 after wave-9 chain $W9CHAIN; or cards 2,3 once wave 10 is gone for 15 min"
free23=0
while :; do
  if ! [ -e "/proc/$W9CHAIN" ] && ! alive_on logs/mist_w3po_v1/wave9/w9_workers.pids 0 1; then CA=0; CB=1; break; fi
  if ! alive_on logs/mist_w3po_v1/wave10/w10_workers.pids 2 3 && only_placeholders 2 3; then
    free23=$((free23 + 1)); [ "$free23" -ge 15 ] && { CA=2; CB=3; break; }
  else
    free23=0
  fi
  sleep 60
done
log "cards $CA,$CB free; starting the wave-5 evaluation chain for $ARM"
( : ) & dead=$!; wait $dead
ARM=$ARM CA=$CA CB=$CB TPID=$dead exec bash experiments/mist_w3po_v1/voicecond/wave5_m_eval_orch.sh
