#!/bin/bash
# Guard v3 (user 2026-11-12 ~07:30 Prague: "改守护脚本补上，防止空卡没有不保存checkpoint的训练占卡程序占领").
# v2 behaviour is kept: a card with no compute process of ours and no placeholder supervisor gets the
# no-checkpoint training placeholder (scripts/gpu_load/run_mist_workload.sh, same flags as v2) at once.
# New: a card that holds ONLY small, long-running processes of ours (e.g. idle CTC scorers left at the tail of an
# eval; 2026-11-12 01:22 incident) is also filled, with --tolerate-small-mib SMALL_MIB, when for GRACE consecutive
# polls all of these hold:
#   * no placeholder supervisor with a running worker for the card (a stand-by supervisor without a worker, which
#     with v2 flags would wait forever behind the small process, is stopped by PID and replaced);
#   * every non-Kubernetes compute process on it uses <= SMALL_MIB MiB and has run >= MIN_AGE s;
#   * none of them is a training (train_crf_sft / torch.distributed / train_mist_workload / megatron);
#   * the card is not a training-slot card (TRAIN_EXCL, kept in sync by sync_train_exclude.sh).
# The placeholder tolerates only the small processes present at its start; any new process (or a tolerated one
# that grows above SMALL_MIB) makes it yield, and our eval orchestrators / takeover launchers stop it by PID first.
# OLD_GUARD_PID: wait (<= 120 s) until that guard has exited before the first pass (no two guards at once).
# DRY_RUN=1: one read-only pass that prints each card's state and what would be done; starts nothing.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-30}
CARDS=${CARDS:-"0 1 2 3 4 5 6 7"}
SMALL_MIB=${SMALL_MIB:-8192}
MIN_AGE=${MIN_AGE:-300}
GRACE=${GRACE:-3}
TRAIN_EXCL=${TRAIN_EXCL:-logs/mist_w3po_v1/talker_crf_v1/exclude_cards}
DRY_RUN=${DRY_RUN:-0}
FLAGS="--wait-idle --ignore-k8s-pods --tolerate-keeper --smi-retries 6 --idle-seconds 5 --poll-seconds 10 --idle-memory-mib 100000"

if [ "$DRY_RUN" = 1 ]; then
  log() { echo "$(date -u +%FT%TZ) [dry] $*"; }
else
  LOG="logs/mist_w3po_v1/keep_cards_busy_v3_$(date -u +%Y%m%dT%H%M%SZ)"
  if [ -e "$LOG" ]; then echo "refusing: $LOG exists"; exit 1; fi
  mkdir -p "$LOG"; echo $$ > "$LOG/guard.pid"
  log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG/guard.log"; }
fi
log "guard v3 $$ cards [$CARDS] poll ${POLL}s small<=${SMALL_MIB}MiB age>=${MIN_AGE}s grace ${GRACE} excl $TRAIN_EXCL"

if [ "$DRY_RUN" != 1 ] && [ -n "${OLD_GUARD_PID:-}" ]; then
  for _ in $(seq 1 120); do kill -0 "$OLD_GUARD_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$OLD_GUARD_PID" 2>/dev/null; then log "old guard $OLD_GUARD_PID still alive after 120 s; continuing"
  else log "old guard $OLD_GUARD_PID has exited"; fi
fi

supervisor_pids() {   # PIDs of train_mist_workload.py supervisors (not --worker) for card $1
  ps -eo pid=,args= | grep "train_mist_workload.py" | grep -v -- "--worker" | grep -v grep \
    | awk -v k="$1" '{for(i=2;i<NF;i++) if($i=="--devices" && $(i+1)==k) {print $1; break}}'
}

has_worker() {   # supervisor $1 has a running --worker child
  ps --ppid "$1" -o args= 2>/dev/null | grep -q -- "--worker"
}

stop_standby() {   # $1 supervisor PID, $2 card: stop a stand-by supervisor (no worker) by PID after a cmdline check
  local s=$1 k=$2 cmd
  cmd=$(tr '\0' ' ' < "/proc/$s/cmdline" 2>/dev/null)
  case "$cmd" in *train_mist_workload.py*"--devices $k "*) ;; *) log "card $k: $s is not a stand-by supervisor for this card; not stopped"; return 1 ;; esac
  case "$cmd" in *--worker*) return 1 ;; esac
  has_worker "$s" && { log "card $k: supervisor $s has a worker now; not stopped"; return 1; }
  if [ "$DRY_RUN" = 1 ]; then log "card $k: would stop stand-by supervisor $s"; return 0; fi
  kill -TERM "$s" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$s" 2>/dev/null || break; sleep 1; done
  kill -0 "$s" 2>/dev/null && { log "card $k: stand-by supervisor $s did not exit"; return 1; }
  log "card $k: stopped stand-by supervisor $s (no worker; it would wait behind small processes)"
  return 0
}

in_train_excl() {
  [ -f "$TRAIN_EXCL" ] && tr ' ,' '\n\n' < "$TRAIN_EXCL" | grep -qx "$1"
}

card_state() {   # prints: empty | small | busy | unknown   (Kubernetes pod processes are ignored, as in v2)
  local k=$1 n=0 big=0 rows p mib age cmd
  if ! rows=$(timeout 30 nvidia-smi -i "$k" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null); then
    echo unknown; return
  fi
  while IFS=, read -r p mib; do
    p=${p// /}; mib=${mib// /}
    [ -z "$p" ] && continue
    grep -q kubepods "/proc/$p/cgroup" 2>/dev/null && continue
    n=$((n + 1))
    case "$mib" in ''|*[!0-9]*) big=1; continue ;; esac
    if [ "$mib" -gt "$SMALL_MIB" ]; then big=1; continue; fi
    age=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -z "$age" ] || [ "$age" -lt "$MIN_AGE" ]; then big=1; continue; fi
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *train_crf_sft*|*torch.distributed*|*train_mist_workload*|*megatron*) big=1 ;; esac
  done <<< "$rows"
  if [ "$n" -eq 0 ]; then echo empty; elif [ "$big" -eq 0 ]; then echo small; else echo busy; fi
}

start_placeholder() {   # $1 card, $2 extra flags, $3 reason
  local k=$1 extra=$2 why=$3
  if [ "$DRY_RUN" = 1 ]; then log "card $k: would start placeholder ($why) flags: $FLAGS $extra"; return; fi
  setsid -f bash -c "echo \$\$ > $LOG/card$k.pid; exec bash scripts/gpu_load/run_mist_workload.sh --devices $k $FLAGS $extra" \
    >> "$LOG/card$k.log" 2>&1 < /dev/null
  sleep 2
  log "card $k: $why: started supervisor $(cat "$LOG/card$k.pid" 2>/dev/null)"
}

declare -A SMALLCNT
while true; do
  for k in $CARDS; do
    sups=$(supervisor_pids "$k")
    if [ -n "$sups" ]; then
      active=0
      for s in $sups; do has_worker "$s" && active=1; done
      if [ "$active" = 1 ]; then
        SMALLCNT[$k]=0; [ "$DRY_RUN" = 1 ] && log "card $k: placeholder running"
        continue
      fi
      # only stand-by supervisors: replace them if the card holds nothing but small long-running processes
      st=$(card_state "$k")
      if [ "$st" = small ] && ! in_train_excl "$k"; then
        SMALLCNT[$k]=$(( ${SMALLCNT[$k]:-0} + 1 ))
        if [ "$DRY_RUN" = 1 ] || [ "${SMALLCNT[$k]}" -ge "$GRACE" ]; then
          ok=1
          for s in $sups; do stop_standby "$s" "$k" || ok=0; done
          [ "$ok" = 1 ] && start_placeholder "$k" "--tolerate-small-mib $SMALL_MIB" \
            "only small long-running processes for ${SMALLCNT[$k]} poll(s); stand-by supervisor replaced"
          SMALLCNT[$k]=0
        fi
      else
        SMALLCNT[$k]=0; [ "$DRY_RUN" = 1 ] && log "card $k: stand-by supervisor waiting; card $st"
      fi
      continue
    fi
    st=$(card_state "$k")
    case "$st" in
      empty)
        SMALLCNT[$k]=0
        start_placeholder "$k" "" "free of our processes and without a placeholder" ;;
      small)
        if in_train_excl "$k"; then
          SMALLCNT[$k]=0; [ "$DRY_RUN" = 1 ] && log "card $k: only small processes, but a training-slot card: skip"
          continue
        fi
        SMALLCNT[$k]=$(( ${SMALLCNT[$k]:-0} + 1 ))
        if [ "$DRY_RUN" = 1 ] || [ "${SMALLCNT[$k]}" -ge "$GRACE" ]; then
          start_placeholder "$k" "--tolerate-small-mib $SMALL_MIB" "only small long-running processes for ${SMALLCNT[$k]} poll(s)"
          SMALLCNT[$k]=0
        fi ;;
      *)
        SMALLCNT[$k]=0; [ "$DRY_RUN" = 1 ] && log "card $k: $st" ;;
    esac
  done
  [ "$DRY_RUN" = 1 ] && exit 0
  sleep "$POLL"
done
