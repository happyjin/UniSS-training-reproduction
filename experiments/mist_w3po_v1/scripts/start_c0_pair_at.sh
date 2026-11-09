#!/usr/bin/env bash
# Start C0 (scripts/run_crf_c0.sh) on cards 0,1,2,3 and, with FP=1, C0-fp (scripts/run_crf_c0_fp.sh: + final-read full P)
# on cards 4,5,6,7 at a given UTC time (LOG 2026-11-10 02:25 UTC plan). Checked with bash -n only; never test-run.
#   AT=2026-11-10T02:25:00Z FP=1 setsid nohup bash experiments/mist_w3po_v1/scripts/start_c0_pair_at.sh \
#     > logs/mist_w3po_v1/start_c0_pair_at_<UTC>.out 2>&1 < /dev/null &
# Sleeps until AT. Then every RETRY_S (60 s), for each card set not yet launched: take flock(gpu_launch.lock), read
# nvidia-smi, check every target card stays <= CAP_GIB (135) with NEED_GIB (70) more; if a non-root (colleague) or
# unidentified process is on a target card, add EXTRA="--mem-cap-gib 88"; launch (the run script detaches itself
# with setsid nohup and refuses existing outputs); wait until the run's own processes hold >= VISIBLE_GIB on every
# target card (or it exits, or VISIBLE_MAX_S); release the lock. A set still not launched RETRY_MAX_S (30 min) after AT
# is given up (logged). Records in a NEW dir logs/mist_w3po_v1/c0_pair_at_<UTC>/ (refuses an existing one):
#   scheduler.pid, <name>.launch.txt (run script output), <name>.pid (the run's root: setsid bash -c; torchrun is its
#   child, the workers torchrun's children in their own sessions), <name>.pids (the whole tree once visible: stop a
#   run by these PIDs), <name>.run_id; the training logs are logs/mist_w3po_v1/<RUN_ID>.log.
# The workload supervisors started 02:30 with --wait-idle only take cards with no other process, so they wait while
# the trainings run (intended).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
AT="${AT:?set AT, e.g. 2026-11-10T02:25:00Z}"
FP="${FP:?set FP=1 (C0 + C0-fp) or FP=0 (C0 only)}"
case "$FP" in 0|1) ;; *) echo "FP must be 0 or 1, got $FP" >&2; exit 1 ;; esac
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CAP_GIB=${CAP_GIB:-135}
NEED_GIB=${NEED_GIB:-70}
VISIBLE_GIB=${VISIBLE_GIB:-50}
VISIBLE_MAX_S=${VISIBLE_MAX_S:-900}
RETRY_S=${RETRY_S:-60}
RETRY_MAX_S=${RETRY_MAX_S:-1800}
LOCK_WAIT_S=${LOCK_WAIT_S:-600}
COLLEAGUE_EXTRA="--mem-cap-gib 88"
TS=$(date -u +%Y%m%dT%H%M%SZ)
D=logs/mist_w3po_v1/c0_pair_at_$TS
mkdir -p logs/mist_w3po_v1
if [ -e "$D" ]; then echo "拒绝覆盖已有目录: $D" >&2; exit 1; fi
mkdir "$D" || { echo "cannot create $D" >&2; exit 1; }
echo "$$" > "$D/scheduler.pid"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/scheduler.log"; }

NAMES=(c0)
declare -A SET=([c0]="0,1,2,3" [c0fp]="4,5,6,7")
declare -A SCRIPT=([c0]="experiments/mist_w3po_v1/scripts/run_crf_c0.sh" [c0fp]="experiments/mist_w3po_v1/scripts/run_crf_c0_fp.sh")
declare -A PREFIX=([c0]="crf_c0" [c0fp]="crf_c0fp")
declare -A DONE=()
[ "$FP" = 1 ] && NAMES+=(c0fp)
for n in "${NAMES[@]}"; do [ -x "${SCRIPT[$n]}" ] || [ -f "${SCRIPT[$n]}" ] || { log "missing ${SCRIPT[$n]}"; exit 1; }; done

now=$(date -u +%s); t_at=$(date -u -d "$AT" +%s) || { log "bad AT=$AT"; exit 1; }
log "scheduler $$: AT=$AT (in $((t_at - now)) s), FP=$FP, sets: $(for n in "${NAMES[@]}"; do printf '%s[%s] ' "$n" "${SET[$n]}"; done)"
log "rule: per card used + ${NEED_GIB} GiB <= ${CAP_GIB} GiB; retry every ${RETRY_S} s for ${RETRY_MAX_S} s; dir $D"
if [ "$t_at" -gt "$now" ]; then sleep $((t_at - now)); fi

# card index -> used MiB
used_mib() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v c="$1" '$1 == c {print $2}'; }
# PIDs of compute processes on card $1
card_pids() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v c="$1" '$1 == c {print $2}')
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$uuid" '$2 == u {print $1}'
}
# 0 if every card of $1 (comma list) stays <= CAP with NEED more; prints a summary
room_ok() {
  local ok=0 c u
  for c in ${1//,/ }; do
    u=$(used_mib "$c")
    if [ -z "$u" ]; then echo -n "card$c=?; "; ok=1; continue; fi
    echo -n "card$c=$((u / 1024))GiB; "
    if [ $((u + NEED_GIB * 1024)) -gt $((CAP_GIB * 1024)) ]; then ok=1; fi
  done
  return $ok
}
# 0 if a non-root or unidentified process is on any card of $1
foreign_on() {
  local c p usr
  for c in ${1//,/ }; do
    for p in $(card_pids "$c"); do
      usr=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      if [ "$usr" != "root" ]; then return 0; fi
    done
  done
  return 1
}
# $1 = root pid: it and all its descendants (torchrun starts its workers with start_new_session=True, so the run's
# processes are found by parentage, not by session)
tree_pids() {
  ps -eo pid=,ppid= | awk -v r="$1" '{par[$1] = $2} END {if (!(r in par)) exit; keep[r] = 1; ch = 1
    while (ch) {ch = 0; for (p in par) if (!(p in keep) && (par[p] in keep)) {keep[p] = 1; ch = 1}}
    for (p in keep) printf "%s ", p}'
}
# GiB held on card $2 by the process tree of $1
tree_gib_on() {
  local root=$1 c=$2 uuid pids
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v c="$c" '$1 == c {print $2}')
  pids=$(tree_pids "$root")
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits | tr -d ' ' \
    | awk -F, -v u="$uuid" -v p="$pids" 'BEGIN {n = split(p, a, " "); for (i = 1; i <= n; i++) s[a[i]] = 1}
        $2 == u && ($1 in s) {m += $3} END {print int(m / 1024)}'
}
alive() { local st; st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$st" ] && [ "${st:0:1}" != Z ]; }

try_launch() {   # $1 = name; 0 = launched (or failed for good), 1 = retry later
  local n=$1 cards=${SET[$1]} summ extra="" run_id out root t0 c g vis state=launched
  exec 9>>"$LOCK"
  if ! flock -w "$LOCK_WAIT_S" 9; then log "$n: gpu_launch.lock busy for ${LOCK_WAIT_S} s"; exec 9>&-; return 1; fi
  if ! summ=$(room_ok "$cards"); then log "$n: no room on [$cards]: $summ"; flock -u 9; exec 9>&-; return 1; fi
  if foreign_on "$cards"; then extra="$COLLEAGUE_EXTRA"; fi
  run_id=${PREFIX[$n]}_$(date -u +%Y%m%dT%H%M%SZ)
  echo "$run_id" > "$D/$n.run_id"
  log "$n: room on [$cards]: $summ launching RUN_ID=$run_id EXTRA='$extra'"
  CARDS=$cards RUN_ID=$run_id EXTRA="$extra" bash "${SCRIPT[$n]}" > "$D/$n.launch.txt" 2>&1 9>&-
  out=$(cat "$D/$n.launch.txt"); log "$n: $out"
  root=$(sed -n 's/.* pid \([0-9][0-9]*\) .*/\1/p' "$D/$n.launch.txt" | head -1)
  if [ -z "$root" ]; then
    log "$n: LAUNCH_FAILED (no pid in the run script output); not retried"
    flock -u 9; exec 9>&-; DONE[$n]=failed; return 0
  fi
  echo "$root" > "$D/$n.pid"
  t0=$(date -u +%s)
  while :; do
    vis=1
    for c in ${cards//,/ }; do
      g=$(tree_gib_on "$root" "$c")
      [ "${g:-0}" -ge "$VISIBLE_GIB" ] || vis=0
    done
    if [ $vis = 1 ]; then log "$n: memory visible on [$cards] after $(( $(date -u +%s) - t0 )) s"; break; fi
    if ! alive "$root"; then
      log "$n: EXITED_EARLY: pid $root ended before its memory showed; see logs/mist_w3po_v1/$run_id.log"
      state=exited_early; break
    fi
    if [ $(( $(date -u +%s) - t0 )) -ge "$VISIBLE_MAX_S" ]; then log "$n: memory not visible after ${VISIBLE_MAX_S} s; releasing the lock anyway"; break; fi
    sleep 10
  done
  for p in $(tree_pids "$root"); do ps -o pid=,ppid=,args= -p "$p"; done > "$D/$n.pids" 2>/dev/null
  flock -u 9; exec 9>&-
  DONE[$n]=$state
  log "$n: $state, root pid $root ($(wc -l < "$D/$n.pids") processes), log logs/mist_w3po_v1/$run_id.log"
  return 0
}

while :; do
  pending=0
  for n in "${NAMES[@]}"; do
    [ -n "${DONE[$n]:-}" ] && continue
    try_launch "$n" || pending=1
  done
  [ $pending = 0 ] && break
  if [ $(( $(date -u +%s) - t_at )) -ge "$RETRY_MAX_S" ]; then
    for n in "${NAMES[@]}"; do
      [ -z "${DONE[$n]:-}" ] && { log "$n: GAVE_UP: no room on [${SET[$n]}] for ${RETRY_MAX_S} s after $AT"; DONE[$n]=gave_up; }
    done
    break
  fi
  sleep "$RETRY_S"
done
log "scheduler done: $(for n in "${NAMES[@]}"; do printf '%s=%s ' "$n" "${DONE[$n]:-none}"; done)"
