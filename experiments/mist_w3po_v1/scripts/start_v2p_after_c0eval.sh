#!/usr/bin/env bash
# Arm SerialCRF v2' (scripts/run_crf_v2p.sh) on all 8 cards right after the C0 / C0-fp dev evaluation
# (reports/mist_w3po_v1/serial_v2p_train.zh-CN.md §4; user decision LOG 2026-11-10 04:55 UTC). Modelled on
# start_c0_pair_at.sh. Checked with bash -n and --dry only; never test-run.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/start_v2p_after_c0eval.sh \
#     > logs/mist_w3po_v1/start_v2p_after_c0eval_<UTC>.out 2>&1 < /dev/null &
#   bash experiments/mist_w3po_v1/scripts/start_v2p_after_c0eval.sh --dry     # what it sees now; no action, no dir
# Events (one process; polls every POLL_S = 60 s; gives up after WAIT_MAX_S = 18 h without launching):
#  1. the evaluation's orchestrator = a python process running $ORCH_NAME (crf_pair_dev_orch.py) --run <dir> (not --dry /
#     --smoke), PID and run dir from ps fields (or ORCH_PID / RUN_DIR given): wait until it is seen (or RUN_DIR's
#     events/DONE exists), then until that PID exits;
#  2. finished = <run>/events/DONE exists (the report reports/mist_w3po_v1/c0_pair_dev_eval.zh-CN.md is only logged: it
#     may be written before the evaluation ends); otherwise wait GRACE_S (20 min) for a restarted orchestrator
#     (followed again from 1), then proceed;
#  3. no evaluation GPU process = every compute process on the cards belongs to a workload supervisor tree (card<k>.pid
#     of every logs/mist_w3po_v1/workload_at_*/ and their descendants); anything else -> wait.
# Then:
#  4. stop the workload supervisors by PID: SIGTERM (each stops its worker process groups), then the leftover PIDs of
#     their trees (listed from ps fields before the signal) SIGTERM / SIGKILL, then GPU processes whose own args name
#     scripts/gpu_load/train_mist_workload.py (orphans), by PID; nvidia-smi must show the cards free;
#  5. flock gpu_launch.lock; every card's used + NEED_MIB (126,976 = cap 120 GiB + 4 GiB context) <= 138,240 MiB,
#     else release, wait, retry;
#  6. smoke A = the v2' objective with C0's packing knobs (TOK 12000, CAP 88, FAST 0, PREFETCH 0, EMPTY 1; 3 steps) and
#     smoke B = the run's knobs (TOK 32000, CAP 120, FAST 1, PREFETCH 1, EMPTY 0; 5 steps): run_crf_v2p.sh, new RUN_IDs,
#     nvidia-smi sampled every 1 s into <dir>/smoke{A,B}.smi.csv; gate: both logs end with CRF_SFT_TRAIN_DONE and
#     CRF_RUN_EXIT rc=0, every loss / sft_loss finite, B's mem_gib_max <= 120 and every sampled card <= 138,240 MiB;
#  7. the formal run (3000 steps, run_crf_v2p.sh defaults) on 0..7; wait until its tree holds >= 50 GiB on every card;
#     release the lock;
#  8. watch its root PID; when it exits (finished or crashed) restart the workload on the cards that have no live
#     supervisor (start_workload_at.sh, AT = now). A failed gate or launch also restarts the workload.
# Records in a NEW dir logs/mist_w3po_v1/v2p_at_<UTC>/ (refuses an existing one): scheduler.pid, scheduler.log,
#   orch.txt, workload_stopped.txt, smoke{A,B}.{run_id,launch.txt,smi.csv}, formal.{run_id,launch.txt,pid,pids}.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CARDS=0,1,2,3,4,5,6,7
CAP_MIB=138240
NEED_MIB=${NEED_MIB:-126976}
VISIBLE_GIB=${VISIBLE_GIB:-50}
VISIBLE_MAX_S=${VISIBLE_MAX_S:-1200}
POLL_S=${POLL_S:-60}
WAIT_MAX_S=${WAIT_MAX_S:-64800}
GRACE_S=${GRACE_S:-1200}
SMOKE_MAX_S=${SMOKE_MAX_S:-1500}
LOCK_WAIT_S=${LOCK_WAIT_S:-900}
WATCH_MAX_S=${WATCH_MAX_S:-108000}
REPORT=reports/mist_w3po_v1/c0_pair_dev_eval.zh-CN.md
ORCH_NAME=${ORCH_NAME:-crf_pair_dev_orch.py}
RUN=scripts/run_crf_v2p.sh
RUN=experiments/mist_w3po_v1/$RUN
T0=$(date -u +%s)
TS=$(date -u +%Y%m%dT%H%M%SZ)
D=logs/mist_w3po_v1/v2p_at_$TS
if [ $DRY = 0 ]; then
  mkdir -p logs/mist_w3po_v1
  if [ -e "$D" ]; then echo "拒绝覆盖已有目录: $D" >&2; exit 1; fi
  mkdir "$D" || { echo "cannot create $D" >&2; exit 1; }
  echo "$$" > "$D/scheduler.pid"
  log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/scheduler.log"; }
else
  log() { echo "$(date -u +%FT%TZ) [dry] $*"; }
fi
[ -f "$RUN" ] || { log "missing $RUN"; exit 1; }

alive() { local st; st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$st" ] && [ "${st:0:1}" != Z ]; }
# $1 = root pid: it and all its descendants (by parentage)
tree_pids() {
  ps -eo pid=,ppid= | awk -v r="$1" '{par[$1] = $2} END {if (!(r in par)) exit; keep[r] = 1; ch = 1
    while (ch) {ch = 0; for (p in par) if (!(p in keep) && (par[p] in keep)) {keep[p] = 1; ch = 1}}
    for (p in keep) printf "%s ", p}'
}
# "pid rundir" of every orchestrator process: field 2 (the executable) is a python, an argument ends with
# crf_pair_dev_orch.py, and --run is given without --dry / --smoke (ps fields; no pattern matching on whole lines)
orch_procs() {
  ps -eo pid=,args= | awk -v nm="$ORCH_NAME" '{
      if ($2 !~ /python[0-9.]*$/) next
      s = 0; run = ""; bad = 0
      for (i = 3; i <= NF; i++) {
        n = split($i, seg, "/"); if (seg[n] == nm) s = 1
        if ($i == "--run" && i < NF) run = $(i + 1)
        if ($i == "--dry" || $i == "--smoke") bad = 1
      }
      if (s && run != "" && !bad) print $1, run
    }'
}
# workload supervisor PIDs alive (all workload_at_* dirs), "card pid" per line
supervisors() {
  local f p k
  for f in logs/mist_w3po_v1/workload_at_*/card*.pid; do
    [ -f "$f" ] || continue
    p=$(tr -dc '0-9' < "$f"); k=$(basename "$f" .pid); k=${k#card}
    [ -n "$p" ] && alive "$p" && echo "$k $p"
  done
}
workload_tree() { local k p; supervisors | while read -r k p; do tree_pids "$p"; done; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | grep -E '^[0-9]+$'; }
# GPU PIDs not in the workload trees (and not in $1 = space-separated allowed PIDs)
foreign_gpu() {
  local wl allow="$1" p
  wl=" $(workload_tree | tr '\n' ' ') $allow "
  for p in $(gpu_pids); do case "$wl" in *" $p "*) ;; *) echo -n "$p ";; esac; done
}
used_all() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' '; }
room_ok() {   # every card: used + NEED_MIB <= CAP_MIB
  local ok=0 line c u
  for line in $(used_all); do
    c=${line%,*}; u=${line#*,}
    if [ $((u + NEED_MIB)) -gt $CAP_MIB ]; then ok=1; fi
  done
  return $ok
}
tree_gib_on() {   # GiB held on card $2 by the process tree of $1
  local root=$1 c=$2 uuid pids
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v c="$c" '$1 == c {print $2}')
  pids=$(tree_pids "$root")
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits | tr -d ' ' \
    | awk -F, -v u="$uuid" -v p="$pids" 'BEGIN {n = split(p, a, " "); for (i = 1; i <= n; i++) s[a[i]] = 1}
        $2 == u && ($1 in s) {m += $3} END {print int(m / 1024)}'
}
restart_workload() {   # cards without a live supervisor get one (start_workload_at.sh, AT = now)
  local have miss="" c
  have=" $(supervisors | awk '{print $1}' | tr '\n' ' ') "
  for c in 0 1 2 3 4 5 6 7; do case "$have" in *" $c "*) ;; *) miss="$miss $c";; esac; done
  miss=${miss# }
  if [ -z "$miss" ]; then log "workload: every card already has a live supervisor; not restarted"; return; fi
  local o=logs/mist_w3po_v1/start_workload_at_$(date -u +%Y%m%dT%H%M%SZ).out
  AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) CARDS="$miss" setsid nohup bash experiments/mist_w3po_v1/scripts/start_workload_at.sh \
    > "$o" 2>&1 < /dev/null 9>&- &
  log "workload restarted on cards [$miss] (start_workload_at.sh pid $!, $o)"
}
timeout_hit() { [ $(( $(date -u +%s) - T0 )) -ge "$WAIT_MAX_S" ]; }

if [ $DRY = 1 ]; then
  log "orchestrators: $(orch_procs | tr '\n' ';') report: $([ -f $REPORT ] && echo present || echo absent)"
  log "workload supervisors (card pid): $(supervisors | tr '\n' ';')"
  log "GPU pids: $(gpu_pids | tr '\n' ' ') | not workload: $(foreign_gpu '')"
  log "cards used MiB: $(used_all) | room for $NEED_MIB MiB on all: $(room_ok && echo yes || echo no)"
  for v in A B F; do
    case $v in
      A) CARDS=$CARDS RUN_ID=crf_v2p_smokeA_DRY STOP_AFTER=3 TOK=12000 CAP=88 FAST=0 PREFETCH=0 EMPTY=1 PORT=29963 DRY_RUN=1 bash "$RUN";;
      B) CARDS=$CARDS RUN_ID=crf_v2p_smokeB_DRY STOP_AFTER=5 PORT=29965 DRY_RUN=1 bash "$RUN";;
      F) CARDS=$CARDS RUN_ID=crf_v2p_DRY PORT=29961 DRY_RUN=1 bash "$RUN";;
    esac
  done
  exit 0
fi

log "scheduler $$: wait for the C0 / C0-fp dev evaluation (crf_pair_dev_orch.py), then smoke A/B + v2' on [$CARDS]; dir $D"
# ---------------------------------------------------------------------------------------------------- 1-3 events
ORCH=${ORCH_PID:-}; RDIR=${RUN_DIR:-}
while :; do
  if [ -z "$ORCH" ]; then
    read -r ORCH RDIR2 < <(orch_procs | head -1) || true
    if [ -n "${ORCH:-}" ]; then RDIR=$RDIR2; log "orchestrator seen: pid $ORCH run $RDIR"; echo "$ORCH $RDIR" >> "$D/orch.txt"; fi
  fi
  if [ -n "${ORCH:-}" ] && alive "$ORCH"; then
    timeout_hit && { log "GAVE_UP: evaluation still running after ${WAIT_MAX_S} s"; exit 1; }
    sleep "$POLL_S"; continue
  fi
  if [ -z "${ORCH:-}" ] && ! { [ -n "$RDIR" ] && [ -f "$RDIR/events/DONE" ]; }; then   # not started yet
    timeout_hit && { log "GAVE_UP: no evaluation orchestrator seen in ${WAIT_MAX_S} s"; exit 1; }
    sleep "$POLL_S"; continue
  fi
  # the followed orchestrator has exited (or RUN_DIR's run is finished)
  if [ -n "$RDIR" ] && [ -f "$RDIR/events/DONE" ]; then
    log "evaluation finished: pid ${ORCH:-?} exited; DONE=$([ -n "$RDIR" ] && [ -f "$RDIR/events/DONE" ] && echo yes || echo no) report=$([ -f $REPORT ] && echo yes || echo no)"
    break
  fi
  log "orchestrator ${ORCH:-?} exited without events/DONE or the report: waiting up to ${GRACE_S} s for a restart"
  t_ex=$(date -u +%s); NEW=""
  while [ $(( $(date -u +%s) - t_ex )) -lt "$GRACE_S" ]; do
    sleep "$POLL_S"
    read -r NEW NRD < <(orch_procs | head -1) || true
    [ -n "${NEW:-}" ] && break
    [ -n "$RDIR" ] && [ -f "$RDIR/events/DONE" ] && break
  done
  if [ -n "${NEW:-}" ]; then ORCH=$NEW; RDIR=$NRD; log "orchestrator restarted: pid $ORCH run $RDIR"; echo "$ORCH $RDIR" >> "$D/orch.txt"; continue; fi
  log "no restart in ${GRACE_S} s: proceeding"
  break
done
while :; do
  F=$(foreign_gpu "")
  [ -z "$F" ] && break
  log "GPU processes outside the workload remain: $F (waiting)"
  timeout_hit && { log "GAVE_UP: non-workload GPU processes still present"; exit 1; }
  sleep "$POLL_S"
done
log "no evaluation GPU process left; cards: $(used_all)"

# ---------------------------------------------------------------------------------------------------- 4 workload
stop_workload() {
  local sup trees p c=0
  sup=$(supervisors)
  trees=$(workload_tree | tr '\n' ' ')
  { echo "$(date -u +%FT%TZ) supervisors (card pid): $(echo "$sup" | tr '\n' ';')"; echo "trees: $trees"
    for p in $trees; do ps -o pid=,ppid=,args= -p "$p" 2>/dev/null | cut -c1-200; done; } >> "$D/workload_stopped.txt"
  for p in $(echo "$sup" | awk '{print $2}'); do kill -TERM "$p" 2>/dev/null; done
  while [ $c -lt 30 ]; do
    local left=""; for p in $trees; do alive "$p" && left="$left $p"; done
    [ -z "$left" ] && break
    sleep 2; c=$((c + 1))
  done
  for p in $trees; do alive "$p" && kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in $trees; do alive "$p" && kill -KILL "$p" 2>/dev/null; done
  sleep 5
  for p in $(gpu_pids); do   # orphaned workload workers (by PID; their own args checked)
    if ps -o args= -p "$p" 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i ~ /scripts\/gpu_load\/train_mist_workload\.py$/) f = 1} END {exit !f}'; then
      log "orphan workload GPU process $p: SIGKILL"; echo "orphan $p" >> "$D/workload_stopped.txt"; kill -KILL "$p" 2>/dev/null
    fi
  done
  sleep 5
}
stop_workload
log "workload stopped; supervisors alive now: $(supervisors | tr '\n' ';') GPU pids: $(gpu_pids | tr '\n' ' ') cards: $(used_all)"

# ---------------------------------------------------------------------------------------------------- 5-7 smoke + launch
run_wait() {   # $1 name, then env assignments for run_crf_v2p.sh; waits for the run's root to exit; 0 = done ok
  local n=$1; shift
  local rid=crf_v2p_smoke${n}_$(date -u +%Y%m%dT%H%M%SZ) root t0 smi lg
  echo "$rid" > "$D/smoke$n.run_id"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits -l 1 \
    > "$D/smoke$n.smi.csv" 2>/dev/null 9>&- &
  smi=$!
  env "$@" CARDS=$CARDS RUN_ID=$rid bash "$RUN" > "$D/smoke$n.launch.txt" 2>&1 9>&-
  log "smoke $n: $(cat "$D/smoke$n.launch.txt")"
  root=$(sed -n 's/.* pid \([0-9][0-9]*\) .*/\1/p' "$D/smoke$n.launch.txt" | head -1)
  if [ -z "$root" ]; then kill -TERM "$smi" 2>/dev/null; log "smoke $n: LAUNCH_FAILED"; return 1; fi
  t0=$(date -u +%s)
  while alive "$root"; do
    if [ $(( $(date -u +%s) - t0 )) -ge "$SMOKE_MAX_S" ]; then
      log "smoke $n: still running after ${SMOKE_MAX_S} s: stopping its tree by PID"
      for p in $(tree_pids "$root"); do kill -TERM "$p" 2>/dev/null; done
      sleep 20; for p in $(tree_pids "$root"); do kill -KILL "$p" 2>/dev/null; done
      break
    fi
    sleep 10
  done
  kill -TERM "$smi" 2>/dev/null
  lg=logs/mist_w3po_v1/$rid.log
  grep -q "CRF_SFT_TRAIN_DONE" "$lg" && grep -q "CRF_RUN_EXIT rc=0" "$lg" || { log "smoke $n: FAILED (see $lg)"; return 1; }
  return 0
}
gate() {   # $1 = smoke B run id; prints the summary; 0 = pass
  local rid=$1
  /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python - "$rid" "$D" "$CAP_MIB" <<'PY'
import json, math, sys
rid, d, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = True
out = {}
for n in ("A", "B"):
    r = open(f"{d}/smoke{n}.run_id").read().strip()
    m = [json.loads(l) for l in open(f"eval_outputs/mist_w3po_v1/crf/train/{r}/metrics.jsonl")]
    fin = all(math.isfinite(x["loss"]) and math.isfinite(x["sft_loss"]) and math.isfinite(x["gnorm"]) for x in m)
    mx = 0
    for l in open(f"{d}/smoke{n}.smi.csv"):
        f = [x.strip() for x in l.split(",")]
        if len(f) == 5 and f[4].isdigit():
            mx = max(mx, int(f[4]))
    out[n] = {"steps": len(m), "finite": fin, "card_max_mib": mx, "mem_gib_max": max(x["mem_gib_max"] for x in m),
              "loss": [round(x["loss"], 4) for x in m], "t_step": [x["t_step"] for x in m]}
    ok &= fin and mx <= cap and len(m) >= (3 if n == "A" else 5)
ok &= out["B"]["mem_gib_max"] <= 120
print(json.dumps(out))
sys.exit(0 if ok else 1)
PY
}
exec 9>>"$LOCK"
while :; do
  if ! flock -w "$LOCK_WAIT_S" 9; then
    log "gpu_launch.lock busy for ${LOCK_WAIT_S} s; retry"
    timeout_hit && { log "GAVE_UP: lock"; restart_workload; exit 1; }
    continue
  fi
  if room_ok; then break; fi
  log "no room for $NEED_MIB MiB on every card: $(used_all); retry in $POLL_S s"
  flock -u 9
  if [ -n "$(supervisors)" ]; then log "workload supervisors (re)started meanwhile: stopping them"; stop_workload; fi
  timeout_hit && { log "GAVE_UP: no room"; restart_workload; exit 1; }
  sleep "$POLL_S"
done
log "lock held; room on all cards: $(used_all)"
if ! run_wait A STOP_AFTER=3 TOK=12000 CAP=88 FAST=0 PREFETCH=0 EMPTY=1 PORT=29963; then
  flock -u 9; restart_workload; log "SMOKE_A_FAILED: formal run not launched"; exit 1
fi
if ! run_wait B STOP_AFTER=5 PORT=29965; then
  flock -u 9; restart_workload; log "SMOKE_B_FAILED: formal run not launched"; exit 1
fi
if ! G=$(gate "$(cat "$D/smokeB.run_id")"); then
  log "GATE_FAILED: $G"; flock -u 9; restart_workload; exit 1
fi
log "gate passed: $G"
if ! room_ok; then log "cards not free after the smokes: $(used_all)"; flock -u 9; restart_workload; exit 1; fi
RID=crf_v2p_$(date -u +%Y%m%dT%H%M%SZ)
echo "$RID" > "$D/formal.run_id"
CARDS=$CARDS RUN_ID=$RID PORT=29961 bash "$RUN" > "$D/formal.launch.txt" 2>&1 9>&-
log "formal: $(cat "$D/formal.launch.txt")"
ROOT=$(sed -n 's/.* pid \([0-9][0-9]*\) .*/\1/p' "$D/formal.launch.txt" | head -1)
if [ -z "$ROOT" ]; then log "FORMAL_LAUNCH_FAILED"; flock -u 9; restart_workload; exit 1; fi
echo "$ROOT" > "$D/formal.pid"
t0=$(date -u +%s)
while :; do
  vis=1
  for c in 0 1 2 3 4 5 6 7; do g=$(tree_gib_on "$ROOT" "$c"); [ "${g:-0}" -ge "$VISIBLE_GIB" ] || vis=0; done
  if [ $vis = 1 ]; then log "formal: memory visible on all cards after $(( $(date -u +%s) - t0 )) s"; break; fi
  if ! alive "$ROOT"; then log "formal: EXITED_EARLY (logs/mist_w3po_v1/$RID.log)"; break; fi
  if [ $(( $(date -u +%s) - t0 )) -ge "$VISIBLE_MAX_S" ]; then log "formal: memory not visible after ${VISIBLE_MAX_S} s"; break; fi
  sleep 10
done
for p in $(tree_pids "$ROOT"); do ps -o pid=,ppid=,args= -p "$p"; done > "$D/formal.pids" 2>/dev/null
flock -u 9; exec 9>&-
log "formal: root pid $ROOT ($(wc -l < "$D/formal.pids") processes), log logs/mist_w3po_v1/$RID.log; lock released"

# ---------------------------------------------------------------------------------------------------- 8 watcher
t0=$(date -u +%s)
while alive "$ROOT"; do
  if [ $(( $(date -u +%s) - t0 )) -ge "$WATCH_MAX_S" ]; then log "watcher: ${WATCH_MAX_S} s passed, run still alive; watcher ends"; exit 0; fi
  sleep "$POLL_S"
done
log "formal run exited: $(grep -c . eval_outputs/mist_w3po_v1/crf/train/$RID/metrics.jsonl 2>/dev/null) steps logged; $(grep -E 'CRF_SFT_TRAIN_DONE|CRF_RUN_EXIT' logs/mist_w3po_v1/$RID.log | tr '\n' ' ')"
sleep 30
restart_workload
log "scheduler done"
