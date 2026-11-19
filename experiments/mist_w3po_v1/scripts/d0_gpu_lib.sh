# D0 (OpenBroadcast-SI dev twin) shared helpers, sourced by d0_build.sh / d0_runs.sh, never run.
# gpu_one: run ONE small GPU command on the card with the most usable memory, counting a no-checkpoint placeholder
# (scripts/gpu_load/train_mist_workload.py) as free; the placeholder there is stopped by PID (its whole process tree,
# taken from ps fields) only if the card's real free memory is below the need. Never touches other processes, never
# uses pattern kills. The guard refills the card afterwards.
d0_log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
d0_pick_card() {   # <need MiB> -> "<card> <real free MiB>" of the card with the most (free + placeholder) memory >= need
  $PY - "$1" <<'EOF'
import subprocess, sys
need = int(sys.argv[1])
q = lambda a: subprocess.run(["nvidia-smi", a, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
free, idx = {}, {}
for l in q("--query-gpu=index,memory.free,pci.bus_id").strip().splitlines():
    i, f, b = [x.strip() for x in l.split(",")]
    free[int(i)] = int(f)
    idx[b.lower()] = int(i)
ph = {i: 0 for i in free}
for l in q("--query-compute-apps=pid,gpu_bus_id,used_memory").strip().splitlines():
    p = [x.strip() for x in l.split(",")]
    if len(p) != 3 or not p[0].isdigit():
        continue
    try:
        cmd = open(f"/proc/{p[0]}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if "train_mist_workload.py" in cmd and p[1].lower() in idx and p[2].isdigit():
        ph[idx[p[1].lower()]] += int(p[2])
best = max(free, key=lambda i: (free[i] + ph[i], free[i]))
if free[best] + ph[best] >= need:
    print(best, free[best])
EOF
}
d0_tree() {   # <pid>: the pid and all its descendants (ps fields)
  local p=$1 c
  echo $p
  for c in $(ps -eo pid=,ppid= | awk -v p=$p '$2 == p {print $1}'); do d0_tree $c; done
}
d0_stop_placeholder() {   # <card>: stop the placeholder supervisor(s) of that card and their trees, by PID
  local c=$1 sup pids="" s
  sup=$(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2 == 1 && index($0, "train_mist_workload.py") && index($0, c) {print $1}')
  [ -z "$sup" ] && return 0
  for s in $sup; do pids="$pids $(d0_tree $s)"; done
  pids=$(echo $pids | tr ' ' '\n' | sort -un | tr '\n' ' ')
  d0_log "card $c: stop placeholder $sup (tree $pids)"
  declare -A st0
  for s in $pids; do st0[$s]=$(cut -d' ' -f22 /proc/$s/stat 2>/dev/null); done   # start times: no PID reuse hits
  kill -TERM $pids 2>/dev/null
  for _ in $(seq 1 60); do [ -z "$(ps -o pid= -p $(echo $pids | tr ' ' ',') 2>/dev/null)" ] && break; sleep 1; done
  for s in $pids; do
    [ -n "${st0[$s]}" ] && [ "$(cut -d' ' -f22 /proc/$s/stat 2>/dev/null)" = "${st0[$s]}" ] && kill -KILL $s 2>/dev/null
  done
  sleep 3
}
d0_gpu_one() {   # <need MiB> <marker> <command...>: rc into <marker> ("rc=<n> card=<c>")
  local need=$1 mark=$2 x card fr
  shift 2
  [ -e "$mark" ] && { d0_log "marker $mark exists: not rerun"; return 0; }
  for _ in $(seq 1 120); do
    x=$(d0_pick_card $need)
    [ -n "$x" ] && break
    sleep 30
  done
  [ -z "$x" ] && { d0_log "no card with $need MiB usable"; echo "rc=nocard" > "$mark"; return 1; }
  read card fr <<< "$x"
  if [ "$fr" -lt "$need" ]; then
    d0_stop_placeholder $card
    for _ in $(seq 1 30); do
      fr=$(nvidia-smi -i $card --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
      [ "${fr:-0}" -ge "$need" ] && break
      sleep 2
    done
  fi
  [ "${fr:-0}" -ge "$need" ] || { d0_log "card $card: only $fr MiB free"; echo "rc=nomem card=$card" > "$mark"; return 1; }
  d0_log "card $card: $fr MiB free; start: $*"
  CUDA_VISIBLE_DEVICES=$card "$@"
  local rc=$?
  echo "rc=$rc card=$card" > "$mark"
  d0_log "exit rc=$rc ($mark)"
  return $rc
}
