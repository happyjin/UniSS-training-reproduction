#!/usr/bin/env bash
# Night switch after the joint stage-2 selection (user 2026-11-13 ~20:10 Prague: decide by the best option overnight;
# deliver strict baseline vs method, both directions, dev + test; never leave a card idle). Detached; waits for
# logs/mist_w3po_v1/joint/JOINT_S2_SELECTED (JOINT_STOP s2 in the driver log -> logs and exits, nothing done), then:
#   1. the stage-2 dev5 proxies still queued or running (checkpoints after the decision; their outputs are not used by
#      anything once the selection is written) are stopped by PID: proxy bash (cmdline + environ RUN checked), its
#      orchestrators (joint_orch.py --set dev5 ... <run>), and their running decode jobs (process group of each job
#      shell from <orch run>/state/<job>.a<n>.pid without .rc; cmdline must name the run). Nothing is deleted.
#   2. the baseline-row dev orchestrators (logs/mist_w3po_v1/joint/dev/{bz,be,v8z,v8e}) get all 8 cards back
#      (exclude_cards emptied; the previous contents kept as exclude_cards.before_night_<UTC>)
#   3. (a)'s En->Zh chain resumes (prereg joint 修正 4 §13): en2zh_after_select.sh with its paused environ
#      (logs/mist_w3po_v1/paused_20261113/paused.tsv) + RESUME_ORCH=1, and the en2zh_test.sh waiter with its environ
#   -> line NIGHT_SWITCH_DONE
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_night_after_s2.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=joint_v2p_20261113T120415Z
JL=logs/mist_w3po_v1/joint
M2=$JL/JOINT_S2_SELECTED
R=$JL/$RUN
E=logs/mist_w3po_v1/en2zh
PT=logs/mist_w3po_v1/paused_20261113/paused.tsv
log() { echo "$(date -u +%FT%TZ) $*"; }
proxies() {   # pids of the stage-2 dev5 proxies of RUN
  local p
  for p in $(ps -eo pid=,args= | awk '$2 == "bash" && $3 == "experiments/mist_w3po_v1/scripts/joint_proxy.sh" {print $1}'); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "RUN=$RUN" && echo $p
  done
}
orchs() {   # pids of RUN's dev5 orchestrators
  ps -eo pid=,args= | awk -v r="$RUN" '$3 ~ /joint_orch.py$/ && index($0, "--set dev5") && index($0, r) {print $1}'
}
jobs_running() {   # job-shell pids (process groups) of running jobs in RUN's proxy orchestrator dirs
  local f p
  for f in $R/proxy/[ze]_s?????/state/*.pid; do
    [ -e "$f" ] || continue
    [ -e "${f%.pid}.rc" ] && continue
    p=$(cat $f 2>/dev/null) || continue
    [ -n "$p" ] && [ -e /proc/$p ] || continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$RUN" && echo $p
  done
}
relaunch() {   # <paused pid> <log> [KEY=VAL ...]: the paused process's cmdline / cwd / environ (+ overrides), detached
  local pid=$1 lf=$2; shift 2
  /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python - "$PT" "$pid" "$lf" "$@" <<'EOF'
import csv, json, subprocess, sys
csv.field_size_limit(10 ** 9)
pt, pid, lf, over = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
r = [x for x in csv.DictReader(open(pt), delimiter="\t") if x["pid"] == pid][0]
env = {k: v for k, v in json.loads(r["environ_json"]).items() if not k.startswith("BASH_FUNC_")}
for kv in over:
    k, v = kv.split("=", 1)
    env[k] = v
with open(lf, "ab") as fh:
    fh.write(f"=== relaunch of paused pid {pid} (night switch, overrides {over}) ===\n".encode())
    fh.flush()
    p = subprocess.Popen(json.loads(r["cmdline_json"]), env=env, cwd=r["cwd"], stdin=subprocess.DEVNULL, stdout=fh,
                         stderr=fh, start_new_session=True)
print(p.pid)
EOF
}
if [ -n "${DRY_RUN:-}" ]; then
  echo "marker: $(cat $M2 2>/dev/null || echo 'not yet')"
  echo "proxies now: $(proxies | tr '\n' ' ')"
  echo "orchestrators now: $(orchs | tr '\n' ' ')"
  echo "running jobs now: $(jobs_running | tr '\n' ' ')"
  for d in bz be v8z v8e; do echo "dev/$d exclude: $(cat $JL/dev/$d/exclude_cards)"; done
  awk -F'\t' '$2 == "a_en2zh_driver" {print "paused (a) driver " $3 ": " $6}' $PT
  exit 0
fi
log "night switch: waiting for $M2"
until [ -e $M2 ]; do
  grep -q "JOINT_STOP" $JL/stage2_s2full_r3.log 2>/dev/null && { log "JOINT_STOP in the stage-2 driver log: nothing done"; exit 1; }
  sleep 60
done
log "selected: $(cat $M2)"
# 1. leftover dev5 work
P=$(proxies); O=$(orchs)
[ -n "$P" ] && kill -TERM $P && log "proxies stopped: $(echo $P)"
sleep 2
[ -n "$O" ] && kill -TERM $O && log "dev5 orchestrators stopped: $(echo $O)"
sleep 5
for g in $(jobs_running); do kill -TERM -- -$g 2>/dev/null && log "dev5 decode job group $g stopped"; done
# 2. baseline dev rows: all cards
T=$(date -u +%Y%m%dT%H%M%SZ)
for d in bz be v8z v8e; do
  f=$JL/dev/$d/exclude_cards
  [ -e $f ] || continue
  cp $f $f.before_night_$T && echo "" > $f.tmp && command mv -f $f.tmp $f && log "dev/$d: exclude_cards emptied (was '$(cat $f.before_night_$T)')"
done
# 3. (a) En->Zh
A=$(relaunch 1222510 $E/after_select.log RESUME_ORCH=1) && log "(a) en2zh_after_select relaunched pid $A (RESUME_ORCH=1)"
B=$(relaunch 3192302 $E/test.log) && log "(a) en2zh_test waiter relaunched pid $B"
log "NIGHT_SWITCH_DONE"
