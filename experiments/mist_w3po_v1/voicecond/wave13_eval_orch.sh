#!/usr/bin/env bash
# Robust wave 13 orchestrator (reports/mist_w3po_v1/robust_wave13_cleandrop_diag_prereg.zh-CN.md; modelled on
# wave12_eval_orch.sh, Thinker only): merge dzh on CPU, then two card lanes in parallel, each job after the previous:
#   lane CA: oracle erm zh2en (a) -> oracle erm_wise05 zh2en (a) -> la2 base train600 -> la2 dzh zh2en (a) (after merge)
#   lane CB: oracle erm en2zh (a) -> oracle erm_wise05 en2zh (a) -> la2 erm train600
# Every job goes through wave13_eval.sh (detached, pid registered for wave13_noplaceholder.sh) and must leave 600 rows.
# Writes wave13/EVAL_DONE, or EVAL_FAILED on a failed stage. No Talker, no ASR.
#   CA=0 CB=1 bash wave13_eval_orch.sh     (itself run detached with setsid nohup). Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${CA:?}" "${CB:?}"
M=eval_outputs/mist_w3po_v1/wave13
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
E=experiments/mist_w3po_v1/voicecond/wave13_eval.sh
fail() { log "FAILED: $*"; echo "$*" >> "$M/EVAL_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }
job() {   # card sys policy dir items
  local f p n
  [ "$3" = oracle ] && f=blocks_$4_w1_oraclecommit.jsonl || f=blocks_$4_w1.jsonl
  log "start $3 $2 $5 $4 on card $1"
  p=$(launch STAGE=thinker GPU=$1 SYS=$2 POLICY=$3 DIR=$4 ITEMS=$5); [ -n "$p" ] || fail "no pid $*"; waitp $p
  n=$(wc -l < "$M/${3}_${2}_$5/thinker_$4/$f" 2>/dev/null || echo 0)
  [ "$n" = 600 ] || fail "$3 $2 $5 $4: $n of 600 rows"
  log "done $3 $2 $5 $4 ($n rows)"
}
log "merge dzh"
pm=$(launch STAGE=merge GPU=-)
laneA() {
  job "$CA" erm oracle zh2en a; job "$CA" erm_wise05 oracle zh2en a; job "$CA" base la2 zh2en train600
  waitp $pm
  { [ -s "$M/merged_dzh/model.safetensors.index.json" ] && grep -q '"scale": 2.0' logs/mist_w3po_v1/wave13/w13_eval_merge_dzh.log; } || fail "merge dzh"
  job "$CA" dzh la2 zh2en a
}
laneB() { job "$CB" erm oracle en2zh a; job "$CB" erm_wise05 oracle en2zh a; job "$CB" erm la2 zh2en train600; }
laneA & a=$!
sleep 30
laneB & b=$!
wait $a; ra=$?; wait $b; rb=$?
[ $ra = 0 ] && [ $rb = 0 ] || { log "lane failed: A=$ra B=$rb"; exit 1; }
date > "$M/EVAL_DONE"
log "wave 13 Thinker chain done"
