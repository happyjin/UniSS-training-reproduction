#!/usr/bin/env bash
# Robust wave 5 main line M: one arm's whole evaluation chain on two cards CA, CB, once its training has written
# adapters_final.pt (and its trainer pid TPID has exited): merge (CPU) -> LA2 Thinker en2zh on CA, zh2en on CB ->
# Talker 2 dirs x 4 shards (4 per card) -> speech-only ASR (one dir per card) -> <root>/EVAL_DONE. Every job goes
# through wave5_m_eval.sh (detached, pid registered for the placeholder watcher); a failed stage writes EVAL_FAILED.
#   ARM=erm CA=0 CB=1 TPID=<trainer pid> bash wave5_m_eval_orch.sh     (itself run detached with setsid nohup)
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${CA:?}" "${CB:?}" "${TPID:?}"
M=eval_outputs/mist_w3po_v1/wave5/M
ROOT=$M/eval_$ARM
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
E=experiments/mist_w3po_v1/voicecond/wave5_m_eval.sh
fail() { log "FAILED: $*"; mkdir -p "$ROOT"; echo "$*" > "$ROOT/EVAL_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
[ -s "$M/run_$ARM/adapters_final.pt" ] || fail "no adapters_final.pt for $ARM"
log "training of $ARM finished; merge"
p=$(launch STAGE=merge GPU=- ARM=$ARM); waitp $p
[ -s "$M/merged_$ARM/model.safetensors.index.json" ] || fail "merge"
grep -q '"adapted"' logs/mist_w3po_v1/wave5/m_eval_merge_$ARM.log || fail "merge log"
log "thinker"
p1=$(launch STAGE=thinker GPU=$CA ARM=$ARM DIR=en2zh); p2=$(launch STAGE=thinker GPU=$CB ARM=$ARM DIR=zh2en); waitp $p1 $p2
for d in en2zh zh2en; do
  n=$(wc -l < "$ROOT/thinker_$d/blocks_${d}_w1.jsonl" 2>/dev/null || echo 0)
  m=$(wc -l < eval_outputs/mist_w3po_v1/wave2_S1/data/items_$d.jsonl)
  [ "$n" = "$m" ] || fail "thinker $d: $n of $m rows"
done
log "talker"
ps_=()
for k in 0 1 2 3; do
  ps_+=($(launch STAGE=talker GPU=$CA ARM=$ARM DIR=en2zh SHARD=$k)); sleep 20
  ps_+=($(launch STAGE=talker GPU=$CB ARM=$ARM DIR=zh2en SHARD=$k)); sleep 20
done
waitp "${ps_[@]}"
log "asr"
p1=$(launch STAGE=asr GPU=$CA ARM=$ARM DIR=en2zh); p2=$(launch STAGE=asr GPU=$CB ARM=$ARM DIR=zh2en); waitp $p1 $p2
for d in en2zh zh2en; do [ -s "$ROOT/talker_${d}_s20261021/asr_speech.jsonl" ] || fail "asr $d"; done
date > "$ROOT/EVAL_DONE"
log "evaluation chain of $ARM done"
