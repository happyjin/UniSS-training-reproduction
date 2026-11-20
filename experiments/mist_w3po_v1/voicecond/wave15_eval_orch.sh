#!/usr/bin/env bash
# Robust wave 15 (copy of wave11_eval_orch.sh without the Fisher / merge stages): the S1 chain of one CV-LA arm on two
# cards CA (en2zh) and CB (zh2en): CV-LA Thinker en2zh on CA, zh2en on CB -> Talker 2 dirs x 4 shards (en2zh shards on CA,
# zh2en shards on CB) -> speech-only ASR (one dir per card) -> <root>/EVAL_DONE. Every job goes through wave15_eval.sh
# (detached, pid registered for wave15_noplaceholder.sh); a failed stage writes EVAL_FAILED.
#   ARM=cvla_base CA=4 CB=5 bash wave15_eval_orch.sh     (run by wave15_chain_g1.sh / wave15_wait_g2.sh)
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${CA:?}" "${CB:?}"
M=eval_outputs/mist_w3po_v1/wave15
ROOT=$M/eval_$ARM
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
E=experiments/mist_w3po_v1/voicecond/wave15_eval.sh
fail() { log "FAILED: $*"; mkdir -p "$ROOT"; echo "$*" > "$ROOT/EVAL_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
for d in en2zh zh2en; do [ -s "$M/views/$d/DONE.json" ] || fail "no views for $d"; done
log "thinker $ARM"
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
