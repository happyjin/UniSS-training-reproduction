#!/usr/bin/env bash
# Robust wave 14 (copy of wave10_eval_orch.sh): the whole evaluation chain of one arm (unist_wise05 | unist) on FOUR
# cards CA CB CC CD, once run_<arm>/adapters_final.pt exists: merge (CPU, wave5_m_merge.py) -> LA2 Thinker en2zh on CA,
# zh2en on CB -> Talker 2 dirs x 4 shards (en2zh shards 0,1 on CA and 2,3 on CC; zh2en shards 0,1 on CB and 2,3 on CD;
# same --shard/--num-shards 4 as on one card) -> speech-only ASR (en2zh on CA, zh2en on CB) -> <root>/EVAL_DONE.
# Every job goes through wave14_eval.sh (detached, pid registered for wave14_noplaceholder.sh); a failed stage writes
# EVAL_FAILED.
#   ARM=unist_wise05 CA=0 CB=1 CC=2 CD=3 bash wave14_eval_orch.sh     (itself run detached by wave14_chain.sh)
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${CA:?}" "${CB:?}" "${CC:?}" "${CD:?}"
M=eval_outputs/mist_w3po_v1/wave14
ROOT=$M/eval_$ARM
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
E=experiments/mist_w3po_v1/voicecond/wave14_eval.sh
fail() { log "FAILED: $*"; mkdir -p "$ROOT"; echo "$*" > "$ROOT/EVAL_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
[ -s "$M/run_$ARM/adapters_final.pt" ] || fail "no adapters_final.pt for $ARM"
log "merge $ARM"
p=$(launch STAGE=merge GPU=- ARM=$ARM); waitp $p
[ -s "$M/merged_$ARM/model.safetensors.index.json" ] || fail "merge"
grep -q '"adapted"' logs/mist_w3po_v1/wave14/w14_eval_merge_$ARM.log || fail "merge log"
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
  if [ "$k" -lt 2 ]; then ge=$CA; gz=$CB; else ge=$CC; gz=$CD; fi
  ps_+=($(launch STAGE=talker GPU=$ge ARM=$ARM DIR=en2zh SHARD=$k)); sleep 20
  ps_+=($(launch STAGE=talker GPU=$gz ARM=$ARM DIR=zh2en SHARD=$k)); sleep 20
done
waitp "${ps_[@]}"
log "asr"
p1=$(launch STAGE=asr GPU=$CA ARM=$ARM DIR=en2zh); p2=$(launch STAGE=asr GPU=$CB ARM=$ARM DIR=zh2en); waitp $p1 $p2
for d in en2zh zh2en; do [ -s "$ROOT/talker_${d}_s20261021/asr_speech.jsonl" ] || fail "asr $d"; done
date > "$ROOT/EVAL_DONE"
log "evaluation chain of $ARM done"
