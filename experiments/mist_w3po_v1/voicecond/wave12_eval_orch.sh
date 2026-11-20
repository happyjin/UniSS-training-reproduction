#!/usr/bin/env bash
# Robust wave 12 (copy of wave11_eval_orch.sh, Thinker only): for each ERM resume checkpoint 00500..02000 in turn:
# merge at alpha = 1 (CPU; the next merge overlaps the current Thinker runs) -> LA2 Thinker en2zh on CA, zh2en on CB on
# the (a) unit -> check 600 rows per direction. Writes wave12/EVAL_DONE, or EVAL_FAILED on a failed stage. Every job
# goes through wave12_eval.sh (detached, pid registered for wave12_noplaceholder.sh). No Talker, no ASR.
#   CA=0 CB=1 bash wave12_eval_orch.sh     (itself run detached with setsid nohup)
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${CA:?}" "${CB:?}"
M=eval_outputs/mist_w3po_v1/wave12
CKPTS=(00500 01000 01500 02000)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
E=experiments/mist_w3po_v1/voicecond/wave12_eval.sh
fail() { log "FAILED: $*"; echo "$*" > "$M/EVAL_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }
checkm() { [ -s "$M/merged_ckpt_$1/model.safetensors.index.json" ] && grep -q '"scale": 2.0' logs/mist_w3po_v1/wave12/w12_eval_merge_ckpt_$1.log || fail "merge $1"; }
log "merge ${CKPTS[0]}"
pm=$(launch STAGE=merge GPU=- CKPT=${CKPTS[0]})
for i in 0 1 2 3; do
  c=${CKPTS[$i]}
  waitp $pm; checkm $c
  if [ $i -lt 3 ]; then log "merge ${CKPTS[$((i + 1))]}"; pm=$(launch STAGE=merge GPU=- CKPT=${CKPTS[$((i + 1))]}); fi
  log "thinker ckpt_$c"
  p1=$(launch STAGE=thinker GPU=$CA CKPT=$c DIR=en2zh); p2=$(launch STAGE=thinker GPU=$CB CKPT=$c DIR=zh2en); waitp $p1 $p2
  for d in en2zh zh2en; do
    n=$(wc -l < "$M/eval_ckpt_$c/thinker_$d/blocks_${d}_w1.jsonl" 2>/dev/null || echo 0)
    [ "$n" = 600 ] || fail "thinker ckpt_$c $d: $n of 600 rows"
  done
done
date > "$M/EVAL_DONE"
log "wave 12 Thinker chain done"
