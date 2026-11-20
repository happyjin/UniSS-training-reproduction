#!/usr/bin/env bash
# RESUME 3 (2026-11-20 18:59 CET): in resume 2 the en2zh Talker could not start on card 2 (the guard restarted
# placeholders during the 120 s wait; launcher v3 now stops those too), so resume 2's en2zh pipe stopped while its zh2en
# pipe (Talker on cards 1, 4, then ASR on card 1) goes on. This script runs the en2zh pipe only -- Talker shards on cards
# 2 and 5 (5 only if extracards clears it, else 2), ASR on card 2 -- then waits for resume 2 (pid 63490) to exit. If resume
# 2 found both ASR files and finished (DONE), nothing more is done; else: check, EVAL_DONE, NCWI_ROOT, wave16e_score.py,
# DONE. Never cards 0, 3 (combo arm C), 6, 7. Own watcher STOP file.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave16/erm
W14=eval_outputs/mist_w3po_v1/wave5/M   # ncwi_erm: CoVoST ERM endpoints
LOGD=logs/mist_w3po_v1/wave16/erm
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=experiments/mist_w3po_v1/voicecond/wave16e_eval.sh
TOOLS=experiments/mist_w3po_v1/voicecond/wave16e_tools.py
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/ORCH_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
nrows() { wc -l < "$1" 2>/dev/null || echo 0; }
extras() { "$PYQ" "$TOOLS" extracards --min-gb "$1" 2>> "$LOGD/extracards.log"; }
STOP=$LOGD/w16e_resume3_noplaceholder.stop
[ -e "$STOP" ] && { echo "$STOP exists"; exit 1; }
PIDS=$LOGD/w16e_workers.pids STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/wave16_noplaceholder.sh > "$LOGD/noplaceholder_resume3.log" 2>&1 < /dev/null 8>&- 9>&- &
log "resume 3: watcher pid $!"
trap 'touch "$STOP"' EXIT
ROOTN=eval_ncwi_merged
ROOT=$M/$ROOTN
R2=63490
EC=(2); case " $(extras 60) " in *" 5 "*) EC+=(5);; esac
log "en2zh Talker shards on cards ${EC[*]}"
ps_=()
for k in 0 1 2 3; do
  [ -e "$ROOT/talker_en2zh_s20261021/results.jsonl" ] && log "WARNING: en2zh results.jsonl exists before shard $k"
  ps_+=($(launch STAGE=talker GPU=${EC[$((k % ${#EC[@]}))]} ROOTN=$ROOTN DIR=en2zh SHARD=$k)); sleep 20
done
waitp "${ps_[@]}"
log "talker en2zh done; asr en2zh on card 2"
p=$(launch STAGE=asr GPU=2 ROOTN=$ROOTN DIR=en2zh); waitp $p
[ -s "$ROOT/talker_en2zh_s20261021/asr_speech.jsonl" ] || fail "asr en2zh"
log "waiting for resume 2 (pid $R2: zh2en Talker + ASR)"
while [ "$(tr '\0' ' ' < /proc/$R2/cmdline 2>/dev/null)" = "bash experiments/mist_w3po_v1/voicecond/wave16e_resume2.sh " ]; do sleep 20; done
if [ -e "$M/DONE" ]; then log "resume 2 finished the chain (DONE exists)"; exit 0; fi
for d in en2zh zh2en; do [ -s "$ROOT/talker_${d}_s20261021/asr_speech.jsonl" ] || fail "asr $d"; done
[ -e "$M/ORCH_FAILED" ] && command mv -n "$M/ORCH_FAILED" "$M/ORCH_FAILED.superseded_v2"
date > "$ROOT/EVAL_DONE"
echo "$ROOTN" > "$M/NCWI_ROOT"
log "evaluation chain done on $ROOTN"
touch "$STOP"
log "cards back to the guard; scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave16e_score.py --out "$M/summary.json" > "$LOGD/score.log" 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/DONE"
log "ncwi_erm done"
