#!/usr/bin/env bash
# RESUME 2 (2026-11-20 18:58 CET, coordinator 18:50): cards 0 and 3 are uniss-de's combo arm C until combo/C/DONE, so this
# script uses ONLY cards 1, 2, 4, 5 (4-5 only if wave16e_tools.py extracards clears them; never 6-7). It adopts the two
# Thinker groups still running from resume 1 (zh2en a0.2304 card 1 pid 31143, en2zh a1.0000 card 2 pid 39903), reruns
# en2zh a0.5926 (its card-0 run was stopped by PID; output renamed *.superseded_v2) on card 4 (or card 2 once free), then
# row check, assemble, Talker (zh2en shards on 1 + 4, en2zh on 2 + 5; 1 / 2 alone if 4-5 are not cleared), ASR (zh2en
# card 1, en2zh card 2), EVAL_DONE, NCWI_ROOT; STOP; wave16e_score.py; DONE. Launches go through wave16e_eval.sh v2
# (waits <= 120 s per card for placeholder release). Own watcher STOP file.
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
STOP=$LOGD/w16e_resume2_noplaceholder.stop
[ -e "$STOP" ] && { echo "$STOP exists"; exit 1; }
PIDS=$LOGD/w16e_workers.pids STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/wave16_noplaceholder.sh > "$LOGD/noplaceholder_resume2.log" 2>&1 < /dev/null 8>&- 9>&- &
log "resume 2: watcher pid $!"
trap 'touch "$STOP"' EXIT
for f in ORCH_FAILED; do [ -e "$M/$f" ] && command mv -n "$M/$f" "$M/$f.superseded_v2"; done
ROOTN=eval_ncwi_merged
P=$M/eval_ncwi_merged/parts
ADOPT=(31143 39903)
X125=$(extras 125); log "cards 4-5 cleared for a Thinker: '${X125}'"
d=en2zh; al=0.5926
line=$(awk -v d=$d -v a=$al '$1==d && $2==a' "$M/groups.txt")
model=$(echo "$line" | awk '{print $3}'); items=$(echo "$line" | awk '{print $4}')
[ -e "$P/thinker_${d}_a$al/blocks_${d}_w1.jsonl" ] && fail "$d a$al already has output"
if [ -n "$X125" ]; then c=${X125%% *}; else log "no extra card; waiting for card 2 (pid 39903)"; waitp 39903; sleep 30; c=2; fi
rp=$(launch STAGE=thinker_merged GPU=$c DIR=$d ALPHA=$al MODEL=$model ITEMS=$items)
waitp "${ADOPT[@]}" $rp
while read -r d al model items n; do
  got=$(nrows "$M/eval_ncwi_merged/parts/thinker_${d}_a$al/blocks_${d}_w1.jsonl")
  [ "$got" = "$n" ] || fail "merged part $d a$al: $got of $n rows"
done < "$M/groups.txt"
"$PYQ" "$TOOLS" assemble > "$LOGD/assemble.log" 2>&1 || fail "assemble (see $LOGD/assemble.log)"
ROOT=$M/$ROOTN
for d in en2zh zh2en; do
  n=$(nrows "$ROOT/thinker_$d/blocks_${d}_w1.jsonl"); m=$(nrows eval_outputs/mist_w3po_v1/wave2_S1/data/items_$d.jsonl)
  [ "$n" = "$m" ] || fail "thinker $d ($ROOTN): $n of $m rows"
done
sleep 30
X=$(extras 60)
ZC=(1); EC=(2)
case " $X " in *" 4 "*) ZC+=(4);; esac
case " $X " in *" 5 "*) EC+=(5);; esac
log "phase 3: talker on $ROOTN; zh2en shards on cards ${ZC[*]}, en2zh on ${EC[*]}; ASR per direction when its shards finish"
pipe() {   # $1 = direction, $2 = ASR card, rest = shard cards
  local d=$1 ga=$2; shift 2; local cs=("$@") ps_=() k p
  for k in 0 1 2 3; do
    ps_+=($(launch STAGE=talker GPU=${cs[$((k % ${#cs[@]}))]} ROOTN=$ROOTN DIR=$d SHARD=$k)) || exit 1; sleep 20
  done
  waitp "${ps_[@]}"
  log "talker $d done; asr $d on card $ga"
  p=$(launch STAGE=asr GPU=$ga ROOTN=$ROOTN DIR=$d) || exit 1
  waitp $p
}
pipe zh2en 1 "${ZC[@]}" & pz=$!
sleep 10
pipe en2zh 2 "${EC[@]}" & pe=$!
wait $pz; wait $pe
for d in en2zh zh2en; do [ -s "$ROOT/talker_${d}_s20261021/asr_speech.jsonl" ] || fail "asr $d"; done
date > "$ROOT/EVAL_DONE"
echo "$ROOTN" > "$M/NCWI_ROOT"
log "evaluation chain done on $ROOTN"
touch "$STOP"
log "cards back to the guard; scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave16e_score.py --out "$M/summary.json" > "$LOGD/score.log" 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/DONE"
log "ncwi_erm done"
