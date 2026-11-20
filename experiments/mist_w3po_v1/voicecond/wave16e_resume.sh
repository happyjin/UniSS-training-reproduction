#!/usr/bin/env bash
# RESUME (2026-11-20 18:55 CET): the ncwi_erm chain stopped at "merged part en2zh a0.5926: 0 of 167 rows" -- 3 of 15
# Thinker groups died at start-up (CUDA OOM on card 3 while other processes appeared there; aten::new_empty failure on
# card 0 right after the previous group). This script reruns ONLY the 3 groups (one per card, cards 0-2; the old logs
# renamed *.superseded_v1, the empty output dirs reused), then continues exactly as wave16e_orch.sh after the lanes:
# row check, assemble, Talker / ASR, EVAL_DONE, NCWI_ROOT; then STOP, wave16e_score.py, DONE. Own watcher STOP file.
# ncwi_erm (change B): sed copy of wave16_orch.sh v2: root wave16/erm, endpoints wave5/M (erm_wise05 / erm, run_erm LoRA),
# wave16e_eval.sh / wave16e_tools.py, NCWI Thinkers on cards 1,2 (coordinator 18:15), alpha-0.5 check on 0,3.
# Robust wave 16 NCWI evaluation chain on cards 0-3 (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md sections 3-5;
# modelled on wave14_eval_orch.sh). Needs wave16/fit/assign_<dir>.jsonl and wave16/lora/a<alpha>/ (wave16_wait.sh makes
# them). Every GPU job goes through wave16_eval.sh (detached, pid registered for wave16_noplaceholder.sh).
#   1. LoRA path, 4 Thinkers in parallel: NCWI en2zh card 0, zh2en card 1; alpha-0.5 check en2zh card 2, zh2en card 3.
#   2. wave16_tools.py agree: accepted -> root eval_ncwi; rejected (engine error or agreement below the base-rerun
#      shares) -> merged fallback: wave5_m_merge.py copies for the alpha* not in {0, 0.5, 1} (CPU, parallel), one LA2 run
#      per (dir, alpha*) item group on 4 card lanes, wave16_tools.py assemble -> root eval_ncwi_merged.
#   3. Talker 2 dirs x 4 shards (en2zh 0,1 card 0 / 2,3 card 2; zh2en 0,1 card 1 / 2,3 card 3), speech-only ASR (en2zh
#      card 0, zh2en card 1) -> <root>/EVAL_DONE, wave16/NCWI_ROOT = the root's name.
# v2 (16:25 CET, coordinator's speed-up): cards 4-5 are stacked on when wave16_tools.py extracards clears them at that
# moment (no training process; free memory incl. killable placeholders >= 60 GB for Talker shards, >= 125 GB for a
# Thinker lane of the merged fallback); cards 6-7 never. Talker: zh2en shards on cards 1,3 + extras (one each when both
# extras are free), en2zh shards on cards 0,2; each direction's ASR starts as soon as its own 4 shards are done (per-
# direction pipeline; same --num-shards 4, so outputs are unchanged). ASR itself is not split: Whisper sorts and batches
# the whole file by duration, so a split would change the batches.
# A failed stage writes wave16/ORCH_FAILED and exits 1.   (run in the foreground of wave16_wait.sh)
# Checked with bash -n only.
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
STOP=$LOGD/w16e_resume_noplaceholder.stop
[ -e "$STOP" ] && { echo "$STOP exists"; exit 1; }
PIDS=$LOGD/w16e_workers.pids STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/wave16_noplaceholder.sh > "$LOGD/noplaceholder_resume.log" 2>&1 < /dev/null 8>&- 9>&- &
log "resume: watcher pid $!"
trap 'touch "$STOP"' EXIT
for f in ORCH_FAILED CHAIN16E_FAILED; do [ -e "$M/$f" ] && command mv -n "$M/$f" "$M/$f.superseded_v1"; done
ROOTN=eval_ncwi_merged
P=$M/eval_ncwi_merged/parts
rp=()
c=0
for g in en2zh:0.5926 zh2en:0.2304 en2zh:1.0000; do
  d=${g%%:*}; al=${g##*:}
  line=$(awk -v d=$d -v a=$al '$1==d && $2==a' "$M/groups.txt")
  model=$(echo "$line" | awk '{print $3}'); items=$(echo "$line" | awk '{print $4}')
  [ -n "$model" ] && [ -n "$items" ] || fail "group $g not in groups.txt"
  [ -e "$P/thinker_${d}_a$al/blocks_${d}_w1.jsonl" ] && fail "$g already has output"
  lf=$LOGD/w16e_eval_thinker_merged_${d}_a$al.log
  [ -e "$lf" ] && { command mv -n "$lf" "$lf.superseded_v1" || fail "rename $lf"; }
  rp+=($(launch STAGE=thinker_merged GPU=$c DIR=$d ALPHA=$al MODEL=$model ITEMS=$items)); c=$((c + 1)); sleep 30
done
waitp "${rp[@]}"
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
X=$(extras 60)
ZC=(1 3 $X); EC=(0 2)
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
pipe en2zh 0 "${EC[@]}" & pe=$!
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
