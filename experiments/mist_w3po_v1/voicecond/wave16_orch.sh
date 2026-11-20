#!/usr/bin/env bash
# Robust wave 16 NCWI evaluation chain on cards 0-3 (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md sections 3-5;
# modelled on wave14_eval_orch.sh). Needs wave16/fit/assign_<dir>.jsonl and wave16/lora/a<alpha>/ (wave16_wait.sh makes
# them). Every GPU job goes through wave16_eval.sh (detached, pid registered for wave16_noplaceholder.sh).
#   1. LoRA path, 4 Thinkers in parallel: NCWI en2zh card 0, zh2en card 1; alpha-0.5 check en2zh card 2, zh2en card 3.
#   2. wave16_tools.py agree: accepted -> root eval_ncwi; rejected (engine error or agreement below the base-rerun
#      shares) -> merged fallback: wave5_m_merge.py copies for the alpha* not in {0, 0.5, 1} (CPU, parallel), one LA2 run
#      per (dir, alpha*) item group on 4 card lanes, wave16_tools.py assemble -> root eval_ncwi_merged.
#   3. Talker 2 dirs x 4 shards (en2zh 0,1 card 0 / 2,3 card 2; zh2en 0,1 card 1 / 2,3 card 3), speech-only ASR (en2zh
#      card 0, zh2en card 1) -> <root>/EVAL_DONE, wave16/NCWI_ROOT = the root's name.
# A failed stage writes wave16/ORCH_FAILED and exits 1.   (run in the foreground of wave16_wait.sh)
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave16
W14=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave16
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=experiments/mist_w3po_v1/voicecond/wave16_eval.sh
TOOLS=experiments/mist_w3po_v1/voicecond/wave16_tools.py
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/ORCH_FAILED"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
nrows() { wc -l < "$1" 2>/dev/null || echo 0; }
for d in en2zh zh2en; do [ -s "$M/fit/assign_$d.jsonl" ] || fail "no fit/assign_$d.jsonl"; done
[ -d "$M/lora" ] || fail "no lora adapters"

log "phase 1: LoRA path (NCWI cards 0,1; alpha-0.5 check cards 2,3)"
p1=$(launch STAGE=thinker_lora GPU=0 ROOTN=eval_ncwi DIR=en2zh); sleep 30
p2=$(launch STAGE=thinker_lora GPU=1 ROOTN=eval_ncwi DIR=zh2en); sleep 30
p3=$(launch STAGE=thinker_lora GPU=2 ROOTN=eval_lora05 DIR=en2zh CONST=0.5); sleep 30
p4=$(launch STAGE=thinker_lora GPU=3 ROOTN=eval_lora05 DIR=zh2en CONST=0.5)
waitp $p1 $p2 $p3 $p4
log "phase 2: agreement check"
if "$PYQ" "$TOOLS" agree --out "$M/agree.json" > "$LOGD/agree.log" 2>&1; then
  ROOTN=eval_ncwi
  log "LoRA path accepted ($M/agree.json)"
else
  rc=$?
  [ "$rc" = 3 ] || fail "agree exited $rc (see $LOGD/agree.log)"
  ROOTN=eval_ncwi_merged
  log "LoRA path rejected ($M/agree.json); merged fallback"
  "$PYQ" "$TOOLS" groups > "$M/groups.txt" 2> "$LOGD/groups.log" || fail "groups"
  mp=()
  for al in $(awk '{print $2}' "$M/groups.txt" | sort -u); do
    case "$al" in 0.0000|0.5000|1.0000) continue;; esac
    A=$M/adapters_a$al
    [ -e "$A" ] && fail "$A exists"
    mkdir -p "$A"
    ln "$W14/run_unist/adapters_final.pt" "$A/adapters_final.pt" || fail "hard link $A"
    "$PYQ" "$TOOLS" wisejson --alpha "$al" >> "$LOGD/wisejson.log" 2>&1 || fail "wisejson $al"
    mp+=($(launch STAGE=merge GPU=- ALPHA=$al))
  done
  waitp "${mp[@]}"
  for al in $(awk '{print $2}' "$M/groups.txt" | sort -u); do
    case "$al" in 0.0000|0.5000|1.0000) continue;; esac
    [ -s "$M/merged_a$al/model.safetensors.index.json" ] && grep -q '"adapted"' "$LOGD/w16_eval_merge_a$al.log" || fail "merge $al"
  done
  for m_ in $(awk '{print $3}' "$M/groups.txt" | sort -u); do
    [ -s "$m_/model.safetensors.index.json" ] || [ -s "$m_/config.json" ] || fail "model $m_ missing"
  done
  # 4 card lanes, jobs round-robin, each lane sequential
  i=0; lanes=("" "" "" "")
  while read -r d al model items n; do
    lanes[$((i % 4))]+="$d|$al|$model|$items;"; i=$((i + 1))
  done < "$M/groups.txt"
  lp=()
  for c in 0 1 2 3; do
    (
      IFS=';' read -ra jobs <<< "${lanes[$c]}"
      for j in "${jobs[@]}"; do
        [ -n "$j" ] || continue
        IFS='|' read -r d al model items <<< "$j"
        p=$(launch STAGE=thinker_merged GPU=$c DIR=$d ALPHA=$al MODEL=$model ITEMS=$items) || exit 1
        waitp $p
        sleep 10
      done
    ) &
    lp+=($!)
  done
  for p in "${lp[@]}"; do wait "$p" || log "WARNING: lane $p exited non-zero"; done
  while read -r d al model items n; do
    got=$(nrows "$M/eval_ncwi_merged/parts/thinker_${d}_a$al/blocks_${d}_w1.jsonl")
    [ "$got" = "$n" ] || fail "merged part $d a$al: $got of $n rows"
  done < "$M/groups.txt"
  "$PYQ" "$TOOLS" assemble > "$LOGD/assemble.log" 2>&1 || fail "assemble (see $LOGD/assemble.log)"
fi
ROOT=$M/$ROOTN
for d in en2zh zh2en; do
  n=$(nrows "$ROOT/thinker_$d/blocks_${d}_w1.jsonl"); m=$(nrows eval_outputs/mist_w3po_v1/wave2_S1/data/items_$d.jsonl)
  [ "$n" = "$m" ] || fail "thinker $d ($ROOTN): $n of $m rows"
done
log "phase 3: talker on $ROOTN"
ps_=()
for k in 0 1 2 3; do
  if [ "$k" -lt 2 ]; then ge=0; gz=1; else ge=2; gz=3; fi
  ps_+=($(launch STAGE=talker GPU=$ge ROOTN=$ROOTN DIR=en2zh SHARD=$k)); sleep 20
  ps_+=($(launch STAGE=talker GPU=$gz ROOTN=$ROOTN DIR=zh2en SHARD=$k)); sleep 20
done
waitp "${ps_[@]}"
log "asr"
p1=$(launch STAGE=asr GPU=0 ROOTN=$ROOTN DIR=en2zh); p2=$(launch STAGE=asr GPU=1 ROOTN=$ROOTN DIR=zh2en); waitp $p1 $p2
for d in en2zh zh2en; do [ -s "$ROOT/talker_${d}_s20261021/asr_speech.jsonl" ] || fail "asr $d"; done
date > "$ROOT/EVAL_DONE"
echo "$ROOTN" > "$M/NCWI_ROOT"
log "evaluation chain done on $ROOTN"
