#!/usr/bin/env bash
# Robust wave 17 post-training chain v2 (replaces wave17_chain.sh, stopped by PID; main session 16:15 CET: run the
# evaluation in parallel, stacking Talker / ASR shards onto other cards that run only evaluation or placeholders and have
# >= 60 GB free). reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md, implementation note 8.
# Same jobs as v1, scheduled as a dependency graph instead of a serial chain:
#   wait for the trainer TPID -> WiSE 0.5 copy run_enc_wise05 (as v1) -> tasks, in priority order:
#     merge enc_wise05, merge enc (CPU, at once, in parallel)
#     mech la2 base train600 (no dependency)
#     thinker enc_wise05 en2zh / zh2en; talker enc_wise05 dir x shard 0..3; asr enc_wise05 dir
#     thinker enc en2zh / zh2en; talker enc ...; asr enc ...
#     mech la2 enc train600, oracle enc a zh2en, oracle enc a en2zh (after merge enc)
#     mech la2 unist train600 (after wave 14's merge of unist has finished; skipped if it never does)
#   every job through wave17_eval.sh (detached, pid -> w17_workers.pids, so wave17_noplaceholder.sh keeps placeholders
#   off whatever card it runs on); outputs verified as in v1 (thinker rows = items, merge self-check, asr file, mech 600).
# Card choice (nvidia-smi, every 15 s; placeholder memory counts as free since the launcher stops them by PID):
#   Thinker-type jobs (vLLM gpu-mem 0.85, about 122 GB): cards 6, 7 only, no non-placeholder process on the card, >= 125 GB
#     effective free.
#   Talker shards / ASR (about 14 / 12 GB each; 16 GB reserved): effective free >= 60 GB and
#     cards 6, 7: only once no Thinker-type job is pending or running (they would block it);
#     cards 0-5: before 19:15 CET only (wave 16 NCWI takes 0-3 at 19:30); never a card with a training process
#       (*_train.py / torch.distributed / /training/ in the argv of the process or its parents); the other processes must
#       all be Talker / ASR (si_infer, asr_transcribe); a placeholder-only card qualifies only among 0-3 and only once
#       wave 14's four Thinker runs are complete (wave 14 starts no Thinker after that); placeholder-only 4, 5 never
#       (wave 15 may start a Thinker there at any time).
#   Effective free = total - used + placeholder memory - my reservations of the last 150 s (240 s for a Thinker), so
#   memory that has not shown up yet is not handed out twice.
# After all GPU jobs: touch STOP -> mech score -> wait for wave 14 unist_wise05 EVAL_DONE (+ up to 2 h for its
# CHAIN_DONE) -> wave17_score.py -> CHAIN_DONE / CHAIN_FAILED. Never touches guard 2726951, gpu_watch 52045, sync 1222453.
#   TPID=<torchrun pid> setsid nohup bash wave17_chain_v2.sh > logs/mist_w3po_v1/wave17/chain_v2.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}"
M=eval_outputs/mist_w3po_v1/wave17
W14=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave17
STOP=$LOGD/w17_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=experiments/mist_w3po_v1/voicecond/wave17_eval.sh
S1=eval_outputs/mist_w3po_v1/wave2_S1/data
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; touch "$STOP"; exit 1; }

log "chain v2 $$: waiting for trainer $TPID"
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
R=$M/run_enc
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; WiSE 0.5 copy"
W=$M/run_enc_wise05
[ -e "$W" ] && fail "$W exists"
mkdir -p "$W"
ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "enc" and m["module"] == "thinker.audio_tower" and m["alpha"] == 32.0 and m["rank"] == 16 \
    and m["steps"] == 2300 and m["world"] == 2, m
m.update({"arm": "enc_wise05", "alpha": m["alpha"] / 2,
          "wise_ft": "W0 + 0.5*(W_enc - W0): LoRA alpha halved, adapters = run_enc/adapters_final.pt"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
print(json.dumps(m))
PY

# ------------------------------------------------------------------ task graph
declare -A ST PID CARD DEPS KIND ARGS RESV
order=()
add() { local n=$1; KIND[$n]=$2; DEPS[$n]=$3; shift 3; ARGS[$n]="$*"; ST[$n]=pending; order+=("$n"); }
add merge_enc_wise05 cpu "" STAGE=merge GPU=- ARM=enc_wise05
add merge_enc cpu "" STAGE=merge GPU=- ARM=enc
add mech_la2_base_train600_zh2en thinker "" STAGE=mech SYS=base POLICY=la2 DIR=zh2en MSET=train600
for arm in enc_wise05 enc; do
  for d in en2zh zh2en; do add th_${arm}_$d thinker "merge_$arm" STAGE=thinker ARM=$arm DIR=$d; done
  for d in en2zh zh2en; do
    for k in 0 1 2 3; do add tk_${arm}_${d}_$k shard "th_${arm}_$d" STAGE=talker ARM=$arm DIR=$d SHARD=$k; done
    add asr_${arm}_$d shard "tk_${arm}_${d}_0 tk_${arm}_${d}_1 tk_${arm}_${d}_2 tk_${arm}_${d}_3" STAGE=asr ARM=$arm DIR=$d
  done
done
add mech_la2_enc_train600_zh2en thinker "merge_enc" STAGE=mech SYS=enc POLICY=la2 DIR=zh2en MSET=train600
add mech_oracle_enc_a_zh2en thinker "merge_enc" STAGE=mech SYS=enc POLICY=oracle DIR=zh2en MSET=a
add mech_oracle_enc_a_en2zh thinker "merge_enc" STAGE=mech SYS=enc POLICY=oracle DIR=en2zh MSET=a
add mech_la2_unist_train600_zh2en thinker "W14U" STAGE=mech SYS=unist POLICY=la2 DIR=zh2en MSET=train600

w14u_ready() { [ -s "$W14/merged_unist/model.safetensors.index.json" ] && grep -q '"adapted"' logs/mist_w3po_v1/wave14/w14_eval_merge_unist.log 2>/dev/null; }
w14_thinkers_done() {
  local a d n m
  for a in unist_wise05 unist; do for d in en2zh zh2en; do
    n=$(wc -l < "$W14/eval_$a/thinker_$d/blocks_${d}_w1.jsonl" 2>/dev/null || echo 0); m=$(wc -l < "$S1/items_$d.jsonl")
    [ "$n" = "$m" ] || return 1
  done; done
}
depstate() {   # ok | wait | fail
  local n=$1 x
  for x in ${DEPS[$n]}; do
    if [ "$x" = W14U ]; then w14u_ready || { echo wait; return; }; continue; fi
    case ${ST[$x]} in done) ;; failed|skipped) echo fail; return;; *) echo wait; return;; esac
  done
  echo ok
}
verify() {   # task -> 0 ok
  local n=$1 a d f
  set -- ${ARGS[$n]}
  local STAGE= ARM= DIR= SYS= POLICY= MSET=
  for a in "$@"; do case $a in STAGE=*|ARM=*|DIR=*|SYS=*|POLICY=*|MSET=*) eval "${a%%=*}='${a#*=}'";; esac; done
  case $STAGE in
    merge) [ -s "$M/merged_$ARM/model.safetensors.index.json" ] && grep -q '"adapted": 128' $LOGD/w17_eval_merge_$ARM.log \
             && grep -q '"ok": true' $LOGD/w17_eval_merge_$ARM.log ;;
    thinker) [ "$(wc -l < "$M/eval_$ARM/thinker_$DIR/blocks_${DIR}_w1.jsonl" 2>/dev/null || echo 0)" = "$(wc -l < $S1/items_$DIR.jsonl)" ] ;;
    talker) return 0 ;;
    asr) [ -s "$M/eval_$ARM/talker_${DIR}_s20261021/asr_speech.jsonl" ] ;;
    mech) [ "$POLICY" = oracle ] && f=blocks_${DIR}_w1_oraclecommit.jsonl || f=blocks_${DIR}_w1.jsonl
          [ "$(wc -l < "$M/${POLICY}_${SYS}_$MSET/thinker_$DIR/$f" 2>/dev/null || echo 0)" = 600 ] ;;
  esac
}
reserved() {   # card -> MiB reserved by my launches that may not show in nvidia-smi yet
  local c=$1 now e tot=0 r
  now=$(date +%s)
  for r in ${RESV[$c]:-}; do e=${r#*:}; [ "$e" -gt "$now" ] && tot=$((tot + ${r%%:*})); done
  echo $tot
}
th_open() { local n; for n in "${order[@]}"; do [ "${KIND[$n]}" = thinker ] && case ${ST[$n]} in pending|run) return 0;; esac; done; return 1; }
argv3() { local p=$1 s="" i; for i in 1 2 3; do [ -r /proc/$p/cmdline ] || break; s="$s $(tr '\0' ' ' < /proc/$p/cmdline)"; p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -n "$p" ] && [ "$p" -gt 1 ] || break; done; echo "$s"; }
pick() {   # kind -> card or empty
  local kind=$1 map apps c tot used ph other train talkish uuid pid mem a free hm
  map=$(nvidia-smi --query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader,nounits | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | tr -d ' ')
  hm=$(TZ=Europe/Prague date +%H%M)
  for c in 6 7 0 1 2 3 4 5; do
    if [ "$kind" = thinker ]; then case $c in 6|7) ;; *) continue;; esac; fi
    if [ "$kind" = shard ]; then
      case $c in 6|7) th_open && continue;; *) [ "$((10#$hm))" -lt 1915 ] || continue;; esac
    fi
    IFS=, read -r _ uuid tot used <<< "$(echo "$map" | awk -F, -v i=$c '$1==i')"
    ph=0; other=0; train=0; talkish=1
    while IFS=, read -r u pid mem; do
      [ "$u" = "$uuid" ] || continue
      a=$(argv3 "$pid")
      case "$a" in
        *scripts/gpu_load/*) ph=$((ph + mem)) ;;
        *) other=$((other + 1))
           case "$a" in *_train.py*|*torch.distributed*|*/training/*) train=1;; esac
           case "$a" in *si_infer*|*asr_transcribe*) ;; *) talkish=0;; esac ;;
      esac
    done <<< "$apps"
    free=$((tot - used + ph - $(reserved $c)))
    if [ "$kind" = thinker ]; then
      [ "$other" = 0 ] && [ "$free" -ge 128000 ] && { echo $c; return; }
      continue
    fi
    [ "$train" = 0 ] && [ "$free" -ge 61440 ] || continue
    case $c in
      6|7) echo $c; return ;;
      *) [ "$talkish" = 1 ] || continue
         if [ "$other" = 0 ]; then case $c in 0|1|2|3) w14_thinkers_done || continue;; *) continue;; esac; fi
         echo $c; return ;;
    esac
  done
}
launch() {   # task card
  local n=$1 c=$2 o p
  o=$(env ${ARGS[$n]/GPU=-/} $( [ "$c" = - ] && echo GPU=- || echo GPU=$c ) bash "$E" 2>&1) || { log "launch $n failed: $o"; ST[$n]=failed; return; }
  echo "$o" | sed 's/^/  /'
  p=$(echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}')
  [ -n "$p" ] || { log "launch $n: no pid"; ST[$n]=failed; return; }
  PID[$n]=$p; CARD[$n]=$c; ST[$n]=run
  if [ "$c" != - ]; then
    local now; now=$(date +%s)
    case ${KIND[$n]} in thinker) RESV[$c]="${RESV[$c]:-} 128000:$((now + 240))";; shard) RESV[$c]="${RESV[$c]:-} 16384:$((now + 150))";; esac
  fi
  log "started $n on card $c pid $p"
}

log "scheduler start: ${#order[@]} tasks"
while :; do
  for n in "${order[@]}"; do
    [ "${ST[$n]}" = run ] || continue
    kill -0 "${PID[$n]}" 2>/dev/null && continue
    if verify "$n"; then ST[$n]=done; log "done $n (card ${CARD[$n]})"; else ST[$n]=failed; log "FAILED $n (card ${CARD[$n]})"; fi
  done
  for n in "${order[@]}"; do
    [ "${ST[$n]}" = pending ] || continue
    case $(depstate "$n") in
      fail) ST[$n]=skipped; log "skip $n (dependency failed)"; continue;;
      wait) continue;;
    esac
    if [ "${KIND[$n]}" = cpu ]; then launch "$n" -; continue; fi
    c=$(pick "${KIND[$n]}")
    [ -n "$c" ] && launch "$n" "$c"
  done
  npend=0; nrun=0; nother=0
  for n in "${order[@]}"; do
    case ${ST[$n]} in pending) npend=$((npend + 1)); [ "${DEPS[$n]}" = W14U ] || nother=$((nother + 1));; run) nrun=$((nrun + 1));; esac
  done
  if [ $nrun = 0 ] && [ $nother = 0 ] && [ $npend -gt 0 ]; then
    for n in "${order[@]}"; do [ "${ST[$n]}" = pending ] && { ST[$n]=skipped; log "skip $n (wave 14 merged_unist not ready when all else was done)"; }; done
    npend=0
  fi
  [ $npend = 0 ] && [ $nrun = 0 ] && break
  sleep 15
done
touch "$STOP"
log "all GPU jobs finished; watcher STOP touched, cards back to the guard"
for n in "${order[@]}"; do echo "  $n ${ST[$n]} card ${CARD[$n]:--}"; done
for arm in enc_wise05 enc; do
  [ "${ST[asr_${arm}_en2zh]}" = done ] && [ "${ST[asr_${arm}_zh2en]}" = done ] && date > "$M/eval_$arm/EVAL_DONE"
done
for n in "${order[@]}"; do case $n in mech_*) [ "${ST[$n]}" = done ] || echo "$n ${ST[$n]}" >> "$M/MECH_FAILED";; esac; done
"$PYQ" experiments/mist_w3po_v1/voicecond/wave17_mech_score.py --out "$M/mech.json" > $LOGD/mech_score.log 2>&1 \
  && log "mechanism scoring done" || log "WARNING: mechanism scoring failed (see $LOGD/mech_score.log)"
[ -e "$M/eval_enc_wise05/EVAL_DONE" ] || fail "enc_wise05 evaluation"
[ -e "$M/eval_enc/EVAL_DONE" ] || log "WARNING: enc (alpha 1) evaluation incomplete; scoring without it"
log "waiting for wave 14 unist_wise05 dev outputs"
while [ ! -e "$W14/eval_unist_wise05/EVAL_DONE" ]; do
  [ -e "$W14/eval_unist_wise05/EVAL_FAILED" ] && fail "wave 14 unist_wise05 evaluation failed"
  [ -e "$W14/CHAIN_FAILED" ] && fail "wave 14 chain failed: $(cat $W14/CHAIN_FAILED)"
  sleep 60
done
log "wave 14 unist_wise05 EVAL_DONE; waiting up to 2 h for wave 14 CHAIN_DONE (summary.json cross-check)"
t0=$(date +%s)
while [ ! -e "$W14/CHAIN_DONE" ] && [ ! -e "$W14/CHAIN_FAILED" ] && [ $(( $(date +%s) - t0 )) -lt 7200 ]; do sleep 60; done
log "scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave17_score.py --out "$M/summary.json" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/CHAIN_DONE"
log "chain done"
