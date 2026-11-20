#!/usr/bin/env bash
# Robust R4 chain (reports/mist_w3po_v1/robust_r4_route_prereg.zh-CN.md), detached. Coordinator 21:58 CET (user): stacked
# on the R2 / R3 TRAINING cards 4, 5 (R2) and 0, 6 (R3), with lowered vLLM memory; cards 1, 2, 3, 7 untouched.
#   0. wait until every card of 0 4 5 6 holds a training process (non gpu_load, >= 20 GiB) for 3 min; peak = max used
#      memory over those 3 min; vLLM fraction per card = (total - peak - 10 GiB) / total; if that is below NEED (weights
#      59.4 GiB + graphs + >= 4 GiB KV = 66 GiB) on any card -> r4/NOFIT, exit (report, never force); D-U views DONE.json
#   1. D-U references, si_thinker_stream la2 w1: base / enc_wise05 / unist_wise05 x {en2zh, zh2en} -> r4/ref_<n>
#   2. self-check r4_si_thinker --route base on D-U vs ref_base (blocks-equal share >= 0.9035 / 0.9095) -> else stop
#   3. R4e then R4w, --route witness: en2zh base card 4 + adapter card 0 | zh2en base card 5 + adapter card 6 (parallel),
#      items = D-U + bench_dev (b)
#   4. r4_score.py -> r4/score.json ; r4/DONE (failure: r4/FAILED)
# Placeholders stopped by PID right before each launch, then wait (<= 120 s) until the card has no gpu_load process;
# own watcher r4_noplaceholder.sh (POLL 1 s). Never touches guard 2726951, gpu_watch 52045, sync 1222453;
# The watcher stops only scripts/gpu_load/ placeholders, never the R2 / R3 trainers.   setsid nohup bash .../r4_orch.sh > logs/mist_w3po_v1/r4/orch.log 2>&1 < /dev/null 8>&- 9>&- &
# DRY_RUN=1 prints the commands. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
M=eval_outputs/mist_w3po_v1/r4
LOGD=logs/mist_w3po_v1/r4
export PIDS=$LOGD/r4.pids
STOP=$LOGD/r4.stop
TR=eval_outputs/mist_w3po_v1/test_robust
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
ENC=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
UNI=eval_outputs/mist_w3po_v1/wave14/merged_unist_wise05
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "STOP: $*"; [ -n "$DRY" ] && return 0; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
PROTECT=" 2726951 52045 1222453 $$ "
clear_card() {   # stop placeholders of card $1 by PID, wait until none is left (<= 120 s)
  local g=$1 k left
  for k in $(seq 1 25); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && return 0
    [ "$k" = 1 ] && log "SIGTERM placeholders card $g:$left" >&2
    sleep 5
  done
  log "WARNING: placeholders still on card $g:$left" >&2
}
launch() {   # cards(comma) name cmd... ; prints pid
  local cs=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] cards %s %s: ' "$cs" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  for g in ${cs//,/ }; do clear_card "$g"; done
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$cs $p $name" >> "$PIDS"
  log "$name cards $cs pid $p" >&2
  echo $p
}
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }

if [ -z "$DRY" ]; then
  mkdir -p "$LOGD"
  for f in DONE FAILED NOFIT score.json ref_base ref_enc ref_uni sc_base R4e R4w; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
  [ -e "$STOP" ] && { echo "exists: $STOP" >&2; exit 1; }
fi
log "r4 orch $$"
# 0. wait for the trainers, measure their peak, set the vLLM fraction per card
CARDS4="0 4 5 6"
declare -A GM
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do until [ -s $M/views/$d/DONE.json ]; do sleep 20; done; done
  t0=$(date +%s)
  while :; do
    ok=1
    for g in $CARDS4; do
      n=$(nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r p mb; do
            a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$a" in *scripts/gpu_load/*|'') ;; *) [ "$mb" -ge 20480 ] && echo x;; esac; done | wc -l)
      [ "$n" -ge 1 ] || ok=0
    done
    [ $ok = 1 ] && break
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "no trainers on all of $CARDS4 after 90 min" > $M/NOFIT; log "NOFIT: no trainers"; exit 1; }
    sleep 30
  done
  log "trainers present on $CARDS4; sampling peak memory for 180 s"
  declare -A PK
  for k in $(seq 1 36); do
    for g in $CARDS4; do
      u=$(nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r p mb; do
            a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$a" in *scripts/gpu_load/*) ;; *) echo "$mb";; esac; done | awk '{s+=$1} END {print s+0}')
      [ "$u" -gt "${PK[$g]:-0}" ] && PK[$g]=$u
    done
    sleep 5
  done
  nofit=""
  for g in $CARDS4; do
    tot=$(nvidia-smi -i $g --query-gpu=memory.total --format=csv,noheader,nounits)
    GM[$g]=$(python3 -c "t=$tot; p=${PK[$g]}; f=(t-p-10240)/t; print(f'{int(f*100)/100:.2f}')")
    avail=$(( tot - PK[$g] - 10240 ))
    log "card $g: total $tot MiB, trainer peak ${PK[$g]} MiB, vLLM fraction ${GM[$g]} ($avail MiB)"
    [ "$avail" -ge 67584 ] || nofit="$nofit card $g: trainer peak ${PK[$g]} MiB leaves $avail MiB < 66 GiB;"
  done
  if [ -n "$nofit" ]; then echo "$nofit" > $M/NOFIT; log "NOFIT:$nofit"; exit 1; fi
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/r4_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher $!"
else
  for g in $CARDS4; do GM[$g]=0.48; done
fi
# 1. D-U references + 2. self-check: single-engine jobs, 4 at a time on cards 4 0 5 6
single() {   # card name thinker direction outroot [r4]
  local g=$1 n=$2 th=$3 d=$4 o=$5 kind=${6:-stream}
  if [ "$kind" = stream ]; then
    launch $g ${n}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
      --thinker "$th" --direction $d --windows 1 --policy la2 --gpu-mem ${GM[$g]} --out $o/thinker_$d --items-file $M/data/items_devU_$d.jsonl
  else
    launch $g ${n}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.r4_si_thinker \
      --route base --adapter $ENC --cards $g,$g --gpu-mem ${GM[$g]} --direction $d --items-file $M/data/items_devU_$d.jsonl --out $o/thinker_$d
  fi
}
ps_=($(single 4 ref_base $QWEN3 en2zh $M/ref_base) $(single 5 ref_base $QWEN3 zh2en $M/ref_base)
     $(single 0 ref_enc $ENC en2zh $M/ref_enc) $(single 6 ref_enc $ENC zh2en $M/ref_enc))
waitp "${ps_[@]}"
ps_=($(single 4 ref_uni $UNI en2zh $M/ref_uni) $(single 5 ref_uni $UNI zh2en $M/ref_uni)
     $(single 0 sc_base - en2zh $M/sc_base r4) $(single 6 sc_base - zh2en $M/sc_base r4))
waitp "${ps_[@]}"
[ -n "$DRY" ] || for n in ref_base ref_enc ref_uni sc_base; do for d in en2zh zh2en; do
  [ "$(wc -l < $M/$n/thinker_$d/blocks_${d}_w1.jsonl 2>/dev/null || echo 0)" = 600 ] || die "$n $d"; done; done
log "references and self-check runs done"
if [ -z "$DRY" ]; then
  "$PYQ" - <<'EOF' > $LOGD/selfcheck.log 2>&1 || die "self-check FAIL (see $LOGD/selfcheck.log)"
import json, sys
M = "eval_outputs/mist_w3po_v1/r4"
THR = {"en2zh": 0.9035, "zh2en": 0.9095}
res, ok = {}, True
for d in THR:
    ld = lambda r: {x["id"]: json.dumps(x["blocks"], ensure_ascii=False) for x in map(json.loads, open(f"{M}/{r}/thinker_{d}/blocks_{d}_w1.jsonl"))}
    a, b = ld("sc_base"), ld("ref_base")
    assert set(a) == set(b)
    s = sum(a[i] == b[i] for i in a) / len(a)
    res[d] = {"blocks_equal": round(s, 4), "threshold": THR[d], "PASS": s >= THR[d]}
    ok &= res[d]["PASS"]
json.dump(res, open(f"{M}/selfcheck.json", "x"), indent=1)
print(json.dumps(res))
sys.exit(0 if ok else 1)
EOF
  log "self-check PASS: $(cat $LOGD/selfcheck.log)"
fi
# 3. R4e, R4w: en2zh on (4 base, 0 adapter) | zh2en on (5 base, 6 adapter), in parallel
for arm in "R4e $ENC" "R4w $UNI"; do
  set -- $arm; a=$1; ad=$2
  p1=$(launch 4,0 ${a}_en2zh "${VENV[@]}" CUDA_VISIBLE_DEVICES=4,0 "$PYV" -m experiments.mist_w3po_v1.voicecond.r4_si_thinker \
       --route witness --adapter "$ad" --cards 4,0 --gpu-mem $(python3 -c "print(min(${GM[4]}, ${GM[0]}))") \
       --views $M/views/en2zh,eval_outputs/mist_w3po_v1/wave15/views/en2zh --direction en2zh --items-file $M/data/items_en2zh.jsonl --out $M/$a/thinker_en2zh)
  p2=$(launch 5,6 ${a}_zh2en "${VENV[@]}" CUDA_VISIBLE_DEVICES=5,6 "$PYV" -m experiments.mist_w3po_v1.voicecond.r4_si_thinker \
       --route witness --adapter "$ad" --cards 5,6 --gpu-mem $(python3 -c "print(min(${GM[5]}, ${GM[6]}))") \
       --views $M/views/zh2en,eval_outputs/mist_w3po_v1/wave15/views/zh2en --direction zh2en --items-file $M/data/items_zh2en.jsonl --out $M/$a/thinker_zh2en)
  waitp $p1 $p2
  [ -n "$DRY" ] || for d in en2zh zh2en; do [ -s $M/$a/thinker_$d/blocks_${d}_w1.r4.json ] || die "$a $d"; done
done
touch "$STOP"
log "R4e / R4w done; watcher STOP touched"
# 4. score
[ -n "$DRY" ] || { PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.r4_score --out $M/score.json > $LOGD/score.log 2>&1 || die "score"; }
[ -n "$DRY" ] || date > $M/DONE
log "r4 done"
