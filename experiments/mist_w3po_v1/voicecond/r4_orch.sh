#!/usr/bin/env bash
# Robust R4 chain (reports/mist_w3po_v1/robust_r4_route_prereg.zh-CN.md, change 2), detached, on the two cards CA, CB
# handed over by r4_wait.sh (the pair of R2 = 4,5 or R3 = 0,6, whichever chain finished first). Never cards 1, 2, 3, 7 (R1).
#   1. D-U references, si_thinker_stream la2 w1, --gpu-mem 0.85: base / enc_wise05 / unist_wise05 (en2zh on CA, zh2en on
#      CB) -> r4/ref_<n>; then the self-check run r4_si_thinker --route base on D-U -> r4/sc_base
#   2. self-check: sc_base vs ref_base blocks-equal share >= 0.9035 (en2zh) / 0.9095 (zh2en), else stop (r4/FAILED)
#   3. R4e then R4w, --route witness, base on CA + adapter on CB, en2zh then zh2en, items = D-U + bench_dev (b)
#   4. r4_score.py -> r4/score.json ; r4/DONE
# Placeholders stopped by PID right before each launch, then wait (<= 120 s) until the card has no gpu_load process;
# own watcher r4_noplaceholder.sh (POLL 1 s; stops only scripts/gpu_load/ placeholders). Never touches guard 2726951,
# gpu_watch 52045, sync 1222453.
#   CA=4 CB=5 setsid nohup bash .../r4_orch.sh > logs/mist_w3po_v1/r4/orch.log 2>&1 < /dev/null 8>&- 9>&- &
# DRY_RUN=1 prints the commands. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${CA:?}" "${CB:?}"
for c in $CA $CB; do case " 0 4 5 6 " in *" $c "*) ;; *) echo "bad card $c (R4 may use only 0 4 5 6)" >&2; exit 1;; esac; done
DRY=${DRY_RUN:-}
M=eval_outputs/mist_w3po_v1/r4
LOGD=logs/mist_w3po_v1/r4
export PIDS=$LOGD/r4.pids
STOP=$LOGD/r4.stop
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
ENC=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
UNI=eval_outputs/mist_w3po_v1/wave14/merged_unist_wise05
GPU_MEM=0.85
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
stream() {   # card name thinker direction
  launch $1 ${2}_$4 "${VENV[@]}" CUDA_VISIBLE_DEVICES=$1 "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
    --thinker "$3" --direction $4 --windows 1 --policy la2 --gpu-mem $GPU_MEM --out $M/$2/thinker_$4 --items-file $M/data/items_devU_$4.jsonl
}
scbase() {   # card direction
  launch $1 sc_base_$2 "${VENV[@]}" CUDA_VISIBLE_DEVICES=$1 "$PYV" -m experiments.mist_w3po_v1.voicecond.r4_si_thinker \
    --route base --adapter $ENC --cards $1,$1 --gpu-mem $GPU_MEM --direction $2 --items-file $M/data/items_devU_$2.jsonl --out $M/sc_base/thinker_$2
}

if [ -z "$DRY" ]; then
  mkdir -p "$LOGD"
  for f in DONE FAILED score.json selfcheck.json ref_base ref_enc ref_uni sc_base R4e R4w; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
  [ -e "$STOP" ] && { echo "exists: $STOP" >&2; exit 1; }
  for d in en2zh zh2en; do [ -s $M/views/$d/DONE.json ] || { echo "D-U views $d missing" >&2; exit 1; }; done
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/r4_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher $!"
fi
log "r4 orch $$ on cards $CA (base / en2zh singles) and $CB (adapter / zh2en singles)"
# 1. references + self-check run
for pair in "ref_base $QWEN3" "ref_enc $ENC" "ref_uni $UNI"; do
  set -- $pair
  p1=$(stream $CA $1 $2 en2zh); p2=$(stream $CB $1 $2 zh2en)
  waitp $p1 $p2
done
p1=$(scbase $CA en2zh); p2=$(scbase $CB zh2en)
waitp $p1 $p2
[ -n "$DRY" ] || for n in ref_base ref_enc ref_uni sc_base; do for d in en2zh zh2en; do
  [ "$(wc -l < $M/$n/thinker_$d/blocks_${d}_w1.jsonl 2>/dev/null || echo 0)" = 600 ] || die "$n $d incomplete"; done; done
log "references and self-check run done"
# 2. self-check
if [ -z "$DRY" ]; then
  "$PYQ" - <<'EOF' > $LOGD/selfcheck.log 2>&1 || die "self-check FAIL (see $LOGD/selfcheck.log)"
import json, sys
M = "eval_outputs/mist_w3po_v1/r4"
THR = {"en2zh": 0.9035, "zh2en": 0.9095}
res, ok = {}, True
for d in THR:
    ld = lambda r: {x["id"]: json.dumps(x["blocks"], ensure_ascii=False)
                    for x in map(json.loads, open(f"{M}/{r}/thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8"))}
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
# 3. R4e, R4w (base on CA, adapter on CB)
for arm in "R4e $ENC" "R4w $UNI"; do
  set -- $arm
  for d in en2zh zh2en; do
    p=$(launch $CA,$CB ${1}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$CA,$CB "$PYV" -m experiments.mist_w3po_v1.voicecond.r4_si_thinker \
        --route witness --adapter "$2" --cards $CA,$CB --gpu-mem $GPU_MEM \
        --views $M/views/$d,eval_outputs/mist_w3po_v1/wave15/views/$d --direction $d --items-file $M/data/items_$d.jsonl \
        --out $M/$1/thinker_$d)
    waitp $p
    [ -n "$DRY" ] || [ -s $M/$1/thinker_$d/blocks_${d}_w1.r4.json ] || die "$1 $d"
  done
done
[ -n "$DRY" ] || touch "$STOP"
log "R4e / R4w done; watcher STOP touched, cards $CA $CB back to the guard"
# 4. score
[ -n "$DRY" ] || { PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.r4_score --out $M/score.json > $LOGD/score.log 2>&1 || die "score"; }
[ -n "$DRY" ] || date > $M/DONE
log "r4 done"
