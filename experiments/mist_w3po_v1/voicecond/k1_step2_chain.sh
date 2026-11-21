#!/usr/bin/env bash
# Robust K1 step 2 chain (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md, section 5), detached; launched only if
# step 1 passed (path check + rule):
#   own watcher k1_noplaceholder.sh (POLL 1 s, own PIDS / STOP) -> wait until cards 1,2,3,7 carry no compute process other
#   than scripts/gpu_load placeholders -> k1_run.sh tau_noisy' (k1_train.py --noisy-only, world 4, mb 16) -> wait ->
#   k1_make_lora.py (tau_noisy' - tau_clean, rank 32, fp32 check) -> LoRA Thinker en2zh card 4 / zh2en card 5 -> STOP ->
#   k1_score.py (k1p vs k1) -> k1/step2/DONE (any failure: k1/step2/FAILED + STOP).
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/k1_step2_chain.sh > logs/mist_w3po_v1/k1/step2/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; only cards 1, 2, 3, 7 (training) and 4, 5 (Thinker).
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
K=eval_outputs/mist_w3po_v1/k1
M=$K/step2
R1=eval_outputs/mist_w3po_v1/r1
LOGD=logs/mist_w3po_v1/k1/step2
export PIDS=$LOGD/k1s2_workers.pids
STOP=$LOGD/k1s2_noplaceholder.stop
V=experiments/mist_w3po_v1/voicecond
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
TRAIN_CARDS="1 2 3 7"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
mkdir -p "$M" "$LOGD"
for f in DONE FAILED run_noisy6 lora_k1p eval_k1p score.json; do
  [ -e "$M/$f" ] && { log "refuse: $M/$f exists"; exit 1; }
done
[ -e "$STOP" ] && { log "refuse: $STOP exists"; exit 1; }
[ -e "$K/DONE" ] || { log "refuse: step 1 not done"; exit 1; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done; }
busy_cards() {   # cards of TRAIN_CARDS with a compute process that is not a gpu_load placeholder
  declare -A idx
  while IFS=', ' read -r i u; do idx[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  local out="" u p a c
  while IFS=', ' read -r u p; do
    [ -n "${p:-}" ] || continue
    c=${idx[$u]:-}
    case " $TRAIN_CARDS " in *" $c "*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*) continue;; esac
    out="$out $c:$p"
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  echo "$out"
}
log "chain $$"
STOP=$STOP POLL=1 setsid nohup bash $V/k1_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher started"
n=0
while :; do
  b=$(busy_cards)
  [ -z "$b" ] && break
  [ $((n % 10)) = 0 ] && log "waiting for cards $TRAIN_CARDS: busy$b"
  n=$((n + 1)); sleep 60
done
log "cards $TRAIN_CARDS free of non-placeholder jobs"
GPUS=1,2,3,7 STEPS=2300 OUT=$M/run_noisy6 NAME=noisy6 MB=16 PORT=29693 WORKERS=12 bash $V/k1_run.sh > $LOGD/run_launch.log 2>&1 \
  || fail "k1_run.sh"
cat $LOGD/run_launch.log
TPID=$(awk '$3 == "noisy6" {p = $2} END {print p}' "$PIDS")
[ -n "$TPID" ] && kill -0 "$TPID" 2>/dev/null || fail "trainer pid not alive"
log "trainer torchrun pid $TPID"
waitp "$TPID"
[ -s "$M/run_noisy6/adapters_final.pt" ] && [ -s "$M/run_noisy6/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; building tau_noisy' - tau_clean LoRA"
PYTHONPATH=. "$PYQ" $V/k1_make_lora.py --noisy $M/run_noisy6/adapters_final.pt --clean $R1/run_clean/adapters_final.pt \
  --out $M/lora_k1p > $LOGD/make_lora.log 2>&1 || fail "make lora (see $LOGD/make_lora.log)"
cat $LOGD/make_lora.log
p1=$(LOGD=$LOGD NAME=k1p LORA=$M/lora_k1p CARD=4 D=en2zh OUT=$M/eval_k1p/thinker_en2zh bash $V/k1_thinker.sh) || fail "launch en2zh"
sleep 20
p2=$(LOGD=$LOGD NAME=k1p LORA=$M/lora_k1p CARD=5 D=zh2en OUT=$M/eval_k1p/thinker_zh2en bash $V/k1_thinker.sh) || fail "launch zh2en"
waitp "$p1" "$p2"
for d in en2zh zh2en; do
  [ -s "$M/eval_k1p/thinker_$d/blocks_${d}_w1.jsonl" ] || fail "no Thinker output k1p $d (see $LOGD/thinker_k1p_$d.log)"
done
touch "$STOP"
log "Thinker runs done; watcher STOP touched, cards back to the guard"
T='thinker_{d}/blocks_{d}_w1.jsonl'
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.k1_score --out "$M/score.json" --method k1p --ref k1 \
  --sys base=$R1/eval_base/$T --sys unist_wise05=$R1/eval_unist_wise05/$T --sys r1=$R1/eval_r1/$T \
  --sys k1=$K/eval_k1/$T --sys k1p=$M/eval_k1p/$T > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
cat $LOGD/score.log
date > "$M/DONE"
log "chain done"
