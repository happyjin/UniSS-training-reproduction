#!/usr/bin/env bash
# Robust K1 step 1 chain (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md, sections 3-4, 6), detached:
#   own watcher k1_noplaceholder.sh (POLL 1 s, own PIDS / STOP) -> K1 LoRA Thinker en2zh card 4 / zh2en card 5 ->
#   path check (tau_noisy alone as LoRA, k1/lora_unist) en2zh card 4 / zh2en card 5 -> STOP -> k1_score.py -> k1/DONE
#   (any failure: k1/FAILED + STOP). LoRAs already built (k1/lora_k1, k1/lora_unist).
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/k1_chain.sh > logs/mist_w3po_v1/k1/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; only cards 4, 5. Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/k1
R1=eval_outputs/mist_w3po_v1/r1
LOGD=logs/mist_w3po_v1/k1
export PIDS=$LOGD/k1_workers.pids
STOP=$LOGD/k1_noplaceholder.stop
V=experiments/mist_w3po_v1/voicecond
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
for f in DONE FAILED eval_k1 eval_unist_lora score.json; do
  [ -e "$M/$f" ] && { log "refuse: $M/$f exists"; exit 1; }
done
[ -e "$STOP" ] && { log "refuse: $STOP exists"; exit 1; }
for l in lora_k1 lora_unist/a1.0000; do [ -s "$M/$l/adapter_model.safetensors" ] || { log "missing $M/$l"; exit 1; }; done
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
log "chain $$"
STOP=$STOP POLL=1 setsid nohup bash $V/k1_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher started"
for job in "k1 $M/lora_k1" "unist_lora $M/lora_unist/a1.0000"; do
  read -r s lora <<< "$job"
  p1=$(NAME=$s LORA=$lora CARD=4 D=en2zh OUT=$M/eval_$s/thinker_en2zh bash $V/k1_thinker.sh) || fail "launch $s en2zh"
  sleep 20
  p2=$(NAME=$s LORA=$lora CARD=5 D=zh2en OUT=$M/eval_$s/thinker_zh2en bash $V/k1_thinker.sh) || fail "launch $s zh2en"
  waitp "$p1" "$p2"
  for d in en2zh zh2en; do
    [ -s "$M/eval_$s/thinker_$d/blocks_${d}_w1.jsonl" ] || fail "no Thinker output $s $d (see $LOGD/thinker_${s}_$d.log)"
  done
  log "$s done"
done
touch "$STOP"
log "Thinker runs done; watcher STOP touched, cards back to the guard"
T='thinker_{d}/blocks_{d}_w1.jsonl'
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.k1_score --out "$M/score.json" --method k1 --ref r1 \
  --check-lora unist_lora --check-merged unist \
  --sys base=$R1/eval_base/$T --sys unist=$R1/eval_unist/$T --sys unist_wise05=$R1/eval_unist_wise05/$T \
  --sys r1=$R1/eval_r1/$T --sys k1=$M/eval_k1/$T --sys unist_lora=$M/eval_unist_lora/$T > $LOGD/score.log 2>&1 \
  || fail "score (see $LOGD/score.log)"
cat $LOGD/score.log
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.rx_score --out "$M/score_rx.json" --method k1 --erm unist_wise05 \
  --sys base=$R1/eval_base/$T --sys unist_wise05=$R1/eval_unist_wise05/$T --sys k1=$M/eval_k1/$T > $LOGD/score_rx.log 2>&1 \
  || fail "rx score (see $LOGD/score_rx.log)"
tail -3 $LOGD/score_rx.log
date > "$M/DONE"
log "chain done"
