#!/usr/bin/env bash
# Robust R13 chain (reports/mist_w3po_v1/robust_r13_fishermask_prereg.zh-CN.md, section 5), detached, cards 1,2,3,7 only:
#   own watcher (rx_noplaceholder.sh, own PIDS / STOP, POLL 1 s) -> r13_fisher.py (torchrun world 4) -> r13_merge.py (CPU;
#   dW = wave14/run_unist_wise05) -> Thinker si_thinker_stream la2 w1 --gpu-mem 0.85 on r1/data (en2zh card 1, zh2en
#   card 2) -> STOP -> rx_score.py (base / unist_wise05 / r1 / r10) -> r13/DONE (failure: r13/FAILED + STOP).
# Placeholders are stopped by PID only, then wait (<= 120 s) until none is left on the card. Never touches guard 2726951,
# gpu_watch 52045, sync 1222453. setsid nohup bash .../r13_chain.sh > logs/mist_w3po_v1/r13/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/mist_w3po_v1
M=$E/r13
LOGD=logs/mist_w3po_v1/r13
V=experiments/mist_w3po_v1/voicecond
export PIDS=$LOGD/r13_workers.pids
export STOP=$LOGD/r13_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then log "[dry] fisher $M/fisher (1,2,3,7) -> merge $M/merged_r13 -> eval $M/eval_r13 (cards 1,2) -> score"; exit 0; fi
mkdir -p "$M" "$LOGD"
[ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
for f in DONE FAILED fisher merged_r13 eval_r13 score.json; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
clear_card() {
  local g=$1 k left pid args
  for k in $(seq 1 25); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && return 0
    [ "$k" = 1 ] && log "SIGTERM placeholders on card $g:$left"
    sleep 5
  done
  log "WARNING: placeholders still on card $g after 120 s:$left"
}
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
log "r13 chain $$"
STOP=$STOP POLL=1 setsid nohup bash $V/rx_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
for g in 1 2 3 7; do clear_card $g; done
env CUDA_VISIBLE_DEVICES=1,2,3,7 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
    HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=. \
    setsid nohup "$PYQ" -m torch.distributed.run --nproc-per-node 4 --master-port 29709 $V/r13_fisher.py --out $M/fisher \
    > $LOGD/fisher.log 2>&1 < /dev/null 8>&- 9>&- &
TP=$!; echo "1,2,3,7 $TP r13_fisher" >> "$PIDS"; log "fisher torchrun pid $TP"
waitp $TP
[ -s "$M/fisher/fisher.pt" ] || fail "fisher (see $LOGD/fisher.log)"
log "fisher: $(cat $M/fisher/stats.json | tr -d '\n' | cut -c1-400)"
PYTHONPATH=. "$PYQ" $V/r13_merge.py --adapters $E/wave14/run_unist_wise05/adapters_final.pt --fisher $M/fisher/fisher.pt \
  --out $M/merged_r13 > $LOGD/merge.log 2>&1 || fail "merge (see $LOGD/merge.log)"
log "merged: $(tail -n 1 $LOGD/merge.log | cut -c1-500)"
thinker() {   # dir card ; prints pid
  local d=$1 g=$2 lf=$LOGD/thinker_r13_$1.log
  [ -e "$lf" ] && fail "log exists: $lf"
  clear_card "$g" >&2
  setsid nohup "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
    --thinker "$M/merged_r13" --direction "$d" --windows 1 --policy la2 --gpu-mem 0.85 --out "$M/eval_r13/thinker_$d" \
    --items-file "$E/r1/data/items_$d.jsonl" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p thinker_r13_$d" >> "$PIDS"
  log "thinker r13 $d card $g pid $p" >&2
  echo $p
}
p1=$(thinker en2zh 1); sleep 20; p2=$(thinker zh2en 2)
waitp $p1 $p2
for d in en2zh zh2en; do
  [ -s "$M/eval_r13/thinker_$d/blocks_${d}_w1.jsonl" ] || fail "no Thinker output $d (see $LOGD/thinker_r13_$d.log)"
done
touch "$STOP"
log "Thinker runs done; watcher STOP touched, cards back to the guard"
B="$E/r1/eval_{s}/thinker_{d}/blocks_{d}_w1.jsonl"
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.rx_score --out "$M/score.json" --method r13 --erm unist_wise05 \
  --sys "base=${B/\{s\}/base}" --sys "unist_wise05=${B/\{s\}/unist_wise05}" --sys "r1=${B/\{s\}/r1}" \
  --sys "r10=$E/r10/eval_r10/thinker_{d}/blocks_{d}_w1.jsonl" --sys "r13=$M/eval_r13/thinker_{d}/blocks_{d}_w1.jsonl" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
tail -n 3 $LOGD/score.log
date > "$M/DONE"
log "chain done"
