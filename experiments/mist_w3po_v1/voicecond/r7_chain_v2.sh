#!/usr/bin/env bash
# v2 (2026-11-21 00:40 CET): R4 already ended (FAILED at its self-check); v1 was stopped after its smoke because clean-env
# slots were selected through numerical noise (fixed in r7_train.py). Identical to v1 otherwise.
# Robust R7 chain (reports/mist_w3po_v1/robust_r7_noiseexcess_prereg.zh-CN.md, section 5), detached, cards 4,5 only:
#   wait until the R4 chain has finished (r4/DONE or r4/FAILED), no r4_* process is alive and the 2 cards hold no compute
#   process except scripts/gpu_load/ placeholders (3 checks, 30 s apart) -> own watcher (rx_noplaceholder.sh with own
#   PIDS / STOP, POLL 1 s) -> smoke 6 steps (mb 16, else 8) -> r7_train.py 2300 steps (world 2) -> WiSE 0.5 copy (alpha
#   16, hard-linked adapters) -> wave5_m_merge.py -> Thinker si_thinker_stream la2 w1 --gpu-mem 0.85 on r1/data
#   (en2zh card 4, zh2en card 5) -> STOP -> rx_score.py (base / unist_wise05 / r1 from r1/eval_*) -> r7/DONE
#   (any failure: r7/FAILED + STOP).
# Placeholders are stopped by PID only, then wait (<= 120 s) until none is left on the card. Never touches guard 2726951,
# gpu_watch 52045, sync 1222453. setsid nohup bash .../r7_chain.sh > logs/mist_w3po_v1/r7/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/mist_w3po_v1
M=$E/r7
LOGD=logs/mist_w3po_v1/r7
V=experiments/mist_w3po_v1/voicecond
export PIDS=$LOGD/r7_workers.pids
export STOP=$LOGD/r7_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
CARDS=4,5
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  log "[dry] wait r4 -> smoke $M/smoke16 -> train $M/run_r7 on $CARDS -> wise $M/run_r7_wise05 -> merge $M/merged_r7_wise05 -> eval $M/eval_r7_wise05 -> score"
  exit 0
fi
mkdir -p "$M" "$LOGD"
[ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
for f in DONE FAILED run_r7 run_r7_wise05 merged_r7_wise05 eval_r7_wise05 score.json; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
alive_r4() { ps -eo args= | grep -E "voicecond/r4_[a-z_]+\.(sh|py)" | grep -vq grep; }
card_clear() {
  local p mb a
  while IFS=', ' read -r p mb; do
    [ -n "$p" ] || continue
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*|'') ;; *) return 1;; esac
  done < <(nvidia-smi -i "$1" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  return 0
}
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
log "r7 chain $$: waiting for R4 (cards $CARDS)"
while :; do
  if { [ -e $E/r4/DONE ] || [ -e $E/r4/FAILED ]; } && ! alive_r4; then
    ok=1
    for k in 1 2 3; do
      for c in ${CARDS//,/ }; do card_clear $c || ok=0; done
      [ $ok = 1 ] || break
      [ $k = 3 ] || sleep 30
    done
    [ $ok = 1 ] && break
  fi
  sleep 60
done
log "R4 finished ($(ls $E/r4 | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')); cards $CARDS clear"
STOP=$STOP POLL=1 setsid nohup bash $V/rx_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
train() {   # out mb [smoke steps]
  for g in ${CARDS//,/ }; do clear_card $g; done
  env CUDA_VISIBLE_DEVICES=$CARDS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
      HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=. \
      setsid nohup "$PYQ" -m torch.distributed.run --nproc-per-node 2 --master-port 29699 $V/r7_train.py \
      --steps 2300 --out "$1" --mb "$2" --workers 12 ${3:+--max-steps-run $3} \
      > "$LOGD/r7_$(basename $1).log" 2>&1 < /dev/null 8>&- 9>&- &
  TP=$!; echo "$CARDS $TP r7_train_$(basename $1)" >> "$PIDS"; log "$(basename $1) (mb $2) torchrun pid $TP"
}
MB=""
for mb in 16 8; do
  train "$M/smoke$mb" $mb 6; waitp $TP
  n=$(wc -l < "$M/smoke$mb/metrics_rank0.jsonl" 2>/dev/null || echo 0)
  log "smoke mb $mb: $n steps; $(tail -n 1 $M/smoke$mb/metrics_rank0.jsonl 2>/dev/null | cut -c1-300)"
  [ "$n" -ge 6 ] && { MB=$mb; break; }
done
[ -n "$MB" ] || fail "smoke failed at mb 16 and 8"
train "$M/run_r7" $MB; waitp $TP
R=$M/run_r7
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt (see $LOGD/r7_run_r7.log)"
log "training done (mb $MB)"
W=$M/run_r7_wise05
mkdir -p "$W"; ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "r7" and m["alpha"] == 32.0 and m["rank"] == 16 and m["steps"] == 2300, m
m.update({"arm": "r7_wise05", "alpha": m["alpha"] / 2, "wise_ft": "W0 + 0.5*(W_r7 - W0): alpha halved"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
PY
PYTHONPATH=. "$PYQ" $V/wave5_m_merge.py --adapters "$W/adapters_final.pt" --out "$M/merged_r7_wise05" > $LOGD/merge.log 2>&1 \
  || fail "merge (see $LOGD/merge.log)"
log "merged: $(tail -n 1 $LOGD/merge.log | cut -c1-300)"
thinker() {   # dir card ; prints pid
  local d=$1 g=$2 lf=$LOGD/thinker_r7_wise05_$1.log
  [ -e "$lf" ] && fail "log exists: $lf"
  clear_card "$g" >&2
  setsid nohup "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
    --thinker "$M/merged_r7_wise05" --direction "$d" --windows 1 --policy la2 --gpu-mem 0.85 --out "$M/eval_r7_wise05/thinker_$d" \
    --items-file "$E/r1/data/items_$d.jsonl" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p thinker_r7_wise05_$d" >> "$PIDS"
  log "thinker r7_wise05 $d card $g pid $p" >&2
  echo $p
}
p1=$(thinker en2zh 4); sleep 20; p2=$(thinker zh2en 5)
waitp $p1 $p2
for d in en2zh zh2en; do
  [ -s "$M/eval_r7_wise05/thinker_$d/blocks_${d}_w1.jsonl" ] || fail "no Thinker output $d (see $LOGD/thinker_r7_wise05_$d.log)"
done
touch "$STOP"
log "Thinker runs done; watcher STOP touched, cards back to the guard"
B="$E/r1/eval_{s}/thinker_{d}/blocks_{d}_w1.jsonl"
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.rx_score --out "$M/score.json" --method r7_wise05 --erm unist_wise05 \
  --sys "base=${B/\{s\}/base}" --sys "unist_wise05=${B/\{s\}/unist_wise05}" --sys "r1=${B/\{s\}/r1}" \
  --sys "r5_wise05=$E/r5/eval_r5_wise05/thinker_{d}/blocks_{d}_w1.jsonl" --sys "r7_wise05=$M/eval_r7_wise05/thinker_{d}/blocks_{d}_w1.jsonl" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
tail -n 3 $LOGD/score.log
date > "$M/DONE"
log "chain done"
