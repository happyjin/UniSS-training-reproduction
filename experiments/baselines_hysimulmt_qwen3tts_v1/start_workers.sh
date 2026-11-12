#!/usr/bin/env bash
# Qwen3-TTS workers (tts_worker.py) on CARDS, PER_CARD workers per card, run under setsid nohup (each worker detached).
# On each card: the guard's placeholder supervisor (train_mist_workload.py --devices <card>, not --worker) is stopped by
# PID after a cmdline check, the card's memory is checked against the 135 GiB budget, and the first worker starts at
# once; further workers on the same card start 60 s apart. A card with a train_crf_sft process is skipped.
# Usage: CARDS="4 5 6 7" PER_CARD=3 [DRY_RUN=1] bash experiments/baselines_hysimulmt_qwen3tts_v1/start_workers.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
CARDS="${CARDS:?}"; PER_CARD="${PER_CARD:-3}"; DRY_RUN="${DRY_RUN:-0}"; RESERVE="${RESERVE:-10}"
BUDGET_MIB=$(( (135 - 5) * 1024 )); NEED_MIB=$(( (RESERVE + 3) * 1024 ))
L=logs/baselines_hysimulmt_qwen3tts; mkdir -p "$L"
JOBS=eval_outputs/baselines_hysimulmt_qwen3tts_v1/a1/jobs.jsonl
PY=/opt/dlami/nvme/jasonleeeli/venvs/tts_schemes/bin/python
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*" | tee -a "$L/start_workers.log"; }

supervisors() {   # placeholder supervisors for card $1 (ps fields, no pattern on our own shell)
  ps -eo pid=,args= | awk -v k="$1" '$0 ~ /train_mist_workload\.py/ && $0 !~ /--worker/ {
    for (i = 2; i < NF; i++) if ($i == "--devices" && $(i + 1) == k) { print $1; break } }'
}

mem_used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

has_training() {   # a train_crf_sft compute process on card $1
  local p
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q train_crf_sft && return 0
  done
  return 1
}

launch() {   # $1 card, $2 worker index
  local k=$1 w=$2 out="$L/a1${TAG:-}_worker_c$1_w$2.log"
  if [ -e "$out" ]; then log "card $k worker $w: $out exists; not started"; return 1; fi
  if [ "$DRY_RUN" = 1 ]; then log "card $k worker $w: would start (mem $(mem_used "$k") MiB)"; return 0; fi
  CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
    setsid nohup "$PY" experiments/baselines_hysimulmt_qwen3tts_v1/tts_worker.py "$JOBS" --reserve-gib "$RESERVE" --max-gib "${MAX_GIB:-0}" \
    > "$out" 2>&1 < /dev/null &
  log "card $k worker $w: started pid $! (mem before $(mem_used "$k") MiB)"
}

first_on_card() {   # stop the placeholder of card $1, then start worker 1
  local k=$1 s cmd m
  if has_training "$k"; then log "card $k: train_crf_sft present; skipped"; return 1; fi
  for s in $(supervisors "$k"); do
    cmd=$(tr '\0' ' ' < "/proc/$s/cmdline" 2>/dev/null)
    case "$cmd" in *train_mist_workload.py*"--devices $k "*) ;; *) log "card $k: $s cmdline mismatch; skipped card"; return 1 ;; esac
    case "$cmd" in *--worker*) log "card $k: $s is a worker; skipped card"; return 1 ;; esac
    if [ "$DRY_RUN" = 1 ]; then log "card $k: would stop placeholder supervisor $s"; continue; fi
    kill -TERM "$s"; log "card $k: stopped placeholder supervisor $s"
  done
  [ "$DRY_RUN" = 1 ] || for _ in $(seq 1 60); do m=$(mem_used "$k"); [ "$m" -lt $((BUDGET_MIB - NEED_MIB * PER_CARD)) ] && break; sleep 1; done
  m=$(mem_used "$k")
  if [ "$DRY_RUN" != 1 ] && [ "$m" -ge $((BUDGET_MIB - NEED_MIB)) ]; then log "card $k: $m MiB used, no room; skipped"; return 1; fi
  launch "$k" 1
}

OK=""
for k in $CARDS; do first_on_card "$k" && OK="$OK $k"; done
for w in $(seq 2 "$PER_CARD"); do
  [ "$DRY_RUN" = 1 ] || sleep 60
  for k in $OK; do
    m=$(mem_used "$k")
    if [ "$m" -ge $((BUDGET_MIB - NEED_MIB)) ]; then log "card $k: $m MiB used; worker $w not started"; continue; fi
    launch "$k" "$w"
  done
done
log "START_WORKERS_DONE cards [$OK]"
