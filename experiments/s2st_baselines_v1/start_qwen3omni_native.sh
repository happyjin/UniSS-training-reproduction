#!/bin/bash
# Baseline (c) of paper/baselines_run_plan_v1.md: Qwen3-Omni native offline S2ST on CVSS-T test (both directions),
# split into NUM_SHARDS shards (qwen3omni_native_cvss.py, items[k::n] over en2zh+zh2en). One vLLM-Omni engine = the
# thinker on one card + its talker/code2wav on another (qwen3omni_native_split.yaml). Each engine worker takes the next
# unclaimed shard (claim file, noclobber) until none is left, so engines added later share the remaining shards.
#
#   ENGINES="4:5 6:7" bash experiments/s2st_baselines_v1/start_qwen3omni_native.sh       # thinker:talker card pairs
#   DRY_RUN=1 ...    prints what would be started (memory check, placeholder supervisors) and starts nothing
#
# Before an engine starts on a card: the card's train_mist_workload.py supervisor (placeholder) is stopped by PID after a
# cmdline check, and the card must have room: used + need <= CAP_MIB (thinker card NEED_T, talker card NEED_K; these
# are the real footprints, above the declared fractions). Engines start STAGGER s apart. Never overwrites a log.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
VENV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030
OUT=${OUT:-eval_outputs/s2st_baselines_v1/qwen3omni_native_cvss}
LOGD=${LOGD:-logs/s2st_baselines_v1/qwen3omni_native}
NUM_SHARDS=${NUM_SHARDS:-4}
SHARDS=${SHARDS:-"0 1 2 3"}
ENGINES=${ENGINES:-"4:5 6:7"}
STAGGER=${STAGGER:-60}
CAP_MIB=${CAP_MIB:-138240}      # 135 GiB a card
NEED_T=${NEED_T:-88000}         # thinker stage at 0.52 (75 GB declared) + context/cudagraph overhead
NEED_K=${NEED_K:-34000}         # talker 0.10 + code2wav 0.05 (21.5 GB declared) + two processes' overhead
DEPLOY=experiments/s2st_baselines_v1/qwen3omni_native_split.yaml
DRY_RUN=${DRY_RUN:-0}
ts() { TZ=Europe/Prague date +%H:%M:%S; }

used_mib() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

stop_placeholders() {   # $1 card: stop its placeholder supervisors (not workers) by PID after a cmdline check
  local k=$1 p cmd
  for p in $(ps -eo pid=,args= | grep "train_mist_workload.py" | grep -v -- "--worker" | grep -v grep \
             | awk -v k="$k" '{for(i=2;i<NF;i++) if($i=="--devices" && $(i+1)==k) {print $1; break}}'); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*"--devices $k "*) ;; *) echo "[$(ts)] card $k: $p not a placeholder supervisor; left"; continue ;; esac
    if [ "$DRY_RUN" = 1 ]; then echo "[$(ts)] card $k: would stop placeholder supervisor $p"; continue; fi
    kill -TERM "$p" 2>/dev/null && echo "[$(ts)] card $k: stopped placeholder supervisor $p"
    for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  done
}

room() {   # $1 card, $2 MiB needed -> 0 if it fits under CAP_MIB
  local u; u=$(used_mib "$1") || return 1
  [ $((u + $2)) -le "$CAP_MIB" ]
}

worker() {   # $1 thinker card, $2 talker card
  local T=$1 K=$2 k log
  mkdir -p "$OUT/claims" "$LOGD"
  for k in $SHARDS; do
    if ! ( set -o noclobber; echo "$T:$K $$ $(date -u +%FT%TZ)" > "$OUT/claims/shard_${k}of${NUM_SHARDS}" ) 2>/dev/null; then
      continue
    fi
    while true; do
      stop_placeholders "$T"; stop_placeholders "$K"
      if room "$T" "$NEED_T" && room "$K" "$NEED_K"; then break; fi
      echo "[$(ts)] engine $T:$K shard $k: no room yet (card $T $(used_mib "$T") MiB, card $K $(used_mib "$K") MiB); waiting"
      sleep 60
    done
    log="$LOGD/shard_${k}of${NUM_SHARDS}_e${T}${K}_$(date -u +%Y%m%dT%H%M%SZ).log"
    [ -e "$log" ] && { echo "refusing: $log exists"; return 1; }
    echo "[$(ts)] engine $T:$K starts shard $k -> $log"
    env CUDA_VISIBLE_DEVICES="$T,$K" PATH="$VENV/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 \
        VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. PYTHONUNBUFFERED=1 \
        TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
        "$VENV/bin/python" -m experiments.s2st_baselines_v1.qwen3omni_native_cvss --out "$OUT" \
        --shard "$k" --num-shards "$NUM_SHARDS" --deploy-config "$DEPLOY" > "$log" 2>&1
    echo "[$(ts)] engine $T:$K shard $k exit $? ($(grep -c . "$OUT"/cvss_*/part_${k}of${NUM_SHARDS}.jsonl 2>/dev/null | tr '\n' ' '))"
  done
  echo "[$(ts)] engine $T:$K: no unclaimed shard left; QWEN3OMNI_ENGINE_DONE"
}

if [ "${1:-}" = "__worker" ]; then worker "$2" "$3"; exit $?; fi

mkdir -p "$LOGD"
first=1
for e in $ENGINES; do
  T=${e%%:*}; K=${e##*:}
  if [ "$DRY_RUN" = 1 ]; then
    echo "engine $T:$K: card $T used $(used_mib "$T") MiB (+$NEED_T), card $K used $(used_mib "$K") MiB (+$NEED_K), cap $CAP_MIB"
    stop_placeholders "$T"; stop_placeholders "$K"
    continue
  fi
  [ $first = 1 ] || sleep "$STAGGER"
  first=0
  wl="$LOGD/worker_e${T}${K}_$(date -u +%Y%m%dT%H%M%SZ).log"
  setsid nohup bash "$0" __worker "$T" "$K" > "$wl" 2>&1 < /dev/null &
  echo "[$(ts)] worker for engine $T:$K pid $! log $wl"
done
