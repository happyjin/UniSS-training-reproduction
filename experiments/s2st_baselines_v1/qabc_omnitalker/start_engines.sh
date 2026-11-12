#!/bin/bash
# QABC blocks spoken by Qwen3-Omni's native Talker (paper/baselines_run_plan_qabc_omnitalker_v1.md): start vLLM-Omni
# engines, each one synth.py process that loads the model once and claims shards of <ROOT>/jobs until none is left.
#
#   ENGINES="0:0 1:4" ROOT=eval_outputs/s2st_baselines_v1/qabc_omnitalker_realsi \
#       bash experiments/s2st_baselines_v1/qabc_omnitalker/start_engines.sh       # thinker:talker cards
#   T:T = all three stages on one card (qwen3omni_native_onecard.yaml, needs NEED_T + NEED_K there);
#   T:K = thinker on T, talker + code2wav on K (qwen3omni_native_split.yaml). Both yamls are the native CVSS-T run's.
#   DRY_RUN=1 ... prints the memory check and the placeholder supervisors it would stop; starts nothing.
#
# Before an engine starts: the cards' train_mist_workload.py supervisors (placeholders started by the guard) are
# stopped by PID after a cmdline check, and each card must have room (used + need <= CAP_MIB, real footprints). Never two
# engine starts at once: STAGGER s between starts (vLLM memory profiling). Never overwrites a log.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
VENV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030
ROOT=${ROOT:?ROOT=<work folder with jobs/>}
LOGD=${LOGD:-logs/s2st_baselines_v1/qabc_omnitalker}
ENGINES=${ENGINES:?ENGINES=\"T:K ...\"}
STAGGER=${STAGGER:-90}
CAP_MIB=${CAP_MIB:-138240}      # 135 GiB a card
NEED_T=${NEED_T:-88000}         # thinker 0.52 + overhead (native CVSS-T run)
NEED_K=${NEED_K:-34000}         # talker 0.10 + code2wav 0.05 + overhead
DRY_RUN=${DRY_RUN:-0}
ts() { TZ=Europe/Prague date +%H:%M:%S; }
used_mib() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

stop_placeholders() {   # $1 card
  local k=$1 p cmd
  for p in $(ps -eo pid=,args= | awk -v k="$k" '$0 ~ /scripts\/gpu_load\/train_mist_workload\.py/ && $0 !~ /--worker/ && $0 !~ /awk/ {for(i=2;i<NF;i++) if($i=="--devices" && $(i+1)==k) {print $1; break}}'); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*"--devices $k "*) ;; *) echo "[$(ts)] card $k: $p not a placeholder supervisor; left"; continue ;; esac
    case "$cmd" in *--worker*) echo "[$(ts)] card $k: $p is a worker; left"; continue ;; esac
    if [ "$DRY_RUN" = 1 ]; then echo "[$(ts)] card $k: would stop placeholder supervisor $p"; continue; fi
    kill -TERM "$p" 2>/dev/null && echo "[$(ts)] card $k: stopped placeholder supervisor $p"
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  done
}

room() { local u; u=$(used_mib "$1") || return 1; [ $((u + $2)) -le "$CAP_MIB" ]; }

mkdir -p "$LOGD"
first=1
for e in $ENGINES; do
  T=${e%%:*}; K=${e##*:}
  if [ "$T" = "$K" ]; then vis=$T; dep=experiments/s2st_baselines_v1/qwen3omni_native_onecard.yaml
  else vis="$T,$K"; dep=experiments/s2st_baselines_v1/qwen3omni_native_split.yaml; fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "engine $T:$K: card $T used $(used_mib "$T") MiB, card $K used $(used_mib "$K") MiB (need $NEED_T + $NEED_K), cap $CAP_MIB, $dep"
    stop_placeholders "$T"; [ "$K" = "$T" ] || stop_placeholders "$K"
    continue
  fi
  [ $first = 1 ] || sleep "$STAGGER"
  first=0
  tries=0
  while true; do
    stop_placeholders "$T"; [ "$K" = "$T" ] || stop_placeholders "$K"
    sleep 15   # let a stopped placeholder's memory go
    if [ "$T" = "$K" ]; then room "$T" $((NEED_T + NEED_K)) && break
    elif room "$T" "$NEED_T" && room "$K" "$NEED_K"; then break; fi
    tries=$((tries + 1))
    echo "[$(ts)] engine $T:$K: no room yet (card $T $(used_mib "$T") MiB, card $K $(used_mib "$K") MiB); waiting"
    [ $tries -ge 20 ] && { echo "[$(ts)] engine $T:$K: gave up after $tries checks"; continue 2; }
    sleep 45
  done
  log="$LOGD/engine_e${T}${K}_$(basename "$ROOT")_$(date -u +%Y%m%dT%H%M%SZ).log"
  [ -e "$log" ] && { echo "refusing: $log exists"; exit 1; }
  setsid nohup env CUDA_VISIBLE_DEVICES="$vis" PATH="$VENV/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 \
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
      "$VENV/bin/python" -m experiments.s2st_baselines_v1.qabc_omnitalker.synth --root "$ROOT" \
      --deploy-config "$dep" --tag "e$T$K" > "$log" 2>&1 < /dev/null &
  echo "[$(ts)] engine $T:$K pid $! log $log"
done
echo "[$(ts)] START_ENGINES_DONE"
