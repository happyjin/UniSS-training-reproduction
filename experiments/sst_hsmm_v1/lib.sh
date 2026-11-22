# sst_hsmm_v1 shared shell helpers (sourced): log / die / launch (cards 0-7; placeholders stopped by PID) / waitp.
# Needs LOGD, PIDS, STOP, R set by the caller. Never touches guard 2726951, gpu_watch 52045, sync 1222453.
PROTECT=" 2726951 52045 1222453 $$ "
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
QENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "${DRY:-}" ] || { mkdir -p "$R"; touch "$R/FAILED"; touch "$STOP"; }; exit 1; }
free_card() {   # stop the placeholders of one card by PID
  local g=$1 k left pid args a
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
  for k in $(seq 1 24); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && break
    sleep 5
  done
  [ -n "$left" ] && log "WARNING: placeholders still on card $g after 120 s:$left" >&2
  return 0
}
launch() {   # cards(comma) name cmd... ; prints pid
  local gs=$1 name=$2; shift 2
  local lf=$LOGD/$name.log g
  if [ -n "${DRY:-}" ]; then printf '[dry] cards %s %s: ' "$gs" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  for g in ${gs//,/ }; do case " 0 1 2 3 4 5 6 7 " in *" $g "*) ;; *) die "card $g not allowed";; esac; free_card "$g"; done
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$gs $p $name" >> "$PIDS"
  log "$name cards $gs pid $p" >&2
  echo $p
}
waitp() { local p; for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }
start_watcher() {
  [ -n "${DRY:-}" ] && return 0
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 PIDS=$PIDS STOP=$STOP setsid nohup bash experiments/sst_hsmm_v1/noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher $!"
}
