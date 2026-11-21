#!/usr/bin/env bash
# Robust K1 step 2 launcher (copy of r1_run.sh): one torchrun job of k1_train.py --noisy-only (tau_noisy', 6 noisy envs,
# UniST pilot data) on the cards GPUS, detached (setsid nohup, fds 8/9 closed); "GPUS pid name" goes to $PIDS for k1_noplaceholder.sh.
# The placeholders of these cards are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 /
# sync 1222453).
#   GPUS=1,2,3,7 STEPS=2300 OUT=<dir> NAME=<n> MB=16 [SMOKE=3] [PORT=29693] [WORKERS=12] PIDS=<file> bash k1_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPUS:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}" "${MB:?}"
LOGD=logs/mist_w3po_v1/k1/step2
mkdir -p "$LOGD"
: "${PIDS:?}"
LOGF=$LOGD/k1_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CMD=(env CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node "$NP" --master-port "${PORT:-29693}"
     experiments/mist_w3po_v1/voicecond/k1_train.py --arm erm --noisy-only --steps "$STEPS" --out "$OUT" --mb "$MB" --workers "${WORKERS:-12}"
     ${SMOKE:+--max-steps-run "$SMOKE"})
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PROTECT=" 2726951 52045 1222453 $$ "
for g in ${GPUS//,/ }; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && echo "SIGTERM placeholder $pid card $g";; esac
  done < <(ps -eo pid=,args=)
done
sleep 8
# wait until no gpu_load process remains on these cards (re-SIGTERM any respawn), at most 120 s (combo_orch.sh fix)
for k in $(seq 1 24); do
  left=""
  for g in ${GPUS//,/ }; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
  done
  [ -z "$left" ] && break
  sleep 5
done
[ -n "$left" ] && echo "WARNING: placeholders still present after 120 s:$left" >&2
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPUS $! $NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on cards $GPUS pid $!"
