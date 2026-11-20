#!/usr/bin/env bash
# Robust wave 9 launcher (copy of wave5_m_run.sh, wave-9 paths, MB): one torchrun training job of wave9_train.py on the cards GPUS, detached
# (setsid nohup, fds 8/9 closed); "GPUS pid name" goes to $PIDS for wave9_noplaceholder.sh. The placeholders of these
# cards are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 / sync 1222453).
#   ARM=A|C GPUS=0,1 STEPS=2300 OUT=<dir> NAME=<n> [MB=16] [SMOKE=6] [PORT=29641] bash wave9_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${GPUS:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}"
LOGD=logs/mist_w3po_v1/wave9
PIDS=${PIDS:-$LOGD/w9_workers.pids}
LOGF=$LOGD/w9_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CMD=(env CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node "$NP" --master-port "${PORT:-29641}"
     experiments/mist_w3po_v1/voicecond/wave9_train.py --arm "$ARM" --steps "$STEPS" --out "$OUT" --mb "${MB:-8}"
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
sleep 10
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPUS $! $NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on cards $GPUS pid $!"
