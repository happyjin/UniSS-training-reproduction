#!/usr/bin/env bash
# Robust wave 8 AUL launcher (copy of wave5_m_run.sh): one torchrun training job of wave8_aul_train.py on the cards GPUS, detached
# (setsid nohup, fds 8/9 closed); "GPUS pid name" goes to $PIDS for wave8_aul_noplaceholder.sh. The placeholders of these
# cards are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 / sync 1222453).
#   GPUS=4,5,6,7 STEPS=2300 OUT=<dir> NAME=<n> [SMOKE=10] [PORT=29631] bash wave8_aul_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPUS:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}"
LOGD=logs/mist_w3po_v1/wave8
PIDS=${PIDS:-$LOGD/aul_workers.pids}
LOGF=$LOGD/aul_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CMD=(env CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node "$NP" --master-port "${PORT:-29631}"
     experiments/mist_w3po_v1/voicecond/wave8_aul_train.py --steps "$STEPS" --out "$OUT"
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
