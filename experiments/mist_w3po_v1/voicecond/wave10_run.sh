#!/usr/bin/env bash
# Robust wave 10 launcher (copy of wave5_m_run.sh): one single-direction ERM LoRA (wave10_dir_train.py) on ONE card,
# world 1, detached (setsid nohup, fds 8/9 closed); "GPU pid name" goes to $PIDS for wave10_noplaceholder.sh. The
# placeholders of the card are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 / sync 1222453).
#   DIRN=zh2en GPU=2 STEPS=2300 OUT=<dir> NAME=<n> MB=32 [SMOKE=15] [PORT=29631] bash wave10_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${DIRN:?}" "${GPU:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}" "${MB:?}"
LOGD=logs/mist_w3po_v1/wave10
mkdir -p "$LOGD"
PIDS=${PIDS:-$LOGD/w10_workers.pids}
LOGF=$LOGD/w10_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
CMD=(env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 1 --master-port "${PORT:-29631}"
     experiments/mist_w3po_v1/voicecond/wave10_dir_train.py --direction "$DIRN" --steps "$STEPS" --out "$OUT" --mb "$MB"
     ${SMOKE:+--max-steps-run "$SMOKE"})
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PROTECT=" 2726951 52045 1222453 $$ "
for g in ${GPU//,/ }; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && echo "SIGTERM placeholder $pid card $g";; esac
  done < <(ps -eo pid=,args=)
done
sleep 10
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on card $GPU pid $!"
