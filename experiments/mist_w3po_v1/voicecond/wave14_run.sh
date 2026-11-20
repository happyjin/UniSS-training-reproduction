#!/usr/bin/env bash
# Robust wave 14 launcher (copy of wave8_aul_run.sh): one torchrun ERM training job of wave14_train.py (UniST pilot data)
# on the cards GPUS, detached (setsid nohup, fds 8/9 closed); "GPUS pid name" goes to $PIDS for wave14_noplaceholder.sh.
# The placeholders of these cards are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 /
# sync 1222453).
#   GPUS=0,1,2,3,4,5,6,7 STEPS=2300 OUT=<dir> NAME=<n> MB=8 [SMOKE=10] [PORT=29641] bash wave14_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPUS:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}" "${MB:?}"
LOGD=logs/mist_w3po_v1/wave14
mkdir -p "$LOGD"
PIDS=${PIDS:-$LOGD/w14_workers.pids}
LOGF=$LOGD/w14_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CMD=(env CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node "$NP" --master-port "${PORT:-29641}"
     experiments/mist_w3po_v1/voicecond/wave14_train.py --arm erm --steps "$STEPS" --out "$OUT" --mb "$MB"
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
