#!/usr/bin/env bash
# Robust wave 17b launcher (copy of wave17_run.sh; CoVoST data via wave17b_enc_train.py): one torchrun ERM training job of wave17b_enc_train.py (encoder-only
# LoRA, wave 5 CoVoST data) on the cards GPUS, detached (setsid nohup, fds 8/9 closed); "GPUS pid name" goes to
# $PIDS for wave17_noplaceholder.sh. After the launch the pid's argv is re-checked (it must be the torchrun itself).
# The placeholders of these cards are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 /
# sync 1222453).
#   GPUS=6,7 STEPS=2300 OUT=<dir> NAME=<n> MB=32 [SMOKE=6] [PORT=29667] [WORKERS=12] bash wave17_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPUS:?}" "${STEPS:?}" "${OUT:?}" "${NAME:?}" "${MB:?}"
LOGD=logs/mist_w3po_v1/wave17b
mkdir -p "$LOGD"
PIDS=${PIDS:-$LOGD/w17b_workers.pids}
LOGF=$LOGD/w17b_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CMD=(env CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
     HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.
     /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node "$NP" --master-port "${PORT:-29667}"
     experiments/mist_w3po_v1/voicecond/wave17b_enc_train.py --arm erm --steps "$STEPS" --out "$OUT" --mb "$MB"
     --workers "${WORKERS:-12}"
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
P=$!
echo "$GPUS $P $NAME" >> "$PIDS"
sleep 3
a=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
case "$a" in *torch.distributed.run*wave17b_enc_train.py*) ;; *) echo "WARNING: pid $P argv is not the torchrun: '$a'" >&2;; esac
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on cards $GPUS pid $P"
