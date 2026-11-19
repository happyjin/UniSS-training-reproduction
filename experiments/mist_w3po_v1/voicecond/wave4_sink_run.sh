#!/usr/bin/env bash
# Wave 4 diagnosis #3 launcher: one wave4_sink.py run (direction DIR) on card GPU, detached (setsid nohup, fds 8/9
# closed); "card pid name" -> logs/mist_w3po_v1/wave4/sink_workers.pids for the placeholder watcher
# (wave2_s1_noplaceholder.sh with PIDS / STOP set). The card's placeholder is stopped by PID (argv re-checked); never the
# guard 2726951, gpu_watch 52045 or sync 1222453.
#   GPU=2 DIR=zh2en [LIMIT=2] bash wave4_sink_run.sh        DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPU:?}" "${DIR:?}"
LOGD=logs/mist_w3po_v1/wave4
NAME=sink_${DIR}${LIMIT:+_smoke}
LOGF=$LOGD/$NAME.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
CMD=(env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
     CUDA_VISIBLE_DEVICES=$GPU /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -I
     experiments/mist_w3po_v1/voicecond/wave4_sink.py run --dir "$DIR" --out eval_outputs/mist_w3po_v1/wave4_diag/sink
     ${LIMIT:+--limit "$LIMIT"})
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PROTECT=" 2726951 52045 1222453 $$ "
while read -r pid args; do
  case "$PROTECT" in *" $pid "*) continue;; esac
  case "$args" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
  a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$a" in *scripts/gpu_load/*"--devices $GPU "*) kill -TERM "$pid" 2>/dev/null && echo "SIGTERM placeholder $pid card $GPU";; esac
done < <(ps -eo pid=,args=)
sleep 10
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$LOGD/sink_workers.pids"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on card $GPU pid $!"
