#!/usr/bin/env bash
# Wave 3 LA2 commit diagnosis (reports/mist_w3po_v1/robust_wave3_la2diag_prereg.zh-CN.md), D-GPU: the S1 LA2 Thinker
# (base Qwen3-Omni, window 1 s) with --policy oracle on the S1 items, one job per direction, detached (setsid nohup,
# fds 8/9 closed); "card pid name" goes to $PIDS for the placeholder watcher (wave2_s1_noplaceholder.sh with PIDS/STOP).
#   GPU=6 DIR=zh2en bash wave3_d_oracle.sh      DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPU:?}" "${DIR:?}"
LOGD=logs/mist_w3po_v1/wave3
PIDS=$LOGD/d_workers.pids
OUTD=eval_outputs/mist_w3po_v1/wave3_combos/D/oracle_$DIR
LOGF=$LOGD/d_oracle_$DIR.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
[ -e "$OUTD" ] && { echo "拒绝覆盖已有输出: $OUTD" >&2; exit 1; }
CMD=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
     VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
     TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface CUDA_VISIBLE_DEVICES=$GPU
     /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
     --direction "$DIR" --windows 1 --policy oracle --gpu-mem 0.85 --out "$OUTD"
     --items-file eval_outputs/mist_w3po_v1/wave2_S1/data/items_$DIR.jsonl)
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
echo "$GPU $! oracle_$DIR" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') oracle_$DIR on card $GPU pid $!"
