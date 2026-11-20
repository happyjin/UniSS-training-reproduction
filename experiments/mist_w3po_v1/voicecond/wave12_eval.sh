#!/usr/bin/env bash
# Robust wave 12 launcher (copy of wave11_eval.sh; own paths, logs and PIDS; only merge + Thinker): the ERM resume
# checkpoints at alpha = 1, Thinker-only on the (a) unit (reports/mist_w3po_v1/robust_wave12_overfit_diag_prereg.zh-CN.md).
#   STAGE=merge   GPU=-  CKPT=00500                 wave5_m_merge.py on wave12/ckpt_<CKPT>/adapters.pt (CPU)
#   STAGE=thinker GPU=0  CKPT=00500 DIR=en2zh       si_thinker_stream --thinker <merged> la2 w1 on items_<dir>_a.jsonl
# No Talker, no ASR. Root eval_outputs/mist_w3po_v1/wave12/eval_ckpt_<CKPT>/thinker_<dir>/. Detached (setsid nohup,
# fds 8/9 closed); "card pid name" -> $PIDS (wave12_noplaceholder.sh). Placeholders of the card are stopped by PID first.
# DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${STAGE:?}" "${GPU:?}" "${CKPT:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
LOGD=logs/mist_w3po_v1/wave12
PIDS=${PIDS:-$LOGD/w12_workers.pids}
M=eval_outputs/mist_w3po_v1/wave12
ROOT=$M/eval_ckpt_$CKPT
MERGED=$M/merged_ckpt_$CKPT
ITEMS=$M/data/items_${DIR:-x}_a.jsonl
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
mkdir -p "$LOGD"
case "$STAGE" in
  merge)
    NAME=merge_ckpt_$CKPT
    CMD=(env PYTHONPATH=. "$PYQ" experiments/mist_w3po_v1/voicecond/wave5_m_merge.py --adapters "$M/ckpt_$CKPT/adapters.pt"
         --out "$MERGED") ;;
  thinker)
    : "${DIR:?}"; NAME=thinker_ckpt_${CKPT}_$DIR
    mkdir -p "$ROOT"
    [ -e "$ROOT/data" ] || ln -s ../../wave2_S1/data "$ROOT/data"
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         --thinker "$MERGED" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}"
         --out "$ROOT/thinker_$DIR" --items-file "$ITEMS") ;;
  *) echo "STAGE: merge|thinker" >&2; exit 1 ;;
esac
LOGF=$LOGD/w12_eval_$NAME.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
if [ "$GPU" != "-" ]; then
  PROTECT=" 2726951 52045 1222453 $$ "
  for g in ${GPU//,/ }; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/run_mist_workload.sh*|*scripts/gpu_load/train_mist_workload.py*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && echo "SIGTERM placeholder $pid card $g";; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
fi
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "${GPU/-/none} $! $NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on card $GPU pid $!"
