#!/usr/bin/env bash
# Robust wave 13 launcher (copy of wave12_eval.sh; own paths, logs and PIDS; merge + Thinker only), see
# reports/mist_w3po_v1/robust_wave13_cleandrop_diag_prereg.zh-CN.md:
#   STAGE=merge   GPU=-                              wave5_m_merge.py on wave10/run_zh2en/adapters_final.pt -> wave13/merged_dzh
#   STAGE=thinker GPU=0 SYS=erm|erm_wise05|dzh|base POLICY=oracle|la2 DIR=zh2en ITEMS=a|train600
#       si_thinker_stream --thinker <SYS model> --policy <POLICY> w1 on wave12 items_<dir>_a / wave13 items_zh2en_train600
#       -> eval_outputs/mist_w3po_v1/wave13/<POLICY>_<SYS>_<ITEMS>/thinker_<dir>/
# No Talker, no ASR. Detached (setsid nohup, fds 8/9 closed); "card pid name" -> $PIDS (wave13_noplaceholder.sh).
# Placeholders of the card are stopped by PID first. DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${STAGE:?}" "${GPU:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
LOGD=logs/mist_w3po_v1/wave13
PIDS=${PIDS:-$LOGD/w13_workers.pids}
E=eval_outputs/mist_w3po_v1
M=$E/wave13
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
mkdir -p "$LOGD"
case "$STAGE" in
  merge)
    NAME=merge_dzh
    CMD=(env PYTHONPATH=. "$PYQ" experiments/mist_w3po_v1/voicecond/wave5_m_merge.py
         --adapters "$E/wave10/run_zh2en/adapters_final.pt" --out "$M/merged_dzh") ;;
  thinker)
    : "${SYS:?}" "${POLICY:?}" "${DIR:?}" "${ITEMS:?}"
    case "$SYS" in
      base) TH=() ;;
      erm) TH=(--thinker "$E/wave5/M/merged_erm") ;;
      erm_wise05) TH=(--thinker "$E/wave5/M/merged_erm_wise05") ;;
      dzh) TH=(--thinker "$M/merged_dzh") ;;
      *) echo "SYS: base|erm|erm_wise05|dzh" >&2; exit 1 ;;
    esac
    case "$ITEMS" in
      a) IF=$E/wave12/data/items_${DIR}_a.jsonl ;;
      train600) IF=$M/data/items_${DIR}_train600.jsonl ;;
      *) echo "ITEMS: a|train600" >&2; exit 1 ;;
    esac
    [ -s "$IF" ] || { echo "no items file $IF" >&2; exit 1; }
    case "$POLICY" in oracle|la2) ;; *) echo "POLICY: oracle|la2" >&2; exit 1 ;; esac
    NAME=${POLICY}_${SYS}_${ITEMS}_$DIR
    OUTD=$M/${POLICY}_${SYS}_$ITEMS/thinker_$DIR
    [ -e "$OUTD" ] && { echo "拒绝覆盖已有输出: $OUTD" >&2; exit 1; }
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         "${TH[@]}" --direction "$DIR" --windows 1 --policy "$POLICY" --gpu-mem "${GPU_MEM:-0.85}"
         --out "$OUTD" --items-file "$IF") ;;
  *) echo "STAGE: merge|thinker" >&2; exit 1 ;;
esac
LOGF=$LOGD/w13_eval_$NAME.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
[ "$STAGE" = thinker ] && mkdir -p "$M/${POLICY}_${SYS}_$ITEMS"
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
