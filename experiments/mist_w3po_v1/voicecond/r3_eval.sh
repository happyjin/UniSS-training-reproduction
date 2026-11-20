#!/usr/bin/env bash
# R3 (reports/mist_w3po_v1/robust_r3_noisesub_prereg.zh-CN.md, section 5): one Thinker-only evaluation job, detached.
#   GPU=0 SYS=r3_wise05|base|enc_wise05 DSET=devU|s1 DIR=en2zh|zh2en bash r3_eval.sh
# si_thinker_stream la2 w1 (same flags as wave 17) -> eval_outputs/mist_w3po_v1/r3/eval_<SYS>/<DSET>/thinker_<DIR>/.
# Placeholders of the card are stopped by PID first, then wait (<= 120 s) until no gpu_load process is left on the card.
# "card pid name" -> $PIDS (r3_noplaceholder.sh). Prints "... pid <P>". DRY_RUN=1 prints the command. bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${GPU:?}" "${SYS:?}" "${DSET:?}" "${DIR:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
LOGD=logs/mist_w3po_v1/r3
PIDS=${PIDS:-$LOGD/r3_workers.pids}
M=eval_outputs/mist_w3po_v1/r3
case "$SYS" in
  r3_wise05) TH=(--thinker "$M/merged_r3_wise05") ;;
  base) TH=() ;;
  enc_wise05) TH=(--thinker eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05) ;;
  *) echo "SYS: r3_wise05|base|enc_wise05" >&2; exit 1 ;;
esac
case "$DSET" in
  devU) ITEMS=eval_outputs/mist_w3po_v1/devU/data/items_$DIR.jsonl ;;
  s1) ITEMS=eval_outputs/mist_w3po_v1/wave2_S1/data/items_$DIR.jsonl ;;
  *) echo "DSET: devU|s1" >&2; exit 1 ;;
esac
OUTD=$M/eval_$SYS/$DSET/thinker_$DIR
NAME=thinker_${SYS}_${DSET}_$DIR
LOGF=$LOGD/r3_eval_$NAME.log
[ -e "$OUTD" ] && { echo "拒绝覆盖已有输出: $OUTD" >&2; exit 1; }
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
     "${TH[@]}" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}" --out "$OUTD" --items-file "$ITEMS")
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$LOGD" "$(dirname "$OUTD")"
PROTECT=" 2726951 52045 1222453 $$ "
# coordinator: R4 (vLLM) stacks on cards 0 / 6 -- before a 0.85 Thinker wait (<= 3 h) until no non-placeholder compute
# process of another job is on the card (R4 / other evals are never stopped)
for k in $(seq 1 1080); do
  busy=""
  for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*) ;; "") ;; *) busy="$busy $p";; esac
  done
  [ -z "$busy" ] && break
  [ $((k % 60)) = 1 ] && echo "$(TZ=Europe/Prague date '+%T') card $GPU busy (pids$busy); waiting"
  sleep 10
done
for k in $(seq 1 24); do
  left=""
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/*"--devices $GPU "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null && echo "SIGTERM placeholder $pid card $GPU";; esac
  done < <(ps -eo pid=,args=)
  [ -z "$left" ] && break
  sleep 5
done
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
P=$!
echo "$GPU $P $NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on card $GPU pid $P"
