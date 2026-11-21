#!/usr/bin/env bash
# Robust K1 (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md): one LoRA Thinker (k1_si_thinker.py over
# si_thinker_stream la2 w1, base Qwen3-Omni + LORA, --gpu-mem 0.85) on ONE card, detached (setsid nohup, fds 8/9 closed).
# Placeholders (scripts/gpu_load/) of the card are SIGTERMed by PID first and the card is waited on until none is left
# (<= 120 s, as combo_orch.sh launch()); "CARD pid NAME" goes to $PIDS for k1_noplaceholder.sh. Prints the pid.
#   NAME=<n> LORA=<peft dir> CARD=4 D=en2zh OUT=<dir> [LIMIT=3] [ITEMS=<jsonl>] PIDS=<file> bash k1_thinker.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
# Never touches guard 2726951, gpu_watch 52045, sync 1222453.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${NAME:?}" "${LORA:?}" "${CARD:?}" "${D:?}" "${OUT:?}" "${PIDS:?}"
LOGD=${LOGD:-logs/mist_w3po_v1/k1}
ITEMS=${ITEMS:-eval_outputs/mist_w3po_v1/r1/data/items_$D.jsonl}
LF=$LOGD/thinker_${NAME}_$D.log
[ -e "$LF" ] && { echo "拒绝覆盖已有日志: $LF" >&2; exit 1; }
[ -e "$OUT" ] && { echo "拒绝覆盖已有输出: $OUT" >&2; exit 1; }
CMD=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
     VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
     PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
     CUDA_VISIBLE_DEVICES=$CARD /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
     -m experiments.mist_w3po_v1.voicecond.k1_si_thinker --lora "$LORA"
     --thinker pretrained_models/Qwen3-Omni-30B-A3B-Instruct --direction "$D" --windows 1 --policy la2 --gpu-mem 0.85
     --out "$OUT" --items-file "$ITEMS" ${LIMIT:+--limit "$LIMIT"})
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PROTECT=" 2726951 52045 1222453 $$ "
for k in $(seq 1 25); do
  left=""
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $CARD "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
  done < <(ps -eo pid=,args=)
  [ -z "$left" ] && break
  [ "$k" = 1 ] && echo "$(TZ=Europe/Prague date '+%F %T %Z') SIGTERM placeholders on card $CARD:$left" >&2
  sleep 5
done
[ -n "$left" ] && echo "WARNING: placeholders still on card $CARD after 120 s:$left" >&2
setsid nohup "${CMD[@]}" > "$LF" 2>&1 < /dev/null 8>&- 9>&- &
p=$!
echo "$CARD $p thinker_${NAME}_$D" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') thinker $NAME $D card $CARD pid $p" >&2
echo "$p"
