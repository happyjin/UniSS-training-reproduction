#!/usr/bin/env bash
# Robust wave 6 (reports/mist_w3po_v1/robust_wave6_disen_prereg.zh-CN.md): disentangled voice Talker vs T-ft.
#   STAGE=train  GPU=4,5 ARM=disen|tft OUTD=<dir> [SMOKE=1] [PORT=29561] NAME=<n> bash wave6_run.sh
#   STAGE=talker GPU=4 DIR=en2zh OUTD=<dir> MODEL=<exported talker dir> TABLE=<tokens jsonl> [SHARD=0 NSHARDS=4]
#                [ITEMS=<items jsonl>] NAME=<n> ...      (S1's flags, shards and seed 20261021, as wave3_v_run.sh)
#   STAGE=asr    GPU=4 RUN=<si_infer output dir> NAME=<n> ...
# Detached (setsid nohup, fds 8/9 closed); appends "card pid name" to $PIDS; stops the placeholders of the card(s) by
# PID (argv re-checked; never guard 2726951, gpu_watch 52045, sync 1222453). DRY_RUN=1 prints the command only.
# Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
LOGD=logs/mist_w3po_v1/wave6
PIDS=${PIDS:-$LOGD/workers.pids}
BLK=eval_outputs/mist_w3po_v1/wave2_S1
DATA=$BLK/data
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${STAGE:?}" "${GPU:?}" "${NAME:?}"
mkdir -p "$LOGD"
LOGF=$LOGD/${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }

case "$STAGE" in
  train)
    : "${ARM:?}" "${OUTD:?}"
    CMD=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
         HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface CUDA_VISIBLE_DEVICES=$GPU
         "$PYQ" -m torch.distributed.run --nproc_per_node 2 --master_port "${PORT:-29561}"
         experiments/mist_w3po_v1/voicecond/wave6_disen_train.py --arm "$ARM" --out "$OUTD" ${SMOKE:+--smoke}) ;;
  talker)
    : "${DIR:?}" "${OUTD:?}" "${MODEL:?}" "${TABLE:?}"
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo
         --voice-override "$TABLE"
         --model "$MODEL" --direction "$DIR" --window 1 --output "$OUTD" --shard "${SHARD:-0}" --num-shards "${NSHARDS:-4}"
         --text-policy file --blocks-file "$BLK/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "${ITEMS:-$DATA/items_$DIR.jsonl}" --seed 20261021) ;;
  asr)
    : "${RUN:?}"
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE must be train, talker or asr" >&2; exit 1 ;;
esac

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

PROTECT=" 2726951 52045 1222453 $$ "
for g in ${GPU//,/ }; do
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g: ${a:0:120}";; esac
    done < <(ps -eo pid=,args=)
  done
done
sleep 8
log "start $NAME on card $GPU: ${CMD[*]}" | tee -a "$LOGD/launch.log"
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
log "$NAME pid $!" | tee -a "$LOGD/launch.log"
