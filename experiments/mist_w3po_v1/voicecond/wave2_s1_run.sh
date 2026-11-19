#!/usr/bin/env bash
# Robust wave 2, diagnosis S1 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): runs the single-speaker
# baseline UNCHANGED -- the paper's LocalAgreement-2 (LA2) at a 1 s read window:
#   Thinker  experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --policy la2 --windows 1
#            (base pretrained_models/Qwen3-Omni-30B-A3B-Instruct, no adapter, greedy, prompt base)
#   Talker   experiments.uniss_qwen3_omni_v1.evaluation.si_infer --text-policy file --speed-policy block --window 1
#            --model eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss (T1c),
#            all other flags at their defaults (temperature 0.7, history ban off, speed shift 0); TTS seeds = default
#            (20261021) and SEED=1, the paper's seed naming.
#   ASR      reports/uniss_si_v4/speech_only.py -> evaluation.asr_transcribe --batch-size 4 (paper's speech-only ASR)
# Each call places ONE job (detached with setsid nohup, fds 8/9 closed) and appends "card pid name" to $PIDS, after
# stopping the placeholder supervisor + worker of that card by PID (argv re-checked; never the guard 2726951,
# gpu_watch 52045, sync 1222453 or anything else).
#   STAGE=thinker GPU=0 DIR=en2zh OUTD=<dir> [ITEMS=<items jsonl>] [LIMIT=60] [GPU_MEM=0.6] NAME=<n> bash wave2_s1_run.sh
#   STAGE=talker  GPU=0 DIR=en2zh OUTD=<dir> BLOCKS=<blocks jsonl> [ITEMS=..] [SEED=1] [SHARD=0 NSHARDS=1] NAME=<n> ...
#   STAGE=asr     GPU=0 RUN=<si_infer output dir with part_*.jsonl> NAME=<n> ...
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
LOGD=logs/mist_w3po_v1/wave2
PIDS=${PIDS:-$LOGD/s1_workers.pids}
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${STAGE:?}" "${GPU:?}" "${NAME:?}"
mkdir -p "$LOGD"
LOGF=$LOGD/s1_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }

case "$STAGE" in
  thinker)
    : "${DIR:?}" "${OUTD:?}"
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.6}" --out "$OUTD"
         ${ITEMS:+--items-file "$ITEMS"} ${LIMIT:+--limit "$LIMIT"}) ;;
  talker)
    : "${DIR:?}" "${OUTD:?}" "${BLOCKS:?}"
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer
         --model "$TALKER" --direction "$DIR" --window 1 --output "$OUTD" --shard "${SHARD:-0}" --num-shards "${NSHARDS:-1}"
         --text-policy file --blocks-file "$BLOCKS" --speed-policy block --gpu-mem 0.1
         ${ITEMS:+--items-file "$ITEMS"} ${SEED:+--seed "$SEED"}) ;;
  asr)
    : "${RUN:?}"
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE must be thinker, talker or asr" >&2; exit 1 ;;
esac

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

# stop the placeholder of this card (supervisor first, then its worker), by PID, argv re-checked
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
log "start $NAME on card $GPU: ${CMD[*]}" | tee -a "$LOGD/s1_launch.log"
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
log "$NAME pid $!" | tee -a "$LOGD/s1_launch.log"
