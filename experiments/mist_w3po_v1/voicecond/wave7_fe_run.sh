#!/usr/bin/env bash
# Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): launcher. Each call places ONE job, detached
# (setsid nohup, fds 8/9 closed), appends "card pid name" to $PIDS (for wave7_fe_noplaceholder.sh) after stopping the
# placeholders of that card by PID (argv re-checked; never guard 2726951, gpu_watch 52045, sync 1222453).
#   STAGE=smoke   GPU=0                                NAME=<n>  pseudo-target smoke (2 sentences per direction, smoke/ only)
#   STAGE=pseudo  GPU=0 WORKER=0 [MB=4]                 NAME=<n>  pseudo-target worker (claims units)
#   STAGE=train   GPU=0 ARM=pseudo|clean [STEPS=20000]   NAME=<n>  front-end training
#   STAGE=process GPU=0 ARM=pseudo|clean                 NAME=<n>  front-end on the S1 items
#   STAGE=rtf     GPU=0 ARM=pseudo                       NAME=<n>  real-time factor (exclusive card)
#   STAGE=thinker GPU=0 ARM=.. DIR=en2zh                 NAME=<n>  si_thinker_stream la2 w1 on the processed items
#   STAGE=talker  GPU=0 ARM=.. DIR=en2zh SHARD=k         NAME=<n>  si_infer + V1 C tokens, original items, 4 shards, seed 20261021
#   STAGE=asr     GPU=0 ARM=.. DIR=en2zh                 NAME=<n>  speech-only ASR of the arm's talker run
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
LOGD=logs/mist_w3po_v1/wave7
PIDS=${PIDS:-$LOGD/fe_workers.pids}
FE=eval_outputs/mist_w3po_v1/wave7/FE
S1D=eval_outputs/mist_w3po_v1/wave2_S1/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
QENV=(env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${STAGE:?}" "${GPU:?}" "${NAME:?}"
mkdir -p "$LOGD"
LOGF=$LOGD/fe_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }

case "$STAGE" in
  pseudo)
    : "${WORKER:?}"
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave7_fe_pseudo.py
         --worker "$WORKER" --device cuda:0 --mb "${MB:-4}") ;;
  smoke)
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave7_fe_pseudo.py
         --device cuda:0 --smoke 2) ;;
  train)
    : "${ARM:?}"
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave7_fe_train.py
         --arm "$ARM" --out "$FE/train_$ARM" --steps "${STEPS:-20000}") ;;
  process|rtf)
    : "${ARM:?}"
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave7_fe_eval.py "$STAGE"
         --arm "$ARM" --device cuda:0) ;;
  thinker)
    : "${ARM:?}" "${DIR:?}"
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}" --out "$FE/eval_$ARM/thinker_$DIR"
         --items-file "$FE/eval_$ARM/items_$DIR.jsonl") ;;
  talker)
    : "${ARM:?}" "${DIR:?}" "${SHARD:?}"
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo
         --voice-override "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens/tokens_v1_$DIR.jsonl"
         --model "$TALKER" --direction "$DIR" --window 1 --output "$FE/eval_$ARM/talker_$DIR" --shard "$SHARD" --num-shards 4
         --text-policy file --blocks-file "$FE/eval_$ARM/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block
         --gpu-mem 0.1 --items-file "$S1D/items_$DIR.jsonl" --seed 20261021) ;;
  asr)
    : "${ARM:?}" "${DIR:?}"
    RUN=$FE/eval_$ARM/talker_$DIR
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE must be smoke, pseudo, train, process, rtf, thinker, talker or asr" >&2; exit 1 ;;
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
log "start $NAME on card $GPU: ${CMD[*]}" | tee -a "$LOGD/fe_launch.log"
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
log "$NAME pid $!" | tee -a "$LOGD/fe_launch.log"
