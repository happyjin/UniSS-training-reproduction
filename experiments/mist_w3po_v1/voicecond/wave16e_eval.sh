#!/usr/bin/env bash
# ncwi_erm (change B): sed copy of wave16_eval.sh with root wave16/erm, logs wave16/erm, PIDS w16e_workers.pids.
# Robust wave 16 NCWI evaluation launcher (copy of wave14_eval.sh; v2 16:25 CET: cards 4-5 allowed for stacking; own paths, logs and PIDS;
# reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md sections 3-4):
#   STAGE=thinker_lora GPU=0 ROOTN=eval_ncwi|eval_lora05 DIR=en2zh [CONST=0.5]
#         wave16_si_thinker_ncwi (base Qwen3-Omni + per-request LoRA from fit/assign_<dir>.jsonl) -> <root>/thinker_<dir>
#   STAGE=merge   GPU=-  ALPHA=0.3333        adapters_a<alpha>/ (hard link of run_unist adapters + json alpha 32*alpha)
#                                             -> merged_a<alpha> (CPU, wave5_m_merge.py)
#   STAGE=thinker_merged GPU=0 DIR=en2zh ALPHA=0.3333 MODEL=<dir> ITEMS=<subset jsonl>
#         si_thinker_stream --thinker MODEL on the items of one alpha group -> eval_ncwi_merged/parts/thinker_<dir>_a<alpha>
#   STAGE=talker  GPU=0  ROOTN=.. DIR=en2zh SHARD=k     si_infer T1c, S1 flags, 4 shards, seed default
#   STAGE=asr     GPU=0  ROOTN=.. DIR=en2zh             speech-only ASR (paper protocol)
# Roots eval_outputs/mist_w3po_v1/wave16/<rootn>/ mirror wave2_S1 (thinker_<dir>/, talker_<dir>_s20261021/, data ->
# wave2_S1/data). Detached (setsid nohup, fds 8/9 closed); "card pid name" -> $PIDS (wave16_noplaceholder.sh).
# Placeholders of the card are stopped by PID first. DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${STAGE:?}" "${GPU:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
LOGD=logs/mist_w3po_v1/wave16/erm
PIDS=${PIDS:-$LOGD/w16e_workers.pids}
M=eval_outputs/mist_w3po_v1/wave16/erm
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
S1ITEMS=eval_outputs/mist_w3po_v1/wave2_S1/data/items_${DIR:-x}.jsonl
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
mkroot() { mkdir -p "$M/$1"; [ -e "$M/$1/data" ] || ln -s ../../../wave2_S1/data "$M/$1/data"; }
case "$STAGE" in
  thinker_lora)
    : "${DIR:?}" "${ROOTN:?}"; mkroot "$ROOTN"; ROOT=$M/$ROOTN; NAME=thinker_lora_${ROOTN}_$DIR
    EXTRA=(); [ -n "${CONST:-}" ] && EXTRA=(--const-alpha "$CONST")
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.mist_w3po_v1.voicecond.wave16_si_thinker_ncwi
         --assign "$M/fit/assign_$DIR.jsonl" --lora-root "$M/lora" "${EXTRA[@]}"
         --thinker "$QWEN3" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}"
         --out "$ROOT/thinker_$DIR" --items-file "$S1ITEMS") ;;
  merge)
    : "${ALPHA:?}"; NAME=merge_a$ALPHA
    CMD=(env PYTHONPATH=. "$PYQ" experiments/mist_w3po_v1/voicecond/wave5_m_merge.py --adapters "$M/adapters_a$ALPHA/adapters_final.pt"
         --out "$M/merged_a$ALPHA") ;;
  thinker_merged)
    : "${DIR:?}" "${ALPHA:?}" "${MODEL:?}" "${ITEMS:?}"; mkroot eval_ncwi_merged; NAME=thinker_merged_${DIR}_a$ALPHA
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         --thinker "$MODEL" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}"
         --out "$M/eval_ncwi_merged/parts/thinker_${DIR}_a$ALPHA" --items-file "$ITEMS") ;;
  talker)
    : "${DIR:?}" "${SHARD:?}" "${ROOTN:?}"; ROOT=$M/$ROOTN; NAME=talker_${ROOTN}_${DIR}_$SHARD
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer
         --model "$TALKER" --direction "$DIR" --window 1 --output "$ROOT/talker_${DIR}_s20261021" --shard "$SHARD" --num-shards 4
         --text-policy file --blocks-file "$ROOT/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "$S1ITEMS") ;;
  asr)
    : "${DIR:?}" "${ROOTN:?}"; ROOT=$M/$ROOTN; NAME=asr_${ROOTN}_$DIR; RUN=$ROOT/talker_${DIR}_s20261021
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE: thinker_lora|merge|thinker_merged|talker|asr" >&2; exit 1 ;;
esac
mkdir -p "$LOGD"
LOGF=$LOGD/w16e_eval_$NAME.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
if [ "$GPU" != "-" ]; then
  # cards 0-3; 4-5 only when wave16_tools.py extracards cleared them (coordinator 16:15 CET); never 6-7 (wave 17 training)
  case " 0 1 2 3 4 5 " in *" $GPU "*) ;; *) echo "wave 16 runs on cards 0-5 only (got $GPU)" >&2; exit 1;; esac
fi
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
