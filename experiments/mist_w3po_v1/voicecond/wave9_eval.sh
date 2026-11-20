#!/usr/bin/env bash
# Robust wave 9 job launcher (modelled on wave5_m_eval.sh, wave-9 roots/logs, plus the anchor stages):
#   STAGE=fit     GPU=0                          PCA + bins of the anchor (wave9_anchor_fit.py)
#   STAGE=table   GPU=0 DIR=en2zh                anchor tokens of the evaluation items (wave9_anchor_table.py)
#   STAGE=check   GPU=0 DIR=en2zh MODE=none      wrapper on the BASE Thinker (S1 settings, gpu-mem 0.6) -> wave9/wrapper_check/<MODE>_<DIR>
#   STAGE=merge   GPU=- ARM=A_wise05|C_wise05    adapters (+ anchor rows) -> merged copy (CPU)
#   STAGE=thinker GPU=0 ARM=.. DIR=en2zh         LA2 w1: arm A through wave9_si_thinker_anchor.py --anchor table,
#                                                arm C through si_thinker_stream itself (no anchor), as wave 5
#   STAGE=talker  GPU=0 ARM=.. DIR=en2zh SHARD=k si_infer T1c, S1 flags, 4 shards
#   STAGE=asr     GPU=0 ARM=.. DIR=en2zh         speech-only ASR (paper protocol)
# Detached (setsid nohup, fds 8/9 closed); "card pid name" -> $PIDS (wave9_noplaceholder.sh). The placeholders of the
# card are stopped by PID first (argv re-checked; never guard 2726951 / gpu_watch 52045 / sync 1222453).
# DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${STAGE:?}" "${GPU:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
LOGD=logs/mist_w3po_v1/wave9
PIDS=${PIDS:-$LOGD/w9_workers.pids}
M=eval_outputs/mist_w3po_v1/wave9
ARM=${ARM:-x}
ROOT=$M/eval_$ARM
MERGED=$M/merged_$ARM
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
ITEMS=eval_outputs/mist_w3po_v1/wave2_S1/data/items_${DIR:-x}.jsonl
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
QENV=(env PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
mkdir_root() { mkdir -p "$ROOT"; [ -e "$ROOT/data" ] || ln -s ../../wave2_S1/data "$ROOT/data"; }
case "$STAGE" in
  fit)
    NAME=fit
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave9_anchor_fit.py) ;;
  table)
    : "${DIR:?}"; NAME=table_$DIR
    CMD=("${QENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave9_anchor_table.py --dir "$DIR") ;;
  check)
    : "${DIR:?}" "${MODE:?}"; NAME=check_${MODE}_$DIR
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.mist_w3po_v1.voicecond.wave9_si_thinker_anchor
         --anchor "$MODE" --anchor-check 64 --direction "$DIR" --windows 1 --policy la2 --gpu-mem 0.6
         --out "$M/wrapper_check/${MODE}_$DIR" --items-file "$ITEMS") ;;
  merge)
    mkdir_root; NAME=merge_$ARM
    CMD=("${QENV[@]}" "$PYQ" experiments/mist_w3po_v1/voicecond/wave9_merge.py --adapters "$M/run_$ARM/adapters_final.pt" --out "$MERGED") ;;
  thinker)
    : "${DIR:?}"; mkdir_root; NAME=thinker_${ARM}_$DIR
    case "$ARM" in
      A*) MOD=(-m experiments.mist_w3po_v1.voicecond.wave9_si_thinker_anchor --anchor table) ;;
      *)  MOD=(-m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream) ;;
    esac
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" "${MOD[@]}"
         --thinker "$MERGED" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}"
         --out "$ROOT/thinker_$DIR" --items-file "$ITEMS") ;;
  talker)
    : "${DIR:?}" "${SHARD:?}"; mkdir_root; NAME=talker_${ARM}_${DIR}_$SHARD
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer
         --model "$TALKER" --direction "$DIR" --window 1 --output "$ROOT/talker_${DIR}_s20261021" --shard "$SHARD" --num-shards 4
         --text-policy file --blocks-file "$ROOT/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "$ITEMS") ;;
  asr)
    : "${DIR:?}"; mkdir_root; NAME=asr_${ARM}_$DIR; RUN=$ROOT/talker_${DIR}_s20261021
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE: fit|table|check|merge|thinker|talker|asr" >&2; exit 1 ;;
esac
LOGF=$LOGD/w9_$NAME.log
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
echo "${GPU/-/none} $! wave9_$NAME" >> "$PIDS"
echo "$(TZ=Europe/Prague date '+%F %T %Z') $NAME on card $GPU pid $!"
