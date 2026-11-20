#!/usr/bin/env bash
# Robust wave 10 evaluation launcher (copy of wave5_m_eval.sh; own paths, logs and PIDS): the S1 chain on the TIES+WiSE
# merged Thinker.
#   STAGE=merge   GPU=-   ARM=ties_wise05                                 two direction LoRAs -> TIES -> WiSE 0.5 copy (CPU)
#   STAGE=thinker GPU=0   ARM=.. DIR=en2zh                                si_thinker_stream --thinker <merged> la2 w1
#   STAGE=talker  GPU=0   ARM=.. DIR=en2zh SHARD=k                         si_infer T1c, S1 flags, 4 shards, seed default
#   STAGE=asr     GPU=0   ARM=.. DIR=en2zh                                speech-only ASR (paper protocol)
# Root eval_outputs/mist_w3po_v1/wave10/eval_<arm>/ mirrors wave2_S1 (thinker_<dir>/, talker_<dir>_s20261021/, data ->
# wave2_S1/data). Detached (setsid nohup, fds 8/9 closed); "card pid name" -> $PIDS (wave10_noplaceholder.sh).
# Placeholders of the card are stopped by PID first. DRY_RUN=1 prints the command. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${STAGE:?}" "${GPU:?}" "${ARM:?}"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
LOGD=logs/mist_w3po_v1/wave10
PIDS=${PIDS:-$LOGD/w10_workers.pids}
M=eval_outputs/mist_w3po_v1/wave10
ROOT=$M/eval_$ARM
MERGED=$M/merged_$ARM
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
ITEMS=eval_outputs/mist_w3po_v1/wave2_S1/data/items_${DIR:-x}.jsonl
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
mkdir -p "$ROOT"
[ -e "$ROOT/data" ] || ln -s ../../wave2_S1/data "$ROOT/data"
case "$STAGE" in
  merge)
    NAME=merge_$ARM
    [ "$ARM" = ties_wise05 ] || { echo "merge: ARM must be ties_wise05" >&2; exit 1; }
    CMD=(env PYTHONPATH=. "$PYQ" experiments/mist_w3po_v1/voicecond/wave10_ties_merge.py --zh2en "$M/run_zh2en/adapters_final.pt"
         --en2zh "$M/run_en2zh/adapters_final.pt" --k 0.2 --lam 1 --wise 0.5 --out "$MERGED") ;;
  thinker)
    : "${DIR:?}"; NAME=thinker_${ARM}_$DIR
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
         --thinker "$MERGED" --direction "$DIR" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}"
         --out "$ROOT/thinker_$DIR" --items-file "$ITEMS") ;;
  talker)
    : "${DIR:?}" "${SHARD:?}"; NAME=talker_${ARM}_${DIR}_$SHARD
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer
         --model "$TALKER" --direction "$DIR" --window 1 --output "$ROOT/talker_${DIR}_s20261021" --shard "$SHARD" --num-shards 4
         --text-policy file --blocks-file "$ROOT/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "$ITEMS") ;;
  asr)
    : "${DIR:?}"; NAME=asr_${ARM}_$DIR; RUN=$ROOT/talker_${DIR}_s20261021
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE: merge|thinker|talker|asr" >&2; exit 1 ;;
esac
LOGF=$LOGD/w10_eval_$NAME.log
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
