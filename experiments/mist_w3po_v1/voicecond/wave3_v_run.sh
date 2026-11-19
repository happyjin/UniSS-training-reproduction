#!/usr/bin/env bash
# Robust wave 3, V group (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md): S1's single-speaker LA2 base with the
# voice tokens of arm V1/V2/V3 (wave3_v_tokens.py tables) put in by idea1_si_infer_vo.py; every other si_infer flag,
# the shard layout (4 shards) and the seed (20261021) are those of S1's B talker (wave2_s1_run.sh), so arms pair with B.
#   STAGE=tokens GPU=3 DIR=en2zh OUTD=<dir> [LIMIT=n] NAME=<n> bash wave3_v_run.sh
#   STAGE=talker GPU=3 DIR=en2zh OUTD=<dir> TABLE=<tokens jsonl>|none [SHARD=0 NSHARDS=4] [SEED=20261021] [ITEMS=..] NAME=<n> ...
#   STAGE=asr    GPU=3 RUN=<si_infer output dir> NAME=<n> ...
# Detached (setsid nohup, fds 8/9 closed); appends "card pid name" to $PIDS; stops the placeholders of the card by PID
# (argv re-checked; never guard 2726951, gpu_watch 52045, sync 1222453). DRY_RUN=1 prints the command only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
LOGD=logs/mist_w3po_v1/wave3
PIDS=${PIDS:-$LOGD/v_workers.pids}
BLK=eval_outputs/mist_w3po_v1/wave2_S1
DATA=$BLK/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${STAGE:?}" "${GPU:?}" "${NAME:?}"
mkdir -p "$LOGD"
LOGF=$LOGD/v_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }

case "$STAGE" in
  tokens)
    : "${DIR:?}" "${OUTD:?}"
    CMD=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$GPU /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -I
         experiments/mist_w3po_v1/voicecond/wave3_v_tokens.py --dir "$DIR" --out "$OUTD" --device cuda:0 ${LIMIT:+--limit "$LIMIT"}) ;;
  talker)
    : "${DIR:?}" "${OUTD:?}" "${TABLE:?}"
    if [ "$TABLE" = none ]; then HOOK=(-m experiments.uniss_qwen3_omni_v1.evaluation.si_infer)   # B: no override
    else HOOK=(-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override "$TABLE"); fi
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" "${HOOK[@]}"
         --model "$TALKER" --direction "$DIR" --window 1 --output "$OUTD" --shard "${SHARD:-0}" --num-shards "${NSHARDS:-4}"
         --text-policy file --blocks-file "$BLK/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "${ITEMS:-$DATA/items_$DIR.jsonl}" --seed "${SEED:-20261021}") ;;
  asr)
    : "${RUN:?}"
    CMD=(bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE must be tokens, talker or asr" >&2; exit 1 ;;
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
log "start $NAME on card $GPU: ${CMD[*]}" | tee -a "$LOGD/v_launch.log"
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
log "$NAME pid $!" | tee -a "$LOGD/v_launch.log"
