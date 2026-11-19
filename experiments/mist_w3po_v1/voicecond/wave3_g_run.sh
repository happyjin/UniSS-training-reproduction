#!/usr/bin/env bash
# Robust wave 3, group G (robust_wave3_single_prereg.zh-CN.md): place ONE job on card 5 (G's only card), detached
# (setsid nohup, fds 8/9 closed), after stopping card 5's placeholder supervisor + worker by PID (argv re-checked; never
# the guard 2726951, gpu_watch 52045, sync 1222453). Appends "5 pid name" to $PIDS (read by wave3_g_noplaceholder.sh).
#   STAGE=gen DIR=en2zh W=1 OUTD=<dir> [BACKEND=hf|vllm] [LIMIT=20] [ROWS=64] NAME=<n> bash wave3_g_run.sh
#   STAGE=asr RUN=<si_infer output dir with part_*.jsonl> NAME=<n> bash wave3_g_run.sh
#   DRY_RUN=1 prints the command only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
GPU=5
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
LOGD=logs/mist_w3po_v1/wave3
PIDS=${PIDS:-$LOGD/g_workers.pids}
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
S1=eval_outputs/mist_w3po_v1/wave2_S1
GD=eval_outputs/mist_w3po_v1/wave3_combos/G
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${STAGE:?}" "${NAME:?}"
mkdir -p "$LOGD"
LOGF=$LOGD/g_${NAME}.log
[ -e "$LOGF" ] && { echo "拒绝覆盖已有日志: $LOGF" >&2; exit 1; }
case "$STAGE" in
  gen)
    : "${DIR:?}" "${W:?}" "${OUTD:?}"
    [ -e "$OUTD" ] && { echo "拒绝覆盖已有输出: $OUTD" >&2; exit 1; }
    CMD=("${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$GPU "$PYV" -m experiments.mist_w3po_v1.voicecond.wave3_g_cfg
         --w "$W" --rows "${ROWS:-64}" --backend "${BACKEND:-hf}" --seed 20261021
         --model "$TALKER" --direction "$DIR" --window 1 --output "$OUTD" --text-policy file
         --blocks-file "$S1/thinker_$DIR/blocks_${DIR}_w1.jsonl" --speed-policy block --gpu-mem 0.1
         --items-file "$GD/data/items_$DIR.jsonl" ${LIMIT:+--limit "$LIMIT"}) ;;
  asr)
    : "${RUN:?}"
    CMD=(bash -c "set -e; cat '$RUN'/part_*.jsonl > '$RUN/results.jsonl' && $PYE reports/uniss_si_v4/speech_only.py '$RUN' && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$RUN/results_speechonly.jsonl' --output '$RUN/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$RUN/asr_speech.jsonl'") ;;
  *) echo "STAGE must be gen or asr" >&2; exit 1 ;;
esac
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PROTECT=" 2726951 52045 1222453 $$ "
for pat in run_mist_workload.sh train_mist_workload.py; do
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/$pat*"--devices $GPU "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $GPU: ${a:0:120}";; esac
  done < <(ps -eo pid=,args=)
done
sleep 8
log "start $NAME on card $GPU: ${CMD[*]}" | tee -a "$LOGD/g_launch.log"
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null 8>&- 9>&- &
echo "$GPU $! $NAME" >> "$PIDS"
log "$NAME pid $!" | tee -a "$LOGD/g_launch.log"
