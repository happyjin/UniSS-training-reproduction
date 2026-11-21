#!/usr/bin/env bash
# Robust D1 Talker-sensitivity orchestrator (reports/mist_w3po_v1/robust_d1_talker_sensitivity_prereg.zh-CN.md section 4),
# cards 4 and 5 only (copy of oracle_voice_orch.sh with a K1 waiter and three arms):
#  0. wait until K1 step 2 has finished (eval_outputs/mist_w3po_v1/k1/step2/DONE or FAILED, per k1_step2_chain.sh) and no
#     K1 step-2 process (k1_step2_chain / k1_thinker / k1_score / lora_k1p) is alive; gives up after MAX_WAIT_S.
#  1. d1ts_tokens (card 4) -> d1/tokens/tokens_{C0,C10,C45}_<d>.jsonl + items_clean_<d>.jsonl
#  2. per direction (en2zh card 4, zh2en card 5), in parallel: for arm C0, C10, C45 in turn 4 Talker shards (S1 B layout,
#     seed 20261021, fixed KV, 10 s apart); the arm's speech-only ASR -> SQUIM PESQ (CPU) + TitaNet similarity (same card)
#     run in the background while the next arm's Talkers run -> d1/metrics/<arm>_<d>_*
#  3. d1ts_score -> d1/summary.json, touch d1/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher d1ts_noplaceholder.sh (POLL 1 s). Never touches guard 2726951, gpu_watch 52045, sync 1222453.
# Refuses an existing d1 root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/d1ts_orch.sh > logs/mist_w3po_v1/d1/talker_sens/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/d1ts_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/d1
LOGD=logs/mist_w3po_v1/d1/talker_sens
K1M=eval_outputs/mist_w3po_v1/k1/step2
MAX_WAIT_S=${MAX_WAIT_S:-28800}
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
GA=4; GB=5                       # tokens / en2zh card, zh2en card
S1=eval_outputs/mist_w3po_v1/wave2_S1
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { touch $R/FAILED; touch $STOP; }; exit 1; }

launch() {   # card name cmd... ; prints pid (DRY_RUN: prints the command to stderr and "DRY")
  local g=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
  local k left=""
  for k in $(seq 1 24); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && break
    sleep 5
  done
  [ -n "$left" ] && log "WARNING: placeholders still on card $g after 120 s:$left" >&2
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p $name" >> "$PIDS"
  log "$name card $g pid $p" >&2
  echo $p
}
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }
k1_alive() {   # pids of live K1 step-2 processes (argv from ps fields)
  local out="" pid args
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *k1_step2_chain*|*k1_thinker*|*k1_score*|*lora_k1p*) out="$out $pid";; esac
  done < <(ps -eo pid=,args=)
  echo "$out"
}

if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "d1 root exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p "$LOGD"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  [ -e "$PIDS" ] && { echo "PIDS exists: $PIDS (refuse to overwrite)" >&2; exit 1; }
  # 0. wait for K1 step 2 (cards 4, 5 carry its Thinker eval)
  log "orch $$: waiting for $K1M/DONE or FAILED and no K1 step-2 process"
  t0=$(date +%s); n=0
  while :; do
    if [ -e "$K1M/DONE" ] || [ -e "$K1M/FAILED" ]; then
      a=$(k1_alive); [ -z "$a" ] && break
      [ $((n % 20)) = 0 ] && log "marker present, K1 processes still alive:$a"
    fi
    [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT_S" ] && { log "FAILED: K1 step 2 not finished after $MAX_WAIT_S s"; exit 1; }
    n=$((n + 1)); sleep 30
  done
  log "K1 step 2 finished ($(ls $K1M | tr '\n' ' ')); starting D1"
  mkdir -p "$R/tokens" "$R/metrics"
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/d1ts_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher $!"
fi

# 1. tokens
p=$(launch $GA tokens "${VENV[@]}" CUDA_VISIBLE_DEVICES=$GA $PYV -I experiments/mist_w3po_v1/voicecond/d1ts_tokens.py \
  --out $R/tokens --device cuda:0)
waitp "$p"
[ -n "$DRY" ] || [ -s $R/tokens/tokens_stats.json ] || die "tokens missing"
log "tokens done"

# 2. per direction
post() {   # d card arm  (ASR -> spksim + pesq); exit status = success
  local d=$1 g=$2 arm=$3 out=$R/talker_$3_$1 p
  p=$(launch $g asr_${arm}_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
  waitp "$p"
  [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || { log "asr $arm $d produced nothing"; return 1; }
  p=$(launch $g spksim_${arm}_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $out --out $R/metrics/${arm}_${d}_spksim.jsonl --device cuda:0)
  if [ -n "$DRY" ]; then echo "[dry] pesq $arm $d: $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out --out $R/metrics/${arm}_${d}_squim.jsonl --procs 16" >&2
  else PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out \
         --out $R/metrics/${arm}_${d}_squim.jsonl --procs 16 > $LOGD/pesq_${arm}_$d.log 2>&1 || { log "pesq $arm $d failed"; return 1; }; fi
  waitp "$p"
  [ -n "$DRY" ] || [ -s $R/metrics/${arm}_${d}_spksim.jsonl ] || { log "spksim $arm $d produced nothing"; return 1; }
  log "metrics $arm $d done"
}
pipe_dir() {   # d card
  local d=$1 g=$2 arm s out ps posts=()
  for arm in C0 C10 C45; do
    out=$R/talker_${arm}_$d; ps=()
    for s in 0 1 2 3; do
      ps+=("$(launch $g talk_${arm}_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
        -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $R/tokens/tokens_${arm}_$d.jsonl \
        --model $TALKER --direction $d --window 1 --output $out --shard $s --num-shards 4 --text-policy file \
        --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --speed-policy block --gpu-mem 0.1 \
        --items-file $R/tokens/items_clean_$d.jsonl --seed 20261021)")
      [ -n "$DRY" ] || sleep 10
    done
    waitp "${ps[@]}"
    for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $out/part_$s.jsonl ] || die "talker $arm $d shard $s ended without part file"; done
    log "talkers $arm $d done"
    post $d $g $arm & posts+=($!)
  done
  for p in "${posts[@]}"; do wait $p || die "post-processing $d (pid $p)"; done
}
pipe_dir en2zh $GA & PA=$!
[ -n "$DRY" ] || sleep 30
pipe_dir zh2en $GB & PB=$!
wait $PA || die "pipeline en2zh"
wait $PB || die "pipeline zh2en"

# 3. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.d1ts_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.d1ts_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
