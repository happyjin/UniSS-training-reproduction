#!/usr/bin/env bash
# K2 step 2 (reports/mist_w3po_v1/robust_k2_w2map_prereg.zh-CN.md): W2 arm on S1 dev, cards 4 and 5 only (2026-11-21 07:40 CET: cards 2, 3 went to K1 training).
#   1. W2 token tables (k2_tokens.py on GPU cards 4/5, same prompts / sha1 as wave3_combos/V V1 tables) -> $R/tokens/
#   2. T1c Talker with the W2 tokens (idea1_si_infer_vo, S1 flags of wave3_v_run.sh, seed 20261021, 4 shards, 2 per card)
#      -> $R/w2/talker_<d>
#   3. speech-only ASR, 4. SQUIM PESQ (CPU) + TitaNet similarity -> $R/metrics/w2_<d>_{squim,spksim}.jsonl
#   5. k2_score.py (B, V1, W2 point estimates) -> $R/summary.json; DONE / FAILED.
# Placeholders (scripts/gpu_load/) of a card stopped by PID right before each launch, wait <= 120 s until none remain
# (combo_orch.sh launch()); watcher combo_noplaceholder.sh with POLL=1, PIDS $LOGD/k2.pids, STOP $LOGD/k2.stop.
# Never touches guard 2726951, gpu_watch 52045, sync 1222453.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/k2_orch.sh > logs/mist_w3po_v1/k2/orch.log 2>&1 < /dev/null 8>&- 9>&- &
# DRY_RUN=1 prints the commands. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
read -ra CA <<< "${CARDS:-4 5}"; NC=${#CA[@]}
for c in "${CA[@]}"; do case " 4 5 " in *" $c "*) ;; *) echo "card $c not allowed for K2 (cards 1,2,3,7 = K1 training)" >&2; exit 1;; esac; done
[ "$NC" -ge 1 ] || exit 1
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/k2
LOGD=logs/mist_w3po_v1/k2
export PIDS=$LOGD/k2.pids
STOP=$LOGD/k2.stop
MAP=$R/fit/w2_map.npz
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
S1=eval_outputs/mist_w3po_v1/wave2_S1
DATA=$S1/data
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
DIRS=(en2zh zh2en)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() {
  log "STOP: $*"
  [ -n "$DRY" ] && return 0
  echo "$*" > "$R/FAILED"; touch "$STOP"; exit 1
}
PROTECT=" 2726951 52045 1222453 $$ "
launch() {   # card name cmd... ; prints pid
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
nap() { [ -n "$DRY" ] || sleep "$1"; }
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }

# ---------------------------------------------------------------- set-up
if [ -z "$DRY" ]; then
  [ -s "$MAP" ] || { echo "missing $MAP" >&2; exit 1; }
  [ -e "$R/w2" ] && { echo "exists: $R/w2 (refuse to overwrite)" >&2; exit 1; }
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  mkdir -p "$R/w2" "$R/tokens" "$R/metrics"
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/combo_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
  git rev-parse HEAD > "$R/w2/config.txt"
fi

# ---------------------------------------------------------------- 1. tokens (GPU, so the V1 ids reproduce the GPU-made V1 table)
tp=(); k=0
for d in "${DIRS[@]}"; do
  g=${CA[$k]}; k=$((k + 1))
  tp+=($(launch "$g" tok_$d env PYTHONPATH=. CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=16 "$PYV" -I \
         experiments/mist_w3po_v1/voicecond/k2_tokens.py --dir "$d" --map "$MAP" --out "$R/tokens/tokens_w2_$d.jsonl" --device cuda:0))
done
waitp "${tp[@]}"
for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/tokens/tokens_w2_$d.jsonl" ] || die "no W2 tokens $d"; done
log "tokens done"

# ---------------------------------------------------------------- 2. talkers, 2 per card
ps_=(); k=0
for d in "${DIRS[@]}"; do for s in 0 1 2 3; do
  g=${CA[$(( (k / 2) % NC ))]}; k=$((k + 1))
  ps_+=($(launch "$g" talk_w2_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g "$PYV" \
    -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override "$R/tokens/tokens_w2_$d.jsonl" \
    --model "$TALKER" --direction "$d" --window 1 --output "$R/w2/talker_$d" --shard "$s" --num-shards 4 \
    --text-policy file --blocks-file "$S1/thinker_$d/blocks_${d}_w1.jsonl" --speed-policy block --gpu-mem 0.1 \
    --items-file "$DATA/items_$d.jsonl" --seed 20261021))
  nap 25
  if [ $((k % (2 * NC))) = 0 ]; then waitp "${ps_[@]}"; ps_=(); fi
done; done
waitp "${ps_[@]}"
for d in "${DIRS[@]}"; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s "$R/w2/talker_$d/part_$s.jsonl" ] || die "no talker part $d $s"; done; done
log "talkers done"

# ---------------------------------------------------------------- 3. ASR
ps_=(); k=0
for d in "${DIRS[@]}"; do
  g=${CA[$((k % NC))]}; k=$((k + 1)); T=$R/w2/talker_$d
  ps_+=($(launch "$g" asr_w2_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$T' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$T/results_speechonly.jsonl' --output '$T/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$T/asr_speech.jsonl'"))
  nap 10
done
waitp "${ps_[@]}"
for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/w2/talker_$d/asr_speech.jsonl" ] || die "no asr $d"; done
log "asr done"

# ---------------------------------------------------------------- 4. PESQ (CPU) + similarity
ps_=(); k=0
for d in "${DIRS[@]}"; do
  if [ -n "$DRY" ]; then log "[dry] pesq $d"; else
    PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run "$R/w2/talker_$d" \
      --out "$R/metrics/w2_${d}_squim.jsonl" --procs 24 > "$LOGD/pesq_w2_$d.log" 2>&1 & ps_+=($!)
  fi
  g=${CA[$((k % NC))]}; k=$((k + 1))
  ps_+=($(launch "$g" spksim_w2_$d bash -c "CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir $d --run $R/w2/talker_$d --out $R/metrics/w2_${d}_spksim.jsonl --device cuda:0"))
done
waitp "${ps_[@]}"
for d in "${DIRS[@]}"; do for m in squim spksim; do
  [ -n "$DRY" ] || [ -s "$R/metrics/w2_${d}_$m.jsonl" ] || die "no $m $d"; done; done
log "metrics done"
[ -n "$DRY" ] || touch "$STOP"

# ---------------------------------------------------------------- 5. score
if [ -n "$DRY" ]; then log "[dry] score"; else
  PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.k2_score --out "$R/summary.json" > "$LOGD/score.log" 2>&1 \
    || die "score"
  date > "$R/DONE"
fi
log "K2 done: $R/summary.json"
