#!/usr/bin/env bash
# Robust K4 orchestrator (reports/mist_w3po_v1/robust_k4_tokencls_prereg.zh-CN.md section 5), cards 0 and 6 only
# (copy of oracle_voice_orch.sh plus the k3_orch.sh training / precheck steps):
#  1. k4_tokencls_train (card 0) -> k4/encoder/encoder.pt
#  2. k4_precheck (card 0) -> k4/precheck.json; stop rule: go = false -> k4/STOPPED_PRECHECK, no Talker eval
#  3. k4_tokens (card 0) -> k4/tokens/tokens_k4_<d>.jsonl (+ .stats.json, match with oracle tokens, record only)
#  4. per direction (en2zh card 0, zh2en card 6), in parallel: 4 Talker shards (S1 B layout, seed 20261021, 45 s apart)
#     -> speech-only ASR -> SQUIM PESQ (CPU) + TitaNet similarity (same card) -> k4/metrics/k4_<d>_*
#  5. k4_score -> k4/summary.json, touch k4/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher k4_noplaceholder.sh (POLL 1 s) keeps them off while jobs run. Never touches guard
# 2726951, gpu_watch 52045, sync 1222453, nor K1 step-2 training (cards 1,2,3,7). Refuses an existing k4 root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/k4_orch.sh > logs/mist_w3po_v1/k4/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/k4_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/k4
LOGD=logs/mist_w3po_v1/k4
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
GT=0; GZ=6                       # training / en2zh card, zh2en card
S1=eval_outputs/mist_w3po_v1/wave2_S1
V=eval_outputs/mist_w3po_v1/wave3_combos/V
I=data/open_voicecond_v1/idea1
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

if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "k4 root exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/tokens" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/k4_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

# 1. training
p=$(launch $GT train "${VENV[@]}" CUDA_VISIBLE_DEVICES=$GT $PYV -m experiments.mist_w3po_v1.voicecond.k4_tokencls_train \
  --data $I/train_en $I/train_zh --dev $I/diag_en $I/diag_zh --out $R/encoder --device cuda:0)
waitp "$p"
[ -n "$DRY" ] || [ -s $R/encoder/encoder.pt ] || die "training produced no encoder.pt"
log "training done"

# 2. precheck + stop rule
p=$(launch $GT precheck "${VENV[@]}" CUDA_VISIBLE_DEVICES=$GT $PYV -m experiments.mist_w3po_v1.voicecond.k4_precheck \
  --k4 $R/encoder/encoder.pt --out $R/precheck.json --device cuda:0)
waitp "$p"
if [ -z "$DRY" ]; then
  [ -s $R/precheck.json ] || die "precheck produced nothing"
  go=$($PYQ -c "import json; print(json.load(open('$R/precheck.json'))['go'])")
  log "precheck go=$go"
  [ "$go" = True ] || { touch $R/STOPPED_PRECHECK $STOP; log "stop rule: K4 noisy token match not >= V1 + 5 points in both languages"; exit 0; }
fi

# 3. tokens
p=$(launch $GT tokens "${VENV[@]}" CUDA_VISIBLE_DEVICES=$GT $PYV -I experiments/mist_w3po_v1/voicecond/k4_tokens.py \
  --k4 $R/encoder/encoder.pt --out $R/tokens --device cuda:0)
waitp "$p"
for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $R/tokens/tokens_k4_$d.jsonl ] || die "tokens $d missing"; done
log "tokens done"

# 4. per direction
pipe_dir() {   # d card
  local d=$1 g=$2 out=$R/talker_$1 s ps=()
  for s in 0 1 2 3; do
    ps+=("$(launch $g talk_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
      -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $R/tokens/tokens_k4_$d.jsonl \
      --model $TALKER --direction $d --window 1 --output $out --shard $s --num-shards 4 --text-policy file \
      --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --speed-policy block --gpu-mem 0.1 \
      --items-file $S1/data/items_$d.jsonl --seed 20261021)")
    [ -n "$DRY" ] || sleep 45
  done
  waitp "${ps[@]}"
  for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $out/part_$s.jsonl ] || die "talker $d shard $s ended without part file"; done
  log "talkers $d done"
  p=$(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
  waitp "$p"
  [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || die "asr $d produced nothing"
  log "asr $d done"
  p=$(launch $g spksim_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $out --out $R/metrics/k4_${d}_spksim.jsonl --device cuda:0)
  if [ -n "$DRY" ]; then echo "[dry] pesq $d: $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out --out $R/metrics/k4_${d}_squim.jsonl --procs 24" >&2
  else PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out \
         --out $R/metrics/k4_${d}_squim.jsonl --procs 24 > $LOGD/pesq_$d.log 2>&1 || die "pesq $d"; fi
  waitp "$p"
  [ -n "$DRY" ] || [ -s $R/metrics/k4_${d}_spksim.jsonl ] || die "spksim $d produced nothing"
  log "metrics $d done"
}
pipe_dir en2zh $GT & PA=$!
[ -n "$DRY" ] || sleep 60
pipe_dir zh2en $GZ & PB=$!
wait $PA || die "pipeline en2zh"
wait $PB || die "pipeline zh2en"

# 5. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.k4_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.k4_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
