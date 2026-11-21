#!/usr/bin/env bash
# AbS-C orchestrator (reports/mist_w3po_v1/robust_abs_c_prereg.zh-CN.md section 6), cards 4, 5, 6, 7 ONLY
# (cards 0-3 belong to another session; launch() refuses any other card). Built from b_abs_orch.sh / b_abs_seed2_orch.sh:
#  1. abs_c_search: one process per card 4-7, process k = shard k (rows mod 4) of en2zh then zh2en -> abs_c/search
#  2. abs_c_precheck (card 4); go = false -> abs_c/STOPPED_PRECHECK, no Talker eval
#  3. Talkers, AbS-C only, TTS seeds 20261021 and 20261022: en2zh shards 0,1 card 4 / 2,3 card 6; zh2en shards 0,1 card 5 /
#     2,3 card 7; 2 seeds x 2 shards = 4 Talkers per card (fixed KV), same-card launches >= 45 s apart
#  4. per (seed, direction): speech-only ASR -> TitaNet + ECAPA2 (GPU) + SQUIM PESQ (CPU), 4 chains in parallel
#  5. abs_c_score -> abs_c/summary.json, touch abs_c/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher abs_c_noplaceholder.sh (POLL 1 s). Never touches guard 2726951, gpu_watch 52045,
# sync 1222453. Refuses an existing abs_c root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/abs_c_orch.sh > logs/mist_w3po_v1/abs_c/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/abs_c_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/abs_c
LOGD=logs/mist_w3po_v1/abs_c
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
S1=eval_outputs/mist_w3po_v1/wave2_S1
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
NENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
SENV=(env PYTHONPATH=.:.tmp/b_abs_pyshim OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { touch $R/FAILED; touch $STOP; }; exit 1; }

launch() {   # card name cmd... ; prints pid (DRY_RUN: prints the command to stderr and "DRY")
  local g=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  case " 4 5 6 7 " in *" $g "*) ;; *) die "card $g not allowed";; esac
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

for d in en2zh zh2en; do [ -s eval_outputs/mist_w3po_v1/b_abs/tokens/tokens_abs_$d.jsonl ] || { echo "missing AbS table $d" >&2; exit 1; }; done
if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "abs_c root exists: $R (refuse to overwrite)" >&2; exit 1; }
  [ -e .tmp/b_abs_pyshim/einx/__init__.py ] && [ -e .tmp/b_abs_pyshim/frozendict/__init__.py ] || { echo "pyshim missing" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/abs_c_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

# 1. search (one process per card)
ps_=()
for k in 0 1 2 3; do
  g=$((4 + k))
  ps_+=("$(launch $g search_$k env CUDA_VISIBLE_DEVICES=$g "${SENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.abs_c_search \
    --dirs en2zh zh2en --shard $k --num-shards 4 --out $R/search --device cuda:0)")
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $R/search/search_${d}_$k.jsonl ] || die "missing search_${d}_$k"; done; done
fi
log "search done"

# 2. precheck
p=$(launch 4 precheck env CUDA_VISIBLE_DEVICES=4 "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.abs_c_precheck \
  --root $R --device cuda:0)
waitp "$p"
if [ -z "$DRY" ]; then
  [ -s $R/precheck.json ] || die "precheck produced no precheck.json"
  go=$($PYQ -c "import json; print(int(json.load(open('$R/precheck.json'))['go']))") || die "precheck read"
  if [ "$go" != 1 ]; then
    log "PRECHECK STOP (see $R/precheck.json)"
    touch $R/STOPPED_PRECHECK
    touch $STOP
    exit 0
  fi
fi
log "precheck go"

# 3. Talkers: 4 slots (seed x local shard); each slot launches on all 4 cards, then 15 s (+ 4 x >= 8 s in launch)
tp=()
for seed in 20261021 20261022; do
  for sh in 0 1; do
    for gd in "4 en2zh $sh" "6 en2zh $((2 + sh))" "5 zh2en $sh" "7 zh2en $((2 + sh))"; do
      read -r g d s <<< "$gd"
      tp+=("$(launch $g talk_absc_${d}_s${seed}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
        -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $R/tokens/tokens_absc_$d.jsonl \
        --model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed $seed \
        --direction $d --output $R/talker_absc_${d}_s$seed --shard $s --num-shards 4 \
        --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --items-file $S1/data/items_$d.jsonl)")
    done
    [ -n "$DRY" ] || sleep 15
  done
done
waitp "${tp[@]}"
if [ -z "$DRY" ]; then
  for seed in 20261021 20261022; do for d in en2zh zh2en; do for s in 0 1 2 3; do
    [ -s $R/talker_absc_${d}_s$seed/part_$s.jsonl ] || die "talker absc $d seed $seed shard $s ended without part file"
  done; done; done
fi
log "talkers done"

# 4. metrics, one chain per (seed, direction)
metrics() {   # card d seed
  local g=$1 d=$2 seed=$3 out=$R/talker_absc_$2_s$3 name=absc_$2_s$3 p p1 p2
  p=$(launch $g asr_$name bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
  waitp "$p"
  [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || die "asr $name produced nothing"
  p1=$(launch $g spksim_$name env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $out --out $R/metrics/${name}_spksim.jsonl --device cuda:0)
  p2=$(launch $g ecapa2_$name env CUDA_VISIBLE_DEVICES=$g "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_spksim \
    --dir $d --run $out --out $R/metrics/ecapa2_${name}_spksim.jsonl --device cuda:0)
  if [ -n "$DRY" ]; then echo "[dry] pesq $name" >&2
  else PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out \
         --out $R/metrics/${name}_squim.jsonl --procs 16 > $LOGD/pesq_$name.log 2>&1 || die "pesq $name"; fi
  waitp "$p1" "$p2"
  [ -n "$DRY" ] || { [ -s $R/metrics/${name}_spksim.jsonl ] && [ -s $R/metrics/ecapa2_${name}_spksim.jsonl ]; } || die "spksim $name produced nothing"
  log "metrics $name done"
}
mp=()
for gds in "4 en2zh 20261021" "5 zh2en 20261021" "6 en2zh 20261022" "7 zh2en 20261022"; do
  read -r g d seed <<< "$gds"
  metrics $g $d $seed & mp+=($!)
  [ -n "$DRY" ] || sleep 10
done
for p in "${mp[@]}"; do wait $p || die "a metrics chain failed"; done

# 5. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.abs_c_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.abs_c_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
