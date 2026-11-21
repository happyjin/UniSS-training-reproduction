#!/usr/bin/env bash
# Voice seeds 3-4 measurement (reports/mist_w3po_v1/robust_voice_seeds34_prereg.zh-CN.md), cards 4, 5, 6, 7 ONLY.
# TTS seeds 20261023, 20261024 for B (si_infer, no override), V1 (wave3_combos/V/tokens/tokens_v1_<d>.jsonl), AbS
# (b_abs/tokens/tokens_abs_<d>.jsonl), AbS-C (abs_c/tokens/tokens_absc_<d>.jsonl); every other flag as b_abs_seed2_orch.sh /
# abs_c_orch.sh. One wave per seed: en2zh shards 0,1 card 4 / 2,3 card 6; zh2en shards 0,1 card 5 / 2,3 card 7; 4 arms x 2
# shards = 8 Talkers per card (fixed KV), same-card launches >= 45 s apart. Then 16 metric chains (arm x dir x seed), 4 per
# card: speech-only ASR -> TitaNet + ECAPA2 (GPU) + SQUIM PESQ (CPU) -> voice_seeds34_score -> DONE (FAILED on any error).
# launch() copied from b_abs_seed2_orch.sh: placeholders of the card SIGTERMed by PID (argv re-checked), wait <= 120 s until
# no gpu_load process is left; own watcher instance b_abs_noplaceholder.sh (POLL 1 s, PIDS / STOP in this run's log dir).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453. Refuses an existing voice_seeds34 root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/voice_seeds34_orch.sh > logs/mist_w3po_v1/voice_seeds34/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/voice_seeds34_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
SEEDS="20261023 20261024"
R=eval_outputs/mist_w3po_v1/voice_seeds34
LOGD=logs/mist_w3po_v1/voice_seeds34
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
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { touch $R/FAILED; touch $STOP; }; exit 1; }
table() {   # arm d -> token table ("" for B)
  case $1 in B) echo "";; v1) echo eval_outputs/mist_w3po_v1/wave3_combos/V/tokens/tokens_v1_$2.jsonl;;
             abs) echo eval_outputs/mist_w3po_v1/b_abs/tokens/tokens_abs_$2.jsonl;;
             absc) echo eval_outputs/mist_w3po_v1/abs_c/tokens/tokens_absc_$2.jsonl;; esac
}

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

for d in en2zh zh2en; do for arm in v1 abs absc; do [ -s "$(table $arm $d)" ] || { echo "missing table $arm $d" >&2; exit 1; }; done; done
if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "root exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/b_abs_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

# 1. Talkers: one wave per seed; 8 slots (arm x local shard), each slot launches on all 4 cards, then 15 s
for SEED in $SEEDS; do
  COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed $SEED)
  tp=()
  for arm in B v1 abs absc; do
    for sh in 0 1; do
      for gd in "4 en2zh $sh" "6 en2zh $((2 + sh))" "5 zh2en $sh" "7 zh2en $((2 + sh))"; do
        read -r g d s <<< "$gd"
        tab=$(table $arm $d)
        if [ -z "$tab" ]; then hook=(-m experiments.uniss_qwen3_omni_v1.evaluation.si_infer)
        else hook=(-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $tab); fi
        tp+=("$(launch $g talk_${arm}_${d}_s${SEED}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV "${hook[@]}" \
          "${COMMON[@]}" --direction $d --output $R/talker_${arm}_${d}_s$SEED --shard $s --num-shards 4 \
          --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --items-file $S1/data/items_$d.jsonl)")
      done
      [ -n "$DRY" ] || sleep 15
    done
  done
  waitp "${tp[@]}"
  if [ -z "$DRY" ]; then
    for arm in B v1 abs absc; do for d in en2zh zh2en; do for s in 0 1 2 3; do
      [ -s $R/talker_${arm}_${d}_s$SEED/part_$s.jsonl ] || die "talker $arm $d seed $SEED shard $s ended without part file"
    done; done; done
  fi
  log "talkers seed $SEED done"
done

# 2. metrics, one chain per (arm, direction, seed)
metrics() {   # card arm d seed
  local g=$1 arm=$2 d=$3 seed=$4 out=$R/talker_$2_$3_s$4 name=$2_$3_s$4 p p1 p2
  p=$(launch $g asr_$name bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
  waitp "$p"
  [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || die "asr $name produced nothing"
  p1=$(launch $g spksim_$name env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $out --out $R/metrics/${name}_spksim.jsonl --device cuda:0)
  p2=$(launch $g ecapa2_$name env CUDA_VISIBLE_DEVICES=$g "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_spksim \
    --dir $d --run $out --out $R/metrics/ecapa2_${name}_spksim.jsonl --device cuda:0)
  if [ -n "$DRY" ]; then echo "[dry] pesq $name" >&2
  else PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out \
         --out $R/metrics/${name}_squim.jsonl --procs 8 > $LOGD/pesq_$name.log 2>&1 || die "pesq $name"; fi
  waitp "$p1" "$p2"
  [ -n "$DRY" ] || { [ -s $R/metrics/${name}_spksim.jsonl ] && [ -s $R/metrics/ecapa2_${name}_spksim.jsonl ]; } || die "spksim $name produced nothing"
  log "metrics $name done"
}
mp=()
k=0
for d in zh2en en2zh; do for seed in $SEEDS; do for arm in B v1 abs absc; do
  g=$((4 + k % 4)); k=$((k + 1))
  metrics $g $arm $d $seed & mp+=($!)
  [ -n "$DRY" ] || sleep 10
done; done; done
for p in "${mp[@]}"; do wait $p || die "a metrics chain failed"; done

# 3. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.voice_seeds34_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.voice_seeds34_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
