#!/usr/bin/env bash
# Robust A orchestrator (reports/mist_w3po_v1/robust_a_svanchor_prereg.zh-CN.md section 7), cards 4, 5, 6, 7 ONLY
# (cards 0-3 belong to another session; launch() refuses any other card). Built from k5_orch_b.sh / k4_orch.sh:
#  1. in parallel: a_sv_titanet (card 4) and a_sv_codes (card 5) -> a_sv/feats; ECAPA2 similarity of the existing
#     B / V1 (card 6) and K5-full / oracle (card 7) outputs -> a_sv/metrics/ecapa2_<arm>_<d>_spksim.jsonl
#  2. a_sv_train (card 4) -> a_sv/train/{mlp.pt, precheck.json, tokens/tokens_a_<d>.jsonl};
#     precheck go = false -> a_sv/STOPPED_PRECHECK, no Talker eval
#  3. in parallel: A en2zh (card 4), A zh2en (card 5), A-soft en2zh (card 6), A-soft zh2en (card 7); 4 Talker shards
#     each (S1 layout, seed 20261021, 45 s apart) -> speech-only ASR -> SQUIM PESQ (CPU) + TitaNet + ECAPA2 (same card)
#  4. a_sv_score -> a_sv/summary.json, touch a_sv/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher a_sv_noplaceholder.sh (POLL 1 s). Never touches guard 2726951, gpu_watch 52045,
# sync 1222453. Refuses an existing a_sv root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/a_sv_orch.sh > logs/mist_w3po_v1/a_sv/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/a_sv_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/a_sv
LOGD=logs/mist_w3po_v1/a_sv
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
S1=eval_outputs/mist_w3po_v1/wave2_S1
V=eval_outputs/mist_w3po_v1/wave3_combos/V
K5=eval_outputs/mist_w3po_v1/k5
OR=eval_outputs/mist_w3po_v1/oracle_voice
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

if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "a_sv root exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/a_sv_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

ecapa_seq() {   # card name "arm:d:run" ...
  local g=$1 name=$2; shift 2
  local cmd="set -e" x arm d run
  for x in "$@"; do
    IFS=: read -r arm d run <<< "$x"
    cmd="$cmd; CUDA_VISIBLE_DEVICES=$g ${NENV[*]} $PYN -m experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_spksim --dir $d --run $run --out $R/metrics/ecapa2_${arm}_${d}_spksim.jsonl --device cuda:0"
  done
  launch $g $name bash -c "$cmd"
}

# 1. features + ECAPA2 on the existing arms
pT=$(launch 4 titanet env CUDA_VISIBLE_DEVICES=4 "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.a_sv_titanet \
  --out $R/feats --device cuda:0)
pC=$(launch 5 codes "${VENV[@]}" CUDA_VISIBLE_DEVICES=5 $PYV -m experiments.mist_w3po_v1.voicecond.a_sv_codes \
  --out $R/feats --device cuda:0)
pE1=$(ecapa_seq 6 ecapa2_old_a B:en2zh:$S1/talker_en2zh_s20261021 B:zh2en:$S1/talker_zh2en_s20261021 \
  v1:en2zh:$V/v1/talker_en2zh v1:zh2en:$V/v1/talker_zh2en)
pE2=$(ecapa_seq 7 ecapa2_old_b k5full:en2zh:$K5/talker_k5full_en2zh k5full:zh2en:$K5/talker_k5full_zh2en \
  oracle:en2zh:$OR/talker_en2zh oracle:zh2en:$OR/talker_zh2en)
waitp "$pT" "$pC"
if [ -z "$DRY" ]; then
  for f in titanet_train.npy titanet_mon.npy titanet_dev_en2zh.npy titanet_dev_zh2en.npy codes_train.npz codes_mon.npz \
           codes_dev_en2zh.npz codes_dev_zh2en.npz; do [ -s $R/feats/$f ] || die "missing feature file $f"; done
fi
log "features done"

# 2. training + precheck + token tables
p=$(launch 4 train "${VENV[@]}" CUDA_VISIBLE_DEVICES=4 $PYV -m experiments.mist_w3po_v1.voicecond.a_sv_train \
  --feats $R/feats --out $R/train --device cuda:0)
waitp "$p"
if [ -z "$DRY" ]; then
  [ -s $R/train/precheck.json ] || die "train produced no precheck.json"
  go=$($PYQ -c "import json; print(int(json.load(open('$R/train/precheck.json'))['go']))") || die "precheck read"
  if [ "$go" != 1 ]; then
    log "PRECHECK STOP (see $R/train/precheck.json); waiting for the ECAPA2 jobs of the existing arms"
    touch $R/STOPPED_PRECHECK
    waitp "$pE1" "$pE2"
    touch $STOP
    exit 0
  fi
fi
log "precheck go"

# 3. Talker runs + metrics
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed 20261021)
run_arm() {   # card arm d
  local g=$1 arm=$2 d=$3 out=$R/talker_$2_$3 s ps=() hook p tab=$R/train/tokens/tokens_a_$3.jsonl
  case $arm in
    A) hook=(-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $tab);;
    Asoft) hook=(-m experiments.mist_w3po_v1.voicecond.k5_si_infer_soft --k5-table $tab --k5-lm 1 --k5-dec 0);;
  esac
  for s in 0 1 2 3; do
    ps+=("$(launch $g talk_${arm}_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV "${hook[@]}" \
      "${COMMON[@]}" --direction $d --output $out --shard $s --num-shards 4 \
      --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --items-file $S1/data/items_$d.jsonl)")
    [ -n "$DRY" ] || sleep 45
  done
  waitp "${ps[@]}"
  for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $out/part_$s.jsonl ] || die "talker $arm $d shard $s ended without part file"; done
  log "talkers $arm $d done"
  local name=${arm}_$d
  p=$(launch $g asr_$name bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
  waitp "$p"
  [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || die "asr $name produced nothing"
  local p1 p2
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
run_arm 4 A en2zh & PA=$!
[ -n "$DRY" ] || sleep 20
run_arm 5 A zh2en & PB=$!
[ -n "$DRY" ] || sleep 20
run_arm 6 Asoft en2zh & PC=$!
[ -n "$DRY" ] || sleep 20
run_arm 7 Asoft zh2en & PD=$!
for p in $PA $PB $PC $PD; do wait $p || die "a chain failed"; done
waitp "$pE1" "$pE2"
if [ -z "$DRY" ]; then
  for arm in B v1 k5full oracle; do for d in en2zh zh2en; do
    [ -s $R/metrics/ecapa2_${arm}_${d}_spksim.jsonl ] || die "missing ECAPA2 metric $arm $d"
  done; done
fi

# 4. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.a_sv_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.a_sv_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
