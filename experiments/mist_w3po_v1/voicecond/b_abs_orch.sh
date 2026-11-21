#!/usr/bin/env bash
# Robust B (AbS-Voice) orchestrator (reports/mist_w3po_v1/robust_b_abs_prereg.zh-CN.md section 6), cards 4, 5, 6, 7 ONLY
# (cards 0-3 belong to another session; launch() refuses any other card). Built from a_sv_orch.sh:
#  1. b_abs_search: one process per card 4-7, process k = shard k (rows mod 4) of en2zh then zh2en -> b_abs/search
#  2. b_abs_precheck (card 4): ECAPA2 resynthesis similarity vs clean twin, token tables, precheck.json;
#     go = false -> b_abs/STOPPED_PRECHECK, no Talker eval
#  3. Talker (S1 layout, seed 20261021, 45 s apart): AbS en2zh shards 0,1 card 4 / 2,3 card 6; zh2en shards 0,1 card 5 /
#     2,3 card 7 -> speech-only ASR -> SQUIM PESQ (CPU) + TitaNet + ECAPA2 (card 4 / 5)
#  4. b_abs_score -> b_abs/summary.json, touch b_abs/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher b_abs_noplaceholder.sh (POLL 1 s). Never touches guard 2726951, gpu_watch 52045,
# sync 1222453. Refuses an existing b_abs root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/b_abs_orch.sh > logs/mist_w3po_v1/b_abs/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/b_abs_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/b_abs
LOGD=logs/mist_w3po_v1/b_abs
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

if [ -z "$DRY" ]; then
  [ -e "$R" ] && { echo "b_abs root exists: $R (refuse to overwrite)" >&2; exit 1; }
  [ -e .tmp/b_abs_pyshim/einx/__init__.py ] && [ -e .tmp/b_abs_pyshim/frozendict/__init__.py ] || { echo "pyshim missing" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/b_abs_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

# 1. search (one process per card)
ps_=()
for k in 0 1 2 3; do
  g=$((4 + k))
  ps_+=("$(launch $g search_$k env CUDA_VISIBLE_DEVICES=$g "${SENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.b_abs_search \
    --dirs en2zh zh2en --shard $k --num-shards 4 --out $R/search --device cuda:0)")
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $R/search/search_${d}_$k.jsonl ] || die "missing search_${d}_$k"; done; done
fi
log "search done"

# 2. precheck
p=$(launch 4 precheck env CUDA_VISIBLE_DEVICES=4 "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.b_abs_precheck \
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

# 3. Talker runs + metrics
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed 20261021)
run_arm() {   # cardA cardB d   (shards 0,1 on cardA, 2,3 on cardB; metrics on cardA)
  local g=$1 g2=$2 d=$3 out=$R/talker_abs_$3 s ps=() p tab=$R/tokens/tokens_abs_$3.jsonl gc
  for s in 0 1 2 3; do
    gc=$g; [ $s -ge 2 ] && gc=$g2
    ps+=("$(launch $gc talk_abs_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$gc $PYV \
      -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $tab \
      "${COMMON[@]}" --direction $d --output $out --shard $s --num-shards 4 \
      --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl --items-file $S1/data/items_$d.jsonl)")
    [ -n "$DRY" ] || sleep 45
  done
  waitp "${ps[@]}"
  for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $out/part_$s.jsonl ] || die "talker abs $d shard $s ended without part file"; done
  log "talkers abs $d done"
  local name=abs_$d
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
run_arm 4 6 en2zh & PA=$!
[ -n "$DRY" ] || sleep 20
run_arm 5 7 zh2en & PB=$!
for p in $PA $PB; do wait $p || die "a chain failed"; done

# 4. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.b_abs_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.b_abs_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
