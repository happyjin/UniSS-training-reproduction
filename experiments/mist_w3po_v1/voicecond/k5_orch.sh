#!/usr/bin/env bash
# Robust K5 orchestrator (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md section 6), cards 0, 6, 4, 5 only
# (copy of oracle_voice_orch.sh):
#  1. k5_tokens (card 0) -> k5/tokens/tokens_k5_<d>.jsonl (V1 hard ids + bounded code b; decoder-centre gate inside)
#  2. sanity (card 0): per direction K5-full in sanity mode and the plain token path, --limit 48, 1 shard, --dump-codes
#     -> speech_only + TitaNet -> k5_sanity_check (gate |median paired sim diff| <= 0.003); fail -> k5/FAILED_SANITY
#  3. main, 4 chains in parallel, each run 4 Talker shards (S1 layout, seed 20261021, 45 s apart) -> speech-only ASR ->
#     SQUIM PESQ (CPU) + TitaNet (same card): card 0 k5full en2zh -> k5dec en2zh; card 6 k5full zh2en -> k5dec zh2en;
#     card 4 k5lm en2zh; card 5 k5lm zh2en
#  4. k5_score -> k5/summary.json, touch k5/DONE (FAILED on any error).
# Every GPU job: placeholders of its card SIGTERMed by PID (argv re-checked), wait <= 120 s until no gpu_load process is
# left on the card; own watcher k5_noplaceholder.sh (POLL 1 s) keeps them off while jobs run. Never touches guard
# 2726951, gpu_watch 52045, sync 1222453, K1 step 2 (cards 1,2,3,7). Refuses an existing k5 root.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/k5_orch.sh > logs/mist_w3po_v1/k5/orch.log 2>&1 < /dev/null 8>&- 9>&- &
#   DRY_RUN=1 bash experiments/mist_w3po_v1/voicecond/k5_orch.sh   (prints the commands only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/k5
LOGD=logs/mist_w3po_v1/k5
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
S1=eval_outputs/mist_w3po_v1/wave2_S1
VT=eval_outputs/mist_w3po_v1/wave3_combos/V/tokens
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 1602215 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { touch $R/FAILED; touch $STOP; }; exit 1; }

launch() {   # card name cmd... ; prints pid (DRY_RUN: prints the command to stderr and "DRY")
  local g=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  case " 0 4 5 6 " in *" $g "*) ;; *) die "card $g not allowed";; esac
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
  [ -e "$R" ] && { echo "k5 root exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p "$LOGD" "$R/tokens" "$R/metrics"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/k5_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
fi

COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed 20261021)
talk_cmd() {   # d out shard nshards table mode(soft|tok) [extra...]
  local d=$1 out=$2 s=$3 ns=$4 tab=$5 mode=$6; shift 6
  local g=$CARD
  local hook
  if [ "$mode" = tok ]; then hook=(-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $tab)
  else hook=(-m experiments.mist_w3po_v1.voicecond.k5_si_infer_soft --k5-table $tab); fi
  echo "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV "${hook[@]}" "$@" "${COMMON[@]}" --direction $d \
    --output $out --shard $s --num-shards $ns --blocks-file $S1/thinker_$d/blocks_${d}_w1.jsonl \
    --items-file $S1/data/items_$d.jsonl
}
metrics() {   # card d run name  (speech_only + ASR if $3 needs it, TitaNet, PESQ)
  local g=$1 d=$2 out=$3 name=$4 asr=$5 p
  if [ "$asr" = 1 ]; then
    p=$(launch $g asr_$name bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$out' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$out/results_speechonly.jsonl' --output '$out/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$out/asr_speech.jsonl'")
    waitp "$p"
    [ -n "$DRY" ] || [ -s $out/asr_speech.jsonl ] || die "asr $name produced nothing"
  else
    if [ -n "$DRY" ]; then echo "[dry] speech_only $out" >&2
    else $PYE reports/uniss_si_v4/speech_only.py "$out" > $LOGD/speechonly_$name.log 2>&1 || die "speech_only $name"; fi
  fi
  p=$(launch $g spksim_$name env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $out --out $R/metrics/${name}_spksim.jsonl --device cuda:0)
  if [ "$asr" = 1 ]; then
    if [ -n "$DRY" ]; then echo "[dry] pesq $name" >&2
    else PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $out \
           --out $R/metrics/${name}_squim.jsonl --procs 16 > $LOGD/pesq_$name.log 2>&1 || die "pesq $name"; fi
  fi
  waitp "$p"
  [ -n "$DRY" ] || [ -s $R/metrics/${name}_spksim.jsonl ] || die "spksim $name produced nothing"
}

# 1. tokens
p=$(launch 0 tokens "${VENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYV -I experiments/mist_w3po_v1/voicecond/k5_tokens.py \
  --out $R/tokens --device cuda:0)
waitp "$p"
for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $R/tokens/tokens_k5_$d.jsonl ] || die "tokens $d missing"; done
log "tokens done"

# 2. sanity (card 0)
CARD=0; ps=()
for d in en2zh zh2en; do
  read -ra c <<< "$(talk_cmd $d $R/sanity_soft_$d 0 1 $R/tokens/tokens_k5_$d.jsonl soft --k5-lm 1 --k5-dec 1 --k5-sanity 1)"
  ps+=("$(launch 0 sanity_soft_$d "${c[@]}" --limit 48 --dump-codes)"); [ -n "$DRY" ] || sleep 45
  read -ra c <<< "$(talk_cmd $d $R/sanity_tok_$d 0 1 $VT/tokens_v1_$d.jsonl tok)"
  ps+=("$(launch 0 sanity_tok_$d "${c[@]}" --limit 48 --dump-codes)"); [ -n "$DRY" ] || sleep 45
done
waitp "${ps[@]}"
for d in en2zh zh2en; do for k in soft tok; do
  [ -n "$DRY" ] || [ -s $R/sanity_${k}_$d/part_0.jsonl ] || die "sanity $k $d ended without part file"
  metrics 0 $d $R/sanity_${k}_$d sanity_${k}_$d 0
done; done
if [ -n "$DRY" ]; then echo "[dry] sanity check" >&2
else
  $PYQ -m experiments.mist_w3po_v1.voicecond.k5_sanity_check --root $R --out $R/sanity.json > $LOGD/sanity_check.log 2>&1
  rc=$?
  [ $rc = 3 ] && { log "SANITY FAILED (see $R/sanity.json)"; touch $R/FAILED_SANITY $STOP; exit 1; }
  [ $rc = 0 ] || die "sanity check error rc=$rc"
fi
log "sanity passed"

# 3. main chains
run_arm() {   # card arm d
  local g=$1 arm=$2 d=$3 out=$R/talker_$2_$3 s ps=() flags
  case $arm in k5full) flags=(--k5-lm 1 --k5-dec 1);; k5dec) flags=(--k5-lm 0 --k5-dec 1);; k5lm) flags=(--k5-lm 1 --k5-dec 0);; esac
  CARD=$g
  for s in 0 1 2 3; do
    read -ra c <<< "$(talk_cmd $d $out $s 4 $R/tokens/tokens_k5_$d.jsonl soft "${flags[@]}")"
    ps+=("$(launch $g talk_${arm}_${d}_$s "${c[@]}")")
    [ -n "$DRY" ] || sleep 45
  done
  waitp "${ps[@]}"
  for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $out/part_$s.jsonl ] || die "talker $arm $d shard $s ended without part file"; done
  log "talkers $arm $d done"
  metrics $g $d $out ${arm}_$d 1
  log "metrics $arm $d done"
}
( run_arm 0 k5full en2zh && run_arm 0 k5dec en2zh ) & PA=$!
[ -n "$DRY" ] || sleep 20
( run_arm 6 k5full zh2en && run_arm 6 k5dec zh2en ) & PB=$!
[ -n "$DRY" ] || sleep 20
run_arm 4 k5lm en2zh & PC=$!
[ -n "$DRY" ] || sleep 20
run_arm 5 k5lm zh2en & PD=$!
for p in $PA $PB $PC $PD; do wait $p || die "a chain failed"; done

# 4. score
if [ -n "$DRY" ]; then echo "[dry] $PYQ -m experiments.mist_w3po_v1.voicecond.k5_score --out $R/summary.json" >&2; exit 0; fi
touch $STOP
$PYQ -m experiments.mist_w3po_v1.voicecond.k5_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
