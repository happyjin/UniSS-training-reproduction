#!/usr/bin/env bash
# Robust line, full-system combination: one arm (reports/mist_w3po_v1/robust_combo_plan_20261120.zh-CN.md, sections 3-4).
# Generalises wave14_combo_orch.sh / wave5_m_combo_orch.sh (same chain, same flags) to any Thinker configuration:
#   1. Thinker   reuse THINKER_FROM=<root with thinker_<d>/> (symlinked), or run combo_si_thinker.py with
#                SV=none|enh|same|budget [BASE=<model dir>] [NCWI=1] [BUDGET_FROM=<root>] on cards CARDS[0] (en2zh) and
#                CARDS[1] (zh2en); NCWI=1 follows wave 16's own outcome (NCWI_ROOT): eval_ncwi -> vLLM multi-LoRA with
#                wave16/fit/assign_<d>.jsonl + wave16/lora; eval_ncwi_merged -> one job per (direction, alpha*) on the
#                merged copy of that alpha (wave16/groups.txt), lanes round-robin over CARDS, then combo_tools assemble.
#                SELFCHECK_REF="<root> [<root> ...]" -> combo_selfcheck.py after the Thinker; FAIL stops the arm.
#                THINKER_ONLY=1 stops after step 1 (self-check arms); NO_V1=1 runs only the raw-voice Talker + ASR
#                (Thinker-level control arms such as the CV-LA equal-latency control).
#   2. raw voice Talker (T1c, S1 flags, si_infer default seed 20261021, 4 shards): reuse RAW_FROM=<root with
#                talker_<d>_s20261021/> or run (2 shards per card at a time) -> <root>/b/talker_<d>
#   3. V1 voice tokens on THIS Thinker's first writes (wave5_m_v1tokens.py) -> V1 Talker (idea1_si_infer_vo, --seed 20261021)
#                -> <root>/v1/talker_<d>
#   4. speech-only ASR (paper protocol) for every new talker run; 5. SQUIM PESQ (CPU) + TitaNet similarity for b and v1
#   -> <root>/DONE. Scoring across arms: combo_score.py (separately).
# Root eval_outputs/mist_w3po_v1/combo/<ARM>/. Logs logs/mist_w3po_v1/combo/<ARM>_*.log; pids -> logs/.../<ARM>.pids, kept
# placeholder-free by combo_noplaceholder.sh (STOP logs/.../<ARM>.stop, touched at the end or on failure). Placeholders of a
# card are stopped by PID right before each launch. Never touches guard 2726951, gpu_watch 52045, sync 1222453.
#   ARM=full SV=enh NCWI=1 CARDS="0 1 2 3" setsid nohup bash .../combo_orch.sh > logs/mist_w3po_v1/combo/full_orch.log 2>&1 < /dev/null 8>&- 9>&- &
# DRY_RUN=1 prints every command instead of launching (no file is written except nothing; checks on inputs are reported,
# not enforced). Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${CARDS:?}"
read -ra CA <<< "$CARDS"; NC=${#CA[@]}
[ "$NC" -ge 2 ] || { echo "CARDS needs >= 2 cards" >&2; exit 1; }
for c in "${CA[@]}"; do case " 0 1 2 3 4 5 6 7 " in *" $c "*) ;; *) echo "bad card $c" >&2; exit 1;; esac; done
DRY=${DRY_RUN:-}
M=eval_outputs/mist_w3po_v1/combo
R=$M/$ARM
LOGD=logs/mist_w3po_v1/combo
export PIDS=$LOGD/$ARM.pids
STOP=$LOGD/$ARM.stop
W15=eval_outputs/mist_w3po_v1/wave15
W16=${COMBO_W16:-eval_outputs/mist_w3po_v1/wave16}   # override only for DRY_RUN tests
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
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
need() { [ -e "$1" ] || die "missing $1"; }
nrows() { wc -l < "$1" 2>/dev/null || echo 0; }
PROTECT=" 2726951 52045 1222453 $$ "
launch() {   # card name cmd... ; prints pid (DRY_RUN: prints the command to stderr and "DRY")
  local g=$1 name=$2; shift 2
  local lf=$LOGD/${ARM}_$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  if [ "$g" != "-" ]; then
    for pat in run_mist_workload.sh train_mist_workload.py; do
      while read -r pid args; do
        case "$PROTECT" in *" $pid "*) continue;; esac
        case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
        a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
        case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
      done < <(ps -eo pid=,args=)
    done
    sleep 8
    # 18:50 CET fix (arm C failed: a 126 GB placeholder still held card 0 when its Talker started): wait until no
    # gpu_load process remains on card g (re-SIGTERM any respawn), at most 120 s
    local k
    for k in $(seq 1 24); do
      local left=""
      while read -r pid args; do
        case "$PROTECT" in *" $pid "*) continue;; esac
        case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
      done < <(ps -eo pid=,args=)
      [ -z "$left" ] && break
      sleep 5
    done
    [ -n "${left:-}" ] && log "WARNING: placeholders still on card $g after 120 s:$left" >&2
  fi
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
  mkdir -p "$LOGD"
  [ -e "$R" ] && die "arm root exists: $R (refuse to overwrite)"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  mkdir -p "$R"
  POLL=${WATCH_POLL:-1} STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/combo_noplaceholder.sh > "$LOGD/${ARM}_noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$ arm $ARM; watcher $!"
  env | grep -E '^(ARM|CARDS|SV|BASE|NCWI|BUDGET_FROM|THINKER_FROM|RAW_FROM|SELFCHECK_REF|THINKER_ONLY|NO_V1)=' | sort > "$R/config.env"
  git rev-parse HEAD >> "$R/config.env"
else
  log "[dry] arm $ARM root $R cards ${CA[*]}"
fi

# ---------------------------------------------------------------- 1. Thinker
thinker_args() {   # $1 = direction ; echoes the switch arguments of combo_si_thinker (one per line)
  local d=$1
  echo --second-view; echo "${SV:?SV needed when THINKER_FROM is unset}"
  [ "$SV" = enh ] && { echo --views; echo "$W15/views/$d"; }
  [ "$SV" = budget ] && { echo --budget-from; echo "${BUDGET_FROM:?}/thinker_$d/blocks_${d}_w1.jsonl"; }
  [ "$SV" = budget ] && [ -n "${BUDGET_POOL:-}" ] && { echo --budget-pool-meta; echo "eval_outputs/mist_w3po_v1/wave2_S1/data/meta_$d.jsonl"; }
  return 0
}
if [ -n "${THINKER_FROM:-}" ]; then
  for d in "${DIRS[@]}"; do
    need "$THINKER_FROM/thinker_$d/blocks_${d}_w1.jsonl"
    [ -n "$DRY" ] || ln -s "$(realpath "$THINKER_FROM/thinker_$d")" "$R/thinker_$d"
  done
  log "thinker reused from $THINKER_FROM"
else
  MODE=single
  if [ "${NCWI:-0}" = 1 ]; then
    need "$W16/NCWI_ROOT"
    nr=$(cat "$W16/NCWI_ROOT" 2>/dev/null || echo "?")
    case "$nr" in eval_ncwi) MODE=lora;; eval_ncwi_merged) MODE=merged;; *) die "wave 16 NCWI_ROOT = $nr";; esac
    [ -z "${BASE:-}" ] || [ "$MODE" = lora ] || die "BASE with the merged NCWI fallback is not defined (plan section 5)"
  fi
  log "thinker mode $MODE, second view ${SV:-?}"
  if [ "$MODE" != merged ]; then
    tp=(); k=0
    for d in "${DIRS[@]}"; do
      mapfile -t SW < <(thinker_args "$d")
      EX=()
      [ "$MODE" = lora ] && EX=(--assign "$W16/fit/assign_$d.jsonl" --lora-root "$W16/lora")
      g=${CA[$k]}; k=$((k + 1))
      tp+=($(launch "$g" thinker_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.combo_si_thinker \
             "${SW[@]}" "${EX[@]}" --thinker "${BASE:-$QWEN3}" --direction "$d" --gpu-mem "${GPU_MEM:-0.85}" \
             --out "$R/thinker_$d" --items-file "$DATA/items_$d.jsonl"))
      nap 30
    done
    waitp "${tp[@]}"
  else
    jobs_=()
    for d in "${DIRS[@]}"; do
      while read -r dd al model items n; do jobs_+=("$dd|$al|$model|$items|$n"); done < <("$PYQ" experiments/mist_w3po_v1/voicecond/combo_tools.py groups --dir "$d")
    done
    [ "${#jobs_[@]}" -gt 0 ] || die "no merged groups"
    lp=()
    for li in "${!CA[@]}"; do
      (
        for ((j = li; j < ${#jobs_[@]}; j += NC)); do
          IFS='|' read -r d al model items n <<< "${jobs_[$j]}"
          mapfile -t SW < <(thinker_args "$d")
          g=${CA[$li]}
          p=$(launch "$g" thinker_${d}_a$al "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.combo_si_thinker \
              "${SW[@]}" --thinker "$model" --direction "$d" --gpu-mem "${GPU_MEM:-0.85}" \
              --out "$R/parts/thinker_${d}_a$al" --items-file "$items") || exit 1
          waitp $p
          [ -n "$DRY" ] || [ "$(nrows "$R/parts/thinker_${d}_a$al/blocks_${d}_w1.jsonl")" = "$n" ] || exit 1
          nap 10
        done
      ) &
      lp+=($!)
    done
    for p in "${lp[@]}"; do wait "$p" || die "merged lane $p failed"; done
    for d in "${DIRS[@]}"; do
      if [ -n "$DRY" ]; then log "[dry] assemble $d"; else
        "$PYQ" experiments/mist_w3po_v1/voicecond/combo_tools.py assemble --root "$R" --dir "$d" > "$LOGD/${ARM}_assemble_$d.log" 2>&1 || die "assemble $d"
      fi
    done
  fi
  for d in "${DIRS[@]}"; do
    [ -n "$DRY" ] || [ "$(nrows "$R/thinker_$d/blocks_${d}_w1.jsonl")" = "$(nrows "$DATA/items_$d.jsonl")" ] || die "thinker $d incomplete"
  done
  log "thinker done"
fi
if [ -n "${SELFCHECK_REF:-}" ]; then
  RA=(); for x in $SELFCHECK_REF; do RA+=(--ref "$x"); done
  if [ -n "$DRY" ]; then log "[dry] selfcheck vs $SELFCHECK_REF"; else
    "$PYQ" experiments/mist_w3po_v1/voicecond/combo_selfcheck.py --run "$R" "${RA[@]}" --out "$R/selfcheck.json" > "$LOGD/${ARM}_selfcheck.log" 2>&1 \
      || die "self-check FAIL (see $R/selfcheck.json)"
  fi
  log "self-check PASS"
fi
if [ "${THINKER_ONLY:-0}" = 1 ]; then
  [ -n "$DRY" ] || { date > "$R/DONE"; touch "$STOP"; }
  log "thinker-only arm done"; exit 0
fi

# ---------------------------------------------------------------- 2.-3. talkers (raw voice if needed, V1)
talk_b() {   # raw voice; $1 = direction, $2 = shard, $3 = card
  local d=$1 s=$2 g=$3
  launch "$g" talk_b_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer \
    --model "$TALKER" --direction "$d" --window 1 --output "$R/b/talker_$d" --shard "$s" --num-shards 4 \
    --text-policy file --blocks-file "$R/thinker_$d/blocks_${d}_w1.jsonl" --speed-policy block --gpu-mem 0.1 \
    --items-file "$DATA/items_$d.jsonl"
}
talk_v1() {
  local d=$1 s=$2 g=$3
  launch "$g" talk_v1_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g "$PYV" \
    -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override "$R/tokens/tokens_v1_$d.jsonl" \
    --model "$TALKER" --direction "$d" --window 1 --output "$R/v1/talker_$d" --shard "$s" --num-shards 4 \
    --text-policy file --blocks-file "$R/thinker_$d/blocks_${d}_w1.jsonl" --speed-policy block --gpu-mem 0.1 \
    --items-file "$DATA/items_$d.jsonl" --seed 20261021
}
NEW_RUNS=(v1)
[ -n "$DRY" ] || mkdir -p "$R/b" "$R/v1" "$R/metrics"
if [ -n "${RAW_FROM:-}" ]; then
  for d in "${DIRS[@]}"; do
    need "$RAW_FROM/talker_${d}_s20261021/asr_speech.jsonl"
    [ -n "$DRY" ] || ln -s "$(realpath "$RAW_FROM/talker_${d}_s20261021")" "$R/b/talker_$d"
  done
  log "raw-voice talker reused from $RAW_FROM"
else
  NEW_RUNS=(b v1)
fi
if [ "${NO_V1:-0}" = 1 ]; then   # Thinker-level control arms: raw voice + ASR only (no V1, no PESQ / similarity)
  [ -z "${RAW_FROM:-}" ] || die "NO_V1 with RAW_FROM leaves nothing to run"
  NEW_RUNS=(b)
fi
# V1 tokens (cards 0, 1 of the list)
tp=(); k=0
[ "${NO_V1:-0}" = 1 ] || {
for d in "${DIRS[@]}"; do
  g=${CA[$k]}; k=$((k + 1))
  tp+=($(launch "$g" v1tok_$d env PYTHONPATH=. CUDA_VISIBLE_DEVICES=$g "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
         --dir "$d" --blocks "$R/thinker_$d/blocks_${d}_w1.jsonl" --out-root "$R"))
done
waitp "${tp[@]}"
for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/tokens/tokens_v1_$d.jsonl" ] || die "no V1 tokens $d"; done
}
# talkers in waves of 2 per card
for run in "${NEW_RUNS[@]}"; do
  ps_=(); k=0
  for d in "${DIRS[@]}"; do for s in 0 1 2 3; do
    g=${CA[$(( (k / 2) % NC ))]}; k=$((k + 1))
    ps_+=($(talk_$run "$d" "$s" "$g"))
    nap 25
    if [ $((k % (2 * NC))) = 0 ]; then waitp "${ps_[@]}"; ps_=(); fi
  done; done
  waitp "${ps_[@]}"
  for d in "${DIRS[@]}"; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s "$R/$run/talker_$d/part_$s.jsonl" ] || die "no talker part $run $d $s"; done; done
  log "talkers $run done"
done

# ---------------------------------------------------------------- 4. ASR
ps_=(); k=0
for run in "${NEW_RUNS[@]}"; do for d in "${DIRS[@]}"; do
  g=${CA[$((k % NC))]}; k=$((k + 1)); T=$R/$run/talker_$d
  ps_+=($(launch "$g" asr_${run}_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$T' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$T/results_speechonly.jsonl' --output '$T/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$T/asr_speech.jsonl'"))
  nap 10
done; done
waitp "${ps_[@]}"
for run in "${NEW_RUNS[@]}"; do for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/$run/talker_$d/asr_speech.jsonl" ] || die "no asr $run $d"; done; done
log "asr done"

if [ "${NO_V1:-0}" = 1 ]; then
  [ -n "$DRY" ] || { date > "$R/DONE"; touch "$STOP"; }
  log "arm $ARM (raw voice only) done: $R"; exit 0
fi
# ---------------------------------------------------------------- 5. PESQ (CPU) + similarity
ps_=()
for run in b v1; do for d in "${DIRS[@]}"; do
  if [ -n "$DRY" ]; then log "[dry] pesq $run $d"; continue; fi
  PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run "$R/$run/talker_$d" \
    --out "$R/metrics/${run}_${d}_squim.jsonl" --procs 24 > "$LOGD/${ARM}_pesq_${run}_$d.log" 2>&1 & ps_+=($!)
done; done
k=0
for run in b v1; do
  g=${CA[$((k % NC))]}; k=$((k + 1))
  ps_+=($(launch "$g" spksim_$run bash -c "set -e; for d in en2zh zh2en; do CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir \$d --run $R/$run/talker_\$d --out $R/metrics/${run}_\${d}_spksim.jsonl --device cuda:0; done"))
done
waitp "${ps_[@]}"
for run in b v1; do for d in "${DIRS[@]}"; do for m in squim spksim; do
  [ -n "$DRY" ] || [ -s "$R/metrics/${run}_${d}_$m.jsonl" ] || die "no $m $run $d"; done; done; done
[ -n "$DRY" ] || { date > "$R/DONE"; touch "$STOP"; }
log "arm $ARM done: $R (score with combo_score.py)"
