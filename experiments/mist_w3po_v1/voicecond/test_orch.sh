#!/usr/bin/env bash
# Robust line, FINAL TEST: one system of reports/mist_w3po_v1/robust_test_prereg_20261120.zh-CN.md section 6, run ONCE on the
# merged test data eval_outputs/mist_w3po_v1/test_robust/data/{items,meta}_<dir>.jsonl (test_data_build.py).
# A copy of combo_orch.sh (same launch() with the placeholder wait, same Talker / V1 / ASR / PESQ / similarity flags) with:
#   * data root = the test data (S1_DATA exported for wave5_m_v1tokens / wave15_enh_views / wave3_v_spksim);
#   * Thinker TK=stream: si_thinker_stream --policy la2 --windows 1 --thinker ${BASE:-base}  (systems B, EW; as wave2 S1 /
#     wave5_m_eval.sh), or TK=combo SV=enh: combo_si_thinker --second-view enh (system C) after DeepFilterNet3 views of the
#     test audio (wave15_enh_views.py, CPU, xtranslator venv) -> <M>/views/<dir>;
#   * RUNS="b" (raw voice) or "b v1"; PESQ (CPU) + TitaNet similarity for every run in RUNS;
#   * own watcher test_noplaceholder.sh (POLL 1 s; patterns incl. si_thinker_stream).
# Root eval_outputs/mist_w3po_v1/test_robust/<ARM>/, logs logs/mist_w3po_v1/test_robust/<ARM>_*.log; refuses an existing root.
#   ARM=B TK=stream RUNS=b CARDS="0 1" setsid nohup bash .../test_orch.sh > logs/mist_w3po_v1/test_robust/B_orch.log 2>&1 < /dev/null 8>&- 9>&- &
# Never touches guard 2726951, gpu_watch 52045, sync 1222453. DRY_RUN=1 prints the commands. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${ARM:?}" "${CARDS:?}" "${TK:?}" "${RUNS:?}"
read -ra CA <<< "$CARDS"; NC=${#CA[@]}
[ "$NC" -ge 2 ] || { echo "CARDS needs >= 2 cards" >&2; exit 1; }
for c in "${CA[@]}"; do case " 0 1 2 3 4 5 " in *" $c "*) ;; *) echo "bad card $c (test uses cards 0-5 only)" >&2; exit 1;; esac; done
read -ra RU <<< "$RUNS"
DRY=${DRY_RUN:-}
M=eval_outputs/mist_w3po_v1/test_robust
R=$M/$ARM
LOGD=logs/mist_w3po_v1/test_robust
export PIDS=$LOGD/$ARM.pids
STOP=$LOGD/$ARM.stop
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
PYX=/opt/dlami/nvme/jasonleeeli/venvs/xtranslator/bin/python
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
DATA=$M/data
export S1_DATA=/opt/dlami/nvme/jasonleeeli/projects/UniSS/$DATA
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
for d in "${DIRS[@]}"; do need "$DATA/items_$d.jsonl"; need "$DATA/meta_$d.jsonl"; done
if [ -z "$DRY" ]; then
  mkdir -p "$LOGD"
  [ -e "$R" ] && die "arm root exists: $R (refuse to overwrite)"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  mkdir -p "$R"
  POLL=${WATCH_POLL:-1} STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/test_noplaceholder.sh > "$LOGD/${ARM}_noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$ arm $ARM; watcher $!"
  env | grep -E '^(ARM|CARDS|TK|SV|BASE|RUNS|S1_DATA)=' | sort > "$R/config.env"
  git rev-parse HEAD >> "$R/config.env"
else
  log "[dry] arm $ARM root $R cards ${CA[*]}"
fi

# ---------------------------------------------------------------- 0. enhanced views (system C only; shared by arms)
if [ "$TK" = combo ] && [ "${SV:-}" = enh ]; then
  vp=()
  for d in "${DIRS[@]}"; do
    if [ -s "$M/views/$d/DONE.json" ]; then log "views $d exist"; continue; fi
    [ -e "$M/views/$d" ] && die "views $d incomplete (no DONE.json): $M/views/$d"
    vp+=($(launch - views_$d env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 S1_DATA="$S1_DATA" "$PYX" -I \
           experiments/mist_w3po_v1/voicecond/wave15_enh_views.py --direction "$d" --out "$M/views/$d" --workers "${WORKERS:-24}"))
  done
  waitp "${vp[@]}"
  for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$M/views/$d/DONE.json" ] || die "views $d"; done
  log "views done"
fi

# ---------------------------------------------------------------- 1. Thinker
tp=(); k=0
for d in "${DIRS[@]}"; do
  g=${CA[$k]}; k=$((k + 1))
  case "$TK" in
    stream) tp+=($(launch "$g" thinker_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
                   --thinker "${BASE:-$QWEN3}" --direction "$d" --windows 1 --policy la2 --gpu-mem "${GPU_MEM:-0.85}" \
                   --out "$R/thinker_$d" --items-file "$DATA/items_$d.jsonl")) ;;
    combo)  tp+=($(launch "$g" thinker_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.combo_si_thinker \
                   --second-view "${SV:?}" --views "$M/views/$d" --thinker "${BASE:-$QWEN3}" --direction "$d" --gpu-mem "${GPU_MEM:-0.85}" \
                   --out "$R/thinker_$d" --items-file "$DATA/items_$d.jsonl")) ;;
    *) die "TK = $TK" ;;
  esac
  nap 30
done
waitp "${tp[@]}"
for d in "${DIRS[@]}"; do
  [ -n "$DRY" ] || [ "$(nrows "$R/thinker_$d/blocks_${d}_w1.jsonl")" = "$(nrows "$DATA/items_$d.jsonl")" ] || die "thinker $d incomplete"
done
log "thinker done"

# ---------------------------------------------------------------- 2.-3. V1 tokens, talkers
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
[ -n "$DRY" ] || mkdir -p "$R/metrics"
for run in "${RU[@]}"; do [ -n "$DRY" ] || mkdir -p "$R/$run"; done
case " $RUNS " in *" v1 "*)
  tp=(); k=0
  for d in "${DIRS[@]}"; do
    g=${CA[$k]}; k=$((k + 1))
    tp+=($(launch "$g" v1tok_$d env PYTHONPATH=. S1_DATA="$S1_DATA" CUDA_VISIBLE_DEVICES=$g "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
           --dir "$d" --blocks "$R/thinker_$d/blocks_${d}_w1.jsonl" --out-root "$R"))
  done
  waitp "${tp[@]}"
  for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/tokens/tokens_v1_$d.jsonl" ] || die "no V1 tokens $d"; done ;;
esac
for run in "${RU[@]}"; do
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
for run in "${RU[@]}"; do for d in "${DIRS[@]}"; do
  g=${CA[$((k % NC))]}; k=$((k + 1)); T=$R/$run/talker_$d
  ps_+=($(launch "$g" asr_${run}_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$T' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$T/results_speechonly.jsonl' --output '$T/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$T/asr_speech.jsonl'"))
  nap 10
done; done
waitp "${ps_[@]}"
for run in "${RU[@]}"; do for d in "${DIRS[@]}"; do [ -n "$DRY" ] || [ -s "$R/$run/talker_$d/asr_speech.jsonl" ] || die "no asr $run $d"; done; done
log "asr done"

# ---------------------------------------------------------------- 5. PESQ (CPU) + similarity
ps_=()
for run in "${RU[@]}"; do for d in "${DIRS[@]}"; do
  if [ -n "$DRY" ]; then log "[dry] pesq $run $d"; continue; fi
  PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run "$R/$run/talker_$d" \
    --out "$R/metrics/${run}_${d}_squim.jsonl" --procs 16 > "$LOGD/${ARM}_pesq_${run}_$d.log" 2>&1 & ps_+=($!)
done; done
k=0
for run in "${RU[@]}"; do
  g=${CA[$((k % NC))]}; k=$((k + 1))
  ps_+=($(launch "$g" spksim_$run bash -c "set -e; for d in en2zh zh2en; do S1_DATA=$S1_DATA CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir \$d --run $R/$run/talker_\$d --out $R/metrics/${run}_\${d}_spksim.jsonl --device cuda:0; done"))
done
waitp "${ps_[@]}"
for run in "${RU[@]}"; do for d in "${DIRS[@]}"; do for m in squim spksim; do
  [ -n "$DRY" ] || [ -s "$R/metrics/${run}_${d}_$m.jsonl" ] || die "no $m $run $d"; done; done; done
[ -n "$DRY" ] || { date > "$R/DONE"; touch "$STOP"; }
log "arm $ARM done: $R (score with test_score.py)"
