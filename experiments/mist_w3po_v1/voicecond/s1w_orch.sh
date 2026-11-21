#!/usr/bin/env bash
# Robust S1 "witness algebra" orchestrator (reports/mist_w3po_v1/robust_s1_witness_prereg.zh-CN.md). Detached waiter +
# launcher + scorer, inference only:
#   0. wait until K1 step 2 has its final marker (eval_outputs/mist_w3po_v1/k1/step2/DONE or FAILED, k1_step2_chain.sh)
#      AND no process running experiments/mist_w3po_v1/voicecond/k1_* remains (poll 60 s, at most MAX_WAIT_S = 30 h);
#   1. own watcher s1w_noplaceholder.sh (POLL=1) keeps scripts/gpu_load placeholders off cards with a live s1w Thinker;
#   2. per card (1, 2, 3, 7; never 0, 4, 5, 6): SIGTERM the card's placeholders by PID (argv re-checked in /proc), wait
#      <= 120 s until none remain, then one s1w_si_thinker.py process (one vLLM Thinker, --gpu-mem 0.85) with its job list;
#   3. all 10 outputs present -> STOP the watcher, s1w_score.py -> <R>/score.json + score.txt, <R>/DONE (else <R>/FAILED).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453. Refuses to overwrite. Checked with bash -n / DRY_RUN=1 only.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/s1w_orch.sh > logs/mist_w3po_v1/s1w/orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/mist_w3po_v1/s1w
LOGD=logs/mist_w3po_v1/s1w
K1M=eval_outputs/mist_w3po_v1/k1/step2
PIDS=$LOGD/s1w.pids
STOP=$LOGD/s1w.stop
MAX_WAIT_S=${MAX_WAIT_S:-108000}
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
declare -A JOBS=([1]="aov:zh2en,aov:en2zh" [2]="either:zh2en,either:en2zh"
                 [3]="la1:zh2en,la1:en2zh,la3:en2zh" [7]="la2:zh2en,la2:en2zh,la3:zh2en")
CARDS="1 2 3 7"
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; [ -n "$DRY" ] && return 0; echo "$*" > "$R/FAILED"; touch "$STOP"; exit 1; }

# ---------------------------------------------------------------- 0. set-up + K1 waiter
for rule in la1 la2 la3 aov either; do [ -e "$R/$rule" ] && { log "refuse: $R/$rule exists"; exit 1; }; done
for f in DONE FAILED score.json; do [ -e "$R/$f" ] && { log "refuse: $R/$f exists"; exit 1; }; done
[ -e "$STOP" ] && { log "refuse: $STOP exists"; exit 1; }
[ -e "$PIDS" ] && { log "refuse: $PIDS exists"; exit 1; }
if [ -z "$DRY" ]; then mkdir -p "$R" "$LOGD"; fi
k1_live() {   # pids of processes running a k1_* script of voicecond (ps fields, argv from /proc)
  local pid args out=""
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *voicecond/k1_*|*voicecond.k1_*) out="$out $pid";; esac
  done < <(ps -eo pid=,args=)
  echo "$out"
}
log "orch $$: waiting for $K1M/DONE|FAILED and no k1_ process"
t0=$(date +%s)
while :; do
  if [ -e "$K1M/DONE" ] || [ -e "$K1M/FAILED" ]; then
    left=$(k1_live)
    [ -z "$left" ] && break
  fi
  [ -n "$DRY" ] && { log "[dry] K1 not finished (marker: $(ls $K1M/DONE $K1M/FAILED 2>/dev/null)); live k1:$(k1_live | wc -w) procs; continuing dry"; break; }
  [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT_S" ] && { log "give up: K1 not finished after $MAX_WAIT_S s"; exit 1; }
  sleep 60
done
log "K1 step 2 finished: $(ls $K1M/DONE $K1M/FAILED 2>/dev/null | tr '\n' ' ')"
if [ -z "$DRY" ]; then
  : > "$PIDS"
  POLL=1 PIDS=$PIDS STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/s1w_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher $!"
  git rev-parse HEAD > "$R/config.txt"
  declare -p JOBS >> "$R/config.txt"
fi

# ---------------------------------------------------------------- 1. launch
stop_placeholders() {   # card
  local g=$1 pid args a k left
  while read -r pid args; do
    case "$PROTECT" in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g";; esac
  done < <(ps -eo pid=,args=)
  for k in $(seq 1 24); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && return 0
    sleep 5
  done
  log "WARNING: placeholders still on card $g after 120 s:$left"
}
TP=()
for g in $CARDS; do
  lf=$LOGD/thinker_card$g.log
  cmd=("${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.s1w_si_thinker --jobs "${JOBS[$g]}"
       --out-root "$R" --data eval_outputs/mist_w3po_v1/wave2_S1/data --views-root eval_outputs/mist_w3po_v1/wave15/views --gpu-mem 0.85)
  if [ -n "$DRY" ]; then printf '[dry] card %s: ' "$g"; printf '%q ' "${cmd[@]}"; echo; continue; fi
  [ -e "$lf" ] && fail "log exists: $lf"
  stop_placeholders "$g"
  setsid nohup "${cmd[@]}" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  p=$!
  echo "$g $p s1w_card$g" >> "$PIDS"
  log "card $g pid $p jobs ${JOBS[$g]}"
  TP+=("$p")
  sleep 20   # stagger engine starts
done
[ -n "$DRY" ] && { log "[dry] would wait for Thinkers, then $PYE experiments/mist_w3po_v1/voicecond/s1w_score.py --out $R/score.json"; exit 0; }
for p in "${TP[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done
touch "$STOP"
log "Thinkers exited; watcher STOP touched"

# ---------------------------------------------------------------- 2. check + score
miss=""
for g in $CARDS; do
  for j in ${JOBS[$g]//,/ }; do
    rule=${j%%:*}; d=${j##*:}
    f=$R/$rule/thinker_$d/blocks_${d}_w1.jsonl
    [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -eq "$(wc -l < eval_outputs/mist_w3po_v1/wave2_S1/data/items_$d.jsonl)" ] || miss="$miss $f"
  done
done
[ -n "$miss" ] && fail "missing / short outputs:$miss"
PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/s1w_score.py --out "$R/score.json" > "$R/score.txt" 2> "$LOGD/score.err" \
  || fail "score failed (see $LOGD/score.err)"
date > "$R/DONE"
log "DONE: $R/score.txt"
