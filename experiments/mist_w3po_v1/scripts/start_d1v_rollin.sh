#!/usr/bin/env bash
# D1v roll-in (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md §1.2, §2.2-2.3): the VE1 learner
# (crf/train/ve1_20261111T214454Z step_03000) decides WHEN to write on VE1's training windows (crf/sched_t5_merge3_w10_v1,
# every window of the 3000-step plan) with crf/d1_rollin.py --score --scorer v2 (forward passes only), one 2-card shard
# per pair in PAIRS; every shard first runs the GPU selfcheck (20 windows with >= 3 reads, per-boundary prefix forwards
# vs the one-forward batch: must agree 20/20, else the shard exits and no manifest is made); then the manifest (CPU,
# --manifest --lw-mode v2p) with the R4 line (>= 5 % of the plan windows ever overdue).
# Eval-class job: stacks on NON-training cards only (a pair touching a slot whose owner file says "training" is refused);
# per shard, under flock(gpu_launch.lock), every card's nvidia-smi used + NEED_MIB <= 138240 MiB (135 GiB), launched
# detached with the lock fd closed, the lock kept until the shard's memory shows (or 300 s) -> launches are staggered.
# Refuses to start unless the R0 CPU precheck (crf/d1v_scorer_precheck_v1.json) passed.
#   PAIRS="4,5 6,7" [NEED_MIB=40000] [MEMCAP=60] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1v_rollin.sh
# Run detached: setsid nohup bash .../start_d1v_rollin.sh > logs/mist_w3po_v1/d1v/start_d1v_rollin.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
  TRITON_CACHE_DIR=$J/cache/triton
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
PAIRS=${PAIRS:?set PAIRS, e.g. "4,5 6,7"}
NEED=${NEED_MIB:-40000}
C=eval_outputs/mist_w3po_v1/crf
S=$C/sched_t5_merge3_w10_v1
M=v2p
CK=$C/train/ve1_20261111T214454Z/ckpt/step_03000
ROOT=$C/d1_rollin_ve1_v1
MAN=$C/d1_ve1_w10_v1
R0=$C/d1v_scorer_precheck_v1.json
L=logs/mist_w3po_v1/d1v
SL=logs/mist_w3po_v1/slots
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
slot_of() { case $1 in 0|1|2|3) echo slotA;; *) echo slotB;; esac; }
training_slot() { local s; s=$(slot_of $1); [ -e $SL/$s.owner ] && grep -qi 'training' $SL/$s.owner; }
read -r -a PA <<< "$PAIRS"
N=${#PA[@]}
cmd_of() { echo "CUDA_VISIBLE_DEVICES=${PA[$1]} $PY -m experiments.mist_w3po_v1.crf.d1_rollin --score --scorer v2 --sched $S --ckpt $CK --root $ROOT --shard $1/$N --selfcheck 20 --selfcheck-min-reads 3 --mem-cap-gib ${MEMCAP:-60}"; }
BADPAIR=""
for p in "${PA[@]}"; do for c in ${p//,/ }; do training_slot $c && BADPAIR="$BADPAIR $c"; done; done
if [ -n "${DRY_RUN:-}" ]; then
  echo "R0 precheck pass: $(field $R0 "d['GO']['pass']" 2>/dev/null || echo missing); learner $CK DONE: $([ -e $CK/DONE ] && echo yes || echo no)"
  echo "root $ROOT ($([ -e $ROOT ] && echo EXISTS || echo new)); manifest $MAN ($([ -e $MAN ] && echo EXISTS || echo new))"
  echo "cards of PAIRS in a training slot (refused):${BADPAIR:- none}"
  for c in $(echo $PAIRS | tr ', ' '  '); do echo "card $c used MiB: $(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"; done
  for k in $(seq 0 $((N - 1))); do echo "shard $k: $(cmd_of $k)"; done
  echo "manifest: $PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN"
  exit 0
fi
[ "$(field $R0 "d['GO']['pass']" 2>/dev/null)" = True ] || { log "R0 CPU precheck missing or failed: not started"; exit 1; }
[ -e $CK/DONE ] || { log "learner checkpoint $CK/DONE missing"; exit 1; }
[ -z "$BADPAIR" ] || { log "cards$BADPAIR belong to a training slot: not started (eval-class job, never on training cards)"; exit 1; }
[ -e $ROOT ] || [ -e $MAN ] && { log "拒绝覆盖已有输出: $ROOT / $MAN"; exit 1; }
mkdir -p $L
for k in $(seq 0 $((N - 1))); do
  LOGK=$L/rollin_ve1_${k}of${N}.log
  [ -e $LOGK ] && { log "拒绝覆盖已有日志: $LOGK"; exit 1; }
done
for k in $(seq 0 $((N - 1))); do
  LOGK=$L/rollin_ve1_${k}of${N}.log
  exec 9>"$LOCK"
  flock -w 1800 9 || { log "gpu_launch.lock busy for 1800 s"; exit 1; }
  ok=0
  until [ $ok = 1 ]; do      # wait (lock held only briefly per try) until every card of the pair has room
    ok=1
    for c in ${PA[$k]//,/ }; do
      used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ $((used + NEED)) -le 138240 ] || ok=0
      training_slot $c && { flock -u 9; log "card $c became a training card: shard $k not started"; exit 1; }
    done
    if [ $ok = 0 ]; then flock -u 9; sleep 60; flock -w 1800 9 || { log "lock busy"; exit 1; }; fi
  done
  first=${PA[$k]%%,*}
  base=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  setsid nohup bash -c "$(cmd_of $k); echo D1_ROLLIN_EXIT rc=\$?" < /dev/null >> "$LOGK" 2>&1 9>&- &
  log "shard $k/$N on ${PA[$k]} pid $! log $LOGK"
  for i in $(seq 1 60); do
    sleep 5
    now=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ $((now - base)) -gt 12000 ] && break
  done
  flock -u 9
  exec 9>&-
done
log "all $N shards launched; waiting for their exits"
fail=0
for k in $(seq 0 $((N - 1))); do
  until grep -q D1_ROLLIN_EXIT $L/rollin_ve1_${k}of${N}.log 2>/dev/null; do sleep 30; done
  if grep -q 'D1_ROLLIN_EXIT rc=0' $L/rollin_ve1_${k}of${N}.log; then
    log "shard $k done: $(grep -m1 '^selfcheck' $L/rollin_ve1_${k}of${N}.log | cut -c1-200)"
  else
    log "shard $k FAILED: $(grep -m1 -E '^selfcheck|Error|error' $L/rollin_ve1_${k}of${N}.log | cut -c1-300)"; fail=1
  fi
done
[ $fail = 0 ] || { log "D1V_ROLLIN_FAILED: no manifest (selfcheck details: $ROOT/rollin_*/selfcheck.json)"; exit 1; }
$PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN > $L/manifest_ve1.log 2>&1 \
  && log "D1V_ROLLIN_DONE manifest $MAN: R4 $(field $MAN/summary.json "d['R4_pass']") (windows with an overdue read: $(field $MAN/summary.json "d['share_windows_with_overdue']"))" \
  || { log "manifest failed (see $L/manifest_ve1.log)"; exit 1; }
