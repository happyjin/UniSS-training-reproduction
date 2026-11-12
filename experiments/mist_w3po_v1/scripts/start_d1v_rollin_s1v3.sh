#!/usr/bin/env bash
# D1v roll-in, shard 1/2 rerun WITHOUT the GPU selfcheck (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md
# amendment 2: the coordinator / user accepted shard 1's amendment-1 selfcheck (749 decisions, 99.07 %, one confident
# flip off the roll-in trajectory) as a documented deviation). Shard 0/2 passed amendment 1 and is reused as is
# (crf/d1_rollin_ve1_v2/rollin_0of2, linked into the new root). Additive: new root crf/d1_rollin_ve1_v3, new log; the
# manifest goes to crf/d1_ve1_w10_v1 (the path start_d1v.sh waits for), with the R4 line.
# Eval-class job on PAIR (default 2,3): refused on a training slot; under flock(gpu_launch.lock) every card's used +
# NEED_MIB <= 135 GiB; launched detached with the lock fd closed.
#   DECISION="<coordinator decision>" [PAIR=2,3] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1v_rollin_s1v3.sh
# Run detached: setsid nohup bash .../start_d1v_rollin_s1v3.sh > logs/mist_w3po_v1/d1v/start_d1v_rollin_s1v3.log 2>&1 < /dev/null &
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
DECISION=${DECISION:?the coordinator decision text}
PAIR=${PAIR:-2,3}
NEED=${NEED_MIB:-40000}
C=eval_outputs/mist_w3po_v1/crf
S=$C/sched_t5_merge3_w10_v1
M=v2p
CK=$C/train/ve1_20261111T214454Z/ckpt/step_03000
SRC=$C/d1_rollin_ve1_v2
ROOT=$C/d1_rollin_ve1_v3
MAN=$C/d1_ve1_w10_v1
L=logs/mist_w3po_v1/d1v
L0=$L/rollin_ve1_v2_0of2.log
LOGK=$L/rollin_ve1_v3_1of2.log
SL=logs/mist_w3po_v1/slots
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
slot_of() { case $1 in 0|1|2|3) echo slotA;; *) echo slotB;; esac; }
training_slot() { local s; s=$(slot_of $1); [ -e $SL/$s.owner ] && grep -qi 'training' $SL/$s.owner; }
CMD="CUDA_VISIBLE_DEVICES=$PAIR $PY -m experiments.mist_w3po_v1.crf.d1_rollin --score --scorer v2 --sched $S --ckpt $CK --root $ROOT --shard 1/2 --selfcheck 0 --mem-cap-gib ${MEMCAP:-60}"
BAD=""; for c in ${PAIR//,/ }; do training_slot $c && BAD="$BAD $c"; done
if [ -n "${DRY_RUN:-}" ]; then
  echo "decision: $DECISION"
  echo "shard 0 source $SRC/rollin_0of2: rho $([ -s $SRC/rollin_0of2/rho.jsonl ] && wc -l < $SRC/rollin_0of2/rho.jsonl || echo missing) lines, done.json $([ -e $SRC/rollin_0of2/done.json ] && echo yes || echo 'not yet'); selfcheck pass $(field $SRC/rollin_0of2/selfcheck.json "d['margin_mode']['pass']" 2>/dev/null)"
  echo "root $ROOT ($([ -e $ROOT ] && echo EXISTS || echo new)); manifest $MAN ($([ -e $MAN ] && echo EXISTS || echo new)); log $LOGK ($([ -e $LOGK ] && echo EXISTS || echo new))"
  echo "cards of PAIR in a training slot (refused):${BAD:- none}"
  for c in ${PAIR//,/ }; do echo "card $c used MiB: $(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"; done
  echo "shard 1: $CMD"
  echo "link: $ROOT/rollin_0of2 -> ../$(basename $SRC)/rollin_0of2"
  echo "manifest: $PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN"
  exit 0
fi
[ -z "$BAD" ] || { log "cards$BAD belong to a training slot: not started"; exit 1; }
[ -e $ROOT ] || [ -e $MAN ] || [ -e $LOGK ] && { log "拒绝覆盖已有输出: $ROOT / $MAN / $LOGK"; exit 1; }
[ "$(field $SRC/rollin_0of2/selfcheck.json "d['margin_mode']['pass']" 2>/dev/null)" = True ] || { log "shard 0 selfcheck did not pass amendment 1"; exit 1; }
log "decision: $DECISION"
mkdir -p $ROOT
exec 9>"$LOCK"
flock -w 1800 9 || { log "gpu_launch.lock busy for 1800 s"; exit 1; }
ok=0
until [ $ok = 1 ]; do
  ok=1
  for c in ${PAIR//,/ }; do
    used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ $((used + NEED)) -le 138240 ] || ok=0
    training_slot $c && { flock -u 9; log "card $c became a training card: not started"; exit 1; }
  done
  if [ $ok = 0 ]; then flock -u 9; sleep 20; flock -w 1800 9 || { log "lock busy"; exit 1; }; fi
done
first=${PAIR%%,*}
base=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
setsid nohup bash -c "$CMD; echo D1_ROLLIN_EXIT rc=\$?" < /dev/null >> "$LOGK" 2>&1 9>&- &
log "shard 1/2 on $PAIR pid $! log $LOGK"
for i in $(seq 1 60); do
  sleep 5
  now=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ $((now - base)) -gt 12000 ] && break
done
flock -u 9
exec 9>&-
log "shard 1 launched; waiting for shard 1 and shard 0 to exit"
until grep -q D1_ROLLIN_EXIT $LOGK 2>/dev/null; do sleep 30; done
grep -q 'D1_ROLLIN_EXIT rc=0' $LOGK || { log "shard 1 FAILED: $(grep -m1 -E 'Error|error' $LOGK | cut -c1-300)"; exit 1; }
log "shard 1 done: $(grep -m1 D1_ROLLIN_DONE $LOGK | cut -c1-200)"
until grep -q D1_ROLLIN_EXIT $L0 2>/dev/null; do sleep 30; done
grep -q 'D1_ROLLIN_EXIT rc=0' $L0 && [ -s $SRC/rollin_0of2/done.json ] || { log "shard 0 did not finish cleanly"; exit 1; }
ln -s ../$(basename $SRC)/rollin_0of2 $ROOT/rollin_0of2 && log "linked $ROOT/rollin_0of2 -> ../$(basename $SRC)/rollin_0of2"
$PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN > $L/manifest_ve1_v3.log 2>&1 \
  && log "D1V_ROLLIN_DONE manifest $MAN: R4 $(field $MAN/summary.json "d['R4_pass']") (windows with an overdue read: $(field $MAN/summary.json "d['share_windows_with_overdue']"))" \
  || { log "manifest failed (see $L/manifest_ve1_v3.log)"; exit 1; }
