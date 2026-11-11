#!/usr/bin/env bash
# D1 roll-in (reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §1.2, §4.2): the learner picked by j1_decide.py
# (logs/mist_w3po_v1/d1/base_decision.json; schedule.json must say run_D1, unless FORCE=1) decodes WHEN to write on its
# own training windows (crf/d1_rollin.py --score, forward passes only), one 2-card shard per pair in PAIRS, then the
# manifest (CPU, crf/d1_rollin.py --manifest) with the R4 line. Eval-class job: stacks on non-training cards; per shard,
# under flock(gpu_launch.lock), every card's nvidia-smi used + NEED_MIB <= 138240 MiB (135 GiB), launched detached with
# the lock fd closed, the lock kept until the shard's memory shows (or 300 s) -> launches are staggered.
#   PAIRS="0,1 2,3" [NEED_MIB=40000] [MEMCAP=60] [FORCE=1] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1_rollin.sh
# Run detached: setsid nohup bash .../start_d1_rollin.sh > logs/mist_w3po_v1/d1/start_d1_rollin.log 2>&1 < /dev/null &
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
PAIRS=${PAIRS:?set PAIRS, e.g. "0,1 2,3"}
NEED=${NEED_MIB:-40000}
DEC=${DEC:-logs/mist_w3po_v1/d1/base_decision.json}      # overridable for DRY_RUN checks only
SCH=${SCH:-logs/mist_w3po_v1/d1/schedule.json}
L=logs/mist_w3po_v1/d1
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
mkdir -p $L
if [ -z "${DRY_RUN:-}" ]; then
  log "waiting for $DEC and $SCH"
  until [ -s $DEC ] && [ -s $SCH ]; do sleep 30; done
fi
[ -s $DEC ] || { echo "no $DEC yet (j1_decide --wait writes it after both quick checks)"; exit 0; }
RUN_D1=$(field $SCH "d['run_D1']" 2>/dev/null || echo "?")
if [ "$RUN_D1" != True ] && [ "${FORCE:-}" != 1 ]; then log "schedule.json: run_D1 = $RUN_D1 -> no roll-in (FORCE=1 overrides)"; exit 0; fi
BASE=$(field $DEC "d['base']")
[ "$BASE" != None ] || { log "no D1 base (neither T4 nor T5 trained cleanly)"; exit 1; }
S=$(field $DEC "d['data']['sched']"); M=$(field $DEC "d['data']['lw_mode']"); RUN=$(field $DEC "d['run']")
CK=eval_outputs/mist_w3po_v1/crf/train/$RUN/ckpt/step_03000
ROOT=eval_outputs/mist_w3po_v1/crf/d1_rollin_${BASE}_v1
MAN=eval_outputs/mist_w3po_v1/crf/d1_${BASE}_w10_v1
read -r -a PA <<< "$PAIRS"
N=${#PA[@]}
cmd_of() { echo "CUDA_VISIBLE_DEVICES=${PA[$1]} $PY -m experiments.mist_w3po_v1.crf.d1_rollin --score --sched $S --ckpt $CK --root $ROOT --shard $1/$N --selfcheck 20 --mem-cap-gib ${MEMCAP:-60}"; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "base $BASE ($RUN, $S, lattice $M); run_D1 $RUN_D1; root $ROOT; manifest $MAN; learner ckpt DONE: $([ -e $CK/DONE ] && echo yes || echo 'not yet')"
  for k in $(seq 0 $((N - 1))); do echo "shard $k: $(cmd_of $k)"; done
  echo "manifest: $PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN"
  exit 0
fi
[ -e $CK/DONE ] || { log "learner checkpoint $CK/DONE missing"; exit 1; }
[ -e $ROOT ] || [ -e $MAN ] && { log "拒绝覆盖已有输出: $ROOT / $MAN"; exit 1; }
for k in $(seq 0 $((N - 1))); do
  LOGK=$L/rollin_${BASE}_${k}of${N}.log
  [ -e $LOGK ] && { log "拒绝覆盖已有日志: $LOGK"; exit 1; }
  exec 9>"$LOCK"
  flock -w 1800 9 || { log "gpu_launch.lock busy for 1800 s"; exit 1; }
  ok=0
  until [ $ok = 1 ]; do      # wait (lock held only briefly per try) until every card of the pair has room
    ok=1
    for c in ${PA[$k]//,/ }; do
      used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ $((used + NEED)) -le 138240 ] || ok=0
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
for k in $(seq 0 $((N - 1))); do
  until grep -q D1_ROLLIN_EXIT $L/rollin_${BASE}_${k}of${N}.log 2>/dev/null; do sleep 30; done
  grep -q 'D1_ROLLIN_EXIT rc=0' $L/rollin_${BASE}_${k}of${N}.log || { log "shard $k failed: no manifest"; exit 1; }
done
$PY -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched $S --root $ROOT --lw-mode $M --out $MAN > $L/manifest_${BASE}.log 2>&1 \
  && log "manifest $MAN: R4 $(field $MAN/summary.json "d['R4_pass']") (windows with an overdue read: $(field $MAN/summary.json "d['share_windows_with_overdue']"))" \
  || log "manifest failed (see $L/manifest_${BASE}.log)"
