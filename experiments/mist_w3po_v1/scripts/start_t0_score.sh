#!/usr/bin/env bash
# T0 scoring shard launcher (crf/t0_agree_labels.py --score; reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T0).
# Stacks one 2-card shard on cards that are NOT a training slot. Under flock(gpu_launch.lock) it checks every card's
# nvidia-smi total + NEED_MIB <= 138240 MiB (135 GiB), launches detached (setsid nohup, the lock fd closed in the child),
# and keeps the lock until the job's memory shows on its first card (or 300 s), so launches are staggered.
#   CARDS=4,5 SHARD=0/2 ROOT=<root> [TARGETS=<targets.jsonl>] [SPLIT=dev|train] [NEED_MIB=40000] [DRY_RUN=1]
#     bash experiments/mist_w3po_v1/scripts/start_t0_score.sh
# Log: logs/mist_w3po_v1/t0/<root name>_score_<k>of<n>.log (refuses an existing log).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
  TRITON_CACHE_DIR=$J/cache/triton
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
CARDS=${CARDS:?set CARDS, e.g. 4,5}
SHARD=${SHARD:?set SHARD, e.g. 0/2}
ROOT=${ROOT:?set ROOT}
SPLIT=${SPLIT:-dev}
NEED=${NEED_MIB:-40000}
TG=
[ -n "${TARGETS:-}" ] && TG="--targets $TARGETS"
K=${SHARD%/*}; N=${SHARD#*/}
mkdir -p logs/mist_w3po_v1/t0
LOG=logs/mist_w3po_v1/t0/$(basename "$ROOT")_score_${K}of${N}.log
[ -e "$LOG" ] && { echo "拒绝覆盖已有日志: $LOG" >&2; exit 1; }
CMD="CUDA_VISIBLE_DEVICES=$CARDS $PY -m experiments.mist_w3po_v1.crf.t0_agree_labels --score --split $SPLIT --root $ROOT $TG --shard $SHARD"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; echo "LOG=$LOG"; exit 0; fi
exec 9>"$LOCK"
flock -w 1800 9 || { echo "gpu_launch.lock busy for 1800 s" >&2; exit 1; }
for c in ${CARDS//,/ }; do
  used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ $((used + NEED)) -gt 138240 ]; then echo "card $c: ${used} MiB used + ${NEED} > 138240" >&2; exit 1; fi
done
first=${CARDS%%,*}
base=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
setsid nohup bash -c "$CMD; echo T0_SCORE_EXIT rc=\$?" < /dev/null >> "$LOG" 2>&1 9>&- &
pid=$!
echo "$(date -u +%FT%TZ) launched T0 score $SHARD on cards $CARDS pid $pid log $LOG"
for i in $(seq 1 60); do
  sleep 5
  now=$(nvidia-smi -i "$first" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ $((now - base)) -gt 12000 ] && break
  kill -0 "$pid" 2>/dev/null || break
done
echo "$(date -u +%FT%TZ) released the lock (card $first +$((now - base)) MiB)"
