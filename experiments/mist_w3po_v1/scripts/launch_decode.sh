#!/usr/bin/env bash
# Launch online_decode.py shards detached, one per card in CARDS, 60 s apart (model starts staggered).
#   NAME=<run name> THINKER=<merged> WINDOWS=<jsonl> CARDS="0 1 2 3" [POLICY=wait] [LAG_MS=2000] [CONTEXT=0]
#   [SHARD_BY=meeting] [BATCH=16] [EXTRA="..."] [OUTROOT=eval_outputs/mist_w3po_v1/decode] bash launch_decode.sh
# Outputs <OUTROOT>/<NAME>/hyp_s<k>.jsonl and infer_s<k>.log; refuses an existing <OUTROOT>/<NAME>. DRY_RUN=1 prints.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$J/conda_envs/qwen3omni/bin/python
NAME=${NAME:?}; THINKER=${THINKER:?}; WINDOWS=${WINDOWS:?}; CARDS=${CARDS:?}
OUTROOT=${OUTROOT:-eval_outputs/mist_w3po_v1/decode}
D=$OUTROOT/$NAME
if [ -e "$D" ]; then echo "拒绝覆盖已有输出: $D" >&2; exit 1; fi
set -- $CARDS; N=$#
CMDS=()
k=0
for c in $CARDS; do
  CMDS+=("CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.evaluation.online_decode --thinker $THINKER --windows $WINDOWS --out $D/hyp_s$k.jsonl --policy ${POLICY:-wait} --lag-ms ${LAG_MS:-2000} --context ${CONTEXT:-0} --dir zh2en --batch ${BATCH:-16} --max-gib ${MAX_GIB:-120} --shard $k --num-shards $N --shard-by ${SHARD_BY:-meeting} ${EXTRA:-} > $D/infer_s$k.log 2>&1")
  k=$((k+1))
done
if [ -n "${DRY_RUN:-}" ]; then printf '%s\n' "${CMDS[@]}"; exit 0; fi
mkdir -p "$D"
echo "$(date -u +%FT%TZ) NAME=$NAME THINKER=$THINKER WINDOWS=$WINDOWS CARDS=$CARDS POLICY=${POLICY:-wait} CONTEXT=${CONTEXT:-0}" > $D/launch.txt
for i in "${!CMDS[@]}"; do
  setsid nohup bash -c "${CMDS[$i]}" < /dev/null > /dev/null 2>&1 &
  echo "shard $i pid $!" >> $D/launch.txt
  [ $i -lt $((N-1)) ] && sleep 60
done
