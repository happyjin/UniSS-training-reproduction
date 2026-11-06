#!/usr/bin/env bash
# W3PO / RL baseline training (experiments/mist_w3po_v1/rl/train_w3po.py), one process per card in CARDS.
#   RUN=<name> BASE=<stage-0 merged> CARDS="0,1,2,3" [MODE=w3po] [STEPS=50] [WPS=64] [G=8] [EXTRA="..."] [RESUME=1]
#   setsid nohup bash run_rl.sh > logs/mist_w3po_v1/rl_<name>.log 2>&1 < /dev/null &
# Output eval_outputs/mist_w3po_v1/rl/<RUN> (refused if it exists, unless RESUME=1). DRY_RUN=1 prints the command.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
  TRITON_CACHE_DIR=$J/cache/triton
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=$J/conda_envs/qwen3omni/bin/python
RUN=${RUN:?}; BASE=${BASE:?}; CARDS=${CARDS:?}
OUT=eval_outputs/mist_w3po_v1/rl/$RUN
if [ -e "$OUT" ] && [ -z "${RESUME:-}" ]; then echo "拒绝覆盖已有输出: $OUT" >&2; exit 1; fi
N=$(echo "$CARDS" | tr ',' '\n' | wc -l)
CMD="CUDA_VISIBLE_DEVICES=$CARDS $PY -m torch.distributed.run --nproc_per_node $N --master_port ${PORT:-29871} \
 -m experiments.mist_w3po_v1.rl.train_w3po --base $BASE --windows ${WINDOWS:-eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl} \
 --out $OUT --mode ${MODE:-w3po} --steps ${STEPS:-50} --windows-per-step ${WPS:-64} --G ${G:-8} ${RESUME:+--resume} ${EXTRA:-}"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; exit 0; fi
echo "$(date -u +%FT%TZ) $CMD"
eval "$CMD"
echo "RL_EXIT $?"
