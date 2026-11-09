#!/usr/bin/env bash
# C0-fp: C0 (scripts/run_crf_c0.sh: continue v8 with plain SFT on the heuristic labels of sched_v1, v8 recipe, 3000
# steps, 4 cards) + train_crf_sft.py --final-read-full-post 1: at each window's FINAL read the posteriors P are visible
# up to t // 80 (the whole window) instead of (t - 1040) // 80 (the training analogue of
# evaluation/online_decode_finalpost.py --final-read-post close_detect). Everything else is identical to C0.
#   CARDS=4,5,6,7 [RUN_ID=crf_c0fp_<UTC time>] [PORT=29951] [STEPS=3000] [SAVE_EVERY=500] [EXTRA="--mem-cap-gib 88"] \
#     bash experiments/mist_w3po_v1/scripts/run_crf_c0_fp.sh
# Default PORT differs from run_crf_c0.sh (29941), so C0 and C0-fp can run at the same time.
# Detaches itself (setsid nohup). Output eval_outputs/mist_w3po_v1/crf/train/<RUN_ID> (ckpt/step_NNNNN, metrics.jsonl),
# log logs/mist_w3po_v1/<RUN_ID>.log; refuses an existing output dir or log. RESUME=1 RUN_ID=<existing>: continue it
# (--resume, log appended). DRY_RUN=1 prints the command only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
  TRITON_CACHE_DIR=$J/cache/triton
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
PY=$J/conda_envs/qwen3omni/bin/python
OBJ=heuristic
CARDS=${CARDS:?set CARDS, e.g. CARDS=4,5,6,7}
N=$(echo "$CARDS" | tr ',' ' ' | wc -w)
RUN_ID=${RUN_ID:-crf_c0fp_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=eval_outputs/mist_w3po_v1/crf/train/$RUN_ID
LOG=logs/mist_w3po_v1/$RUN_ID.log
if [ -n "${RESUME:-}" ]; then
  [ -d "$OUT" ] || { echo "RESUME=1 but $OUT does not exist" >&2; exit 1; }
  RES=--resume
else
  if [ -e "$OUT" ] || [ -e "$LOG" ]; then echo "拒绝覆盖已有输出: $OUT / $LOG" >&2; exit 1; fi
  RES=
fi
CMD="CUDA_VISIBLE_DEVICES=$CARDS $PY -m torch.distributed.run --nproc_per_node $N --master_port ${PORT:-29951} \
experiments/mist_w3po_v1/crf/train_crf_sft.py --objective $OBJ --final-read-full-post 1 --out $OUT --steps ${STEPS:-3000} \
--save-every ${SAVE_EVERY:-500} $RES ${EXTRA:-}"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; echo "LOG=$LOG"; exit 0; fi
mkdir -p logs/mist_w3po_v1
setsid nohup bash -c "$CMD; echo CRF_RUN_EXIT rc=\$?" < /dev/null >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) launched $RUN_ID (objective $OBJ + final-read-full-post, cards $CARDS) pid $! -> $OUT, log $LOG"
