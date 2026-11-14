#!/usr/bin/env bash
# PLAN B COPY (2026-11-14, user: "先A后B"): run_crf_v2p.sh with the trainer entry crf/train_v2p_lat.py (installs
# crf/serial_v2p_lat.py: the v2' posterior with a per-window latency budget, ratio V2P_LAT_RATIO, required) -- every
# other line is run_crf_v2p.sh's. Original header below.
# SerialCRF v2' training (reports/mist_w3po_v1/serial_v2p_train.zh-CN.md): continue v8 with
# crf/train_crf_sft.py --objective serial_v2p (crf/serial_v2p_train.py), C0's recipe and data stream (3000 steps, ~64 read
# rows / step, lr 1e-4 -> 1e-5 cosine, warm-up 50, seed 20261109, v8 init, LoRA r32 / alpha 64 + projector + new rows),
# --final-read-full-post 1 (the precheck's train-window protocol, = C0-fp), tau 0.5 (pre-registered), on 8 cards.
# The remaining flags change packing / speed only (tested: same posterior and gradients):
#   --feat-per-row 1 (each row's log-mel on its own audio, cached per read), --fast-moe-bwd 1 (no frozen expert-weight
#   gradients in transformers' grouped_mm fallback; bit-identical), --prefetch 1 (pass-2 batches collated during pass 1),
#   --empty-cache 0, --tok-budget $TOK padded tokens per micro-batch, --mem-cap-gib $CAP (allocator cap per process).
#   CARDS=0,1,2,3,4,5,6,7 [RUN_ID=crf_v2p_<UTC>] [PORT=29961] [STEPS=3000] [SAVE_EVERY=500] [STOP_AFTER=n] [TOK=32000]
#     [CAP=120] [FAST=1] [PREFETCH=1] [EMPTY=0] [EXTRA=...] bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh
# Detaches itself (setsid nohup) and prints "... pid <root pid> ...". Output eval_outputs/mist_w3po_v1/crf/train/<RUN_ID>
# (ckpt/step_NNNNN in C0's format, metrics.jsonl), log logs/mist_w3po_v1/<RUN_ID>.log; refuses an existing output dir or
# log. RESUME=1 RUN_ID=<existing>: continue it (--resume, log appended). DRY_RUN=1 prints the command only.
# Smoke: STOP_AFTER=5 RUN_ID=crf_v2p_smoke_<UTC> (steps 1..5 of the same 3000-step schedule, a checkpoint at step 5).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
  TRITON_CACHE_DIR=$J/cache/triton
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
PY=$J/conda_envs/qwen3omni/bin/python
OBJ=serial_v2p
V2P_LAT_RATIO=${V2P_LAT_RATIO:?plan B: set V2P_LAT_RATIO in [0, 1]}; export V2P_LAT_RATIO
CARDS=${CARDS:?set CARDS, e.g. CARDS=0,1,2,3,4,5,6,7}
N=$(echo "$CARDS" | tr ',' ' ' | wc -w)
RUN_ID=${RUN_ID:-crf_v2p_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=eval_outputs/mist_w3po_v1/crf/train/$RUN_ID
LOG=logs/mist_w3po_v1/$RUN_ID.log
if [ -n "${RESUME:-}" ]; then
  [ -d "$OUT" ] || { echo "RESUME=1 but $OUT does not exist" >&2; exit 1; }
  RES=--resume
else
  if [ -e "$OUT" ] || [ -e "$LOG" ]; then echo "拒绝覆盖已有输出: $OUT / $LOG" >&2; exit 1; fi
  RES=
fi
STOP=
[ -n "${STOP_AFTER:-}" ] && STOP="--stop-after $STOP_AFTER"
CMD="CUDA_VISIBLE_DEVICES=$CARDS $PY -m torch.distributed.run --nproc_per_node $N --master_port ${PORT:-29961} \
experiments/mist_w3po_v1/crf/train_v2p_lat.py --objective $OBJ --tau 0.5 --final-read-full-post 1 --out $OUT \
--steps ${STEPS:-3000} --save-every ${SAVE_EVERY:-500} --feat-per-row 1 --fast-moe-bwd ${FAST:-1} \
--prefetch ${PREFETCH:-1} --empty-cache ${EMPTY:-0} --tok-budget ${TOK:-32000} --mem-cap-gib ${CAP:-120} $STOP $RES \
${EXTRA:-}"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; echo "LOG=$LOG"; exit 0; fi
mkdir -p logs/mist_w3po_v1
setsid nohup bash -c "$CMD; echo CRF_RUN_EXIT rc=\$?" < /dev/null >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) launched $RUN_ID (objective $OBJ + latency budget ratio $V2P_LAT_RATIO, cards $CARDS) pid $! -> $OUT, log $LOG"
