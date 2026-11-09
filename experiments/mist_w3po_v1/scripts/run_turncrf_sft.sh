#!/usr/bin/env bash
# Scheme 2 formal run (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §3, §11; code: reports/mist_w3po_v1/turn_crf_e2e_code.zh-CN.md):
# end-to-end TurnCRF = crf/train_turncrf_sft.py (the shared trainer crf/train_crf_sft.py, --objective heuristic, plus the
# TurnCRF layer through its HOOK API), continued from v8, 3000 steps, one pre-registered setting (the defaults), 4 cards,
# torchrun, detached (setsid nohup). Control: the shared C0 run (train_crf_sft.py --objective heuristic, same seed/data).
# Layer init: neutral (independent P). NOTE: the TurnCRF CPU stage failed its pre-registered stop rule (<= 1 s turn recall
# -0.21 vs raw P), so this formal run is NOT recommended under the current design; launch only on the user's decision.
#   CARDS="4 5 6 7" OUT=eval_outputs/mist_w3po_v1/crf/turncrf_sft_v1 [MASTER_PORT=29871] [MEM_CAP_GIB=0] [EXTRA="..."] \
#     bash experiments/mist_w3po_v1/scripts/run_turncrf_sft.sh
# Refuses an existing OUT or log. DRY_RUN=1 prints the command only. Log: logs/mist_w3po_v1/<basename OUT>.log,
# exit code: <log>.rc. Checkpoints: OUT/ckpt/step_NNNNN (lora.pt, extra.pt with the layer, optim.pt, DONE); layer side log:
# OUT/turn_layer.jsonl. Decode a checkpoint (LoRA + projector + token rows + layer, in memory on top of v8):
#   python -m experiments.mist_w3po_v1.crf.turn_decode --thinker <v8 merged> ... --turn-ckpt OUT/ckpt/step_NNNNN
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache
export TRITON_CACHE_DIR=$J/cache/triton PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=$J/conda_envs/qwen3omni/bin/python
CARDS=${CARDS:?set CARDS, e.g. CARDS=\"4 5 6 7\"}
OUT=${OUT:-eval_outputs/mist_w3po_v1/crf/turncrf_sft_v1}
LOG=logs/mist_w3po_v1/$(basename "$OUT").log
set -- $CARDS; N=$#
if [ "$N" -ne 4 ]; then echo "预登记为 4 卡，CARDS 给了 $N 张: $CARDS" >&2; exit 1; fi
if [ -e "$OUT" ] || [ -e "$LOG" ]; then echo "拒绝覆盖已有输出: $OUT / $LOG" >&2; exit 1; fi
VIS=$(echo $CARDS | tr ' ' ',')
CMD="CUDA_VISIBLE_DEVICES=$VIS $PY -m torch.distributed.run --nproc_per_node 4 --master_port ${MASTER_PORT:-29871} \
-m experiments.mist_w3po_v1.crf.train_turncrf_sft --out $OUT --steps 3000 --save-every 500 \
--mem-cap-gib ${MEM_CAP_GIB:-0} ${EXTRA:-}"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; exit 0; fi
mkdir -p "$(dirname "$LOG")"
setsid nohup bash -c "$CMD; echo \$? > $LOG.rc" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! cards $CARDS out $OUT log $LOG"
