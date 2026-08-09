#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2_nar_ctc_smoke_v1}
TRAIN_ITERS=${TRAIN_ITERS:-40}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
EVAL_ITERS=${EVAL_ITERS:-2}
LOG_INTERVAL=${LOG_INTERVAL:-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
SMOKE_ROOT=$ROOT/data/processed/phase3_whisper_streamspeech_joint_v1/smoke_manifest_128
SAVE_DIR=$ROOT/checkpoints/simul_s2st_route_v1/$RUN_NAME
TB_DIR=$ROOT/runs/simul_s2st_route_v1/$RUN_NAME
LOG=$ROOT/logs/simul_s2st_route_v1/$RUN_NAME.log
MASTER_PORT=${MASTER_PORT:-29811}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR" \
  "$(dirname "$LOG")" "$ROOT/checkpoints/simul_s2st_route_v1" "$ROOT/runs/simul_s2st_route_v1"

SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBRARY_DIRS=()
shopt -s nullglob
for directory in "$SITE_PACKAGES"/nvidia/*/lib; do
  [[ -d "$directory" ]] && NVIDIA_LIBRARY_DIRS+=("$directory")
done
shopt -u nullglob
NVIDIA_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIBRARY_DIRS[*]}")"
export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON" - <<'PY'
import ctypes
ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402
PY

test ! -e "$SAVE_DIR"
test ! -e "$TB_DIR"
test ! -e "$LOG"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$MASTER_PORT" \
  "$ROOT/experiments/simul_s2st_route_v1/step2_nar_ctc_head/pretrain_nar_ctc_megatron.py" \
  --nar-train-manifest "$SMOKE_ROOT/joint_train.jsonl" \
  --nar-valid-manifest "$SMOKE_ROOT/joint_valid.jsonl" \
  --nar-phase3-model "$ROOT/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf" \
  --nar-frames-per-second 75 \
  --nar-max-frames 900 \
  --nar-max-audio-seconds 10 \
  --nar-max-unit-tokens 800 \
  --tokenizer-type NullTokenizer \
  --vocab-size 180407 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 896 \
  --ffn-hidden-size 4864 \
  --num-attention-heads 14 \
  --group-query-attention \
  --num-query-groups 2 \
  --normalization RMSNorm \
  --swiglu \
  --disable-bias-linear \
  --add-qkv-bias \
  --position-embedding-type rope \
  --rotary-base 1000000 \
  --bf16 \
  --seq-length 4096 \
  --max-position-embeddings 32768 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --train-iters "$TRAIN_ITERS" \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --lr-warmup-iters "$LR_WARMUP_ITERS" \
  --lr-decay-iters "$TRAIN_ITERS" \
  --lr-decay-style cosine \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --dataloader-type cyclic \
  --num-workers 2 \
  --no-create-attention-mask-in-dataloader \
  --no-gradient-accumulation-fusion \
  --save "$SAVE_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-iters "$EVAL_ITERS" \
  --log-interval "$LOG_INTERVAL" \
  --tensorboard-dir "$TB_DIR" \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --seed 20260809 \
  2>&1 | tee "$LOG"
