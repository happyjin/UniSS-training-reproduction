#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
SOURCE=$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl
DIRECTION_INDEX=$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_v1
RUN_NAME=${RUN_NAME:-stage08_step2_qwen_lora_replay_r8_replay30_research_v1}
TRAIN_ITERS=${TRAIN_ITERS:-200}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_ITERS=${EVAL_ITERS:-8}
LOG_INTERVAL=${LOG_INTERVAL:-10}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-20}
MASTER_PORT=${MASTER_PORT:-29670}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
OFFLINE_REPLAY_RATIO=${OFFLINE_REPLAY_RATIO:-0.30}
SAVE_DIR=$ROOT/checkpoints/uniss_streamspeech_ctc_v1/$RUN_NAME
TB_DIR=$ROOT/runs/uniss_streamspeech_ctc_v1/$RUN_NAME
LOG=$ROOT/logs/uniss_streamspeech_ctc_v1/$RUN_NAME.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR" "$(dirname "$LOG")"

SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBRARY_DIRS=()
shopt -s nullglob
for directory in "$SITE_PACKAGES"/nvidia/*/lib; do
  [[ -d "$directory" ]] && NVIDIA_LIBRARY_DIRS+=("$directory")
done
shopt -u nullglob
(( ${#NVIDIA_LIBRARY_DIRS[@]} > 0 )) || {
  echo "No NVIDIA library directories found under $SITE_PACKAGES" >&2
  exit 1
}
NVIDIA_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIBRARY_DIRS[*]}")"
export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON" - <<'PY'
import ctypes

ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402
PY

test -f "$DIRECTION_INDEX/metadata.json"
test ! -e "$SAVE_DIR"
test ! -e "$TB_DIR"
test ! -e "$LOG"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$MASTER_PORT" \
  "$ROOT/experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step2_qwen_lora_replay_v1/pretrain_lora_replay_megatron.py" \
  --step2-dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
  --step2-source-manifest "$SOURCE" \
  --step2-source-offsets "${SOURCE}.offsets.bin" \
  --step2-direction-index-dir "$DIRECTION_INDEX" \
  --step2-ctc-tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
  --step2-stage03b-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
  --step2-historical-stage-b-checkpoint "$ROOT/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt" \
  --step2-stage04-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1/best.pt" \
  --step2-stage06-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage06_b1_megatron_v2/iter_0000600" \
  --step2-step1-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_p3w2_zhen1p25_v1/iter_0000350" \
  --step2-codebook-model "$ROOT/pretrained_models/UniSS/glm4_tokenizer" \
  --step2-phase3-model "$ROOT/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf" \
  --step2-lora-rank "$LORA_RANK" \
  --step2-lora-alpha "$LORA_ALPHA" \
  --step2-lora-dropout "$LORA_DROPOUT" \
  --step2-lora-targets q_proj,v_proj \
  --step2-offline-replay-ratio "$OFFLINE_REPLAY_RATIO" \
  --step2-research-only-override \
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
  --seq-length 18000 \
  --max-position-embeddings 32768 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters "$TRAIN_ITERS" \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --lr-warmup-iters "$LR_WARMUP_ITERS" \
  --lr-decay-iters "$TRAIN_ITERS" \
  --lr-decay-style cosine \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 0.5 \
  --dataloader-type cyclic \
  --num-workers 4 \
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
  2>&1 | tee "$LOG"
