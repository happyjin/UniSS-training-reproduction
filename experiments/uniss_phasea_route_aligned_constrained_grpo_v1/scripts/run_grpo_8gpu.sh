#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=${1:?usage: run_grpo_8gpu.sh RUN_ID PACK_ID WARMUP_ROOT [EPOCHS]}
PACK_ID=${2:?missing pack ID}
WARMUP_ROOT=${3:?missing warm-up checkpoint root}
EPOCHS=${4:-3}
TRAIN=${REPO_ROOT}/data/processed/uniss_phasea_route_aligned_constrained_grpo_v1/${PACK_ID}/train_packs.jsonl
VALID=${REPO_ROOT}/data/processed/uniss_phasea_route_aligned_constrained_grpo_v1/${PACK_ID}/valid_packs.jsonl
SAVE=${REPO_ROOT}/checkpoints/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}
TB=${REPO_ROOT}/runs/uniss_phasea_route_aligned_constrained_grpo_v1/tensorboard/${RUN_ID}
LOG=${REPO_ROOT}/logs/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}.log
for path in "${TRAIN}" "${TRAIN}.offsets.bin" "${VALID}" "${VALID}.offsets.bin"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
[[ -f "${WARMUP_ROOT}/latest_checkpointed_iteration.txt" ]] || {
  echo "missing warm-up checkpoint root ${WARMUP_ROOT}" >&2; exit 2;
}
[[ ! -e "${SAVE}" && ! -e "${TB}" && ! -e "${LOG}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2; exit 3;
}
mkdir -p "$(dirname "${SAVE}")" "${TB}" "$(dirname "${LOG}")"
TRAIN_RECORDS=$(($(stat -c %s "${TRAIN}.offsets.bin") / 8))
STEPS_PER_EPOCH=$(( (TRAIN_RECORDS + 8 - 1) / 8 ))
TRAIN_ITERS=$(( STEPS_PER_EPOCH * EPOCHS ))

export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON}"):${PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export UNISS_E2E_COMPILE_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/.cache/uniss_route_aligned/${RUN_ID}
SITE_PACKAGES=$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_LIBRARY_PATH=$(find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CMD=(
  "$(dirname "${PYTHON}")/torchrun" --nproc_per_node 8 --master_port 29982
  "${REPO_ROOT}/experiments/uniss_phasea_stateful_longepisode_rl_v1/training/pretrain_megatron.py"
  --sft --episode-grpo-train "${TRAIN}" --episode-grpo-valid "${VALID}"
  --episode-whispervq-model "${WHISPERVQ_MODEL}"
  --episode-lora-rank 16 --episode-lora-alpha 32 --episode-lora-dropout 0.05
  --episode-top-layers 8 --episode-adapter-lr 8e-6
  --episode-clip-epsilon 0.20 --episode-kl-beta 0.03
  --episode-replay-weight 0.35 --episode-anchor-weight 0.03
  --tokenizer-type NullTokenizer --vocab-size 180407
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 18000 --max-position-embeddings 32768
  --micro-batch-size 1 --global-batch-size 8 --train-iters "${TRAIN_ITERS}"
  --lr 8e-6 --min-lr 8e-7 --lr-warmup-iters 1
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type cyclic --no-data-sharding --num-workers 0
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 0.5
  --bf16 --use-flash-attn --attention-backend fused
  --no-create-attention-mask-in-dataloader --no-gradient-accumulation-fusion
  --recompute-activations --attention-dropout 0.0 --hidden-dropout 0.0
  --dist-ckpt-strictness log_all --finetune --no-load-optim --no-load-rng
  --load "${WARMUP_ROOT}" --save "${SAVE}" --save-interval "${STEPS_PER_EPOCH}"
  --eval-iters 2 --eval-interval "${STEPS_PER_EPOCH}"
  --eval-micro-batch-size 1 --eval-global-batch-size 8
  --log-interval 1 --tensorboard-dir "${TB}" --tensorboard-log-interval 1
  --log-timers-to-tensorboard --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard --log-memory-interval 1
  --log-world-size-to-tensorboard --log-throughput --seed 20260827
)
{
  echo "TRAIN_RECORDS=${TRAIN_RECORDS}"
  echo "STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
  echo "EPOCHS=${EPOCHS}"
  echo "TRAIN_ITERS=${TRAIN_ITERS}"
  printf '%q ' "${CMD[@]}"; printf '\n'
} >"${LOG}.command"
"${CMD[@]}" 2>&1 | tee "${LOG}"

