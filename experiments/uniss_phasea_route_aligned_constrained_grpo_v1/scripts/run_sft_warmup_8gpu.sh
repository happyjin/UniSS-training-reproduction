#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
RUN_ID=${1:-route_aligned_sft64_v1}
TRAIN_ITERS=${TRAIN_ITERS:-64}
WARMUP_ITERS=${WARMUP_ITERS:-4}
if (( WARMUP_ITERS >= TRAIN_ITERS )); then WARMUP_ITERS=1; fi
SAVE=${REPO_ROOT}/checkpoints/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}
TB=${REPO_ROOT}/runs/uniss_phasea_route_aligned_constrained_grpo_v1/tensorboard/${RUN_ID}
LOG=${REPO_ROOT}/logs/uniss_phasea_route_aligned_constrained_grpo_v1/${RUN_ID}.log
[[ ! -e "${SAVE}" && ! -e "${TB}" && ! -e "${LOG}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2; exit 3;
}
mkdir -p "${SAVE}" "${TB}" "$(dirname "${LOG}")"

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
  "$(dirname "${PYTHON}")/torchrun" --nproc_per_node 8 --master_port 29981
  "${EXPERIMENT_ROOT}/training/pretrain_joint_sft.py"
  --sft --joint-mode sft --joint-group-size 4 --joint-bootstrap-updates 1
  --joint-candidate-width 16 --joint-clip-epsilon 0.20 --joint-kl-beta 0.02
  --joint-sft-replay-weight 1.0 --joint-reference-anchor-weight 0.02
  --joint-lora-rank 16 --joint-lora-alpha 32 --joint-lora-dropout 0.05
  --joint-top-layers 8 --joint-adapter-lr 2e-5 --joint-smoke
  --e2e-train-build-report "${TRAIN_REPORT}"
  --e2e-valid-build-report "${VALID_REPORT}"
  --e2e-phase3-train-cache-audit "${PHASE3_TRAIN_CACHE}"
  --e2e-phase3-valid-cache-audit "${PHASE3_VALID_CACHE}"
  --e2e-whispervq-model "${WHISPERVQ_MODEL}"
  --e2e-checkpoint-fingerprints "${FINGERPRINTS}"
  --e2e-asr-weight 1.0 --e2e-mt-weight 1.0 --e2e-semantic-weight 1.0
  --e2e-replay-weight 0.0 --e2e-v1-asr-kl-weight 0.25
  --e2e-phase3-kl-weight 0.25 --e2e-commit-weight 0.10
  --e2e-boundary-eos-weight 0.10 --e2e-speaker-continuity-weight 0.0
  --tokenizer-type NullTokenizer --vocab-size 180407
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 18000 --max-position-embeddings 32768
  --micro-batch-size 1 --global-batch-size 16 --train-iters "${TRAIN_ITERS}"
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type cyclic --no-data-sharding --num-workers 0
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 0.5
  --bf16 --use-flash-attn --attention-backend fused
  --no-create-attention-mask-in-dataloader --no-gradient-accumulation-fusion
  --recompute-activations --attention-dropout 0.0 --hidden-dropout 0.0
  --dist-ckpt-strictness log_all --finetune --no-load-optim --no-load-rng
  --load "${PHASE_A_ROOT}" --save "${SAVE}" --save-interval 32
  --eval-iters 4 --eval-interval 32 --eval-micro-batch-size 1
  --eval-global-batch-size 8 --log-interval 1 --tensorboard-dir "${TB}"
  --tensorboard-log-interval 1 --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard --log-memory-to-tensorboard
  --log-memory-interval 1 --log-world-size-to-tensorboard --log-throughput
  --seed 20260827
)
printf '%q ' "${CMD[@]}" >"${LOG}.command"; printf '\n' >>"${LOG}.command"
"${CMD[@]}" 2>&1 | tee "${LOG}"
