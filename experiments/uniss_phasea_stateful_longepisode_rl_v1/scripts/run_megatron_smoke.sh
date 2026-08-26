#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

RUN_ID=${1:-episode_grpo_megatron_smoke_v1}
GPU=${2:-4}
TRAIN=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/smoke/train_packs.jsonl
VALID=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/smoke/valid_packs.jsonl
SAVE=${REPO_ROOT}/checkpoints/uniss_phasea_stateful_longepisode_rl_v1/${RUN_ID}
TB=${REPO_ROOT}/runs/uniss_phasea_stateful_longepisode_rl_v1/tensorboard/${RUN_ID}
LOG=${REPO_ROOT}/logs/uniss_phasea_stateful_longepisode_rl_v1/${RUN_ID}.log
[[ ! -e "${SAVE}" && ! -e "${TB}" && ! -e "${LOG}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2
  exit 3
}
mkdir -p "$(dirname "${SAVE}")" "${TB}" "$(dirname "${LOG}")"

export HF_HOME=/opt/dlami/nvme/jasonleeeli/.cache/huggingface
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON}"):${PATH}
export CUDA_VISIBLE_DEVICES=${GPU}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export UNISS_E2E_COMPILE_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/.cache/uniss_phasea_stateful_longepisode_rl_v1/${RUN_ID}

SITE_PACKAGES=$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_LIBRARY_DIRS=()
shopt -s nullglob
for directory in "${SITE_PACKAGES}"/nvidia/*/lib; do
  [[ -d "${directory}" ]] && NVIDIA_LIBRARY_DIRS+=("${directory}")
done
shopt -u nullglob
(( ${#NVIDIA_LIBRARY_DIRS[@]} > 0 )) || {
  echo "no NVIDIA library directories found under ${SITE_PACKAGES}" >&2
  exit 4
}
NVIDIA_LIBRARY_PATH=$(IFS=:; echo "${NVIDIA_LIBRARY_DIRS[*]}")
SYSTEM_CUDA_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64
export LD_LIBRARY_PATH=${SYSTEM_CUDA_LIBRARY_PATH}:${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
"${PYTHON}" - <<'PY'
import ctypes

cudnn = ctypes.CDLL("libcudnn.so.9")
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
if cudnn.cudnnGetVersion() < 90700:
    raise SystemExit(f"cuDNN 9.7+ is required, found {cudnn.cudnnGetVersion()}")
ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402
PY

CMD=(
  "$(dirname "${PYTHON}")/torchrun" --nproc_per_node 1 --master_port 29971
  "${EXPERIMENT_ROOT}/training/pretrain_megatron.py"
  --sft
  --episode-smoke
  --episode-grpo-train "${TRAIN}"
  --episode-grpo-valid "${VALID}"
  --episode-whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer"
  --episode-lora-rank 16
  --episode-lora-alpha 32
  --episode-lora-dropout 0.05
  --episode-top-layers 8
  --episode-adapter-lr 1e-5
  --episode-clip-epsilon 0.20
  --episode-kl-beta 0.02
  --episode-replay-weight 0.25
  --episode-anchor-weight 0.02
  --tokenizer-type NullTokenizer
  --vocab-size 180407
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers 24
  --hidden-size 896
  --ffn-hidden-size 4864
  --num-attention-heads 14
  --group-query-attention
  --num-query-groups 2
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --add-qkv-bias
  --position-embedding-type rope
  --rotary-base 1000000
  --seq-length 18000
  --max-position-embeddings 32768
  --micro-batch-size 1
  --global-batch-size 1
  --train-iters 1
  --lr 1e-5
  --min-lr 1e-6
  --lr-warmup-iters 0
  --lr-decay-iters 1
  --lr-decay-style cosine
  --dataloader-type cyclic
  --no-data-sharding
  --num-workers 0
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 0.5
  --bf16
  --use-flash-attn
  --attention-backend fused
  --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --recompute-activations
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --dist-ckpt-strictness log_all
  --finetune
  --no-load-optim
  --no-load-rng
  --load "$(dirname "${PHASE_A_CHECKPOINT}")"
  --save "${SAVE}"
  --save-interval 1
  --eval-iters 1
  --eval-interval 1
  --eval-micro-batch-size 1
  --eval-global-batch-size 1
  --log-interval 1
  --tensorboard-dir "${TB}"
  --tensorboard-log-interval 1
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval 1
  --log-world-size-to-tensorboard
  --log-throughput
  --seed 20260826
)

printf '%q ' "${CMD[@]}" > "${LOG}.command"
printf '\n' >> "${LOG}.command"
"${CMD[@]}" 2>&1 | tee "${LOG}"
