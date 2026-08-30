#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
RUN_ID=${1:?usage: run_coverage_grpo_8gpu.sh RUN_ID PACK_ID LOAD_ROOT}
PACK_ID=${2:?missing pack ID}
LOAD_ROOT=${3:?missing checkpoint root}
DATA=${REPO_ROOT}/data/processed/uniss_phase3_content_first_joint_s2st_v1/${PACK_ID}
TRAIN=${DATA}/train_packs.jsonl
VALID=${DATA}/valid_packs.jsonl
SAVE=${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1/${RUN_ID}
TB=${REPO_ROOT}/runs/uniss_phase3_content_first_joint_s2st_v1/tensorboard/${RUN_ID}
LOG=${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1/${RUN_ID}.log
for path in "${TRAIN}" "${TRAIN}.offsets.bin" "${VALID}" "${VALID}.offsets.bin" \
  "${LOAD_ROOT}/latest_checkpointed_iteration.txt"; do
  [[ -f "${path}" ]] || { echo "missing GRPO input: ${path}" >&2; exit 2; }
done
[[ ! -e "${SAVE}" && ! -e "${TB}" && ! -e "${LOG}" ]] || {
  echo "refusing to overwrite ${RUN_ID}" >&2; exit 3;
}
mkdir -p "${SAVE}" "${TB}" "$(dirname "${LOG}")"
TRAIN_RECORDS=$(($(stat -c %s "${TRAIN}.offsets.bin") / 8))
STEPS_PER_ROUND=$(( (TRAIN_RECORDS + 8 - 1) / 8 ))
TRAIN_ITERS=${TRAIN_ITERS:-${STEPS_PER_ROUND}}
SMOKE_ARGS=()
[[ ${EVENT_SMOKE:-0} == 1 ]] && SMOKE_ARGS=(--event-smoke)

export HF_HOME=${USER_ROOT}/.cache/huggingface TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export PATH=$(dirname "${PYTHON}"):${PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
export UNISS_E2E_COMPILE_CACHE_ROOT=${USER_ROOT}/.cache/uniss_content_first_grpo/${RUN_ID}
SITE_PACKAGES=$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_LIBRARY_PATH=$(find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CMD=(
  "$(dirname "${PYTHON}")/torchrun" --nproc_per_node 8 --master_port "${MASTER_PORT:-29996}"
  -m experiments.uniss_phasea_coverage_constrained_grpo_v3.training.pretrain_event_grpo
  --sft --event-grpo-train "${TRAIN}" --event-grpo-valid "${VALID}"
  --event-whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer"
  --event-lora-rank 16 --event-lora-alpha 32 --event-lora-dropout 0.05
  --event-top-layers 8 --event-adapter-lr 5e-6 --event-clip-epsilon 0.20
  --event-kl-beta 0.03 --event-replay-weight 0.50 --event-anchor-weight 0.03
  "${SMOKE_ARGS[@]}"
  --tokenizer-type NullTokenizer --vocab-size 180407
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 18000 --max-position-embeddings 32768
  --micro-batch-size 1 --global-batch-size 8 --train-iters "${TRAIN_ITERS}"
  --lr 5e-6 --min-lr 5e-7 --lr-warmup-iters 1 --lr-decay-iters "${TRAIN_ITERS}"
  --lr-decay-style cosine --dataloader-type cyclic --no-data-sharding --num-workers 0
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 0.5
  --bf16 --use-flash-attn --attention-backend fused --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion --recompute-activations --attention-dropout 0 --hidden-dropout 0
  --dist-ckpt-strictness log_all --finetune --no-load-optim --no-load-rng
  --load "${LOAD_ROOT}" --save "${SAVE}" --save-interval "${TRAIN_ITERS}"
  --eval-iters 2 --eval-interval "${TRAIN_ITERS}" --eval-micro-batch-size 1 --eval-global-batch-size 8
  --log-interval 1 --tensorboard-dir "${TB}" --tensorboard-log-interval 1
  --log-timers-to-tensorboard --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard --log-memory-interval 1 --log-world-size-to-tensorboard
  --log-throughput --seed 20260830
)
{
  echo "TRAIN_RECORDS=${TRAIN_RECORDS}"
  echo "FRESH_ROLLOUT_EPOCHS=1"
  echo "TRAIN_ITERS=${TRAIN_ITERS}"
  printf '%q ' "${CMD[@]}"; printf '\n'
} >"${LOG}.command"
"${CMD[@]}" 2>&1 | tee "${LOG}"
