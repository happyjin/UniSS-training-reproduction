#!/usr/bin/env bash
# T1: UniSS as a streaming Talker, from Plan 1's R2, on the read/write data
# (data/unist_simul_train_v1) with offline replay. See training/si_model.py.
#
# Only the talker is checkpointed (about 22 GB a save with its optimizer
# state); set SAVE_INTERVAL with the disk in mind. EXIT_INTERVAL stops the
# run at that step after saving -- used for the first probe; launching again
# with the same RUN_ID and the same settings resumes from that save.
#
#   RUN_ID=si_t1_$(date -u +%Y%m%dT%H%M%SZ) bash experiments/uniss_qwen3_omni_v1/scripts/run_si.sh
#   DRY_RUN=1 ... prints the command and starts nothing.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

R2_EXPORT="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/e2_clean1_kl05_teacher_20261021T095400Z/full_iter2000_natclip_loop8_min30/export/uniss"
RUN_ID="${RUN_ID:?set RUN_ID, e.g. si_t1_20261021T150000Z}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen3_omni_v1/${RUN_ID}}"
TB_DIR="${TB_DIR:-${REPO_ROOT}/runs/uniss_qwen3_omni_v1/${RUN_ID}}"
NPROC="${NPROC:-4}"; MBS="${MBS:-16}"; GBS="${GBS:-256}"
TRAIN_ITERS="${TRAIN_ITERS:-3000}"; WARMUP_ITERS="${WARMUP_ITERS:-100}"
LR="${LR:-1e-5}"; MIN_LR="${MIN_LR:-1e-6}"
LOAD_ARG=""
[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && LOAD_ARG="--load ${SAVE_DIR}"

CMD=("${ENV_ROOT}/bin/python" -m torch.distributed.run
  --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT:-29741}"
  experiments/uniss_qwen3_omni_v1/training/pretrain_si.py
  --uniss-path "${UNISS_PATH:-${R2_EXPORT}}"
  --teacher-path "${TEACHER_PATH:-${R2_EXPORT}}"
  --anchor-path "${ANCHOR_PATH:-${REPO_ROOT}/pretrained_models/UniSS_full}"
  --qwen3-path "${REPO_ROOT}/pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
  --si-index "${SI_INDEX:-${REPO_ROOT}/data/unist_simul_train_v1}"
  --omni-audio-index "${AUDIO_INDEX:-${REPO_ROOT}/data/unist_clean_v1/audio_index_clean_v1}"
  --omni-parquet-root "${REPO_ROOT}/data/raw/UniST"
  --omni-audio-root "${REPO_ROOT}/data/unist_omni_v1/speech"
  --si-share "${SI_SHARE:-0.7}" --si-delta-ms "${DELTA_MS:-200}" --si-holdback "${HOLDBACK:-12}"
  --si-m-max "${M_MAX:-6}" --si-read-cap "${READ_CAP:-0.25}"
  --si-w-future "${W_FUTURE:-0.5}" --si-w-anchor "${W_ANCHOR:-0.5}" --si-anchor-cmn-mult "${ANCHOR_CMN_MULT:-2.0}"
  ${INIT_WAIT_FROM_END:+--si-init-wait-from-end} --si-close-weight "${CLOSE_WEIGHT:-1.0}"
  --si-word-share "${WORD_SHARE:-0}" --si-word-windows "${WORD_WINDOWS:-1,2,3}" --si-word-min "${WORD_MIN:-1,2,3}"
  ${BLOCK_SPEED:+--si-block-speed}
  --si-lookahead-max "${LOOKAHEAD_MAX:-0}" --si-lookahead-noise "${LOOKAHEAD_NOISE:-0.3}" --si-lookahead-none "${LOOKAHEAD_NONE:-0.2}"
  ${INIT_LOOK:+--si-init-look}
  --tokenizer-type NullTokenizer --vocab-size 8448
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864
  --num-attention-heads 14 --group-query-attention --num-query-groups 2
  --normalization RMSNorm --swiglu --disable-bias-linear --add-qkv-bias
  --position-embedding-type rope --rotary-base 1000000
  --seq-length 640 --max-position-embeddings 32768
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR}" --min-lr "${MIN_LR}"
  --lr-warmup-iters "${WARMUP_ITERS}"
  --lr-decay-iters "${TRAIN_ITERS}" --lr-decay-style cosine
  --dataloader-type single --no-data-sharding
  --num-workers "${NUM_WORKERS:-6}"
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
  --bf16 --no-create-attention-mask-in-dataloader
  --no-gradient-accumulation-fusion
  --use-distributed-optimizer
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-500}" ${NO_SAVE_OPTIM:+--no-save-optim --no-save-rng}
  ${LOAD_ARG} ${NO_LOAD_OPTIM:+--no-load-optim --no-load-rng}
  --eval-iters 0 --eval-interval "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL:-10}"
  --tensorboard-dir "${TB_DIR}" --log-throughput
  ${EXIT_INTERVAL:+--exit-interval "${EXIT_INTERVAL}"}
  --seed 20261021)

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
  echo "SAVE_DIR=${SAVE_DIR}  LOAD=${LOAD_ARG:-none}"
  exit 0
fi
mkdir -p "${SAVE_DIR}" "${TB_DIR}"
"${CMD[@]}"
status=$?
[ "${status}" -ne 0 ] && { echo "SI_T1_MEGATRON_FAILED status=${status}"; exit "${status}"; }
echo "SI_T1_MEGATRON_DONE"
