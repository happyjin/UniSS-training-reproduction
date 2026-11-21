#!/usr/bin/env bash
# E10 v3: LiveInterpret S5 trained with baseline B's data, hyper-parameters and trainable parts (e10/v3.py).
#   STAGE=o2     B's O2: S4 + merged S5 adapter, fresh LoRA, 2000 steps x GBS 64 on li_v3_o2 (B's 67,795 O2 rows)
#   STAGE=joint  B's joint stage: + merged O2 adapter and its speaker projector, fresh LoRA, 6000 steps x GBS 24 on
#                li_v3_joint (one streaming record = all reads of a B window; 24 records ~ B's 64 rows of text
#                supervision; 6000 x 24 ~ 1.4 passes over the unique records ~ B's 1.34 over its unique rows)
#   both         LoRA r32 / alpha 64 on Thinker attention (q/k/v/o + GDN in_proj_qkv/in_proj_z/out_proj), 8 speaker
#                rows, B's SDProjector; lr 1e-4 -> 1e-5 cosine (B), warm-up 50 / 100 steps (B), text + decision loss
#                (decision masked on offline / sd records), speech loss 0 (Talker frozen); last step, adapter only.
# Foreground only; a formal run is wrapped by takeover_e10v3.sh.
#
#   DRY_RUN=1 STAGE=o2 bash e10_multispk/train_e10v3.sh
#   SMOKE=1 STAGE=o2 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash e10_multispk/train_e10v3.sh
set -euo pipefail
E10_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LI_ROOT="$(dirname "${E10_DIR}")"
: "${SIMULS2S:=/opt/dlami/nvme/jasonleeeli/venvs/simuls2s}"
export PATH="${SIMULS2S}/bin:${PATH}"
: "${STAGE:?STAGE=o2|joint}"

: "${MODEL_PATH:=${LI_ROOT}/checkpoints/S4_1031v2_ckpt93767}"
: "${S5_ADAPTER:=${LI_ROOT}/checkpoints/p4mix_2n_20261117/checkpoint-29773}"
export MODEL_PATH
case "${STAGE}" in
    o2)
        : "${E10_RECORDS:=${E10_DIR}/data/li_v3_o2}"
        export E10V3_MERGES="${S5_ADAPTER}" E10V3_SPK_INIT=""
        : "${E10_GBS:=64}"; : "${MAX_STEPS:=2000}"; : "${WARMUP_RATIO:=0.025}"; : "${RUN_LABEL:=e10v3_o2}" ;;
    joint)
        : "${O2_ADAPTER:?O2_ADAPTER=<stage-1 final checkpoint dir>}"
        : "${E10_RECORDS:=${E10_DIR}/data/li_v3_joint}"
        export E10V3_MERGES="${S5_ADAPTER},${O2_ADAPTER}" E10V3_SPK_INIT="${O2_ADAPTER}"
        : "${E10_GBS:=24}"; : "${MAX_STEPS:=6000}"; : "${WARMUP_RATIO:=0.0167}"; : "${RUN_LABEL:=e10v3_joint}" ;;
    *) echo "STAGE=${STAGE}?" >&2; exit 1 ;;
esac
export E10V3_STAGE="${STAGE}"
if [[ "${SMOKE:-0}" == 1 ]]; then
    : "${MAX_STEPS_SMOKE:=4}"; MAX_STEPS="${MAX_STEPS_SMOKE}"
    : "${SAVE_STEPS:=${MAX_STEPS}}"; : "${PER_DEVICE_TRAIN_BATCH_SIZE:=2}"; : "${VAL_INTERVAL:=${MAX_STEPS}}"
    : "${VAL_BATCHES:=1}"; : "${LOGGING_STEPS:=1}"; : "${DATALOADER_NUM_WORKERS:=1}"
    : "${OUTPUT_ROOT:=${E10_DIR}/runs}"; RUN_LABEL="smoke_${RUN_LABEL}"; GRADIENT_ACCUMULATION_STEPS=1
fi
R="$(cd "${E10_RECORDS}" && pwd)"
export SI_S4_DATA="${R}/records/en2zh/train/records,${R}/records/zh2en/train/records"
export SI_S4_DATA_ROOT="${R}" SI_S4_VAL_DIR="${R}/val" SI_S4_INDEX_DIR=""

# ---- objective (B form; decision masking per record in e10.v3) ----
export SI_TEXT_LOSS_WEIGHT=1 SI_SPEECH_LOSS_WEIGHT=0 SI_S4_DECISION_WEIGHT=1 SI_COND_MODE=full SI_HIDDEN_NOISE=0.0
export SI_S4_TRAIN_SCOPE=thinker SI_S4_P_DROP=0 SI_S4_PAIRED_CONTROL=0 SI_S4_PREFIX_CONTEXT_TURNS=16
: "${MAX_LENGTH:=1280}"   # v1 records <= 1024 tokens; + one speaker slot per 5 frames (<= 75 for 30 s)
export SI_MAX_LENGTH="${MAX_LENGTH}" SI_MAX_AUDIO_SEC=30 SI_MAX_SPEECH_LEN=750 SI_S4_USE_SRC_END=1 SI_S4_STRICT_TEXT=0
export SI_S4_LATENCY_MULTIPLIER=1 SI_S4_M_MAX=4 SI_S4_M_SCHEDULE=uniform SI_S4_STRATIFIED=0 SI_S4_MAX_REPEAT=2.0
export SI_S4_MIN_SOURCE_MS=0 SI_S4_SEED=20261026
export E10_POST_PERTURB=0.2
# ---- optimisation (B) ----
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=8}"
: "${NPROC_PER_NODE:=4}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
    PER_STEP=$(( PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * ${NNODES:-1} ))
    (( E10_GBS % PER_STEP == 0 )) || { echo "[e10v3] GBS ${E10_GBS} not divisible by ${PER_STEP}" >&2; exit 1; }
    GRADIENT_ACCUMULATION_STEPS=$(( E10_GBS / PER_STEP ))
fi
: "${LEARNING_RATE:=1e-4}"
: "${MIN_LR:=1e-5}"
: "${NUM_TRAIN_EPOCHS:=100}"          # max_steps decides
: "${SAVE_STEPS:=500}"
: "${SAVE_TOTAL_LIMIT:=20}"
: "${SAVE_ONLY_MODEL:=true}"
: "${VAL_INTERVAL:=500}"
: "${VAL_BATCHES:=10}"
: "${OUTPUT_ROOT:=${E10_DIR}/output}"
export NPROC_PER_NODE OUTPUT_ROOT
LORA_RANK=32; LORA_ALPHA=64; LORA_DROPOUT=0
export PLUGIN_PATH="${E10_DIR}/plugin_e10v3.py" REPO_ROOT="${LI_ROOT}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    swift() { printf '[DRY_RUN] swift'; printf ' %q' "$@"; printf '\n'; }
    mkdir() { printf '[DRY_RUN] mkdir %s\n' "$*"; }
fi
source "${LI_ROOT}/pipelines/streaming_si/common_args.sh"

WORLD=$(( ${NPROC_PER_NODE:-1} * ${NNODES:-1} ))
export SI_S4_GLOBAL_BATCH=$(( PER_DEVICE_TRAIN_BATCH_SIZE * WORLD * GRADIENT_ACCUMULATION_STEPS ))
export SI_S4_MAX_STEPS="${MAX_STEPS}"
export SI_S4_VAL_LIMIT=$(( VAL_BATCHES * PER_DEVICE_TRAIN_BATCH_SIZE * WORLD ))

EXTRA=(--tuner_type e10v3 --seed "${SI_S4_SEED}" --dataloader_drop_last true --strict true
       --val_dataset streaming_si_s4_val --eval_strategy steps --eval_steps "${VAL_INTERVAL}"
       --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" --metric_for_best_model dec_macro_f1
       --greater_is_better true)
echo "[e10v3] stage=${STAGE} records=${R} gbs=${SI_S4_GLOBAL_BATCH} max_steps=${MAX_STEPS} lr=${LEARNING_RATE}->${MIN_LR}" \
     "warmup=${WARMUP_RATIO} merges=${E10V3_MERGES} spk_init=${E10V3_SPK_INIT:-fresh}" >&2
run_streaming_si streaming_si_s4 "${RUN_LABEL}" stream_lora "${EXTRA[@]}" "$@"
