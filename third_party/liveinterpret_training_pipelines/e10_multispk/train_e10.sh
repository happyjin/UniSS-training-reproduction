#!/usr/bin/env bash
# E10: LiveInterpret S5 trained in baseline B's form on B's AMI data (Thinker only; Talker frozen).
#   start      : S4 base + the S5 adapter checkpoint-29773 (warm start, e10.tuner), 8 speaker rows, zero e10_sd_proj
#   loss       : text 1 / speech 0 / decision 1 (SI_SPEECH_LOSS_WEIGHT=0: the Talker never runs)
#   trainable  : Thinker LoRA (r64/a128 of the S5 adapter) + 12 token rows + e10_sd_proj   (SI_S4_TRAIN_SCOPE=thinker)
#   optimiser  : lr 1.2e-5 -> 2e-6 cosine (the S5 adapter's run), warmup 3 %; e10_sd_proj at 1e-4 (B's new-module lr)
#   batch      : 8 GPUs x 8 = 64 (B's GBS); 1.5 epochs (~1670 steps) over the balanced records; last step (as B)
# Foreground only (no setsid/nohup here); a formal run is wrapped by launch_e10_detached.sh.
#
#   DRY_RUN=1 bash e10_multispk/train_e10.sh                       # print the swift command, run nothing
#   SMOKE=1 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash e10_multispk/train_e10.sh   # 5 steps on the smoke records
set -euo pipefail
E10_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LI_ROOT="$(dirname "${E10_DIR}")"
: "${SIMULS2S:=/opt/dlami/nvme/jasonleeeli/venvs/simuls2s}"
export PATH="${SIMULS2S}/bin:${PATH}"

: "${MODEL_PATH:=${LI_ROOT}/checkpoints/S4_1031v2_ckpt93767}"
: "${E10_INIT_ADAPTER:=${LI_ROOT}/checkpoints/p4mix_2n_20261117/checkpoint-29773}"
export MODEL_PATH E10_INIT_ADAPTER
if [[ "${SMOKE:-0}" == 1 ]]; then
    : "${E10_RECORDS:=${E10_DIR}/runs/smoke_records_v1}"
    : "${MAX_STEPS:=5}"; : "${SAVE_STEPS:=5}"; : "${PER_DEVICE_TRAIN_BATCH_SIZE:=2}"; : "${VAL_INTERVAL:=5}"
    : "${VAL_BATCHES:=1}"; : "${LOGGING_STEPS:=1}"; : "${DATALOADER_NUM_WORKERS:=1}"; : "${RUN_LABEL:=smoke_train}"
    : "${OUTPUT_ROOT:=${E10_DIR}/runs}"
fi
: "${E10_RECORDS:=${E10_DIR}/data/li_s5_records_v1}"
R="$(cd "${E10_RECORDS}" && pwd)"
export SI_S4_DATA="${R}/records/en2zh/train/records,${R}/records/zh2en/train/records"
export SI_S4_DATA_ROOT="${R}" SI_S4_VAL_DIR="${R}/val" SI_S4_INDEX_DIR=""

# ---- objective (B form) ----
export SI_TEXT_LOSS_WEIGHT=1 SI_SPEECH_LOSS_WEIGHT=0 SI_S4_DECISION_WEIGHT=1 SI_COND_MODE=full SI_HIDDEN_NOISE=0.0
export SI_S4_TRAIN_SCOPE=thinker SI_S4_P_DROP=0 SI_S4_PAIRED_CONTROL=0 SI_S4_PREFIX_CONTEXT_TURNS=16
# ---- samples (S5 train_simul.sh values; B: uniform shuffle -> no stratified index) ----
: "${MAX_LENGTH:=1024}"
export SI_MAX_LENGTH="${MAX_LENGTH}" SI_MAX_AUDIO_SEC=30 SI_MAX_SPEECH_LEN=750 SI_S4_USE_SRC_END=1 SI_S4_STRICT_TEXT=0
export SI_S4_LATENCY_MULTIPLIER=1 SI_S4_M_MAX=4 SI_S4_M_SCHEDULE=uniform SI_S4_STRATIFIED=0 SI_S4_MAX_REPEAT=2.0
export SI_S4_MIN_SOURCE_MS=0 SI_S4_SEED=20261026
export E10_POST_PERTURB=0.2 E10_SD_LR="${E10_SD_LR:-1e-4}"
# ---- optimisation ----
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=8}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LEARNING_RATE:=1.2e-5}"
: "${MIN_LR:=2e-6}"
: "${WARMUP_RATIO:=0.03}"
: "${NUM_TRAIN_EPOCHS:=1.5}"   # ~B's exposure: B saw each direction's rows ~1.2-1.5 times (6000 x 64 rows)
: "${SAVE_STEPS:=250}"
: "${SAVE_TOTAL_LIMIT:=20}"
: "${VAL_INTERVAL:=250}"
: "${VAL_BATCHES:=10}"
: "${NPROC_PER_NODE:=8}"
: "${OUTPUT_ROOT:=${E10_DIR}/output}"
: "${RUN_LABEL:=e10_thinker_v1}"
export NPROC_PER_NODE OUTPUT_ROOT
LORA_RANK=64; LORA_ALPHA=128; LORA_DROPOUT=0        # must equal the S5 adapter (checked in e10.tuner)
export PLUGIN_PATH="${E10_DIR}/plugin_e10.py" REPO_ROOT="${LI_ROOT}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    swift() { printf '[DRY_RUN] swift'; printf ' %q' "$@"; printf '\n'; }
    mkdir() { printf '[DRY_RUN] mkdir %s\n' "$*"; }
fi
source "${LI_ROOT}/pipelines/streaming_si/common_args.sh"

WORLD=$(( ${NPROC_PER_NODE:-1} * ${NNODES:-1} ))
export SI_S4_GLOBAL_BATCH=$(( PER_DEVICE_TRAIN_BATCH_SIZE * WORLD * GRADIENT_ACCUMULATION_STEPS ))
export SI_S4_MAX_STEPS="${MAX_STEPS}"
export SI_S4_VAL_LIMIT=$(( VAL_BATCHES * PER_DEVICE_TRAIN_BATCH_SIZE * WORLD ))

EXTRA=(--tuner_type e10_ms --optimizer e10_grouped --seed "${SI_S4_SEED}" --dataloader_drop_last true --strict true
       --val_dataset streaming_si_s4_val --eval_strategy steps --eval_steps "${VAL_INTERVAL}"
       --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" --metric_for_best_model dec_macro_f1
       --greater_is_better true)
echo "[e10] records=${R} gbs=${SI_S4_GLOBAL_BATCH} epochs=${NUM_TRAIN_EPOCHS} max_steps=${MAX_STEPS} lr=${LEARNING_RATE}" \
     "sd_lr=${E10_SD_LR} init=${E10_INIT_ADAPTER}" >&2
run_streaming_si streaming_si_s4 "${RUN_LABEL}" stream_lora "${EXTRA[@]}" "$@"
