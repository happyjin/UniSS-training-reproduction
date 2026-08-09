#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2b_existing_nar_head_v1}
V6=$ROOT/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6
STAGE_A=${STAGE_A:-$V6/phase3_joint_v6_stage_a_heads_only_full198_v1}
STAGE_B=${STAGE_B:-$V6/phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1}

export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"

cd "$ROOT"

CHECKPOINTS=()
if [[ -z "${SKIP_DEFAULT_CHECKPOINTS:-}" ]]; then
  for iteration in ${STAGE_A_ITERS:-500}; do
    CHECKPOINTS+=(--checkpoint "stage_a_iter${iteration}=$STAGE_A/$(printf 'iter_%07d' "$iteration")")
  done
  for iteration in ${STAGE_B_ITERS:-2500 5000}; do
    CHECKPOINTS+=(--checkpoint "stage_b_iter${iteration}=$STAGE_B/$(printf 'iter_%07d' "$iteration")")
  done
fi

"$PYTHON" -m experiments.simul_s2st_route_v1.step2_nar_ctc_head.evaluate_existing_head \
  --run-name "$RUN_NAME" \
  --samples-per-direction "${SAMPLES_PER_DIRECTION:-16}" \
  --min-audio-seconds "${MIN_AUDIO_SECONDS:-2.0}" \
  --max-audio-seconds "${MAX_AUDIO_SECONDS:-10.0}" \
  --device "${DEVICE:-cuda:0}" \
  "${CHECKPOINTS[@]}" \
  --output-json "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.json" \
  --output-md "$ROOT/reports/simul_s2st_route_v1/${RUN_NAME}.md" \
  "$@"
