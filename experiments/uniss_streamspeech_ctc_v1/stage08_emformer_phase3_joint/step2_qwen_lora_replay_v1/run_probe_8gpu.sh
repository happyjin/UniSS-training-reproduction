#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
STEP=$ROOT/experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step2_qwen_lora_replay_v1
SOURCE=$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-stage08_step2_qwen_lora_replay_r8_replay30_research_v1}
REPORT_RUN_NAME=${REPORT_RUN_NAME:-stage08_step2_qwen_lora_replay_probe32_research_v1}
CHECKPOINT_ITERS=${CHECKPOINT_ITERS:-"50 100 150 200"}
SAMPLES_PER_WORKER=${SAMPLES_PER_WORKER:-4}
RUN_DIR=$ROOT/reports/uniss_streamspeech_ctc_v1/$REPORT_RUN_NAME
MEGATRON_ROOT=$ROOT/checkpoints/uniss_streamspeech_ctc_v1/$TRAIN_RUN_NAME
REFERENCE=$ROOT/reports/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_gate32_v1/iter_0000350/report.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_KERNEL_CACHE_PATH=$USER_ROOT/cache/torch_kernel
export TMPDIR=$USER_ROOT/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"

SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBRARY_DIRS=()
shopt -s nullglob
for directory in "$SITE_PACKAGES"/nvidia/*/lib; do
  [[ -d "$directory" ]] && NVIDIA_LIBRARY_DIRS+=("$directory")
done
shopt -u nullglob
(( ${#NVIDIA_LIBRARY_DIRS[@]} > 0 )) || exit 1
NVIDIA_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIBRARY_DIRS[*]}")"
export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$REFERENCE"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"

candidate_jsons=()
for iteration in $CHECKPOINT_ITERS; do
  printf -v checkpoint_name 'iter_%07d' "$iteration"
  checkpoint=$MEGATRON_ROOT/$checkpoint_name
  test -f "$checkpoint/.metadata"
  candidate_dir=$RUN_DIR/$checkpoint_name
  parts=$candidate_dir/parts
  mkdir -p "$parts"
  pids=()
  for rank in 0 1 2 3 4 5 6 7; do
    direction=$((rank / 4))
    direction_offset=$(((rank % 4) * SAMPLES_PER_WORKER))
    CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" "$STEP/evaluate_step2_text_probe.py" \
      --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
      --source-manifest "$SOURCE" \
      --source-offsets "${SOURCE}.offsets.bin" \
      --ctc-tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
      --stage03b-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
      --historical-stage-b-checkpoint "$ROOT/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt" \
      --stage04-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1/best.pt" \
      --stage06-initialize-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage06_b1_megatron_v2/iter_0000600" \
      --step1-megatron-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_p3w2_zhen1p25_v1/iter_0000350" \
      --step2-megatron-checkpoint "$checkpoint" \
      --codebook-model "$ROOT/pretrained_models/UniSS/glm4_tokenizer" \
      --phase3-model "$ROOT/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf" \
      --direction-id "$direction" \
      --direction-offset "$direction_offset" \
      --max-samples "$SAMPLES_PER_WORKER" \
      --output-json "$parts/part_${rank}.json" \
      >"$parts/part_${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || {
    echo "Stage08 Step2 probe failed for $checkpoint_name" >&2
    exit 1
  }
  "$PYTHON" "$STEP/merge_step2_text_probe.py" \
    --parts "$parts"/part_?.json \
    --output-json "$candidate_dir/report.json" \
    --output-md "$candidate_dir/report.md"
  candidate_jsons+=("$candidate_dir/report.json")
done

"$PYTHON" "$STEP/compare_step2_research.py" \
  --reference-json "$REFERENCE" \
  --candidate-json "${candidate_jsons[@]}" \
  --output-json "$RUN_DIR/comparison.json" \
  --output-md "$RUN_DIR/comparison.md"
