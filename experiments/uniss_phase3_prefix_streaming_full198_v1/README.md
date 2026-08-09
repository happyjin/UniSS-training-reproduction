# Phase3 Full198 Prefix-Streaming Joint V1

This experiment is isolated from all historical UniSS training and evaluation
directories. It initializes the existing best Phase3 model and trains only
rank-16 Qwen LoRA parameters with a single 12,000-iteration curriculum:

1. exact Phase3 Quality/Performance replay;
2. random-prefix streaming S2TT;
3. frozen full-context Phase3 top-k distillation;
4. adjacent-prefix consistency;
5. target BiCodec short-block continuation;
6. automatically derived WAIT/WRITE supervision.

It does not add a Talker, expand the vocabulary, mutate historical packed data,
or create word-aligned READ/WRITE trajectories.

## Provenance

```text
source checkpoint:
  checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf

train data:
  data/raw/UniST/train-00000.parquet ... train-00197.parquet
  19,785,924 rows

validation:
  data/raw/UniST/dev-00000.parquet

formal output:
  checkpoints/uniss_phase3_prefix_streaming_full198_joint_v1
  runs/uniss_phase3_prefix_streaming_full198_joint_v1/tensorboard
  logs/uniss_phase3_prefix_streaming_full198_joint_v1.log
```

## Safety

All launchers refuse to overwrite a checkpoint, TensorBoard directory, or log.
`RESUME=1` is accepted only when the experiment checkpoint tracker exists. The
GPU stop helper matches only this repository's known synthetic holder script.

## Commands

```bash
python -m experiments.uniss_phase3_prefix_streaming_full198_v1.build_direction_index \
  --input-dir data/raw/UniST \
  --output-dir data/processed/uniss_phase3_prefix_streaming_full198_v1/direction_index \
  --workers 32

bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_smoke_1gpu.sh
bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_smoke_8gpu.sh

tmux new-session -d -s uniss_phase3_prefix_streaming_full198_joint_v1 \
  "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && \
   bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_megatron.sh"
```

TensorBoard defaults to port `6065`:

```bash
bash experiments/uniss_phase3_prefix_streaming_full198_v1/start_tensorboard.sh
```

