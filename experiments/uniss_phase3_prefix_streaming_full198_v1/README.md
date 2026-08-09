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
  19,785,924 source rows
  19,285,109 valid training rows (12,421,395 EN source; 6,863,714 ZH source)
  500,815 rejected incomplete/incompatible rows

validation:
  data/raw/UniST/dev-00000.parquet
  deterministic 1:1 EN-source/ZH-source interleaving (1024 rows)

current formal output:
  checkpoints/uniss_phase3_prefix_streaming_full198_joint_v3
  runs/uniss_phase3_prefix_streaming_full198_joint_v3/tensorboard
  logs/uniss_phase3_prefix_streaming_full198_joint_v3.log
```

## Run history

- `joint_v1` stopped before its first checkpoint when a malformed one-token
  target BiCodec sequence reached the semantic continuation builder.  Its log
  and TensorBoard output are retained.  The immutable v3 direction index adds
  the corresponding data-quality gate.
- `joint_v2` safely reached checkpoint 2500, but audit logs showed that its
  task mix stayed at the first curriculum point after iteration 1500.  The
  cause was reading Megatron's checkpoint/start field `args.iteration` rather
  than its live loop field `args.curr_iteration`.  Its checkpoints, log, and
  TensorBoard output are retained for diagnosis and must not be treated as the
  completed curriculum experiment.
- `joint_v3` includes the live-iteration fix.  The real 8-GPU run changed from
  the initial approximately `40/50/10/0` replay/prefix/semantic/commit mix to
  approximately `30/50/15/5` after iteration 1500, with nonzero commit and
  action losses.  This is the current formal run.

## Safety

All launchers refuse to overwrite a checkpoint, TensorBoard directory, or log.
`RESUME=1` is accepted only when the experiment checkpoint tracker exists. The
GPU stop helper matches only this repository's known synthetic holder script.

## Commands

```bash
python -m experiments.uniss_phase3_prefix_streaming_full198_v1.build_direction_index \
  --input-dir data/raw/UniST \
  --output-dir data/processed/uniss_phase3_prefix_streaming_full198_v1/direction_index_valid_v3 \
  --workers 32

bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_smoke_1gpu.sh
bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_smoke_8gpu.sh

tmux new-session -d -s uniss_phase3_prefix_streaming_full198_joint_v3 \
  "cd /opt/dlami/nvme/jasonleeeli/projects/UniSS && \
   env RUN_NAME=uniss_phase3_prefix_streaming_full198_joint_v3 \
       MASTER_PORT=29666 TENSORBOARD_PORT=6066 \
   bash experiments/uniss_phase3_prefix_streaming_full198_v1/run_megatron.sh"
```

The current v3 TensorBoard uses port `6066` (v2 remains preserved on `6065`):

```bash
env RUN_NAME=uniss_phase3_prefix_streaming_full198_joint_v3 \
    TENSORBOARD_PORT=6066 \
  bash experiments/uniss_phase3_prefix_streaming_full198_v1/start_tensorboard.sh
```

The formal run uses 64-row direction-local blocks.  With global batch 128,
each optimizer step receives 64 EN-source and 64 ZH-source examples while the
block order remains deterministically shuffled across all 198 shards.
The v3 index excludes records with empty text/source GLM, fewer than two target
BiCodec semantic tokens, or a BiCodec global sequence not exactly 32 tokens;
it never rewrites the source parquet files.  The formal global batch remains
128, using micro-batch 8 with two accumulation steps to leave safe memory
headroom for dynamic-length samples.

An 8-GPU micro-batch-16 stress run reached about 140.5 GiB on a 143.8 GiB
H200 and was therefore rejected for the formal run.  Micro-batch 8 completed a
400-step regression, validation, and distributed checkpoint with roughly
57--72 GiB observed per GPU.
