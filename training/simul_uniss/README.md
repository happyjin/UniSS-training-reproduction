# Simul-UniSS bootstrap implementation

This directory is isolated from the existing Phase1–3 implementation. It does
not change legacy sample layouts, packers, datasets, entrypoints, checkpoints,
or experiment configurations.

The initial schedule builder uses proportional token timing because public
UniST parquet rows contain token sequences but no source/target word timestamps.
Every generated schedule is marked:

```text
alignment_kind=pseudo_proportional_token_alignment
```

It is intended to validate the complete data, packing, policy, TensorBoard, and
training plumbing. It must be replaced with audio timestamp or contextual
alignment before reporting formal simultaneous translation results.

Prepare the fixed 15-shard bootstrap set:

```bash
scripts/simul_uniss/prepare_bootstrap_15shard.sh
```

Run a small smoke preparation:

```bash
scripts/simul_uniss/prepare_bootstrap_15shard.sh --limit-records 20
```

Start its TensorBoard service:

```bash
scripts/simul_uniss/start_tensorboard.sh
```

Train the causal token bootstrap student and Source/Target CTC heads:

```bash
scripts/simul_uniss/train_stage1_bootstrap_student.sh
```

The Stage1 bootstrap consumes 50 Hz `source_bicodec` tokens and distills the
existing `source_glm` targets. This is a runnable plumbing and loss-validation
stage, not a substitute for the later audio Streaming GLM student.
