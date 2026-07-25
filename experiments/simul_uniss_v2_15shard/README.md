# Simul-UniSS v2 — isolated 15-shard experiments

This directory is the version-controlled control plane for the corrected
Simul-UniSS experiments. It never writes into the previous
`simul_uniss_v1` checkpoint, log, run, or TensorBoard directories.

## Immutable inputs

- Prepared 15-shard train/validation data: reused read-only from `simul_uniss_v1`.
- Qwen initialization: completed UniST198 Phase3 v4, iteration 9075.
- Previous Simul-UniSS v1 outputs: retained unchanged for reproduction.

## Output namespace

- Checkpoints: `checkpoints/simul_uniss_v2_15shard/`
- Logs: `logs/simul_uniss_v2_15shard/`
- TensorBoard: `runs/simul_uniss_v2_15shard/tensorboard/`
- TensorBoard port/session: `6016` / `simul_uniss_v2_15shard_tensorboard`

## Stage layout

- `stage00_baselines/`: isolated reconstruction and prefix re-encoding baseline.
- `stage01_02_streaming_student/`: eight-GPU causal student plus Source/Target CTC heads.
- `stage03_action_sft/`: WAIT/WRITE action SFT.
- `stage04_interleaved_s2st/`: phrase-level interleaved S2ST SFT.
- `stage06_joint_refinement/`: low-learning-rate joint refinement.
- `orchestration/`: eight-GPU smoke, sequential training, and TensorBoard launchers.

Additional Stage 5/7/8 folders are added only with their distributed
training adapters, so a folder never suggests an unsupported eight-GPU path.

## Shuffle invariant

All Qwen stages in this experiment require:

```text
dataloader_type = cyclic
data_sharding = False
seed = 20260725
full_validation = True
```

The generic training implementation remains shared with historical runs, but
these settings are opt-in through `experiment.env` so old configurations retain
their original behavior.

## Execution

```bash
experiments/simul_uniss_v2_15shard/orchestration/run_shuffle_smoke_8gpu.sh
experiments/simul_uniss_v2_15shard/orchestration/start_tensorboard.sh
experiments/simul_uniss_v2_15shard/orchestration/launch_qwen_pipeline_tmux.sh
```

The smoke must finish before the formal Stage 3 → Stage 4 → Stage 6 pipeline
can start. Every stage has an independent output directory and completion
marker; an existing partial directory causes a safe failure instead of an
overwrite.

The first live eight-GPU attempt is preserved under `shuffle_smoke_8gpu`; it
stopped before iteration 1 because a two-step smoke inherited ten warmup steps.
The corrected non-overwriting attempt uses `shuffle_smoke_8gpu_v2` with zero
warmup steps.

If the controlling shell is disconnected after all three checkpoints finish,
recover only the completion state without retraining or overwriting outputs:

```bash
experiments/simul_uniss_v2_15shard/orchestration/run_shuffle_smoke_8gpu.sh \
  --verify-existing
```

This mode rechecks iteration, sampler, seed, full-validation, skipped-iteration,
and NaN invariants before writing the completion marker.
