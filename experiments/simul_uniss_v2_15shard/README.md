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
- `stage05_streaming_bicodec/`: overlap baseline and eight-GPU boundary refinement.
- `stage06_joint_refinement/`: low-learning-rate joint refinement.
- `stage07_grpo/`: eight-GPU WAIT/WRITE policy GRPO bootstrap.
- `stage08_nar_optional/`: profiling-gated eight-GPU NAR semantic branch.
- `orchestration/`: eight-GPU smoke, sequential training, and TensorBoard launchers.

Every implemented training stage now has an isolated local folder and an
eight-GPU launcher. Scope limitations of bootstrap implementations are stated
inside the corresponding stage README instead of being hidden.

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
experiments/simul_uniss_v2_15shard/orchestration/launch_component_pipeline_when_ready.sh
```

The smoke must finish before the formal Stage 3 → Stage 4 → Stage 6 pipeline
can start. Every stage has an independent output directory and completion
marker; an existing partial directory causes a safe failure instead of an
overwrite.

The component waiter does not contend with Qwen training. It starts Stage 0,
Stage 1/2, Stage 5, and the Stage 7 policy bootstrap only after the formal Qwen
pipeline writes its completion marker. Stage 8 remains profiling-gated and is
never auto-started.

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

If a formal stage reached its target checkpoint and validation but the launcher
ended before writing its pipeline marker, recover only after verifying all
eight distributed checkpoint shards, the final training iteration, final
validation, and zero skipped/NaN iterations:

```bash
experiments/simul_uniss_v2_15shard/orchestration/launch_qwen_pipeline_tmux.sh \
  --recover-completed
```

This never overwrites or resumes a completed stage. Missing later stages still
run normally, and a nonzero post-training launcher status is accepted only when
the completed checkpoint and log pass the same verification.
