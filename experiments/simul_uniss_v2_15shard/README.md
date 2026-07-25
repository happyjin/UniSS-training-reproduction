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

- `stage03_action_sft/`: WAIT/WRITE action SFT.
- `stage04_interleaved_s2st/`: phrase-level interleaved S2ST SFT.
- `stage06_joint_refinement/`: low-learning-rate joint refinement.
- `orchestration/`: eight-GPU smoke, sequential training, and TensorBoard launchers.

Additional Stage 0/1/2/5/7/8 folders are added only with their distributed
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

