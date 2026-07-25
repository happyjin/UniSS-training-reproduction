# Simul-UniSS v3 — isolated full UniST198 experiment

This experiment makes all 198 UniST training shards available to the
Simul-UniSS stages while preserving every v1/v2 input and output.

## What is shared and what is different

The model and reader implementations are shared:

- `training/pretrain_simul_uniss_megatron.py` and its packed JSONL dataset;
- the streaming token/audio student, BiCodec refinement, GRPO, and NAR modules;
- the same token IDs, `simul_uniss_packed_v1` schema, model architecture, Phase3
  iteration-9075 initialization, fixed dev set, and eight-GPU launch pattern.

The full experiment is not only a configuration copy. It adds:

- 198-shard parallel, resumable, checksummed schedule generation;
- per-shard action/interleaved packing and deterministic atomic assembly;
- a separate full-data manifest and readiness marker;
- exact Stage 3/4/6 iteration and warmup generation from real packed counts;
- validated byte-offset sidecars, avoiding an eight-process scan of each
  hundreds-of-GB JSONL file at every training startup;
- independent data/checkpoint/log/run/TensorBoard namespaces and ports;
- stratified Stage 0 audio reconstruction across all shards.

## Isolated namespace

- Prepared data: `data/processed/simul_uniss_v3_full198/`
- Megatron data: `data/megatron/simul_uniss_v3_full198/`
- Checkpoints: `checkpoints/simul_uniss_v3_full198/`
- Logs: `logs/simul_uniss_v3_full198/`
- TensorBoard: `runs/simul_uniss_v3_full198/tensorboard/`, port `6017`

Validation data and the policy tokenizer are intentionally reused read-only so
the 15-shard/full198 comparison changes the training set, not the dev set or
vocabulary.

## Safety gates and order

```bash
# 1. Inspect commands.
experiments/simul_uniss_v3_full198/data_preparation/run_full_preparation.sh --dry-run

# 2. Validate the real pipeline on 2 shards x 4 records.
experiments/simul_uniss_v3_full198/orchestration/run_data_smoke.sh

# 3. Prepare all 198 shards in tmux.
experiments/simul_uniss_v3_full198/data_preparation/launch_tmux.sh

# 4. After FULL_DATA_READY.json exists, run the eight-GPU shuffle smoke.
experiments/simul_uniss_v3_full198/orchestration/run_shuffle_smoke_8gpu.sh

# 5. Start TensorBoard and the formal Stage 3 -> 4 -> 6 pipeline.
experiments/simul_uniss_v3_full198/orchestration/start_tensorboard.sh
experiments/simul_uniss_v3_full198/orchestration/launch_qwen_pipeline_tmux.sh
```

Stage 3 and Stage 4 initially target one true packed-data epoch. Stage 6 is a
quarter-epoch low-LR refinement. These values are explicit protocol defaults in
`experiment.env`; the generated `training_schedule.env` contains the exact
iterations and can be regenerated under a new isolated experiment if the
protocol changes.

Stage 8 remains profiling-gated and is never auto-started. Full preparation or
training is deliberately not auto-launched merely by adding this directory.
