# UniSS Phase3 Whisper-StreamSpeech Joint v1

This experiment is the isolated implementation of the single-stage plan in:

`docs/uniss_training_reproduction/uniss_phase3_whisper_streamspeech_single_stage_joint_training_plan.md`

It keeps the successful Phase3 architecture (WhisperVQ, GLM codebook, Qwen
Phase3, BiCodec) and runs through the repository's Megatron-Core training
framework. Historical Phase1/2/3 and earlier streaming experiment directories
are read-only inputs.

## Loss and sampling

```text
joint microbatch (80%):
  1 * BiCodec Unit CTC
  8 * policy-conditioned AR S2TT
  4 * source ASR CTC
  4 * target NAR S2TT CTC

replay microbatch (20%):
  0.5 * exact old Phase3 packed causal CE
```

The Megatron entrypoint is:

```text
training/phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py
```

## Safe execution order

1. Build isolated smoke manifests and an eight-row partial replay index:

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/prepare_smoke_data.sh
   ```

2. Run a two-update, eight-GPU Megatron smoke test:

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/run_smoke_8gpu.sh
   ```

3. Build the complete full198 replay offset index in tmux (CPU/NVMe only):

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/launch_full_replay_index_tmux.sh
   ```

4. Prepare the full198 Stage-A source audio in eight GPU lanes, assemble it,
   then build the final joint manifest. These outputs are new and do not touch
   the 15-shard Stage-A tree:

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/launch_full198_stage_a_tmux.sh
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/prepare_full198_joint_manifest.sh
   ```

   For H200 nodes, the isolated high-throughput launcher keeps exact
   single-sample BiCodec decoding while running two disjoint workers per GPU:

   ```bash
   WORKERS_PER_GPU=2 bash \
     experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/launch_full198_stage_a_high_throughput_tmux.sh
   ```

   To let Stage-A completion, manifest assembly, validation and formal
   Megatron training continue automatically, launch the isolated watcher after
   the eight Stage-A lanes:

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/launch_full198_pipeline_tmux.sh
   ```

5. Start the formal single-stage run only after the smoke gate and both full
   data completion markers pass:

   ```bash
   bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/run_full198_8gpu.sh
   ```

## TensorBoard

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/start_tensorboard.sh
```

Default local endpoint: `http://127.0.0.1:6031/`. Use SSH forwarding from a
client machine when needed:

```bash
ssh -L 6031:127.0.0.1:6031 root@SERVER
```

All checkpoints, TensorBoard events and logs use new names under
`checkpoints/`, `runs/` and `logs/`; every runner refuses to overwrite an
existing destination.
