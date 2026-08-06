# Phase3 Whisper + StreamSpeech joint V5 stabilization

V5 is an isolated repair experiment. It never resumes or overwrites the
diverged `phase3_whisper_streamspeech_joint_full198_v4` run.

The repair changes are opt-in and leave the historical V1/V4 launch scripts
reproducible:

- keep the original WhisperVQ/Phase3 semantic codebook immutable;
- disable EMA mutation and dead-code restart for V5;
- replace the unconstrained learned projection STE with a hard-forward,
  top-k-soft-backward surrogate;
- optimize a masked fixed-codebook commitment loss with weight `0.25`;
- train only the four Whisper layers immediately before the historical
  pooling/VQ boundary;
- reduce the base LR from `1e-4` to `2e-5`, with Qwen and Whisper using lower
  per-group multipliers;
- stop immediately if per-microbatch masked commitment exceeds `5.0`;
- refuse to reuse any checkpoint, TensorBoard, or log output directory.

Execution order:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v5/scripts/run_smoke_8gpu.sh
bash experiments/uniss_phase3_whisper_streamspeech_joint_v5/scripts/prepare_15shard_joint_manifest.sh
bash experiments/uniss_phase3_whisper_streamspeech_joint_v5/scripts/start_tensorboard.sh
bash experiments/uniss_phase3_whisper_streamspeech_joint_v5/scripts/launch_15shard_tmux.sh
```

TensorBoard defaults to `http://127.0.0.1:6032/`.
