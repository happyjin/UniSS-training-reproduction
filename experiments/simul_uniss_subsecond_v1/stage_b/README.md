# Simul-UniSS Subsecond v1 — Stage B

Stage B trains an isolated Emformer causal audio student. The historical
WhisperVQ, Qwen, BiCodec, Phase1–3, and prior Simul-UniSS checkpoints remain
read-only.

Architecture defaults:

- 16 kHz, 128-bin log-Mel, `center=False`;
- four-frame stacking, one hidden frame per 40 ms;
- 160 ms Emformer segment;
- 80 ms right context;
- bounded 2-second left context;
- GLM CTC, source CTC, target-capacity, and stability heads.

## Eight-GPU smoke

```bash
scripts/simul_uniss_subsecond_v1/train_stage_b.sh --smoke
```

The smoke uses all eight GPUs and then runs cache-parity and future-
perturbation tests.

## Formal 15-shard training

After Stage A source preparation is complete:

```bash
scripts/simul_uniss_subsecond_v1/train_stage_b.sh --formal
```

Resume safely with:

```bash
scripts/simul_uniss_subsecond_v1/train_stage_b.sh --formal --resume
```

## TensorBoard

```bash
scripts/simul_uniss_subsecond_v1/start_stage_b_tensorboard.sh
```

Default port: `6050`.
