# Simul-UniSS Subsecond v1 — Stage B

Stage B trains an isolated Emformer causal audio student. The historical
WhisperVQ, Qwen, BiCodec, Phase1–3, and prior Simul-UniSS checkpoints remain
read-only.

Formal Student-B defaults:

- 16 kHz, 128-bin log-Mel, `center=False`;
- four-frame stacking, one hidden frame per 40 ms;
- 160 ms Emformer segment;
- 80 ms right context;
- bounded 2-second left context;
- hidden size 768, 16 Emformer layers, 12 heads, FFN 3072;
- BF16, 64 examples per GPU, eight DDP ranks (global batch 512);
- eight DataLoader workers per rank;
- GLM CTC, source CTC, target-capacity, and stability heads.

The 2026-07-30 H200 scan compared 32 and 64 examples per rank on an isolated
1,920-record manifest. Batch 64 achieved about 86% mean active GPU utility,
100% peak utility, 295 W mean active power, and 39 GB peak process memory on
the GPU that also hosted the demo. Batch 32 achieved about 82%, 100%, 283 W,
and 25 GB respectively. The larger batch was selected for the formal pilot;
the historical experiments are unaffected.

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

## Automatic Stage A then Stage B

The resilient pipeline waits for the Stage-A completion marker, then starts
one eight-GPU Stage-B DDP job. A failed attempt resumes/retries after 30
seconds without deleting completed shards or checkpoints:

```bash
scripts/simul_uniss_subsecond_v1/run_stage_ab_pipeline.sh
```

## TensorBoard

```bash
scripts/simul_uniss_subsecond_v1/start_stage_b_tensorboard.sh
```

Default port: `6050`.
