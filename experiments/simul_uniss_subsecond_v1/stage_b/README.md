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

The later full-manifest profile measured the original batch-64 implementation
at about 7,800 audio seconds/s, 91.8% active utility, and 459 W active power.
Increasing the audio cap to 16 seconds raised resident memory beyond 100 GB on
some ranks but reduced throughput to about 4,300 audio seconds/s. Batch 128
only reached about 8,300 audio seconds/s and changed the global batch to 1,024.
A 1,024-wide student reduced throughput to about 6,300 audio seconds/s. These
variants were rejected.

The selected v2 removes the per-example GPU Python loop used to pack variable
utterance and right-context frames. A batched mask/gather implementation is
forward- and gradient-identical to the reference implementation and reaches
about 8,450--8,590 audio seconds/s at 92--100% utility. Power remains roughly
440--500 W because 160 ms Emformer segments are sequential, small-kernel
streaming work; the 700 W board limit is not a meaningful utilization target
for this architecture. The best full198 Phase3 used Qwen sequence length
18,000 (not 13,000), which is a different dense long-token workload.

The isolated continuation config is:

```text
configs/experiments/simul_uniss_subsecond_v1/stage_ab_vectorized_v2.env
```

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
