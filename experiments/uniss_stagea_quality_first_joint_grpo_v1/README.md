# Stage-A quality-first joint GRPO v1

This isolated experiment starts every comparison arm from the immutable Stage A
`iter_0000381` checkpoint and consumes the audited 15-shard interleaved E2E
task pool. It does not edit or overwrite Stage A, Stage B, Phase 1--3, prior
GRPO, evaluation, or demo assets.

Four matched 2-GPU arms run concurrently:

| Arm | GPUs | Objective |
|---|---:|---|
| A1 | 0,1 | 2510 updates of matched continued SFT |
| A2 | 2,3 | 256-update SFT bootstrap then quality-first GRPO G4 |
| A3 | 4,5 | 256-update SFT bootstrap then quality-first GRPO G8 |
| A4 | 6,7 | G8 with seed-2 and stronger completeness/reference protection |

All arms use the same strict global shuffle geometry, one complete coverage of
the 40,150 interleaved 18k packs, the same Stage-A initialization, top-eight
Qwen LoRA capacity, optimizer budget, validation split, and final evaluation.
Only NaN, invalid checkpoint/data structure, and process failure abort a run.
Quality gates are reported after training and never truncate an arm.

TensorBoard default port: `6017`.

