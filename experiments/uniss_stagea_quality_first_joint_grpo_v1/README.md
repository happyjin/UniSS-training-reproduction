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

Formal multi-arm launches default to `DATA_WORKERS=0`. The task-pool records
are mmap-backed and contain many tensor fields; using multiprocessing workers
for all eight ranks can exhaust the file-descriptor transport used by PyTorch's
pinned-memory queues (`received 0 items of ancdata`). This setting changes only
how an already-selected batch is read, not the global shuffle, sample order,
optimization geometry, or experiment comparison. A non-overwriting recovery
run can be named with `RUN_VARIANT`, for example:

```bash
RUN_VARIANT=full_recovery1 DATA_WORKERS=0 scripts/launch_all_tmux.sh
```

TensorBoard default port: `6017`.

The non-overwriting post-training pipeline waits for all four formal
`iter_0002510` checkpoints and for Megatron final validation to release the
GPUs. It then runs the frozen routed validation/E2E protocol, paired Stage-A
comparison, short multi-chunk listening suite, four bilingual 60-second strict
prefixes, and the best-arm versus Stage-A complete bounded-window long-form
comparison. Check dependencies without waiting or using a GPU with:

```bash
scripts/run_post_train_pipeline.sh --check
```
