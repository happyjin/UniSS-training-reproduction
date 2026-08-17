# Stage A V7 post-decay canary preflight report

- Run: `stage_a_v7_post_decay64_canary8_20260817T051442Z`
- Start: 2026-08-17 05:14:42 UTC
- Training updates completed: **0**
- Checkpoints produced: **none**
- Stage B authorization: **blocked**

The first V7 launch was rejected by the inherited Stage A argument validator
before model construction or optimizer updates. The wrapper supplied
`stage_a_coverage_epochs=2`, while strict Stage A requires the formal
three-coverage-epoch metadata.

This was a wrapper preflight defect, not a numerical training failure. The
`train_iters=191`, curriculum horizon 127, optimizer horizon 127, and warmup
6 settings were accepted and remain unchanged. The repair restores
`RUN_COVERAGE_EPOCHS=3`; the explicit 191-update cap still defines the actual
canary duration.

The failed run directory is retained as immutable diagnostic evidence and
must never be resumed or overwritten. A new run ID is required after the
repair passes regression tests.
