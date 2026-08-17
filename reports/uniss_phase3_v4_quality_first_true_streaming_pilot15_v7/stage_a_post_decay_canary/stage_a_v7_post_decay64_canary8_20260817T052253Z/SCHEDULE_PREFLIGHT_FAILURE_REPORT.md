# Stage A V7 post-decay canary schedule-preflight report

- Run: `stage_a_v7_post_decay64_canary8_20260817T052253Z`
- Start: 2026-08-17 05:22:53 UTC
- Training updates completed: **0**
- Checkpoints produced: **none**
- Stage B authorization: **blocked**

The repaired launch passed argument validation and reached dataset
construction, then the strict schedule gate rejected it before iteration 1.
Three full shuffled coverage epochs contain 48,768 samples (381 global
updates), while the diagnostic canary requests 24,448 samples (191 updates).
The inherited provider intentionally forbids an implicit truncation.

V7 now adds an explicit canary-only prefix schedule. It first constructs the
same deterministic three-epoch globally shuffled schedule used by formal
training, then exposes exactly the first 24,448 samples while retaining the
data-parallel group geometry, collator, and synchronized sampler contract.
Formal training never enables this mode and keeps the original exact-length
gate unchanged.

The failed run is immutable diagnostic evidence and must never be resumed or
overwritten. A fresh run ID is required after regression and launch tests.
