# Phase3 v4 quality-first true-streaming pilot15 v2

This experiment repairs the failed Stage A v1 gate without modifying any v1
code, packs, checkpoints, logs, diagnosis, or reports.

The repair order is intentionally gated:

1. build an exact matching-sample Phase3 offline ASR anchor for the 334 formal
   Stage A validation samples;
2. make same-prefix teacher supervision active and fail fast when its
   denominator is zero;
3. add final-checkpoint cached/full, future-perturbation, and rollback gates;
4. run an isolated 8-GPU smoke and formal Stage A v2 training;
5. start Stage B only after Stage A v2 writes a passing selection artifact.

All v2 data and outputs use paths containing
`uniss_phase3_v4_quality_first_true_streaming_pilot15_v2`.
