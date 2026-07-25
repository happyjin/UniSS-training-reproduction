# Stage 8 — optional NAR semantic generator

This branch is gated by profiling. Run it only if Stage 4/6 autoregressive
semantic generation fails the target real-time threshold.

`run_8gpu.sh` uses eight-GPU DDP. The bounded shuffled iterable is partitioned
without overlap, gradients and logged metrics are global, and only rank 0 writes
the optional NAR checkpoint/TensorBoard events.

This remains a NAST-S2x-style bootstrap generator over prepared phrase/semantic
pairs. It must be compared with the AR system for quality, RTF, and boundary
discontinuity before being selected.

