# full198 MBS=2 smoke validation

Validated on 2026-08-07 with 8 x NVIDIA H200 (144 GB each).

| Stage | Iterations | Rank-0 peak allocated | Max train commitment | NaN / skipped / OOM | Checkpoint |
|---|---:|---:|---:|---|---|
| heads-only | 4 | 72,306.85 MB | 0.01516 | 0 / 0 / no | saved |
| guarded joint | 16 | 112,661.53 MB | 0.02152 | 0 / 0 / no | saved |

During guarded joint compute, an external `nvidia-smi` snapshot showed
approximately 123--126 GB used per GPU, 89--100% GPU utilization, and
484--583 W power.  The short smoke did not encounter the worst-case variable
length batch: the formal Stage B MBS=2 run later needed an additional 24.21
GiB after iteration 10 and OOMed.  The formal Stage B continuation therefore
uses the previously stable MBS=1 with GBS=128; Stage A retains MBS=2.

The Stage B smoke loaded the full198 Stage A checkpoint as model-only state and
started at iteration zero.  The final validation commitment was 0.01843, well
below the V6 absolute stop gate of 0.10.  No old checkpoint, log, TensorBoard
directory, script, or experiment output was overwritten.
