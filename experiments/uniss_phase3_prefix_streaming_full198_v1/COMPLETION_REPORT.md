# Full198 Phase3 Prefix-Streaming Joint V3 Completion Report

## Outcome

The corrected formal run completed all 12,000 Megatron iterations on eight
H200 GPUs without a traceback, CUDA OOM, skipped iteration, or NaN iteration.
The final validation and distributed checkpoint both completed successfully.

```text
run name:       uniss_phase3_prefix_streaming_full198_joint_v3
initial model:  checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
train start:    2026-08-09 18:06:46 UTC
train end:      2026-08-09 20:21:06 UTC
elapsed:        approximately 2 h 14 min 20 s
samples:        1,536,000
final LR:       1.0e-6
NaN / skipped:  0 / 0
```

Artifacts are isolated from every historical run:

```text
checkpoint:  checkpoints/uniss_phase3_prefix_streaming_full198_joint_v3
final step:  checkpoints/uniss_phase3_prefix_streaming_full198_joint_v3/iter_0012000
train log:   logs/uniss_phase3_prefix_streaming_full198_joint_v3.log
GPU log:     logs/uniss_phase3_prefix_streaming_full198_joint_v3_gpu_power_utility.csv
TensorBoard: runs/uniss_phase3_prefix_streaming_full198_joint_v3/tensorboard
```

TensorBoard remains available on port 6066.  On the completion host it was
served at `http://10.1.6.203:6066/`.

## Correctness fix validated by the run

Megatron advances `args.curr_iteration` inside its training loop, while
`args.iteration` remains the checkpoint/start iteration.  The stopped v2 run
read the latter and consequently stayed at the first curriculum point.
Commit `5b3ed30` changed the experiment to use the live loop iteration and
added regression tests.

The real v3 run verified every curriculum transition:

| Logged iterations | Intended replay/prefix/semantic/commit mix | Observed behavior |
|---:|---:|---|
| 1--1500 | 40/50/10/0 | commit and action losses were zero |
| 1501--4000 | 30/50/15/5 | iteration 1510 was 30/52/13/4 with nonzero commit/action losses |
| 4001--7000 | 30/30/30/10 | iteration 4020 was 30/30/30/10; shorter 0.40 prefix was active |
| 7001--10000 | 30/25/25/20 | iteration 7050 was 29/24/25/22; 0.25 prefix and stronger WRITE supervision were active |
| 10001--12000 | 35/20/20/25 | iteration 12000 was 36/18/20/26 |

Every logged optimizer step retained a 50/50 EN-source/ZH-source direction
mix.  LoRA update RMS increased from zero at initialization to 0.006005 at the
end, confirming that the intended trainable parameters changed.

## Validation checkpoints

The table reports the deterministic balanced UniST dev evaluation.  Stage
boundaries can cause temporary changes because progressively shorter prefixes
and stronger commit supervision make the task harder.

| Iteration | Replay CE | Prefix CE | Semantic CE | Commit CE | Teacher KL | Adjacent | Action CE | Boundary |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 3.8155 | 2.3673 | 5.6200 | 0.2059 | 0.3922 | 0.1586 | 4.1073 | 4.2107 |
| 1500 | 3.8156 | 1.9269 | 5.1866 | 0.1831 | 0.2855 | 0.1292 | 4.2971 | 1.6167 |
| 2000 | 3.8195 | 1.9240 | 4.9036 | 0.2057 | 0.2692 | 0.1199 | 1.1098 | 1.0723 |
| 4000 | 3.8030 | 1.8103 | 4.9152 | 0.1161 | 0.2714 | 0.1206 | 1.0547 | 0.3145 |
| 4250 | 3.7970 | 1.8552 | 4.8845 | 0.1429 | 0.2603 | 0.1133 | 0.9627 | 0.3478 |
| 7000 | 3.8056 | 1.8095 | 4.7744 | 0.1620 | 0.2557 | 0.1127 | 0.5990 | 0.2637 |
| 7250 | 3.8060 | 1.7678 | 4.9104 | 0.1455 | 0.2555 | 0.1101 | 0.5777 | 0.1789 |
| 10000 | 3.7875 | 1.8760 | 4.7573 | 0.1410 | 0.2550 | 0.1086 | 0.5703 | 0.2072 |
| 10250 | 3.7971 | 1.8031 | 4.8178 | 0.1340 | 0.2510 | 0.1087 | 0.5635 | 0.2408 |
| 11500 | 3.7956 | 1.8561 | 4.7853 | 0.1340 | 0.2479 | 0.1059 | 0.5859 | 0.2306 |
| 12000 | 3.8206 | 1.9053 | 4.7393 | 0.2255 | 0.2517 | 0.1134 | 0.5669 | 0.2332 |

Training completion alone does not establish the best inference checkpoint.
The final checkpoint has the lowest listed semantic CE, while iteration 10000
has the lowest replay CE and iteration 11500 has the best listed teacher KL
and adjacent-prefix consistency.  Downstream simultaneous S2ST evaluation
should therefore compare at least iterations 10000, 11500, and 12000 instead
of selecting the last checkpoint solely by iteration number.

## GPU utilization and safety

Across GPU-log samples with allocated training memory:

```text
average GPU utility: 39.9%
peak GPU utility:    100%
average power:       202.1 W
peak power:          384.1 W
peak memory:         90,152 MiB per sampled GPU
```

This workload trains LoRA parameters on a 0.52B model and uses dynamically
bounded samples, so utilization is bursty.  An earlier micro-batch-16 stress
test reached roughly 140.5 GiB on a 143.8 GiB H200.  The formal micro-batch-8
configuration deliberately retained memory headroom for full198 outliers and
completed safely.  Forcing sustained 700 W would have increased OOM risk
without establishing a quality benefit.

## Version control

The isolated experiment implementation, data guards, live curriculum fix, and
run-lineage documentation were committed without staging unrelated workspace
changes.  The final relevant commits are:

```text
8d47c03 feat: add isolated full198 streaming curriculum data path
a44b355 feat: add Megatron full198 multi-view joint trainer
ec2cd89 fix: support full198 parquet reads in curriculum dataset
bf046b5 feat: add safe launch and monitoring for full198 streaming run
2a2b75b fix: stabilize full198 Megatron validation startup
9ebabaf fix: balance and sanitize full198 streaming data
af31f3d fix: harden full198 streaming outlier handling
5b3ed30 fix: advance full198 streaming curriculum with Megatron step
7bf76db docs: record full198 streaming run lineage
```

At completion, local `master` and `private/main` were synchronized.
