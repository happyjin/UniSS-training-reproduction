# Stage A V7 post-decay canary pass report

## Verdict

- Run: `stage_a_v7_post_decay64_canary8_20260817T055122Z`
- Result: **PASS**
- Gate schema: `uniss_stage_a_v7_post_decay_canary_gate_v1`
- Training updates: **191/191**
- Optimizer/curriculum horizon: **127 updates**
- Post-decay LR-floor hold: **64 updates**
- Final validation chunk: **160 ms**
- NaN/skipped iterations: **0 / 0**
- Formal Stage A: **authorized**
- Stage B: **not authorized**

The run started from the immutable Phase3 v4 checkpoint at iteration 9075.
It did not resume either V6 failure evidence or a V7 canary checkpoint. The
canary used eight GPUs, Megatron, sequence length 18,000, micro-batch 1,
global batch 128, and a deterministic prefix of the formal three-epoch global
shuffle schedule.

## Final strict-gate metrics

| Metric | Final value | Gate | Result |
|---|---:|---:|---|
| AR-ASR loss | 0.577423 | < 3.0 | pass |
| source CTC loss | 6.273035 | < 15.0 | pass |
| CTC blank ratio | 0.124283 | <= 0.25 | pass |
| blank posterior | 0.200605 | <= target + 0.05 | pass |
| blank budget target | 0.863281 | reference | pass |
| causal GLM agreement | 0.096475 | >= 0.02 | pass |
| teacher-code cosine | 0.927539 | >= 0.85 | pass |
| adapter RMS | 0.158721 | <= 0.50 | pass |
| curriculum progress | 1.0 | exactly 1.0 | pass |
| curriculum chunk | 160 ms | exactly 160 ms | pass |

All 17 machine checks in `CANARY_GATE.json` passed, including the final
checkpoint, complete metric set, finite values, exact 64-update post-decay
hold, and zero NaN/skipped iterations.

## Comparison with the successful V6 hold-canary

| Metric | V6 iteration 127 | V7 iteration 191 | Direction |
|---|---:|---:|---|
| AR-ASR loss | 0.752788 | 0.577423 | better |
| source CTC loss | 7.308588 | 6.273035 | better |
| causal GLM agreement | 0.075721 | 0.096475 | better |
| teacher-code cosine | 0.924229 | 0.927539 | slightly better |
| adapter RMS | 0.157726 | 0.158721 | essentially unchanged |
| CTC blank ratio | 0.002069 | 0.124283 | higher, but inside gate |

V7 fixes the optimizer-clock defect that caused V6 formal to collapse around
updates 104-108. V7 crossed that region with blank ratio below 0.01 and
teacher cosine near 0.94. At update 128, when the optimizer reached its floor,
validation blank ratio was 0.005437 and teacher cosine was 0.938302. After 64
additional floor updates, blank ratio rose to 0.124283 while teacher cosine
remained 0.927539. This passes the designed stress gate but is a trend that
must be monitored during formal training; formal should be stopped if blank
ratio exceeds 0.95 or teacher cosine falls below 0.85.

## Runtime evidence

- Training window: approximately 05:52-06:18 UTC on 2026-08-17.
- Checkpoint footprint: approximately 67 GiB.
- Per-GPU active-sample average utilization: approximately 66%-71%.
- Per-GPU observed maximum utilization: 99%.
- Per-GPU active-sample average power: approximately 332-366 W.
- Per-GPU observed maximum power: approximately 422-461 W.
- Active memory was approximately 95-99 GiB per GPU on average.

The power profile is below the long-sequence Phase3 peak because the causal
curriculum intentionally moves from 1280 ms to 160/320 ms chunks, reducing
per-update arithmetic while retaining the 18,000-token packed Megatron batch.

## Artifacts

- Gate: `CANARY_GATE.json`
- Training log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v7/stage_a_post_decay_canary/stage_a_v7_post_decay64_canary8_20260817T055122Z/train.log`
- Checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v7/stage_a_post_decay_canary/stage_a_v7_post_decay64_canary8_20260817T055122Z/iter_0000191`
- TensorBoard: `http://10.1.6.203:6119/`

Formal training must start again from the immutable Phase3 v4 checkpoint, not
from this canary checkpoint.
