# Stage A V9 bridge-freeze canary pass report

## Decision

PASS. The 255-update V9 canary passed every strict checker condition and
authorizes one fresh 381-update V9 formal Stage A run. It does not authorize
Stage B.

The authorization artifact is `CANARY_GATE.json` in this directory. Its
schema is `uniss_stage_a_v9_bridge_freeze_canary_gate_v1`, with
`formal_v9_authorized=true` and `stage_b_authorized=false`.

## Run identity

- Run ID: `stage_a_v9_bridgefreeze128_canary8_20260817T121622Z`
- Code commits: `e107f43`, `181301e`
- Initialization: immutable Phase3 checkpoint
  `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4`
- Framework/GPU count: Megatron / 8 H200 GPUs
- Sequence length: 18,000
- Micro/global batch: 1 / 128
- Exact shuffled prefix: 32,640 samples
- Global shuffle seed: `20260816`
- Optimizer/curriculum horizon: 127 updates
- Post-curriculum hold: 128 updates
- Total: 255 updates
- Final checkpoint:
  `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_bridge_freeze_canary/stage_a_v9_bridgefreeze128_canary8_20260817T121622Z/iter_0000255`

## Strict gate result

| Check | Required | Final | Result |
|---|---:|---:|---|
| CTC blank ratio | `<= 0.25` | `0.2202124` | PASS |
| CTC blank posterior | `<= 0.25` | `0.2321676` | PASS |
| Teacher-code cosine | `>= 0.85` | `0.8990183` | PASS |
| Causal GLM agreement | `>= 0.02` | `0.09228273` | PASS |
| Code-adapter RMS | `<= 0.50` | `0.2054663` | PASS |
| Final chunk | `160 ms` | `160 ms` | PASS |
| Skipped / NaN updates | `0 / 0` | `0 / 0` | PASS |
| Final checkpoint | iteration 255 | iteration 255 | PASS |

All 20 machine-checked conditions passed; `failed_checks` is empty.

## Comparison with failed long-hold runs

| Metric | V7 formal | V8 canary | V9 canary | V9 interpretation |
|---|---:|---:|---:|---|
| Blank ratio | `0.998594` | `0.318492` | `0.220212` | V9 crosses the strict `0.25` boundary |
| Blank posterior | not controlling collapse | `0.241635` | `0.232168` | remains controlled |
| Teacher cosine | `0.832528` | `0.847486` | `0.899018` | bridge geometry retained with margin |
| GLM agreement | — | `0.092326` | `0.092283` | retained rather than traded away |
| Adapter RMS | — | `0.270982` | `0.205466` | less adapter drift |

V9's stronger differentiable blank margin fixed the residual decision-level
blank failure. Freezing the bridge after the 127-update curriculum prevented
the late hold from eroding teacher geometry while leaving the new heads and
Qwen groups trainable.

## Authorization boundary

The next permitted action is a fresh formal V9 run from immutable Phase3,
covering all 48,768 scheduled samples across 381 updates. The canary checkpoint
must not be resumed. Stage B remains blocked until that complete formal run
passes the same strict final gate.
