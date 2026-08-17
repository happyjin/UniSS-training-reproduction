# Stage A v5 frozen-Whisper adapter canary pass

## Decision

The v5 canary passed every strict final gate on the iteration-127 validation at
160 ms. Formal v5 Stage A is authorized. Stage B remains blocked until the
formal Stage A run completes and passes its own validation and streaming
evaluation gates.

V5 repairs the v4 geometry failure by freezing the complete released
WhisperVQ frontend and training a zero-initialized `1280 -> 128 -> 1280`
residual adapter after causal pooling. This kept the initial causal code path
identical to Phase3 while allowing short-chunk corrections outside the
immutable frontend.

## Run identity

- run ID: `stage_a_v5_adapter_canary8_20260817T024000Z`
- initialization: immutable Phase3 v4 iteration 9075
- framework/devices: Megatron, 8 x H200
- sequence length: 18000
- micro/global batch: 1 / 128
- data order: globally shuffled 15-shard coverage plan
- planned/completed updates: 127 / 127
- final validation chunk: 160 ms
- final checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_canary/stage_a_v5_adapter_canary8_20260817T024000Z/iter_0000127`
- log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_canary/stage_a_v5_adapter_canary8_20260817T024000Z/train.log`
- GPU telemetry: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_canary/stage_a_v5_adapter_canary8_20260817T024000Z/train.gpu.csv`
- TensorBoard events: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_canary/stage_a_v5_adapter_canary8_20260817T024000Z/tensorboard`
- TensorBoard service at completion: `http://10.1.6.203:6115/`
- machine-readable gate: `CANARY_GATE.json`

## Strict final gate

| Check | Requirement | Final value | Result |
|---|---:|---:|---|
| Training reached iteration 127 | yes | 127 | pass |
| Final checkpoint saved | yes | iteration 127 | pass |
| Final validation reached iteration 127 | yes | 127 | pass |
| Final validation chunk | 160 ms | 160 ms | pass |
| Complete finite metrics | yes | complete and finite | pass |
| Skipped iterations | 0 | 0 | pass |
| NaN iterations | 0 | 0 | pass |
| CTC blank ratio | <= 0.95 | 0.002069 | pass |
| Blank posterior | <= budget + 0.05 | 0.089444 <= 0.921094 | pass |
| Causal GLM agreement | >= 0.02 | 0.073299 | pass |
| Teacher code cosine | >= 0.85 | 0.924994 | pass |
| Adapter RMS | <= 0.50 | 0.155880 | pass |
| AR-ASR | < 3.0 | 0.880258 | pass |
| Source CTC | < 15.0 | 7.281893 | pass |

The gate output is `passed=true`. Its `stage_b_authorized=false` field is
intentional: a successful canary authorizes only the formal Stage A run.

## Validation trajectory

| Iteration | Validation chunk | AR-ASR | Source CTC | Identity CE | GLM agreement | Teacher cosine | Adapter RMS | CTC blank ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 960 ms | 2.321362 | 10.038230 | 7.104705 | 0.239454 | 0.970051 | 0.016454 | 0.000000 |
| 64 | 640 ms | 1.719934 | 6.983644 | 6.910399 | 0.158774 | 0.961430 | 0.082834 | 0.000000 |
| 96 | 640 ms | 0.938659 | 8.020359 | 6.771588 | 0.143515 | 0.937131 | 0.119593 | 0.001462 |
| 127 | **160 ms** | **0.880258** | **7.281893** | **6.839264** | **0.073299** | **0.924994** | **0.155880** | **0.002069** |

AR-ASR learned throughout the curriculum, while the identity and geometry
metrics degraded gradually rather than collapsing when 160-ms chunks were
introduced. The final agreement remains 3.66 times the minimum gate, teacher
cosine remains well above the geometry floor, and the adapter uses only about
31% of its allowed RMS budget.

## Comparison with failed repairs

| Run | Final inspected chunk | AR-ASR | GLM agreement | Teacher cosine | CTC blank ratio | Decision |
|---|---:|---:|---:|---:|---:|---|
| v3 anti-blank | 160 ms, iter 127 | 0.806612 | 0.009229 | 0.892115 | 0.017959 | fail: identity below 2% |
| v4 direct Whisper adaptation | 640 ms, iter 64 | 1.450472 | 0.005572 | 0.816551 | 0.001307 | stopped: geometry collapse |
| **v5 frozen Whisper + adapter** | **160 ms, iter 127** | **0.880258** | **0.073299** | **0.924994** | **0.002069** | **pass** |

V5 therefore validates the parameter-routing hypothesis: the discrete
identity objective is useful when its gradients are confined to a bounded new
adapter, but harmful when they directly rewrite the released Whisper frontend.

## Compute evidence

During the memory-resident training interval, telemetry recorded approximately:

- mean memory: 98,861 MiB/GPU;
- mean sampled utilization: 67.0%;
- mean sampled power: 344.9 W/GPU;
- median utilization: 70%;
- 90th-percentile utilization: 100%;
- 90th-percentile power: 403.9 W;
- maximum sampled power: 458.8 W.

The workload completed without reducing sequence length, global batch, GPU
count, or real model computation. Instantaneous utilization oscillation comes
from Megatron gradient accumulation, variable packed acoustic work, validation,
and checkpoint boundaries; it is not an idle placeholder workload.

## Next authorized action

Launch the isolated 381-update formal v5 Stage A run with this exact
`CANARY_GATE.json` passed as `CANARY_AUTHORIZATION`. The formal run must start
again from immutable Phase3 iteration 9075 and must not resume the canary
checkpoint. Do not start Stage B until formal Stage A has completed and passed
its separate evaluation gate.
