# Stage A v6 independent-horizon hold-canary pass

## Decision

The v6 hold-canary passed every strict gate on the final iteration-127
validation at 160 ms. A new isolated v6 formal Stage A run is authorized.
Stage B remains blocked until formal Stage A completes and passes its separate
quality and streaming gates.

V6 changes only curriculum time. The v5 objective, frozen released WhisperVQ
frontend, zero-initialized residual adapter, loss weights, Megatron topology,
training packs, teacher cache, and immutable Phase3 initialization are
unchanged. Unlike the failed v5 formal run, v6 measures curriculum progress
against an explicit horizon rather than against total training iterations.

The hold-canary deliberately uses a 42-update curriculum horizon inside a
127-update run. It therefore reaches the 320/160-ms target regime at iteration
43 and spends the remaining 85 updates testing whether short-chunk training is
stable after the curriculum has saturated. This is a stricter stress test of
the repaired scheduling semantics than simply repeating the original v5
canary.

## Run identity

- run ID: `stage_a_v6_horizon42_hold_canary8_20260817T033200Z`
- initialization: immutable Phase3 v4 iteration 9075
- framework/devices: Megatron, 8 x H200
- sequence length: 18000
- micro/global batch: 1 / 128
- data order: globally shuffled 15-shard coverage plan
- planned/completed updates: 127 / 127
- independent curriculum horizon: 42 updates
- saturated short-chunk interval: iterations 43-127
- final validation chunk: 160 ms
- final checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_hold_canary/stage_a_v6_horizon42_hold_canary8_20260817T033200Z/iter_0000127`
- log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_hold_canary/stage_a_v6_horizon42_hold_canary8_20260817T033200Z/train.log`
- GPU telemetry: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_hold_canary/stage_a_v6_horizon42_hold_canary8_20260817T033200Z/train.gpu.csv`
- TensorBoard events: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_hold_canary/stage_a_v6_horizon42_hold_canary8_20260817T033200Z/tensorboard`
- TensorBoard service at completion: `http://10.1.6.203:6117/`
- machine-readable gate: `CANARY_GATE.json`

## Strict final gate

| Check | Requirement | Final value | Result |
|---|---:|---:|---|
| Training reached iteration 127 | yes | 127 | pass |
| Final checkpoint saved | yes | iteration 127 | pass |
| Final validation reached iteration 127 | yes | 127 | pass |
| Final validation chunk | 160 ms | 160 ms | pass |
| Effective curriculum progress | exactly 1.0 | 1.0 | pass |
| Complete finite metrics | yes | complete and finite | pass |
| Skipped iterations | 0 | 0 | pass |
| NaN iterations | 0 | 0 | pass |
| CTC blank ratio | <= 0.25 | 0.002069 | pass |
| Blank posterior | <= budget + 0.05 | 0.089301 <= 0.921094 | pass |
| Causal GLM agreement | >= 0.02 | 0.075721 | pass |
| Teacher code cosine | >= 0.85 | 0.924229 | pass |
| Adapter RMS | <= 0.50 | 0.157726 | pass |
| AR-ASR | < 3.0 | 0.752788 | pass |
| Source CTC | < 15.0 | 7.308588 | pass |

The checker returned `passed=true` and all 17 machine-readable checks are
true. Its `stage_b_authorized=false` field is intentional: this gate
authorizes only v6 formal Stage A.

## Validation trajectory

| Iteration | Curriculum progress | Validation chunk | AR-ASR | Source CTC | Identity CE | GLM agreement | Teacher cosine | Adapter RMS | CTC blank ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.761905 | 160 ms | 2.307890 | 10.055480 | 7.171108 | 0.126877 | 0.955560 | 0.016680 | 0.000000 |
| 64 | 1.000000 | 320 ms | 1.469644 | 7.095201 | 6.953024 | 0.126759 | 0.951861 | 0.075204 | 0.000000 |
| 96 | 1.000000 | 320 ms | 0.755356 | 8.218425 | 6.836248 | 0.121268 | 0.932070 | 0.117478 | 0.001254 |
| 127 | **1.000000** | **160 ms** | **0.752788** | **7.308588** | **6.834860** | **0.075721** | **0.924229** | **0.157726** | **0.002069** |

The curriculum was already saturated at both the iteration-64 and
iteration-96 validations. AR-ASR continued to improve, source CTC remained
well inside its learning gate, and the hard blank ratio stayed three orders of
magnitude below the v5 formal collapse. Identity and teacher geometry degraded
gradually but remained bounded after 85 consecutive updates in the target
short-chunk regime.

## Comparison with the v5 controls

| Run / point | Effective curriculum | Chunk | AR-ASR | GLM agreement | Teacher cosine | Adapter RMS | CTC blank ratio | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v5 canary final, iter 127 | 1.000 | 160 ms | 0.880258 | 0.073299 | 0.924994 | 0.155880 | 0.002069 | pass |
| v5 formal train, iter 115 | 0.299 | 960 ms | 0.652247 | 0.174016 | **0.811503** | 0.301900 | **0.999782** | stopped |
| **v6 hold-canary final, iter 127** | **1.000** | **160 ms** | **0.752788** | **0.075721** | **0.924229** | **0.157726** | **0.002069** | **pass** |

V6 closely reproduces the successful v5 endpoint while remaining in the
short-chunk regime for substantially longer. At the same approximate amount
of optimization where v5 formal had collapsed to almost all blank predictions,
v6 retains the Phase3 teacher geometry and non-blank causal code identity.
This isolates curriculum stretching, rather than total update count or the v5
objective itself, as the cause of the failed formal attempt.

## Compute evidence

During the memory-resident training interval, telemetry recorded
approximately:

- mean memory: 98,870 MiB/GPU;
- mean sampled utilization: 66.1%;
- mean sampled power: 347.9 W/GPU;
- median utilization: 67%;
- 90th-percentile utilization: 100%;
- 90th-percentile power: 401.4 W;
- maximum sampled power: 456.2 W.

The run retained 8 real GPUs, sequence length 18000, global batch 128, the
complete 0.52B active model path, global shuffle, validation, and checkpoint
work. Instantaneous utilization varies with Megatron accumulation, packed
acoustic length, evaluation, and distributed checkpoint boundaries.

## Next authorized action

Launch a new isolated 381-update v6 formal Stage A run using this exact
`CANARY_GATE.json` as `CANARY_AUTHORIZATION`. The formal run must:

1. start again from immutable Phase3 v4 iteration 9075, not from the canary;
2. use three globally shuffled coverage epochs and 381 total updates;
3. use an independent 127-update curriculum horizon;
4. complete the proven curriculum during epoch one;
5. remain in the 320/160-ms regime for epochs two and three;
6. stop if hard blank ratio exceeds 0.95 or teacher cosine falls below 0.85.

Stage B remains explicitly unauthorized.
