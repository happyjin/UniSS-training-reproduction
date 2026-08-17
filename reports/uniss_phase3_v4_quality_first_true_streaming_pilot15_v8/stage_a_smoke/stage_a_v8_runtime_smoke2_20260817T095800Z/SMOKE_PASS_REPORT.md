# Stage A V8 two-update runtime smoke

## Decision

**PASS.** The isolated V8 objective completed real 8-GPU Megatron forward,
backward, optimizer, validation, and distributed checkpoint paths.

- Run ID: `stage_a_v8_runtime_smoke2_20260817T095800Z`
- Initialization: immutable Phase3 checkpoint
  `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4`
- Updates: 2/2
- GPUs: 8
- Sequence length: 18000
- Micro/global batch: 1/128
- NaN / skipped updates: 0 / 0
- Final checkpoint: saved at iteration 2
- Checkpoint root:
  `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v8/stage_a_smoke/stage_a_v8_runtime_smoke2_20260817T095800Z`
- Log:
  `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v8/stage_a_smoke/stage_a_v8_runtime_smoke2_20260817T095800Z/train.log`

## Runtime evidence

The second training update reported:

| Metric | Value |
|---|---:|
| AR-ASR | 8.443466 |
| Source CTC | 17.447650 |
| CTC blank budget | 0.000000 |
| CTC blank ratio | 0.000000 |
| CTC blank posterior | 0.000485 |
| Causal GLM agreement | 0.204859 |
| Teacher-code cosine | 0.968664 |
| Adapter RMS | 0.000000 |
| NaN / skipped | 0 / 0 |

The final validation also remained finite, with blank ratio 0, blank
posterior approximately 0.000544, teacher cosine 0.956141, and adapter RMS
0.000550. The logged seed strength is approximately 0.10 at saturated smoke
progress, confirming that the V8 non-blank floor is active rather than
annealing to zero.

This two-update run is a runtime integration check, not a model-quality gate.
It authorizes launching the 255-update long-hold canary, which must cross the
V7 failure boundary before any V8 formal run can start. It does not authorize
Stage B.
