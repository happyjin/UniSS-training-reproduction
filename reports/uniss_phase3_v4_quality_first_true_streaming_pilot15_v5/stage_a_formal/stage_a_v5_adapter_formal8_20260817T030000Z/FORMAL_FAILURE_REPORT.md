# Stage A v5 formal curriculum-horizon failure

## Decision

The v5 formal run was deliberately stopped after iteration 115. The valid
iteration-100 checkpoint is retained as failure evidence, but it must not be
used to resume training. Formal Stage A and Stage B remain blocked.

The v5 canary itself remains a valid one-epoch success. The failure was caused
by scaling the canary curriculum over all 381 formal updates instead of
finishing that curriculum during the first 127-update coverage epoch.

## Run identity

- run ID: `stage_a_v5_adapter_formal8_20260817T030000Z`
- initialization: immutable Phase3 v4 iteration 9075
- framework/devices: Megatron, 8 x H200
- sequence length: 18000
- micro/global batch: 1 / 128
- globally shuffled source packs: 16195
- configured coverage: 3 epochs, 381 updates
- stopped after: iteration 115
- retained evidence checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_formal/stage_a_v5_adapter_formal8_20260817T030000Z/iter_0000100`
- log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_formal/stage_a_v5_adapter_formal8_20260817T030000Z/train.log`
- GPU telemetry: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_formal/stage_a_v5_adapter_formal8_20260817T030000Z/train.gpu.csv`
- TensorBoard events: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v5/stage_a_formal/stage_a_v5_adapter_formal8_20260817T030000Z/tensorboard`
- TensorBoard service at failure: `http://10.1.6.203:6116/`

## Evidence

| Point | Curriculum progress | Chunk | AR-ASR | CTC blank ratio | Blank posterior | GLM agreement | Teacher cosine | Adapter RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation 50 | 0.131 | 1280 ms | 2.203718 | 0.000000 | 0.014429 | 0.252578 | 0.971831 | 0.038238 |
| validation 100 | 0.262 | 960 ms | 0.744420 | **0.749355** | 0.303921 | 0.162316 | 0.876530 | 0.262860 |
| train 107 | 0.278 | 960 ms | 0.753424 | **0.990306** | 0.447186 | 0.164514 | **0.845398** | 0.273022 |
| train 112 | 0.291 | 640 ms | 0.664230 | **0.999248** | 0.565808 | 0.157776 | **0.824742** | 0.288915 |
| train 115 | 0.299 | 960 ms | 0.652247 | **0.999782** | 0.622683 | 0.174016 | **0.811503** | 0.301900 |

The stop conditions were crossed repeatedly:

- CTC blank ratio exceeded the `0.95` anti-collapse ceiling;
- teacher-code cosine fell below the `0.85` geometry floor;
- the failure persisted across alternating 960-ms and 640-ms batches;
- no NaN or skipped iteration occurred, so this was semantic collapse rather
  than numerical instability.

## Root cause

The successful v5 canary used 127 updates and reached the target curriculum by
the end of one globally shuffled coverage epoch. Its final validation was at
160 ms with a CTC blank ratio of `0.002069` and teacher cosine of `0.924994`.

The formal launcher changed both the training horizon and the curriculum
horizon to 381 updates. Therefore, after almost one complete epoch of data at
iteration 100, the formal run had advanced through only 26% of the curriculum
and was still dominated by 1280/960-ms chunks. The same amount of model
optimization that took the canary into 320/160-ms supervision instead
over-trained the easy long-chunk regime. CTC converged toward blank-dominated
argmax predictions while the adapter moved progressively away from the
released Phase3 teacher-code geometry.

The existing blank-posterior budget did not prevent this failure because it
constrains the mean blank probability against a high length-derived budget.
Hard argmax blank ratio can approach one while mean blank posterior remains
below that budget. The monotonic seed was also decaying with the stretched
global progress and could not reverse the established blank preference.

## Required isolated v6 repair

V6 must start again from immutable Phase3 iteration 9075 and must not resume
the failed checkpoint. It must:

1. introduce an explicit curriculum horizon independent of `train_iters`;
2. set the horizon to 127 updates for the 381-update formal run;
3. reproduce the proven v5 canary schedule during the first coverage epoch;
4. clamp effective curriculum progress at 1.0 afterward, retaining alternating
   320/160-ms target training for epochs two and three;
5. use the same effective progress for chunk selection, CTC seed scheduling,
   and parameter-group curriculum gates;
6. add tests proving that update 127 reaches progress 1.0 and updates 128-381
   remain at target short chunks;
7. run a bounded v6 continuation canary before authorizing another full formal
   attempt.

Stage B remains explicitly unauthorized.
