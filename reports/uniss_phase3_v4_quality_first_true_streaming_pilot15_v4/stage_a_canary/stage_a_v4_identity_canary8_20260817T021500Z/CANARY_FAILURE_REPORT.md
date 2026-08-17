# Stage A v4 direct-identity canary early failure

## Decision

The v4 canary was deliberately stopped after the iteration-64 checkpoint and
validation completed.  Formal Stage A and Stage B remain blocked.

V4 tested the hypothesis that full 16,384-way codebook cross-entropy applied
directly through the Whisper top layers would preserve discrete teacher-code
identity.  The experiment falsified that hypothesis: the loss decreased, but
the original WhisperVQ geometry degraded faster than the classifier objective
could restore it.

## Run identity

- run ID: `stage_a_v4_identity_canary8_20260817T021500Z`
- initialization: immutable Phase3 v4 iteration 9075
- framework/devices: Megatron, 8 x H200
- sequence length: 18000
- micro/global batch: 1 / 128
- globally shuffled coverage plan: one epoch, 127 planned updates
- stopped after: iteration 65, after iteration-64 checkpoint and validation
- saved evidence checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v4/stage_a_canary/stage_a_v4_identity_canary8_20260817T021500Z/iter_0000064`
- log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v4/stage_a_canary/stage_a_v4_identity_canary8_20260817T021500Z/train.log`
- TensorBoard events: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v4/stage_a_canary/stage_a_v4_identity_canary8_20260817T021500Z/tensorboard`

## Validation evidence

| Iteration | Validation chunk | AR-ASR | Source CTC | Identity CE | GLM agreement | Teacher cosine | Teacher margin | CTC blank ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 960 ms | 2.315723 | 9.903848 | 6.847233 | 0.046099 | 0.934799 | -0.034163 | 0.000000 |
| 64 | 640 ms | 1.450472 | 6.408794 | 6.797567 | **0.005572** | **0.816551** | **-0.077005** | 0.001307 |

The run had zero skipped and zero NaN iterations.  CTC remained healthy, so
this is not a recurrence of the v2 blank-collapse failure.

## Diagnosis

The v4 cross-entropy is mathematically aligned with top-1 code identity, but
its trainable path was the pre-existing Whisper top stack.  Updating that stack
changes every pooled code vector, including vectors that were already close to
the released codebook.  Evidence is internally consistent:

- identity CE improved only slightly: `6.847 -> 6.798`;
- teacher cosine fell sharply: `0.935 -> 0.817`;
- teacher margin became more negative: `-0.034 -> -0.077`;
- exact agreement collapsed: `4.61% -> 0.56%`.

The problem is therefore parameter routing, not lack of a discrete objective.
Directly adapting Whisper trades away the Phase3/WhisperVQ geometry that the
new objective is trying to preserve.

## Required v5 repair

V5 must start again from immutable Phase3 iteration 9075, not resume v4.  It
must:

1. freeze all Whisper encoder and convolution parameters;
2. insert a zero-initialized residual causal-code adapter after causal pooling
   and before nearest-code quantization;
3. apply MSE commitment and full-codebook CE to the adapted code vectors;
4. keep v3 CTC anti-collapse terms;
5. retain early repeated 160-ms exposure and require final 160-ms validation;
6. regularize and report adapter residual magnitude.

Zero initialization makes the v5 initial code path exactly equal to the
released Phase3/WhisperVQ path.  Only the new adapter can change code geometry,
so failed short-chunk corrections cannot silently corrupt the acoustic
frontend.

