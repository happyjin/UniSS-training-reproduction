# Stage A v2 iteration-32 checkpoint frontend gate

## Decision

**PASS.** The trained Stage A v2 frontend and bridge can execute through the
actual 160 ms persistent-K/V runtime without changing the recomputed causal
reference, and future PCM cannot alter already committed acoustic states.

## Evaluated checkpoint and sample

- checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_smoke/stage_a_v2_teacher_smoke8_20260817T003700Z/iter_0000032`;
- sample: `NCSSD_R_EN_0000000261`;
- duration: 4,540 ms;
- runtime: FP32 eager, 160 ms PCM blocks, zero right context;
- future mutation: all PCM from 480 ms onward replaced by seeded noise.

## Results

| Check | Result | Evidence |
|---|---|---|
| Recomputed vs cached pre-VQ hidden | PASS | shape `[1,57,1280]`, max/mean absolute error `0` |
| Recomputed vs cached quantized hidden | PASS | shape `[1,57,1280]`, max/mean absolute error `0` |
| Recomputed vs cached GLM token identity | PASS | 57/57 exact |
| Recomputed vs cached trained bridge residual | PASS | shape `[1,57,896]`, max/mean absolute error `0` |
| Future-PCM recomputed hidden invariance | PASS | first 6 tokens, max absolute error `0` |
| Future-PCM cached hidden invariance | PASS | first 6 tokens, max absolute error `0` |
| Future-PCM token invariance | PASS | 6/6 exact on both execution paths |
| Cached K/V frame growth | PASS | 29 blocks, `8,16,...,232`; no unexpected reset |
| Partial final block exercised | PASS | 4,540 ms is not block aligned |

The training single-mask path differs from the strict runtime reference by at
most `1.4305e-05` because one large masked GEMM and eight-frame cached GEMMs use
different floating-point reduction orders. This is recorded as a non-gating
diagnostic. The independent block-recomputed reference and persistent cached
runtime are bit-identical in FP32 and are the deployment gate.

## Scope

This gate proves checkpoint-level frontend causality and cached execution
parity. It does not yet prove free-running ASR content quality or event-level
append-only behavior. Stage B remains blocked until the trained Qwen export,
matching-sample content evaluation, and rollback gate are complete.
