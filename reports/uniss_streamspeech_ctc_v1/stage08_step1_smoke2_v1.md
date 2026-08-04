# Stage08 Step1 two-iteration Megatron smoke

Date: 2026-08-04 UTC

## Result

The isolated frozen-Qwen Step1 path completed successfully on eight H200 GPUs.
Megatron reported DP=8, TP=PP=1, global batch 128 and consumed exactly 128
samples per iteration, confirming 16 micro-batch-one accumulation steps per
optimizer update.

All train and validation components were finite. Megatron reported zero skipped
iterations and zero NaN iterations. Validation and torch-dist checkpoint saves
succeeded at both iterations.

| Metric | Train iter 1 | Train iter 2 | Final validation |
|---|---:|---:|---:|
| joint multitask | 51.21619 | 52.82292 | 54.11036 |
| ASR CTC | 1.513582 | 1.562595 | 1.422346 |
| NAR-S2TT CTC | 4.518938 | 4.575142 | 4.730384 |
| AR-S2TT CE | 3.385764 | 3.533996 | 3.687430 |
| frozen Phase3 NLL | 4.053855 | 4.064550 | 4.093561 |
| B1 residual RMS | 0.01204206 | 0.01204454 | 0.01234043 |
| gradient norm | 14.764 | 15.343 | n/a |

## Artifacts

```text
checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_smoke2_v1/
runs/uniss_streamspeech_ctc_v1/stage08_step1_smoke2_v1/
logs/uniss_streamspeech_ctc_v1/stage08_step1_smoke2_v1.log
```

The smoke artifacts are intentionally separate from both Stage03--07 and the
formal Stage08 run. They are runtime evidence and are not committed to Git.
