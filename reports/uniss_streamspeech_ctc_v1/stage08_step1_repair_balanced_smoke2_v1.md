# Stage08 Step1-R balanced repair smoke

Date: 2026-08-04 UTC

The two-iteration eight-H200 smoke completed successfully with DP=8, TP=PP=1,
micro batch 1, global batch 128 and 16 gradient-accumulation micro-batches.

## Initialization and objective

- restored all 157 trainable tensors / 116,454,660 parameters from the original
  Step1 iteration 800 checkpoint;
- exact observed EN→ZH / ZH→EN fractions: 0.5 / 0.5 in train and validation;
- loss weights: ASR 4, NAR-S2TT 4, AR-S2TT 8, frozen Phase3 NLL 2,
  residual MSE 1e-4 and ZH→EN sample weight 1.25;
- Qwen, BiCodec and the Stage04 bridge remained frozen.

## Smoke evidence

| Metric | Iteration 1 | Iteration 2 | Final validation |
|---|---:|---:|---:|
| joint multitask | 55.50076 | 55.21736 | 54.99857 |
| ASR CTC | 1.555924 | 1.602798 | 1.449436 |
| NAR-S2TT CTC | 4.687539 | 4.702949 | 4.856915 |
| AR-S2TT CE | 3.815863 | 3.749297 | 3.721645 |
| Phase3 NLL | 4.150585 | 4.126327 | 4.110051 |
| EN→ZH fraction | 0.5 | 0.5 | 0.5 |
| ZH→EN fraction | 0.5 | 0.5 | 0.5 |
| gradient norm | 16.676 | 17.538 | n/a |

Both iterations reported zero skipped iterations and zero NaN iterations.
Validation and torch-dist checkpoint saving succeeded at iterations 1 and 2.

Runtime artifacts are isolated under:

```text
checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_smoke2_v1/
runs/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_smoke2_v1/
logs/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_smoke2_v1.log
```
