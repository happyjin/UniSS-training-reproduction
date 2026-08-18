# Stage05 real-CTC policy evaluation

This report feeds actual causal Stage03b ASR/NAR-S2TT logits into the isolated
monotonic Stage05 policy. It measures the policy/CTC proxy only; Phase3 text and
BiCodec audio generation are deliberately not claimed here.

| Metric | Value |
| --- | ---: |
| Samples | 1 |
| Samples with a WRITE | 1.0000 |
| First WRITE mean (model-frame ms) | 560.0 |
| First WRITE p50 (model-frame ms) | 560.0 |
| First WRITE p95 (model-frame ms) | 560.0 |
| Committed target unigram recall | 0.2353 |
| Source/target conflict events | 0 / 0 |
| Rollback events | 0 |

`model-frame ms` uses the 40 ms encoder frame clock and includes the configured
right-context frames. It excludes wall-clock compute and downstream synthesis.
