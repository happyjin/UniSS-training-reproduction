# Stage05 real-CTC policy evaluation

This report feeds actual causal Stage03b ASR/NAR-S2TT logits into the isolated
monotonic Stage05 policy. It measures the policy/CTC proxy only; Phase3 text and
BiCodec audio generation are deliberately not claimed here.

| Metric | Value |
| --- | ---: |
| Samples | 1 |
| Samples with a WRITE | 1.0000 |
| First WRITE mean (model-frame ms) | 2160.0000 |
| First WRITE p50 (model-frame ms) | 2160.0000 |
| First WRITE p95 (model-frame ms) | 2160.0000 |
| Committed target unigram recall | 0.1579 |
| Source/target conflict events | 0 / 0 |
| Rollback events | 0 |

| Direction | Samples | WRITE coverage | First WRITE p50 ms | Recall |
| --- | ---: | ---: | ---: | ---: |
| EN→ZH | 0 | 0.0000 | n/a | n/a |
| ZH→EN | 1 | 1.0000 | 2160.0000 | 0.1579 |

`model-frame ms` uses the 40 ms encoder frame clock and includes the configured
right-context frames. It excludes wall-clock compute and downstream synthesis.
