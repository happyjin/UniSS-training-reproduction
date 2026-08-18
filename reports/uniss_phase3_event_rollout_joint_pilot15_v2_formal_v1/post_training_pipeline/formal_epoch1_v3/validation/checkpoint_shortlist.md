# Fixed15 exact-runtime checkpoint shortlist

- Final checkpoint: `not_selected`
- This table is a probe shortlist, not a best-checkpoint claim.

| order | iteration | rank sum | trajectory loss | text acc | action acc | EOS recall | safe-commit F1 | frontend RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 700 | 13 | 5.760265 | 0.355642 | 0.759314 | 0.785277 | 0.605338 | 0.042882 |
| 2 | 650 | 16 | 5.768689 | 0.355041 | 0.759153 | 0.792954 | 0.601483 | 0.042761 |
| 3 | 600 | 23 | 5.800465 | 0.350887 | 0.757765 | 0.792423 | 0.600400 | 0.042607 |

Teacher-forced rank sum only creates a probe shortlist. Final selection requires natural exact-runtime WRITE, target-language useful-audio ASR, p50/p90/p95 latency, valid PCM, EOS, collapse, quality metrics, runtime parity and Phase3 replay retention.
