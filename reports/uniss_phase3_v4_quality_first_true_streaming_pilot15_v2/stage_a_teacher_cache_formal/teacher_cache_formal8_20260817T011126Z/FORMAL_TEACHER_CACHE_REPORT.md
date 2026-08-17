# Stage A v2 formal same-prefix teacher cache report

## Decision

**PASS.** The complete formal train and validation caches were built on eight
H200 GPUs, merged with exact pack/acoustic coverage, indexed, and read through
the real 18k dataset/collator. Formal Stage A v2 training is authorized.

## Immutable configuration

| Setting | Value |
|---|---:|
| Phase3 teacher | iteration 9075 |
| Top-k | 32 |
| Temperature | 1.5 |
| Reference must be in top-k | yes |
| Reference one-hot anchor | 0.5 |
| Speaker source | immutable Stage A pack prompt |
| Train coverage epochs | 3 |
| Validation coverage epoch | deterministic epoch 0 |
| Maximum acoustics per pack | 2 |

## Cache audit

| Split | Packs | Acoustic records | Candidate positions | Retained positions | Retention | Raw teacher top-1 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Train | 16,195 | 94,587 | 2,440,059 | 2,416,510 | 99.0349% | 38.6444% |
| Validation | 167 | 334 | 8,344 | 8,276 | 99.1850% | 39.0940% |

Train merged manifest SHA-256:
`c71c065592846fbc32e06b69f4befd9cf7ed43734a53119ff95ac25a1b97b579`.

Validation merged manifest SHA-256:
`58e1ddd836a4b9e81519fa1be737716a5de87f7d440014627e463dccdc6b6ba3`.

## Real 18k dataset/collator read

Both splits were opened with sequence length 18,000, max acoustics 2, real PCM
loading, teacher LRU bundles, and the production collator.

| Split | Token tensor | PCM tensor | Selected acoustics | Teacher positions | Top-k width | Probability-sum range |
|---|---|---|---:|---:|---:|---:|
| Train pack 0 | `[1,18000]` | `[2,143040]` | 2 | 31 | 32 | 0.999645–1.000330 |
| Validation pack 0 | `[1,18000]` | `[2,72640]` | 2 | 23 | 32 | 0.999737–1.000272 |

The small deviation from exactly 1.0 is the expected FP16 cache serialization
rounding; the objective renormalizes retained probabilities before KL.

## Formal launch geometry

The validated dry-run resolves to native Megatron with 8 processes, sequence
length 18,000, micro batch 1, global batch 128, three strict globally shuffled
coverage epochs, 381 iterations, 19 warmup iterations, max acoustics 2, and
initialization from the Phase3 native iteration-9075 checkpoint. It saves every
50 steps and evaluates every 50 steps plus the final checkpoint.
