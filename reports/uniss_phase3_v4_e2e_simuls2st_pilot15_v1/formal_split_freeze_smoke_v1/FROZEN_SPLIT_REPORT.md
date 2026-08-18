# Formal 15-shard split freeze audit

Status: **PASSED**

The new E2E pipeline independently re-read and SHA256-hashed the complete
Stage-A formal source assets before any large trajectory generation. It also
loaded the sorted 64-bit ID digests and recomputed the train/validation
intersection.

| split | records | duration | `cmn→eng` | `eng→cmn` | manifest SHA256 |
|---|---:|---:|---:|---:|---|
| train | 1,325,243 | 2,393.25 h | 565,268 | 759,975 | `cc2c850e20f5d5b346be5fd44b8dfa98093fd44be1f99d0276e57cc28a3716f3` |
| valid | 13,469 | 24.32 h | 5,708 | 7,761 | `2981204ad0db9ca08bbc5ec206203ed1c8f7a410ad7a211cd632661e14d91483` |

Additional immutable identities:

| asset | SHA256 |
|---|---|
| train offset index | `ba8389c194ae8ff8783635dc6f08a2c15ba7d5e6615b143adc9394714b10013f` |
| valid offset index | `24098ce527176e34ca08f489a90e572fa0aaa96fd47a53916f1573cf0593d0d9` |
| train sorted ID hashes | `83ce0449138b1bfeaa4b12bce6791d53db1540ace96241fa17501bc12309c151` |
| valid sorted ID hashes | `54c64dd3bdb92196387744ada4f13b892dc7bba54e1c3f8ca468daaf6eba42b9` |
| Stage-A data audit | `fab404612a88154aabc736be5d85008bbe90370314db6cce6b2f01d6f55a074f` |

The recomputed train/validation ID overlap is exactly zero. The formal builder
therefore has authorization to create gold trajectories, but GPU training is
still blocked until real V1 free-running rollouts and teacher caches pass their
later gates.
