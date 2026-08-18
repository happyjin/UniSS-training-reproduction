# Formal gold trajectory build and audit

Status: **DATA GATE PASSED; GPU TRAINING INTENTIONALLY BLOCKED**

Run ID: `formal_gold_20260818T090515Z`

The complete 15-shard aligned source manifests were converted with 56 train
workers and 8 validation workers. Every source FLAC was both SHA256-hashed and
decoded; the independent second-pass auditor then reloaded every generated JSON
trajectory and rechecked the schema invariants.

| split | records | events | source GLM | target semantic | pre-EOS target WRITEs | output size |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,325,243 | 25,997,984 | 108,489,720 | 467,071,808 | 7,666,418 | 23,329,392,455 bytes |
| valid | 13,469 | 263,391 | 1,102,419 | 4,746,713 | 77,703 | 236,415,029 bytes |

Direction coverage is preserved:

- train: 565,268 `cmn→eng`, 759,975 `eng→cmn`;
- valid: 5,708 `cmn→eng`, 7,761 `eng→cmn`.

## Hard gates

All of the following passed on 100% of rows:

- train/validation ID overlap is zero;
- source audio exists, hashes successfully, decodes as finite mono 16 kHz PCM,
  and has the exact expected frame count;
- source event times and PCM offsets are contiguous and exact;
- gold source and target text prefixes are append-only;
- source GLM deltas concatenate exactly to the original source sequence;
- target semantic deltas have zero gap/overlap and concatenate exactly to the
  original `target_bicodec` sequence;
- non-final target content never appears before its bilingual source support;
- only the final event carries source/target final state.

## Intentional stop condition

`GOLD_TRAJECTORY_GATE.json` has `status=passed` but
`formal_training_authorized=false`. This is correct: `v1_source_delta` is still
explicitly pending, and V1/Phase3 top-k teacher caches do not yet exist. Starting
Megatron training now would silently turn the intended noisy-history E2E task
into teacher-forced gold-only training, so the pipeline fails closed.

Large immutable data is stored outside Git at:

```text
data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/
  formal_gold_20260818T090515Z/
```
