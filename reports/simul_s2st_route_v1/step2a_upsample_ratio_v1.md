# Step 2a — sizing the NAR CTC upsample ratio

> Run `step2a_upsample_ratio_v1` · 2026-08-09T03:40:27+0000 · research only.

399,987 target utterances from joint_train.jsonl, joint_valid.jsonl.

## 1. What the corpus needs

| Direction | Rows | Text tokens p50/p95/max | BiCodec tokens p50/p95/max | Required frames per text token p50/p95/p99/max |
|---|---:|---|---|---|
| overall | 399,987 | 14 / 47 / 133 | 261 / 807 / 3000 | 18.4 / 37.5 / 65.3 / 2997.0 |
| eng->cmn | 167,859 | 19 / 51 / 119 | 314 / 854 / 1720 | 16.6 / 25.3 / 32.3 / 217.0 |
| cmn->eng | 232,128 | 11 / 43 / 133 | 239 / 747 / 3000 | 20.7 / 43.7 / 82.0 / 2997.0 |

## 2. Feasibility and cost per candidate ratio

`relative attention cost` is the mean of `(ratio x text_length)^2` divided by the same quantity at the currently shipped ratio 48, i.e. the unit decoder's self-attention work.

| Ratio | Feasible (all) | Feasible (healthy) | Infeasible | Mean frames | p99 frames | Max frames | Lattice occupancy | Relative attention cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.190% | 0.191% | 399,226 | 146 | 528 | 1,064 | 85.7% | 0.028x |
| 12 | 5.715% | 5.736% | 377,129 | 219 | 792 | 1,596 | 88.9% | 0.062x |
| 16 | 31.763% | 31.879% | 272,940 | 291 | 1056 | 2,128 | 85.2% | 0.111x |
| 20 | 60.122% | 60.342% | 159,507 | 364 | 1320 | 2,660 | 78.3% | 0.174x |
| 24 | 77.377% | 77.660% | 90,488 | 437 | 1584 | 3,192 | 71.0% | 0.250x |
| 28 | 86.415% | 86.731% | 54,339 | 510 | 1848 | 3,724 | 64.2% | 0.340x |
| 32 | 91.543% | 91.878% | 33,826 | 583 | 2112 | 4,256 | 58.2% | 0.444x |
| 40 | 96.042% | 96.393% | 15,833 | 729 | 2640 | 5,320 | 48.5% | 0.694x |
| 48 (current) | 97.748% | 98.105% | 9,007 | 874 | 3168 | 6,384 | 41.3% | 1.000x |
| 64 | 98.973% | 99.335% | 4,107 | 1166 | 4224 | 8,512 | 31.7% | 1.778x |
| 80 | 99.369% | 99.733% | 2,522 | 1457 | 5280 | 10,640 | 25.6% | 2.778x |
| 96 **← recommended** | 99.573% | 99.937% | 1,707 | 1749 | 6336 | 12,768 | 21.5% | 4.000x |
| 128 | 99.749% | 100.000% | 1,004 | 2332 | 8448 | 17,024 | 16.2% | 7.111x |

## 3. Smallest ratio meeting a coverage target

`healthy` excludes the 1,456 rows (0.36%) whose required frames exceed 100.0 per text token — those are misaligned pairs, not evidence that the head needs a larger ratio.

| Coverage | Smallest ratio (all rows) | Smallest ratio (healthy rows) |
|---|---:|---:|
| 99.0% | 80 | 64 |
| 99.5% | 96 | 80 |
| 99.9% | none in grid | 96 |
| 100.0% | none in grid | 128 |

### Rows excluded as degenerate

| Metric | p50 | p95 | max |
|---|---:|---:|---:|
| Text tokens | 11 | 38 | 53 |
| BiCodec tokens | 3000 | 3000 | 3000 |
| Required frames per text token | 197 | 969 | 2997 |

## 4. Is text length the right thing to size from?

Measured on healthy rows only. A wide distribution means the constant ratio has to be set for the tail, and every typical utterance pays for that in padded CTC frames.

| Anchor | Frames per anchor p50 | p95 | p99 | Coefficient of variation | p95/p50 | p99/p50 |
|---|---:|---:|---:|---:|---:|---:|
| target text tokens | 18.4 | 36.8 | 57.3 | 0.442 | 2.00x | 3.11x |
| source audio seconds | 52.8 | 70.0 | 74.6 | 0.200 | 1.33x | 1.41x |

## 5. Configuration

```json
{
  "manifests": [
    "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_train.jsonl",
    "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl"
  ],
  "ratio_grid": [
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    64,
    80,
    96,
    128
  ],
  "sample_rows_per_manifest": 200000,
  "current_ratio": 48,
  "coverage_targets": [
    0.99,
    0.995,
    0.999,
    1.0
  ],
  "text_length_verification": {
    "checked": 2000,
    "mismatched": 0
  },
  "degenerate_ratio_limit": 100.0
}
```
