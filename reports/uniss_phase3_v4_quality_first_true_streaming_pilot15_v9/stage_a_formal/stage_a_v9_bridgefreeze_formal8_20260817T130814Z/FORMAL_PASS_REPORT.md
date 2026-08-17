# Stage A V9 bridge-freeze formal pass report

## Decision

PASS. The complete 381-update V9 formal Stage A run passed all 20 strict
machine checks. The structural Stage A gate sets `stage_b_authorized=true`.

This decision proves complete short-chunk training stability. It does not by
itself prove acceptable free-running WER/CER; the fixed 334-ID, four-chunk
diagnosis must still be run before claiming streaming ASR quality parity with
offline Phase3.

## Run identity

- Run: `stage_a_v9_bridgefreeze_formal8_20260817T130814Z`
- Initialization: immutable Phase3 iteration 9075
- Framework: Megatron, 8 H200 GPUs
- Sequence length: 18,000
- Micro/global batch: 1 / 128
- Source packs: 16,195
- Coverage epochs: 3
- Total consumed samples: 48,768
- Global shuffle seed: `20260816`
- Curriculum/optimizer horizon: 127 updates
- Post-horizon hold: 254 updates
- Final checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`

## Final 160-ms validation

| Metric | Required | Final | Result |
|---|---:|---:|---|
| AR-ASR | `< 3.0` | `0.5173882` | PASS |
| Source CTC | `< 15.0` | `6.783165` | PASS |
| CTC blank ratio | `<= 0.25` | `0.1880285` | PASS |
| CTC blank posterior | `<= target + 0.05` | `0.2395359` | PASS |
| Causal GLM agreement | `>= 0.02` | `0.09014972` | PASS |
| Teacher-code cosine | `>= 0.85` | `0.8994169` | PASS |
| Code-adapter RMS | `<= 0.50` | `0.2047903` | PASS |
| Skipped / NaN updates | `0 / 0` | `0 / 0` | PASS |

The run completed exact three-epoch global shuffle geometry, saved the final
checkpoint, reached final 160-ms validation, and retained finite metrics.

## Interpretation

V9 is the first revision in this series to retain both non-blank CTC behavior
and Phase3 teacher geometry through the entire 254-update post-curriculum
hold. Compared with V7 formal, final blank ratio improved from `0.998594` to
`0.188029` and teacher cosine improved from `0.832528` to `0.899417`.

The next evidence required for an ASR-quality conclusion is free-running
decoding on the exact V1 protocol: 334 fixed validation IDs at 160, 320, 640,
and 1280 ms, reporting Chinese CER and English WER against the matching
Phase3 offline anchor.
