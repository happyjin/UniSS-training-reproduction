# Stage A V8 128-update LR-floor hold canary failure

## Decision

**FAIL. V8 formal and Stage B remain blocked.**

- Run: `stage_a_v8_hold128_canary8_20260817T104000Z`
- Updates: 255/255
- LR-floor hold: 128 updates
- Consumed shuffled prefix: 32,640 samples
- Final checkpoint: saved at iteration 255
- NaN / skipped updates: 0 / 0
- TensorBoard: `http://10.1.6.203:6122/`
- Machine gate: `CANARY_GATE.json`

The gate passed 18 of 20 checks and failed:

- `strict_sustained_ctc_not_blank`: `0.3184921 > 0.25`
- `teacher_geometry_retained`: `0.8474859 < 0.85`

It records `passed=false`, `formal_v8_authorized=false`, and
`stage_b_authorized=false`. The V8 formal launcher must not run from this
artifact.

## Final validation

| Metric | V8 result | Required | Decision |
|---|---:|---:|---|
| AR-ASR | 0.469841 | < 3.0 | PASS |
| Source CTC | 6.882926 | < 15.0 | PASS |
| Blank ratio | 0.318492 | <= 0.25 | **FAIL** |
| Blank posterior | 0.241635 | <= 0.25 | PASS |
| Blank target | 0.200195 | — | active |
| Causal GLM agreement | 0.092326 | >= 0.02 | PASS |
| Teacher-code cosine | 0.847486 | >= 0.85 | **FAIL** |
| Adapter RMS | 0.270982 | <= 0.50 | PASS |
| Persistent seed strength | 0.100098 | > 0 | PASS |

## Long-hold trajectory

| Update | Blank budget | Blank ratio | Blank posterior | Teacher cosine | Adapter RMS |
|---:|---:|---:|---:|---:|---:|
| 127 | 0.000000 | 0.003508 | 0.090057 | 0.898119 | 0.197179 |
| 160 | 0.000000 | 0.010044 | 0.113495 | 0.880123 | 0.219333 |
| 181 | 0.000000 | 0.023005 | 0.136097 | 0.872393 | 0.228669 |
| 200 | ~0.000000 | 0.055940 | 0.165595 | 0.861106 | 0.239222 |
| 224 | 0.000248 | 0.165885 | 0.212477 | 0.851597 | 0.252095 |
| 231 | 0.000845 | 0.227294 | 0.226005 | 0.858763 | 0.252010 |
| 240 | 0.001897 | 0.322340 | 0.239767 | 0.846021 | 0.260227 |
| 249 | 0.002410 | 0.370250 | 0.244355 | 0.849011 | 0.258770 |
| 255 | 0.001939 | 0.333063 | 0.239919 | 0.846592 | 0.260274 |

First per-update threshold crossings:

- blank posterior exceeded 0.20 at update 219;
- teacher cosine first fell below 0.85 at update 234;
- blank ratio first exceeded 0.25 at update 235.

## What V8 fixed

V8 materially improved the V7 CTC failure. At comparable long-hold time, V7
was moving rapidly toward all blank and its final formal blank ratio was
0.998594. V8 retained a non-zero monotonic seed, activated its blank budget,
and ended this canary at 0.318492 rather than near 1.0. The mean blank
posterior also passed its new 0.25 gate. This validates the repair direction.

## What remains wrong

The differentiable decision-margin component is too weak. It activates near
the failure boundary, but the combined blank-budget magnitude remains only
about 0.002 and cannot keep framewise blank argmax below 0.25.

The adapter continues to drift after the 127-update optimizer horizon. RMS
increases from 0.197 at update 127 to 0.271 at final validation while teacher
cosine falls below the gate. Increasing geometry loss weights alone did not
stop the slow LR-floor drift.

## Minimal next repair

The next isolated version should again initialize from immutable Phase3 and
must not resume this V8 checkpoint.

1. Increase the decision-margin contribution substantially while retaining
   the already-successful 0.20 posterior target and persistent seed floor.
2. Lower the bridge/adapter learning rate and strengthen its residual trust
   region; the objective should preserve the useful adapter but prevent
   post-horizon drift.
3. Keep all data, shuffle, Phase3 replay, Qwen/CTC settings, 18000 sequence
   length, and Megatron batch geometry unchanged.
4. Repeat the same 255-update canary. Formal remains forbidden unless both
   blank ratio and teacher cosine pass at the final 160-ms validation.
