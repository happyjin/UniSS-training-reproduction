# Stage A v3 anti-blank canary failure report

## Decision

The 127-update, eight-GPU Stage A v3 canary completed normally and saved its
final checkpoint, but **did not authorize formal Stage A training**.  Stage B
also remains blocked.

V3 fixed the v2 CTC all-blank collapse.  It did not retain enough exact
source-GLM identity after the curriculum reached the final 160 ms causal
chunk.  The final validation `causal_glm_agreement` was `0.009229417`, below
the required `0.02` threshold.

Machine-readable decision:

- `CANARY_GATE.json`: `passed=false`
- failed check: `causal_code_identity_retained=false`
- `stage_b_authorized=false`

## Run identity

- run ID: `stage_a_v3_antiblank_canary8_20260817T014922Z`
- immutable initialization: Phase3 v4 native checkpoint, iteration 9075
- devices: 8 x H200
- framework: Megatron
- sequence length: 18000
- micro/global batch: 1 / 128
- updates: 127, one globally shuffled coverage epoch
- maximum acoustics per pack: 2
- train log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v3/stage_a_canary/stage_a_v3_antiblank_canary8_20260817T014922Z/train.log`
- GPU log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v3/stage_a_canary/stage_a_v3_antiblank_canary8_20260817T014922Z/train.gpu.csv`
- TensorBoard: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v3/stage_a_canary/stage_a_v3_antiblank_canary8_20260817T014922Z/tensorboard`
- final checkpoint: `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v3/stage_a_canary/stage_a_v3_antiblank_canary8_20260817T014922Z/iter_0000127`

## Validation trajectory

| Validation | Chunk regime | AR-ASR | Source CTC | Teacher KL | CTC blank ratio | Blank posterior | GLM agreement | Teacher cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1280 ms | 2.318427 | 10.03148 | 8.046454 | 0.000000 | 0.005616 | 0.266181 | 0.974938 |
| 64 | 960 ms | 1.679314 | 6.854292 | 3.864700 | 0.000000 | 0.030938 | 0.163108 | 0.970724 |
| 96 | 640 ms | 0.852718 | 7.903368 | 3.157530 | 0.004977 | 0.072031 | 0.049357 | 0.918365 |
| 127 | 160 ms | 0.806612 | 7.080750 | 2.058350 | 0.017959 | 0.099355 | **0.009229** | 0.892115 |

All validation values were finite.  The run recorded zero skipped iterations
and zero NaN iterations.

## What worked

The v3 anti-collapse changes were effective for CTC:

- v2 final validation CTC greedy blank ratio: `1.0`
- v3 final validation CTC greedy blank ratio: `0.01795902`
- v3 final blank posterior: `0.09935516`
- v3 final dynamic blank budget: `0.8710938`

The fresh negative blank bias, early monotonic seed, differentiable blank
budget, and delayed Whisper unfreezing therefore prevented the all-blank
basin.  AR-ASR and teacher KL also improved over the canary.

## What failed

Exact causal code identity degraded monotonically as the chunk curriculum
became more difficult:

- 1280 ms validation: 26.62% exact agreement
- 960 ms validation: 16.31%
- 640 ms validation: 4.94%
- 160 ms final validation: 0.92%

The final teacher cosine of `0.892115` shows that representations remain
geometrically close to the teacher codebook, but nearest-code identity is not
preserved.  A small continuous-vector error can cross a discrete codebook
boundary, so MSE commitment with weight `0.10` is insufficient at 160 ms.

This is not evidence that CTC failed again.  It is a separate discrete
identity failure caused by the combination of progressively shorter context,
frontend unfreezing, and a commitment term that optimizes vector distance but
does not directly optimize the teacher code ID decision boundary.

## Gate parser defect found and fixed

Megatron prints intermediate validation as:

`validation loss at iteration 96 | ...`

but final validation as:

`validation loss at iteration 127 on validation set | ...`

The original v3 checker recognized only the first form.  It therefore retained
iteration-96 metrics and could have issued a false authorization.  The parser
now recognizes both forms, and a regression test proves that a failing final
validation overrides an earlier passing validation.

Verification:

- checker tests: `2 passed`
- Python compilation: passed
- real run checker exit: `1` as required for failure

## Compute behavior

For samples with more than 90 GiB allocated per GPU, the GPU monitor recorded:

- mean GPU utilization: 68.06%
- fraction of samples at or above 90% utilization: 25.83%
- mean power: 350.36 W/GPU
- maximum observed power: 653.16 W
- maximum observed memory: 98,611 MiB

The alternating 320/640/960 ms consistency work and checkpoint/evaluation
intervals create expected utilization variance.  No attempt should be made to
hide the failed identity gate by adding unrelated synthetic GPU load.

## Required next repair

Do not launch the 381-step v3 formal run.  The next experiment must be isolated
as v4 and start again from immutable Phase3 iteration 9075.  Its objective must
directly preserve discrete teacher-code identity at short chunks, while
retaining the successful v3 CTC anti-collapse terms.  A new v4 canary must pass
the final 160 ms validation, not merely the iteration-96 validation, before any
formal run is authorized.

