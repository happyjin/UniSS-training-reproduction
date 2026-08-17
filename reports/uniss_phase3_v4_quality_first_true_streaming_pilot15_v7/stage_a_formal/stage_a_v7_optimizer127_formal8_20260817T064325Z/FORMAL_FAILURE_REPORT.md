# Stage A V7 formal run failure report

## Decision

- Run: `stage_a_v7_optimizer127_formal8_20260817T064325Z`
- Decision: **FAIL**
- Stage B authorized: **false**
- Blocked next stage: `stage_b_incremental_mt`
- Strict gate: `STRICT_FORMAL_GATE.json`
- TensorBoard: `http://10.1.6.203:6120/`

The formal job itself completed normally: it reached iteration 381, saved the
final checkpoint, ran final validation, consumed the exact three-epoch global
shuffle, and reported no NaN or skipped iterations. The model nevertheless
failed two mandatory quality checks:

1. `strict_sustained_ctc_not_blank`: final validation CTC blank ratio was
   `0.9985937`, above the maximum `0.25`.
2. `teacher_geometry_retained`: final validation teacher-code cosine was
   `0.8325284`, below the minimum `0.85`.

The final checkpoint must not be selected for Stage B. Earlier V7 checkpoints
remain diagnostic artifacts only and must not be treated as a passed Stage A.

## Exact execution evidence

| Item | Result |
|---|---:|
| Megatron updates | 381/381 |
| Source packs | 16,195 |
| Coverage epochs | 3 |
| Padded samples per epoch | 16,256 |
| Total consumed samples | 48,768 |
| Global shuffle seed | 20,260,816 |
| Prefix/diagnostic mode | disabled |
| Optimizer/curriculum horizon | 127 updates |
| LR-floor hold | 254 updates |
| Final checkpoint | saved at iteration 381 |
| Final validation | completed at iteration 381, 160 ms |
| NaN iterations | 0 |
| Skipped iterations | 0 |

Training ran from approximately 06:43 to 07:35 UTC on 2026-08-17
(14:43–15:35 Beijing time). The exact checkpoint root is:

`checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v7/stage_a_formal/stage_a_v7_optimizer127_formal8_20260817T064325Z`

The run was initialized from the immutable Phase3 checkpoint and did not
resume any V6 failure or V7 canary checkpoint.

## Validation trajectory

| Iteration | AR-ASR | Source CTC | Blank ratio | Agreement | Teacher cosine | Adapter RMS | Chunk |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2.099969 | 8.225383 | 0.000000 | 0.237691 | 0.966284 | 0.048872 | 960 ms |
| 100 | 1.056385 | 6.360610 | 0.001934 | 0.113744 | 0.934021 | 0.135002 | 320 ms |
| 150 | 0.733805 | 7.660909 | 0.009752 | 0.115100 | 0.916549 | 0.154491 | 320 ms |
| 200 | 0.730792 | 6.918190 | 0.044701 | 0.089225 | 0.902239 | 0.199367 | 320 ms |
| 250 | 0.765203 | 5.611505 | 0.590422 | 0.124348 | 0.879723 | 0.229032 | 320 ms |
| 300 | 0.587839 | 4.898502 | 0.967678 | 0.125596 | 0.872440 | 0.250926 | 320 ms |
| 350 | 0.679031 | 3.925426 | 0.994804 | 0.110022 | 0.839922 | 0.271697 | 320 ms |
| 381 | 0.457490 | 4.757348 | 0.998594 | 0.086895 | 0.832528 | 0.285273 | 160 ms |

The per-update trace identifies the first threshold crossings:

- blank ratio first exceeded `0.25` at iteration **231**;
- blank ratio first exceeded `0.95` at iteration **297**;
- teacher cosine first fell below `0.85` at iteration **326**;
- agreement never fell below `0.02`;
- adapter RMS never exceeded `0.50`.

Iteration 200 is the last saved checkpoint whose validation still satisfies
the blank and teacher-cosine thresholds. It is useful for diagnosis only; it
does not satisfy the required complete three-epoch formal protocol.

## Why the post-decay canary passed but formal failed

The V7 optimizer-clock repair worked as intended. Unlike V6, V7 stayed healthy
through the old failure region and through the successful 191-update canary.
The failure is a slower long-horizon objective drift:

- the canary ended after 64 LR-floor updates at iteration 191;
- the formal blank gate failed at iteration 231, after 104 LR-floor updates;
- therefore the canary stopped 40 updates before the first formal gate
  violation.

This rules out the V6 optimizer-horizon mismatch as the remaining cause, but
shows that 64 LR-floor hold updates were not sufficient to prove stability.

## Root-cause analysis

### 1. The CTC anti-collapse constraint becomes inactive

The explicit monotonic non-blank seed decays to zero after 40% curriculum
progress and remains zero for the long short-chunk hold. The blank-budget loss
is driven by mean blank posterior, with a permissive target near `0.86`.

At final validation:

- mean blank posterior: `0.6165922`;
- blank-posterior budget target: `0.8632812`;
- recorded blank-budget loss: `0.0`;
- argmax blank ratio: `0.9985937`.

Thus the differentiable budget considers the state acceptable even though
blank is the highest-logit class on virtually every frame. The optimized
surrogate and the deployment/gate statistic are misaligned.

The CTC loss itself also gives a misleadingly positive signal. From iteration
200 to 350, validation source CTC improves from `6.918190` to `3.925426` while
blank ratio worsens from `0.044701` to `0.994804`. Standard CTC can lower
sequence loss while concentrating framewise argmax predictions on blank, as
long as small non-blank probabilities still support target paths. Therefore
falling CTC loss is not evidence of healthy streaming decoding here.

### 2. The residual code adapter drifts for too long

Adapter RMS grows monotonically from `0.199367` at iteration 200 to `0.285273`
at iteration 381, while teacher-code cosine falls from `0.902239` to
`0.832528`. The existing adapter residual weight (`0.01`) and codebook
commitment/identity terms are adequate for the short canary but do not bound
254 LR-floor updates. Agreement stays above its minimum because discrete code
agreement is less sensitive than continuous geometry; it cannot replace the
cosine gate.

### 3. This is not a data-order, checkpoint, or numerical failure

The strict checker verified all of the following:

- exact 16,195-pack, three-epoch, globally shuffled schedule;
- exact 48,768 consumed samples with seed 20,260,816;
- formal prefix mode disabled;
- initialization from immutable Phase3;
- final checkpoint and final validation present;
- all required metrics finite;
- zero NaN and zero skipped iterations.

The evidence therefore points to long-horizon objective design, not corrupted
input, accidental resume, incomplete coverage, or an optimizer-clock bug.

## Required repair before another formal run

The next isolated Stage A revision should start again from immutable Phase3
and should not resume this run. At minimum it must:

1. Add a differentiable blank-vs-best-nonblank margin penalty, so the training
   loss directly penalizes blank being the framewise winner. Mean blank
   posterior alone is insufficient.
2. Retain a small non-zero monotonic anchor during the short-chunk hold, or
   reactivate it when a differentiable blank health statistic deteriorates.
3. Tighten continuous code geometry with a stronger/scheduled adapter residual
   constraint and/or direct cosine-distance term. The target is to prevent the
   slow `adapter RMS up / teacher cosine down` drift, not merely cap RMS at
   `0.50`.
4. Extend the pre-formal hold beyond the observed failure boundary. A repaired
   canary should cover at least 128 LR-floor updates and must pass validation
   after the hold; 64 updates are now proven insufficient.
5. Add fail-fast monitoring for sustained blank-ratio and teacher-cosine
   violations so a future bad run is stopped near the first crossing rather
   than allowed to finish all 381 updates.

No Stage B job should be launched until a new formal Stage A final gate has
`passed=true` and `stage_b_authorized=true`.
