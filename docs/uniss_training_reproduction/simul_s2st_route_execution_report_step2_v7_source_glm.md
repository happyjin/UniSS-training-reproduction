# Simul-S2ST route — Step2 v7 source-GLM result

> 2026-08-09

## Settings

- Speaker + source-GLM frame conditioning
- `ctc_weight=0`, `guided_ce_weight=1.0`, `blank_penalty=0.5`, lr=5e-4 constant, 3000 iters

## Decode (iter3000)

| Dir | UER | Empty | Blank frames | Pred units |
|---|---:|---:|---:|---:|
| zh2en | 99.9% | 0 | 0% | ~22 |
| en2zh | 99.2% | 0 | 0% | ~28 |

Final `guided_ce≈8.60` (still ≈ ln V).

## Reading

Source-GLM conditioning did **not** move content quality. Blank remains solved; the bottleneck is supervision (frame-stretched CE invents a bad alignment / under-specified mapping), not missing discrete source codes alone.

## Next

v8: **unit-pooled CE** (`adaptive_avg_pool` frames → unit length, then CE), optional; defaults keep v1–v7 unchanged.
