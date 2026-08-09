# Simul-S2ST route — Step2 v5 CE-dominant result

> 2026-08-09T08:10:09.156649+00:00

- Settings: ctc_weight=0.25, guided_ce=5.0, blank_penalty=2.0, lr=5e-4 constant
- Final valid nar_ctc / blank_mass / guided_ce: `9.993706E+00` / `1.070887E-02` / `8.802889E+00`

| Ckpt | UER | Empty | Blank frames | Blank-sup UER | Distinct (blank-sup) |
|---|---:|---:|---:|---:|---:|
| `iter1000` | 99.8% | 0/32 | 67.7% | 99.6% | 7.1 |
| `iter2000` | 99.7% | 1/32 | 66.3% | 99.6% | 6.5 |
| `iter3000` | 99.7% | 0/32 | 68.3% | 99.5% | 8.4 |

## Reading

- Blank collapse partially broken: empty predictions 0, blank frames ~45–89% (was ~100%).
- best_nonblank_prob > blank_prob on average, but guided_ce still ≈8.8 and UER ≈99.5%.
- Next: speaker-conditioned CE-only warm (v6); AR Pareto continues with Λ=0.

