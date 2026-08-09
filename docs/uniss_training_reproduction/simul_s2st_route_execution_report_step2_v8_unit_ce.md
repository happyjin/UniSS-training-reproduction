# Simul-S2ST route — Step2 v8 unit-pooled CE

> 2026-08-09T09:44:52.850768+00:00

- Settings: unit_ce=1.0, ctc=0.25, blank_penalty=0.5, guided_ce=0
- Train finished 3000 iters; unit_ce plateaued ~8.56 (≈ln V)

| Ckpt | Dir | UER | Empty | Blank frames | Pred units |
|---|---|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.9% | 0.0 |
| `iter1000` | en2zh | 16 | 100.0% | 0 | 275 | 0.002 | 11 | 99.8% | 99.3% | 0.4 |
| `iter2000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.9% | 0.0 |
| `iter2000` | en2zh | 16 | 99.8% | 2 | 275 | 0.009 | 2 | 98.3% | 99.1% | 2.0 |
| `iter3000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.8% | 0.0 |
| `iter3000` | en2zh | 16 | 99.7% | 4 | 275 | 0.015 | 0 | 96.7% | 99.0% | 3.1 |

## Reading

- Unit-pooled CE did **not** break the ln(V) plateau.
- Decode worsened vs v6/v7: blank frames returned (~97–100%), many empty preds.
- CTC weight 0.25 likely re-introduced blank collapse; unit CE alone did not peak classes.

