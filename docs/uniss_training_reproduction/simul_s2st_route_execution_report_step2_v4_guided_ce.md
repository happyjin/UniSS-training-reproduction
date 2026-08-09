# Simul-S2ST route — Step2 v4 guided-CE result

> 2026-08-09T07:48:34.985940+00:00

## Train

- 15-shard, mbs=64/gbs=512, blank_penalty=1.0, guided_ce_weight=1.0, 3000 iters
- Final valid nar_ctc / blank_mass / guided_ce: `9.029999E+00` / `1.145593E-01` / `8.880948E+00`
- blank_mass improved vs v3 (~0.176 → ~0.115) but guided_ce stayed ≈ log(V)

## Decode

| Ckpt | UER | Empty | Blank frames | Blank-sup UER | Distinct (blank-sup) |
|---|---:|---:|---:|---:|---:|
| `iter1000` | 100.0% | 28/32 | 99.9% | 99.8% | 3.8 |
| `iter2000` | 100.0% | 29/32 | 99.9% | 99.7% | 4.2 |
| `iter3000` | 100.0% | 28/32 | 99.9% | 99.7% | 5.0 |

## Reading

- Greedy still blank-collapsed (blank wins argmax while mean blank prob ~0.13).
- Blank-suppressed emits a few units but UER still ~99.5% — CE did not peak class mass.
- Next: diagnose optimizer/grad + raise guided CE weight / overfit smoke before v5 full train.
