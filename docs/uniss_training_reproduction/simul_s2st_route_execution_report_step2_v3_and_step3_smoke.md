# Simul-S2ST route — Step2 v3 blankpen + Step3 AR Pareto smoke

> Auto-generated 2026-08-09T07:15:06.191606+00:00

## Step2 v3 blankpen train

- Data: 15-shard joint (`pilot_15shard_joint`), mbs=64 / gbs=512, blank_penalty=1.0
- Final valid `nar_ctc` / `nar_blank_mass`: 8.918175E+00 / 1.764056E-01
- Checkpoint: `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v3_blankpen/iter_0003000`

## Decode probe

| Ckpt | UER | Empty | Blank frames | Blank-sup UER | Distinct (blank-sup) |
|---|---:|---:|---:|---:|---:|
| `iter1000` | 100.0% | 31/32 | 100.0% | 99.8% | 4.2 |
| `iter2000` | 100.0% | 31/32 | 100.0% | 99.7% | 4.3 |
| `iter3000` | 100.0% | 31/32 | 100.0% | 99.7% | 6.1 |

## Step3 AR Pareto smoke

| k | Λ window | BLEU | chrF | First WRITE ms | Fallback | RTF |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.14 | 2.26 | 680 | 0% | 3.90 |
| 0 | 512 | 1.27 | 2.51 | 680 | 0% | 2.87 |
| 2 | 0 | 3.14 | 3.78 | 2927 | 38% | 3.64 |
| 2 | 512 | 0.42 | 0.50 | 2927 | 38% | 3.14 |
| 4 | 0 | 6.90 | 7.74 | 4760 | 62% | 3.52 |
| 4 | 512 | 0.32 | 0.78 | 4760 | 62% | 2.88 |
| 8 | 0 | 9.52 | 9.68 | 6560 | 75% | 2.54 |
| 8 | 512 | 0.11 | 0.18 | 6560 | 75% | 2.55 |
