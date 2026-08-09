# Step 2 — Megatron smoke for the duration-anchored causal NAR CTC head

> Run `step2_nar_ctc_smoke_v1` · 2026-08-09 · research only.

## Setup

- Entry: `experiments/simul_s2st_route_v1/step2_nar_ctc_head/pretrain_nar_ctc_megatron.py`
- Data: `smoke_manifest_128` (256 train rows)
- Backbone: frozen Phase3 `qwen0p5b_phase3_unist198_iter_0009075_hf`
- Head: `DurationAnchoredCausalNARCTC` at 75 frames/s, max 900 frames
- 8× H200, micro-batch 1, global batch 8, 40 iters

## Result

| Metric | Iter 1 | Iter 40 | Valid |
|---|---:|---:|---:|
| `nar_ctc` | 14.43 | **9.98** | 12.68 |
| `nar_infeasible` | **0** | **0** | **0** |
| lattice occupancy | 0.65 | 0.66 | 0.67 |
| grad norm | 19.5 | 8.8 | — |

Loss fell ~31% in 40 steps with zero infeasible CTC paths. Checkpoints:
`checkpoints/simul_s2st_route_v1/step2_nar_ctc_smoke_v1/{iter_0000020,iter_0000040}`.

This is only a wiring smoke (tiny data, short run). The 15shard epoch is the first
real test of whether the head learns units rather than blanks.
