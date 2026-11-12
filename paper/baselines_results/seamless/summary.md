# SeamlessStreaming on RealSI sentence level (R1): results and tiers

Written 2026-11-12 16:05 (Prague). Registration: `paper/baselines_run_plan_seamless_v1.md`. Official weights and agent, inference only; one run per point (greedy, deterministic). ASR-BLEU: speech-only audio, Whisper-large-v3 (En) / Paraformer-large (Zh), `evaluation.text_metrics.corpus_bleu`. D1 = the paper's segment LAAL (unit at its segment's playback start, queue included); D0w = SimulEval word LAAL (unit at its segment's emission); D0 = SimulEval default speech LAAL (one delay per speech segment), for reference only. Computation-unaware, as in the paper.

## En→Zh (346 sentences)

| point | decision_threshold | min_starting_wait_w2vbert | min_unit_chunk_size | ASR-BLEU | text BLEU | D1 (s) | D0w (s) | D0 (s) | StartOffset p50 (s) | ASR cover |
|---|---|---|---|---|---|---|---|---|---|---|
| t0.1_w192_c50 | 0.1 | 192 | 50 | 14.62 | 16.04 | 2.19 | 1.67 | 4.32 | 2.24 | 1.0 |
| t0.3_w192_c50 | 0.3 | 192 | 50 | 15.08 | 16.75 | 2.26 | 1.78 | 4.37 | 2.24 | 1.0 |
| t0.5_w192_c50 | 0.5 | 192 | 50 | 15.03 | 16.61 | 2.30 | 1.85 | 4.40 | 2.24 | 1.0 |
| t0.7_w192_c50 | 0.7 | 192 | 50 | 15.01 | 16.72 | 2.34 | 1.91 | 4.41 | 2.24 | 1.0 |
| t0.9_w192_c50 | 0.9 | 192 | 50 | 15.09 | 16.8 | 2.41 | 2.00 | 4.45 | 2.24 | 1.0 |
| t0.1_w32_c25 | 0.1 | 32 | 25 | 13.21 | 14.77 | 1.98 | 1.29 | 3.20 | 0.96 | 1.0 |
| t0.3_w32_c25 | 0.3 | 32 | 25 | 13.36 | 14.88 | 2.09 | 1.46 | 3.37 | 1.28 | 1.0 |
| t0.5_w32_c25 | 0.5 | 32 | 25 | 13.69 | 15.16 | 2.11 | 1.50 | 3.42 | 1.28 | 1.0 |

Tiers (En→Zh; best ASR-BLEU within budget):

| tier | budget (s) | SimulS2ST-Omni (reported) | SeamlessStreaming by D1: ASR-BLEU (D1) [point] | by D0w (secondary): ASR-BLEU (D0w) [point] |
|---|---|---|---|---|
| m1 | 1.18 | 19.80 | -- | -- |
| m2 | 1.98 | 24.21 | 13.21 (1.98) [t0.1_w32_c25] | 15.08 (1.78) [t0.3_w192_c50] |
| m3 | 2.46 | 25.01 | 15.09 (2.41) [t0.9_w192_c50] | 15.09 (2.00) [t0.9_w192_c50] |
| m4 | 2.98 | 25.54 | 15.09 (2.41) [t0.9_w192_c50] | 15.09 (2.00) [t0.9_w192_c50] |

## Zh→En (431 sentences)

| point | decision_threshold | min_starting_wait_w2vbert | min_unit_chunk_size | ASR-BLEU | text BLEU | D1 (s) | D0w (s) | D0 (s) | StartOffset p50 (s) | ASR cover |
|---|---|---|---|---|---|---|---|---|---|---|
| t0.1_w192_c50 | 0.1 | 192 | 50 | 13.38 | 13.86 | 2.24 | 2.11 | 4.30 | 2.88 | 1.0 |
| t0.3_w192_c50 | 0.3 | 192 | 50 | 13.9 | 13.89 | 2.58 | 2.46 | 4.56 | 2.88 | 1.0 |
| t0.5_w192_c50 | 0.5 | 192 | 50 | 13.38 | 13.72 | 2.71 | 2.60 | 4.62 | 3.2 | 1.0 |
| t0.7_w192_c50 | 0.7 | 192 | 50 | 14.09 | 13.95 | 2.85 | 2.75 | 4.71 | 3.2 | 1.0 |
| t0.9_w192_c50 | 0.9 | 192 | 50 | 14.17 | 13.91 | 3.07 | 2.96 | 4.84 | 3.26 | 1.0 |
| t0.1_w32_c25 | 0.1 | 32 | 25 | 13.73 | 14.11 | 1.84 | 1.67 | 3.49 | 1.92 | 1.0 |
| t0.3_w32_c25 | 0.3 | 32 | 25 | 13.92 | 14.01 | 2.24 | 2.08 | 3.81 | 2.24 | 1.0 |
| t0.5_w32_c25 | 0.5 | 32 | 25 | 13.34 | 13.74 | 2.40 | 2.23 | 3.93 | 2.24 | 1.0 |

Tiers (Zh→En; best ASR-BLEU within budget):

| tier | budget (s) | SimulS2ST-Omni (reported) | SeamlessStreaming by D1: ASR-BLEU (D1) [point] | by D0w (secondary): ASR-BLEU (D0w) [point] |
|---|---|---|---|---|
| m1 | 1.18 | 14.89 | -- | -- |
| m2 | 1.92 | 19.55 | 13.73 (1.84) [t0.1_w32_c25] | 13.73 (1.67) [t0.1_w32_c25] |
| m3 | 2.40 | 21.07 | 13.92 (2.24) [t0.3_w32_c25] | 13.92 (2.08) [t0.3_w32_c25] |
| m4 | 2.98 | 22.47 | 14.09 (2.85) [t0.7_w192_c50] | 14.17 (2.96) [t0.9_w192_c50] |

