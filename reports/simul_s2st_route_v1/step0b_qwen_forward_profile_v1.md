# Step 0b — what the 25 ms Qwen forward is made of

> Run `step0b_qwen_forward_profile_v1` · 2026-08-09T03:09:37+0000 · NVIDIA H200 · research only.

Model: 24 layers, hidden 896, vocab 180480, 48 unmerged LoRA adapters. KV cache held at 1024 positions.

## 1. Batch scaling of a one-token forward

A cost that barely moves from batch 1 to batch 32 is dominated by fixed per-call
overhead rather than by arithmetic.

| Batch | With LoRA adapters (ms) | With LoRA merged (ms) | Merged speed-up | Per-sequence merged (ms) |
|---:|---:|---:|---:|---:|
| 1 | 24.40 | 18.63 | 1.31x | 18.63 |
| 2 | 26.83 | 19.96 | 1.34x | 9.98 |
| 4 | 26.44 | 19.82 | 1.33x | 4.95 |
| 8 | 26.48 | 19.96 | 1.33x | 2.49 |
| 16 | 26.02 | 19.81 | 1.31x | 1.24 |
| 32 | 26.14 | 19.75 | 1.32x | 0.62 |

## 2. Where one forward goes at batch 1

| Component | Mean ms | Median ms |
|---|---:|---:|
| `backbone_only` | 24.52 | 24.52 |
| `lm_head_only` | 0.15 | 0.15 |

## 3. LoRA merge is an identity, not an approximation

- merged modules: 48
- max absolute logit change: 0.2812 (logit magnitude up to 15.38)
- argmax agreement: 100.00%

## 4. Repetition penalty over the expanded vocabulary

The Stage10 helper issues three CUDA operations per distinct generated token.

| History tokens | Python loop (ms) | Vectorised (ms) | Speed-up | Max abs delta |
|---:|---:|---:|---:|---:|
| 16 | 0.91 | 0.23 | 4.0x | 0 |
| 64 | 3.47 | 0.23 | 14.9x | 0 |
| 256 | 13.67 | 0.25 | 54.3x | 0 |
| 512 | 27.25 | 0.28 | 98.8x | 0 |
