# Step 1 (D1) — joint-V6 checkpoints under a frozen Phase3 BLEU probe

> Run `step1_v6_bleu_recheck_v1` · 2026-08-09T03:56:29+0000 · research only.

Backend held fixed at `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf` for every row. 16 samples per direction (32 total) from `joint_valid.jsonl`.

## 1. Agreement against downstream BLEU

| Stream | Chunk | Agreement EN→ZH | BLEU EN→ZH | Agreement ZH→EN | BLEU ZH→EN |
|---|---|---:|---:|---:|---:|
| `manifest_teacher_glm` | offline | 100.00% | 44.10 | 100.00% | 38.13 |
| `pretrained_frontend` | 320ms | 18.61% | 41.27 | 7.67% | 31.28 |
| `pretrained_frontend` | offline | 25.48% | 44.40 | 19.30% | 31.80 |
| `stage_a_iter500` | 320ms | 17.42% | 41.38 | 7.62% | 31.42 |
| `stage_a_iter500` | offline | 25.98% | 41.98 | 18.12% | 29.42 |
| `stage_b_iter250` | 320ms | 18.97% | 40.72 | 8.15% | 27.05 |
| `stage_b_iter250` | offline | 26.75% | 41.04 | 21.23% | 32.77 |
| `stage_b_iter1250` | 320ms | 16.43% | 40.38 | 6.54% | 26.59 |
| `stage_b_iter1250` | offline | 24.50% | 43.32 | 15.88% | 31.77 |
| `stage_b_iter2500` | 320ms | 13.02% | 36.33 | 4.72% | 28.08 |
| `stage_b_iter2500` | offline | 19.24% | 42.09 | 10.24% | 36.99 |
| `stage_b_iter3750` | 320ms | 10.51% | 40.44 | 3.46% | 29.43 |
| `stage_b_iter3750` | offline | 15.66% | 40.89 | 6.40% | 36.51 |
| `stage_b_iter5000` | 320ms | 8.59% | 42.98 | 2.57% | 22.35 |
| `stage_b_iter5000` | offline | 13.27% | 44.73 | 4.75% | 39.89 |

## 2. Detail

| Stream | Chunk | Dir | Samples | BLEU | chrF | Agreement | Length ratio | Empty hyp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `manifest_teacher_glm` | offline | eng->cmn | 16 | 44.10 | 36.93 | 100.00% | 1.000 | 0 |
| `manifest_teacher_glm` | offline | cmn->eng | 16 | 38.13 | 52.46 | 100.00% | 1.000 | 0 |
| `pretrained_frontend` | 320ms | eng->cmn | 16 | 41.27 | 35.32 | 18.61% | 0.998 | 0 |
| `pretrained_frontend` | 320ms | cmn->eng | 16 | 31.28 | 45.63 | 7.67% | 0.995 | 0 |
| `pretrained_frontend` | offline | eng->cmn | 16 | 44.40 | 37.93 | 25.48% | 0.998 | 0 |
| `pretrained_frontend` | offline | cmn->eng | 16 | 31.80 | 45.03 | 19.30% | 0.995 | 0 |
| `stage_a_iter500` | 320ms | eng->cmn | 16 | 41.38 | 35.63 | 17.42% | 0.998 | 0 |
| `stage_a_iter500` | 320ms | cmn->eng | 16 | 31.42 | 46.76 | 7.62% | 0.995 | 0 |
| `stage_a_iter500` | offline | eng->cmn | 16 | 41.98 | 36.17 | 25.98% | 0.998 | 0 |
| `stage_a_iter500` | offline | cmn->eng | 16 | 29.42 | 47.03 | 18.12% | 0.995 | 0 |
| `stage_b_iter250` | 320ms | eng->cmn | 16 | 40.72 | 34.29 | 18.97% | 0.998 | 0 |
| `stage_b_iter250` | 320ms | cmn->eng | 16 | 27.05 | 44.78 | 8.15% | 0.995 | 0 |
| `stage_b_iter250` | offline | eng->cmn | 16 | 41.04 | 34.85 | 26.75% | 0.998 | 0 |
| `stage_b_iter250` | offline | cmn->eng | 16 | 32.77 | 50.35 | 21.23% | 0.995 | 0 |
| `stage_b_iter1250` | 320ms | eng->cmn | 16 | 40.38 | 34.14 | 16.43% | 0.998 | 0 |
| `stage_b_iter1250` | 320ms | cmn->eng | 16 | 26.59 | 42.63 | 6.54% | 0.995 | 0 |
| `stage_b_iter1250` | offline | eng->cmn | 16 | 43.32 | 37.10 | 24.50% | 0.998 | 0 |
| `stage_b_iter1250` | offline | cmn->eng | 16 | 31.77 | 47.57 | 15.88% | 0.995 | 0 |
| `stage_b_iter2500` | 320ms | eng->cmn | 16 | 36.33 | 30.87 | 13.02% | 0.998 | 0 |
| `stage_b_iter2500` | 320ms | cmn->eng | 16 | 28.08 | 44.63 | 4.72% | 0.995 | 0 |
| `stage_b_iter2500` | offline | eng->cmn | 16 | 42.09 | 35.35 | 19.24% | 0.998 | 0 |
| `stage_b_iter2500` | offline | cmn->eng | 16 | 36.99 | 50.93 | 10.24% | 0.995 | 0 |
| `stage_b_iter3750` | 320ms | eng->cmn | 16 | 40.44 | 34.29 | 10.51% | 0.998 | 0 |
| `stage_b_iter3750` | 320ms | cmn->eng | 16 | 29.43 | 44.54 | 3.46% | 0.995 | 0 |
| `stage_b_iter3750` | offline | eng->cmn | 16 | 40.89 | 34.53 | 15.66% | 0.998 | 0 |
| `stage_b_iter3750` | offline | cmn->eng | 16 | 36.51 | 50.21 | 6.40% | 0.995 | 0 |
| `stage_b_iter5000` | 320ms | eng->cmn | 16 | 42.98 | 36.05 | 8.59% | 0.998 | 0 |
| `stage_b_iter5000` | 320ms | cmn->eng | 16 | 22.35 | 40.66 | 2.57% | 0.995 | 0 |
| `stage_b_iter5000` | offline | eng->cmn | 16 | 44.73 | 37.74 | 13.27% | 0.998 | 0 |
| `stage_b_iter5000` | offline | cmn->eng | 16 | 39.89 | 52.32 | 4.75% | 0.995 | 0 |

## 3. Did the checkpoint's own Qwen move?

| Checkpoint | Iteration | Loaded tensors | Changed Qwen tensors | Max abs delta |
|---|---:|---:|---:|---:|
| `stage_a_iter500` | 500 | 604 | 0/291 | 0 |
| `stage_b_iter250` | 250 | 604 | 178/291 | 9.537e-07 |
| `stage_b_iter1250` | 1250 | 604 | 221/291 | 1.526e-05 |
| `stage_b_iter2500` | 2500 | 604 | 223/291 | 1.526e-05 |
| `stage_b_iter3750` | 3750 | 604 | 226/291 | 3.052e-05 |
| `stage_b_iter5000` | 5000 | 604 | 227/291 | 3.052e-05 |

## 4. Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "whisper_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/pretrained_models/UniSS/glm4_tokenizer",
  "tokenizer_map_dir": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/tokenizer_maps",
  "samples_per_direction": 16,
  "total_samples": 32,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "max_new_tokens": 192,
  "chunks": [
    "320ms",
    "offline"
  ],
  "device": "cuda:0"
}
```
