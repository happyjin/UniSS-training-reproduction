# Step 2b — the NAR CTC head already trained inside joint V6

> Run `step2b_existing_nar_head_v1` · 2026-08-09T04:10:28+0000 · research only.

Best case for the head: frozen Phase3 backbone, teacher GLM source tokens, reference translation, ordinary causal attention. 16 samples per direction.

## 1. Does the head emit anything?

| Checkpoint | Dir | Samples | Unit error rate | Predicted units | Reference units | Length ratio | Empty | Blank frames | CTC infeasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage_a_iter500` | eng->cmn | 16 | 100.0% | 0 | 295 | 0.001 | 15 | 100.0% | 0 |
| `stage_a_iter500` | cmn->eng | 16 | 100.0% | 0 | 223 | 0.000 | 16 | 100.0% | 1 |
| `stage_b_iter2500` | eng->cmn | 16 | 100.0% | 0 | 295 | 0.000 | 15 | 100.0% | 0 |
| `stage_b_iter2500` | cmn->eng | 16 | 100.0% | 0 | 223 | 0.000 | 16 | 100.0% | 1 |
| `stage_b_iter5000` | eng->cmn | 16 | 100.0% | 0 | 295 | 0.000 | 16 | 100.0% | 0 |
| `stage_b_iter5000` | cmn->eng | 16 | 100.0% | 0 | 223 | 0.000 | 16 | 100.0% | 1 |

A unit error rate at or above 100% with a near-empty prediction is the degenerate all-blank CTC solution — the head has not learnt to emit units at all. Values meaningfully below 100% with a length ratio near 1.0 mean it has.

## 2. Is there any signal under the blank prior?

`blank suppressed` decodes the best non-blank token per frame. If the head learnt the content but mis-calibrated its blank prior, this stream resembles the reference and the collapse is a loss-balance problem. If it stays at ~100% error with a handful of distinct tokens, the head learnt nothing and has to be retrained.

| Checkpoint | Dir | Mean blank prob | Mean best non-blank prob | Blank-suppressed UER | Blank-suppressed units | Distinct |
|---|---|---:|---:|---:|---:|---:|
| `stage_a_iter500` | eng->cmn | 0.6177 | 0.0095 | 99.7% | 19 | 16.4 |
| `stage_a_iter500` | cmn->eng | 0.6492 | 0.0111 | 99.8% | 11 | 10.0 |
| `stage_b_iter2500` | eng->cmn | 0.7210 | 0.0034 | 99.7% | 18 | 13.2 |
| `stage_b_iter2500` | cmn->eng | 0.6438 | 0.0058 | 99.6% | 10 | 8.6 |
| `stage_b_iter5000` | eng->cmn | 0.7069 | 0.0037 | 99.6% | 18 | 12.7 |
| `stage_b_iter5000` | cmn->eng | 0.6341 | 0.0054 | 99.6% | 10 | 7.9 |

## 3. Vocabulary use

| Checkpoint | Dir | Distinct predicted | Distinct reference | Lattice occupancy |
|---|---|---:|---:|---:|
| `stage_a_iter500` | eng->cmn | 0.1 | 272.8 | 34.5% |
| `stage_a_iter500` | cmn->eng | 0.0 | 179.1 | 53.9% |
| `stage_b_iter2500` | eng->cmn | 0.1 | 272.8 | 34.5% |
| `stage_b_iter2500` | cmn->eng | 0.0 | 179.1 | 53.9% |
| `stage_b_iter5000` | eng->cmn | 0.0 | 272.8 | 34.5% |
| `stage_b_iter5000` | cmn->eng | 0.0 | 179.1 | 53.9% |

## 4. Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "total_samples": 32,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "upsample_ratio": 48,
  "blank_id": 8192,
  "device": "cuda:0"
}
```
