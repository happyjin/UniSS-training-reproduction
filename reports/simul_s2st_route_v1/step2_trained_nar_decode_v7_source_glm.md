# Step 2 — trained duration-anchored NAR CTC head decode probe

> Run `step2_trained_nar_decode_v7_source_glm` · 2026-08-09T08:53:56.250559+00:00

Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b (V6 head was all-blank / ~100% UER).

| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | Empty | Blank frames | Blank-sup UER | Distinct pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 99.9% | 16 | 193 | 0.089 | 0 | 0.0% | 99.9% | 6.4 |
| `iter1000` | en2zh | 16 | 99.2% | 36 | 275 | 0.136 | 0 | 0.0% | 99.2% | 15.8 |
| `iter2000` | zh2en | 16 | 99.9% | 20 | 193 | 0.109 | 0 | 0.0% | 99.9% | 10.5 |
| `iter2000` | en2zh | 16 | 99.2% | 32 | 275 | 0.119 | 0 | 0.0% | 99.2% | 15.8 |
| `iter3000` | zh2en | 16 | 99.9% | 22 | 193 | 0.116 | 0 | 0.0% | 99.9% | 10.1 |
| `iter3000` | en2zh | 16 | 99.2% | 28 | 275 | 0.112 | 0 | 0.0% | 99.2% | 14.8 |

## Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "checkpoints": [
    "iter1000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v7_source_glm/iter_0001000",
    "iter2000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v7_source_glm/iter_0002000",
    "iter3000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v7_source_glm/iter_0003000"
  ]
}
```
