# Step 2 — trained duration-anchored NAR CTC head decode probe

> Run `step2_trained_nar_decode_v8_unit_ce` · 2026-08-09T09:26:27.347244+00:00

Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b (V6 head was all-blank / ~100% UER).

| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | Empty | Blank frames | Blank-sup UER | Distinct pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.9% | 0.0 |
| `iter1000` | en2zh | 16 | 100.0% | 0 | 275 | 0.002 | 11 | 99.8% | 99.3% | 0.4 |
| `iter2000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.9% | 0.0 |
| `iter2000` | en2zh | 16 | 99.8% | 2 | 275 | 0.009 | 2 | 98.3% | 99.1% | 2.0 |
| `iter3000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 99.8% | 0.0 |
| `iter3000` | en2zh | 16 | 99.7% | 4 | 275 | 0.015 | 0 | 96.7% | 99.0% | 3.1 |

## Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "checkpoints": [
    "iter1000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v8_unit_ce/iter_0001000",
    "iter2000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v8_unit_ce/iter_0002000",
    "iter3000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v8_unit_ce/iter_0003000"
  ]
}
```
