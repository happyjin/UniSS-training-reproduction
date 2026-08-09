# Step 2 — trained duration-anchored NAR CTC head decode probe

> Run `step2_trained_nar_decode_v6_speaker_ce` · 2026-08-09T08:33:46.101426+00:00

Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b (V6 head was all-blank / ~100% UER).

| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | Empty | Blank frames | Blank-sup UER | Distinct pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 100.0% | 11 | 193 | 0.059 | 0 | 0.0% | 100.0% | 5.2 |
| `iter1000` | en2zh | 16 | 99.4% | 20 | 275 | 0.080 | 0 | 0.0% | 99.4% | 11.8 |
| `iter2000` | zh2en | 16 | 99.9% | 12 | 193 | 0.060 | 0 | 0.0% | 99.9% | 6.9 |
| `iter2000` | en2zh | 16 | 99.3% | 22 | 275 | 0.092 | 0 | 0.0% | 99.3% | 13.6 |
| `iter3000` | zh2en | 16 | 99.9% | 12 | 193 | 0.065 | 0 | 0.0% | 99.9% | 7.2 |
| `iter3000` | en2zh | 16 | 98.9% | 25 | 275 | 0.105 | 0 | 0.0% | 98.9% | 14.6 |

## Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "checkpoints": [
    "iter1000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v6_speaker_ce/iter_0001000",
    "iter2000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v6_speaker_ce/iter_0002000",
    "iter3000=checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v6_speaker_ce/iter_0003000"
  ]
}
```
