# Step 2 — trained duration-anchored NAR CTC head decode probe

> Run `step2_trained_nar_decode_v5_guided_ce_dominant` · 2026-08-09T08:07:16.889456+00:00

Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b (V6 head was all-blank / ~100% UER).

| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | Empty | Blank frames | Blank-sup UER | Distinct pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 100.0% | 5 | 193 | 0.025 | 0 | 46.1% | 100.0% | 2.3 |
| `iter1000` | en2zh | 16 | 99.6% | 4 | 275 | 0.019 | 0 | 89.3% | 99.3% | 3.1 |
| `iter2000` | zh2en | 16 | 100.0% | 3 | 193 | 0.018 | 1 | 45.2% | 100.0% | 2.3 |
| `iter2000` | en2zh | 16 | 99.5% | 5 | 275 | 0.022 | 0 | 87.5% | 99.4% | 3.5 |
| `iter3000` | zh2en | 16 | 100.0% | 5 | 193 | 0.025 | 0 | 47.5% | 100.0% | 3.0 |
| `iter3000` | en2zh | 16 | 99.5% | 7 | 275 | 0.028 | 0 | 89.2% | 99.1% | 4.3 |

## Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "checkpoints": [
    "iter1000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v5_guided_ce_dominant/iter_0001000",
    "iter2000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v5_guided_ce_dominant/iter_0002000",
    "iter3000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v5_guided_ce_dominant/iter_0003000"
  ]
}
```
