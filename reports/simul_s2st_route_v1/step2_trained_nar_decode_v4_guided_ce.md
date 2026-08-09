# Step 2 — trained duration-anchored NAR CTC head decode probe

> Run `step2_trained_nar_decode_v4_guided_ce` · 2026-08-09T07:45:09.307808+00:00

Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b (V6 head was all-blank / ~100% UER).

| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | Empty | Blank frames | Blank-sup UER | Distinct pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iter1000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 100.0% | 0.0 |
| `iter1000` | en2zh | 16 | 100.0% | 0 | 275 | 0.001 | 12 | 99.8% | 99.6% | 0.3 |
| `iter2000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 100.0% | 0.0 |
| `iter2000` | en2zh | 16 | 100.0% | 0 | 275 | 0.001 | 13 | 99.8% | 99.5% | 0.2 |
| `iter3000` | zh2en | 16 | 100.0% | 0 | 193 | 0.000 | 16 | 100.0% | 100.0% | 0.0 |
| `iter3000` | en2zh | 16 | 100.0% | 0 | 275 | 0.001 | 12 | 99.7% | 99.4% | 0.3 |

## Configuration

```json
{
  "manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
  "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
  "samples_per_direction": 16,
  "min_audio_seconds": 2.0,
  "max_audio_seconds": 10.0,
  "checkpoints": [
    "iter1000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v4_guided_ce/iter_0001000",
    "iter2000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v4_guided_ce/iter_0002000",
    "iter3000=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v4_guided_ce/iter_0003000"
  ]
}
```
