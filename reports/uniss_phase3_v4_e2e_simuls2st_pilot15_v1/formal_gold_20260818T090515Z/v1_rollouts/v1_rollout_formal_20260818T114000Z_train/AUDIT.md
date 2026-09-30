# V1 append-only ASR rollout audit

- Status: **passed**
- Gold: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/source_events/train_gold_trajectories.jsonl`
- Rollouts: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/v1_rollouts/v1_rollout_formal_20260818T114000Z_train/train_v1_rollouts.jsonl`
- Samples: **1,325,243**
- Events: **25,997,984**
- Append-only rollback count: **0**
- Empty event rate: **0.1683**
- Malformed WRITE rate: **0.0252**
- Early EOS rate: **0.0023**
- Final EOS sample rate: **0.9999**

| source language | metric | samples | errors | reference units | weighted error rate |
|---|---|---:|---:|---:|---:|
| cmn | cer | 565,268 | 4,341,048 | 17,257,462 | 0.2515 |
| eng | wer | 759,975 | 4,999,024 | 12,772,752 | 0.3914 |

The rollout is an immutable sidecar. It uses the gold event clock only to decide when the trained V1 ASR is queried; generated text is fully free-running and every accepted delta is append-only. Empty gold text events are deliberately not queried because that is the exact Stage A training protocol.
