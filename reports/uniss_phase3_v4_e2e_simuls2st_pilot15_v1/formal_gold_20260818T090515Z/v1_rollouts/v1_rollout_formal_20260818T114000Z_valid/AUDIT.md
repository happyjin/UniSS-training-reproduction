# V1 append-only ASR rollout audit

- Status: **passed**
- Gold: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl`
- Rollouts: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z/v1_rollouts/v1_rollout_formal_20260818T114000Z_valid/valid_v1_rollouts.jsonl`
- Samples: **13,469**
- Events: **263,391**
- Append-only rollback count: **0**
- Empty event rate: **0.1688**
- Malformed WRITE rate: **0.0242**
- Early EOS rate: **0.0024**
- Final EOS sample rate: **1.0000**

| source language | metric | samples | errors | reference units | weighted error rate |
|---|---|---:|---:|---:|---:|
| cmn | cer | 5,708 | 44,081 | 176,169 | 0.2502 |
| eng | wer | 7,761 | 50,084 | 128,500 | 0.3898 |

The rollout is an immutable sidecar. It uses the gold event clock only to decide when the trained V1 ASR is queried; generated text is fully free-running and every accepted delta is append-only. Empty gold text events are deliberately not queried because that is the exact Stage A training protocol.
