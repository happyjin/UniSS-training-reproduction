# Stage A v2 cached-runtime free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_smoke/stage_a_v2_teacher_smoke8_20260817T003700Z/iter_0000032`
- HF Qwen: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/uniss_stage_a_v2_teacher_smoke8_iter_0000032_hf`
- Evaluations: 4
- Committed rollback count: **0**
- Append-only rows: **1.0000**
- Cached/recomputed token parity: **1.0000**
- Cached/recomputed free-generation parity: **1.0000**
- Streaming WER/CER: **1.0000**
- Causal-full WER/CER: **0.8824**

| task | language | sample | CTC blank | AR text | metric | error rate | rollback |
|---|---|---|---:|---|---|---:|---:|
| streaming_asr | eng | NCSSD_R_EN_0000000261 | 0.0308 |  | wer | 1.0000 | 0 |
| causal_full_asr | eng | NCSSD_R_EN_0000000402 | 0.0491 | Yes, you. What about the matter in that? | wer | 0.7143 | 0 |
| causal_full_asr | eng | NCSSD_R_EN_0000004455 | 0.0613 | Don't you? I'm worried you're broke. | wer | 1.0000 | 0 |
| streaming_asr | eng | NCSSD_R_EN_0000005155 | 0.0195 |  | wer | 1.0000 | 0 |

Rollback is measured on the persistent accepted-token ledger: each new WRITE event may only append; every earlier committed token and its hash must remain unchanged.
