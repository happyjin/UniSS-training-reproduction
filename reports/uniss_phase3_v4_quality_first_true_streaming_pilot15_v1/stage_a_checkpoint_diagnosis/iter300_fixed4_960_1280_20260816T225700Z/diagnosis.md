# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000300`
- Evaluations: 8
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9040**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 960 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9515 | 5 | As too laid now that just get this over way | wer | 0.4000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9251 | 9 | As till late now that just get this over way this | wer | 0.5000 |
| 960 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9202 | 9 | That's true What about the magnetic impacts | wer | 0.2857 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9202 | 10 | That's true What about in that case | wer | 0.4286 |
| 960 | streaming_asr | NCSSD_R_EN_0000001077 | 0.8140 | 26 | Yeah is stridulous as slapped everyone disappointed to come here at the same time | wer | 0.3571 |
| 1280 | streaming_asr | NCSSD_R_EN_0000001077 | 0.8110 | 28 | Here is str ridiculous as slap everyone disappointed to come here at the same time | wer | 0.4286 |
| 960 | causal_full_asr | NCSSD_R_EN_0000004455 | 0.9693 | 2 | Thank you all but you how much you're there work | wer | 0.5000 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000004455 | 0.9693 | 2 | Thank you all but you how much you're there work | wer | 0.5000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
