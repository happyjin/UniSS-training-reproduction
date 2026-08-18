# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000200`
- Evaluations: 8
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8974**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 960 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9648 | 4 | It'stolaidnowthatjustgetthisoverway | wer | 1.0000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9648 | 4 | It'stolaid innowthatjustgetthisoverway | wer | 1.0000 |
| 960 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9448 | 6 | That's true What about the magnetic impacts | wer | 0.2857 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9693 | 3 | That's true What about the magnetic impacts | wer | 0.2857 |
| 960 | streaming_asr | NCSSD_R_EN_0000001077 | 0.9207 | 14 | Hereisstridulousasslappedeveryonedisappointedtocomehereatthesametime | wer | 1.0000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000001077 | 0.9177 | 14 | Hereisstr ridiculousasslapeveryonedisappointedtocomehereatthesametime | wer | 1.0000 |
| 960 | causal_full_asr | NCSSD_R_EN_0000004455 | 1.0000 | 0 | Thank you all but you how much you're there big | wer | 0.6000 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000004455 | 0.9877 | 1 | Thank you all but you how much you're there work | wer | 0.5000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
