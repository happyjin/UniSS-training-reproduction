# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 2
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8095**
- Weighted CTC blank ratio: **0.2282**
- Weighted streaming WER/CER: **0.6000**
- Weighted causal-full WER/CER: **0.7143**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2511 | 18 | It's two laid now let just yeah this over way is | wer | 0.6000 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.1963 | 8 | That's true. What's about the mentality and that's | wer | 0.7143 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
