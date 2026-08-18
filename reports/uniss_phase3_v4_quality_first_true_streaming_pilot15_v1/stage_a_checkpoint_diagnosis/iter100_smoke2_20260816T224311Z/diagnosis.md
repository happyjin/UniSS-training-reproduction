# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000100`
- Evaluations: 2
- CTC blank collapse: **True**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.7619**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 1.0000 | 0 | Andandandandthethethethethethethe | wer | 1.0000 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 1.0000 | 0 |  | wer | 1.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
