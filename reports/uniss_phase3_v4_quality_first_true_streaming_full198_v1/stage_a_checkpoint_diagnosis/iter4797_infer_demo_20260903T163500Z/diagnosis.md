# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 8
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9274**
- Weighted CTC blank ratio: **0.7065**
- Weighted streaming WER/CER: **0.3684**
- Weighted causal-full WER/CER: **0.0750**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 960 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7665 | 34 | As june late now let Jeff get this over way | wer | 0.5000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7797 | 33 | As jewell laid now let just get this over way | wer | 0.5000 |
| 960 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7485 | 26 | That's true What about the genetic impacts | wer | 0.2857 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7669 | 26 | That's true What about the genetic impact | wer | 0.1429 |
| 960 | streaming_asr | CommonVoice_EN_0000042263 | 0.7000 | 39 | There remain of the year was spent in port | wer | 0.2222 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.7053 | 38 | There remain of the year was spent in port | wer | 0.2222 |
| 960 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6392 | 78 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6297 | 82 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
