# Stage B E2 真流式延迟评估

- checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_uniss_subsecond_v1/stage_b_pilot_15shard_vectorized_v2/best.pt`
- manifest: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/simul_uniss_subsecond_e2_v1/unist_dev/stage_a_source/manifest.jsonl`
- 口径：PCM 增量输入、causal log-Mel、Emformer cache；NCA 只计音频时间，CA 加入实测计算排队。
- 当前 E2 只评估 causal frontend / First WRITE 下界，不等于 Qwen + BiCodec 端到端首个可播放翻译音频。

## 汇总

| chunk/right | records | First GLM NCA p50/p95 | stable首token覆盖 | wait-k2 stable First WRITE CA p50/p95 | <1s比例 | active RTF p50 | GLM agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| 160/80 ms | 7965 | 320.0/1920.0 ms | 31.0% | 337.4/337.9 ms | 31.0% | 0.1055 | 17.85% |
| 320/80 ms | 7965 | 640.0/1920.0 ms | 31.7% | 657.5/658.0 ms | 31.6% | 0.0541 | 16.70% |
| 160/0 ms | 7965 | 320.0/2080.0 ms | 31.6% | 337.2/337.8 ms | 31.5% | 0.1059 | 12.98% |
| 320/0 ms | 7965 | 640.0/2240.0 ms | 32.1% | 657.3/657.7 ms | 32.1% | 0.0543 | 13.69% |

## 判定规则

- `First predicted GLM` 很低只证明模型较早发出 token，可能是错误 token。
- 方案是否满足 `<1 s`，至少应查看 `wait-k stable First WRITE CA` 的覆盖率和比例。
- Stage B 独立质量门要求最终 GLM token agreement ≥90%；未通过时不能据此声称端到端同传已满足1秒且质量合格。
- 真正端到端 `<1 s` 还必须在 E4/E5 接入 Qwen micro-WRITE、Streaming BiCodec、网络和播放器缓冲后重新测量 Useful First Audio CA。
