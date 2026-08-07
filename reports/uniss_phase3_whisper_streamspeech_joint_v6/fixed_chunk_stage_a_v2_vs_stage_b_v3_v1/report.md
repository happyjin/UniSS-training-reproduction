# Phase3 Whisper StreamSpeech Joint V6 固定 chunk 评估报告

本报告使用相同 15-shard 双语 validation、相同 8-GPU Megatron 入口，分别固定 Whisper chunk，避免训练期间随机 chunk 导致的不可比性。数值为 loss/诊断指标，不等同于端到端 BLEU、语音质量或真实播放延迟。

- Stage A checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_a_heads_only_15shard_v2`
- Stage B checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_b_guarded_joint_15shard_v3`
- right context: `80 ms`
- validation: `8 × global batch 128 = 1024` samples per operating point

## 1. 固定 chunk 绝对结果

| 模型 | chunk | BiCodec CTC ↓ | AR S2TT ↓ | ASR CTC ↓ | NAR S2TT CTC ↓ | unit infeasible ↓ | commitment ↓ | teacher agreement ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_a | 320 ms | 10.4380 | 5.2006 | 34.8821 | 33.4400 | 0.0244 | 0.02107 | 12.22% |
| stage_a | 640 ms | 10.4461 | 5.2054 | 34.8854 | 33.4498 | 0.0244 | 0.02101 | 14.42% |
| stage_a | 960 ms | 10.4514 | 5.2041 | 34.8890 | 33.4562 | 0.0244 | 0.02054 | 15.63% |
| stage_a | 1280 ms | 10.4511 | 5.2056 | 34.8963 | 33.4602 | 0.0244 | 0.02011 | 16.70% |
| stage_a | offline | 10.4485 | 5.2121 | 34.8454 | 33.4048 | 0.0244 | 0.01283 | 22.96% |
| stage_b | 320 ms | 9.7669 | 5.1790 | 20.8738 | 20.2962 | 0.0244 | 0.02145 | 12.02% |
| stage_b | 640 ms | 9.7710 | 5.1839 | 20.8801 | 20.3095 | 0.0244 | 0.02139 | 14.07% |
| stage_b | 960 ms | 9.7736 | 5.1831 | 20.8866 | 20.3184 | 0.0244 | 0.02091 | 15.50% |
| stage_b | 1280 ms | 9.7723 | 5.1857 | 20.8990 | 20.3254 | 0.0244 | 0.02046 | 16.31% |
| stage_b | offline | 9.7672 | 5.1907 | 20.8034 | 20.2192 | 0.0244 | 0.01315 | 22.32% |

## 2. Stage B 相对 Stage A 的变化

loss 的负数表示 Stage B 改善；teacher agreement 的正数表示改善。

| chunk | Δ BiCodec | Δ AR | Δ ASR | Δ NAR | Δ commitment | Δ agreement (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 320 | -0.6711 | -0.0216 | -14.0083 | -13.1438 | +0.00038 | -0.20 |
| 640 | -0.6750 | -0.0216 | -14.0052 | -13.1404 | +0.00037 | -0.35 |
| 960 | -0.6779 | -0.0210 | -14.0023 | -13.1378 | +0.00037 | -0.13 |
| 1280 | -0.6788 | -0.0199 | -13.9973 | -13.1349 | +0.00036 | -0.39 |
| offline | -0.6813 | -0.0214 | -14.0420 | -13.1856 | +0.00032 | -0.64 |

## 3. 自动诊断

- ASR CTC: Stage B 在 5/5 个 chunk 上改善。
- NAR S2TT CTC: Stage B 在 5/5 个 chunk 上改善。
- AR S2TT: Stage B 在 5/5 个 chunk 上保持或改善。
- ASR CTC 五点平均相对改善：`40.17%`。
- NAR S2TT CTC 五点平均相对改善：`39.32%`。
- Teacher agreement: Stage B 仅在 0/5 个 chunk 上改善，平均变化 `-0.34` 个百分点。
- Stage B 最大 commitment: `0.02145`，绝对安全阈值为 `0.10`。
- 结论：Stage B 通过数值稳定、CTC 学习和 AR loss 保持门，但没有通过 teacher agreement 改善门；不能把本轮表述为 semantic-code agreement 已修复。
- 下一质量门：使用固定 operating point 做 Phase3 old-protocol replay、端到端文本/语音生成、BLEU/ASR-BLEU、speaker/AutoPCP/SLC 与真实 latency 评估。
