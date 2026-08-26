# Stage A 质量优先 SFT / GRPO 四组完整对照实验报告

## 1. 结论摘要

固定质量优先排序选择的最佳实验为 **a3_g8_full_recovery1**。排序先比较结构错误、非静音率和 source EOS 前语义输出，再比较相对 Stage A 的配对质量，最后才比较首语义时延；训练过程中没有使用该排序提前停止。

**直接回答：A3（GRPO G8）相对 Stage A 在本次冻结 validation64/E2E16 协议下有效，也优于 matched SFT A1 的质量优先综合排序，但不是所有指标全面支配。** A3 相对 A1 保持相同 98 个结构错误、16/16 非静音和 640ms 首语义 p50，同时 free-source MT、quality retention、AL/DAL 和实现 RTF 更好；代价是 gold-source BLEU 略低。A3 相对 Stage A 的结构错误由 100 降至 98，free-source MT 明显提升且 AL 更低；代价是 DAL 与当前未优化 wall-clock RTF 变差。

本实验回答两个问题：第一，继续 SFT 或 GRPO 是否能相对不可变 Stage A `iter_0000381` 改善 incremental MT / semantic TTS / WAIT-WRITE；第二，GRPO 是否优于同训练预算的 matched continued SFT。所有结论均来自相同 checkpoint 初始化、相同 15-shard 全局 shuffle、相同 2-GPU/arm 预算和相同固定评估样本。

## 2. 训练设置与完整性

| arm | 方法 | steps | NaN / skipped | GPU util mean / p95 | power mean / p95 / max | max memory |
|---|---|---:|---:|---:|---:|---:|
| a1_sft_full_recovery1 | matched continued SFT | 2510/2510 | 0/0 | 37.45%/100.00% | 264.0/297.3/349.9 W | 56400 MiB |
| a2_g4_full_recovery1 | GRPO G4 | 2510/2510 | 0/0 | 44.86%/100.00% | 290.9/378.6/415.7 W | 62816 MiB |
| a3_g8_full_recovery1 | GRPO G8 | 2510/2510 | 0/0 | 44.32%/100.00% | 302.2/398.2/438.4 W | 62816 MiB |
| a4_g8_seed2_full_recovery1 | GRPO G8 seed-2 + stronger anchor | 2510/2510 | 0/0 | 44.56%/100.00% | 288.5/375.7/411.1 W | 62818 MiB |

共同训练几何：Megatron 单机 2 GPU/arm；4 arm 并发使用 8×H200；GBS=16、MBS=1、sequence length=18,000、2,510 updates、40,150 packs 一次严格全局 shuffle coverage。A2–A4 前 256 updates 为共同 SFT bootstrap，之后 GRPO reference 与 group reward 激活。

功率不以无关 synthetic kernel 填充；上表只报告真实训练监控，因此 H200 未达到 700W 并不代表程序空闲。变长 18k pack、同步、checkpoint 和 GRPO reference forward 会造成 utility/power 波动。

## 3. 固定 64 条 validation 定量评估

评估集按方向和 short/medium/long 时长分层冻结为 64 条，其中 16 条运行完整 E2E S2S 与音频解码。ASR route 固定禁用 adapter；MT、semantic TTS 与 control route 启用 adapter。每个 candidate 同时在相同 worker 内计算 Stage A adapter-off 配对基线。

### 3.1 ASR 与 MT

| arm | 中文 CER | 英文 WER | gold cmn→eng BLEU/chrF | gold eng→cmn BLEU/chrF | free cmn→eng BLEU/chrF | free eng→cmn BLEU/chrF |
|---|---:|---:|---:|---:|---:|---:|
| a1_sft_full_recovery1 | 19.08% | 36.18% | 0.16/8.45 | 0.47/6.36 | 0.13/7.50 | 0.26/3.96 |
| a2_g4_full_recovery1 | 19.08% | 36.18% | 0.18/8.85 | 0.53/7.09 | 0.11/7.42 | 0.40/4.31 |
| a3_g8_full_recovery1 | 19.08% | 36.18% | 0.16/8.68 | 0.41/6.29 | 0.11/7.42 | 0.41/4.30 |
| a4_g8_seed2_full_recovery1 | 19.08% | 36.18% | 0.13/8.36 | 0.35/5.58 | 0.13/7.85 | 0.63/4.57 |

ASR 理论上应在四组完全一致，因为 adapter 在 ASR route 关闭；若存在仅为浮点/运行噪声。这里的主要可学习差异是 incremental MT、TTS semantic 与外部 control。

### 3.2 E2E S2S、结构与时延

| arm | semantic coverage mean/min | pre-EOS text | pre-EOS semantic | non-silent | malformed | first text p50 | first semantic p50 | AL / DAL | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a1_sft_full_recovery1 | 1.000/1.000 | 100.00% | 100.00% | 100.00% | 98 | 640.0 ms | 640.0 ms | 408.6/975.3 ms | 18.899 |
| a2_g4_full_recovery1 | 1.000/1.000 | 100.00% | 100.00% | 100.00% | 110 | 640.0 ms | 640.0 ms | 352.4/928.6 ms | 20.467 |
| a3_g8_full_recovery1 | 1.000/1.000 | 100.00% | 100.00% | 100.00% | 98 | 640.0 ms | 640.0 ms | 366.8/938.4 ms | 18.539 |
| a4_g8_seed2_full_recovery1 | 1.000/1.000 | 100.00% | 100.00% | 93.75% | 106 | 640.0 ms | 640.0 ms | 381.5/919.6 ms | 19.940 |

### 3.3 相对 Stage A 的配对结论

| arm | quality retention mean | non-silent | pre-EOS semantic | structure errors candidate/Stage A | first semantic Δ | 判定 |
|---|---:|---:|---:|---:|---:|---|
| a1_sft_full_recovery1 | 1.3763 | 100.00% | 100.00% | 98/100 | 0.0 ms | 质量优先口径相对 Stage A 有效 |
| a2_g4_full_recovery1 | 1.5830 | 100.00% | 100.00% | 110/100 | 0.0 ms | 未证明质量优先有效 |
| a3_g8_full_recovery1 | 1.5213 | 100.00% | 100.00% | 98/100 | 0.0 ms | 质量优先口径相对 Stage A 有效 |
| a4_g8_seed2_full_recovery1 | 1.7744 | 93.75% | 100.00% | 106/100 | 0.0 ms | 未证明质量优先有效 |

GRPO 是否优于 matched SFT 必须直接比较 A2–A4 与 A1，而不是只看各自相对 Stage A。若 GRPO 的 quality retention、结构健康度或首语义时延没有同时优于 A1，则只能说明 GRPO reward 在训练内有效激活，不能说明其外部性能优于 SFT。

### 3.4 最佳 A3 与 A1 / Stage A 的直接差值

| comparison | gold BLEU | gold chrF | free BLEU | free chrF | structure errors | non-silent | first semantic p50 | AL | DAL | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 − A1 | -0.0298 | 0.0854 | 0.0659 | 0.1283 | +0 | 0.0000 | 0.0 ms | -41.8 ms | -36.9 ms | -0.360 |
| A3 − Stage A | 0.0454 | -0.0983 | 0.1807 | 1.0031 | -2 | 0.0000 | 0.0 ms | -25.2 ms | 54.6 ms | 3.871 |

负的 AL/DAL/RTF 差值代表更低；正的 BLEU/chrF 代表更高。A3 相对 A1 的主要收益来自 free-running MT 和 lagging，gold BLEU 存在小幅回退，因此结论是综合有效而非全面占优。

## 4. 短音频 160/320/640/1280ms 可试听结果

| arm | chunk | ASR WER/CER | 首音频 p50 | pre-final 发声 | WAV 健康 | RTF |
|---|---:|---:|---:|---:|---:|---:|
| a1_sft_full_recovery1 | 160 ms | 41.20% | 2640.0 ms | 75.00% | 100.00% | 9.392 |
| a1_sft_full_recovery1 | 320 ms | 33.33% | 3200.0 ms | 87.50% | 100.00% | 5.312 |
| a1_sft_full_recovery1 | 640 ms | 35.65% | 3840.0 ms | 87.50% | 100.00% | 3.641 |
| a1_sft_full_recovery1 | 1280 ms | 40.28% | 5480.0 ms | 50.00% | 100.00% | 2.923 |
| a2_g4_full_recovery1 | 160 ms | 41.20% | 2640.0 ms | 75.00% | 100.00% | 9.023 |
| a2_g4_full_recovery1 | 320 ms | 33.33% | 3200.0 ms | 75.00% | 100.00% | 4.932 |
| a2_g4_full_recovery1 | 640 ms | 35.65% | 3840.0 ms | 87.50% | 100.00% | 3.755 |
| a2_g4_full_recovery1 | 1280 ms | 40.28% | 5480.0 ms | 50.00% | 100.00% | 2.909 |
| a3_g8_full_recovery1 | 160 ms | 41.20% | 2640.0 ms | 75.00% | 100.00% | 9.114 |
| a3_g8_full_recovery1 | 320 ms | 33.33% | 3200.0 ms | 87.50% | 100.00% | 5.124 |
| a3_g8_full_recovery1 | 640 ms | 35.65% | 3840.0 ms | 87.50% | 100.00% | 3.600 |
| a3_g8_full_recovery1 | 1280 ms | 40.28% | 5480.0 ms | 50.00% | 100.00% | 3.001 |
| a4_g8_seed2_full_recovery1 | 160 ms | 41.20% | 2640.0 ms | 75.00% | 100.00% | 9.247 |
| a4_g8_seed2_full_recovery1 | 320 ms | 33.33% | 3200.0 ms | 87.50% | 100.00% | 5.179 |
| a4_g8_seed2_full_recovery1 | 640 ms | 35.65% | 3840.0 ms | 87.50% | 100.00% | 3.683 |
| a4_g8_seed2_full_recovery1 | 1280 ms | 40.28% | 5480.0 ms | 50.00% | 100.00% | 2.865 |

该表的首音频时延是相对该条源音频起点的 source-availability 时刻；RTF 还包含当前 Python 自回归实现，不能等同于优化后的线上服务吞吐。

## 5. 四条中英文 60 秒严格因果前缀

| arm | chunk | ASR WER/CER | 首音频 p50 | pre-final 发声 | WAV 健康 | RTF |
|---|---:|---:|---:|---:|---:|---:|
| a1_sft_full_recovery1 | 160 ms | — | 7600.0 ms | 100.00% | 100.00% | 24.107 |
| a1_sft_full_recovery1 | 320 ms | — | 5440.0 ms | 75.00% | 100.00% | 12.168 |
| a1_sft_full_recovery1 | 640 ms | — | 11840.0 ms | 75.00% | 100.00% | 6.106 |
| a1_sft_full_recovery1 | 1280 ms | — | 8960.0 ms | 100.00% | 100.00% | 3.383 |
| a2_g4_full_recovery1 | 160 ms | — | 7600.0 ms | 100.00% | 100.00% | 24.089 |
| a2_g4_full_recovery1 | 320 ms | — | 5440.0 ms | 75.00% | 100.00% | 11.967 |
| a2_g4_full_recovery1 | 640 ms | — | 11840.0 ms | 75.00% | 100.00% | 5.864 |
| a2_g4_full_recovery1 | 1280 ms | — | 8320.0 ms | 100.00% | 100.00% | 3.530 |
| a3_g8_full_recovery1 | 160 ms | — | 7600.0 ms | 100.00% | 100.00% | 23.756 |
| a3_g8_full_recovery1 | 320 ms | — | 5440.0 ms | 75.00% | 100.00% | 11.903 |
| a3_g8_full_recovery1 | 640 ms | — | 11840.0 ms | 75.00% | 100.00% | 5.981 |
| a3_g8_full_recovery1 | 1280 ms | — | 8320.0 ms | 100.00% | 100.00% | 3.435 |
| a4_g8_seed2_full_recovery1 | 160 ms | — | 7600.0 ms | 100.00% | 100.00% | 22.971 |
| a4_g8_seed2_full_recovery1 | 320 ms | — | 5440.0 ms | 75.00% | 100.00% | 11.857 |
| a4_g8_seed2_full_recovery1 | 640 ms | — | 11840.0 ms | 75.00% | 100.00% | 5.875 |
| a4_g8_seed2_full_recovery1 | 1280 ms | — | 8320.0 ms | 100.00% | 100.00% | 3.367 |

该表的首音频时延是相对该条源音频起点的 source-availability 时刻；RTF 还包含当前 Python 自回归实现，不能等同于优化后的线上服务吞吐。

## 6. 完整 5–7 分钟有界滑窗

| model | audio | source | plan | windows | silent windows | first audio | RTF | max internal silence | stereo |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| a3_g8_full_recovery1 | long_en_helen_keller_full | 351.3s | silence_seeking | 15 | 0 | 21420.0ms | 5.724 | 38300.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/a3_g8_full_recovery1/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav` |
| a3_g8_full_recovery1 | long_en_shimon_peres_full | 415.3s | equal_partition_relaxed_minimum | 14 | 0 | 17280.0ms | 3.952 | 51400.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/a3_g8_full_recovery1/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav` |
| a3_g8_full_recovery1 | long_zh_singapore_vietnam_full | 413.4s | silence_seeking | 17 | 0 | 5120.0ms | 4.727 | 39100.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/a3_g8_full_recovery1/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav` |
| a3_g8_full_recovery1 | long_zh_zhangheqiao_full | 352.6s | silence_seeking | 14 | 0 | 6400.0ms | 5.593 | 44600.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/a3_g8_full_recovery1/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav` |
| Stage A | long_en_helen_keller_full | 351.3s | silence_seeking | 15 | 0 | 21420.0ms | 5.264 | 39800.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/stage_a_iter381/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav` |
| Stage A | long_en_shimon_peres_full | 415.3s | equal_partition_relaxed_minimum | 14 | 0 | 17280.0ms | 3.778 | 51600.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/stage_a_iter381/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav` |
| Stage A | long_zh_singapore_vietnam_full | 413.4s | silence_seeking | 17 | 1 | 25720.0ms | 4.381 | 44900.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery2/stage_a_iter381/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav` |
| Stage A | long_zh_zhangheqiao_full | 352.6s | silence_seeking | 14 | 0 | 6400.0ms | 5.070 | 45200.0ms | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/stage_a_iter381/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav` |

完整长音频模式优先使用 18–30 秒静音边界窗口；当录音长度在数学上无法同时满足最小/最大窗口约束时，使用不超过 30 秒的等分兜底。窗口内部遵守 640ms PCM 逐块可见性，但窗口间重置模型状态；因此它是 bounded-window pseudo-streaming，不是因果 encoder/KV cache 的严格长时 streaming。60 秒前缀表才用于严格因果长前缀判断。

## 7. 音频与报告路径

- 定量评估根目录：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/routed64_e2e16`
- 短音频根目录：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/short_audio_multichunk`
- 60 秒严格前缀根目录：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/long_audio4_prefix60_multichunk`
- 最佳 arm 完整长音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery1/a3_g8_full_recovery1`
- Stage A 完整长音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery2/stage_a_iter381_merged`

每个短/前缀样本目录均包含 `source.wav`、`translation_continuous.wav`、`translation_timeline.wav`、`stereo_left_source_right_translation.wav` 与逐 event JSON；立体声左声道是源语音，右声道是翻译语音。

## 8. 方法边界与限制

1. 训练 route mask 使用 gold next-token loss family；自由运行无法提前知道下一 token 的 oracle family，因此评估采用确定性状态机近似：ASR 内 adapter off，MT/TTS/control on。
2. 当前 GRPO 是 utterance-level grouped token/action surrogate，不是逐事件真实音频 rollout 的 on-policy GRPO。reward 有方差、KL 和 policy update 不为零能证明优化路径生效，但最终是否有效只由外部评估决定。
3. 64 条 validation 是冻结的配对对照集，不等同于 CVSS-T 或全量 UniST test；四条外部长音频没有参考译文，因此只报告运行、时延、空白与可试听音频，不报告 BLEU。
4. 独立 TTS segment 使用固定 32-token speaker condition，condition 本身不变化；这减少显式音色漂移来源，但不等价于客观 speaker-similarity 指标。

## 9. 最终回答

本次固定质量优先选择为 **a3_g8_full_recovery1**。在冻结 validation64/E2E16 上，A3 相对 Stage A 达到质量优先有效，且综合排序优于 A1 matched SFT；但 gold-source BLEU 与当前实现吞吐存在 trade-off，因此不能表述为全面支配，也不能仅凭训练 reward 上升或 loss 下降下结论。
