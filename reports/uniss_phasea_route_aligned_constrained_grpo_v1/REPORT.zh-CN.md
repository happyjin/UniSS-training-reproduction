# Phase A 路由一致约束式 GRPO：训练内长音频评估

## 1. 声明边界

本报告只评估训练流程明确使用的 8 条 train-seen 长 episode，目标是判断小数据条件下方法本身能否改善，不用于宣称 validation 或外部泛化。所有 arm 固定使用 640 ms decision chunk、160 ms 物理声学 block、24 s acoustic ring、同一 speaker token 和 Runtime v2。

本实验与旧 RL 的关键区别是：adapter 在 ASR、MT、semantic TTS 和 control 路由全部启用；reward 只有在 ASR、MT、完整性和音频健康保持时才奖励低 first-WRITE 和较短静音。

## 2. 内容质量

| 系统 | 中→英 CER↓ | 英→中 WER↓ | 中→英 BLEU/chrF↑ | 英→中 BLEU/chrF↑ | LCS覆盖↑ | 4-gram重复↓ |
|---|---:|---:|---:|---:|---:|---:|
| Phase A | 0.763 | 0.689 | 2.36/21.38 | 9.55/17.19 | 0.196 | 0.004 |
| SFT64 | 0.695 | 0.792 | 5.52/27.82 | 3.44/11.58 | 0.193 | 0.002 |
| RL epoch1 | 0.679 | 0.787 | 5.67/28.63 | 3.71/12.23 | 0.199 | 0.006 |
| RL epoch2 | 0.682 | 0.744 | 5.45/27.96 | 5.90/14.82 | 0.208 | 0.010 |
| RL epoch3 | 0.697 | 0.745 | 4.49/26.24 | 5.19/14.66 | 0.201 | 0.003 |

## 3. WRITE、静音和音频

| 系统 | 首次发声p50/p95 ms↓ | WRITE gap p95 ms↓ | 最大静音均值 ms↓ | 音频覆盖↑ | WRITE | pending/TTS失败 | WAV健康 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase A | 4480/12512 | 24064 | 26875 | 0.395 | 86 | 0/0 | 1.000 |
| SFT64 | 4480/20800 | 24960 | 20975 | 0.398 | 89 | 0/0 | 1.000 |
| RL epoch1 | 4480/12064 | 23040 | 20788 | 0.401 | 91 | 0/0 | 1.000 |
| RL epoch2 | 4800/12064 | 25600 | 22988 | 0.454 | 97 | 0/0 | 1.000 |
| RL epoch3 | 4800/12096 | 25600 | 26025 | 0.445 | 93 | 0/0 | 1.000 |

## 4. 相对 Phase A 的自动审计

| 系统 | CER变化↓ | WER变化↓ | 中→英chrF变化↑ | 英→中chrF变化↑ | 覆盖变化↑ | 首次p50变化ms↓ | 最大静音变化ms↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT64 | -0.068 | 0.103 | 6.44 | -5.62 | -0.003 | 0 | -5900 |
| RL epoch1 | -0.083 | 0.098 | 7.25 | -4.96 | 0.003 | 0 | -6088 |
| RL epoch2 | -0.081 | 0.055 | 6.57 | -2.37 | 0.012 | 320 | -3888 |
| RL epoch3 | -0.066 | 0.057 | 4.86 | -2.53 | 0.005 | 320 | -850 |

## 5. 相对 SFT64 的 RL 增益与 checkpoint 选择

| 系统 | ASR error变化↓ | BLEU变化↑ | chrF变化↑ | LCS覆盖变化↑ | 首次p95变化ms↓ | 最大静音变化ms↓ | 音频覆盖变化↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL epoch1 | -0.012 | 0.15 | 0.77 | 0.006 | -8736 | -188 | 0.003 |
| RL epoch2 | -0.026 | -0.07 | 0.94 | 0.015 | -8736 | 2012 | 0.056 |
| RL epoch3 | -0.016 | -1.02 | -0.37 | 0.008 | -8704 | 5050 | 0.047 |

结论：RL 确实改变了真实 ASR/MT/TTS 路由，不再是旧实验中 ASR 输出逐字不变的无效更新。若以本次用户指定的 train-seen 目标优先选择内容完整度和双向折中，推荐 `RL epoch2 / iter_0000082`：它相对 SFT64 取得最低整体 ASR error、最高 LCS 覆盖、最高音频覆盖，并显著恢复英→中 BLEU/chrF。若更强调较短内部静音和中→英质量，则 `RL epoch1 / iter_0000041` 更稳健。`RL epoch3 / iter_0000123` 的 BLEU、chrF 和静音开始回退，不推荐作为部署 checkpoint。

严格边界：相对原始 Phase A，所有新 arm 都改善了中→英 CER/chrF，但英→中 WER 和 chrF 仍未完全恢复，因此没有通过‘双向质量均不退化’的严格门。当前实验只证明约束式 GRPO 能在这 8 条训练样本上学习并部分修复 SFT64，不证明外部泛化，也不代表已经达到低于 1 秒的同传延迟。

### 推荐试听与失败样本

- `episode_000006_cmn_eng`：RL epoch2 是最清楚的中→英正例；相对 Phase A，ASR error、chrF、覆盖和最大静音均改善，但首次发声仍为 14.08 s。
- `episode_000033_eng_cmn`：RL epoch2 相对 SFT64 明显恢复长段英文识别与中文翻译，适合听 RL 的修复作用；但仍弱于原始 Phase A，且内部最大静音达到 38 s。
- `episode_000028_cmn_eng`：三个 RL arm 都较稳定，epoch3 单样本分数最好，但不能据此覆盖其整体过训结论。
- `episode_000004_cmn_eng`：SFT64 已大幅优于 Phase A，继续 RL 后逐 epoch 回落，是过度优化的反例。
- `episode_000035_eng_cmn`：所有 arm 都很差，RL epoch2 仅 5 次 WRITE、音频覆盖 0.148，是当前最明显的失败样本。

## 6. 逐样本试听

### episode_000006_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000006_cmn_eng.wav`
- Phase A：ASR error=0.796，chrF=20.88，覆盖=0.119，首次发声=14080 ms，最大静音=22000 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.677，chrF=29.90，覆盖=0.232，首次发声=27520 ms，最大静音=13500 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.663，chrF=31.90，覆盖=0.226，首次发声=14080 ms，最大静音=14100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.655，chrF=32.80，覆盖=0.237，首次发声=14080 ms，最大静音=12200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.743，chrF=25.43，覆盖=0.181，首次发声=13440 ms，最大静音=21700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`

### episode_000028_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
- Phase A：ASR error=0.588，chrF=30.29，覆盖=0.232，首次发声=5120 ms，最大静音=17800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.612，chrF=29.79，覆盖=0.243，首次发声=5120 ms，最大静音=17600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.564，chrF=31.23，覆盖=0.243，首次发声=5120 ms，最大静音=15800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.564，chrF=30.08，覆盖=0.221，首次发声=5120 ms，最大静音=13700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.488，chrF=33.07，覆盖=0.271，首次发声=5120 ms，最大静音=14900 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`

### episode_000004_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000004_cmn_eng.wav`
- Phase A：ASR error=0.883，chrF=12.02，覆盖=0.125，首次发声=4480 ms，最大静音=40000 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.722，chrF=26.92，覆盖=0.245，首次发声=4480 ms，最大静音=18800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.722，chrF=25.59，覆盖=0.234，首次发声=4480 ms，最大静音=22700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.744，chrF=24.28，覆盖=0.228，首次发声=5120 ms，最大静音=24600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.778，chrF=21.26，覆盖=0.223，首次发声=5120 ms，最大静音=25500 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`

### episode_000002_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
- Phase A：ASR error=0.781，chrF=21.87，覆盖=0.142，首次发声=4480 ms，最大静音=20400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.763，chrF=25.23，覆盖=0.156，首次发声=4480 ms，最大静音=20400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.760，chrF=26.28，覆盖=0.156，首次发声=4480 ms，最大静音=20600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.760，chrF=25.16，覆盖=0.156，首次发声=4480 ms，最大静音=20800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.765，chrF=25.23，覆盖=0.156，首次发声=4480 ms，最大静音=24400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`

### episode_000007_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000007_eng_cmn.wav`
- Phase A：ASR error=0.797，chrF=13.29，覆盖=0.162，首次发声=4480 ms，最大静音=21100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.743，chrF=15.19，覆盖=0.189，首次发声=4480 ms，最大静音=29800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.748，chrF=15.59，覆盖=0.198，首次发声=4480 ms，最大静音=29100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.748，chrF=15.59，覆盖=0.198，首次发声=4480 ms，最大静音=29300 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.757，chrF=17.32，覆盖=0.177，首次发声=4480 ms，最大静音=27100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`

### episode_000033_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000033_eng_cmn.wav`
- Phase A：ASR error=0.448，chrF=28.54，覆盖=0.404，首次发声=9600 ms，最大静音=40400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.797，chrF=9.59，覆盖=0.164，首次发声=8320 ms，最大静音=22500 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.797，chrF=9.54，覆盖=0.173，首次发声=8320 ms，最大静音=23700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.613，chrF=21.00，覆盖=0.302，首次发声=8320 ms，最大静音=38000 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.623，chrF=17.20，覆盖=0.250，首次发声=9600 ms，最大静音=35100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`

### episode_000023_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000023_eng_cmn.wav`
- Phase A：ASR error=0.606，chrF=19.46，覆盖=0.313，首次发声=3200 ms，最大静音=16800 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.706，chrF=16.83，覆盖=0.255，首次发声=3200 ms，最大静音=23700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.694，chrF=18.32，覆盖=0.280，首次发声=3200 ms，最大静音=19900 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.694，chrF=16.84，覆盖=0.258，首次发声=3200 ms，最大静音=23600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.694，chrF=18.32，覆盖=0.280，首次发声=3200 ms，最大静音=19400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`

### episode_000035_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000035_eng_cmn.wav`
- Phase A：ASR error=0.899，chrF=5.12，覆盖=0.074，首次发声=3200 ms，最大静音=36500 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- SFT64：ASR error=0.920，chrF=4.19，覆盖=0.063，首次发声=3200 ms，最大静音=21500 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/sft64_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch1：ASR error=0.905，chrF=5.14，覆盖=0.081，首次发声=3200 ms，最大静音=20400 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch1_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch2：ASR error=0.925，chrF=4.20，覆盖=0.063，首次发声=3200 ms，最大静音=21700 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch2_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL epoch3：ASR error=0.910，chrF=4.64，覆盖=0.074，首次发声=3200 ms，最大静音=40100 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_route_aligned_constrained_grpo_v1/rl_epoch3_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`

## 7. 判定原则

只有当 ASR error 不升、双向 chrF 和文本覆盖不下降、pending/TTS failure 为零时，first-WRITE、WRITE gap 或静音改善才计为有效。训练内提升仅证明当前方法能在给定数据上学到目标，不等于外部泛化。
