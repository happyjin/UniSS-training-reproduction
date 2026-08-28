# Phase A 事件约束长 Episode GRPO：64×4 统一评估

## 1. 结论边界

本报告严格复用同一批 64 条双向长 episode（中→英 32、英→中 32），每个 policy 生成 4 个候选，共 256 candidates。结果只说明 train-seen 方法有效性，不证明 validation 或外部泛化。`first WRITE` 是源音频时间轴上的决策时延，不是 wall-clock 服务时延；当前没有 LLM KV cache，且 TTS 同步执行，因此不能据此宣称真实 wall-clock 低于 1 秒。

`all 256` 衡量随机采样 policy 的总体行为；`best-of-4` 是每条 episode 按同一 reward 选出的试听上界，不能当成单次部署性能。
历史 baseline 使用旧 reward 定义，而 fresh arms 使用当前带质量保留、连续时延和 failure penalty 的 reward；因此历史行的 reward 只作原始记录，不能与 fresh arms 的 reward 数值直接排序。fresh arms 之间使用同一定义，可以横向比较。

## 2. 全部 256 candidates

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ | RTF↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round1 | -4.849 | 0.261 | 0.178 | 0.339 | 2880/8320 | 18793 | 0.360 | 11.840 |
| Round2 | -4.857 | 0.261 | 0.179 | 0.338 | 3200/7360 | 18762 | 0.357 | 11.882 |
| Round3 | -4.924 | 0.261 | 0.178 | 0.351 | 3200/9920 | 18625 | 0.369 | 12.035 |

## 3. 分方向全部 candidates

| Policy | 方向 | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Round1 | cmn->eng | 0.316 | 0.249 | 0.401 | 2880/8320 | 17981 | 0.399 |
| Round1 | eng->cmn | 0.206 | 0.108 | 0.278 | 3200/8320 | 19605 | 0.320 |
| Round2 | cmn->eng | 0.316 | 0.247 | 0.394 | 3200/7040 | 17808 | 0.393 |
| Round2 | eng->cmn | 0.206 | 0.111 | 0.282 | 2880/11520 | 19716 | 0.322 |
| Round3 | cmn->eng | 0.316 | 0.248 | 0.417 | 3200/8640 | 17910 | 0.413 |
| Round3 | eng->cmn | 0.206 | 0.109 | 0.284 | 2880/10560 | 19339 | 0.326 |

## 4. 每条 episode 的 best-of-4

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | pending/TTS失败↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round1 | -3.395 | 0.261 | 0.180 | 0.344 | 2880/6080 | 16650 | 0.1/0.3 |
| Round2 | -3.460 | 0.261 | 0.180 | 0.345 | 2880/5760 | 17603 | 0.1/0.3 |
| Round3 | -3.256 | 0.261 | 0.182 | 0.346 | 3200/9280 | 18073 | 0.1/0.3 |

## 5. 每轮双向最佳试听样本

### Round1

- `episode_000028_cmn_eng` / group 0 / cmn->eng：reward=1.600，first WRITE=2560 ms，MT=0.353。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000024_cmn_eng` / group 0 / cmn->eng：reward=1.205，first WRITE=3200 ms，MT=0.272。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000024_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_3/audio/episode_000024_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_3/audio/episode_000024_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_3/audio/episode_000024_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000018_cmn_eng` / group 0 / cmn->eng：reward=1.180，first WRITE=2240 ms，MT=0.375。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000032_cmn_eng` / group 1 / cmn->eng：reward=0.779，first WRITE=2560 ms，MT=0.278。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000032_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 0 / eng->cmn：reward=0.351，first WRITE=2560 ms，MT=0.075。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 2 / eng->cmn：reward=0.341，first WRITE=1920 ms，MT=0.233。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 0 / eng->cmn：reward=0.019，first WRITE=2560 ms，MT=0.248。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000015_eng_cmn` / group 1 / eng->cmn：reward=-0.715，first WRITE=3840 ms，MT=0.058。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000015_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_57/audio/episode_000015_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_57/audio/episode_000015_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round1_g4_w64_formal_v1/workers/worker_57/audio/episode_000015_eng_cmn_g1/stereo_left_source_right_translation.wav`

### Round2

- `episode_000018_cmn_eng` / group 0 / cmn->eng：reward=1.392，first WRITE=2240 ms，MT=0.373。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000046_cmn_eng` / group 2 / cmn->eng：reward=1.314，first WRITE=2240 ms，MT=0.369。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000046_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 1 / cmn->eng：reward=1.027，first WRITE=2560 ms，MT=0.368。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000008_cmn_eng` / group 0 / cmn->eng：reward=0.886，first WRITE=2240 ms，MT=0.280。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000008_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 1 / eng->cmn：reward=0.598，first WRITE=1920 ms，MT=0.309。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 3 / eng->cmn：reward=0.253，first WRITE=2560 ms，MT=0.071。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 1 / eng->cmn：reward=0.062，first WRITE=2560 ms，MT=0.217。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000041_eng_cmn` / group 3 / eng->cmn：reward=-0.675，first WRITE=1920 ms，MT=0.138。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000041_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_13/audio/episode_000041_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_13/audio/episode_000041_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round2_g4_w64_formal_v1/workers/worker_13/audio/episode_000041_eng_cmn_g3/stereo_left_source_right_translation.wav`

### Round3

- `episode_000018_cmn_eng` / group 1 / cmn->eng：reward=1.105，first WRITE=2240 ms，MT=0.368。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_18/audio/episode_000018_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000008_cmn_eng` / group 2 / cmn->eng：reward=1.097，first WRITE=3200 ms，MT=0.272。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000008_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_1/audio/episode_000008_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000032_cmn_eng` / group 0 / cmn->eng：reward=0.855，first WRITE=2560 ms，MT=0.323。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000032_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_4/audio/episode_000032_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 0 / cmn->eng：reward=0.775，first WRITE=2560 ms，MT=0.375。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 3 / eng->cmn：reward=0.388，first WRITE=2560 ms，MT=0.065。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 1 / eng->cmn：reward=0.160，first WRITE=1920 ms，MT=0.349。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000051_eng_cmn` / group 0 / eng->cmn：reward=-0.014，first WRITE=8000 ms，MT=0.217。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000051_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_30/audio/episode_000051_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_30/audio/episode_000051_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_30/audio/episode_000051_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 0 / eng->cmn：reward=-0.403，first WRITE=2560 ms，MT=0.223。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_event_constrained_grpo_long_v2/fresh_round3_g4_w64_formal_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/stereo_left_source_right_translation.wav`

## 6. 训练终点 validation

- Round1 / iter 142：total=0.028986，policy=0.027924，KL=0.035398，control clipping=0.000992；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round1_g4_w64_formal_v1.log`。
- Round2 / iter 142：total=-0.047514，policy=-0.047568，KL=0.001811，control clipping=0.000992；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round2_g4_w64_formal_v1.log`。
- Round3 / iter 142：total=0.052486，policy=0.052462，KL=0.000801，control clipping=0.000992；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round3_g4_w64_formal_v1.log`。

## 7. 选择原则

只在 ASR/MT 相似度、文本与译音完整度、音频健康不明显下降时，才把更早 first WRITE、更短内部静音视为有效提升。训练 validation loss 只用于检查优化稳定性；最终试听选择必须同时查看自由运行 64×4 指标，不能仅按 loss 最低选 checkpoint。
