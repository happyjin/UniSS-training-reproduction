# formal_valid16_g4_v1：真实自由运行长 episode rollout 分析

## 协议

共 16 条 45–90 秒 episode，8 个 GPU worker，group size=4，产生 64 条完整自由运行候选。策略在自己的 ASR/MT/TTS 历史上继续生成，不使用 gold prefix。质量门只记录与排序，不会中断后续打包、训练或评估。

## 汇总

| 指标 | mean | p50 | p95 |
|---|---:|---:|---:|
| episode reward | 2.4077 | 2.4156 | 3.1073 |
| ASR teacher similarity | 0.3207 | 0.3140 | 0.5677 |
| MT teacher chrF/100 | 0.1980 | 0.2053 | 0.3724 |
| first WRITE (ms) | 22552.2 | 19200.0 | 53760.0 |
| 最大内部静音 (ms) | 28770.3 | 28200.0 | 51500.0 |
| 已发音文本比例 | 0.9979 | 1.0000 | 1.0000 |

## 最好候选试听

### episode_000010_cmn_eng / group 1 / cmn->eng

- reward=3.3098；first WRITE=5760 ms；MT chrF/100=0.3387；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g1/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g1/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g1/stereo_left_source_right_translation.wav`

### episode_000010_cmn_eng / group 3 / cmn->eng

- reward=3.1529；first WRITE=7680 ms；MT chrF/100=0.3222；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g3/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g3/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g3/stereo_left_source_right_translation.wav`

### episode_000010_cmn_eng / group 2 / cmn->eng

- reward=3.1421；first WRITE=5760 ms；MT chrF/100=0.3088；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g2/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g2/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g2/stereo_left_source_right_translation.wav`

### episode_000010_cmn_eng / group 0 / cmn->eng

- reward=3.1073；first WRITE=5760 ms；MT chrF/100=0.2768；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_2/audio/episode_000010_cmn_eng_g0/stereo_left_source_right_translation.wav`

### episode_000008_cmn_eng / group 3 / cmn->eng

- reward=3.0915；first WRITE=29440 ms；MT chrF/100=0.3600；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g3/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g3/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g3/stereo_left_source_right_translation.wav`

### episode_000008_cmn_eng / group 1 / cmn->eng

- reward=3.0748；first WRITE=29440 ms；MT chrF/100=0.3535；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g1/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g1/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g1/stereo_left_source_right_translation.wav`

### episode_000008_cmn_eng / group 2 / cmn->eng

- reward=3.0283；first WRITE=30720 ms；MT chrF/100=0.3459；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g2/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g2/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g2/stereo_left_source_right_translation.wav`

### episode_000008_cmn_eng / group 0 / cmn->eng

- reward=3.0001；first WRITE=29440 ms；MT chrF/100=0.3705；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000008_cmn_eng_g0/stereo_left_source_right_translation.wav`

## 最差候选诊断

### episode_000009_eng_cmn / group 3 / eng->cmn

- reward=1.5813；first WRITE=19200 ms；最大内部静音=24000 ms；MT chrF/100=0.0855；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_1/audio/episode_000009_eng_cmn_g3/stereo_left_source_right_translation.wav`

### episode_000009_eng_cmn / group 1 / eng->cmn

- reward=1.6488；first WRITE=19200 ms；最大内部静音=22300 ms；MT chrF/100=0.1209；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_1/audio/episode_000009_eng_cmn_g1/stereo_left_source_right_translation.wav`

### episode_000009_eng_cmn / group 0 / eng->cmn

- reward=1.7523；first WRITE=53120 ms；最大内部静音=5700 ms；MT chrF/100=0.0725；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_1/audio/episode_000009_eng_cmn_g0/stereo_left_source_right_translation.wav`

### episode_000000_cmn_eng / group 0 / cmn->eng

- reward=1.7564；first WRITE=3840 ms；最大内部静音=33200 ms；MT chrF/100=0.3814；已发音比例=0.8866。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_0/audio/episode_000000_cmn_eng_g0/stereo_left_source_right_translation.wav`

### episode_000015_eng_cmn / group 2 / eng->cmn

- reward=1.7814；first WRITE=26240 ms；最大内部静音=47800 ms；MT chrF/100=0.0531；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_7/audio/episode_000015_eng_cmn_g2/stereo_left_source_right_translation.wav`

### episode_000009_eng_cmn / group 2 / eng->cmn

- reward=1.7889；first WRITE=52480 ms；最大内部静音=6800 ms；MT chrF/100=0.1066；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_1/audio/episode_000009_eng_cmn_g2/stereo_left_source_right_translation.wav`

### episode_000015_eng_cmn / group 3 / eng->cmn

- reward=1.7932；first WRITE=29440 ms；最大内部静音=41600 ms；MT chrF/100=0.0615；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_7/audio/episode_000015_eng_cmn_g3/stereo_left_source_right_translation.wav`

### episode_000015_eng_cmn / group 0 / eng->cmn

- reward=1.8516；first WRITE=7040 ms；最大内部静音=66600 ms；MT chrF/100=0.0408；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/workers/worker_7/audio/episode_000015_eng_cmn_g0/stereo_left_source_right_translation.wav`

## 本阶段用途

这些候选不是最终模型结论，而是带 old-policy log-probability 的训练轨迹。组内优势会偏好质量、完整发音、稳定提交和健康音频，同时保留较弱的首 WRITE/内部静音项；Phase3 replay 与 KL 在下一阶段抑制灾难性遗忘。
