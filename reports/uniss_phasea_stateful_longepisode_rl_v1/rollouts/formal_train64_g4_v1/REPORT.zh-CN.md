# formal_train64_g4_v1：真实自由运行长 episode rollout 分析

## 协议

共 64 条 45–90 秒 episode，8 个 GPU worker，group size=4，产生 256 条完整自由运行候选。策略在自己的 ASR/MT/TTS 历史上继续生成，不使用 gold prefix。质量门只记录与排序，不会中断后续打包、训练或评估。

## 汇总

| 指标 | mean | p50 | p95 |
|---|---:|---:|---:|
| episode reward | 2.5048 | 2.4488 | 3.1639 |
| ASR teacher similarity | 0.2998 | 0.2471 | 0.6088 |
| MT teacher chrF/100 | 0.1895 | 0.1813 | 0.3636 |
| first WRITE (ms) | 21945.5 | 8320.0 | 67120.0 |
| 最大内部静音 (ms) | 31074.6 | 33500.0 | 60000.0 |
| 已发音文本比例 | 0.9999 | 1.0000 | 1.0000 |

## 最好候选试听

### episode_000058_cmn_eng / group 1 / cmn->eng

- reward=3.7146；first WRITE=7040 ms；MT chrF/100=0.4597；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/stereo_left_source_right_translation.wav`

### episode_000058_cmn_eng / group 2 / cmn->eng

- reward=3.7054；first WRITE=7680 ms；MT chrF/100=0.4848；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g2/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g2/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g2/stereo_left_source_right_translation.wav`

### episode_000058_cmn_eng / group 0 / cmn->eng

- reward=3.6702；first WRITE=7040 ms；MT chrF/100=0.4693；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g0/stereo_left_source_right_translation.wav`

### episode_000058_cmn_eng / group 3 / cmn->eng

- reward=3.5753；first WRITE=7040 ms；MT chrF/100=0.4596；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g3/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g3/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g3/stereo_left_source_right_translation.wav`

### episode_000046_cmn_eng / group 2 / cmn->eng

- reward=3.5007；first WRITE=4480 ms；MT chrF/100=0.3644；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/stereo_left_source_right_translation.wav`

### episode_000062_cmn_eng / group 0 / cmn->eng

- reward=3.3783；first WRITE=34560 ms；MT chrF/100=0.3380；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/stereo_left_source_right_translation.wav`

### episode_000046_cmn_eng / group 1 / cmn->eng

- reward=3.3570；first WRITE=7040 ms；MT chrF/100=0.3655；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g1/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g1/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g1/stereo_left_source_right_translation.wav`

### episode_000046_cmn_eng / group 3 / cmn->eng

- reward=3.3518；first WRITE=7040 ms；MT chrF/100=0.3474；已发音比例=1.0000。
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g3/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g3/translation_global_timeline.wav`
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g3/stereo_left_source_right_translation.wav`

## 最差候选诊断

### episode_000004_cmn_eng / group 2 / cmn->eng

- reward=1.5832；first WRITE=7040 ms；最大内部静音=60200 ms；MT chrF/100=0.1136；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g2/stereo_left_source_right_translation.wav`

### episode_000059_eng_cmn / group 1 / eng->cmn

- reward=1.6025；first WRITE=5120 ms；最大内部静音=33600 ms；MT chrF/100=0.1049；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_3/audio/episode_000059_eng_cmn_g1/stereo_left_source_right_translation.wav`

### episode_000004_cmn_eng / group 0 / cmn->eng

- reward=1.6034；first WRITE=5120 ms；最大内部静音=65800 ms；MT chrF/100=0.1038；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g0/stereo_left_source_right_translation.wav`

### episode_000059_eng_cmn / group 2 / eng->cmn

- reward=1.6134；first WRITE=5120 ms；最大内部静音=33500 ms；MT chrF/100=0.1127；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_3/audio/episode_000059_eng_cmn_g2/stereo_left_source_right_translation.wav`

### episode_000004_cmn_eng / group 1 / cmn->eng

- reward=1.6265；first WRITE=51200 ms；最大内部静音=18300 ms；MT chrF/100=0.1206；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g1/stereo_left_source_right_translation.wav`

### episode_000045_eng_cmn / group 2 / eng->cmn

- reward=1.7374；first WRITE=30720 ms；最大内部静音=34300 ms；MT chrF/100=0.0452；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_5/audio/episode_000045_eng_cmn_g2/stereo_left_source_right_translation.wav`

### episode_000005_eng_cmn / group 3 / eng->cmn

- reward=1.8132；first WRITE=28160 ms；最大内部静音=33400 ms；MT chrF/100=0.0565；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_5/audio/episode_000005_eng_cmn_g3/stereo_left_source_right_translation.wav`

### episode_000022_cmn_eng / group 3 / cmn->eng

- reward=1.8152；first WRITE=26240 ms；最大内部静音=34800 ms；MT chrF/100=0.0819；已发音比例=1.0000。
- 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000022_cmn_eng_g3/stereo_left_source_right_translation.wav`

## 本阶段用途

这些候选不是最终模型结论，而是带 old-policy log-probability 的训练轨迹。组内优势会偏好质量、完整发音、稳定提交和健康音频，同时保留较弱的首 WRITE/内部静音项；Phase3 replay 与 KL 在下一阶段抑制灾难性遗忘。
