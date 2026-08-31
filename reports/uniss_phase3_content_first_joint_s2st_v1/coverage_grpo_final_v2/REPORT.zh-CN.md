# Phase A 内容覆盖约束长 Episode GRPO：64×4 统一评估

## 1. 结论边界

本报告严格复用同一批 64 条双向长 episode（中→英 32、英→中 32），每个 policy 生成 4 个候选，共 256 candidates。结果只说明 train-seen 方法有效性，不证明 validation 或外部泛化。`target coverage` 是生成译文以单调 token 匹配覆盖冻结 teacher target 的比例；`spoken target coverage` 是健康 TTS emission 覆盖该 teacher target 的比例。`first WRITE` 是源音频时间轴上的决策时延，不是 wall-clock 服务时延；当前没有 LLM KV cache，且 TTS 同步执行，因此不能据此宣称真实 wall-clock 低于 1 秒。

`all 256` 衡量随机采样 policy 的总体行为；`best-of-4` 是每条 episode 按同一 reward 选出的试听上界，不能当成单次部署性能。
历史 baseline 使用旧 reward 定义，而 fresh arms 使用当前带质量保留、连续时延和 failure penalty 的 reward；因此历史行的 reward 只作原始记录，不能与 fresh arms 的 reward 数值直接排序。fresh arms 之间使用同一定义，可以横向比较。

## 2. 全部 256 candidates

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | target/spoken coverage↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | RTF↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre_GRPO | -10.482 | 0.048 | 0.026 | 0.031/0.030 | 0.068 | 11840/64280 | 28465 | 0.088 | 7.798 |
| round1 | -10.376 | 0.048 | 0.029 | 0.031/0.031 | 0.072 | 9600/60480 | 28388 | 0.101 | 7.802 |
| round2 | -10.564 | 0.048 | 0.030 | 0.032/0.032 | 0.074 | 10240/62120 | 27415 | 0.104 | 7.849 |

## 3. 分方向全部 candidates

| Policy | 方向 | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| pre_GRPO | cmn->eng | 0.051 | 0.030 | 0.053 | 10880/64520 | 28597 | 0.067 |
| pre_GRPO | eng->cmn | 0.044 | 0.022 | 0.082 | 12800/54720 | 28333 | 0.109 |
| round1 | cmn->eng | 0.053 | 0.036 | 0.065 | 10560/64300 | 29004 | 0.087 |
| round1 | eng->cmn | 0.044 | 0.022 | 0.079 | 8960/58560 | 27772 | 0.115 |
| round2 | cmn->eng | 0.051 | 0.037 | 0.064 | 9920/64300 | 26862 | 0.091 |
| round2 | eng->cmn | 0.045 | 0.023 | 0.084 | 11520/40640 | 27968 | 0.117 |

## 4. 每条 episode 的 best-of-4

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | pending/TTS失败↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre_GRPO | -8.340 | 0.051 | 0.029 | 0.073 | 14400/64300 | 28061 | 0.0/0.0 |
| round1 | -8.307 | 0.053 | 0.035 | 0.082 | 9920/57600 | 28534 | 0.0/0.0 |
| round2 | -8.583 | 0.053 | 0.036 | 0.084 | 15680/64280 | 25311 | 0.0/0.0 |

## 5. 每轮双向最佳试听样本

### pre_GRPO

- `episode_000022_cmn_eng` / group 0 / cmn->eng：reward=-5.475，first WRITE=31360 ms，MT=0.005。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000022_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_50/audio/episode_000022_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_50/audio/episode_000022_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_50/audio/episode_000022_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000004_cmn_eng` / group 0 / cmn->eng：reward=-6.363，first WRITE=39360 ms，MT=0.032。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000004_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000002_cmn_eng` / group 3 / cmn->eng：reward=-6.477，first WRITE=4800 ms，MT=0.028。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g3/stereo_left_source_right_translation.wav`
- `episode_000008_cmn_eng` / group 0 / cmn->eng：reward=-6.612，first WRITE=10880 ms，MT=0.018。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000008_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000017_eng_cmn` / group 0 / eng->cmn：reward=-2.622，first WRITE=5120 ms，MT=0.083。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000017_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000059_eng_cmn` / group 0 / eng->cmn：reward=-2.969，first WRITE=13760 ms，MT=0.039。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000059_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 0 / eng->cmn：reward=-3.064，first WRITE=6080 ms，MT=0.024。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000009_eng_cmn` / group 1 / eng->cmn：reward=-3.701，first WRITE=40640 ms，MT=0.010。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000009_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_pre_grpo_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/stereo_left_source_right_translation.wav`

### round1

- `episode_000006_cmn_eng` / group 2 / cmn->eng：reward=-5.480，first WRITE=9920 ms，MT=0.051。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000006_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_48/audio/episode_000006_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_48/audio/episode_000006_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_48/audio/episode_000006_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000008_cmn_eng` / group 0 / cmn->eng：reward=-5.811，first WRITE=13760 ms，MT=0.032。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000008_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000002_cmn_eng` / group 2 / cmn->eng：reward=-5.863，first WRITE=3840 ms，MT=0.052。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000060_cmn_eng` / group 3 / cmn->eng：reward=-6.174，first WRITE=15040 ms，MT=0.100。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000060_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_39/audio/episode_000060_cmn_eng_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_39/audio/episode_000060_cmn_eng_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_39/audio/episode_000060_cmn_eng_g3/stereo_left_source_right_translation.wav`
- `episode_000017_eng_cmn` / group 1 / eng->cmn：reward=-2.896，first WRITE=5440 ms，MT=0.072。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000017_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 0 / eng->cmn：reward=-3.094，first WRITE=6400 ms，MT=0.024。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_59/audio/episode_000031_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000059_eng_cmn` / group 1 / eng->cmn：reward=-3.274，first WRITE=32640 ms，MT=0.039。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000059_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000009_eng_cmn` / group 0 / eng->cmn：reward=-3.482，first WRITE=7040 ms，MT=0.016。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000009_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round1_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g0/stereo_left_source_right_translation.wav`

### round2

- `episode_000004_cmn_eng` / group 2 / cmn->eng：reward=-4.170，first WRITE=35520 ms，MT=0.059。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000004_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_32/audio/episode_000004_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000008_cmn_eng` / group 0 / cmn->eng：reward=-5.060，first WRITE=40640 ms，MT=0.057。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000008_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_1/audio/episode_000008_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000048_cmn_eng` / group 0 / cmn->eng：reward=-5.505，first WRITE=15360 ms，MT=0.080。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000048_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_6/audio/episode_000048_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_6/audio/episode_000048_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_6/audio/episode_000048_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000002_cmn_eng` / group 2 / cmn->eng：reward=-5.841，first WRITE=3840 ms，MT=0.062。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_16/audio/episode_000002_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000017_eng_cmn` / group 2 / eng->cmn：reward=-2.828，first WRITE=5120 ms，MT=0.078。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000017_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_10/audio/episode_000017_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000059_eng_cmn` / group 2 / eng->cmn：reward=-2.964，first WRITE=35840 ms，MT=0.042。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000059_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_31/audio/episode_000059_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000009_eng_cmn` / group 1 / eng->cmn：reward=-3.668，first WRITE=7040 ms，MT=0.012。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000009_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_9/audio/episode_000009_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000043_eng_cmn` / group 3 / eng->cmn：reward=-4.090，first WRITE=8320 ms，MT=0.036。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000043_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_29/audio/episode_000043_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_29/audio/episode_000043_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phase3_content_first_joint_s2st_v1/content_first_post_round2_g4_w64_v2/workers/worker_29/audio/episode_000043_eng_cmn_g3/stereo_left_source_right_translation.wav`

## 6. 训练终点 validation

- round1 / iter 99：total=-0.133421，policy=-0.133743，KL=0.010723，control clipping=0.360178；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phase3_content_first_joint_s2st_v1/content_first_coverage_grpo_round1_v2.log`。
- round2 / iter 99：total=0.020294，policy=0.020069，KL=0.007499，control clipping=0.404378；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phase3_content_first_joint_s2st_v1/content_first_coverage_grpo_round2_v2.log`。

## 7. 选择原则

只在 ASR/MT 相似度、文本与译音完整度、音频健康不明显下降时，才把更早 first WRITE、更短内部静音视为有效提升。训练 validation loss 只用于检查优化稳定性；最终试听选择必须同时查看自由运行 64×4 指标，不能仅按 loss 最低选 checkpoint。
