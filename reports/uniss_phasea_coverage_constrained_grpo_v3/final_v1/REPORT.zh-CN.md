# Phase A 内容覆盖约束长 Episode GRPO：64×4 统一评估

## 1. 结论边界

本报告严格复用同一批 64 条双向长 episode（中→英 32、英→中 32），每个 policy 生成 4 个候选，共 256 candidates。结果只说明 train-seen 方法有效性，不证明 validation 或外部泛化。`target coverage` 是生成译文以单调 token 匹配覆盖冻结 teacher target 的比例；`spoken target coverage` 是健康 TTS emission 覆盖该 teacher target 的比例。`first WRITE` 是源音频时间轴上的决策时延，不是 wall-clock 服务时延；当前没有 LLM KV cache，且 TTS 同步执行，因此不能据此宣称真实 wall-clock 低于 1 秒。

`all 256` 衡量随机采样 policy 的总体行为；`best-of-4` 是每条 episode 按同一 reward 选出的试听上界，不能当成单次部署性能。
历史 baseline 使用旧 reward 定义，而 fresh arms 使用当前带质量保留、连续时延和 failure penalty 的 reward；因此历史行的 reward 只作原始记录，不能与 fresh arms 的 reward 数值直接排序。fresh arms 之间使用同一定义，可以横向比较。

## 2. 全部 256 candidates

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | target/spoken coverage↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | RTF↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HistoricalStateful64x4 | 2.505 | 0.300 | 0.189 | —/— | 0.363 | 8320/67120 | 31075 | 0.336 | 4.634 |
| PostR3Baseline | -5.554 | 0.261 | 0.178 | 0.171/0.170 | 0.339 | 2880/9600 | 18895 | 0.359 | 11.945 |
| CoverageRound2 | -5.605 | 0.261 | 0.179 | 0.171/0.169 | 0.339 | 3200/8000 | 18302 | 0.359 | 11.842 |
| CoverageRound3 | -5.663 | 0.261 | 0.179 | 0.169/0.167 | 0.350 | 3200/9920 | 17974 | 0.368 | 11.857 |
| FinalPostRound3 | -5.851 | 0.261 | 0.178 | 0.170/0.169 | 0.340 | 3200/8320 | 18453 | 0.359 | 11.845 |

## 3. 分方向全部 candidates

| Policy | 方向 | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | 译音覆盖↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| HistoricalStateful64x4 | cmn->eng | 0.374 | 0.271 | 0.431 | 7040/62600 | 35526 | 0.389 |
| HistoricalStateful64x4 | eng->cmn | 0.225 | 0.108 | 0.294 | 11520/67120 | 26623 | 0.283 |
| PostR3Baseline | cmn->eng | 0.316 | 0.247 | 0.396 | 3200/6080 | 17961 | 0.394 |
| PostR3Baseline | eng->cmn | 0.206 | 0.109 | 0.282 | 2880/11840 | 19829 | 0.324 |
| CoverageRound2 | cmn->eng | 0.316 | 0.247 | 0.396 | 3200/8000 | 17369 | 0.398 |
| CoverageRound2 | eng->cmn | 0.206 | 0.111 | 0.283 | 2880/8000 | 19235 | 0.321 |
| CoverageRound3 | cmn->eng | 0.316 | 0.247 | 0.416 | 3200/7360 | 16773 | 0.414 |
| CoverageRound3 | eng->cmn | 0.206 | 0.110 | 0.283 | 2880/10560 | 19175 | 0.322 |
| FinalPostRound3 | cmn->eng | 0.316 | 0.247 | 0.399 | 3200/13120 | 17088 | 0.397 |
| FinalPostRound3 | eng->cmn | 0.206 | 0.109 | 0.281 | 2880/8320 | 19819 | 0.322 |

## 4. 每条 episode 的 best-of-4

| Policy | reward↑ | ASR相似度↑ | MT相似度↑ | 文本完整度↑ | 首次WRITE p50/p95 ms↓ | 最大静音均值 ms↓ | pending/TTS失败↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistoricalStateful64x4 | 2.595 | 0.300 | 0.195 | 0.377 | 7680/68680 | 25936 | 0.0/0.0 |
| PostR3Baseline | -3.656 | 0.261 | 0.183 | 0.340 | 3200/11840 | 19688 | 0.0/0.1 |
| CoverageRound2 | -3.854 | 0.261 | 0.185 | 0.346 | 2880/5760 | 17927 | 0.1/0.4 |
| CoverageRound3 | -3.531 | 0.261 | 0.182 | 0.345 | 3200/9920 | 17747 | 0.0/0.1 |
| FinalPostRound3 | -3.906 | 0.261 | 0.184 | 0.347 | 2880/8320 | 18983 | 0.1/0.2 |

## 5. 每轮双向最佳试听样本

### HistoricalStateful64x4

- `episode_000058_cmn_eng` / group 1 / cmn->eng：reward=3.715，first WRITE=7040 ms，MT=0.460。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000058_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000058_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000046_cmn_eng` / group 2 / cmn->eng：reward=3.501，first WRITE=4480 ms，MT=0.364。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000046_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000046_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000062_cmn_eng` / group 0 / cmn->eng：reward=3.378，first WRITE=34560 ms，MT=0.338。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000062_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000062_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000054_cmn_eng` / group 1 / cmn->eng：reward=3.349，first WRITE=5120 ms，MT=0.364。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000054_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000054_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000054_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000054_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000049_eng_cmn` / group 0 / eng->cmn：reward=3.121，first WRITE=62680 ms，MT=0.165。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000049_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000049_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000049_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000049_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000023_eng_cmn` / group 1 / eng->cmn：reward=2.961，first WRITE=3200 ms，MT=0.197。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000023_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000039_eng_cmn` / group 2 / eng->cmn：reward=2.942，first WRITE=16640 ms，MT=0.200。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000039_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000039_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000039_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000039_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 2 / eng->cmn：reward=2.935，first WRITE=5760 ms，MT=0.233。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000001_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000001_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000001_eng_cmn_g2/stereo_left_source_right_translation.wav`

### PostR3Baseline

- `episode_000058_cmn_eng` / group 0 / cmn->eng：reward=4.167，first WRITE=2880 ms，MT=0.476。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000058_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 0 / cmn->eng：reward=3.234，first WRITE=2560 ms，MT=0.399。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_35/audio/episode_000028_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000018_cmn_eng` / group 2 / cmn->eng：reward=2.511，first WRITE=3200 ms，MT=0.346。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000046_cmn_eng` / group 2 / cmn->eng：reward=2.361，first WRITE=2240 ms，MT=0.360。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000046_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 2 / eng->cmn：reward=5.154，first WRITE=1920 ms，MT=0.315。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 0 / eng->cmn：reward=1.913，first WRITE=2560 ms，MT=0.234。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000025_eng_cmn` / group 3 / eng->cmn：reward=0.112，first WRITE=8000 ms，MT=0.242。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000025_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_11/audio/episode_000025_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_11/audio/episode_000025_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_11/audio/episode_000025_eng_cmn_g3/stereo_left_source_right_translation.wav`
- `episode_000041_eng_cmn` / group 2 / eng->cmn：reward=-0.534，first WRITE=1920 ms，MT=0.132。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000041_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_13/audio/episode_000041_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_13/audio/episode_000041_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/post_round3_checkpoint_g4_w64_v1/workers/worker_13/audio/episode_000041_eng_cmn_g2/stereo_left_source_right_translation.wav`

### CoverageRound2

- `episode_000058_cmn_eng` / group 1 / cmn->eng：reward=4.135，first WRITE=2880 ms，MT=0.466。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000058_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 1 / cmn->eng：reward=3.219，first WRITE=2560 ms，MT=0.364。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_35/audio/episode_000028_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000018_cmn_eng` / group 2 / cmn->eng：reward=2.367，first WRITE=2240 ms，MT=0.352。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000046_cmn_eng` / group 2 / cmn->eng：reward=2.219，first WRITE=2240 ms，MT=0.384。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000046_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_53/audio/episode_000046_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 1 / eng->cmn：reward=4.638，first WRITE=1920 ms，MT=0.292。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 0 / eng->cmn：reward=2.177，first WRITE=2560 ms，MT=0.263。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000025_eng_cmn` / group 1 / eng->cmn：reward=-0.565，first WRITE=5440 ms，MT=0.214。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000025_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 2 / eng->cmn：reward=-0.686，first WRITE=2560 ms，MT=0.071。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_59/audio/episode_000031_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_59/audio/episode_000031_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/workers/worker_59/audio/episode_000031_eng_cmn_g2/stereo_left_source_right_translation.wav`

### CoverageRound3

- `episode_000058_cmn_eng` / group 3 / cmn->eng：reward=4.306，first WRITE=2880 ms，MT=0.467。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000058_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 2 / cmn->eng：reward=2.764，first WRITE=2560 ms，MT=0.365。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000018_cmn_eng` / group 2 / cmn->eng：reward=2.057，first WRITE=2240 ms，MT=0.330。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000006_cmn_eng` / group 0 / cmn->eng：reward=1.392，first WRITE=2880 ms，MT=0.278。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000006_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_48/audio/episode_000006_cmn_eng_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_48/audio/episode_000006_cmn_eng_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_48/audio/episode_000006_cmn_eng_g0/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 1 / eng->cmn：reward=4.977，first WRITE=1920 ms，MT=0.345。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 2 / eng->cmn：reward=1.736，first WRITE=2560 ms，MT=0.232。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000025_eng_cmn` / group 1 / eng->cmn：reward=0.188，first WRITE=9280 ms，MT=0.224。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000025_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g1/stereo_left_source_right_translation.wav`
- `episode_000031_eng_cmn` / group 3 / eng->cmn：reward=-0.893，first WRITE=2560 ms，MT=0.065。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000031_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/workers/worker_59/audio/episode_000031_eng_cmn_g3/stereo_left_source_right_translation.wav`

### FinalPostRound3

- `episode_000058_cmn_eng` / group 1 / cmn->eng：reward=4.021，first WRITE=2880 ms，MT=0.484。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000058_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_23/audio/episode_000058_cmn_eng_g1/stereo_left_source_right_translation.wav`
- `episode_000018_cmn_eng` / group 2 / cmn->eng：reward=2.752，first WRITE=2240 ms，MT=0.373。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000018_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_18/audio/episode_000018_cmn_eng_g2/stereo_left_source_right_translation.wav`
- `episode_000028_cmn_eng` / group 3 / cmn->eng：reward=2.420，first WRITE=2560 ms，MT=0.354。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_35/audio/episode_000028_cmn_eng_g3/stereo_left_source_right_translation.wav`
- `episode_000032_cmn_eng` / group 3 / cmn->eng：reward=1.973，first WRITE=2560 ms，MT=0.326。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000032_cmn_eng.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_4/audio/episode_000032_cmn_eng_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_4/audio/episode_000032_cmn_eng_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_4/audio/episode_000032_cmn_eng_g3/stereo_left_source_right_translation.wav`
- `episode_000037_eng_cmn` / group 2 / eng->cmn：reward=4.481，first WRITE=2240 ms，MT=0.283。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000037_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/stereo_left_source_right_translation.wav`
- `episode_000001_eng_cmn` / group 0 / eng->cmn：reward=1.987，first WRITE=2560 ms，MT=0.236。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000001_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_8/audio/episode_000001_eng_cmn_g0/stereo_left_source_right_translation.wav`
- `episode_000011_eng_cmn` / group 3 / eng->cmn：reward=-0.238，first WRITE=3840 ms，MT=0.096。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000011_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_25/audio/episode_000011_eng_cmn_g3/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_25/audio/episode_000011_eng_cmn_g3/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_25/audio/episode_000011_eng_cmn_g3/stereo_left_source_right_translation.wav`
- `episode_000025_eng_cmn` / group 2 / eng->cmn：reward=-0.548，first WRITE=7680 ms，MT=0.199。
  - 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000025_eng_cmn.wav`
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g2/translation_continuous.wav`
  - 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g2/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/workers/worker_11/audio/episode_000025_eng_cmn_g2/stereo_left_source_right_translation.wav`

## 6. 训练终点 validation

- CoverageRound1 / iter 142：total=-0.052864，policy=-0.052884，KL=0.000674，control clipping=0.000992；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round1_v1.log`。
- CoverageRound2 / iter 142：total=-0.057586，policy=-0.057608，KL=0.000715，control clipping=0.000992；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round2_v1.log`。
- CoverageRound3 / iter 142：total=0.070432，policy=0.070412，KL=0.000656，control clipping=0.000977；日志 `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round3_v1.log`。

## 7. 选择原则

只在 ASR/MT 相似度、文本与译音完整度、音频健康不明显下降时，才把更早 first WRITE、更短内部静音视为有效提升。训练 validation loss 只用于检查优化稳定性；最终试听选择必须同时查看自由运行 64×4 指标，不能仅按 loss 最低选 checkpoint。
