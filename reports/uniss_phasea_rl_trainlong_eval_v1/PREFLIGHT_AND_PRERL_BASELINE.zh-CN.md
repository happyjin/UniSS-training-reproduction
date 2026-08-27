# 8 条 train-seen 长 episode：pre-RL Phase A group-0 临时基线

## 重要边界

本报告从已经完成的正式 train64×group4 rollout 中提取固定 group-0，不需要重新占用 GPU。它能立即确认训练长 episode、reference scorer 和现有 Phase A 问题，但 **不是** 新生成的 Phase A/iter15/iter30/iter45 同 seed 正式对照，不能代替最终报告。

全部 8 条样本都来自 RL 正式训练 rollout，属于 train-seen/in-domain；episode 音频和组成 component 均已确认不与 validation 重叠。

## 临时结果

- 中→英流式 ASR CER=0.763；英→中流式 ASR WER=0.689。
- 中→英 MT BLEU/chrF=2.47/21.44；英→中=6.53/13.69。
- 平均 LCS 文本覆盖=0.181；平均 hypothesis/reference 内容长度比=0.345；平均 4-gram 重复率=0.004。
- 首次发声 p50/p95/max=4800/61152/76160 ms。
- WRITE gap p95/max=35223/68720 ms；最大内部静音 mean/max=46138/66900 ms。
- 译音/源音时长比=0.290；总 WRITE=81；pending/TTS failure=0/0；RTF=4.428。
- continuous/timeline/stereo WAV 健康率=1.000/1.000/1.000。

临时结论：声音文件本身全部健康、TTS 队列也能清空，但 Phase A 在这些约 1 分钟拼接 episode 上的 ASR、翻译内容覆盖、首次 WRITE 和长空白均明显有问题。当前最主要瓶颈不是 WAV 写坏，而是 free-running ASR/增量 MT 的内容错误与过晚/稀疏 WRITE。RL 是否真正修复它，必须等 iter15/30/45 在同一协议上的新结果。

当前单条 chrF 最好的是 `episode_000028_cmn_eng`（31.12），最差的是 `episode_000035_eng_cmn`（5.51）；不能只挑最好样本试听。

## 逐样本指标与试听

| episode | 方向 | 秒 | CER/WER↓ | chrF↑ | LCS覆盖↑ | 首次发声ms↓ | 最大静音ms↓ | WRITE | 译音/源音 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| episode_000006_cmn_eng | cmn→eng | 76.04 | 0.796 | 18.73 | 0.119 | 33280 | 40500 | 7 | 0.187 |
| episode_000028_cmn_eng | cmn→eng | 72.20 | 0.588 | 31.12 | 0.227 | 3840 | 36000 | 13 | 0.385 |
| episode_000004_cmn_eng | cmn→eng | 71.50 | 0.883 | 10.38 | 0.092 | 5120 | 65800 | 6 | 0.150 |
| episode_000002_cmn_eng | cmn→eng | 70.82 | 0.781 | 24.35 | 0.179 | 4480 | 38700 | 10 | 0.306 |
| episode_000007_eng_cmn | eng→cmn | 78.96 | 0.797 | 12.00 | 0.155 | 4480 | 66300 | 9 | 0.239 |
| episode_000033_eng_cmn | eng→cmn | 76.96 | 0.448 | 18.15 | 0.290 | 76160 | 0 | 19 | 0.553 |
| episode_000023_eng_cmn | eng→cmn | 70.88 | 0.606 | 18.16 | 0.313 | 11520 | 54900 | 12 | 0.391 |
| episode_000035_eng_cmn | eng→cmn | 70.88 | 0.899 | 5.51 | 0.074 | 3200 | 66900 | 5 | 0.105 |

### 音频路径

#### episode_000006_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000006_cmn_eng.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000006_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000006_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_6/audio/episode_000006_cmn_eng_g0/stereo_left_source_right_translation.wav`

#### episode_000028_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000028_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000028_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000028_cmn_eng_g0/stereo_left_source_right_translation.wav`

#### episode_000004_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000004_cmn_eng.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_4/audio/episode_000004_cmn_eng_g0/stereo_left_source_right_translation.wav`

#### episode_000002_cmn_eng（cmn→eng）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000002_cmn_eng_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000002_cmn_eng_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_2/audio/episode_000002_cmn_eng_g0/stereo_left_source_right_translation.wav`

#### episode_000007_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000007_eng_cmn.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000007_eng_cmn_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000007_eng_cmn_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000007_eng_cmn_g0/stereo_left_source_right_translation.wav`

#### episode_000033_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000033_eng_cmn.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000033_eng_cmn_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000033_eng_cmn_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_1/audio/episode_000033_eng_cmn_g0/stereo_left_source_right_translation.wav`

#### episode_000023_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000023_eng_cmn.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_7/audio/episode_000023_eng_cmn_g0/stereo_left_source_right_translation.wav`

#### episode_000035_eng_cmn（eng→cmn）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000035_eng_cmn.wav`
- 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_3/audio/episode_000035_eng_cmn_g0/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_3/audio/episode_000035_eng_cmn_g0/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/workers/worker_3/audio/episode_000035_eng_cmn_g0/stereo_left_source_right_translation.wav`

## 后续正式对照

GPU device node 恢复后，`run_all_8gpu.sh` 会在相同 8 条 episode、相同 Runtime v2、640 ms/24 s 配置下依次重跑 Phase A、RL iter15、iter30 和 iter45；最终报告还会把之前反复试听的 Helen Keller、Shimon Peres、新加坡—越南关系和张河桥乡四条外部长音频放在独立章节继续比较。
