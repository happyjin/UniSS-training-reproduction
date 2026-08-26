# Phase A 长 episode A/B/C/D 故障归因

## 归因协议

本报告只在带 teacher transcription/translation 的 valid 长 episode 上计算质量，避免给四条外部无人工参考音频伪造 WER/BLEU。A/B/C 使用相同 Phase A `iter_0000381` 与同一 speaker/BiCodec 条件；D 使用真实自由运行 Runtime v2 的固定 group-0 候选。

- A：完整 episode 一次性 full-context ASR，对 teacher transcription 计算相似度。
- B：输入 gold source text 做 MT，对 teacher translation 计算 chrF。
- C：输入 gold target text 分短语做 semantic TTS + BiCodec，统计健康发音覆盖。
- D：stateful Runtime v2 自由运行 ASR→incremental MT→ACK TTS，所有上游误差会级联。

## 汇总结论

| 路由 | 指标 | 结果 |
|---|---|---:|
| A offline/full-context ASR | teacher similarity | 0.2813 |
| D streaming ASR | teacher similarity | 0.3207 |
| A→D ASR 退化 | similarity 差值 | -0.0393 |
| B gold-source MT | chrF/100 | 0.3781 |
| D free-running MT | chrF/100 | 0.1939 |
| B→D MT 级联退化 | chrF/100 差值 | +0.1841 |
| C gold-target TTS | 健康短语覆盖 | 0.9952 |
| D runtime TTS | 健康音频覆盖 | 0.9803 |
| D runtime | 已发音文本比例 | 0.9929 |
| D runtime | mean first WRITE | 21080.0 ms |

如果 A 明显好于 D 的 ASR，主因是流式声学/长会话 ASR；如果 B 明显好于 D 的 MT，主因包含 ASR 误差传播和 incremental MT；如果 C 接近 1 而 D 发音覆盖低，则 TTS 本体可用但输入片段、END 或队列状态有问题。Runtime v1/v2 的窗口重置差异另由四条外部长音频 Stage-1 报告量化。

## 逐 episode 结果与试听

| episode | 方向 | A ASR | B MT chrF | C TTS覆盖 | D ASR | D MT | D首WRITE | D已发音 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| episode_000000_cmn_eng | cmn->eng | 0.3730 | 54.53 | 0.9231 | 0.4662 | 38.14 | 3840 ms | 0.8866 |
| episode_000001_eng_cmn | eng->cmn | 0.3418 | 36.90 | 1.0000 | 0.2785 | 16.43 | 49920 ms | 1.0000 |
| episode_000002_cmn_eng | cmn->eng | 0.2219 | 52.09 | 1.0000 | 0.3562 | 31.77 | 8960 ms | 1.0000 |
| episode_000003_eng_cmn | eng->cmn | 0.2629 | 40.79 | 1.0000 | 0.2113 | 14.80 | 29440 ms | 1.0000 |
| episode_000004_cmn_eng | cmn->eng | 0.2840 | 51.73 | 1.0000 | 0.3086 | 26.39 | 30720 ms | 1.0000 |
| episode_000005_eng_cmn | eng->cmn | 0.2709 | 32.89 | 1.0000 | 0.3399 | 19.15 | 5120 ms | 1.0000 |
| episode_000006_cmn_eng | cmn->eng | 0.2721 | 50.07 | 1.0000 | 0.2574 | 18.69 | 7040 ms | 1.0000 |
| episode_000007_eng_cmn | eng->cmn | 0.3419 | 29.81 | 1.0000 | 0.5677 | 20.92 | 10240 ms | 1.0000 |
| episode_000008_cmn_eng | cmn->eng | 0.2618 | 50.31 | 1.0000 | 0.4416 | 37.05 | 29440 ms | 1.0000 |
| episode_000009_eng_cmn | eng->cmn | 0.2824 | 29.82 | 1.0000 | 0.2290 | 7.25 | 53120 ms | 1.0000 |
| episode_000010_cmn_eng | cmn->eng | 0.3389 | 40.68 | 1.0000 | 0.4799 | 27.68 | 5760 ms | 1.0000 |
| episode_000011_eng_cmn | eng->cmn | 0.3333 | 33.20 | 1.0000 | 0.1321 | 8.65 | 53760 ms | 1.0000 |
| episode_000012_cmn_eng | cmn->eng | 0.2713 | 43.76 | 1.0000 | 0.3140 | 25.24 | 29440 ms | 1.0000 |
| episode_000013_eng_cmn | eng->cmn | 0.3684 | 30.35 | 1.0000 | 0.2281 | 14.08 | 5760 ms | 1.0000 |
| episode_000014_cmn_eng | cmn->eng | 0.0000 | 0.00 | 1.0000 | 0.3956 | 0.00 | 7680 ms | 1.0000 |
| episode_000015_eng_cmn | eng->cmn | 0.2765 | 27.96 | 1.0000 | 0.1244 | 4.08 | 7040 ms | 1.0000 |

## C 路由 gold-target TTS 试听

### episode_000000_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_0/episode_000000_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_0/episode_000000_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：0.9231

### episode_000001_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_1/episode_000001_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_1/episode_000001_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000002_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_2/episode_000002_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_2/episode_000002_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000003_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_3/episode_000003_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_3/episode_000003_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000004_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_4/episode_000004_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_4/episode_000004_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000005_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_5/episode_000005_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_5/episode_000005_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000006_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_6/episode_000006_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_6/episode_000006_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000007_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_7/episode_000007_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_7/episode_000007_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000008_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_0/episode_000008_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_0/episode_000008_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000009_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_1/episode_000009_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_1/episode_000009_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000010_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_2/episode_000010_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_2/episode_000010_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000011_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_3/episode_000011_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_3/episode_000011_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000012_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_4/episode_000012_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_4/episode_000012_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000013_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_5/episode_000013_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_5/episode_000013_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000014_cmn_eng / cmn->eng

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_6/episode_000014_cmn_eng/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_6/episode_000014_cmn_eng/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

### episode_000015_eng_cmn / eng->cmn

- 左源右 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_7/episode_000015_eng_cmn/stereo_left_source_right_gold_target_tts.wav`
- 连续 gold-target TTS：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/workers/worker_7/episode_000015_eng_cmn/gold_target_phasea_tts.wav`
- 健康短语覆盖：1.0000

## D 路由自由运行试听

D 路由的连续、全局时间轴和左源右译文件保存在 formal valid rollout 报告中；本 JSON 直接保留每条 group-0 的绝对路径和完整 observation/reward，便于后续训练前后做完全相同协议对照。
