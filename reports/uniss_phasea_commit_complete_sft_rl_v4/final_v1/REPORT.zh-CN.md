# Phase-A Commit-Complete SFT + constrained GRPO v4：结题评估

## 结论摘要

本轮 v4 的训练、三轮 constrained GRPO 和训练后独立评估均已完成。最终评估严格使用固定的 64 条双向长 episode（中→英 32 条、英→中 32 条），每条 fresh rollout 4 个候选，共 256 个候选；所有 64 个 worker 正常合并，未出现 OOM、NaN、跳步或 worker 失败。

**结论不是“已经得到完整可用的同声传译 S2ST”。** 系统确实可以在源音频未结束时多次产生目标语音，且最终评估的 256 个候选都生成了三种可播放 WAV（双声道、源时间轴、连续目标语音）。但平均已经实际说出的参考目标内容只有 **16.71%**；256 个候选中只有 **1 个**达到 50%，没有任何候选达到 80% 或完整覆盖。因此这只能证明端到端事件式“读入—识别—增量翻译—提交—合成”链路可以发声，不能证明它完成了长音频的完整 speech-to-speech translation。

本次结论的适用范围是 **train-seen long episodes**，不能外推为未见数据泛化结果。表中的 `first write` 与 `maximum internal silence` 都是**源时间轴**上的时机指标，不能等同于用户实际等待的 wall-clock 推理延迟；当前 LLM 声学提示仍会重算，没有 LLM KV cache，且 TTS 是同步调用，不能据此宣称真实端到端延迟低于 1 秒。

## 已完成的训练与评估

1. 动作 warm-up：基于经审计的 15-shard event cache，完成 45 updates，保存 checkpoint。
2. Commit-aware SFT：完成 3 epochs / 132 updates，checkpoint 为 `commit_sft_v4_formal3e_v1/iter_0000132`。
3. constrained GRPO：三轮均使用 fresh rollout；每轮均为 64 episode × 4 candidates = 256 candidates，随后以该轮 rollout 训练一 epoch。
   - Round 1：142/142 updates；
   - Round 2：141/141 updates；
   - Round 3：142/142 updates，最终 checkpoint 为 `commit_v4_grpo_round3_formal_v1/iter_0000142`。
4. 训练后最终评估：以 Round-3 checkpoint 重新做独立 64×4 fresh rollout（round index 4）。`ROLLOUT_MERGED.json` 记录 `workers=64`、`episodes=64`、`group_size=4`、`status=complete`，并且日志错误检索为 0。

## 指标含义

- **ASR teacher similarity**：流式识别文本与 teacher 转录的相似度；越高越好。
- **MT teacher similarity**：流式翻译文本与 teacher 目标译文的相似度；越高越好。它是本实验的文本相似度，不是可直接与论文 BLEU 混用的 BLEU。
- **target / spoken target coverage**：teacher 目标译文中已经出现在生成目标、以及已经送入并实际说出的目标内容比例。后者是判断“是否完整 S2ST”的关键指标。
- **first write**：第一段目标语音首次出现在源音频时间轴上的位置；越小表示越早发声。
- **maximum internal silence**：目标时间轴中两段已生成目标语音之间的最大空白；越小越连贯。
- **TTS failures / healthy audio fraction**：TTS 片段失败计数与成功音频比例。WAV 文件存在不等于翻译完整；健康度只检查音频是否是有限、非静音的有效信号。

## 训练过程中的演化

| checkpoint 的独立 rollout | MT 相似度 ↑ | spoken 覆盖 ↑ | 首次发声 ↓ | 最大中间空白 ↓ | TTS failures ↓ | 覆盖 ≥30% / 256 | 覆盖 ≥50% / 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1（SFT 后、R1 GRPO 前） | 0.1788 | **0.1694** | **4.306 s** | 18.689 s | **0.883** | 19 | 0 |
| R2（R1 GRPO 后） | 0.1781 | 0.1686 | 4.345 s | **18.179 s** | 0.898 | 19 | 0 |
| R3（R2 GRPO 后） | 0.1766 | 0.1665 | 4.516 s | 18.608 s | 1.188 | 18 | 0 |
| 最终（R3 GRPO 后） | 0.1770 | 0.1671 | 4.610 s | 18.895 s | 1.000 | 19 | 1 |

R2 相比 R1 的最大中间空白缩短 0.510 秒，是本轮唯一可见的平均时机改善；但内容覆盖、MT 相似度没有同步改善。最终模型相对 R1：

- MT 相似度下降 0.0018；
- 已发声目标覆盖下降 0.0023（约 1.4% 相对下降）；
- 首次发声反而晚 0.304 秒；
- 最大中间空白增加 0.205 秒；
- TTS failure 从 0.883 增至 1.000。

因此不能把三轮 GRPO 解释为整体有效提升。GRPO 的 surrogate / policy loss 稳定，只说明优化过程数值稳定，**不等价于翻译质量或完整性提升**。

## 最终模型的逐方向结果

| 方向 | ASR 相似度 ↑ | MT 相似度 ↑ | spoken 覆盖 ↑ | 首次发声 ↓ | 最大空白 ↓ | TTS failures ↓ | 健康音频比例 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 中→英 | 0.3163 | 0.2457 | 0.1685 | 4.533 s | 18.153 s | 1.938 | 0.9172 |
| 英→中 | 0.2059 | 0.1083 | 0.1656 | 4.688 s | 19.637 s | 0.063 | 0.9958 |
| 合计 | 0.2611 | 0.1770 | 0.1671 | 4.610 s | 18.895 s | 1.000 | 0.9565 |

最终 256 个候选中：205 个覆盖至少 10%，71 个至少 20%，19 个至少 30%，4 个至少 40%，仅 1 个达到 50%，0 个达到 80%。此外，52 个候选首次发声晚于 5 秒，229 个候选出现超过 10 秒的内部空白。所有 256 个候选的双声道 / 时间轴 / 连续目标语音文件均存在并可被音频库读取，但“可播放”不等于“译文完整”。

## 值得试听的样例

以下都来自训练已见的 64 条长 episode，且每条是 group 内的一个候选；适合听链路、早发声和空白问题，**不应当被当作完整翻译示例**。

1. **中→英，内容与连续性最平衡的最终样例**：`episode_000058_cmn_eng_g3`。ASR 0.708、MT 0.486、实际发声覆盖 43.9%、首次发声 3.20 秒、最大空白 7.2 秒、无 TTS failure。它最适合听到“途中多次发声”的正向现象，但仍缺少约 56% 的目标内容。

   - 双声道（左源语音、右译语音）：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/stereo_left_source_right_translation.wav`
   - 仅目标、按源时间轴放置：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/translation_global_timeline.wav`
   - 仅目标、连续拼接：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_23/audio/episode_000058_cmn_eng_g3/translation_continuous.wav`

2. **英→中，内容覆盖最高的最终样例**：`episode_000037_eng_cmn_g0`。ASR 0.702、MT 0.304、实际发声覆盖 50.0%、首次发声 1.92 秒、无 TTS failure。但中间仍有 13.7 秒空白，且后半段译文缺失；它适合作为“最高覆盖也仍不完整”的反例。

   - 双声道：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g0/stereo_left_source_right_translation.wav`
   - 时间轴目标：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g0/translation_global_timeline.wav`

3. **英→中，连贯性更好的最终对照**：`episode_000037_eng_cmn_g2`。覆盖 44.1%、首次发声 2.24 秒、最大空白 6.3 秒、无 TTS failure。它比同一源音频的 g0 更连续，但覆盖更低，直观展示当前策略仍然存在“更早/更连续”与“更完整”之间的不稳定取舍。

   - 双声道：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/workers/worker_44/audio/episode_000037_eng_cmn_g2/stereo_left_source_right_translation.wav`

## 相比先前 Stateful Long-Episode RL 的结论

历史 baseline 使用同一批 64 条 episode、每条 4 候选，但运行结构为 640 ms 决策、旧状态式前端与不同的 coverage 审计；其没有产生与 v4 完全同定义的 `spoken_target_coverage`，因此不能把 coverage 数字做严格显著性比较。

| 系统 | ASR 相似度 ↑ | MT 相似度 ↑ | 首次发声 ↓ | 最大空白 ↓ | 实际 TTS 音频片段数 ↑ | TTS failures ↓ | 健康音频比例 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史 Stateful 64×4 | **0.2998** | **0.1895** | 21.945 s | 31.075 s | 9.85 | **0.016** | **0.9990** |
| 本次 v4 最终 64×4 | 0.2611 | 0.1770 | **4.610 s** | **18.895 s** | **18.24** | 1.000 | 0.9565 |

v4 的真实正向改动是：第一段目标音频在源时间轴上平均提前约 17.3 秒出现、内部最长空白平均缩短约 12.2 秒，并且 256/256 都在源结束前发过音。这证明提交控制逻辑解决了旧系统经常拖到尾部才发声的问题。v4 平均实际合成 18.24 个音频片段；另有 205.27 次策略级 WRITE 决策，但后者包含未形成独立声学片段的控制事件，不能误作音频片段数。代价是：ASR/MT 文本相似度下降、TTS 片段失败更多、最终目标内容依旧不完整。两套系统的决策粒度和审计定义不同，所以这个表只说明运行行为的方向性变化，不构成严格的质量优胜结论。

## 根因判断

本次 RL 主要优化的是 **何时把已有增量翻译提交给 TTS**。它已修复“对空写入或不可用 token 给予正向 credit”的问题：不可执行控制事件会被 mask，只有实际健康的语音提交及其覆盖增量才获得 action credit。

但 action policy 不能补回上游没有给出的内容：

1. ASR 在 R1、R2、R3 与最终评估中均为 0.2611，说明本轮 GRPO 没有改善上游识别；
2. 增量 MT 是依据该不完整 / 不稳定 ASR 前缀产生，源信息一旦丢失，WRITE 决策只能更早或更晚地说出已有片段，无法生成缺失译文；
3. 策略级 WRITE 很密（平均 205.27 次），但平均只形成 18.24 个实际 TTS 片段；这说明大量 WRITE 没有对应到足够的可说增量。真正送入 TTS 的片段仍可能过短，从而增加音色/韵律边界和失败机会；
4. 现有 reward 虽已覆盖为先，但仍不足以让 action-only 更新改变内容瓶颈；三轮 fresh policy optimization 反而出现小幅退化。

## 下一步建议

不建议继续在当前 **冻结上游、只训动作 LoRA** 的设置上增加更多 GRPO 轮数。最可能的改进是保持完整的 Phase-3 多任务能力，再以事件级 teacher-forced 数据联合训练：

1. 使因果 / chunked ASR 与 incremental MT 的可训练参数参与更新，并以 Phase-3 offline ASR、翻译、语音 token 重放为锚，避免内容能力退化；
2. 在训练样本中保留完整连续目标语音的分段监督，按稳定短语边界（而非极短 token 差分）向 TTS 提交，并显式保持 speaker state；
3. 再在该内容质量已经达标的 checkpoint 上做少量 constrained GRPO，仅微调 WAIT/WRITE，奖励按最终目标覆盖、最大空白、首发声和 TTS 健康度联合计算；
4. 单独报告 source-time 时机和 wall-clock RTF；若目标是实际低于 1 秒，需要因果声学前端、增量 encoder state / KV cache 与可并行或流式 TTS，不能靠当前重算式运行时声称达成。

## 可复现入口

- 实验实现：`experiments/uniss_phasea_commit_complete_sft_rl_v4/`
- 最终 checkpoint：`checkpoints/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_grpo_round3_formal_v1/iter_0000142`
- 最终评估：`eval_outputs/uniss_phasea_commit_complete_sft_rl_v4/commit_v4_post_round3_g4_w64_v1/ROLLOUT_MERGED.json`
- 历史 baseline：`eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/ROLLOUT_MERGED.json`
