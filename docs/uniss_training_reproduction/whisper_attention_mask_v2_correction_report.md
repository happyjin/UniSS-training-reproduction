# ZH→EN Speech-BLEU / Whisper 批处理修复与结果更正报告

> 修复日期：2026-07-29  
> 适用范围：此前使用 batched Whisper-large-v3 计算的英文目标语音 ASR 与 ZH→EN Speech-BLEU  
> 修复协议：`whisper-large-v3-attention-mask-v2`

## 1. 结论

此前报告中约 `1.x--3.x` 的 ZH→EN Speech-BLEU 不能用于判断模型英文语音性能。主要原因不是 checkpoint 训练失败，而是 Transformers Whisper pipeline 在变长英文音频批处理时没有显式 attention mask，部分短音频会在 padding 区域继续解码并循环到 decoder limit；旧 ASR 文本因此膨胀约 16 倍，corpus BLEU 被系统性压低。

本次修复只重新读取已有生成 WAV，重跑英文 Whisper ASR 和 ZH→EN Speech-BLEU。以下内容均未重跑或覆盖：

- Phase2/Phase3、Stage4/Stage6、Stage7A/Reward-v2 训练；
- 文本生成与 BiCodec 音频解码；
- 中文 Paraformer ASR 与 EN→ZH Speech-BLEU；
- AutoPCP、SLC、UTMOS、Text-BLEU 和 streaming latency 指标；
- 旧 `metrics/`、旧音频和旧报告原始产物。

因此本次属于评估协议修复，不需要重新训练 checkpoint，也不需要重新生成或解码音频。

## 2. 根因证据

### 2.1 全量异常形态

以 Phase3 UniST test 为例：

| 项目 | Performance (P) | Quality (Q) |
|---|---:|---:|
| Text-BLEU ZH→EN | 32.4509 | 39.3753 |
| 旧 Speech-BLEU ZH→EN | 1.4790 | 1.7499 |
| 旧 Whisper hypothesis token/word 数量 | 约 197 万 | 约 202 万 |
| reference 长度 | 约 12.5 万 | 约 12.5 万 |
| 旧 system/reference 长度比 | 约 15.78 | 约 16.16 |

进一步审计显示：

- `43.18%` 的英文 ASR 输出长度超过对应 reference 的 10 倍；
- 约 `15%` 输出超过 350 词；
- 大量输出循环到约 444 词，符合 Whisper decoder-limit hallucination，而不是正常英文句长分布。

这解释了为什么 Text-BLEU 正常、音频质量/韵律指标也未整体崩坏，却只有英文 Speech-BLEU 异常；中文目标使用 Paraformer，不经过这一 Whisper padding 路径。

### 2.2 batch 对照与 reference control

| 对照 | Speech-BLEU | system/reference 长度比 | >10× reference |
|---|---:|---:|---:|
| 异常生成 WAV 子集，旧 batch=8 | 约 0.47 | 严重膨胀 | 大量存在 |
| 同一生成 WAV 子集，batch=1 | 16.4948 | 1.53 | 0 |
| 同一子集模型 Text-BLEU | 19.1123 | - | - |
| ground-truth reference WAV，旧 batch=8 | 5.28 | 严重膨胀 | 存在 |
| 同一 reference WAV，batch=1 | 86.4381 | 正常 | 0 |

reference WAV 也能复现 batch 依赖，证明低分不能简单归因于 UniSS 生成音频质量。

显式设置：

```python
recognizer.feature_extractor.return_attention_mask = True
```

后，16 条原异常样本在 batch=8 与 batch=1 的转写逐条一致，子集 BLEU 均为 `16.4948`。

### 2.3 极短音频边界条件

后续真实全量 smoke 还发现：对约 `0.18--1.5 s` 的极短生成音频，即使已有 attention mask，batch=32 仍可能触发 100--300 词循环。第一组 32 条短音频中，batch=32 有 20 条需要单条重试；batch=8 降为 5 条，且单条重试后都恢复为 1--5 词的合理转写。

还有极少数音频在 batch=1 后仍然 hallucinate，这表示输入本身不可稳定转写。此类样本不能跳过（会让 BLEU 虚高），也不能保留数百词循环（会让 BLEU 虚低）。修复协议把它记录为显式空 hypothesis，并通过 `--score-empty-hypotheses` 保留在 corpus BLEU 中，使该模型失败得到正确惩罚。

## 3. 实现与安全设计

### 3.1 ASR 修复

`evaluation/asr_transcribe.py` 现在执行：

1. Whisper feature extractor 强制返回 attention mask；
2. 英文输入按音频时长排序并避免 short/long preprocessing schema 混批；
3. `≤2 s` 极短音频直接 batch=1，其余短音频使用安全 batch size 8；
4. 用 `max(64 words, 12 words/s)` 做物理可解释的 transcript 长度 guard；
5. 可疑批结果自动以 batch=1 重试；
6. 单条仍异常时记录空 hypothesis 和 `asr_rejected_reason`；
7. 每条记录保存 protocol、attention-mask、请求 batch、实际 batch 和 retry/rejection 元数据；
8. 支持 `--target-language eng`，只重跑受影响的英文目标方向。

### 3.2 BLEU 完整性修复

`evaluation/text_metrics.py --score-empty-hypotheses` 会把显式空 ASR hypothesis 与完整 reference 一起送入 corpus SacreBLEU，而不是按 `missing_text` 跳过。这保证：

- 不可懂音频仍计入正式样本数；
- 不用人工占位 token 污染 n-gram 统计；
- 失败样本贡献零输出长度和零匹配，符合端到端 Speech-BLEU 语义。

### 3.3 隔离输出与验证

每个旧 run 只新增：

```text
metrics_whisper_attention_mask_v2/
├── asr_results_eng.jsonl
├── asr_results_eng.summary.json
├── speech_bleu_eng.json
├── verification.json
└── COMPLETE
```

`verify.py` 在写 `COMPLETE` 前检查：

- expected/observed 数量一致；
- 无 duplicate、missing 或 extra key；
- protocol 与 attention-mask 标记正确；
- 没有未经 rejection 处理的异常词速；
- rejection 必须来自 single-item retry 且 hypothesis 为空。
- requested batch 必须为 8，且 `≤2 s` 的样本 effective batch 必须为 1。

21 个主评估使用 GPU 1--7，每个 run 再确定性拆成两个互斥 shard，共 42 个 worker、每卡 6 个模型；GPU 0 保留给公网 Phase3 Demo。已有 canonical rows 作为 `completed-input`，两个 shard 写独立 JSONL，完成后加文件锁原子合并，再统一计算 BLEU、运行 verifier 和写 `COMPLETE`。

## 4. 全量旧值与修正值

下表只列受影响的 ZH→EN Speech-BLEU。EN→ZH 使用 Paraformer，数值不变。`Legacy length ratio` 和 `Corrected length ratio` 分别是 ASR system length / reference length；旧值显著大于 1 是本问题最直接的信号。

<!-- WHISPER_V2_RESULTS_TABLE_START -->

> 最终冻结时间：2026-07-29 23:30 UTC。21/21 个正式 run 均已写入并通过 `COMPLETE`。所有行均通过数量、协议、attention mask、异常长度、空 hypothesis 和 batch policy 验证。

| Run | Mode | N | Legacy Speech-BLEU | Corrected Speech-BLEU | Delta | Legacy length ratio | Corrected length ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| offline_phase2_unist_dev | performance:cmn->eng | 6508 | 1.2552 | 16.3810 | +15.1258 | 16.790 | 1.448 |
| offline_phase2_unist_dev | quality:cmn->eng | 6511 | 1.6176 | 19.3076 | +17.6899 | 15.660 | 1.468 |
| offline_phase2_unist_test | performance:cmn->eng | 14211 | 1.4024 | 16.8206 | +15.4183 | 15.788 | 1.473 |
| offline_phase2_unist_test | quality:cmn->eng | 14223 | 1.7262 | 21.3719 | +19.6457 | 15.493 | 1.408 |
| offline_phase3_unist_dev | performance:cmn->eng | 6513 | 1.4670 | 16.5154 | +15.0484 | 15.605 | 1.547 |
| offline_phase3_unist_dev | quality:cmn->eng | 6518 | 1.8927 | 21.3884 | +19.4957 | 14.536 | 1.451 |
| offline_phase3_unist_test | performance:cmn->eng | 14232 | 1.4790 | 19.3770 | +17.8980 | 15.779 | 1.364 |
| offline_phase3_unist_test | quality:cmn->eng | 14235 | 1.7499 | 22.7268 | +20.9769 | 16.163 | 1.403 |
| offline_phase3_cvss_t_cmn_to_eng | performance:cmn->eng | 4849 | 1.7459 | 6.9897 | +5.2438 | 4.784 | 1.286 |
| offline_phase3_cvss_t_cmn_to_eng | quality:cmn->eng | 4865 | 2.9397 | 12.0453 | +9.1056 | 4.515 | 1.196 |
| stage4_unist_dev | streaming_stage4:cmn->eng | 6528 | 1.7462 | 13.9852 | +12.2390 | 10.344 | 1.430 |
| stage4_unist_test | streaming_stage4:cmn->eng | 14252 | 3.0358 | 16.2733 | +13.2375 | 6.980 | 1.438 |
| stage6_unist_dev | streaming_stage6:cmn->eng | 6527 | 1.8947 | 14.2764 | +12.3817 | 9.715 | 1.428 |
| stage6_unist_test | streaming_stage6:cmn->eng | 14251 | 3.1077 | 16.8825 | +13.7749 | 6.847 | 1.394 |
| stage7a_v1_e0_test | stage7a_e0_stage6:cmn->eng | 14250 | 1.1546 | 16.7046 | +15.5500 | 18.289 | 1.419 |
| stage7a_v1_e1_test | stage7a_e1_continued_sft:cmn->eng | 14250 | 1.1644 | 16.2534 | +15.0890 | 18.089 | 1.451 |
| stage7a_v1_e2_test | stage7a_e2_grpo_g4:cmn->eng | 14250 | 1.1273 | 17.0564 | +15.9291 | 18.506 | 1.378 |
| stage7a_v1_e3_test | stage7a_e3_grpo_g8:cmn->eng | 14252 | 1.2149 | 16.7334 | +15.5185 | 17.288 | 1.407 |
| reward_v2_r0_dev | stage7a_reward_v2_r0_e3_v1_bias:cmn->eng | 6527 | 0.9592 | 14.2101 | +13.2510 | 19.002 | 1.436 |
| reward_v2_r1_dev | stage7a_reward_v2_r1_rebalanced_coverage:cmn->eng | 6527 | 0.9066 | 15.3875 | +14.4809 | 19.909 | 1.326 |
| reward_v2_r2_dev | stage7a_reward_v2_r2_explicit_latency:cmn->eng | 6527 | 0.9518 | 15.8127 | +14.8610 | 19.305 | 1.319 |
| reward_v2_r3_dev | stage7a_reward_v2_r3_bilingual_adaptive:cmn->eng | 6527 | 0.8906 | 15.3865 | +14.4959 | 20.549 | 1.349 |
| reward_v2_r0_test | stage7a_reward_v2_r0_e3_v1_bias:cmn->eng | 14252 | 1.1783 | 16.1344 | +14.9561 | 17.913 | 1.464 |
| reward_v2_r1_test | stage7a_reward_v2_r1_rebalanced_coverage:cmn->eng | 14251 | 1.1531 | 16.4390 | +15.2859 | 18.291 | 1.435 |
| reward_v2_r2_test | stage7a_reward_v2_r2_explicit_latency:cmn->eng | 14251 | 1.2210 | 17.1067 | +15.8857 | 17.513 | 1.402 |
| reward_v2_r3_test | stage7a_reward_v2_r3_bilingual_adaptive:cmn->eng | 14251 | 1.1990 | 17.0340 | +15.8350 | 17.748 | 1.402 |

<!-- WHISPER_V2_RESULTS_TABLE_END -->

### 4.1 最终结论

1. **旧的 `1.x--3.x` ZH→EN Speech-BLEU 已被证实无效。** 已完成 run 的旧 ASR system/reference 长度比为 `4.515--20.549`，修正后收敛到 `1.196--1.547`；这与 attention-mask 缺失导致 decoder-limit 循环的根因完全一致。
2. **Offline Phase3 本身没有出现英文语音能力崩溃。** UniST dev 修正后为 `16.5154`（P）和 `21.3884`（Q），test 为 `19.3770`（P）和 `22.7268`（Q）。相对 Phase2，Phase3 dev 的 P/Q 分别提高 `+0.1344/+2.0808`，test 分别提高 `+2.5564/+1.3549`。
3. **Quality 模式对 ZH→EN 可懂度有稳定收益。** Phase2 dev/test 的 Q 相对 P 分别提高 `+2.9266/+4.5513`，Phase3 dev/test 分别提高 `+4.8730/+3.3498`。这说明增加生成预算对英文目标语音的完整性和可识别性确实有效。
4. **Stage6 在 dev/test 都优于 Stage4，但 streaming 仍低于 offline。** UniST dev 上 Stage6 为 `14.2764`，比 Stage4 的 `13.9852` 高 `+0.2912`；test 上 Stage6 为 `16.8825`，比 Stage4 的 `16.2733` 高 `+0.6092`。Stage6 test 比 offline Phase3-P/Q 分别低 `2.4945/5.8443`。因此 streaming 的质量损失真实存在，但远没有旧报告中 `1.x` 所暗示的严重。
5. **第一轮 GRPO 有小幅正收益，不是“完全无效”。** Stage7A test 中 E2/GRPO-g4 为 `17.0564`，比 E0/Stage6 的 `16.7046` 高 `+0.3518`，是 E0--E3 中最好；E3/GRPO-g8 为 `16.7334`，仅比 E0 高 `+0.0288`，说明扩大 group size 本身没有带来等比例收益。
6. **Reward-v2 的 explicit-latency R2 是当前最稳定方案。** R2 在 dev/test 分别达到 `15.8127/17.1067`，均为 R0--R3 最高；相对 R0 分别提高 `+1.6026/+0.9723`。R3 test 为 `17.0340`，非常接近 R2，但 dev 低 `0.4262`，因此若只按当前修正后的 ZH→EN Speech-BLEU 选型，应优先 R2，再结合 latency、Text-BLEU、AutoPCP、SLC 和 UTMOS 判断最终 Pareto 点。
7. **CVSS-T 的 bug 已修复，但跨域差距仍然存在。** Phase3 CVSS-T 修正后为 `6.9897`（P）和 `12.0453`（Q），明显高于旧值 `1.7459/2.9397`，但仍低于 UniST 内部分布结果和原论文 1.5B 模型结果。因此 CVSS-T 的剩余差距不能再归因于 Whisper padding bug，需要从模型规模、训练域、数据覆盖、semantic/codec 泛化与测试协议继续分析。

### 4.2 最终比较边界

- Stage4-vs-Stage6 的 UniST test ZH→EN Speech-BLEU 已可冻结为 `+0.6092`，但完整模型推荐仍需联合其他质量和 latency 指标。
- Reward-v2 R2 test 的 `17.1067` 相对 offline Phase3-P/Q 分别低 `2.2703/5.6201`；R3 的 `17.0340` 分别低 `2.3430/5.6928`。
- 当前表只修正 ZH→EN Speech-BLEU；不能据此改写 Text-BLEU、EN→ZH Speech-BLEU、AutoPCP、SLC、UTMOS 或 latency 的原始数值。
- Reward-v2 的最终推荐不能只看单一 Speech-BLEU；R2/R3 仍需与已存在的延迟和韵律指标联合组成 Pareto 对比。

## 5. 旧报告影响范围

以下旧结论中的 ZH→EN Speech-BLEU 必须使用第 4 节修正值替代：

- `uniss_full198_phase2_phase3_detailed_evaluation_report.md` 的 Phase2/Phase3 dev/test ZH→EN Speech-BLEU；
- `simul_uniss_stage7a_grpo_15shard_validation_plan.md` 中 offline/streaming、Stage7A 和 Reward-v2 的 ZH→EN Speech-BLEU 比较；
- CVSS-T Phase3 Table 1 本地 ZH→EN Speech-BLEU 对比；
- Stage4/Stage6 streaming 报告内依赖旧英文 Whisper ASR 的 Speech-BLEU。

旧结果文件保留用于审计，但不能继续用来声称“ZH→EN 语音可懂度只有 1.x”或据此判断 GRPO/streaming 没有效果。Text-BLEU、AutoPCP、SLC、UTMOS、延迟指标以及 EN→ZH Speech-BLEU 不受本次 bug 影响。

## 6. 验证与提交记录

已执行：

- metric adapter 与 text metric 单元测试；
- Python compile 与 shell syntax 检查；
- `git diff --check`；
- 真实 Whisper batch=1/batch=8/batch=32 对照；
- CVSS-T 10-pair、P/Q 共 20 条 smoke，完整性 `20/20`、异常长度 `0`。

关键修复提交：

- `07c660e`：启用 Whisper attention mask 和英文-only 重跑入口；
- `d9b17d8`：加入隔离的修正评估流水线；
- `9f9b651`：可疑批输出自动 single-item retry；
- `915cece`：不可懂结果以空 hypothesis 计入 BLEU；
- `54a64c6`：加入 H200 多 worker 并行重算；
- 后续并发调优：每卡三个互斥 worker，21 个 run 同时断点重算。
