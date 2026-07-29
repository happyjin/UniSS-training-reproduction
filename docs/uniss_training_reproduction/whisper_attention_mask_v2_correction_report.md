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

全量重算进行中；完成后由 `experiments/evaluation/whisper_attention_mask_v2/summarize.py` 生成并替换本段。

<!-- WHISPER_V2_RESULTS_TABLE_END -->

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
