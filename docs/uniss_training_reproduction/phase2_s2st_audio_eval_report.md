# Qwen0.5B Phase2 S2ST 音频测试报告

生成时间：2026-07-16

## 测试目的

本次测试的目标不是评估最终论文指标，而是确认 Phase2 checkpoint 是否已经能按 UniSS 的 speech-to-speech translation prompt 生成可解码音频，并检查它是否还存在 Phase1 测试中的问题：

- 输出全部打满 `MAX_NEW_TOKENS=1500`。
- 生成音频固定 30 秒。
- 后半段 token 重复坍缩。
- 解码后大部分近静音。

## Phase2 是否对应官方 UniSS 的 S2ST 推理

结论：Phase2 已经是在模仿官方 `cmots/UniSS` 的 S2ST 任务格式，但它不是官方完整推理效果。

对应关系如下：

| 官方 UniSS 推理 task | 本地评测 mode | 作用 |
| --- | --- | --- |
| `Quality` | `quality` | 慢速质量模式：ASR -> S2TT -> target semantic |
| `Performance` | `performance` | 平衡模式：S2TT -> target semantic |
| `S2ST` | `direct_s2st` | 快速直接模式：source speech tokens -> target semantic |

本地 Phase2 使用 [training/sample_builders.py](../../training/sample_builders.py) 构造 token-id prompt，结构对齐官方 [uniss/cli/prompt.py](../../uniss/cli/prompt.py)。但是当前实验和官方模型仍有关键差异：

- 使用的是 `Qwen2.5-0.5B-Instruct` reproduction checkpoint，不是论文默认 `Qwen2.5-1.5B-Instruct`。
- 当前只用 13 个 UniST train shard 的快照跑通，不是论文完整 44.8k 小时数据。
- 当前是 Phase2 checkpoint，不是最终 Phase3 checkpoint。
- 评测输入来自 UniST parquet 预处理 token row，不是官方 `infer.py` 那种从真实 wav 运行时 tokenize。

因此，这次测试能回答“Phase2 S2ST 生成链路是否开始工作”，不能等价证明“已经达到官方 cmots/UniSS 的翻译质量”。

## 执行计划

1. 使用 Phase2 Megatron-LM checkpoint：

```text
checkpoints/uniss_qwen0p5b_phase2_unist13_full/iter_0001045
```

2. 导出为 Hugging Face checkpoint：

```text
checkpoints/exported_hf/qwen0p5b_phase2_unist13_full_hf
```

3. 使用 UniST dev 前 3 条样本，每条跑 3 个 S2ST mode：

```text
quality
performance
direct_s2st
```

4. 保存三类音频：

```text
source_wav/     源语音重建
reference_wav/  目标语音 reference 重建
wav/            Phase2 模型生成音频
```

5. 自动检查：

- 是否生成 9/9 个音频。
- `source_wav` 和 `reference_wav` 是否不同。
- `wav` 是否仍全部 30 秒。
- `wav` RMS、peak、非静音比例是否正常。
- 生成 token 是否包含 `<|end_semantic_token|>` 和 `<|im_end|>`。
- 是否存在长 token 重复坍缩。

## 执行命令

新增脚本：

```text
scripts/export_and_generate_qwen0p5b_phase2_audio_eval.sh
```

执行命令：

```bash
RUN_ID=20260716T174233Z \
EVAL_CUDA_VISIBLE_DEVICES=3 \
LIMIT_RECORDS=3 \
scripts/export_and_generate_qwen0p5b_phase2_audio_eval.sh
```

输出目录：

```text
eval_outputs/qwen0p5b_phase2_unist13_s2st_dev_20260716T174233Z
```

日志：

```text
logs/qwen0p5b_phase2_s2st_audio_eval_20260716T174233Z.log
```

## 执行结果

`summary.json`：

```json
{
  "failed": 0,
  "generated_audio": 9,
  "reference_audio": 9,
  "source_audio": 9,
  "total": 9
}
```

全部 9 个 mode 输出均成功生成音频，退出码为 0。

## Source / Reference 检查

这次 `source_wav` 和 `reference_wav` 不再相同：

- `source_wav` 使用 `source_bicodec`，是中文源语音重建。
- `reference_wav` 使用 `target_bicodec`，是英文目标语音 reference 重建。

这说明本次确实跑的是 S2ST mode，而不是 Phase1 那次的 `tts` reconstruction sanity check。

参考音频时长：

| 样本 | source 时长 | target reference 时长 |
| --- | ---: | ---: |
| `magicdata_0000000001` | 5.46s | 6.28s |
| `magicdata_0000000002` | 5.46s | 5.04s |
| `magicdata_0000000005` | 3.06s | 2.40s |

## 生成音频自动质检

| 样本 | mode | 生成时长 | RMS | 非静音比例 `>|0.005|` | semantic tokens | 初步判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `0000000001` | `quality` | 7.14s | 0.0418 | 21.8% | 357 | 可解码，有声音 |
| `0000000001` | `performance` | 6.16s | 0.0485 | 33.1% | 308 | 可解码，有声音 |
| `0000000001` | `direct_s2st` | 4.18s | 0.0558 | 63.4% | 209 | 可解码，有声音 |
| `0000000002` | `quality` | 5.96s | 0.0701 | 65.3% | 298 | 可解码，有声音 |
| `0000000002` | `performance` | 25.54s | 0.0356 | 66.0% | 1277 | 有声音但过长，存在重复风险 |
| `0000000002` | `direct_s2st` | 5.90s | 0.0495 | 45.0% | 295 | 可解码，有声音 |
| `0000000005` | `quality` | 2.84s | 0.0472 | 42.8% | 142 | 可解码，有声音 |
| `0000000005` | `performance` | 2.58s | 0.0528 | 45.5% | 129 | 可解码，有声音 |
| `0000000005` | `direct_s2st` | 3.06s | 0.0508 | 38.6% | 153 | 可解码，有声音 |

和 Phase1 的 `tts` 测试相比，Phase2 明显改善：

- 不再全部固定 30 秒。
- 不再全部近静音。
- 绝大多数样本生成了合理长度的 semantic token。
- 生成文本中出现了 `<|end_semantic_token|>` 和 `<|im_end|>`，说明模型多数情况下能停止。

但仍有一个明显异常：

- `00001_performance_magicdata_0000000002.wav` 长达 25.54 秒。
- semantic token 数为 1277，接近 1500 上限但没有打满。
- token `5088` 出现 765 次，最长连续重复 343 次。
- 该条虽然不是静音，但有明显重复坍缩风险，试听时很可能有拖长或重复段。

## 文本输出检查

`performance` mode 的文本翻译较接近目标语言：

| 样本 | reference translation | generated clean text |
| --- | --- | --- |
| `0000000001` | `I want to use Baidu to search for SMS.` | `I want to use Baidu to search for text.` |
| `0000000002` | `How much longer can the water fee be used?` | `The water cost can still be used for a long time.` |
| `0000000005` | `Where am I now?` | `Where am I now?` |

`quality` mode 当前表现不理想：三条都更像中文 ASR/改写，而不是英文翻译：

| 样本 | `quality` generated clean text |
| --- | --- |
| `0000000001` | `我想用百度搜索短信。我想通过百度搜索短信。` |
| `0000000002` | `税费还可以用多长时间？税费还可以使用多长时间？` |
| `0000000005` | `我现在哪儿？我现在去哪儿？` |

这说明 Phase2 的 `quality` 任务链路虽然能生成语音 token，但当前小模型/小数据训练还没有可靠学会 `ASR -> S2TT -> target semantic` 的完整多段输出结构。它倾向在 ASR/中文改写阶段停留，语言控制仍不稳定。

`direct_s2st` 没有文本输出，只能通过音频和 reference 试听判断是否为目标语言。

## 结论

Phase2 生成音频已经比 Phase1 明显正常：9 个输出都成功解码，绝大多数不是 30 秒静音，也不再出现 Phase1 那种全局同一 token 长时间坍缩。

但是当前 Phase2 还不能认定为“已经复现官方 UniSS S2ST 效果”：

- `performance` mode 初步最像可用的 S2ST 翻译路径，但有一条生成过长并出现 token 重复。
- `quality` mode 当前文本输出仍偏中文 ASR/改写，不稳定。
- `direct_s2st` 能生成正常长度音频，但需要人工试听确认目标语言是否正确。
- 当前模型是 0.5B + 13 shard + Phase2，不是官方完整权重或论文最终 Phase3。

下一步建议：

1. 人工试听 `wav/`、`source_wav/`、`reference_wav/` 中同名样本，优先听 `performance` 和 `direct_s2st`。
2. 等 Phase3 checkpoint 后重复同样脚本，比较 `quality/performance/direct_s2st` 是否进一步稳定。
3. 如果继续出现过长输出，可以给评测脚本增加基于 reference token 长度的安全上限，而不是对所有样本统一允许 1500 tokens。
