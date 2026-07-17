# Qwen0.5B Phase3 S2ST 音频测试与 Phase2 对比报告

生成时间：2026-07-17

## 测试结论

最新可用的 Phase3 checkpoint `iter_0004100` 已成功完成与 Phase2 相同条件的 S2ST 音频测试：

- 3 条 UniST dev 样本。
- 每条分别运行 `quality`、`performance`、`direct_s2st`。
- 模型生成音频 9/9 成功。
- source reconstruction 9/9 成功。
- target reference reconstruction 9/9 成功。
- `failed=0`。
- 全部 9 条生成都包含 `<|end_semantic_token|>` 和 `<|im_end|>`。

和 Phase2 相比，最明显的改善是 `magicdata_0000000002` 的 `performance` 输出不再发生严重重复坍缩：音频从 25.54 秒缩短到 5.42 秒，semantic token 从 1277 降到 271，最长连续重复从 343 降到 8。

Phase3 的 `quality` 模式也开始稳定生成“源语言转写 + 英文翻译”两段文本，而 Phase2 的三个 `quality` 样本都停留在中文转写/改写阶段。

## Checkpoint 边界

本次评测使用：

```text
checkpoints/uniss_qwen0p5b_phase3_unist13_full/iter_0004100
```

`latest_checkpointed_iteration.txt` 的值为 `4100`，训练日志确认该 checkpoint 已完整保存。训练进程之后运行到 iteration 4180，但在下次保存前收到 SIGTERM，因此 4180 没有可评测 checkpoint。

所以本报告评测的是“最新完整保存的 Phase3 checkpoint”，不是完成 4341 iterations 的最终 checkpoint。

## 可复现脚本

新增脚本：

```text
scripts/export_and_generate_qwen0p5b_phase3_audio_eval.sh
```

脚本会读取 tracker 并显式锁定对应的 `iter_XXXXXXX` 目录，避免转换器误选未来可能出现的半写 checkpoint。Phase3 使用独立的 HF 导出和评测目录，不会覆盖 Phase1/Phase2 结果。

本次执行命令等价于：

```bash
RUN_ID=20260717T114748Z \
PHASE3_ITERATION=4100 \
EVAL_CUDA_VISIBLE_DEVICES=3 \
CUDA_VISIBLE_DEVICES=3 \
scripts/export_and_generate_qwen0p5b_phase3_audio_eval.sh
```

HF 导出目录：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist13_iter_0004100_hf
```

评测输出目录：

```text
eval_outputs/qwen0p5b_phase3_unist13_s2st_dev_20260717T114748Z
```

## 与 Phase2 相同的测试条件

Phase2 和 Phase3 使用相同设置：

```text
split=dev
limit_records=3
modes=quality performance direct_s2st
max_new_tokens=1500
temperature=0.7
top_p=0.8
repetition_penalty=1.1
dtype=bfloat16
save_source_audio=1
save_reference_audio=1
```

Phase2 对照目录：

```text
eval_outputs/qwen0p5b_phase2_unist13_s2st_dev_20260716T174233Z
```

控制变量检查：

- Phase2/Phase3 的 9 个 `source_wav` 逐文件完全一致。
- Phase2/Phase3 的 9 个 `reference_wav` 逐文件完全一致。
- Phase2/Phase3 的 9 个模型生成 `wav` 全部不同。

因此两次测试使用的是同一批输入语音、目标参考语音和 mode，模型输出差异来自不同 checkpoint 及采样随机性。

注意：两次评测均使用 `temperature=0.7`，且生成器未设置固定随机种子，所以这是保持既有 Phase2 条件的试听/链路对比，不是严格逐 token 的确定性 benchmark。

## 运行结果

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

三个输出目录都包含 9 个非空、16 kHz、mono、16-bit PCM WAV：

```text
wav/
source_wav/
reference_wav/
```

## Phase2 与 Phase3 生成长度和重复对比

`max run` 表示同一个 BiCodec semantic token 的最长连续重复次数。

| 样本 | mode | Phase2 秒数 | Phase3 秒数 | Phase2 tokens | Phase3 tokens | Phase2 max run | Phase3 max run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000000001` | `quality` | 7.14 | 6.82 | 357 | 341 | 1 | 3 |
| `0000000001` | `performance` | 6.16 | 6.74 | 308 | 337 | 3 | 2 |
| `0000000001` | `direct_s2st` | 4.18 | 4.88 | 209 | 244 | 1 | 3 |
| `0000000002` | `quality` | 5.96 | 5.94 | 298 | 297 | 2 | 2 |
| `0000000002` | `performance` | 25.54 | 5.42 | 1277 | 271 | 343 | 8 |
| `0000000002` | `direct_s2st` | 5.90 | 6.00 | 295 | 300 | 2 | 2 |
| `0000000005` | `quality` | 2.84 | 2.38 | 142 | 119 | 1 | 1 |
| `0000000005` | `performance` | 2.58 | 3.74 | 129 | 187 | 1 | 19 |
| `0000000005` | `direct_s2st` | 3.06 | 2.62 | 153 | 131 | 1 | 1 |

Phase3 没有出现 Phase2 中 25.54 秒、连续重复 343 次的严重坍缩。Phase3 最大连续重复为 19，出现在 `0000000005/performance`；该输出只有 3.74 秒并且正常产生 EOS，但仍建议人工试听其局部重复情况。

## Phase3 生成音频自动质检

| 样本 | mode | 时长 | RMS | 非静音比例 `>|0.005|` | semantic tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `0000000001` | `quality` | 6.82s | 0.0532 | 34.9% | 341 |
| `0000000001` | `performance` | 6.74s | 0.0530 | 31.6% | 337 |
| `0000000001` | `direct_s2st` | 4.88s | 0.0446 | 18.9% | 244 |
| `0000000002` | `quality` | 5.94s | 0.0461 | 34.1% | 297 |
| `0000000002` | `performance` | 5.42s | 0.0510 | 45.3% | 271 |
| `0000000002` | `direct_s2st` | 6.00s | 0.0687 | 44.6% | 300 |
| `0000000005` | `quality` | 2.38s | 0.0620 | 38.9% | 119 |
| `0000000005` | `performance` | 3.74s | 0.0494 | 24.5% | 187 |
| `0000000005` | `direct_s2st` | 2.62s | 0.0527 | 43.5% | 131 |

全部输出都有可测量的非静音内容，没有 30 秒打满或近乎全静音的全局异常。`0000000001/direct_s2st` 的非静音比例最低，为 18.9%，需要结合人工试听判断是否存在过多停顿。

## 文本输出对比

### Quality

Phase2 的 `quality` 输出只生成中文 ASR/改写：

```text
我想用百度搜索短信。我想通过百度搜索短信。
税费还可以用多长时间？税费还可以使用多长时间？
我现在哪儿？我现在去哪儿？
```

Phase3 已生成中文源转写与英文翻译两段内容：

```text
我想用百度搜索短信。I want to use Baidu search for text messages.
水费还可以用多长时间？How much longer can water fees be used?
我现在哪儿？Where am I now?
```

这更符合 Quality slow-mode 的预期链路：先输出源端识别文本，再输出目标端翻译，最后生成目标语音 semantic token。

### Performance

Phase3 的三条英文文本为：

```text
I want to use Baidu to search for the text.
How long can water be used?
Where am I now?
```

第三条和 reference 完全一致；前两条语义大致对应 reference，但仍存在 `SMS -> text` 和 `water fee -> water` 的信息损失。

### Direct S2ST

`direct_s2st` 不生成中间文本，只能通过模型音频与 `reference_wav` 人工试听判断语言、内容、音色和韵律。

## 当前实验边界

- 模型是 Qwen2.5-0.5B reproduction，不是论文默认 1.5B 模型。
- 训练只使用当前 13 个 UniST train shard 快照，不是论文完整训练数据。
- 当前 Phase3 训练脚本复用了 Phase2 mixed packed 数据，不等价于论文严格的 high-quality-only Phase3 配方。
- 本次使用的是最新保存的 iteration 4100，不是计划中的 iteration 4341。
- 自动统计只能判断音频是否成功生成、长度和幅度是否异常；翻译正确性、可懂度、音色保持和韵律仍需人工试听或 ASR/客观指标评估。

## 最终判断

Phase3 `iter_0004100` 的生成链路正常，9 个 S2ST 输出全部成功解码。相对 Phase2，它解决了最严重的长音频重复坍缩，并让 `quality` 模式开始输出完整的中文识别与英文翻译结构。

当前最值得人工试听的文件是：

```text
wav/00001_performance_magicdata_0000000002.wav
wav/00002_performance_magicdata_0000000005.wav
wav/00000_direct_s2st_magicdata_0000000001.wav
```

分别用于确认：Phase2 严重坍缩样本是否真正恢复、19-token 局部重复是否可听、低非静音比例是否影响内容完整性。
