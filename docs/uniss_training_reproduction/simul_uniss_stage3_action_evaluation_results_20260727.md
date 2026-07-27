# Simul-UniSS full198 Stage3 action 评估结果与分析

> 完成时间：2026-07-27 UTC
>
> 评估对象：full198 Stage3 action-only SFT，iteration 4753
>
> 评估层级：teacher-forced WAIT/WRITE policy proxy
>
> 数据分配：UniST dev 使用 GPU 0–3，UniST test/eval 使用 GPU 4–7

## 1. 执行结论

Stage3 全量 action 评估已经成功完成。dev 和 test/eval 同时使用两个独立的
4-GPU 进程组运行，所有样本、action event 和 rank 输出均完整，没有 OOM、CUDA error、
Traceback、重复样本或重复 event。

主要结果如下：

| 指标 | UniST dev | UniST test/eval |
| --- | ---: | ---: |
| 样本数 | 7,965 | 23,369 |
| Action events | 62,688 | 175,848 |
| Binary WAIT/WRITE accuracy | 92.05% | 91.80% |
| Macro-F1 | 86.83% | 87.09% |
| WAIT F1 | 95.12% | 94.89% |
| WRITE F1 | 78.53% | 79.30% |
| Premature WRITE / given WAIT | 2.35% | 2.57% |
| Unnecessary WAIT / given WRITE | 29.48% | 28.27% |
| Full-vocabulary invalid top1 | 0.00% | 0.00% |
| Action CE / PPL | 0.1923 / 1.2121 | 0.1973 / 1.2181 |
| Final flush success | 89.87% | 90.77% |
| First-WRITE exact | 64.98% | 65.16% |
| First-WRITE MAE | 591.98 ms | 569.13 ms |

结论是：

1. Stage3 checkpoint 已经学会稳定地把 WAIT/WRITE 作为完整 180,480 词表中的 top1；
   eval 上没有任何 top1 落到非 action token 的情况。
2. dev 到 eval 的 accuracy 只下降 0.26 个百分点，Macro-F1 反而增加 0.26 个百分点，
   没有明显的 dev-only 过拟合迹象。
3. 模型整体偏保守。eval 上真实 WRITE 中有 28.27% 被预测成 WAIT，平均每条样本少预测
   0.315 个 WRITE；相比之下，真实 WAIT 被过早预测为 WRITE 的比例为 2.57%。
4. 当前结果证明的是 pseudo schedule 上的 teacher-forced action policy 能力，不是完整的
   free-running simultaneous S2ST，也不能直接当作 AL、LAAL、ATD 或 ASR-BLEU。

## 2. 可复现对象

### 2.1 Checkpoint

Megatron checkpoint：

```text
checkpoints/simul_uniss_v7_full198_seq18000_mbs2_gbs128_stage3/
  stage03_action_sft/iter_0004753
```

本次评估使用经过校验的独立 HF export：

```text
checkpoints/exported_hf/simul_uniss_stage3_action_v1_iter_0004753_hf
```

权重审计：

| 项目 | 数值 |
| --- | --- |
| Model vocab size | 180,480 |
| Tokenizer size | 180,407 |
| `model.safetensors` size | 1,039,248,688 bytes |
| SHA256 | `fadf5ccd87356cd672dfc2b39f5183fbd76995915a5b7c2fe216373c43c37715` |

运行时仓库 commit：

```text
007b49e9f4f0a9cfcef57637d62a5b3c317ed499
```

运行开始时 worktree 为空，保证结果对应上述 commit。

### 2.2 数据和 schedule

| Split | 输入 | 样本 | 方向 |
| --- | --- | ---: | --- |
| Dev | `data/processed/simul_uniss_v1/validation_dev/action_samples.jsonl` | 7,965 | cmn→eng、eng→cmn |
| Test/eval | `data/raw/UniST/test-00000.parquet` 的独立 Stage3 manifest | 23,369 | cmn→eng、eng→cmn |

Test/eval 的新产物位于：

```text
data/evaluation/simul_uniss_stage3_action_v1/unist_test_chunk640_wait2/
```

配置为：

```text
chunk_ms = 640
wait_k_chunks = 2
max_phrase_tokens = 16
alignment = pseudo_proportional_token_alignment
```

这里的 640 ms 是 schedule 的时间量化步长，不是本次模型计算耗时，也不是实测的
first-audio wall-clock latency。

## 3. 评估实现与正确性保护

代码和脚本全部位于新的独立目录：

```text
evaluation/simultaneous_streaming/
experiments/evaluation/simul_uniss_stage3_action_v1/
```

历史 offline/non-streaming 评估目录、训练脚本、checkpoint 和 `eval_outputs` 均未修改或覆盖。

### 3.1 推理方式

每个样本独立 padding，并使用自己的 attention mask。评估器不把多个独立样本拼到普通
causal mask 下，因此一个样本不可能看到另一个样本的答案。模型先计算 Qwen backbone，
然后只抽取 action token 前一个位置的 hidden state，再通过完整 LM head 计算 180,480
词表 logits。这样既保留完整词表 CE/top1 的正确性，也避免构造
`batch × sequence × vocabulary` 的巨大 logits tensor。

以下行为被明确禁止：

- 不截断过长样本；
- 不重复样本来人为增加 GPU 功率；
- 不添加无意义 token 或计算；
- 不覆盖已有输出目录；
- 不把两个样本 pack 到可能相互 attention 的序列中。

### 3.2 分布式完整性

样本按原始 record index 对 4 个 rank 取模分配。最终审计结果：

| 检查项 | Dev | Test/eval |
| --- | ---: | ---: |
| 期望样本 | 7,965 | 23,369 |
| 实际样本 | 7,965 | 23,369 |
| Unique sample IDs | 7,965 | 23,369 |
| Action events | 62,688 | 175,848 |
| Unique `(sample_id, event_index)` | 62,688 | 175,848 |
| Rank completion markers | 4/4 | 4/4 |

### 3.3 测试

执行并通过：

```text
新增 Stage3 evaluation unit tests: 4/4 passed
历史 Simul-UniSS regression tests: 69/69 passed
Python py_compile: passed
所有新增 shell 脚本 bash -n: passed
4+4 GPU distributed smoke: passed
```

## 4. 指标定义

### 4.1 Action 分类

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| Binary accuracy | 只在 WAIT 与 WRITE 两个 logit 中取 top1 | 策略二分类总体正确率 |
| Macro-F1 | WAIT F1 与 WRITE F1 的平均 | 避免 WAIT 类数量较多掩盖 WRITE 错误 |
| Full-vocabulary top1 | 在完整 180,480 词表中取 top1 | 验证 action 是否真的压过所有其他 token |
| Invalid global top1 | 完整词表 top1 不是 WAIT/WRITE 的比例 | 定位模型是否退回普通 token |
| Action CE/PPL | 真实 action 相对完整词表的交叉熵/困惑度 | 衡量 action 置信度，不只看 argmax |

### 4.2 策略错误

```text
Premature WRITE given WAIT
  = reference WAIT 但 prediction WRITE / reference WAIT

Unnecessary WAIT given WRITE
  = reference WRITE 但 prediction WAIT / reference WRITE
```

前者表示模型在 schedule 认为信息不足时过早输出；后者表示模型在已经允许输出时继续等待。
二者不能只优化一个：极端地永远 WAIT 会获得很低的 premature WRITE，但完全不可用。

### 4.3 First-WRITE 与 final flush

- `First-WRITE exact`：预测第一次 WRITE 与 pseudo reference 落在同一 640 ms chunk。
- `First-WRITE delta`：预测第一次 WRITE 时间减 reference 第一次 WRITE 时间；正数表示更晚。
- `First-WRITE MAE`：上述差值绝对值的均值。
- `Missing predicted first-WRITE`：整条样本没有预测任何 WRITE。
- `Final flush success`：最后一个 source-final event 被预测为 WRITE。

这些是 policy proxy。由于 reference 来自 pseudo proportional alignment，不能与真实词时间戳
产生的 AL/LAAL/ATD 数值直接比较。

## 5. 完整结果

### 5.1 总体 action confusion matrix

UniST test/eval：

| Reference \ Prediction | WAIT | WRITE | 合计 |
| --- | ---: | ---: | ---: |
| WAIT | 133,794 | 3,532 | 137,326 |
| WRITE | 10,891 | 27,631 | 38,522 |
| 合计 | 144,685 | 31,163 | 175,848 |

WAIT 占 eval action 的 78.09%，WRITE 占 21.91%。因此 91.80% accuracy 必须与 Macro-F1、
WRITE recall 和两个错误率一起解释，不能仅靠 accuracy 得出策略已经完全解决的结论。

### 5.2 Dev 与 test/eval 泛化

| 指标 | Dev | Test/eval | Eval - Dev |
| --- | ---: | ---: | ---: |
| Binary accuracy | 92.054% | 91.798% | -0.256 pp |
| Macro-F1 | 86.830% | 87.094% | +0.264 pp |
| Action CE | 0.1923 | 0.1973 | +0.0050 |
| Premature WRITE / WAIT | 2.355% | 2.572% | +0.217 pp |
| Unnecessary WAIT / WRITE | 29.479% | 28.272% | -1.207 pp |
| Final flush success | 89.868% | 90.770% | +0.902 pp |
| First-WRITE exact | 64.984% | 65.159% | +0.175 pp |
| First-WRITE MAE | 591.98 ms | 569.13 ms | -22.85 ms |

dev 与 eval 的变化很小且方向混合：accuracy 和 CE 略差，但 Macro-F1、WRITE 等待错误、
final flush 和 first-WRITE MAE 略好。这更符合 split 组成差异，而不是整体崩溃或明显过拟合。

### 5.3 First-WRITE 行为

Eval 上每条样本的 first-WRITE 分类：

| 行为 | 比例 |
| --- | ---: |
| 与 pseudo reference 完全一致 | 65.16% |
| 比 reference 更晚 | 20.41% |
| 比 reference 更早 | 6.79% |
| 没有预测 first WRITE | 7.64% |

有效 first-WRITE 样本的平均 signed delta 为 `+413.51 ms`，MAE 为 `569.13 ms`，p95
绝对误差为 `3200 ms`。正的 signed delta 和较高的 unnecessary WAIT 一致，都说明模型
总体偏向等待。

### 5.4 按翻译方向

UniST test/eval：

| 指标 | cmn→eng | eng→cmn |
| --- | ---: | ---: |
| Samples | 14,257 | 9,112 |
| Action events | 103,200 | 72,648 |
| Accuracy | 93.16% | 89.86% |
| Macro-F1 | 87.99% | 85.91% |
| WRITE F1 | 80.11% | 78.46% |
| WRITE recall | 69.86% | 73.81% |
| Premature WRITE / WAIT | 1.12% | 4.79% |
| Unnecessary WAIT / WRITE | 30.14% | 26.19% |
| Action CE | 0.1705 | 0.2355 |
| Final flush success | 89.88% | 92.16% |
| Missing first WRITE | 9.32% | 5.02% |
| First-WRITE MAE | 515.64 ms | 649.02 ms |

两个方向表现出不同的错误形态：

- `cmn→eng` action accuracy 和 CE 更好，premature WRITE 很低，但 WRITE recall 更低、
  missing first WRITE 更高，说明它更保守。
- `eng→cmn` 更愿意 WRITE，因此 recall 和 final flush 较好；代价是 premature WRITE
  明显升高，first-WRITE 时间误差更大。
- 后续不能只使用一个全局 action threshold。应先在 dev 上分别审计两个方向的
  WAIT/WRITE calibration，再冻结阈值到 test/eval；test 不能反向用于调参。

## 6. GPU 吞吐、显存和功率

### 6.1 全量 4+4 GPU 结果

| 项目 | Dev GPU 0–3 | Eval GPU 4–7 |
| --- | ---: | ---: |
| Max-rank inference wall time | 4.14 s | 11.72 s |
| Aggregate real tokens/s | 746,760 | 747,633 |
| Padding efficiency | 77.71% | 87.13% |
| Rank PyTorch peak allocated memory | 9.81–9.89 GiB | 9.89 GiB |

全局 GPU active-sample 统计：

```text
utilization mean = 52.8%
utilization p95  = 100.0%
power mean       = 353.6 W
power p95        = 507.4 W
```

每张卡均观测到 99–100% 的峰值利用率。dev 只需约 4 秒，GPU 0–3 的采样点很少且会比
eval 更早结束，所以其平均利用率不能当作稳定长任务平均值。Stage3 只有约 0.5B 参数，
而且只对 action hidden states 计算 LM head；它不会像 seq=18000 的 Phase3/Stage 训练那样
持续接近 H200 700W。这是有效计算量不同，不是漏用 GPU。

### 6.2 Batch 调优审计

在同一 GPU、相同 7,000 条 dev 样本上比较：

| Batch budget | Real tokens/s | Padding efficiency | PyTorch peak | Peak power |
| --- | ---: | ---: | ---: | ---: |
| 262,144 tokens / 512 samples | 206,705 | 90.61% | 9.89 GiB | 525 W |
| 524,288 tokens / 1,024 samples | 181,616 | 84.01% | 18.76 GiB | 662 W |

更大的 batch 虽然短时峰值功率更高，但有效吞吐下降 12.1%，padding 浪费增加，因此正式
评估选择 262,144-token 配置。两种配置在同一 53,169 个 action event 上：

```text
WAIT/WRITE prediction mismatches     = 0
full-vocabulary top1 mismatches      = 0
maximum target-CE absolute difference = 1.907e-6
mean CE difference                    = 6.063e-9
```

这保证吞吐优化没有改变离散评估结论。没有使用重复计算或无效 padding 来人为追求功率。

## 7. 与 simultaneous/streaming 文献的对应与不可比边界

当前 Stage3 使用 UniST 中英数据和 pseudo schedule action labels，而参考论文多使用 CVSS-C、
Europarl-ST 或 Audio-NTREX，并在自由运行 waveform 上报告质量和真实延迟。因此目前没有
同测试集、同输入、同输出模态、同 latency 定义的公开数值可以进行公平排名。

| 方法/论文 | 主要数据 | 论文核心指标 | 当前 Stage3 对应项 | 能否直接数值比较 |
| --- | --- | --- | --- | --- |
| SimulS2S-LLM | CVSS-C Es/Fr/De→En | ASR-BLEU、ATD、Start/End Offset、AL | WAIT/WRITE 与 first-WRITE proxy | 否 |
| Hibiki | CVSS-C Fr→En、Audio-NTREX、VoxPopuli | ASR-BLEU、LAAL、End Offset、speaker similarity、MOS | 仅 action policy | 否 |
| Hibiki-Zero | Europarl-ST、Audio-NTREX-4L | BLEU/XCOMET、ASR-COMET、LAAL、End Offset | premature WRITE 思路相近 | 否 |
| StreamSpeech | CVSS-C | ASR-BLEU、AL/AP/DAL/LAAL/ATD、RTF、Discontinuity | WAIT/WRITE eligibility proxy | 否 |
| NAST-S2x | CVSS-C Fr→En | ASR-BLEU、Unit-BLEU、BLASER、AL、ACT、DC metrics | action proxy，无 waveform | 否 |
| Textless Streaming S2ST | CVSS-C Es/Fr/De→En | ASR-BLEU、BLASER、AL | semantic streaming 尚未执行 | 否 |

Stage3 与这些工作的合理对应方式是验证“何时 READ/WRITE”的策略模块，而不是把本报告的
91.80% action accuracy 与论文的 ASR-BLEU 或 latency 直接比较。详细文献协议见：

```text
docs/uniss_training_reproduction/
  simul_uniss_stage3_stage4_stage6_streaming_evaluation_plan.md
```

## 8. 结果产物

完整运行目录：

```text
/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/
  simul_uniss_stage3_action_v1/full_stage3_20260727T031856Z/
```

重要文件：

| 文件 | 作用 |
| --- | --- |
| `aggregate_metrics.json` | 可机器读取的完整 dev/eval、方向和 GPU 指标 |
| `stage3_action_evaluation_report.md` | 运行时自动生成报告 |
| `dev/events.rank*.jsonl` | dev 逐 event 预测 |
| `dev/samples.rank*.jsonl` | dev 逐样本 first-WRITE/final flush |
| `eval/events.rank*.jsonl` | test/eval 逐 event 预测 |
| `eval/samples.rank*.jsonl` | test/eval 逐样本结果 |
| `gpu_monitor.csv` | 8 张 GPU 的利用率、功率和显存采样 |
| `environment/` | commit、git status、Python packages、GPU 环境和实验配置 |
| `COMPLETE` | 聚合和完整性检查成功标记 |

结果校验值：

```text
aggregate_metrics.json
  SHA256 9260f05541cc674fc707c50ebe87dcb24076d092dd31ec5a93a69794ebb3c7b7

stage3_action_evaluation_report.md
  SHA256 94679119da63e844de03bdbbeaddf13f84795e45cee3944b9b169c0e9f50024c
```

## 9. 后续评估顺序

Stage3 policy proxy 已完成，但整个 simultaneous S2ST 评估计划尚未完成。推荐下一步严格按：

1. 为 Stage4 建立 free-running Qwen streaming adapter，复用相同 manifest 和 event trace。
2. 在 dev 比较 `oracle schedule`、`Stage3 model action`、固定 `wait-k` 三种策略。
3. Stage4 生成目标文本和 BiCodec semantic tokens，报告 prefix/final text quality、token quality、
   理论 latency 和 computation-aware latency。
4. 接入 streaming BiCodec waveform 后，再报告 ASR-BLEU、silence-removed ASR-BLEU、
   BLASER、RTF、Discontinuity、Start/End Offset、AL/LAAL/ATD 和试听样本。
5. 用同一协议评估 Stage6，进行 Stage4 vs Stage6 paired comparison 和 quality-latency Pareto。
6. full198 Stage1/2 接入真实 streaming student/CTC boundary 后，重跑 Stage3，明确比较
   pseudo alignment 与真实 boundary-aware policy。

在完成第 4 步之前，不应把当前结果命名为“完整 simultaneous speech-to-speech 性能”。
