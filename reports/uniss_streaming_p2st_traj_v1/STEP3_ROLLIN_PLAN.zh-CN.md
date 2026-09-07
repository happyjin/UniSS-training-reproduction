# 步骤 3:用 roll-in 解决截断,然后才做 latency 条件化

## 为什么是 roll-in,而不是重做数据

截断率三个阶段没有动过:

| | C | 步骤 1 | 步骤 2 | gold |
|---|---:|---:|---:|---:|
| RealSI 截断率 | 0.510 | 0.555 | **0.550** | **0.000** |

步骤 2 的第 3 条**在数据构造上已经做到了** —— `uniform_chunk_tasks` 合并整个词块的
`target_semantic_delta` 再终止,所以 `END_SEMANTIC` 在训练样本里**只可能**落在词块边界。
做到了却没传导到推理,只剩一个解释:**训练喂 gold 语义前缀,推理用模型自己的前缀。**

这不是猜测,仓库里有直接的量化证据。2026-08-31 的 `uniss_phase3_e2e_rollin_continue_v1`
运行(跑了约 540 步后中止、无报告)记录了同一批样本上的两个数:

```
semantic_end_ce        = 0.384    gold 前缀下的终止决策
semantic_rollin_end_ce = 1.474    模型自己前缀下的同一决策
```

**同一个决策,换成模型自己的前缀贵 3.8 倍。** 这就是暴露偏差的大小,
也解释了为什么用数据构造保证落点在推理时会失效。

重做数据修不了它:数据里的落点已经是对的。

## 机制已经存在,不需要新写

`pretrain_e2e_megatron.py` 里的 `apply_symmetric_model_generated_semantic_rollin`:
每步额外做一次前向拿到模型自己的候选码,按 `rate` 替换进输入,再在**被替换过的前缀**上
监督终止决策。`pretrain_p2st_megatron.py` 把 phase3 训练器作为 `base` 导入,
所以这套机制在 p2st 上直接可用 —— 步骤 2 的命令行里本来就带着
`--e2e-semantic-boundary-rollin-rate 0.0`,只是关着。

入口有一条硬约束:`prefix_corruption` 与 `boundary_rollin` **不能同时开**
(`pretrain_e2e_megatron.py:585`)。本步只开 roll-in。

## 配置

| 项 | 值 | 理由 |
|---|---|---|
| 父 checkpoint | 步骤 2 的 `iter_0003180` | RealSI 上它优于 iter_2350(zh→en 13.48 对 12.35) |
| 数据池 | **复用** `chunk640_asridle_pool_train_20260905T165546Z` | 落点已经是对的,不重建 |
| `--e2e-semantic-boundary-rollin-rate` | **0.25** | 四分之一的样本用模型前缀。全开会让训练完全脱离 gold;历史那次用的是从 0 爬升的 ramp,均值只有 0.002,太小看不出效果 |
| `--e2e-semantic-boundary-rollin-ramp-updates` | **200** | 前 200 步线性爬到 0.25,避免开局就撞上 1.47 的损失 |
| `--e2e-semantic-rollin-end-weight` | **0.25** | 与历史那次一致。这是本步**唯一新开的损失项**,其余 12 项 margin/binary/speaker 仍为 0.0 |
| `--e2e-semantic-rollin-continue-*` | **0.0** | 只解决"何时停",不碰"是否继续"。历史上四次 continue margin 实验全部失败 |
| `COVERAGE_EPOCHS` | **1**(约 1592 步) | roll-in 每步多一次前向,预计 1.5–2 倍耗时。先用半程验证方向 |
| `RUN_MBS` | 1 | 与步骤 2 一致,也是它越过 iter 285 崩溃点的原因 |
| `SAVE_INTERVAL` / `EVAL_INTERVAL` | 50 / 400 | 与步骤 2 一致 |

## 判据

主判据只有一个,而且有 gold 参照:

| 指标 | 步骤 2 | 目标 | gold |
|---|---:|---|---:|
| **RealSI 截断率** | **0.550** | **≤ 0.45** | 0.000 |
| `speech/src` | 0.79 / 0.78 | 向 1.01 靠 | 1.01 |
| ASR-BLEU(k4,先验 1.0) | 17.42 / 12.91 | **不退化** | — |
| `semantic_rollin_end_ce` | — | 应从约 1.47 下降 | — |

诊断量 `semantic_end_ce` **不作判据** —— 777 条已经证明它挑不出更好的 checkpoint
(iter_2350 比 iter_3180 低 34%,RealSI 上却差 1.1 分)。

## 风险

* **roll-in 让训练脱离 gold,可能伤内容。** 所以 rate 只开 0.25、continue 项全部关闭,
  并且 BLEU 不退化是硬判据。
* **速度**:每步多一次前向。若超过 2 倍,减到 0.5 epoch 而不是硬撑。
* **历史那次只跑了 540 步就中止且无报告**,说明它可能有未记录的问题。
  第 50 / 100 / 200 步的外部评测照旧,尽早暴露。

## 之后才做步骤 4

latency 条件化(一个 checkpoint 服务多读步)。论文的写法已经确认是系统提示里的一句
自然语言 `" With Latency: {m}."`(`src/train/prompt_formats.py:68`),**不动词表**。
数据支持做它:同一 checkpoint 在 k1/k4/k25 上差 2.9 BLEU。
但截断是听感的直接来源,先修它。
