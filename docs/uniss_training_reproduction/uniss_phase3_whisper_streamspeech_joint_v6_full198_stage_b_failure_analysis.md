# UniSS Phase3 Whisper–StreamSpeech Joint V6 Full198 Stage B 失败分析

> 分析日期：2026-08-08  
> 分析对象：`phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1`  
> 结论：**完整训练目标失败；数值稳定和安全保护成功；普通任务优化部分成功；streaming semantic alignment 明确失败。**

## 1. 执行摘要

当前 Full198 Stage B 不能判定为“训练成功后正常结束”，也不应把最后的 `iter_0005000` 当作可部署或最佳 checkpoint。

本次运行计划训练 `9075` iterations，实际日志记录到 `5060`，完成约：

\[
\frac{5060}{9075}=55.76\%
\]

随后一个尚未形成完整 iteration 日志的 joint microbatch 触发了 bridge commitment 绝对安全门：

```text
bridge commitment exceeded absolute safety gate
value=0.101447
limit=0.100000
baseline_count=32
```

因此，本次训练的直接停止原因不是 OOM、NaN、CUDA 崩溃或数据进程意外退出，而是模型的 streaming hidden representation 已远离冻结 WhisperVQ codebook，安全门按设计主动终止训练。后续 `torch.distributed.elastic ... ChildFailedError` 只是一个 rank 抛出 `FloatingPointError` 后，多卡 launcher 回收其余进程的连带结果。

更重要的是，安全门触发前已经存在持续数千 iterations 的语义漂移，而不是最后一个 batch 的偶然尖峰。固定 `320 ms` validation 从 iteration 700 到 5000：

- teacher agreement 从 `15.89%` 降至 `5.45%`，绝对下降 `10.45` 个百分点，相对下降 `65.73%`；
- bridge commitment 从 `0.0202` 升至 `0.0401`，增大 `98.77%`；
- teacher commitment 从 `0.0381` 升至 `0.0749`，增大 `96.49%`；
- code perplexity 从 `59.65` 降至 `50.36`，下降 `15.58%`；
- hidden RMS 从 `0.694` 升至 `0.760`，增大 `9.49%`。

与此同时，BiCodec、AR S2TT、ASR CTC 和 NAR S2TT CTC validation loss 多数仍在改善。这说明模型学会了降低普通多任务 loss，但没有保持 Phase3 所依赖的 WhisperVQ/GLM semantic token 接口。对本实验最核心的 simultaneous/streaming 改造目标而言，这属于失败。

## 2. 实验身份与当前状态

### 2.1 路径

| 项目 | 路径 |
|---|---|
| 运行名 | `phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1` |
| 日志 | `logs/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1.log` |
| checkpoint 根目录 | `checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1` |
| TensorBoard | `runs/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_b_guarded_joint_full198_v2_mbs1` |
| Stage B 配置 | `experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/stage_b_env.sh` |
| 8-GPU 入口 | `experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_8gpu.sh` |

### 2.2 初始化与时间线

运行从以下干净 Stage A checkpoint 初始化：

```text
checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/
phase3_joint_v6_stage_a_heads_only_full198_v1
```

日志显示加载的是 Stage A 的 iteration 500，但使用 `--finetune --no-load-optim --no-load-rng`，所以 Stage B 自身 iteration 从 0 重新计数。

| 事件 | UTC | 北京时间 UTC+8 |
|---|---|---|
| Stage B iteration 1 | 2026-08-07 09:02 | 2026-08-07 17:02 |
| 保存 iteration 5000 | 2026-08-07 20:29 | 2026-08-08 04:29 |
| 最后完整日志 iteration 5060 | 2026-08-07 20:38 | 2026-08-08 04:38 |
| safety gate 终止 | 2026-08-07 20:40 | 2026-08-08 04:40 |

当前最新落盘 checkpoint 是：

```text
iter_0005000
```

不存在 `iter_0005060`，因为保存间隔为 250 iterations，而 safety gate 在下一次保存前触发。

### 2.3 当前是否仍在训练

该 Stage B 训练进程已经结束，当前 GPU 上运行的是独立的 8 卡 GPU 占用程序 `uniss_gpu_load_60`，不是本次 Stage B 训练。开始任何正式修复评估或重训前，需要先停止该占用 session：

```bash
tmux kill-session -t uniss_gpu_load_60
```

## 3. 失败判定分解

“实验失败”不能简单理解为所有部分都没有学到。更准确的分解如下。

| 目标 | 判定 | 证据 |
|---|---|---|
| 跑满 9075 iterations | 失败 | 5060/9075 时安全门终止 |
| streaming semantic alignment | 失败 | 320 ms agreement 15.89% → 5.45% |
| hidden/codebook 几何稳定 | 失败 | commitment 近乎翻倍，最终 microbatch 超过 0.10 |
| code 使用多样性 | 退化 | perplexity 59.65 → 50.36 |
| 普通 ASR/S2TT 多任务优化 | 部分成功 | 四项 validation loss 多数下降 |
| 数值稳定性 | 成功 | 训练日志无 NaN、无 skipped iteration |
| 显存稳定性 | 成功 | 无 OOM |
| safety guard | 成功 | 在进一步漂移前主动停止 |
| 最终 checkpoint 可用性 | 未通过 | 未完成训练，核心 agreement 恶化，尚无端到端质量门 |

因此，最准确的总判定是：

> **训练工程没有崩坏，但研究目标没有达成；安全门成功发现并终止了一个正在语义漂移的实验。**

## 4. 停止事件的技术解释

安全门实现位于 [model.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/phase3_whisper_streamspeech_joint/model.py:474)。它先对各 data-parallel rank 当前 microbatch 的 `bridge_commitment` 做 all-reduce 平均，然后与绝对阈值和相对阈值比较。

触发时各 rank 输出相同的全局均值：

```text
value=0.101447
limit=0.100000
baseline_count=32
```

但各 rank 打印的 chunk 不同：

```text
rank0: 640 ms
rank1: offline
rank2: 1280 ms
rank3: 960 ms
rank4: 320 ms
rank5: 1280 ms
rank6: offline
rank7: 960 ms
```

这是因为 chunk 选择使用 `sample_index=consumed_samples+rank`，见 [pretrain_joint_megatron.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py:573)，所以同一训练 step 的不同 rank 可以使用不同 chunk；guard 随后却把它们的 commitment 混合为一个均值。

日志中 iteration 5060 的平均训练指标仍是 `bridge/commitment_mse=0.0236`，并不与下一 microbatch 的 `0.101447` 冲突：

1. iteration 日志是前面一段统计窗口的聚合值；
2. safety gate 检查的是当前 joint microbatch；
3. 触发 gate 的 microbatch 没有成功完成，所以没有形成 iteration 5070 日志。

这说明最后确实存在一个较严重的样本/批次尖峰。但从固定 validation 的长期趋势看，尖峰只是最终触发点，不是全部原因。

## 5. 长期漂移证据

为避免训练期间随机 chunk 造成不可比，下面只比较 validation 中固定的 `320 ms chunk + 80 ms right context`。

### 5.1 核心 streaming 指标

| Stage B iteration | Teacher agreement ↑ | Bridge commitment ↓ | Teacher commitment ↓ | Code perplexity ↑ | Hidden RMS |
|---:|---:|---:|---:|---:|---:|
| 700 | 15.89% | 0.02016 | 0.03810 | 59.65 | 0.6941 |
| 3000 | 9.28% | 0.02967 | 0.05567 | 53.69 | 0.7390 |
| 5000 | 5.45% | 0.04007 | 0.07487 | 50.36 | 0.7599 |

指标含义：

- `teacher agreement`：student hidden 量化后的 top-1 GLM code 与 teacher `source_glm` 在同一位置完全相等的比例；越高越好。
- `bridge commitment`：student hidden 到其最近 codebook vector 的均方距离；越低表示越贴近可量化 codebook 流形。
- `teacher commitment`：student hidden 到指定 teacher code vector 的均方距离；越低表示越接近目标 teacher token。
- `code perplexity`：当前 batch 中有效 code 的使用多样性；持续下降通常意味着 code 使用范围收缩。
- `hidden RMS`：hidden 整体尺度诊断；它本身不是越低越好，但持续漂移并伴随 commitment 恶化说明表示空间发生系统变化。

从 700 到 5000，agreement 单调恶化，两个 commitment 单调增大，code perplexity 单调下降。这四个趋势方向相互一致，因此不能解释为单次 validation 噪声。

### 5.2 普通多任务 validation loss

| Stage B iteration | BiCodec CTC ↓ | AR S2TT ↓ | ASR CTC ↓ | NAR S2TT CTC ↓ |
|---:|---:|---:|---:|---:|
| 700 | 9.5375 | 4.9560 | 9.5782 | 9.4528 |
| 3000 | 9.3075 | 4.3457 | 9.4363 | 9.1998 |
| 5000 | 9.2093 | 4.3089 | 9.2636 | 8.9667 |

从 700 到 5000：

| 指标 | 相对变化 | 表面结论 |
|---|---:|---|
| BiCodec CTC | -3.44% | 改善 |
| AR S2TT | -13.06% | 改善 |
| ASR CTC | -3.29% | 改善 |
| NAR S2TT CTC | -5.14% | 改善 |

### 5.3 为什么普通 loss 下降仍然判失败

Phase3 后端并不是接收任意连续 hidden，而是依赖 WhisperVQ/GLM semantic code 接口。如果 student hidden 为了降低 ASR、AR、NAR 或 BiCodec loss，移动到一个对这些任务更方便、但不再对应原 GLM codebook 的表示空间，则：

```text
普通任务 loss 可以下降
        同时
量化 token 与 Phase3 预训练语义接口越来越不一致
```

例如 teacher token 原本是 `4312`，student hidden 初期最近的 code 也是 `4312`。联合训练后，ASR head 可能更容易从一个偏移后的 hidden 预测文字，但该 hidden 的最近 code 变成 `9871`。ASR CTC loss 会下降，teacher agreement 却会变为 0。对需要继续调用原 Phase3 Qwen/BiCodec 生成路径的系统，这不是可接受的替代表示。

## 6. 根因分析

下面区分“代码和数据中直接确认的事实”与“由指标趋势支持的因果判断”。根因不是一个参数，而是 teacher 数据、信息条件、可训练结构和 loss 设计共同形成的冲突。

### 6.1 根因一：teacher token 与实际训练输入音频不构成闭环

这是当前最高优先级的问题。

Stage A 数据准备直接从 UniST parquet 读取原始 `source_glm`：

```python
source_glm = row["source_glm"]
```

见 [stage_a.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v1/stage_a.py:195)。

但送入当前 Whisper frontend 的音频不是生成该 `source_glm` 时的原始 waveform，而是：

```text
source_bicodec + bicodec_global
    → BiCodec decoder
    → 重建 waveform
    → FLAC
    → 当前 WhisperVQ/student frontend
```

对应实现见 [stage_a.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v1/stage_a.py:203)。

BiCodec 重建会改变波形细节、帧边界和声学表示。原 parquet 的 `source_glm` 未必等于对重建 FLAC 重新运行冻结 offline WhisperVQ 得到的 token。

同一数据链上的固定评估已经给出强证据：冻结 Stage A 即使使用 `offline` 全上下文，teacher agreement 也只有 `22.96%`，而不是接近 100%。报告位于：

```text
reports/uniss_phase3_whisper_streamspeech_joint_v6/
fixed_chunk_stage_a_v2_vs_stage_b_v3_v1/report.md
```

其冻结 Stage A 结果为：

| Chunk | Teacher agreement |
|---:|---:|
| 320 ms | 12.22% |
| 640 ms | 14.42% |
| 960 ms | 15.63% |
| 1280 ms | 16.70% |
| Offline | 22.96% |

该固定评估使用的是 15-shard validation，而不是本次 full198 validation；但二者使用相同的数据构造链，因此足以证明系统性风险。下一版必须在 full198 抽样上重新做同一项闭环审计。

这意味着当前 student 在被要求完成一个内部矛盾的任务：从重建音频预测另一条音频编码链生成的 exact-position token。

### 6.2 根因二：streaming student 被要求复现拥有完整未来信息的 offline teacher

当前 multi-chunk 配置为：

```text
chunk = 320 / 640 / 960 / 1280 ms / offline
right context = 80 ms
```

见 [run_stage_8gpu.sh](/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_8gpu.sh:48)。

对同一句“我们将在下周发布新的模型”，offline teacher 在编码“发布”附近的 frame 时可以看到完整后文；320 ms student 只能看到当前块和 80 ms 右上下文。音素边界、停顿、共发音和 Whisper self-attention hidden 都可能改变。因此，短 chunk hidden 不可能在所有位置严格复制 offline teacher 的 exact code。

这并不意味着 streaming 蒸馏不可做，而是应该：

- 使用与当前重建音频自洽的 teacher；
- 对未来不可辨识位置使用 soft distribution 或 hidden target；
- 对边界位置允许局部时间偏移；
- 把不可避免的 offline/streaming 差异与真正的模型漂移分开统计。

### 6.3 根因三：没有独立可训练的 streaming repair bridge

当前 bridge 使用 `topk_soft` surrogate。代码中只有 `projection` 模式才创建可训练的 `nn.Linear`；`topk_soft` 只保存冻结 codebook 和 Qwen embedding buffer，见 [phase3_ste_bridge.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/phase3_whisper_streamspeech_joint/phase3_ste_bridge.py:62)。

运行参数又明确设置：

```text
--joint-lr-bridge-mult 0
```

见 [run_stage_8gpu.sh](/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_8gpu.sh:72)。

因此，系统没有一个容量受控、专门学习“chunk hidden → offline codebook geometry”修复的 adapter。实际可训练路径主要是：

- 新增 CTC/任务 heads：最大 LR `1e-5`；
- Whisper 最后 1 个 pre-VQ layer：最大 LR `1e-7`；
- Qwen 主体：最大 LR `1e-8`；
- Whisper codebook、post-VQ、其余 Whisper layers 和 Qwen I/O：冻结或 LR 为 0。

ASR、NAR、AR、BiCodec 和 teacher alignment 的梯度共同作用在 Whisper 顶层 hidden 上。任务 heads 能快速降低自己的 loss，而没有独立 adapter 吸收 streaming/offline 表示差异，最终使共享 hidden 逐步离开 codebook。

### 6.4 根因四：优化目标、诊断指标和安全门不完全一致

当前 Stage B 的总 loss 权重是：

\[
\begin{aligned}
L ={}&1L_{BiCodecCTC}+2L_{AR}+1L_{ASR}+1L_{NAR}+1L_{replay}\\
&+0L_{bridge\_commit}+1L_{whisper\_quantize}\\
&+0.25L_{teacher\_CE}+1L_{teacher\_commit}.
\end{aligned}
\]

配置见 [stage_b_env.sh](/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/stage_b_env.sh:2)。

这里存在三个问题：

1. `teacher agreement` 是 argmin 后的 exact-match 诊断，不可微，代码只是记录它，并不直接优化它，见 [phase3_ste_bridge.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/phase3_whisper_streamspeech_joint/phase3_ste_bridge.py:175)。
2. safety guard 监控 `bridge commitment`，但该项的显式 loss 权重为 `0`。虽然 `whisper_quantize` 和 `teacher_commitment` 提供相关约束，guard 监控目标与直接优化目标仍不一致。
3. 普通任务权重合计较强，且新增 heads 使用远高于 Whisper/Qwen 的 LR，优化器很容易优先找到“heads 可用但 code identity 改变”的解。

因此，普通 loss 改善不能自动保证 agreement 改善。

### 6.5 根因五：exact-position agreement 对帧偏移过于敏感

当前 agreement 直接比较：

```python
student_code[t] == teacher_code[t]
```

见 [phase3_ste_bridge.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/phase3_whisper_streamspeech_joint/phase3_ste_bridge.py:158)。

如果重建音频或 chunk boundary 使相同语义 code 整体偏移 1 帧：

```text
teacher: [A, A, B, B, C]
student: [A, B, B, C, C]
```

序列内容高度相似，但 exact-position agreement 会明显降低。因此，当前 5.45% 同时混合了：

- 真正 code identity 错误；
- 1–2 frame 时间偏移；
- token 重复长度不同；
- chunk boundary 造成的局部错位。

这不是当前训练失败的唯一原因，因为 commitment、perplexity 和 hidden RMS 也同步恶化；但它会放大 agreement 数值的悲观程度，并妨碍定位问题到底来自语义错误还是时序错位。

### 6.6 根因六：guard 把不同 chunk 的 rank 混合在一起

同一个 step 中各 rank 使用不同 chunk，但 guard 先跨 rank 求均值，再把同一个均值与各 rank 自己的 chunk 名一起打印。

后果是：

- 可以正确发现“总体 commitment 已危险”；
- 不能准确指出是 320、640、960、1280 还是 offline 导致尖峰；
- 不同 chunk 的正常 commitment 分布可能不同，共用一个 baseline/阈值会增加监控噪声；
- 无法直接回溯触发样本及语言方向。

这是监控设计缺陷，不是 representation drift 的根本成因，但必须在重训前修复，否则下一次触发仍难以定位。

## 7. 对 checkpoint 的处置建议

### 7.1 不建议从 iteration 5000 直接恢复

不应通过简单提高 `MAX_COMMITMENT=0.10` 后从 `iter_0005000` 继续，因为：

- agreement 已从 15.89% 下降到 5.45%；
- bridge/teacher commitment 已持续数千 iterations 恶化；
- code perplexity 持续下降；
- 触发 safety gate 前的状态已不是偶发异常。

提高阈值只会允许模型在已经错误的方向上继续走，并不会修复 teacher 数据、bridge 容量或 loss 冲突。

### 7.2 所有 checkpoint 都应保留

`iter_0005000` 仍有事故分析价值，不应删除或覆盖。建议把以下 checkpoint 做统一固定评估：

```text
Stage A clean initialization
iter_0000750
iter_0001000
iter_0001500
iter_0002000
iter_0002500
iter_0003000
iter_0004000
iter_0005000
```

目的不是从中直接选最终模型，而是定位：

- agreement 从何时开始持续下降；
- 普通 loss 改善与 semantic drift 的拐点；
- 哪个 checkpoint 适合作为对照组；
- safety gate 是否应更早触发。

### 7.3 新实验从哪里开始

推荐新版本从以下干净 Stage A 初始化，而不是从 Stage B iteration 5000 恢复：

```text
checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/
phase3_joint_v6_stage_a_heads_only_full198_v1
```

但在正式 full198 重训前，必须先完成 teacher 标签闭环修复和 15-shard 验证。

## 8. 推荐修复方案

### 8.1 第一步：重建自洽 teacher 标签

对 15-shard 试验集逐条执行：

```text
source_bicodec + bicodec_global
    → 与训练完全相同的 BiCodec decoder
    → 与训练完全相同的 FLAC / resample / mel pipeline
    → frozen offline WhisperVQ
    → source_glm_reencoded
```

新增字段建议为：

```text
source_glm_reencoded
source_glm_reencoded_length
offline_pre_vq_hidden（可选，存储成本允许时）
```

原始 `source_glm` 必须保留，用于旧 Phase3 兼容性监督和审计，不能原地覆盖。

数据闭环质量门：

| 检查 | 初始门槛 |
|---|---:|
| 同一 reconstructed FLAC 重跑 frozen offline WhisperVQ top-1 agreement | ≥99% |
| token length 一致率 | ≥99.9% |
| teacher coverage | 100% 或解释所有缺失样本 |
| 15-shard 随机抽样人工核对 | 至少 100 条 |

如果同一 deterministic pipeline 仍无法达到接近 100%，说明预处理、模型版本、padding、采样率或 VQ codebook 加载仍不一致，不能进入训练。

### 8.2 第二步：增加独立 residual streaming repair adapter

保持原 WhisperVQ 主体和 codebook 冻结，在 pre-VQ hidden 后增加小型 residual adapter：

\[
h_{repair}=h_{chunk}+\alpha A(h_{chunk}),
\]

其中：

- `A` 可使用 2-layer MLP、轻量 causal Conv 或小型 causal Transformer；
- `alpha` 初始设为 0 或很小，保证初始行为接近 Stage A；
- adapter 单独参数组和学习率；
- adapter 输出再进入冻结 codebook quantization 和 Phase3 Qwen。

这样 streaming/offline 差异主要由受控的新模块吸收，而不是直接推动 Whisper 顶层和 Phase3 主干漂移。

### 8.3 第三步：使用平滑 alignment supervision

推荐的 alignment 核心可写为：

\[
L_{align}=
2L_{hidden}
+1L_{topkKL}
+1L_{teacherCE}
+0.25L_{commit}.
\]

含义：

- `L_hidden`：streaming repaired hidden 与 frozen offline pre-VQ hidden 的 masked MSE/cosine loss；
- `L_topkKL`：对 codebook 距离形成的 teacher/student soft distribution 做 top-k KL，而不是只依赖 hard token；
- `L_teacherCE`：保留离散 teacher code 分类监督；
- `L_commit`：限制 hidden 不离开 codebook 流形。

对 chunk boundary 附近可使用较低权重或局部对齐；对稳定中心区域使用完整权重。

### 8.4 第四步：两阶段训练，而不是立即全量联合更新

#### Phase I：alignment warmup，500–1000 iterations

- 数据：15 shards；
- 冻结 Whisper、Qwen、codebook、post-VQ 和所有旧 Phase3 参数；
- 只训练 repair adapter；
- 优化 `L_hidden + L_topkKL + L_teacherCE + L_commit`；
- 每 100 iterations 固定评估五个 chunk。

只有通过 alignment gate 才进入 Phase II。

#### Phase II：guarded joint training

- 继续训练 adapter 和新 CTC heads；
- Whisper 默认继续冻结；若必须解冻，最多只解冻最后一层，最大 LR 不超过 `2e-8`；
- Qwen 保持冻结或极小 LR；
- 保留 Phase3 replay；
- 逐步加入 AR/NAR/ASR/BiCodec loss，而不是 iteration 1 起全部同权竞争。

一种安全的 loss ramp：

| 区间 | Alignment | ASR/NAR | AR/BiCodec | Replay |
|---|---:|---:|---:|---:|
| 0–500 | 1.0 | 0 | 0 | 0.25 |
| 500–1000 | 1.0 | 0.25 | 0.25 | 0.5 |
| 1000+ | 1.0 | 1.0 | 1.0 | 1.0 |

实际权重应由 15-shard 梯度范数和固定评估校准，不能仅按 loss 数值大小猜测。

### 8.5 第五步：修复 guard 和可观测性

重训前至少完成：

1. 由 rank 0 选择 chunk 并 broadcast，保证同一步各 rank 使用相同 operating point；或按 chunk 分组 all-reduce。
2. 为每个 chunk 独立维护 commitment baseline、EMA、绝对阈值和相对阈值。
3. 记录触发样本 ID、语言方向、音频时长、chunk、token 长度和各 rank 指标。
4. 增加 teacher agreement 下降 gate，而不仅监控 commitment。
5. 同时记录 exact、top-k、±1/±2 frame 和 edit/DTW agreement。
6. 触发 guard 时先保存诊断快照到独立目录，再终止训练。

建议的初始 guard：

| 项目 | 建议 |
|---|---|
| commitment absolute | 先保持 0.10，不直接放宽 |
| commitment relative | 每个 chunk 相对自身 baseline |
| agreement trend | 连续 3 次固定评估下降超过 2 pp 则暂停 |
| code perplexity | 连续下降且超过 baseline 20% 时报警 |
| NaN / skipped iteration | 任一出现立即停止并保存上下文 |

## 9. 新版质量门与评估矩阵

### 9.1 15-shard alignment gate

每个 checkpoint 固定评估：

```text
320 / 640 / 960 / 1280 ms / offline
right context = 80 ms
相同 validation IDs
相同 batch 和 seed
```

至少记录：

- exact top-1 agreement；
- top-5/top-8 agreement；
- ±1、±2 frame shift agreement；
- edit-distance/DTW agreement；
- hidden cosine 和 hidden MSE；
- bridge/teacher commitment；
- code perplexity 和 active-code fraction；
- ASR/NAR/AR/BiCodec validation loss。

进入 full198 的初始门槛建议：

| 项目 | 门槛 |
|---|---|
| offline self-consistency | ≥99% |
| 320 ms exact agreement | 比修复后的 frozen Stage A baseline 至少提高 3 pp |
| agreement 趋势 | 最近 3 次评估不得持续下降 |
| bridge commitment | 不超过各 chunk baseline 的 1.5 倍且 <0.05 |
| 普通任务 loss | 相对 Stage A 不恶化超过 5% |
| NaN / skipped / OOM | 0 |

这些是进入 full198 的工程门，不代表最终论文质量。最终仍需端到端生成评估。

### 9.2 Full198 checkpoint gate

Full198 重训时建议每 250–500 iterations 对固定小 validation 做上述快速 gate，并在 1000、3000、5000、最终 iteration 做完整评估。

只有同时满足以下条件才可称为成功：

1. semantic alignment 不随训练持续下降；
2. 普通 Phase3/offline replay 性能不明显回退；
3. streaming 320–1280 ms operating points 均可生成；
4. ASR-BLEU/BLEU、speaker similarity、AutoPCP/SLC 不低于设定门槛；
5. 测得真实 first-write、first-audio、AL/AP、RTF，而不是只用 chunk size 推断延迟。

## 10. 新实验隔离建议

为保证所有 V6 结果和历史脚本可复现，新版不要原地改写 V6 结果目录。建议建立：

```text
experiments/uniss_phase3_whisper_streamspeech_joint_v7_repair/
data/processed/phase3_whisper_streamspeech_joint_v7/
checkpoints/uniss_phase3_whisper_streamspeech_joint_v7/
logs/uniss_phase3_whisper_streamspeech_joint_v7/
runs/uniss_phase3_whisper_streamspeech_joint_v7/
reports/uniss_phase3_whisper_streamspeech_joint_v7/
```

V6 的脚本、日志、TensorBoard 和所有 checkpoint 保持只读，不覆盖、不续写。

## 11. 推荐执行顺序

1. 保留 V6 全部产物，标记 `iter_0005000` 为 failure-analysis checkpoint。
2. 对 V6 Stage A、750、1000、1500、2000、2500、3000、4000、5000 做固定 checkpoint matrix。
3. 在 15 shards 上生成 `source_glm_reencoded`，先通过 offline ≥99% 闭环门。
4. 新增 residual repair adapter 和 soft alignment losses。
5. 只训练 adapter 做 500–1000 iteration alignment warmup。
6. 通过 320/640/960/1280/offline 固定 gate 后，再做 15-shard guarded joint training。
7. 通过端到端生成和离线兼容性评估后，才启动独立 full198 V7。
8. Full198 仍从干净 Stage A 初始化，不从 V6 iteration 5000 续训。

## 12. 最终结论

当前 Full198 Stage B 应判定为失败，但失败不是“代码完全不可用”或“GPU 训练崩溃”：

- Megatron 8 卡训练流程、数据读取、保存、validation 和数值稳定性都正常工作；
- ASR、AR、NAR、BiCodec 多任务 heads 确实学到了东西；
- safety guard 正确阻止了更严重的 codebook drift；
- 失败集中在本实验最关键的目标：保持 Phase3 semantic-code 接口的同时获得 streaming/multi-chunk 能力。

最根本的问题是：当前 teacher token 与重建训练音频不闭环，短 chunk student 又被要求 exact-position 模仿完整上下文 teacher，同时缺少独立的 streaming repair adapter。普通多任务 loss 因而可以下降，semantic agreement 却持续恶化。

所以正确修复方向不是提高 safety 阈值或从 iteration 5000 硬续训，而是先重建自洽 teacher、加入受控 repair adapter、采用 soft/hidden alignment、修复 per-chunk guard，再从干净 Stage A 做独立 V7 实验。

