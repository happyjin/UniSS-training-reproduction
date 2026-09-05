# 本次训练:动机、损失构成、以及与论文训练流程的对比

## 一、本次训练(步骤 1)的 motivation

**只改一件事:训练池的组成。** 模型、目标函数、超参、训练代码、验证池全部与 C 逐项相同,
父检查点是 C 的 `iter_0004236`。

**为什么值得单独做一次训练**:论文 §4.5 的消融把这一条标为它报告过的最强杠杆 ——

> *"**Without monotonicity filtering, low-latency (m1) performance collapses
> entirely, plummeting to 4.59 and 3.56 BLEU** in the two directions...
> Introducing NIR-based filtering immediately recovers this gap, proving that
> high-quality, difficulty-controlled trajectory supervision is **the primary
> driver of low-latency robustness**. Furthermore, varying the multiplier
> sampling schedules yields **negligible** differences... **curating a stable,
> filtered data pool is far more critical than the specific multiplier sampling
> schedule.**"*

而我们的池此前**完全没有这一步**。C 的短板已被多轮实测定位在低延迟档
(RealSI k1 的中文源 CER 0.804、ASR-BLEU 10.37),正是这条消融针对的地方。

## 二、每个 loss 的作用与权重

### 族采样权重(决定每个训练块属于哪个任务)

`p2st_schedule.POOL_WEIGHTS`,与 C 完全相同:

| family | 权重 | 作用 |
|---|---:|---|
| `p2st_streaming_asr` | **0.25** | 前缀到前缀的流式识别:给定源音频前缀,预测新增的源文本 |
| `p2st_incremental_mt` | **0.25** | 增量翻译:给定已提交的源前缀,预测新增的目标文本 |
| `p2st_streaming_tts` | **0.25** | 流式合成:给定目标文本前缀,预测新增的 BiCodec 语义码 |
| `phase3_quality_replay` | **0.15** | 整句 phase3 回放,抗遗忘 |
| `phase3_performance_replay` | **0.10** | 同上,性能侧 |

三个 P2ST 族各 0.25 = 0.75,回放 0.25。**这是 C 的配比,没有改。**

### 损失项权重(本次命令行实测值)

| 损失项 | 权重 | 作用 |
|---|---:|---|
| `asr_ce` | **1.0** | 源文本增量的交叉熵 |
| `mt_ce` | **1.0** | 目标文本增量的交叉熵 |
| `semantic_ce` | **1.0** | 目标语义码增量的交叉熵 |
| `replay_ce` | **1.0** | 回放族的交叉熵 |
| `boundary_eos` | **1.0** | 片段边界终止符(`boundary_ce` + `eos_ce` 合成) |
| `content_end_ce` | **0.0** | 文本终止符,**关闭** |
| `semantic_end_ce` | **0.0** | 语义终止符,**关闭** |
| 全部 margin / rollin / binary / speaker 项 | **0.0** | 共 12 项,全部关闭 |

**这就是"纯 CE"的含义:五个 CE/EOS 项各 1.0,其余十几项全 0。**
C 赢过 m3(12 个损失 + margin 3.0)和 B′ 正是靠这个配方,所以这次一个字没动。

学习率:`--e2e-lr-qwen 2e-6`、`--e2e-lr-qwen-io 5e-7`,与 C 相同。

### 一处必须知道的聚合性质

目标函数是 `total = Σ_i (numerator_i / global_denominator_i) × scale_i` ——
**每一项都是它自己那个桶内的"每位置平均 CE",然后按权重相加**,不是按 token 数加权求和。

后果:一个只在很少 token 上触发的项,仍然贡献它的完整均值。
之前实测过在纯 TTS 窗口里 `END_SEMANTIC` 拿到的每 token 梯度是一个语音码的 **27.3 倍**。
这正是本次把所有 end/margin 项设为 0.0 的原因之一 —— 它们一旦开启就会被过度放大。

## 三、这次训练完能测吗?——能,完全独立可测

**不需要等下一阶段。** 因为步骤 1 只改数据池,推理接口与 C 完全一致,
所以现成的评测链可以原样跑,数字与 C 的历史结果逐项可比:

| 指标 | C 基线 | 步骤 1 的门线 |
|---|---|---|
| RealSI k1 ASR-BLEU | 14.66 / 10.37 | **≥ 16 / 12** |
| RealSI k1 中文源 CER | 0.804 | **≤ 0.60** |
| RealSI k1 英文源 WER | 0.421 | 不退化 |
| 首次发音(k1 + s1t0) | 1573 ms | 不退化 |

训练内已有的信号(同一验证池、同一 iteration 的**配对**比较)是 **20/20 格 NIR 更低**,
asr −5.0%、mt −3.5%、semantic −1.0%、replay −0.2%。方向一致但幅度小且在收窄,
**不足以下结论,要看 RealSI。**

## 四、下一阶段(步骤 2+3)是什么,motivation 是什么

**做什么**:把训练读点从 gold 事件边界改成**固定 640 ms chunk 网格**,
并给"这一步没有新内容"的 chunk **显式监督一个终止符**(即 IDLE/wait 步)。

**依据**(论文 §3.2 Step 2):

> *"group adjacent target words and their codes whose boundaries fall within the
> same pre-defined source chunk intervals of **1 second**... **Chunks without
> newly committed target content act as read/wait steps**; chunks containing
> target content act as write steps."*

**motivation 有三条,都已实测:**

1. **我们的训练从没见过固定读网格。** `task_samples_p2st.py:410` 是
   `for event in trajectory.events`,切在 gold 词边界;而推理必须按固定时钟读。
2. **这个错配可以量化。** 把 gold 事件重分箱到固定 chunk,IDLE 比例
   160 ms **84.1%** / 640 ms **50.7%** / 1920 ms 21.2% ——
   **在 160 ms 上六步里五步该闭嘴**,而模型从没被教过"闭嘴"是一个合法输出。
3. **它是卡顿的唯一解。** 已实测:填静音、淡化、调语速、δ 式无条件写四条推理侧办法
   全部无效或有害;空档出现的地方就是模型当时没有可说的文本
   (源播到一半时 C 只有 12% 的译文)。

**选 640 ms** 是因为 IDLE 率 50.7% 监督最平衡。

## 五、与论文训练流程的对比

| | 论文(主线 Thinker–Talker) | 论文的 Dec-only 基线 | **我们** |
|---|---|---|---|
| 架构 | 双流:Thinker 出文本,Talker 出码 | 单解码器,词表加 16,384 个 code token | **单解码器**(同 Dec-only) |
| 基座 | Qwen2.5-Omni 3B,Talker 继承其 talker body | Qwen2.5-Omni LLM body | **UniSS 派生 Qwen 0.5B** |
| 微调方式 | LoRA(阶段间融合再新建适配器) | LoRA + **双分支预热 + 加权融合** | **全参数微调** |
| 阶段 1 | Talker 预热 | **双分支 LoRA 预热**(理解 70% / 生成 90%) | 无(直接从 C 继续) |
| 阶段 2 | 多任务联合 | 融合后多任务联合 | 无 |
| 阶段 3 | 流式轨迹微调 | 同左 | **本次 = 阶段 3 的数据选择部分** |
| 读点 | **固定 1 秒 chunk** | 同左 | **gold 事件边界**(步骤 2 才改) |
| wait 步 | **显式 IDLE token** | 同左 | **无**(步骤 2 才加) |
| latency | 训练时采样 m∈{1..12},prompt 里写 `With Latency: m` | 同左 | **无**(步骤 4 才加) |
| **NIR 过滤** | **有**(§A.3) | 有 | **本次新增 ✓** |
| 难度/长度分层 | `{.1,.3,.4,.2}` × `{.1,.5,.4}` | 同左 | **逐字照抄 ✓** |
| 目标函数 | chunk 分解的联合 log-likelihood | 同左 | **纯 CE,五项各 1.0** |
| 数据量 | 约 2000 小时配对 S2ST | 同左 | 15shard 池,分层后 503,785 条 |

**所以严格说:本次只对上了表里的最后三行中的两行(NIR 过滤 + 分层采样),
其余全部不同。** 步骤 2–4 会补上读点、IDLE、latency 三行;
架构、基座、LoRA、三阶段这四行**不在移植范围**(见
`reports/uniss_streaming_p2st_realsi_v1/PAPER_READING_2607.19810.zh-CN.md`
里论文对 Dec-only 的判决,以及我们 0.5B 已与他们 3B Dec-only 持平的实测)。

**一处刻意的偏离并已记录**:分层池只有全池的 38%,而 `build_p2st_pools` 拒绝重复的
`sequence_id`,所以只能下采样而不能复制;用 `coverage_epochs=2` 把迭代数补到
3876 = C 的 92%,让优化步数可比。
