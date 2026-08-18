# UniSS Runtime-Parity Streaming 版本演进与失败根因审计

## 1. 报告目的与范围

本文总结当前 UniSS runtime-parity streaming 主线的完整演进，包括：

- dense-aligned 数据与 exact-runtime 前端基座；
- Overfit1–Overfit9；
- Generalize10–Generalize15；
- 每个版本要解决的具体问题、motivation、实际修改、初始化来源、实际完成状态、真实结果以及进入下一版的原因；
- 为什么经过多次迭代，仍没有得到一个在未见输入上同时满足翻译质量、语音质量、自然低延迟 WRITE、自然 EOS、零 revision 和实时吞吐的 streaming/simultaneous S2ST 模型。

本文不总结更早的 wait-k、StreamSpeech/CTC、GRPO 或其他独立路线；它们属于不同实验主线。本文也不是代码或运行教程，不包含实现代码。

审计时间：2026-08-12。

## 2. 一句话结论

当前主线并不是“同一种训练连续失败了十五次”。它通过一系列隔离实验，已经分别证明：

- 单条训练轨迹可以学会完整翻译、semantic、自然 WRITE 和自然 EOS；
- 等价 runtime 优化可以把单条轨迹的首 PCM 和 RTF 压到实时门内；
- deadline 与 model-prefix exposure 可以让 seen 和 held-out 样本都在 320–480ms 自然 WRITE；
- teacher-forced action/EOS 指标可以在验证集上明显改善。

但是，这些能力至今没有在未见输入上同时成立。最核心的阻塞点不是 shuffle、训练 epoch 数或某一个 loss 权重，而是：

> 训练长期在 oracle action/text/semantic 历史上计算损失，真实推理却把模型自己的早期错误写入 persistent KV；第一步内容错误后，后续 action、文本、semantic、长度和 EOS 所依赖的 hidden state 全部偏离训练分布。当前主线尚未真正完成 exact event-level、model-induced persistent-KV rollout 下的联合纠错训练。

Generalize14 只做了固定位置、固定长度的 token 级 prefix 替换，并不等价于完整事件 rollout；Generalize15 又冻结了文本和 semantic 内容路径，只校准 action/EOS，因此仍未触及上述核心问题。

## 3. 什么才算当前意义上的“训练成功”

早期实验的 quality_passed 标记使用过较宽松的门，不能与后期严格门直接等价。按照当前严格 real-PCM gate，一个可用的 streaming S2ST checkpoint 至少必须同时满足：

| 维度 | 当前严格含义 |
|---|---|
| WRITE | 模型自然 argmax 选择 WRITE，不允许 forced WRITE 或降低阈值冒充成功 |
| 首次源时间延迟 | 第一次自然 WRITE 出现在小于 1 秒的 source prefix |
| 首次可播放延迟 | 第一段真实可播放 PCM 的 wall-clock 时间小于 1 秒 |
| 文本内容 | 翻译不能只是通用短词；严格样本门通常要求与参考文本的序列相似度至少 0.98 |
| semantic/audio | 必须产生可播放、非单一 code 塌缩的目标语音 |
| 终止 | 必须自然选择 EOS，不能依靠固定 drain 上限截断 |
| revision | 已提交内容不得回改，当前门要求 revision 为 0 |
| 实时吞吐 | RTF 必须小于 1，不能只做到“很早开始、但越播越积压” |
| 泛化 | 上述条件必须在未见输入上成立，单训练样本记忆不算泛化成功 |

因此：

- Overfit9 是单训练轨迹严格成功；
- Generalize14 是自然低延迟策略成功，但端到端质量和吞吐失败；
- Generalize15 是 teacher-forced 部分策略校准成功，但真实 free-running 内容和 EOS 失败；
- 当前仍没有通过 held-out 端到端严格门的最终模型。

## 4. 前置问题：为什么需要重新构造 dense/runtime-parity 基座

早期 streaming 数据本身存在结构性缺陷。现有审计显示：

- 60.842% 的 session 没有 natural WRITE；
- 平均每句只有约 125.77ms 的目标音频真正受到 streaming semantic loss 监督；
- text delta 与 semantic span 没有形成可靠的一一对应关系；
- forced WRITE 往往只有软文本 target，没有对应音频监督；
- observation 并非从句首开始每 160ms 连续推进到句尾；
- 部分目标 semantic 覆盖存在缺失，无法保证零 gap、零 overlap；
- 训练和部署在 causal frontend、speaker 状态、persistent history、KV 行为和 codec 状态方面并不完全一致。

这意味着旧数据即使 loss 下降，也可能只是在学稀疏的局部片段，而不是学完整的“READ/WRITE—文本—semantic—继续/终止”轨迹。

### 4.1 Dense-aligned 数据基座的 motivation

目标是先消除“数据监督缺失”和“训练/部署输入不一致”这两个最基础的混淆因素，再判断模型本身是否可学。

### 4.2 主要修改

- 内部 policy tick 固定为 160ms；
- 从句首到句尾连续建立 observation；
- 为目标文本和 semantic 提供完整覆盖；
- semantic span 要求无 gap、无 overlap；
- 保留完整 session 内事件顺序，只在完整 pack/session 级做全局 shuffle；
- 保留 Phase3 replay 作为离线翻译和语音质量锚点；
- 从 Phase3 v4 最佳 Megatron checkpoint 初始化；
- 使用 causal WhisperVQ trace、persistent history grammar 和与部署一致的 runtime 表示。

### 4.3 Pilot15 基座实际状态

fixed 15-shard dense-aligned pilot 的数据规模为：

- train sessions：1,325,243；
- validation sessions：13,469；
- Phase3 初始化：iteration 9075；
- 训练：8 张 H200、sequence length 18000、micro batch 2、global batch 128；
- 设计覆盖：三次完整 trajectory epoch；
- 完成的 dense pilot checkpoint：iteration 2151。

这个基座解决了数据覆盖、全局 shuffle 和前端/runtime parity 的第一层问题，但它本身没有证明 free-running held-out streaming 已经成功。后续 Overfit/Generalize 主线就是在这个基座上逐层定位剩余问题。

## 5. 总版本时间线

| 版本 | 实验性质 | 主要问题 | 核心修改 | 最重要结论 |
|---|---|---|---|---|
| 基座 | 数据与前端修复 | 监督稀疏、训练/runtime 不一致 | dense 160ms 轨迹、causal frontend、exact grammar | 排除最基础的数据覆盖问题 |
| Overfit1 | 单轨迹能力诊断 | 普通目标是否能记住完整 grammar | 复用完整 dense objective | 会 WRITE，但文本和 semantic 循环 |
| Overfit2 | 单轨迹 loss 重加权 | 内容和 STOP 被普通 token 稀释 | dropout=0，强化 text/semantic/EOS | 单样本内容和 EOS 可学，仍慢 |
| Overfit3 | 单轨迹 boundary 修复 | source end 后过早 EOS | 同时强调 START_GLM continuation 与 EOS | continuation/EOS grammar 可学 |
| Overfit4 | 单轨迹内容巩固 | boundary 过强导致 drain 内容 underfit | 降 boundary、提高 text/semantic 权重 | 单样本内容完全正确，但 AR 太慢 |
| Overfit5 | semantic 速度实验 | 逐 unit AR 生成太慢 | 24-slot 并行 semantic block | 快速结构难学，content/END 耦合失败 |
| Overfit6 | semantic head 隔离 | frozen tied classifier 适应性不足 | untied classifier，只训新 head | 内容可记忆，但自然 END 失败 |
| Overfit7 | semantic 长度实验 | 固定/隐式长度不自然 | content posterior + length posterior | 正式训练未完成，无法下最终结论 |
| Overfit8 | semantic 长度隔离 | 联合训长度可能破坏已学内容 | 保留 v6 content，只训 length head | 单样本仅差一点 RTF |
| Overfit9 | runtime-only 优化 | 同一 block 多一次 causal forward | semantic block 与 END 一次提交 | 单样本严格通过，held-out 崩溃 |
| Generalize10 | 小规模泛化 | 单样本 head 不泛化 | 128 sessions，仅训 semantic head | 低延迟/EOS 可用，但文本只输出“我” |
| Generalize11 | fixed15 扩大数据 | 怀疑 v10 数据太少 | 59,576 packs、全局 shuffle 一轮 | 仍输出“我”，证伪“只缺数据量” |
| Generalize12 | semantic 结构重构 | 独立 slots 产生高频 code 塌缩 | 4-unit causal microblock、自然 CONTINUE/END | semantic 多样性改善，文本路径仍错 |
| Generalize13 | 联合内容训练 | 冻结 text 无法修翻译 | 联训 LoRA、action、text、semantic | teacher-forced 改善，runtime 变成全 WAIT |
| Generalize14 | prefix exposure | oracle prefix 与 model prefix 错位 | DAgger-style token roll-in、deadline | 低于 1 秒自然 WRITE 成功，内容/RTF 失败 |
| Generalize15 | action/EOS 校准 | WRITE 标签和 source-end continuation 不一致 | 修标签，只训 action + continuation | teacher-forced EOS 学会，runtime 内容/EOS 仍失败 |

## 6. 各版本详细审计

### 6.1 Overfit1：先问“一条完整轨迹能不能学会”

#### Motivation

在扩大数据和继续调参之前，必须先回答最基本的问题：从 Phase3 v4 开始，模型能否在 exact runtime grammar 下记住一条完整 trajectory。如果连一条样本都无法学会，就说明 objective、标签、导出或 runtime 本身仍有错误。

#### 实际修改

- 复用 dense-aligned 的完整多任务 objective；
- 只使用一条 trajectory 做高覆盖 overfit；
- 从 Phase3 v4 checkpoint 初始化；
- 采用 sequence length 18000、micro batch 2、global batch 128；
- 原计划做 100 coverage epochs，实际完成 100 iteration。

Overfit1 没有独立的 pretrain 实现，而是复用根目录的 dense-aligned 训练入口。

#### 真实结果

原始评估：

- 生成文本：“你好。你好。你好。你好。你好。”；
- 文本相似度：0.2162；
- natural WRITE：46；
- semantic tokens：3456；
- RTF：27.36。

修复导出/pre-norm/runtime 语义后的重新评估并没有变成成功：

- 生成文本为空；
- natural WRITE：42；
- semantic tokens：1568；
- RTF：14.37。

#### 结论

模型已经会触发 WRITE，但没有学会稳定的文本、semantic 内容和终止。重复文本和超长 semantic 表明普通 token 占据了主要训练质量，稀有边界和 STOP 监督不足。两个不同 runtime 结果还说明早期导出/runtime parity 本身也会改变结论，不能把训练 loss 当成唯一依据。

### 6.2 Overfit2：强化内容、semantic 和 STOP

#### Motivation

Overfit1 的主要症状是循环、超长输出和不终止，因此 v2 的目标不是泛化，而是验证模型容量是否足以在一条样本上同时学会内容和 STOP。

#### 实际修改

- dropout 设为 0，去掉单样本记忆中的随机干扰；
- 增大 text、action、semantic 和关键 boundary 的权重；
- 对 END_CONTENT、END_SEMANTIC 和 EOS 单独加大权重；
- 降低或关闭与单样本能力诊断无关的 curriculum 项；
- replay 比例降到 10%；
- 从 Phase3 v4 重新初始化，而不是接 Overfit1 的错误状态。

配置曾规划 500 iteration，但实际 checkpoint 只完成到 iteration 100，日志随后进入 101。因此本版应称为“100 iteration 能力诊断”，不能写成完整 500 iteration 实验。

#### 真实结果

iteration 100：

- 参考：“你好，我在寻找一本关于正念的书。你能帮我吗？”；
- 生成：“你好，我在寻找一本关于正念的书。你能帮”；
- 文本相似度：0.9268；
- natural WRITE：10；
- semantic tokens：160；
- natural EOS：是；
- RTF：2.65。

旧 evaluator 给出了 quality_passed=true，但当时没有使用后来完整的首 PCM、相似度 0.98 和 RTF<1 严格门，因此不能把它视为当前标准的通过。

#### 结论

这版非常重要：它证明模型容量和基础 grammar 并非完全不可学。问题从“模型是否能学”收缩为“如何学完整句尾、如何更快、如何泛化”。

### 6.3 Overfit3：同时学习 source-end continuation 与最终 EOS

#### Motivation

Overfit2 强化 EOS 后，模型容易在 source 刚结束时直接终止，但此时目标翻译和目标语音可能尚未 drain 完。模型需要区分：

- source 已结束，但还要选择 START_GLM 继续输出剩余目标；
- 所有目标已经完成，此时才选择 EOS。

#### 实际修改

- 对 START_GLM continuation 和最终 EOS 给予相同强调；
- 让上下文状态决定“继续 drain”还是“真正结束”；
- 从 Phase3 v4 重新 finetune，而不是从 Overfit2 checkpoint 继续；
- 实际完成 120 iteration。

#### 真实结果

- 文本相似度：0.9048；
- natural EOS：是；
- 首 PCM wall time：约 1915ms；
- RTF：2.87。

#### 结论

单样本上 natural continuation/EOS grammar 可以学会，但内容略有退化，速度仍远离实时。下一版需要在不破坏 grammar 的前提下，把训练质量重新分配给文本和 semantic 内容。

### 6.4 Overfit4：巩固文本与 semantic 内容

#### Motivation

Overfit3 对稀有 boundary 的强调过大，使模型更会“决定继续或结束”，但 drain 阶段的具体文本和 semantic 内容 underfit。

#### 实际修改

- 从完成的 Overfit3 checkpoint 继续；
- boundary continuity 总权重从 4 降到 0.5；
- text token 权重从 8 提高到 16；
- semantic token 权重从 2 提高到 4；
- START_GLM、EOS、END_CONTENT 和 END_SEMANTIC 仍保留监督，但不再压过内容；
- 对 runtime 做 dynamic/fused 等价优化；
- 拒绝了非等价的 static-KV “加速”，避免用改变模型决策的错误路径冒充性能提升。

#### 真实结果

未经正确等价 runtime 评估时，曾出现长乱码、相似度 0.03、RTF 9.22。修正 dynamic/fused runtime 后：

- 完整生成参考文本；
- 文本相似度：1.0；
- semantic tokens：178；
- natural EOS：是；
- 首 PCM：约 1.84–1.96 秒，当前选定 summary 为 1960ms；
- RTF：约 2.47–2.52。

#### 结论

这版证明了单条 trajectory 上，文本、semantic 和 grammar 可以一起记住。剩余瓶颈转为逐 semantic token 自回归生成的计算成本：内容正确，但每次 WRITE 需要太多 forward，无法实时。

### 6.5 Overfit5：24-slot 并行 semantic block

#### Motivation

Overfit4 的 AR semantic 生成正确但慢。v5 尝试一次预测最多 24 个 semantic unit，用并行 block 替代逐 unit 自回归，从结构上减少每次 WRITE 的 forward 数。

#### 实际修改

- 从 Overfit4 初始化；
- 增加最多 24 slot 的 parallel semantic block head；
- 把 END 作为额外类别，与 semantic content 一起预测；
- 保留 Overfit4 已学的 grammar 和原 AR semantic anchor；
- 提高新 block head 的 loss 权重和学习率。

配置规划 300 iteration，但实际 checkpoint 只到 100，日志进入 iteration 101 后停止；没有形成可靠的最终 strict PCM 报告。

#### 训练诊断

iteration 100 附近：

- semantic token accuracy 仍为 0；
- END accuracy 约 0.875；
- length MAE 约 14.16。

这意味着 head 更容易学习“何时 END”，却没有学会每个 slot 的真实 semantic 内容。

#### 结论

独立并行 slots 缺少槽位间的因果依赖，而且 classifier 绑定到冻结的 Phase3 embedding rows，适应新 block 预测任务的能力不足。该版主要证明“直接把 AR 改为 24 个独立并行分类”并不能自然得到高质量 semantic。

### 6.6 Overfit6：解开 classifier，并冻结已成功的 v4

#### Motivation

需要判断 v5 失败究竟来自整个模型被破坏，还是新 semantic head 本身不够可学。v6 将问题隔离为：保留 Overfit4 的成功内容和 grammar，只训练一个 untied semantic classifier。

#### 实际修改

- 从 Overfit4 初始化，而不是接 Overfit5；
- 冻结 Overfit4 的全部已有参数；
- semantic classifier 不再绑定冻结的 Phase3 embedding rows；
- 只训练新的 parallel semantic head；
- 使用较高的新 head 学习率；
- 完成 150 iteration。

#### 真实结果

训练样本上的 semantic accuracy 可以升到约 0.865，说明 untied head 的可学习性明显提高。但严格 runtime evaluator 无法得到自然 END_SEMANTIC，并直接以“parallel semantic head did not naturally select END_SEMANTIC”失败；没有可作为最终端到端结论的 strict PCM summary。

#### 结论

semantic 内容记忆与自然长度/停止是两个不同问题。仅让 classifier 更可训练，可以记住内容，却不能保证自然选择输出长度和 END。

### 6.7 Overfit7：显式 natural-length posterior

#### Motivation

v6 的 content head 可以学，但固定长度或通过 END 混在 8192-way content 分类中学习不稳定。v7 将“输出什么 unit”和“输出多少 unit”拆成两个 posterior。

#### 实际修改

- 从 Overfit6 初始化；
- 保留 semantic content posterior；
- 新增显式 1–24 natural-length posterior；
- 同时训练 semantic content loss 与 semantic length loss。

配置规划 300 iteration，但正式 checkpoint 只到 50，日志进入 52；现有 2-step 报告只是 loader smoke，不是正式性能评估。因此不能把 v7 写成已经完成并通过或失败的端到端版本。

#### 结论

联合训练 content 和 length 有可能破坏 v6 已经记住的 semantic 内容。由于正式训练和严格评估不完整，v7 的价值主要是暴露了“长度学习应与 content 保护隔离”的需求，随后进入 v8。

### 6.8 Overfit8：只训练 length head

#### Motivation

保留 v6 已经学会的 semantic content，不让新的长度目标反向破坏 content。

#### 实际修改

- 从 Overfit6 初始化，而不是从不完整的 v7 初始化；
- 冻结 Phase3、文本、action、semantic content 等所有已有参数；
- 只训练 natural-length head；
- 完成 50 iteration。

#### 真实结果

- 文本相似度：1.0；
- natural WRITE：11；
- semantic tokens：178；
- first source WRITE：640ms；
- first PCM wall time：906.94ms；
- natural EOS：是；
- RTF：1.0178；
- 严格失败原因只有 RTF 未小于 1。

#### 结论

单条训练轨迹上，grammar、文本、semantic 内容和自然长度已经可以组合起来，距离严格门只差少量 runtime 性能。这是进入 v9 的直接依据。

### 6.9 Overfit9：不训练，只做等价 runtime 融合

#### Motivation

v8 的 RTF 仅略高于 1。分析发现 semantic block 和 END_SEMANTIC 被分成两次 causal forward，存在可以消除的等价计算开销。

#### 实际修改

- 不重新训练，也不改变 v8 checkpoint；
- 在同一个 causal forward 内提交 semantic block 和 END_SEMANTIC；
- canonical transcript、action 和 semantic 决策保持一致；
- 用三次重复严格评估检查稳定性。

#### 单训练轨迹结果

| 重复 | 文本相似度 | 首 PCM | natural EOS | RTF | 严格结果 |
|---|---:|---:|---|---:|---|
| 1 | 1.0 | 895.63ms | 是 | 0.9867 | 通过 |
| 2 | 1.0 | 880.57ms | 是 | 0.9375 | 通过 |
| 3 | 1.0 | 877.68ms | 是 | 0.9151 | 通过 |

#### Held-out 结果

- 生成文本出现“我来拿……一本一本……”等循环；
- 文本相似度：0.0541；
- natural WRITE：35；
- semantic tokens：514；
- 首 PCM：612.68ms；
- natural EOS：否；
- RTF：1.1876。

#### 结论

Overfit9 是整个主线的分水岭：

- 它严格证明单样本记忆和等价 runtime 加速可以同时成功；
- 它也严格证明单样本通过完全不代表新输入泛化；
- 后续问题不再是“能否记住一条轨迹”，而是“如何在新 source prefix 上生成正确文本和 semantic”。

### 6.10 Generalize10：小规模 exact-runtime semantic 泛化

#### Motivation

从单样本过渡到多样本，判断自然长度 semantic head 是否能在 exact runtime trajectory 上泛化。

#### 实际修改

- 不从 Overfit9 开始，而是从完成的 dense-aligned fixed15 pilot checkpoint iteration 2151 重启；
- 冻结 Phase3、action、text 和 frontend；
- 只训练 natural-length parallel semantic content/length head；
- 训练数据为 5 个 18k packs，共 128 个 exact-runtime sessions；
- validation 为 2 个独立 packs，共 32 个 sessions；
- 完成 300 iteration。

#### Held-out 结果

| checkpoint | 生成文本 | 相似度 | 首 PCM | natural EOS | RTF |
|---|---|---:|---:|---|---:|
| iter 50 | 我 | 0.0526 | 536ms | 是 | 1.020 |
| iter 300 | 我 | 0.0526 | 742ms | 是 | 0.909 |

#### 结论

semantic、natural EOS 和实时速度可以部分工作，但被冻结的 text path 只输出通用首 token“我”。只训练 semantic head 没有任何梯度路径可以修复翻译文本，因此继续堆 semantic epoch 不可能解决根本问题。

### 6.11 Generalize11：fixed15 全部 dense packs

#### Motivation

Generalize10 只有 128 sessions，可能是数据太少导致 semantic collapse。v11 用 fixed15 全部 dense trajectory 验证“只是数据量不足”这一假设。

#### 实际修改

- 同样从 dense pilot checkpoint 重启；
- 仍冻结 Phase3、action、text、frontend 和 speaker 路径；
- 只训练 parallel semantic content/length head；
- 使用 59,576 个完整 18k trajectory packs；
- 对完整 pack 做一次严格全局 permutation；
- GBS 尾部 padding 后 epoch samples 为 60,288，对应 471 iteration；
- replay 保留 1%，但冻结 base 不接收 replay 梯度。

#### Held-out 结果

| checkpoint | 生成文本 | 相似度 | 首 PCM | natural EOS | RTF |
|---|---|---:|---:|---|---:|
| iter 100 | 我 | 0.0526 | 530ms | 是 | 0.867 |
| iter 300 | 我 | 0.0526 | 542ms | 是 | 0.871 |
| iter 471 | 我 | 0.0526 | 538ms | 是 | 0.858 |

#### 结论

扩大到 fixed15 全部 dense packs 仍无法改变文本只输出“我”。这不是说更多数据永远无用，而是严格证伪了：

> 在冻结 text path、只训练当前 semantic head 的 objective 下，单纯扩大 semantic trajectory 数据即可修好端到端翻译。

因此当前问题不能再归因于 shuffle 或 v10 样本量太少。

### 6.12 Generalize12：4-unit causal semantic microblock

#### Motivation

v11 的 24 个独立 slots 容易全部选择同一高频 code。需要让后续 semantic block 依赖已经提交的 semantic history，同时仍比逐 unit AR 更快。

#### 实际修改

- 从 dense pilot checkpoint 重启；
- 冻结 Phase3、LoRA、frontend、action、safe-commit、text 和 speaker；
- 每次预测 4-unit causal microblock；
- block 之间依赖已经提交到 persistent KV 的 semantic；
- block 内部用小型 causal transition 依赖前一个 unit；
- classifier 重新 tied 到 Phase3 semantic embedding，并从 Phase3 next-token classifier 初始化；
- 使用受限的 inverse-square-root class weighting 抑制高频 code 支配；
- 单独预测 CONTINUE/END；
- 最后一个 block 再预测 1–4 的自然长度；
- safety ceiling 只判失败，不把强制截断计为成功。

#### 实际完成状态

- canary 完成 200 iteration；
- full15 的数据/目录准备过，但 checkpoint 目录为空；
- 因 canary 未过门，正式 full15 没有启动；
- 本主线没有进行 full198 runtime-parity 正式训练。

#### 严格结果

held-out：

- 文本相似度约 0.074–0.098；
- first source WRITE：320ms；
- 首 PCM：约 624–640ms；
- natural EOS：是；
- RTF 约 0.789–1.299；
- semantic diversity 比独立 slots 更好，但 token accuracy 约 1.8%，first-slot accuracy 约 4.6%。

seen train 样本也只输出“我”，相似度约 0.054–0.087。

#### 结论

causal microblock 是比 24 个独立 slots 更合理的 semantic 结构，也改善了 unit 多样性和自然长度行为。但是被冻结的 text path 仍然错误，且 semantic head 在错误文本/runtime state 上无法形成高质量条件生成。即使 seen 样本也只输出“我”，排除了“只是 held-out checkpoint 选错”的解释。

### 6.13 Generalize13：联合训练 runtime 文本、action 和 semantic

#### Motivation

Generalize12 已经明确证明：冻结 text path、只修 semantic 不可能得到正确翻译。v13 因此开始联合适配 runtime 内容路径。

#### 实际修改

- 从 Generalize12 canary iteration 200 初始化；
- 保留 4-unit causal semantic microblock；
- 联合训练 Qwen LoRA；
- 联合训练 action、support、safe-commit heads；
- 训练 runtime text content 和 critical boundary；
- 训练 semantic microblock content、length 和 CONTINUE/END；
- Phase3 base、embedding/output matrix 和 causal frontend 继续冻结；
- Phase3 replay 保留 10%。

第一版 canary_v1 同时建立了两份 full-vocabulary CE graph，导致 H200 OOM，只作为诊断残留。有效版本是共享同一份 CE tensor 的 canary_v2，完成 200 iteration。

#### Teacher-forced 结果

held-out 的最佳点约为 iteration 50：

- text accuracy：22.43%；
- text loss：5.6324；
- action accuracy：85.57%。

iteration 200 时，训练 text accuracy 已升到 76.62%，但 held-out text loss 已恶化到 7.8157，显示快速记忆和过拟合。

#### Strict free-running 结果

held-out：

- iter 50/75：一个样本直到 source 6560ms 才第一次 WRITE，另一个完全不 WRITE；
- iter 200：两个样本都 0 WRITE、空输出。

seen train：

- 多数 checkpoint 也是空输出；
- iter 200 只有一个样本输出“你好，”，相似度 0.24，首 PCM 1074ms。

#### 结论

这是 teacher forcing 与 free-running 分布错位的第一组直接强证据：

- 在 oracle packed history 上，action accuracy 可以达到约 84%–86%；
- 一旦 runtime 把自己的 token 写入 persistent KV，模型却几乎全 WAIT；
- 同一个聚合 action 指标并不能预测真实 runtime action。

此外，v13 的 deadline survival 权重为 0，而严格门要求 1 秒内 WRITE，也加剧了“等到句尾”的倾向。

### 6.14 Generalize14：DAgger-style model-prefix roll-in

#### Motivation

Generalize13 训练时始终看到 oracle action/text/semantic prefix，runtime 却看到自己的 prefix。v14 尝试让训练主动暴露于部分 model-induced history。

#### 实际修改

- 从 Generalize13 最佳 held-out 点 iteration 50 初始化；
- 先做无梯度 probe，预测 runtime text 和 semantic token；
- 按 schedule 将部分预测 token 等长替换进 input prefix；
- 保持原 packed sequence 长度和 session boundary 不变；
- 后期增加第二轮 probe；
- 对污染后的 state 仍使用 oracle label 做 DAgger-style correction；
- 新增 runtime-prefix recovery CE；
- 开启 grouped soft/hard deadline survival；
- 继续联合训练 Qwen LoRA、policy heads 和 semantic microblock；
- corruption 后期最高达到约 50%–62% 的实际位置比例。

#### 真实结果

8 次 train/held-out strict trial 全部实现：

- natural first WRITE：320–480ms；
- first PCM wall time：599–901ms；
- natural EOS：是；
- committed revision：0。

但同时：

- 文本相似度仅 0.065–0.20；
- semantic 经常塌缩到 code 7645；
- RTF 为 1.11–1.44；
- iteration 50 的内容通常优于 iteration 200；
- corruption 继续增大后，held-out text/semantic accuracy 明显恶化。

#### 结论

Generalize14 不是“完全没用”。它严格证明 deadline objective 加上 model-prefix exposure，能够把 v13 的全 WAIT 修成 seen/held-out 都低于 1 秒自然 WRITE。

它失败在三个方面：

1. 第一批提交内容已经错误；
2. semantic 仍塌缩；
3. 每个 160ms tick 的 text/semantic 计算会产生 backlog，使 RTF>1。

更深层的问题是：v14 只是固定位置的 token replacement。真实 runtime 会同时改变：

- 本 tick 是 WAIT 还是 WRITE；
- 一次生成多少 text token；
- 一次生成多少 semantic unit；
- semantic 选择 CONTINUE 还是 END；
- source end 后选择 START_GLM 还是 EOS；
- 后续 persistent KV 的实际长度和内容。

因此 v14 学到的是“在 oracle grammar 中修复随机 token 污染”，不是“在完整自生成事件历史上恢复正确轨迹”。

### 6.15 Generalize15：action 标签与 EOS continuation 校准

#### Motivation

v14 暴露出两个更具体的 policy 问题：

1. 数据把同一目标词的 semantic continuation tick 重复标成顶层 WRITE，但 runtime 一次 WRITE 已经生成完整 microblock chain，后续 continuation tick 在顶层应视为 WAIT；
2. source end 后不能无条件 EOS，需要单独判断继续 START_GLM drain 还是最终 EOS。

#### 实际修改

- 从 Generalize14 iteration 50 初始化；
- semantic-only continuation WRITE 折叠为 WAIT；
- 新增 START_GLM-versus-EOS continuation head；
- WAIT false-positive 权重设为 2；
- action WRITE 权重降为 0.5；
- EOS class 权重设为 8；
- prefix roll-in 上限降到一次 10%；
- 只训练 action head 和 continuation head；
- 冻结 Phase3 base、Qwen LoRA/text、frontend、semantic microblock、support/safe-commit、embedding/output；
- 完成 100/100 iteration，无 skipped update、无 NaN。

需要特别指出：Generalize14 的严格报告建议下一版实现完整 event-level runtime rollout，但 Generalize15 实际没有实现该建议，而是先做了局部 action/EOS 校准。这意味着最核心的 state-distribution 问题仍然存在。

#### Teacher-forced validation

| iteration | predicted WRITE | target WRITE | WRITE precision | false-positive rate | EOS precision | EOS recall |
|---|---:|---:|---:|---:|---:|---:|
| 25 | 25.74% | 23.88% | 44.65% | 18.77% | 0 | 0 |
| 100 | 9.46% | 23.88% | 21.61% | 9.72% | 74.29% | 62.64% |

这显示训练过程中出现明显 trade-off：

- 早期 WRITE 比例接近 target，但 EOS 完全不会；
- 后期 teacher-forced EOS 学会了，false positive 下降，但 WRITE 比例收缩过度，precision 也下降。

#### Strict free-running real-PCM 结果

- iteration 25：4 个 train/held-out trial 全部 0 WRITE、0 audio；
- iteration 50–100：first source WRITE 为 320–480ms；
- 首 PCM 为约 599–823ms；
- RTF 基本小于 1；
- 生成文本仅为“我能”“我喜欢”“我明白吗？”等 1–4 个短词；
- 相似度约 0.0476–0.1667；
- 所有 train/held-out trial 都没有 natural EOS；
- 全部达到 16 个 drain ticks；
- iteration 100 的 runtime EOS probability 只有约 0.00025–0.00038。

#### 结论

不能说“EOS head 没学会”。更准确的结论是：

- EOS head 在 oracle 完整目标 prefix 的 teacher-forced hidden state 上学会了；
- runtime 只生成了错误的短文本，永远到不了训练中表示“目标内容已经完整”的 EOS state；
- 因此真实 runtime EOS 概率极低，持续选择 START_GLM；
- 同时 content path 被冻结，v15 没有能力修复“我能/我喜欢”这些错误文本。

这是当前主线中 teacher-forced 指标与真实运行状态不一致的最直接证据之一。

## 7. 为什么迭代这么多版本仍然没有训练出来

### 7.1 端到端成功是多个耦合问题的乘积

真正的 simultaneous S2ST 不是一个二分类 WRITE loss。它要求模型同时解决：

1. 何时 WRITE；
2. WRITE 什么目标文本；
3. 为该文本生成什么 semantic；
4. semantic 生成多长；
5. 何时继续、何时自然 EOS；
6. 是否能在每 160ms 输入 cadence 内完成全部计算。

任何一个环节失败，端到端都失败。例如：

- v4 内容正确，但太慢；
- v9 单样本内容和速度都正确，但不泛化；
- v12 semantic 结构更合理，但文本被冻结且错误；
- v13 teacher-forced 内容指标改善，但 runtime 不 WRITE；
- v14 WRITE 延迟成功，但内容和 semantic 错；
- v15 RTF 和首 PCM 可过门，但文本覆盖和 EOS 失败。

### 7.2 最大根因：oracle teacher forcing 与真实 persistent-KV history 错位

训练时，后续 tick 通常看到正确的：

- action；
- text token；
- semantic unit；
- block length；
- CONTINUE/END；
- START_GLM/EOS。

真实推理时，后续 tick 看到模型自己的输出。若 320ms 时第一个词错成“我能”，这个错误会进入 persistent KV。接下来：

- action head 在错误 history 上决定 WAIT/WRITE；
- text path 在错误翻译前缀上继续；
- semantic head 在错误文本与 semantic history 上生成音频；
- length head 在错误上下文上预测长度；
- EOS head 永远到不了“完整正确目标已经结束”的状态。

G13 中“teacher-forced action accuracy 约 84%，runtime 却 0 WRITE”和 G15 中“teacher-forced EOS recall 62.64%，runtime EOS probability 约 3e-4”是这一问题的直接证据。

### 7.3 Generalize14 并没有真正完成 event-level DAgger

v14 只在保持 oracle grammar 和固定 sequence length 的情况下替换部分 token。它没有让模型真正经历：

- 自己先选 WAIT 或 WRITE；
- 自己决定本次文本长度；
- 自己决定 semantic block 数和最终长度；
- 自己决定 CONTINUE/END；
- 自己决定 source-end continuation/EOS；
- 然后从这个完整事件状态接受 oracle correction。

因此它解决了“模型从未见过错误 token”中的一部分，却没有解决“模型从未见过自己造成的完整事件结构”。

### 7.4 多次冻结策略造成“修 A 不可能修 B”

冻结策略对定位问题是必要的，但也产生明确上限：

- G10/G11/G12 冻结 text path，只训练 semantic，因此无论 semantic loss 多低都不可能修复“我”；
- G15 冻结 text 和 semantic，只训练 action/EOS，因此不可能修复“我能/我喜欢”；
- v6/v8 的冻结适合证明单样本 head 能力，却不代表能联合泛化。

这些实验不是无意义，而是严格证明了局部 head 能做什么、不能做什么。但如果把局部诊断 checkpoint 当成最终训练路线，就会出现“某个指标变好，端到端仍不动”的现象。

### 7.5 Semantic decoder 存在准确率、塌缩和速度三难

| semantic 方案 | 优点 | 主要问题 |
|---|---|---|
| 逐 unit AR | 单样本内容最容易正确 | forward 数太多，RTF 2–27 |
| 24 独立 slots | 理论上快 | slots 缺少因果关系，容易同码塌缩 |
| untied classifier | 新任务更可学 | 容易记忆单样本，仍不会自然 END |
| 显式 length posterior | 可学习自然长度 | 长度正确不代表 semantic 内容正确 |
| 4-unit causal microblock | 比独立 slots 更符合生成过程 | held-out token accuracy 仍低，依赖错误 runtime/text state |

因此 semantic 问题不是只加一个 length loss 就能解决。它必须在真实 model-induced event state 上与文本和 policy 联合学习，同时保留 anti-collapse 和离线质量锚点。

### 7.6 Action 标签曾与 runtime 语义不一致

semantic continuation tick 被重复标为顶层 WRITE，使训练 target WRITE 比实际 runtime 所需更密。G15 修正后，模型的 predicted WRITE 比例从 25.74% 收缩到 9.46%，说明旧标签确实影响策略。

但标签修复并不能自动修复翻译内容。甚至当 content path 冻结时，减少 WRITE 会让错误短文本覆盖更少，形成“速度和 false positive 变好、翻译更不完整”的新问题。

### 7.7 EOS 不是独立于内容的简单分类

EOS 的正确条件不是“source 已经结束”，而是：

- source 已结束；
- 剩余目标文本已经 drain 完；
- 对应 semantic 已经 drain 完；
- 当前 persistent KV 表示一个完整、可结束的目标历史。

如果 runtime 只生成“我能”，它不应与 oracle 完整句末 hidden state 相同。继续增加 EOS class weight，可能只会让 teacher-forced EOS 更好，或造成更早截断；不能让错误内容自动变完整。

### 7.8 低延迟和内容质量存在真实冲突

deadline loss 可以迫使模型在 320–480ms WRITE，但此时 source evidence 可能还不足以确定正确目标词。当前又要求零 revision：

- 过早 commit：容易 hallucinate，错误进入 KV 后不可修复；
- 过于保守：会退化为全 WAIT 或 source end 后才 WRITE；
- 只调 action threshold：只能改变早晚，不能产生缺失的翻译证据。

G13 和 G14 正好展示了冲突两端：

- G13：保守、几乎不说；
- G14：很早说，但说错。

### 7.9 数据量不是唯一问题，shuffle 也不是当前根因

G11 已使用 fixed15 的 59,576 个完整 18k packs，完成一次严格全局 shuffle epoch，仍只输出“我”。这说明：

- v10 的 128 sessions 确实很小；
- 但在 text path 冻结、state objective 错位的前提下，扩大 semantic 数据不会修好翻译；
- 当前问题不能简单归因于“没有 shuffle”。

同时也要保持谨慎：runtime-parity 主线没有完成 full198 正式训练，因此不能声称 full198 已经被证伪。正确表述是：在 canary/固定15 都没有通过机制门之前，直接扩大 full198 只会放大计算成本，不会自动修复 objective。

### 7.10 计算吞吐是独立硬约束

“低于 1 秒开始”与“整段实时处理”是两个指标：

- v8 首 PCM 约 907ms，但 RTF 1.0178；
- v9 通过等价融合才把单样本 RTF 稳定降到 1 以下；
- v14 首 PCM 小于 1 秒，但 RTF 1.11–1.44，说明后续不断积压。

如果每个 160ms tick 的观察、文本投影、semantic microblock、codec 和 continuation 总耗时长期超过 160ms，系统即使早说第一段，也会越来越落后。

### 7.11 评价门在演进中逐步严格化

早期实验主要回答“能否自然 WRITE/STOP、能否记住内容”，部分 summary 的 quality_passed 没有检查当前全部条件。后期才逐步加入：

- natural action；
- real PCM；
- 首 PCM wall time；
- similarity 0.98；
- natural EOS；
- revision=0；
- RTF<1；
- held-out strict free-running。

因此早期“通过”不能与当前端到端通过混为一谈。

### 7.12 Canary 很适合证伪，但不足以证明泛化

G12–G15 主要使用：

- 5 个 train packs，共 128 sessions；
- 2 个 held-out packs，共 32 sessions；
- strict PCM gate 只抽取 train2 和 held-out2 代表样本。

这个规模足以快速发现：

- 全 WAIT；
- 通用首词；
- semantic collapse；
- EOS state mismatch；
- RTF backlog。

但即使 canary 通过，也仍需要 fixed15/full198 和完整 dev/test 证明稳定泛化。当前情况是 canary 本身尚未通过内容门，因此不应把扩大数据当成替代机制修复。

## 8. 为什么“版本很多”但不是无效重复

这些版本实际完成了逐层排错：

1. dense 数据修复：排除监督过稀疏；
2. runtime/frontend parity：排除训练输入与部署输入明显不一致；
3. Overfit1–4：证明一条完整 grammar/content trajectory 可学；
4. Overfit5–8：隔离 semantic 速度、content 和 length；
5. Overfit9：证明等价 runtime 可达到实时；
6. G10–G11：证伪 semantic-only 加数据即可泛化；
7. G12：验证 causal microblock 比独立 slot 更合理；
8. G13：证明 joint teacher-forced 训练仍无法覆盖真实 runtime state；
9. G14：证明 model-prefix exposure 与 deadline 可修低延迟 action；
10. G15：修正 action 标签并证明 teacher-forced EOS 与 runtime EOS 可以完全脱节。

真正的问题是：每个局部假设被逐步证明或证伪后，最终需要的完整 event-level on-policy 联合训练尚未执行。局部成功没有自动组合为端到端泛化。

## 9. 当前已经被实验严格证明的结论

### 9.1 已经证明可行

- Phase3 v4 能在一条 dense runtime trajectory 上适配 streaming grammar；
- 单样本完整文本和 semantic 可以学到；
- natural length 可以单独学到；
- 等价 runtime 融合可以把单样本 RTF 降到 1 以下；
- deadline + model-prefix exposure 可以得到 320–480ms natural WRITE；
- 真实 first PCM 可以做到约 599–901ms；
- teacher-forced EOS head 可以达到较高 precision/recall。

### 9.2 已经证明不充分

- 单样本 strict pass 不足以说明泛化；
- 只训练 semantic head 不会修复被冻结的 text path；
- fixed15 扩大 semantic 数据不会自动修复上述问题；
- 24 个独立并行 semantic slots 容易 collapse；
- 只学习 length 不保证 semantic 内容；
- teacher-forced action accuracy 不代表 free-running 会 WRITE；
- token 级随机 prefix corruption 不等价于完整 runtime rollout；
- 只校准 action/EOS、同时冻结 content，不会修复错误翻译；
- 首 PCM<1s 不代表整段 RTF<1；
- 更高 EOS 权重不等于 runtime 能进入正确 EOS state。

## 10. 当前最具体、最核心的问题到底在哪里

如果只选一个最核心位置，问题位于：

> 训练样本的“后续事件状态生成方式”。

当前 loss 大多在 oracle packed trajectory 的 token 位置上计算。模型被要求预测 action/text/semantic/EOS，但构成下一事件 hidden state 的历史仍主要由正确答案提供。真实部署则由模型自己的 action 和变长输出构成下一事件。

这会形成级联：

1. 早期 source evidence 不足；
2. deadline 促使第一次 WRITE；
3. 第一个文本 token 错误；
4. 错误 text/semantic 被提交到 persistent KV；
5. 下一 tick 的状态离开训练分布；
6. action、semantic length、continuation、EOS 同时失准；
7. 零 revision 不允许回退；
8. 后续训练过的 oracle EOS state再也到不了。

因此当前失败不是一个孤立 head 的分类错误，而是完整自生成事件状态没有被训练覆盖。

## 11. 下一版必须发生的根本变化

下一版如果仍只微调某个 loss 权重、增加 epoch、扩大 shard 或单独校准某个 head，大概率会重复当前模式。真正需要改变的是训练状态分布：

1. 使用 exact runtime grammar 生成完整 model-induced event rollout，而不是只替换固定 token；
2. rollout 必须真实包含 WAIT/WRITE、变长 text、变长 semantic、CONTINUE/END 和 START_GLM/EOS；
3. 从每个 model-induced persistent-KV state 查询 oracle continuation，训练模型恢复正确 action、内容和终止；
4. text、semantic、action 和 continuation/EOS 必须在这些状态上联合获得梯度，不能继续冻结错误 content path；
5. 同时保留 clean oracle trajectory 与 Phase3 replay，避免模型只学会从坏状态恢复却丢失原始离线质量；
6. 保留 semantic anti-collapse、自然长度和 speaker/codec 质量锚点；
7. rollout 概率必须受控，不能像 v14 后期一样让过强 corruption 淹没本来就较弱的内容学习；
8. 训练 gate 必须同时查看 strict free-running PCM，而不是只看 teacher-forced loss；
9. runtime 仍需批处理/融合每 tick 的文本、semantic 和 codec 计算，确保 RTF<1。

换言之，下一阶段的核心不是“继续训练 v15 更久”，而是首次真正训练：

> 在模型自己造成的完整事件历史上，如何恢复正确翻译、semantic、WRITE 和 EOS。

## 12. 最终结论

当前实验没有证明 UniSS Phase3 无法改成 simultaneous S2ST；相反，它已经证明多个必要能力可以分别实现。但它也清楚证明了：

- 单样本记忆不能代表泛化；
- teacher-forced 指标不能代表真实 persistent-KV runtime；
- semantic-only、action-only 或 EOS-only 修复都不足以完成端到端任务；
- 低延迟 action 与正确内容必须在真实 model-induced event history 中联合训练；
- 目前尚未真正执行这一步。

因此，迭代很多仍没有训练出合格模型的最准确解释是：

> 前半段版本在逐个排除数据、grammar、semantic 长度和 runtime 性能问题；后半段虽然发现了 exposure bias，却只实现了 token 级近似和局部 policy 校准，没有完成完整 event-level on-policy 联合训练。核心状态分布错位仍然存在，所以每次只能改善一个局部指标，无法让低延迟、翻译质量、semantic 质量、自然 EOS 和实时吞吐在 held-out 输入上同时成立。

## 13. 主要证据路径

- 总体 dense 数据与 runtime 方案：docs/uniss_training_reproduction/uniss_dense_aligned_continuous_streaming_retraining_solution.md
- Dense fixed15 pilot：experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/README.md
- Runtime-parity 主实验：experiments/uniss_phase3_runtime_parity_streaming_v2/
- Generalize12 严格报告：reports/uniss_phase3_runtime_parity_streaming_v2/generalize12_canary_strict_gate_report.md
- Generalize13 严格报告：reports/uniss_phase3_runtime_parity_streaming_v2/generalize13_joint_runtime_strict_gate_report.md
- Generalize14 严格报告：reports/uniss_phase3_runtime_parity_streaming_v2/generalize14_dagger_prefix_strict_gate_report.md
- Generalize15 训练证据：logs/uniss_phase3_runtime_parity_streaming_v2_generalize15_action_eos_calibration_canary_v1.log
- Generalize15 real-PCM 证据：reports/uniss_phase3_runtime_parity_streaming_v2/ 下名称包含 generalize15_action_eos_calibration_canary_v1_train2_strict_v15_gate_20260812a 和 held2_strict_v15_gate_20260812a 的目录
- 单样本严格成功：reports/uniss_phase3_runtime_parity_streaming_v2/overfit9_v1_fused_semantic_commit_v1/ 及两个 repeat 目录
- Overfit9 held-out 失败：reports/uniss_phase3_runtime_parity_streaming_v2/overfit9_v1_validation_sample0_v1/

## 14. 完成状态特别说明

为避免后续汇报把“计划”误写成“已完成”，当前状态如下：

| 版本 | 计划/名称 | 实际完成状态 |
|---|---|---|
| Overfit2 | 配置规划 500 iter | checkpoint 到 100，不能写完成 500 |
| Overfit5 | 配置规划 300 iter | checkpoint 到 100，无最终 strict PCM |
| Overfit6 | 150 iter | 完成训练，但 strict runtime 因不自然 END_SEMANTIC 失败 |
| Overfit7 | 配置规划 300 iter | checkpoint 到 50，只有 loader smoke |
| Overfit8 | 50 iter | 完成，有严格单轨迹 PCM |
| Overfit9 | v9 | 不训练，是 v8 的 runtime-only 优化 |
| Generalize11 full15 | full15 | fixed15 的 59,576 packs，不是 full198 |
| Generalize12 full15 | 名称/目录已建立 | checkpoint 目录为空，正式 full15 未训练 |
| Generalize13 canary_v1 | 第一版 | 双 CE graph OOM，正式有效结果是 canary_v2 |
| Generalize12–15 | canary | 主要为 5 train packs/128 sessions；不是 full198 |
