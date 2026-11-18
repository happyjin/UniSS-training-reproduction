# 单说话人 QABC 论文 v14 修改方案（基于 bayes_simul_s2st_v13_baselines）

写于 2026-11-18 09:20（布拉格时间）。只做了阅读和统计，没有改动 v13 或任何其他版本。

## 0. 先说结论

- **最新版本确认**：最新的单说话人论文是 `paper/bayes_simul_s2st_v13_baselines/`。
  依据是 `reports/s2st_baselines_v1/CHANGELOG_paper.zh-CN.md`：v13 建于 2026-11-13 00:04（布拉格时间），由 v12 `cp -a` 得到，之后没有更新的 bayes 目录。
  它的来源链是 v11_gpt → v12_baselines → v13_baselines。
- **审稿意见没有找到**。已经查过的位置：
  - 仓库全文；
  - `docs/reviews/`，里面只有 MIST v6 的审稿意见；
  - 各版本的 `docs/`，最新的单说话人审稿是 v7 和 v10 时期的，已经过时；
  - `/tmp` 下各会话的目录；
  - `~/.codex`；
  - 11-12 之后历史会话里的用户消息；
  - 全盘最近一天新增的 pdf、md、txt 和 docx。

  因此本方案目前覆盖两部分：你提出的 6 条要求，以及我通读 v13 时自己发现的问题（第 2 节）。审稿意见请贴给我或给出路径，我再补进第 10 节，逐条对应。
- **三个硬问题**：
  1. **正文超页**：Conclusion 从第 9 页左栏 61% 处开始，Limitations 也在第 9 页，正文约 8.5 页。ICML 规定正文超过 8 页自动拒稿。
  2. **正文图表太少**：v13 正文只有 3 个浮动体（Figure 1、Table 1、Table 2），三篇参考论文是 10–12 个；主结果曲线 Figure 2 放在附录第 17 页。
  3. **比例失衡**：Method 约 3.7 页，占正文约 45%；Setup 约 1.5 页；Results 只有约 1.2 页。
- **模板**：ICML 2027 的模板已经发布，应当换用（第 5 节）。

## 1. 三篇参考论文的版面数据

估算方法：按标题所在的页、栏和纵向位置换算成页数，误差约 ±0.1 页，每节的页数包含排在该节里的浮动体。

| 论文 | 引言 | 相关工作 | 方法 | 实验设置 | 实验结果 | 结论 | 正文浮动体（其中结果类） |
|---|---|---|---|---|---|---|---|
| Hibiki, High-Fidelity Simultaneous S2ST（ICML 2025） | 0.8 | 0.7 | 2.7（3 个子节） | 1.8（4.1–4.5） | 1.5 | 0.2 | 6 图 + 5 表 = 11（8） |
| SEE（ACL 2026） | 1.6（含 Fig 1、2） | 0.6 | 2.1（3 个子节） | 约 0.5 | 2.4 | 0.2 | 6 图 + 6 表 = 12（10） |
| SimulS2ST-Omni（ACL 2026） | 0.9 | 0.8 | 1.9（3 个子节） | 0.65 | 3.4 | 0.2 | 5 图 + 5 表 = 10（7，另有 1 张数据统计表） |
| **v13（本文）** | 0.9 | 0.4 | **3.7（3 个一级节，23 个定义/定理类环境）** | **1.5** | **1.2** | 0.4（**超出第 8 页**） | **1 图 + 2 表 = 3（2）** |

从这张表可以读出几点：

- **方法**：三篇都是一个一级节、3 个子节，篇幅 1.9–2.7 页，占正文 25–33%。
- **实验设置**：SEE 约 0.5 页，Omni 约 0.65 页，都是每段几句话，再指向附录。
  Hibiki 的设置写了 1.8 页，因为它是训练模型的论文，附录只有 1 页，所以架构超参和训练流程都放在正文里。
  我们的附录很长，应按两篇 ACL 论文的做法控制在 0.5 页左右。
- **SEE 的实验子节都以方法组件命名**，结果和方法一一对应：

  | SEE 实验子节 | 验证的方法部分 |
  |---|---|
  | 4.2 Quantifying Noise Intensity with SEE | §3.2 SEE |
  | 4.3 Effectiveness of SEE Across Noise Types | §3.1、§3.2 的泛化性 |
  | 4.4 SEE on Traditional Denoising | SEE 作为诊断工具 |
  | 4.5 Suppressing Noise with SEEN | §3.3 |
  | 4.6 Ablation Analysis | §3.1 的层选择、§3.3 的 λ 和 δ |

  每段的写法都是：要验证什么 → 看哪张图表 → 定性结论（最多给一个数）→ 这证实了方法的哪部分 → 更多结果见附录 X。
- **Omni 的附录 B "Training Implementation Details and Hyperparameters"** 的写法：
  - 先用一个总述段交代硬件、checkpoint 选择方式和初始化；
  - 再按 Stage 1、2、3 各写一段，说明训练什么、冻结什么、用什么数据、训练多少步；
  - 最后用一张表（Table 8）列出：Stage、可训练范围、每卡 token 数、GPU 小时、学习率计划和关键设置。

## 2. 通读 v13 发现的问题（这是我自己的阅读结果，不是审稿意见）

1. **正文超 8 页**，见第 0 节。
2. **摘要有 8 句**，ICML 示例论文要求 4–6 句。
3. **主证据不在正文**：正文只有 3 个浮动体；质量–延迟曲线和与外部系统的对比表都在附录。
4. **方法写得分散**：
   - 方法分成 §3 Problem Formulation、§4 Queue-Aware Bayesian Decision、§5 QABC 三个一级节；
   - 其中还有两个不属于方法的子节：§5.1 FR 是启发式规则，§5.2 是理论适用范围的讨论；
   - 你自己的 Table scope 也把 Prop 1、Lemma 1–2、Prop 4 标为 "standard identities"。
5. **方法里出现实验数字和超前引用**：Remark "myopic vs global" 里的 9–22%，target-shift remark 里的 0.22/0.16 和 1.49/0.73，以及 Table 18 等引用。
6. **内部标签外露**：
   - 正文和附录里有 E1/E2、C1/C2、H1/H2、prereg v4/v5/v6、"second/third registration"、s0/s1、T1c、seq-B、p5 等内部代号；
   - 附录节标题带 "(v11)"；
   - 附录按版本追加（`app:v10`、`app:v11`），没有按方法组织。
7. **结果正文数字太密**：一段里常有好几组 "+0.36/+1.10/+2.13 (Holm p=0.139, p=0.004, p<0.001)" 这样的数字串。
8. **训练设置有缺口**：
   - "belief-state variant" 全文只在 `method.tex:225` 出现一次，从未定义；
   - 优化器、warmup、数值精度、序列长度、Thinker 每读最大生成长度、Talker 采样参数和 GPU 小时都没有写（见第 7 节）。
9. **Table 14/15 是全部工作点的数据罗列**（第 6 节）。

## 3. 要求 1：Method 部分

### 3.1 逐项审计：哪些是你的创新

| 内容 | v13 位置 | 是否你的创新 | v14 处理 |
|---|---|---|---|
| Def 1 读窗协议 | §3 | 否，是 SimulS2ST-Omni 的协议 | 正文写 2 句并引用 |
| Ass 1 精确观测（Borel 路径空间、disintegration） | §3 | 否，测度论技术条件 | 移到附录 A；正文只写一句"已听到的音频无噪声地被观测到" |
| Ass 2 确定性起草、因果策略 | §3 | 否，建模假设 | 正文半句，完整表述放附录 A |
| Def 3 oracle continuation、stable length、survival | §3 | **部分**。用"同一 Thinker 的全上下文输出"打标签，与 Zheng 2019、wait-if-diff、TransLLaMa 相近（Related Work 自己写了 "modest variant"）。把 S_k 当作潜变量求后验，并按单元分解为存活概率，是你的建模 | 保留，作为 3.1 的核心 |
| Def 4 播放队列与感知延迟 D_k | §3 | **是** | 保留 |
| Ass 3 队列模型（平均语速 r） | §3 | 工程近似 | 正文保留一个公式；r 的数值移到附录超参表 |
| Def 5 myopic 单元损失 | §3 | **是** | 保留 |
| Prop 1 稳定长度后验与链式分解 | §4.1 | 推导是标准概率；它给出的结果是方法所需 | 正文保留后验式和链式分解两个公式；命题形式和证明放附录 A |
| Lemma 1 后验与策略无关 | §4.1 | 支撑性结论 | 正文一句，证明放附录 |
| Lemma 2 先验几率 × 似然比 | §4.1 | 否，标准恒等式（原文也写了 "interpretive, not computed online"） | 移到附录，与 TTB 放在一起 |
| **Thm 1 队列感知 Bayes 规则** | §4.2 | **是，核心** | 保留 |
| **Cor 1 队列等于对数几率上的决策偏移** | §4.2 | **是，核心洞见** | 保留 |
| Prop 3 LA 和 Hold-n 是 0/1 编码 | §4.2 | 统一视角，次要 | 正文一句；形式化命题放附录 |
| Remark myopic vs global | §4.2 | 局限说明 | 移到附录 B |
| Remark 块内延迟 D_k(u) | §4.2 | 扩展，测试结果不显著 | 移到附录 B，结果放附录 G |
| 自标注与 wait marker 目标 z_k | §5 | **是** | 保留 |
| Hazard 参数化（ρ = 1 − h） | §5 | **是**。把 LLM 输出 wait 的概率解释为离散时间 hazard 是你的做法；hazard 模型本身是生存分析的标准工具，需要引用 | 保留，并补一条生存分析的引用 |
| Prop 4 MLE 即摊销后验推断（KL 分解） | §5 | 否，是摊销推断中的标准结论 | 正文用一行公式说明"停止项学的是后验"；命题和证明放附录 |
| **Thm 2 训练损失界定任意阈值下的单元决策遗憾** | §5 | **是，主要的理论贡献**。证明用到 Pinsker 不等式和插值阈值遗憾这类经典工具，需要引用 | 保留 |
| Cor 协变量偏移（χ²） | §5 和附录 | 次要 | 移到附录 B；Limitations 写一句 |
| Cor target shift 及其 Remark（含数值） | §5 | 次要，而且结论是"无法给出保证"，放在方法里会削弱主线 | 移到附录 B；Limitations 写一句 |
| **Prop 5 贪心解码加偏置等于对后验取阈值（只会更保守）** | §5 | **是**，它使单次解码可行 | 保留 |
| §5.1 Fluent Realisation | §5.1 | 否，手工启发式；不使用后验，LA 也用 | 移到附录 D；Setup 写一句"所有策略共用同一后处理" |
| §5.2 理论范围与 A1–A6 近似 | §5.2 | 否，元讨论 | 移到附录 B；Limitations 写两句 |
| TTB | 已在附录 | 基线或消融 | 保持在附录 F，作为基线描述 |

两条处理原则：

- 正文只保留你自己的方法。属于你的内容不为省页而删减；省下的篇幅来自移出别人的方法、标准结论和工程细节。
- 正文里的每个命题，前面加一句动机，后面加一句它的用途；所有证明放到附录 A。

### 3.2 现在的子节安排有什么问题

1. **结构上**：三个一级节加两个非方法子节。§3 在给出任何直觉之前，先连续写了 5 个定义或假设块。读者要到第 6 页才知道系统在部署时是怎么运行的。
2. **逻辑上有断点**：
   - Lemma 2 刚说完这一项"不在线计算"，紧接着就引入 TTB 去分别估计两项。TTB 是基线，却出现在方法的核心推导里。
   - target-shift 推论的结论是"不能证明"，放在方法里打断了"后验 → 规则 → 学习 → 解码"这条主线。
3. **标题不体现动机**：子节标题是名词短语，开头没有说明为什么需要这一步。

### 3.3 v14 Method 的结构

一个一级节、4 个子节，加上 Figure 1 约 2.5 页。

**§3 Queue-Aware Amortized Bayesian Commitment**

节首写 3–4 句，配 Figure 1，把全节串成四个依次回答的问题：
(i) 提交的风险是什么？(ii) 什么时候该说？(iii) 后验从哪里来？(iv) 怎样在一次解码里执行？

- **3.1 Commitment Risk as a Stable-Length Posterior**（约 0.7 页）
  - 动机：提交后不能撤回，所以风险就是草稿中将来会被完整上下文改掉的那一部分。
  - 内容：
    - 读窗协议，引用 Omni，写 2 句；
    - 观测 o_k 和草稿 d_k；
    - oracle continuation、稳定长度 S_k、存活概率 ρ_k(u)；
    - 后验式和链式分解式；
    - 一句说明后验与策略无关。
  - 末句：§4.3 用 oracle 实验检验"按 S_k 提交时质量接近离线翻译"。
- **3.2 Queue-Aware Bayes Rule**（约 0.6 页）
  - 动机：后验只回答草稿有多稳定，不回答等待的代价有多大；而在 S2ST 中，等待的时间会被仍在播放的语音掩盖。
  - 内容：
    - 队列 q_k、感知延迟 D_k、队列更新式；
    - myopic 损失；
    - Thm 1；
    - Cor 1：队列只移动阈值，不改变后验；
    - 一句说明 LA 和 Hold-n 是无法利用队列的 0/1 编码。
  - 末句：§4.4 检验队列项的作用。
- **3.3 Amortized Posterior Learning**（约 0.7 页）
  - 动机：后验需要对还没听到的音频积分，在线无法计算。
  - 内容：
    - 用同一 Thinker 的全上下文输出做自标注；
    - 目标 z_k；
    - wait marker 的 hazard 参数化，ρ = 1 − h；
    - 损失分解为停止项加内容项；
    - 一行 KL 等式，说明停止项学到的是后验；
    - Thm 2：只需训练一次，在任意队列阈值下遗憾都有界。
  - 末句：§4.5 检验学到的后验。
- **3.4 Single-Pass Queue-Aware Decoding**（约 0.4 页）
  - 动机：部署时不可能对每个单元逐一枚举阈值。
  - 内容：
    - 用 b_k = logit τ(q_k) 偏置 wait 的 logit；
    - Prop 5：这种实现只会比 Bayes 规则更保守；
    - 每读三步：解码 → 提交块 → 更新队列。
    - τ_max 和 b_0 的取值放到附录 C。
  - 末句：§4.6 检验单次解码是否足够。

写作规则，沿用你给 MIST v15–v18 定的规则中可以通用的部分：

- 每个符号在引入它的那句话里定义，不用符号表；数值一律放到附录 C。
- 标量用普通斜体，向量用粗体小写，集合用 `\mathcal`；同一个字母不重复使用。
- 每个子节以动机句开头，但不写 "Motivation:" 这样的标签。
- 句与句、段与段之间要有承接。

## 4. 要求 2：Experimental Setup 控制在约 0.5 页

写四段，每段约 3 句，其余细节全部指向附录：

1. **Data and benchmarks**：
   - 训练数据：40k 句，写明语料名称，展开为 235k 次读；
   - 开发集：805 句；
   - 测试集：RealSI 两个方向（句级，语音输出），ACL60/60（长文，文本输出）。
   - 细节见附录 E。
2. **Models and training**：
   - Thinker 是冻结的 Qwen3-Omni-30B-A3B，加上 QABC 的 LoRA；
   - Talker 是 simultaneous UniSS，所有策略共用，并保持冻结；
   - 分阶段训练见附录 C。
3. **Baselines**：
   - 同一 Thinker 上的 LA2、Hold-n、Vote；
   - 用我们的协议运行的外部系统：SeamlessStreaming、X-Translator、Hy-SimulMT 接我们的 Talker。
   - 细节见附录 F。
4. **Metrics and statistics**：
   - 质量用 ASR-BLEU；延迟以 segment speech LAAL 为主，长文用 StreamLAAL；
   - 评测不计计算时间，与 SimulEval 的默认设置一致；
   - 统计用配对 bootstrap 加 Holm 校正；主比较的设置在运行测试集之前就已固定，见附录 E。

**移出 Setup 的内容**：

- CVSS 离线表（v13 的 Table 1）移到附录 G，正文用一句话交代离线水平；
- Policies 段的细节、FR、ACL 上的模拟队列、word-aligned LAAL 的讨论，移到附录 E/F；
- 注册细节（E1/E2/C1/C2）和 "Supplementary: latency tiers…" 一句，移到附录 E。

## 5. 要求 3：模板换成 ICML 2027

- ICML 2027 的 Author Instructions 和 `icml2027.zip` 已经发布，其中 `icml2027.sty` 的日期是 2026-10-30。
- 与 v13 用的 `icml2026.sty` 对比：sty 文件只改了版本行，`icml2027.bst` 与 `icml2026.bst` 完全相同。所以换模板几乎没有风险。
- 需要做的改动：
  1. 改成 `\usepackage{icml2027}` 和 `\bibliographystyle{icml2027}`，并替换 sty 和 bst 文件；
  2. **新增必填的 "AI Use Statement"**：无编号，放在参考文献之前，不计入页数；
  3. Impact Statement 保留：它也是必填项，不计入页数；
  4. **正文 ≤ 8 页**：超过即自动拒稿；相机版可以多 1 页；
  5. **摘要压到 4–6 句**。
- 相机版还有一项新要求：在引言末尾加 "Conflict of Interest Disclosure" 段。如果有作者受雇于被评测模型的开发公司，例如 Hy-SimulMT 的开发方，就需要披露。投稿版不需要这一段，但请作者自查。

## 6. 要求 4：Table 14

- **它是什么**：附录 B 第 21 页的 `generated/tab_appendix_full.tex`，列出全部 RealSI 工作点。
  共 34 行，每行最多 15 个点，每个点写成 "均值±sd/LAAL"。全文只有 `appendix.tex:327` 引用过它一次。
- **判断**：对读者没有用。同样的信息已经由质量–延迟曲线（v14 移入正文）和分档对比表给出；它唯一的用处是复现曲线。
- **建议删除**：
  - 数据导出成 CSV，作为投稿的 supplementary code/data 提交；相机版放到代码仓库；
  - 在曲线图的图注里写 "exact values in the supplementary CSV"；
  - Table 15（ACL60/60 的全部工作点）按同样方式处理；
  - `tab_range_tests_full` 和 `tab_sdr_full` 如果与正文的表重复，也一并删除；
  - 同步修改 `appendix.tex:327` 那句引用。

## 7. 要求 5：参数和超参数

### 7.1 现状审计

| 项目 | 定义在哪里 | 数值在哪里 | 缺什么 |
|---|---|---|---|
| Δ、b_0、τ_0、C_err、C_lat | 正文 | Setup 有 Δ 和 b_0 的网格 | 无 |
| τ_max = 0.9975 | 正文 | 写在 method 正文里 | 移到附录 C |
| r（每字 0.19 s，每词 0.34 s） | Ass 3 | 写在 method 正文里 | 移到附录 C |
| n_min（3 个字或 4 个词） | 附录中的 FR | 附录 | 无 |
| LoRA：r=32、α=64、第 32–47 层 | 附录 | Setup 写的是 "upper third"，两处说法不一致 | 统一 |
| GBS 256、LR 1e-4 cosine、8×H200、每 200 步存一次、选第 200 步（共 800 步）、约 25 分钟 | 分散在 3 处 | — | 集中到一处 |
| **adapter 训练** | — | — | 优化器及 β、weight decay、warmup、min LR、梯度裁剪、数值精度、序列长度、micro-batch。<br>epoch 数：235k / 256 ≈ 918 步为一个 epoch，800 步约 0.87 个 epoch，需核实。<br>**第 200 步是按什么标准选出的**。<br>wait marker 是新增 token 还是已有的保留 token，如何初始化。<br>内容项和停止项是否等权。<br>"1/6 为整句样本"是怎样混合进去的。<br>**belief-state roll-in 是什么（未定义）**，TTB roll-in 用的阈值。<br>训练语料的名称和两个方向的比例。 |
| **Talker（T1c）** | 附录 | GBS 256、LR 1e-5、第 1500 步 | warmup 和学习率计划、数据规模（小时）、序列长度、GPU 小时、块是怎样构造的、推理采样参数（温度、top-p，以及"三个 talker seed"具体指什么） |
| **推理** | — | — | Thinker 每读的最大生成 token 数、prompt 格式、KV cache 是否复用 |
| TTB prior head（MLP 17–64–64–1） | 附录 | — | 训练设置 |

### 7.2 新增附录 C：Training Implementation Details and Hyperparameters

按 Omni 附录 B 的写法组织：

- **总述段**：
  - 硬件和训练框架；
  - 哪些部分冻结、哪些部分训练；
  - checkpoint 只按 dev 选择，写清选择标准；
  - 随机种子。
- **C.1 Stage 1：Simultaneous Talker training**
  - 从 UniSS 出发，训练为按块续写；
  - 数据、步数、学习率计划；
  - 得到 T1c，即第 1500 步的 checkpoint。
- **C.2 Stage 2：Oracle label generation**
  - 在冻结的 Thinker 上运行 roll-in 策略：给出 TTB 和 belief-state 两种策略的定义与参数；
  - 读窗取 0.5、1、2、4 s；
  - 在同一已提交前缀下做全上下文续写，得到 S_k，构造目标 z_k；
  - 1/6 的样本为整句；
  - 统计信息：40k 句 → 235k 次读，两个方向的比例，S_k 的分布。
- **C.3 Stage 3：QABC adapter training**
  - LoRA 的插入位置、秩和 α；
  - 完整的优化器设置和学习率计划；
  - batch 和序列长度；
  - 总步数和保存间隔；
  - dev 选择标准；
  - 5 个 seed；
  - 训练耗时。
- **C.4 TTB prior head**：作为基线使用，写清训练设置。
- **表 C1**：列为 Stage、Init、Trainable scope、Data、Steps (epochs)、Batch、LR schedule、GPU-hours、Selection，每个 Stage 一行。
- **表 C2，推理超参**：Δ 网格、b_0 网格、τ_max、r、FR 的 n_min、Thinker 的解码设置、Talker 的采样设置、速度 token、播放缓冲。
- **Algorithm 1**（可选）：用伪代码概括训练和推理，放在附录 C 末尾。
- **数值来源**：所有数值都从启动脚本和训练日志中读取，不凭记忆填写。
  - 需要在 `experiments/uniss_qwen3_omni_v1/` 的 `scripts/` 和 `training/` 下，定位 QABC adapter 和 T1c 的 Megatron 启动脚本和日志；
  - 填完后用检查脚本逐项核对。

## 8. 要求 6：正文图表与结果章节

### 8.1 数量目标

- 参考论文正文有 10–12 个浮动体，其中结果类 7–10 个；v13 只有 3 个（结果类 2 个）。
- v14 目标：正文 9–10 个浮动体，即 1 张方法图加 8–9 个结果图表。
- 全部来自已有结果，**不需要新实验**。只有 Fig 3 是新画的图，数据取自现有的 `tab:oracle`。

### 8.2 新的实验章

只设一章 §4 Experiments，顺序参照 SEE：Setup → Main Results → 与方法逐项对应的分析 → Ablation。

| 子节 | 回答的问题 | 对应的方法 | 正文图表（来源） | 放到附录的内容 |
|---|---|---|---|---|
| 4.1 Experimental Setup（0.5 页） | — | — | 无 | — |
| 4.2 Main Results（0.8 页） | 在相同延迟下，QABC 是否更好？ | 整体 | **Fig 2**：质量–延迟曲线，RealSI 两个方向加 ACL60/60（`figures/curves.tex`，从附录移入）<br>**Table 1**：按延迟档与各系统对比（`tab_realsi_main`）<br>**Table 2**：事先固定的主比较（精简自 `tab_prereg_all`：每个测试集给 s0 和五个 seed 均值两行，列出 E1 和 p） | 全部工作点的 CSV、分档表、按 talk 重采样、word-aligned LAAL、文本端结果、CVSS 离线结果 |
| 4.3 Is the stable length the right target?（0.35 页） | 按 S_k 提交，质量能否接近离线？ | 3.1 | **Fig 3**（新图）：oracle 提交、LA2、QABC 与整句翻译的对比柱状图，数据来自 `tab:oracle` | 早期单元的稳定性（irrev、critical） |
| 4.4 Does the queue-aware rule help?（0.5 页） | 在同一后验上加入队列是否更好？在同一份草稿上，规则本身是否胜过 LA？ | 3.2（Thm 1、Cor 1） | **Table 3**：QABC_Δ 与 QABC 的配对差（压缩 `tab_queue`）加上同一草稿上规则与 LA 的对比（取 `tab_controls` 2 行）。<br>也可以改画成 **Fig 4**：增益随 Δ 变化。Δ 越小，Talker 越常在播放，增益越大，这正是 Cor 1 的预测 | 与二值门的对比（如实写：没有显示出优于二值门，两者只在 3.8% 的读上做出不同决定）、语速 r 的敏感性、延迟的实测 |
| 4.5 Does amortized learning give a usable posterior?（0.5 页） | 学到的后验是否校准？是否优于测试时后验？ | 3.3（Thm 2） | **Table 4**：摊销后验、测试时后验（TTB）、LA 三者对比（精简自 `tab_range`），加 ECE 和 NLL 两列（来自 `tab_v10_calib`） | refine、Platt 重校准、target shift 的数值 |
| 4.6 Is single-pass decoding enough?（0.35 页） | 贪心解码加偏置会不会损失质量？计算代价多大？ | 3.4（Prop 5） | **Table 5**：同一后验上的 4 种决策规则（`tab_v10_rules`），加每读的计算量（`tab_v2_cost`；Vote 需要 8 次采样） | 各规则做出不同决定的频率 |
| 4.7 Ablation Analysis（0.3 页，按段落写） | 各项设计选择是否必要？ | 各部分 | **Table 6**：5 个 adapter seed（`tab_v10_seeds_main`）；也可以把这部分并入 Table 2 | FR 开关、其他 seed 和 talk 的分析 |

另外两项负面结果：一条时钟连续解码的 backlog 问题，以及 Zh→En 的差距。这两项在 Limitations 里各写一两句，完整分析放到附录 H。

### 8.3 写法模板（正文和附录都用）

每段按这个顺序写：
1. 要验证哪个组件；
2. 看哪张图表；
3. 定性结论，最多给一个关键数字；
4. 这对应哪个定理、推论或公式的预测；
5. 细节见附录哪一节。

以 v13 的 "Queue-aware threshold" 段为例。

v13 原文：

> Adding the queue to the same posterior helps in all ten matched RealSI pairs, by up to +2.96/+2.33 ASR-BLEU where the Talker is most often still speaking (four significant after Holm; Table~\ref{tab:queue}), and improves all three adapters (Table~\ref{tab:v2-seeds}) at no cost; without it the registered RealSI gains drop to +0.59/+0.07 (not significant).

v14 改写：

> Corollary 1 predicts that the queue should matter only while earlier speech is still playing. Table 3 compares QABC with its constant-delay variant on the same posterior: the queue-aware threshold is ahead in all ten matched settings, and the gain is largest at the shortest read window, where the Talker is most often still speaking (up to +3.0 ASR-BLEU). Without the queue term, the main RealSI gains are no longer significant. The queue thus changes the decision without changing the posterior, as Corollary 1 states.

其他规则：

- **图注**写 1–2 句：比较了什么、怎么读（加粗表示什么、箭头方向），显著性用一句话说明。协议细节放附录。图注里不列 p 值。
- **正文禁用**：
  - 内部标签：E1/E2、C1/C2、H1/H2、v4–v11、s0、T1c、seq-B、p5、"second/third registration"；
  - 一组组的数字三元组；
  - "comparable"。
- **用词**：说我们的任务和模型时用 "simultaneous"；延迟写成 computation-excluded。

### 8.4 附录按同样的逻辑重排

| 附录 | 内容 |
|---|---|
| A | Proofs：按正文中命题出现的顺序排列 |
| B | Scope of the theory：myopic 与全局、块内延迟、两个偏移推论、A1–A6 近似、Table scope |
| C | Training Implementation Details and Hyperparameters（第 7 节） |
| D | Inference and Fluent Realisation：FR、播放器参数 |
| E | Evaluation protocols：ACL60/60 的偏移恢复、StreamLAAL、统计方法、事先固定的主比较、Table protocol |
| F | Baselines：LA2、Hold-n、Vote、TTB、外部系统、Talker 消融 |
| G | Supplementary results，按方法组件分组：<br>G.1 主结果补充（分档、按 talk 重采样、word-aligned LAAL、文本端、CVSS 离线结果）<br>G.2 稳定长度<br>G.3 队列规则<br>G.4 后验（校准、refine、seed）<br>G.5 解码规则 |
| H | When QABC does not help：一条时钟连续解码、Zh→En 的差距、没有带来提升的扩展 |

排版要求：

- 删除附录标题里的 "(v10)"、"(v11)"。
- 附录中的结果段也按 8.3 的模板写。
- 附录开头加 `\clearpage`；Supplementary Results 放在文字较多的节之前；通过 float 参数控制位置，每栏空白不超过 20%。

## 9. v14 页数预算

| 部分 | v13 | v14 目标 | 参考论文的范围 |
|---|---|---|---|
| 标题和摘要 | 约 0.6 | 0.4（摘要 4–6 句） | — |
| 引言 | 0.9 | 0.85 | 0.8–1.6 |
| 相关工作 | 0.4 | 0.4（补 hazard 和生存分析的引用） | 0.6–0.8 |
| 方法 | 3.7 | 2.5 | 1.9–2.7 |
| 实验设置 | 1.5 | 0.5 | 0.5–0.65（Hibiki 1.8） |
| 实验结果 | 1.2 | 3.0 | 1.5–3.4 |
| 结论和局限 | 0.4 | 0.35 | 约 0.2 |
| **合计** | **约 8.5（超页）** | **8.0** | — |

## 10. 审稿意见（待补）

审稿意见到手后，按下面的格式逐条补入：

| 意见 | 是否同意 | 修改位置 | 是否需要实验 |
|---|---|---|---|

目前已有的单说话人审稿意见都是旧版本的，只能作参考：`paper/bayes_simul_s2st_v11/docs/review_of_v7.zh-CN.md` 和 `response_to_review_v10.zh-CN.md`。

## 11. 需要你决定的事

1. **CVSS 离线表**：放附录（推荐），还是留在正文？
2. **"preregistered" 的说法**：
   - 推荐：只保留一句 "the primary comparison and its statistics were fixed before the test sets were run"；删去 v4/v5/v6、"second/third registration"、E1/E2/C1/C2/H1/H2 等字样；摘要里的 "Preregistered" 改为同样的说法。
   - 另一个选项是完全不提。
3. **Table 14/15**：删除，改为提供 CSV（推荐）。
4. **一条时钟连续解码的负面结果**：在 Limitations 写两句、完整分析放附录（推荐），还是在正文单列一小节？
5. **Fig 3**：是否新画 oracle 柱状图（推荐画）？
6. **Algorithm 框**：放附录 C（推荐），还是放正文？

## 12. 执行步骤

1. 用 `cp -a` 把 v13 复制为 `paper/bayes_simul_s2st_v14/`。v13 及更早的版本都不改动。
2. 换成 icml2027 模板，用 tectonic 编译，先确认基线能正常编译。
3. 按第 3、4 节重写 Method 和 Setup；按第 8 节重写结果章；把移出的内容放进附录。
4. 按 8.4 重排附录；删除 Table 14/15，并导出 CSV。
5. 写新附录 C：从启动脚本和日志中读取数值，用检查脚本逐项核对。
6. 用现有的 `tab:oracle` 数据画 Fig 3。
7. 检查以下各项：
   - 正文（截至 Conclusion 和 Limitations）≤ 8 页；
   - 没有未定义的引用，没有 Overfull；
   - grep 内部标签，结果为 0；
   - 每个数字都与结果 JSON 一致，沿用 `check_paper_v13.py` 的做法。
8. 修改记录写入 `reports/s2st_baselines_v1/CHANGELOG_paper.zh-CN.md`。

整个过程不需要 GPU，也不需要新实验。

## 来源

- ICML 2027 Author Instructions：https://icml.cc/Conferences/2027/AuthorInstructions
- ICML 2027 样式包：https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip
- Hibiki：arXiv 2502.03382（ICML 2025）
- SEE：ACL 2026 Long，第 18976–18995 页
- SimulS2ST-Omni：arXiv 2607.19810v2
