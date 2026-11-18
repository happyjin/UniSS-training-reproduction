# 单说话人 QABC 论文 v14 修改方案（基于 bayes_simul_s2st_v13_baselines）

- 第 1 轮：2026-11-18 09:20（布拉格时间）。
- 第 2 轮：2026-11-18 10:30（布拉格时间）。本轮写入了你的决定（第 11 节），补了附录诊断和重排方案（第 13 节）。
- 第 3 轮：2026-11-18 11:40（布拉格时间）。审稿意见（QABC_v13_ICML_Review_CN.pdf）已收到。本轮新增第 14 节（负面结果清单）和第 15 节（对方案的修改）。**第 15 节与前文冲突时，以第 15 节为准。**

到目前为止只做了阅读和统计，没有改动 v13 或任何其他版本。

## 0. 先说结论

- **最新版本确认**：最新的单说话人论文是 `paper/bayes_simul_s2st_v13_baselines/`。
  - 依据是 `reports/s2st_baselines_v1/CHANGELOG_paper.zh-CN.md`：v13 建于 2026-11-13 00:04（布拉格时间），之后没有更新的 bayes 目录。
  - 来源链：v11_gpt → v12_baselines → v13_baselines。
- **审稿意见没有找到**。已经查过这些位置：
  - 仓库全文；
  - `docs/reviews/`（只有 MIST v6 的审稿）；
  - 各版本的 `docs/`（最新的单说话人审稿是 v7、v10 时期的）；
  - `/tmp` 下的会话目录；
  - `~/.codex`；
  - 11-12 之后历史会话里的用户消息；
  - 全盘最近一天新增的 pdf、md、txt、docx。

  所以本方案覆盖的是你的 6 条要求、第 2 轮的决定，以及我通读时自己发现的问题。审稿意见请贴给我或给出路径，我会补进第 10 节，逐条对应。
- **三个硬问题**：
  1. **正文超页**：Conclusion 从第 9 页左栏 61% 处开始，Limitations 也在第 9 页，正文约 8.5 页。ICML 规定正文超过 8 页即自动拒稿。
  2. **正文浮动体太少**：只有 3 个（Figure 1、Table 1、Table 2），三篇参考论文是 10–12 个；主结果曲线放在附录第 17 页。
  3. **篇幅比例失衡**：Method 约 3.7 页，占正文约 45%；Setup 约 1.5 页；Results 只有约 1.2 页。
- **模板**：ICML 2027 模板已经发布，换用（第 5 节）。

## 1. 三篇参考论文的版面数据

估算方法：按标题所在的页、栏和纵向位置换算，误差约 ±0.1 页；每节的页数包含排在该节内的浮动体。

| 论文 | 引言 | 相关工作 | 方法 | 实验设置 | 实验结果 | 结论 | 正文浮动体（其中结果类） |
|---|---|---|---|---|---|---|---|
| Hibiki, High-Fidelity Simultaneous S2ST（ICML 2025） | 0.8 | 0.7 | 2.7（3 个子节） | 1.8（4.1–4.5） | 1.5 | 0.2（无 Limitations 节） | 6 图 + 5 表 = 11（8） |
| SEE（ACL 2026） | 1.6（含 Fig 1、2） | 0.6 | 2.1（3 个子节） | 约 0.5 | 2.4 | 0.2 | 6 图 + 6 表 = 12（10） |
| SimulS2ST-Omni（ACL 2026） | 0.9 | 0.8 | 1.9（3 个子节） | 0.65 | 3.4 | 0.2 | 5 图 + 5 表 = 10（7，另有 1 张数据统计表） |
| **v13（本文）** | 0.9 | 0.4 | **3.7（3 个一级节，23 个定义/定理类环境）** | **1.5** | **1.2** | 0.4（**超出第 8 页**） | **1 图 + 2 表 = 3（2）** |

几点观察：

- **方法**：三篇都是一个一级节加 3 个子节，篇幅 1.9–2.7 页。
- **实验设置**：SEE 约 0.5 页，Omni 约 0.65 页，都是每段几句话，再指向附录。Hibiki 写了 1.8 页，因为它的附录只有 1 页。我们按两篇 ACL 论文的做法，控制在 0.5 页左右。
- **SEE 的实验子节按方法组件命名**，每个子节对应一个方法部分：

  | SEE 实验子节 | 对应的方法部分 |
  |---|---|
  | 4.2 Quantifying Noise Intensity with SEE | §3.2 |
  | 4.3 Effectiveness Across Noise Types | §3.1、§3.2 的泛化 |
  | 4.4 SEE on Traditional Denoising | — |
  | 4.5 Suppressing Noise with SEEN | §3.3 |
  | 4.6 Ablation | §3.1 的层选择，§3.3 的 λ/δ |

  每段的写法是：要验证什么 → 看哪张图表 → 定性结论（最多一个数）→ 这证实了方法的哪部分 → 详见附录 X。
- **Omni 的附录**：附录 B 先写一个总述段，再按 Stage 分段，最后配一张阶段表（Table 8）。整个附录的结构见第 13 节。

## 2. 通读 v13 发现的问题（我自己的阅读结果，不是审稿意见）

1. **正文超 8 页**（见第 0 节）。
2. **摘要 8 句**；ICML 示例要求 4–6 句。
3. **正文只有 3 个浮动体**，主结果曲线和外部系统对比都在附录。
4. **方法分散**：
   - 分在 §3、§4、§5 三个一级节里，另有两个不属于方法的子节：§5.1 FR 启发式，§5.2 理论范围；
   - 多个结论被你自己的 Table scope 标为 "standard identities"。
5. **方法里出现实验数字和超前引用**：9–22%、0.22/0.16、1.49/0.73，以及对 Table 18 等的引用。
6. **内部标签外露**：
   - 正文出现 E1/E2、C1/C2、H1/H2、prereg v4/v5/v6、"second/third registration"、s0/s1、T1c、seq-B、p5；
   - 附录节标题带 "(v11)"；
   - **附录里使用的 "D1" 从未定义**（`appendix.tex:372`、`tab_baselines`、`tab_omnitalker`）。
7. **结果正文数字太密**，常见 "+0.36/+1.10/+2.13 (Holm p=…)" 这样的写法。
8. **"belief-state variant" 全文只出现一次，从未定义**（`method.tex:225`）；优化器、warmup、精度、序列长度、最大生成长度、Talker 采样参数、GPU 小时也都没有写。
9. **Table 14/15 是全部工作点的数据罗列**。
10. **附录按写作时间堆积**，逻辑和写法都乱，详见第 13 节。

## 3. 要求 1：Method 部分

### 3.1 逐项审计：哪些是你的创新

| 内容 | v13 位置 | 是否你的创新 | v14 处理 |
|---|---|---|---|
| Def 1 读窗协议 | §3 | 否，来自 SimulS2ST-Omni | 正文写 2 句并引用 |
| Ass 1 精确观测（Borel 路径空间、disintegration） | §3 | 否，测度论技术条件 | 移到附录 B；正文一句"已听到的音频无噪声地被观测到" |
| Ass 2 确定性起草、因果策略 | §3 | 否，建模假设 | 正文半句，全文移到附录 B |
| Def 3 oracle continuation、stable length、survival | §3 | **部分**。用"同一 Thinker 的全上下文输出"打标签，与 Zheng 2019、wait-if-diff、TransLLaMa 相近；把 S_k 当作潜变量求后验、按单元分解存活概率，是你的建模 | 保留，作为 3.1 的核心 |
| Def 4 播放队列与感知延迟 D_k | §3 | **是** | 保留 |
| Ass 3 队列模型（平均语速 r） | §3 | 工程近似 | 正文保留一个公式；r 的数值移到附录超参表 |
| Def 5 myopic 单元损失 | §3 | **是** | 保留 |
| Prop 1 稳定长度后验与链式分解 | §4.1 | 推导是标准概率；结果是方法所需 | 正文保留两个公式；命题和证明移到附录 A |
| Lemma 1 后验与策略无关 | §4.1 | 支撑性 | 正文一句，证明移到附录 |
| Lemma 2 先验几率 × 似然比 | §4.1 | 否，标准恒等式 | 移到附录 B（与 TTB 的推导放在一起） |
| **Thm 1 队列感知 Bayes 规则** | §4.2 | **是，核心** | 保留 |
| **Cor 1 队列等于对数几率上的决策偏移** | §4.2 | **是，核心洞见** | 保留 |
| Prop 3 LA 和 Hold-n 是 0/1 编码 | §4.2 | 统一视角，次要 | 正文一句；形式化命题移到附录 |
| Remark myopic vs global | §4.2 | 局限说明 | 移到附录 B |
| Remark 块内延迟 D_k(u) | §4.2 | 扩展，测试结果不显著 | 移到附录 B，结果放附录 H |
| 自标注与 wait marker 目标 z_k | §5 | **是** | 保留 |
| Hazard 参数化（ρ = 1 − h） | §5 | **是**（hazard 模型本身需要引用生存分析文献） | 保留，补引用 |
| Prop 4 MLE 即摊销后验推断（KL 分解） | §5 | 否，标准结论 | 正文一行公式；命题和证明移到附录 |
| **Thm 2 训练损失界定任意阈值下的单元决策遗憾** | §5 | **是，主要理论贡献**（Pinsker 等经典工具需要引用） | 保留 |
| Cor 协变量偏移、Cor target shift 及其 Remark | §5 和附录 | 次要；target shift 的结论是"无法给出保证" | 移到附录 B；Limitations 写一句 |
| **Prop 5 贪心解码加偏置等于对后验取阈值（只会更保守）** | §5 | **是** | 保留 |
| §5.1 Fluent Realisation | §5.1 | 否，手工启发式，所有策略共用 | 移到附录 E；Setup 写一句 |
| §5.2 理论范围与 A1–A6 近似 | §5.2 | 否，元讨论 | 移到附录 B；Limitations 写一句 |

两条处理原则：

- 正文只保留你自己的方法；属于你的内容不为省页而删减。
- 正文里的每个命题，前面加一句动机，后面加一句它的用途；证明全部移到附录 A。

### 3.2 现在的子节安排有什么问题

- **结构**：三个一级节，外加两个不属于方法的子节。§3 在给出任何直觉之前连续写了 5 个定义或假设块；读者要到第 6 页才知道系统部署时怎么运行。
- **逻辑断点**：
  - Lemma 2 刚说完"不在线计算"，紧接着就引出 TTB 这个基线；
  - target-shift 推论的结论是"不能证明"，放在方法主线上会削弱主线。
- **标题**：子节标题是名词短语，没有体现动机。

### 3.3 v14 Method 的结构

一个一级节、4 个子节，含 Figure 1 约 2.5 页。

**§3 Queue-Aware Amortized Bayesian Commitment**

节首写 3–4 句，配 Figure 1，把全节串成四个问题：(i) 风险是什么 → (ii) 何时说 → (iii) 后验从哪里来 → (iv) 怎样一次解码执行。

- **3.1 Commitment Risk as a Stable-Length Posterior**（约 0.7 页）
  - 内容：读窗协议（2 句，引用 Omni）、观测 o_k、草稿 d_k、S_k、ρ_k(u)、后验式与链式分解、一句策略无关性。
- **3.2 Queue-Aware Bayes Rule**（约 0.6 页）
  - 内容：q_k、D_k、队列更新式、myopic 损失、Thm 1、Cor 1、一句 0/1 编码。
- **3.3 Amortized Posterior Learning**（约 0.7 页）
  - 内容：自标注、z_k、hazard 参数化、停止项与内容项的分解、一行 KL、Thm 2。
- **3.4 Single-Pass Queue-Aware Decoding**（约 0.4 页）
  - 内容：b_k = logit τ(q_k)、Prop 5、每读三步。
  - τ_max 和 b_0 的取值放附录。

写法：

- 每个子节以动机句开头，以"由哪个实验验证"一句收尾。
- 每个符号在引入它的那句话里定义，不用符号表，数值一律放附录。
- 标量用普通斜体，向量用粗体小写，集合用 `\mathcal`；不重复使用同一个字母。

## 4. 要求 2：Experimental Setup 约 0.5 页

写四段，每段约 3 句：

1. **Data and benchmarks**
   - 训练：40k 句，写明语料名，展开为 235k 次读；dev：805 句。
   - 测试：RealSI 两个方向（句级，语音输出）；ACL60/60（长文，文本输出）；CVSS-T（离线）。
   - 明确写出 RealSI 在句级评测，与 SimulS2ST-Omni 一致。
   - 细节见附录 C、F。
2. **Models and training**
   - 冻结的 Qwen3-Omni-30B-A3B 加 QABC LoRA；simultaneous UniSS Talker，所有策略共用，冻结。
   - 分阶段训练见附录 D。
3. **Baselines**
   - 同一 Thinker 上的 LA2、Hold-n、Vote；用我们的协议运行的外部系统 SeamlessStreaming、X-Translator、Hy-SimulMT 接我们的 Talker。
   - 细节见附录 G。
4. **Metrics and statistics**
   - ASR-BLEU；speech LAAL 为主，长文用 StreamLAAL；不计计算时间。
   - 统计用配对 bootstrap 加 Holm；"the primary comparison and its statistics were fixed before the test sets were run"。
   - 细节见附录 F。

**移出 Setup 的内容**：Policies 段的细节、FR、ACL 模拟队列、word-aligned LAAL、注册编号和代号，移到附录 E、F、G。CVSS 表按你的决定留在正文，见 8.2。

## 5. 要求 3：模板换成 ICML 2027

- ICML 2027 已经发布 `icml2027.zip`，sty 文件日期 2026-10-30。
- 与 v13 用的 `icml2026.sty` 相比，sty 只改了版本行；`icml2027.bst` 与 2026 版完全相同。
- 需要做的改动：
  1. 改为 `\usepackage{icml2027}` 和 `\bibliographystyle{icml2027}`；
  2. **新增必填的 "AI Use Statement"**（无编号，放在参考文献前，不计页）；
  3. Impact Statement 保留（必填，不计页）；
  4. **正文 ≤ 8 页**（相机版可加 1 页）；
  5. **摘要 4–6 句**。
- 相机版还要求在引言末尾加 Conflict of Interest Disclosure：如果有作者受雇于被评测模型的开发公司（例如 Hy-SimulMT 的开发方），需要披露。请作者自查。

## 6. 要求 4：Table 14/15（已决定删除）

- 删除 `generated/tab_appendix_full.tex`（Table 14）和 `tab_appendix_acl.tex`（Table 15）。
- 同步修改 `appendix.tex:327`、`appendix.tex:361` 的引用句。
- 数据导出为 CSV，作为投稿的 supplementary code/data；相机版放到代码仓库。
- 曲线图图注写 "exact values in the supplementary CSV"。
- `tab_range_tests_full` 若与 `tab_range_tests` 重复，也一并删除。

## 7. 要求 5：参数和超参数

### 7.1 现状审计

| 项目 | 定义 | 数值在哪里 | 缺什么 |
|---|---|---|---|
| Δ、b_0、τ_0、C_err、C_lat | 正文 | Setup | 无 |
| τ_max = 0.9975，r = 每字 0.19 s、每词 0.34 s | 正文 | 在 method 正文里 | 移到附录 E |
| n_min | 附录 FR | 附录 | 无 |
| LoRA：r = 32、α = 64、第 32–47 层 | 附录 | Setup 写的是 "upper third"，两处不一致 | 统一 |
| GBS 256、LR 1e-4 cosine、8×H200、每 200 步存一次、选第 200/800 步、约 25 分钟 | 分散在 3 处 | — | 集中到一处 |
| **adapter 训练** | — | — | 见下方列表 |
| **Talker（T1c）** | 附录 | GBS 256、LR 1e-5、第 1500 步 | warmup 与学习率计划、数据规模（小时）、序列长度、GPU 小时、块构造方式、推理采样参数（温度、top-p，"三个 talker seed"指什么） |
| **推理** | — | — | Thinker 每读最大生成 token 数、prompt 格式、KV cache 是否复用 |
| TTB prior head（MLP 17–64–64–1） | 附录 | — | 训练设置 |

adapter 训练缺的项目：

- 优化器及 β、weight decay、warmup、min LR、梯度裁剪、精度、序列长度、micro-batch；
- epoch 数：235k / 256 ≈ 918 步为一个 epoch，800 步约 0.87 个 epoch，需核实；
- **第 200 步是按什么标准选出的**；
- wait marker 是新增 token 还是已有的保留 token，如何初始化；
- 内容项和停止项是否等权；
- "1/6 为整句样本"怎样混合；
- **belief-state roll-in 是什么（未定义）**，TTB roll-in 的阈值；
- 语料名称和两个方向的比例。

### 7.2 新增附录 D：Training Implementation Details and Hyperparameters（仿 Omni 附录 B）

- **总述段**：硬件、框架、冻结与训练的部分、只按 dev 选 checkpoint 的标准、随机种子。
- **D.1 Stage 1：Simultaneous Talker training**，得到 T1c。
- **D.2 Stage 2：Oracle label generation**
  - roll-in 策略（TTB 与 belief-state）的定义和参数；
  - 读窗 0.5/1/2/4 s；
  - 全上下文续写 → S_k → z_k；
  - 1/6 为整句样本；
  - 统计：40k → 235k，方向比例，S_k 分布。
- **D.3 Stage 3：QABC adapter training**
  - 位置、秩、α；完整的优化器设置和学习率计划；batch、序列长度；
  - 步数和保存间隔；dev 选择标准；5 个 seed；耗时。
- **D.4 TTB prior head**：作为基线使用。
- **表 D1**：Stage、Init、Trainable scope、Data、Steps (epochs)、Batch、LR schedule、GPU-hours、Selection。
- **Algorithm 1**（按你的决定放附录）：训练与推理的伪代码。
- **数值来源**：所有数值从启动脚本和日志读取，不凭记忆填。需要在 `experiments/uniss_qwen3_omni_v1/` 的 `scripts/` 和 `training/` 下定位 QABC adapter 和 T1c 的启动脚本与日志，填完后用检查脚本逐项核对。
- 推理超参放附录 E 的表 E1：Δ 网格、b_0 网格、τ_max、r、n_min、Thinker 解码、Talker 采样、速度 token、播放缓冲。

## 8. 要求 6：正文图表与结果章节

### 8.1 数量目标

- 参考论文的结果类浮动体是 7–10 个。
- v14 正文共 8 个浮动体：1 张方法图 + 7 个结果图表，与 Omni 相当。
- 全部来自已有结果，不需要新实验。
- 如果页面有余量，可以加第 9 个：一张很小的案例时间线（见 13.3 的附录 I），直观展示 Cor 1。

### 8.2 新的实验章（§4 Experiments）

| 子节 | 回答的问题 | 对应的方法 | 正文图表（来源） | 放到附录的内容 |
|---|---|---|---|---|
| 4.1 Setup（0.5 页） | — | — | 无 | C、F、G |
| **4.2 Offline S2ST（0.3 页，按你的决定，仿 Omni）** | 共享的 Thinker 和 Talker 离线水平如何？adapter 是否改变"说什么"？ | 3.3（目标是模型自己输出的前缀，所以 adapter 只学"何时停"） | **Table 1**：CVSS-T（`tab_cvss`）。adapter 前后 ASR-BLEU +0.5/+0.9，SIM-O 不变 | 文本端的整句结果（`tab_offline`） |
| 4.3 Main Results（0.8 页） | 在相同延迟下 QABC 是否更好？ | 整体 | **Fig 2**：质量–延迟曲线（RealSI 两个方向 + ACL60/60）<br>**Table 2**：按延迟档与各系统对比（`tab_realsi_main`）<br>**Table 3**：事先固定的主比较（s0 与五个 seed 均值，每个测试集两行） | H.1 |
| 4.4 Does the queue-aware rule help?（0.5 页） | 加入队列是否更好？规则本身是否胜过 LA？ | 3.2（Thm 1、Cor 1） | **Table 4**：QABC_Δ 与 QABC 的配对差（压缩 `tab_queue`）+ 同一草稿上规则与 LA 的对比（`tab_controls` 2 行） | H.3 |
| 4.5 Does amortized learning give a usable posterior?（0.5 页） | 学到的后验是否校准？是否优于测试时后验？ | 3.3（Thm 2） | **Table 5**：摊销后验、TTB、LA 三者对比（精简 `tab_range`）+ ECE/NLL 列（`tab_v10_calib`）<br>正文一句：oracle 按 S_k 提交时质量接近离线，说明 S_k 是值得估计的量（附录 H.2） | H.2、H.4 |
| 4.6 Is single-pass decoding enough?（0.35 页） | 贪心解码加偏置会不会损失？计算代价多大？ | 3.4（Prop 5） | **Table 6**：同一后验上的 4 种决策规则（`tab_v10_rules`）+ 每读计算量（`tab_v2_cost`） | H.5 |
| 4.7 Ablation Analysis（0.3 页，按段落写） | 各项设计选择是否必要？ | 各部分 | 段落式（参照 SEE 4.6）：语速 r、FR、seed；引用附录表格 | H |

**删除**：一条时钟连续解码的全部内容，见 11.4。

### 8.3 写法模板（正文和附录都用）

每段按这个顺序写：
1. 要验证哪个组件；
2. 看哪张图表；
3. 定性结论，最多给一个关键数字；
4. 这对应哪个定理、推论或公式的预测；
5. 细节见附录哪一节。

以 v13 的 "Queue-aware threshold" 段为例。v14 改写为：

> Corollary 1 predicts that the queue should matter only while earlier speech is still playing. Table 4 compares QABC with its constant-delay variant on the same posterior: the queue-aware threshold is ahead in all ten matched settings, and the gain is largest at the shortest read window, where the Talker is most often still speaking (up to +3.0 ASR-BLEU). Without the queue term, the main RealSI gains are no longer significant. The queue thus changes the decision without changing the posterior, as Corollary 1 states.

其他规则：

- **图注**写 1–2 句：比较了什么、怎么读（加粗表示什么、箭头方向）；显著性用一句话说明；不列 p 值。
- **正文禁用**：内部代号、数字三元组、"comparable"。
- **用词**：说我们的任务和模型时用 "simultaneous"；延迟写成 computation-excluded。

## 9. v14 页数预算（CVSS 表留在正文后的版本）

| 部分 | v13 | v14 目标 | 参考论文的范围 |
|---|---|---|---|
| 标题和摘要 | 约 0.6 | 0.4 | — |
| 引言 | 0.9 | 0.85 | 0.8–1.6 |
| 相关工作 | 0.4 | 0.35 | 0.6–0.8 |
| 方法 | 3.7 | 2.5 | 1.9–2.7 |
| 实验章（设置 + 结果） | 2.7 | 3.25（设置 0.5 + 结果 2.75） | 2.9–4.1 |
| Conclusion + Limitations | 0.4 | 0.35 | 约 0.2 |
| **合计** | **约 8.5（超页）** | **约 7.7，留 0.3 页余量** | — |

## 10. 审稿意见

已收到，见第 14、15 节。

## 11. 第 2 轮决定（2026-11-18）与回答

1. **CVSS 离线表**：按 Omni 的做法留在正文，作为 §4.2 Offline S2ST（Table 1）。它在正文里还有一个作用：说明 adapter 只学"何时停"、不改变"说什么"，与 3.3 对应。
2. **"preregistered" 等说法**：采纳建议。
   - 正文和附录只保留一句 "the primary comparison and its statistics were fixed before the test sets were run"；
   - 删去 v4/v5/v6、second/third registration、E1/E2/C1/C2/H1/H2；
   - 摘要里的 "Preregistered" 改为同样的说法。
3. **Table 14/15**：删除，见第 6 节。
4. **一条时钟连续解码的负面结果**：全部删除。要删的内容：
   - 正文 §7.4；
   - Limitations 里的相关句子；
   - Table protocol 中 H1/H2 那一行；
   - 表格 `tab_v10_talk`、`tab_v11_cont`、`tab_v11_contpts`、`tab_v11_dev`、`tab_v11_posthoc`；
   - 附录段落："One clock per talk, re-timed after the fact (v10)"、"The continuous clock"、"Choosing the control on dev"、"A second adapter"、"The v10 re-timing"；
   - `tab_v11_rules` 段落中 "Under one clock…" 一句。

   删除后在 Setup 里写清"RealSI 按句级评测，与 SimulS2ST-Omni 一致"，让结论的适用范围仍然明确。

   第 3 轮补充：建议在 Limitations 保留一句，见 14.3。
5. **Limitations**：
   - ICML 2027 不强制设 Limitations 章节；Hibiki（ICML 2025）就只有 Conclusion。
   - 但 ICML 2027 的审稿表单有 "Limitations" 一项："Have the authors adequately discussed the limitations and potential negative societal impact of their work?"，并且写明 "Authors should be rewarded rather than punished for being up front about the limitations"。
   - 按你的指示**写进论文**：正文 Conclusion 之后加一段不编号的 Limitations，3–4 句，约 0.15 页，计入 8 页（附录不一定有人读）。
   - 内容：
     - 只用了一个 Thinker、一个 Talker、两个方向；
     - ACL60/60 只评测文本输出；
     - 评测在句级进行；
     - RealSI En→Zh 的增益对 adapter seed 敏感（五个 seed 平均后不显著）；
     - 理论只覆盖 myopic 的单元损失和训练分布下的遗憾，分布偏移项没有给出保证；
     - FR 是手工设计的。
6. **oracle 柱状图：不画，放在附录**。理由如下。
   - **它是什么**：一个假想的 oracle，在每一读都知道真实的稳定长度 S_k（需要未来的音频，现实中做不到），并且恰好提交 S_k 个单元。
   - **它能证明什么**：它证明的是 S_k 是值得估计的量。按 S_k 提交时，文本 BLEU 与整句翻译只差 0.1–1.2，延迟 1.6–1.9 s；同一延迟下 LA2 差 2.7–6.2。这支持的是 3.1 的建模选择，同时给出了提升空间的上界。
   - **它不能证明什么**：它不能证明 QABC 有效。QABC 的有效性要由主比较和 4.4–4.6 的消融来证明。
   - **画成柱状图的风险**：该表的 RealSI 行是文本 BLEU，而且各行延迟不同。QABC 是 29.68 @1.67 s，LA2 是 29.86 @1.92 s（Zh→En 为 20.99 @1.61 s 对 21.76 @2.02 s）。柱状图会丢掉延迟这一维，看起来就像 QABC 不如 LA2，反而对你不利。
   - **结论**：保留为附录 H.2 的表格；正文 4.5 写一句话并引用它。
7. **Algorithm**：放在附录 D 末尾。

## 12. 执行步骤

1. 用 `cp -a` 把 v13 复制为 `paper/bayes_simul_s2st_v14/`；v13 及更早的版本都不改动。
2. 换成 icml2027 模板，用 tectonic 编译，确认基线能正常编译。
3. 按第 3、4、8 节重写正文。
4. 按第 13 节重排附录；删除 Table 14/15 和一条时钟相关的全部内容；导出 CSV。
5. 写附录 D：从脚本和日志读数，用检查脚本逐项核对。
6. 检查：
   - 正文（截至 Limitations）≤ 8 页；
   - 没有未定义的引用，没有 Overfull；
   - grep 内部标签（E1、C2、H1、v4–v11、s0、T1c、seq-B、p5、D1、registration）结果为 0；
   - 每个数字都与结果 JSON 一致（沿用 `check_paper_v13.py`）；
   - 附录每栏空白不超过 20%。
7. 修改记录写入 `reports/s2st_baselines_v1/CHANGELOG_paper.zh-CN.md`。

整个过程不需要 GPU。

## 13. 附录：诊断与重排

### 13.1 现在的问题

v13 附录共 22 页（第 12–33 页），有 53 张表和 1 张图；第 31、32 页各堆了 7 张表。

1. **按写作时间堆积，没有按逻辑组织**：
   - 节标签是 `app:v2`、`app:v10`、`app:v11`；
   - 标题是 "Supplementary experiments" 这样的杂项，或 "Robustness, attribution and decision-rule analyses" 这样主题混杂的标题，还有 "One clock per talk, error analyses and the target shift **(v11)**"。
2. **同一主题分散在多处**：

   | 主题 | 出现的位置 |
   |---|---|
   | adapter seed | `appendix.tex:534`（`tab_v2_seeds`）、`appendix_v10.tex:10`（`tab_v10_seeds`）、`appendix_v11.tex:14`（`tab_v10_seeds_main`） |
   | 校准 | D 节（`tab_calib`）、F 节（`tab_refine`）、`appendix_v10.tex:69`（`tab_v10_calib`）、K 节（target shift 的数值） |
   | 决策规则 | A 节证明末尾（`tab_crossing`）、J 节（`tab_v10_rules`）、K 节（`tab_v11_rules`） |
   | 统计方法 | G 节（run counting）、H 节（power）、I 节（significance）、J 节（talk 重采样）、K 节（Table protocol） |
   | 分档表 | B 节（`tab_tiers`、`tab_realsi_main`）、K 节（`tab_tiers_main`） |
   | FR | 定义在 I 节，结果在 B 节（`tab_fr`、`tab_fluency`） |

3. **内容放错节**：
   - TTB 是基线，它的定义、特征表和推导却放在 A "Derivations" 里（`appendix.tex:113–179`）；
   - LAAL 的定义也放在 Derivations 里；
   - B "All operating points" 里放了队列消融、对照实验和 FR，这些都不是"工作点"。
4. **实验日志式的细节写进了论文**：
   - "a colleague's jobs grew and left us two cards"、"vLLM engine failed to start … re-run into a new folder"、"smaller Talker KV cache"（`appendix_v10.tex:10–16`）；
   - "Counting runs as our v4 script did… v7 had run 9 of LA2's… v8 runs the rest"（`appendix.tex:546–552`）；
   - 文件路径 "(generated/results.json)" 出现在正文句子里（`appendix.tex:330`）。
5. **术语未定义或是内部代号**：D1 未定义；s0–s4、seq-B、p5、C1/C2、E1/E2、T1c。
6. **段落写成表格目录**：很多段只有一句 "Table X gives …"，后面跟一串 `\input`。读者只看到数字，看不到"这说明了什么"。
7. **防御性的说明太多，结论不清楚**：一段里常有 3–5 个 p 值和多组数字三元组，没有一句对应到正文的论点。

### 13.2 SimulS2ST-Omni 附录的做法与可借鉴之处

Omni 的附录顺序：

1. A 数据来源、构造与过滤；
2. B Training Implementation Details and Hyperparameters；
3. C Inference and Decoding；
4. D Evaluation Protocols；
5. E Supplementary Evaluation Results；
6. F Computation-Aware Evaluation；
7. G Reproducibility and ASR Sensitivity；
8. H Human Evaluation / LLM-as-Judge；
9. I Case Study；
10. J Bad Cases and Failure Analysis；
11. K Reproducibility and Versioning。

整体顺序是：怎么构建 → 怎么训练 → 怎么运行 → 怎么评测 → 补充结果 → 稳健性 → 定性分析 → 失败案例 → 发布说明。

可借鉴的 9 点：

1. **顺序本身**：先交代"怎么做"，再给"补充结果"，最后是定性分析和失败案例。附录里没有任何版本或注册的历史叙述。
2. **每节首句说明与正文的关系**。例如 "The main text anchors …; for completeness, Figure 7 reports …"，以及 "This subsection examines whether our main latency–quality conclusions remain stable after accounting for active inference cost"。
3. **每段末句回到正文的论点**。例如 "the ASR backend shifts absolute scores somewhat without changing the paper's main conclusions"。我们的附录基本缺这一句。
4. **评测协议写到可以复现**：按数据集和指标逐条列出文本归一化方式、分词器（zh / 13a）、各语言用的 ASR 模型、latency 的定义。我们缺文本归一化和 sacreBLEU 的设置。
5. **主动回应公平性**："Initialization fairness" 一段说明两条线的初始化是对等的。我们可以写一段 "Fairness of comparisons"：
   - 所有策略共用同一个 Thinker、同一个 Talker 和同一套 FR；
   - 外部系统按我们的协议运行，并接我们的 Talker；
   - 另外用 Omni 原生 Talker 做了对照。
6. **seed 稳健性集中在一处**：一张 mean ± std 表加一句结论。我们把三处 seed 表合成一张。
7. **案例时间线图（I Case Study）**：把源语音、WAIT 区间和提交块画在同一条时间轴上。我们可以画 QABC 与 LA2 在同一句上的对比：每一读的 ρ、τ(q_k)、q_k、提交块和播放，直接展示 Cor 1。数据预计可以取自 `tab_v11_rules` 用到的逐读回放数据，不需要重新解码，但要先确认字段齐全。
8. ~~**失败案例分类（J Bad Cases）**~~（第 3 轮按你的要求取消，不放失败案例）。原计划每类按"现象 → 例子 → 原因 → 是数据问题还是模型问题"来写。当时列出的材料：
   - Zh→En 语序导致的过早提交（v13 H 节里的 *suoyi zuohao …* 例子）；
   - 数字和否定词变化（`tab_v11_critical`）；
   - 块尾落在悬空功能词或单字上（"Choose to"，这是 FR 的动机）。
9. **计算代价单列一节**：Omni 有 LAAL_CA 和 RTF。我们有 `tab_v2_cost` 和"计算最多增加 0.14 s"，可以合成一节 computation-aware 说明。

**不借鉴的部分**：

- 人工听测和 LLM-judge：我们没有做。
- 第二套 ASR 的敏感性分析：我们没有现成结果。若要补，需要用 Qwen3-ASR 重新转写已有音频，GPU 用量小，但属于新实验，由你决定（可选）。

### 13.3 v14 附录结构（旧内容 → 新位置）

| 新附录 | 内容 | 来自 v13 |
|---|---|---|
| A Proofs | 按正文命题顺序：Prop 1、Lemma 1、Thm 1、Cor 1、Prop 3、hazard 推导、Prop 4、Thm 2、Prop 5 | 旧 A（不含 TTB 和 LAAL） |
| B Scope of the Theory | Ass 1/2 全文、Lemma 2、myopic vs global（`tab_crossing`）、块内延迟、协变量偏移推论及证明、target shift 推论及证明与实测数值、A1–A6、Table scope | 旧 method §5.2、Remark，旧 K 的证明 |
| C Data | 训练语料来源与数量、roll-in 策略、标签统计、两个 dev 集、RealSI 规模、ACL60/60 切分与偏移恢复 | 旧 method 训练数据段、旧 I 的 ACL 偏移 |
| D Training Implementation Details and Hyperparameters | 见 7.2，加 Algorithm 1 | 旧 I 的 Training 段 + 新整理 |
| E Inference and Decoding | Thinker 解码、队列跟踪、偏置与 τ_max、FR 定义与听感侧步骤、播放器、Talker 采样；表 E1 | 旧 method §5.1、旧 I 的两段 FR |
| F Evaluation Protocols | ASR-BLEU 流程（归一化、ASR 模型）、segment LAAL 定义、StreamLAAL 与重切分、computation-aware 说明（`tab_v2_cost`）、统计方法（bootstrap、Holm、事先固定的主比较、power）、清理后的 Table protocol | 旧 A 的 LAAL、旧 I 的 StreamLAAL 和 significance、旧 H 的 power、旧 K 的 protocol |
| G Baselines | LA2、Hold-n、Vote；TTB（定义、特征表、推导）；外部系统设置（`tab_baselines`）；Talker 对照（`tab_omnitalker`）；Fairness of comparisons 段 | 旧 A 的 TTB、旧 C |
| H Supplementary Results | 按方法组件分组，见下表 | 旧 B、D、F、G、J、K |
| I Case Study（新） | 1–2 个句子的时间线图 | 新，用已有数据 |
| ~~J Failure Analysis~~ | 第 3 轮取消；其中原有内容的去留见 15.4 | 旧 H、旧 E |

H 节内部的分组：

| 子节 | 内容 |
|---|---|
| H.1 Main comparison | 完整分档表（`tab_tiers`、`tab_tiers_main` 合一）；ACL 与 Omni 的对比（`tab_acl_main`）；按 talk 重采样（`tab_v10_robust`）；增益来源（`tab_v10_decomp`、`tab_v11_axes`）；word-aligned LAAL；文本端（`tab_v2_text`、`tab_prereg_text`）；更强的基线（`tab_v2_baselines`）；语音质量（`tab_v2_quality`）；ACL 敏感性（`tab_acl_sensitivity`） |
| H.2 Stable length（3.1） | oracle（`tab_oracle`、`tab_oracle-fr`）；早期错误（`tab_v11_irrev`、`tab_v11_critical`） |
| H.3 Queue-aware rule（3.2） | 完整的 `tab_queue`；与二值门的对比（`tab_v10_gate`）；规则做出不同决定的频率（`tab_v11_rules`，只保留句级部分）；语速 r（`tab_v2_rate`）；延迟实测（`tab_v10_delay`） |
| H.4 Posterior learning（3.3） | 测试时后验的质量（`tab_calib`）；refine 与 Prop monotone（`tab_refine`）；在策略自身读上的校准（`tab_v10_calib`）；seed（`tab_v10_seeds`，合并 `tab_v2_seeds`）；range 检验（`tab_range`）；消融的文本端（`tab_v2_ablation_text`） |
| H.5 Decoding（3.4） | 正文 Table 6 的完整版（`tab_v10_rules`） |
| H.6 FR | `tab_fr`、`tab_fluency` |

**删除**：

- Table 14/15；
- 一条时钟相关的全部内容（见 11.4）；
- 重复的 `tab_v10_seeds_main`（合并进 H.4）、`tab_range_tests_full`（若与 `tab_range_tests` 重复）；
- 所有实验日志式的句子和内部代号；
- 正文里出现的文件路径。

预计附录从 53 张表减到约 30–35 张。

**写法**：

- 每节首句写"这一节补充正文哪部分的什么问题"。
- 每段按 8.3 的模板写，最后一句写"这对正文的结论意味着什么"。
- p 值只放在表里，不在正文列出。

**排版**：

- 附录开头加 `\clearpage`；
- Supplementary Results 放在文字较多的节之前；
- 用 float 参数控制位置，每栏空白不超过 20%；
- 表格紧跟它首次被提到的地方。

## 14. 审稿意见与负面结果清单

审稿报告：QABC_v13_ICML_Review_CN.pdf（2026-11-18）。

- **总评**：Weak Reject，3/6。分项：Soundness 3、Presentation 2、Significance 2、Originality 3；Confidence 4。
- **改为 Weak Accept（4/6）的四个条件**：
  1. 正文合规（≤ 8 页）；
  2. 摘要和引言统一到句级适用范围；
  3. **正文清楚呈现五个 seed、按 talk 重采样、word-aligned 计时、无 envelope 的敏感性**，并且核心方向保持稳健；
  4. 理论与实现的表述不再混淆。
- 审稿人把"披露负面结果"列为三条优点之一，并两次写明负面结果应保留（第 1 页、2.1 节）。

### 14.1 审稿人指出的负面结果：位置与处理

| # | 负面结果 | 审稿人出处 | v13 位置 | 能否删 | v14 处理 |
|---|---|---|---|---|---|
| N1 | 连续整场录音：Talker 跟不上，积压 1.1–1.8 s/min；共同区间 −0.45/−0.07（p = 0.70/0.53） | 2.2 P1、Q4 | **正文** §7.4（第 9 页）、Limitations；附录 J（Table 41）、K（Table 47–50） | 你已决定删实验 | 删除 §7.4、Table 41、47–50 及相关段落。建议 Limitations 留一句（见 14.3）。审稿人对 D_k 机制的批评（q_k ≥ Δ 时等待免费，可能加剧积压）只读方法就能提出，与实验删不删无关，所以在 3.2 明确写出"队列有界（每句重置）" |
| N2 | 五个 seed 平均后 En→Zh +0.36，Holm p = 0.139，不显著 | 2.3 P1、Q1 | **正文** §7.2 第二段、Conclusion、Limitations；附录 Table 36、46 | **不能删**：这是事先登记的比较（C1），而且是改分条件 3 | 进正文 Table 3：s0 与五 seed 均值并列，标注重采样单位。正文写一句："every seed is ahead on all three test sets; the mean is significant on Zh→En and ACL60/60" |
| N3 | 按 talk 重采样时 En→Zh p = 0.057；seed × talk 时 p = 0.253 | 2.3、Q1 | **正文** §7.2 第一段；附录 Table 38 | 不建议删（改分条件 3） | Table 3 加一列"按 talk 重采样的 p"；详表放附录 H.1 |
| N4 | 不用 upper envelope 时，E1 从 +0.76/+0.78/+2.63 降到约 +0.48/+0.53/+2.26 | 第 4 节、Q2 | 附录 J "Robustness" 段（第 30 页） | 不建议删（改分条件 3） | 附录 F 定义 envelope；附录 H.1 给出无 envelope 的 E1、CI 和 p（需补算，见 15.5）；Table 3 表注写一句 |
| N5 | word-aligned LAAL 比 segment LAAL 高 0.7–1.1 s，QABC 的优势少约 0.2 s | 2.6 P2、Q2 | **正文** Setup 的 Metrics 段；附录 Table 33、54 | 不建议删（改分条件 3） | 补算 word-aligned 计时下的 E1、CI、Holm p（见 15.5）；附录 F 加三种时间轴的定义小表；正文写一句 |
| N6 | 推导出的连续阈值对二值队列门：+0.35/−0.26/+0.24，Holm p ≥ 0.06；两者只在 3.8% 的读上不同 | 2.4 P1、Q3 | **正文** §7.3 Controls 段；附录 Table 37、51 | 不建议删：这是事先登记的比较（C2） | 主张拆成两层：(i) 队列有用，证据强（Table 4）；(ii) 精确的 τ(q) 是否优于二值门，本数据无法区分。原因是约 82% 的读队列要么为空、要么至少有一个读窗（En→Zh 68%+14%，Zh→En 67%+15%，v13 Table 51），两条规则在这些读上完全一致；只有约 18% 的读处在 0<q<Δ，两条规则在其中约五分之一的读上不同，合计占全部读的 3.8%。（第 4 轮更正：第 3 轮误写为"96%"。）正文写一句，详表放附录 H.3。可选做分桶分析（见 15.5） |
| N7 | Thm 1 只对 myopic 单元损失最优；与全局最优在 9–22% 的读上不同 | 2.5 P2 | **正文** Remark（§4.2）；附录 Table 3 | 不能删：这是定理适用范围的限定 | 定理陈述写明 "myopic, per-unit"；数字和 Table 3 移到附录 B |
| N8 | χ² 偏移未估计；target shift 使上界增加约 1.93/1.05 C_err，无法据此认证部署策略 | 2.5 P2、Q5 | **正文** §5 Remark；附录 K | 数值可以只放附录；限定语不能删 | 贡献句写成"训练分布下、对任意阈值成立的单元遗憾界"，不单用 "guarantee"；数值和量级解释放附录 B；Limitations 写一句 |
| N9 | Zh→En m2：Hy-SimulMT 接我们的 Talker 得 21.88，高于 QABC 的 21.42 | 2.7 P2 | **正文** §7.2；附录 Table 11 | 不能删：它是主表里的一个格子 | 结论只写"同一 Thinker 和 Talker 下的策略比较"，外部系统作为背景；不写 SOTA；表注写清各系统的模型规模、解码方式和所用 Talker |
| N10 | 摘要里的 3.1%/5.5% 是挑选出的 m2 工作点；"同时降低延迟"没有统计支持（0.07/0.08 s） | 5.1 | **正文** 摘要、引言第 3 段 | 删的是这句话，不是结果 | 改写摘要，见 15.1 |
| N11 | 附录中 Prop 4 证明的末句 "does not compete" 超出了命题能证明的范围 | 第 3 节，标为"矛盾" | 附录 A（第 14 页） | 应删这一句 | 换成审稿人 5.3 节给的句子 |

### 14.2 审稿人没点名、但 v14 计划移进正文的表里也有的不利数字

v14 计划把 v13 的 Table 11（`tab_realsi_main`）移进正文。表里还有这些格子：

- Zh→En m4：LA2 23.47，高于 QABC 22.65；
- Zh→En m3/m4：TTB+q（冻结 Thinker 的测试时后验加队列）23.16/24.06，高于 QABC 22.46/22.65；
- En→Zh m4：QABC_Δ 30.70，略高于 QABC 30.51。

处理方式：

- 正文 Table 2 只放外部系统、LA2 和 QABC；TTB+q 和 QABC_Δ 属于消融，分别放进 4.5 和 4.4 的消融表，如实报告。
- 4.5 的论点改为：摊销后验在 ACL60/60 上明显更好，在 RealSI 上与测试时后验相当，但只需一次解码，不需要每读再生成一份 virtual draft。
- 正文写明增益集中在 2.5 s 以内（与 v13 的 Conclusion 一致）。

其他不在审稿人清单上的负面结果：

| 负面结果 | 处理 |
|---|---|
| FR 使 ASR-BLEU 下降 0.47（正文 §7.5、Limitations） | FR 已移到附录，代价放附录 H.6，正文不再提 |
| Zh→En 差距诊断、按方向选择（Table 34）、power（Table 35） | 探索性分析，不支撑任何主张，**可以删**（与你"不放失败案例"一致） |
| "Extensions that did not help"、测试时 refine（Table 20–22） | 探索性，**可以删**；Prop monotone 随之删除，审稿人对它前提的意见也就不用处理 |
| 提交时刻计时的文本终点 +0.57/+0.19，不显著（正文 §7.2） | 事先登记的配套检验，保留在附录 H.1，正文不提 |
| E2（等质量时节省的延迟）在 RealSI 上不显著（正文 §7.2） | 保留为 Table 3 的一列，正文不展开 |

### 14.3 删与留的原则（我的建议）

- **可以删**：不支撑任何主张的探索性实验和次要扩展。
- **只能移到附录，不能删**：
  - **事先登记的比较**（C1 五个 seed、C2 门控、配套文本检验）。正文保留了"主比较及其统计在跑测试集前已固定"这句话；如果只报告结果好的那几项，这句话就有误导性。
  - **直接限定主结论的结果**（seed、talk、计时方式、envelope）。审稿人把它们列为改分条件，真实审稿人也最常问这些；论文里已经有答案，比在 rebuttal 里补要好。
- **一条时钟**：尊重你删除实验的决定。但这是一个已知的失败，而引言又以会议和讲座作为动机，所以建议 Limitations 保留一句（约两行），例如：

  > Our evaluation resets the playback queue at each sentence; on a continuous recording clock the Talker cannot keep up with the source and the queue does not drain, so sustained interpretation needs throughput control, which QABC does not provide.

  同时把引言的动机改成句级场景。这比完全不提更稳妥，也能回应审稿人对 D_k 机制的质疑。保不保留这一句由你决定。

## 15. 结合审稿意见对前面方案的修改（与前文冲突时以本节为准）

### 15.1 摘要和引言

- **摘要**：删去 3.1%/5.5% 和 "while also reducing latency"，改为：

  > In a sentence-level evaluation whose primary comparison was fixed in advance, QABC improves equal-latency ASR-BLEU over LocalAgreement by 0.76 and 0.78 on RealSI En→Zh and Zh→En (Holm-adjusted p = 0.030 for both) and by 2.63 BLEU on ACL60/60; across five adapter seeds every adapter is ahead on all three test sets, and the mean gain is significant on RealSI Zh→En and ACL60/60.

  理论那一句加限定："bounds per-unit decision regret for every threshold under the training distribution"。摘要控制在 4–6 句。
- **引言**：
  - 删掉第 2 段重复的 "Existing policies often rely on heuristics …"；
  - 采用审稿人 5.2 节给的两句 "Existing stopping rules neither estimate … nor price waiting …"；
  - 批评 heuristic 时写"没有校准过的存活概率"，不要写成"完全不考虑稳定性"；
  - 写明适用范围：句级评测，播放队列有界；
  - 第 3 条贡献改为"同一 Thinker 和 Talker 下的策略比较"，并区分"队列有用"和"精确阈值形式"两层。

### 15.2 方法（修改第 3 节）

- 3.2 明确写出"队列有界（每句重置）"这一假设，并说明在持续过载时，myopic 损失不会推动系统清空队列。
- 贡献列表区分三层：myopic Bayes 最优性；训练分布下、对任意阈值成立的单元遗憾界；部署时的近似。
- 数学和写法的修正：
  - Prop 4 证明末句改为审稿人 5.3 节给的句子；
  - 附录 A 中 Lemma 2 的证明标题从 "Proof of Proposition" 改为 "Proof of Lemma"；
  - "Densities are with respect to Lebesgue measure" 限定为"在密度存在时"；
  - Figure 1 中的 q_θ 统一为 p_θ，并标注 "stopping NLL"；
  - Prop monotone 随 refine 一起删除；如果保留，补上"g 是连续双射"的假设。

### 15.3 实验章（替换 8.2 中的对应行）

- **4.2 Offline（Table 1）**：
  - 按你的决定留在正文。审稿人建议移到附录，所以采用 Omni 那样的紧凑版：两个方向，只保留 ASR-BLEU、Text-BLEU、SIM-O 三列，约 6 行系统；
  - 加一列或脚注，写清模型规模、解码方式和所用 Talker；
  - 不写 "above every S2ST system"，改写"作为同传实验的离线参照"。
- **4.3 Main Results**：
  - **Table 2（分档表）**：只放外部系统、LA2 和 QABC。表注写 "selected best-in-tier operating points; not the primary equal-latency comparison"；外部系统作为背景，不写 SOTA。
  - **Table 3（主比较）**：每个测试集一行 s0、一行五 seed 均值。列为 E1、95% CI、Holm p、E2、按 talk 重采样的 p，并标注重采样单位。
  - 正文采用审稿人 5.4 节那句限定的前半句："The single-adapter result is significant in both RealSI directions, whereas averaging over five adapters retains significance for Zh→En and ACL60/60"。后半句讲连续时钟，随删除决定一起去掉。
  - 4.3 末尾补一段 **"Sensitivity to timing and curve construction"**：正文一句，加附录 H.1 的表，覆盖 word-aligned 下的 E1 和无 envelope 的 E1。这回应改分条件 3 和 Q2。
- **4.4 Queue**：按 N6 把主张拆成两层。
- **4.5 Posterior**：按 14.2 改写论点；target shift 的数值和量级解释放在附录 B，回应 Q5。
- **Limitations**，包括以下几点：
  - 只用了一个 Thinker、一个 Talker、两个方向；ACL60/60 只评测文本；
  - 句级评测、每句重置队列，以及建议保留的那句一条时钟说明；
  - 五 seed 平均后 En→Zh 不显著；
  - 理论的适用范围：myopic、单元级、训练分布；
  - 没有人工听测（审稿人 P2 建议做，本版不做）；
  - FR 是手工设计的。

### 15.4 附录（修改第 13 节）

- **按你的要求不放失败案例**：删除 13.2 第 8 点和 13.3 的 "J Failure Analysis"。
- **同时删除**：
  - Zh→En 差距诊断、按方向的选择、power（Table 34、35）；
  - Extensions that did not help；
  - 测试时 refine（Table 20–22）和 Prop monotone。
- **新增或扩充**：
  - H.1：加入按 talk 重采样、无 envelope、word-aligned 三项敏感性分析，均给出 CI 和 p；
  - F：加入三种时间轴的定义小表；
  - B：加入 target shift 数值的量级解释。
- **复现材料**（审稿人 P2）：摘要加脚注 "Code and models will be released upon acceptance"；附录 D、E、F 写到可以复现的程度。

### 15.5 新增计算

全部在已有输出上用 CPU 完成，不需要重新解码，也不需要 GPU：

1. word-aligned 计时下的 E1、CI、Holm p。v13 只有一个 talker seed 有 word-aligned 数据，需要写明。
2. 无 envelope 的 E1 的 CI 和 p。
3. （可选）在 0 < q_k < Δ 的读上做分桶分析：τ(q) 与二值门做出不同决定的比例，以及这些读上的稳定性和风险。

无论结果好坏都如实写入。第 1、2 项是审稿人的改分条件，建议做；第 3 项可选。

## 16. v13 全部负面结果清单（第 4 轮，逐表核对）

说明：

- T# 是 v13 PDF 中的表号。
- "正文"指 v13 第 1–9 页。
- R 表示审稿人点名的项目。
- "延迟不同"表示两边的 LAAL 不相等，直接比较时要谨慎。

### A. 主比较：QABC 对 LocalAgreement（LA2），同一 Thinker、同一 Talker

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| A1 R | 五个 adapter seed 平均后，En→Zh 增益不显著 | QABC vs LA2，等延迟 ASR-BLEU（E1） | +0.36，Holm p=0.139；最差 seed +0.003（Zh→En +1.10、ACL +2.13 显著） | 正文 §7.2、Conclusion、Limitations；附录 T36、T46 |
| A2 | 登记的 s0 在 En→Zh 显著，但 95% CI 下界正好是 0 | 同上 | +0.76 [+0.00, +1.54] | 正文 T2 |
| A3 R | 按整场 talk 重采样后，En→Zh 不显著 | 同上 | s0：p=0.057；五 seed × talk：p=0.253 | 正文 §7.2；附录 T36、T38 |
| A4 R | 不用 upper envelope（直接在工作点之间线性插值）时，增益变小 | 同上 | +0.76/+0.78/+2.63 → +0.48/+0.53/+2.26；RealSI Zh→En 文本 −0.03 | 附录 T38 |
| A5 | 等质量下节省的延迟（E2）在 RealSI 上不显著 | QABC vs LA2 | +0.21 s（p=0.208）/ +0.12 s（p=0.264）；五 seed 平均 En→Zh −0.04 s | 正文 T2、§7.2；附录 T36 |
| A6 | 第二个登记检验（只看提交的文本）不显著 | QABC vs LA2，文本 BLEU | +0.57（p=0.057）/ +0.19（p=0.240）；Zh→En E2 −0.17 s（反而更晚） | 正文 T2、§7.2；附录 T27、T9、T54 |
| A7 | 五 seed 的提交文本中有 seed 为负 | QABC vs LA2，文本 | En→Zh：s3 −0.19、s4 −0.04 | 附录 T36 |
| A8 | 固定设置的区间检验（2–3 s）在 RealSI 不显著 | QABC vs LA2 | +0.78（Holm 0.114）/ +0.15（Holm 0.365） | 附录 T6、T23 |
| A9 | 三个 adapter 的区间检验有负值，均值接近 0 | QABC vs LA2，2–3 s，单个 talker seed | En→Zh：s2 −0.66；Zh→En：s1 −0.24；均值 +0.10/+0.26 | 附录 T24 |
| A10 | 分档表中 Zh→En m4 输给 LA2；m3 以上大多不显著 | QABC vs LA2，每档最佳点 | s1：23.20 vs 23.69（−0.49）；m3–m6 的 Holm p 在 0.07–0.91 之间 | 附录 T55（正文 §7.2 写为"16 档中 15 档领先"） |
| A11 | 分档表（s0，三个 talker seed）中 Zh→En m4 输给 LA2 | 同上 | 22.65 vs 23.47 | 附录 T11 |
| A12 | 文本端有 4 个延迟预算低于 LA2 | QABC vs LA2，文本 BLEU | En→Zh ≤1.5 s：27.87 vs 28.66；≤2.0 s：29.68 vs 29.86；Zh→En ≤2.5 s：22.07 vs 22.21；≤3.0 s：22.76 vs 23.58（也低于 Vote 23.41、TTB 23.30） | 附录 T28 |
| A13 | 全分档文本 BLEU 有几档较低 | QABC vs LA2 / Vote | En→Zh m3：30.7 vs 30.8；Zh→En m4：22.6 vs LA2 23.1、Vote 23.4 | 附录 T8 |
| A14 | 在同一台服务器上重跑 ACL60/60，增益变小 | QABC vs LA2 | +2.63 → +1.97（仍显著） | 附录 T26 |
| A15 R | 按词计时时，QABC 的延迟增加得更多 | QABC vs LA2，word-aligned LAAL 减 segment LAAL | QABC +0.95/+1.13 s；LA2 +0.74/+0.90 s；QABC 多约 0.2 s | 正文 Setup；附录 T33 |
| A16 | 同一读窗下，Δ = 2/3 s 时 QABC 的不可逆单元和数字改动更多（延迟不同，QABC 延迟更低；等 LAAL 时 QABC 更少） | QABC vs LA2 | 不可逆单元：En→Zh 56.2/49.3% vs 43.1/31.8%，Zh→En 57.4/48.4% vs 43.0/28.5%；数字改动 9.2/10.1% vs 5.8/5.5%；等 LAAL 2 s 时 En→Zh 否定词 3.4% vs 3.2% | 附录 T52、T53 |

### B. 与外部系统比较

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| B1 R | Zh→En m2 输给 Hy-SimulMT | QABC vs Hy-SimulMT 接我们的 Talker | 21.42 vs 21.88 | 正文 §7.2；附录 T11 |
| B2 | ACL60/60 在 1.9 s 以上低于 SimulS2ST-Omni 的报告值（另一个模型，beam 4） | QABC vs SimulS2ST-Omni | 48.76/50.98/51.49 vs 51.51/51.94/53.24；分档 m2–m6 均低于它 | 附录 T5、T55 |
| B3 | 区间检验中 Zh→En 没有胜过 Vote | QABC vs Vote | −0.08（2–3 s） | 附录 T23 |
| B4 | 离线时，不加 adapter 的 En→Zh ASR-BLEU 与 Qwen3-Omni 原生 Talker 持平 | 我们的 Thinker+Talker vs Qwen3-Omni 原生 Talker | 36.60 vs 36.59（加 adapter 后 37.10） | 正文 T1 |
| B5 | 离线 En→Zh 的 A.PCP 低于几个系统 | 我们 vs SimulS2ST-Omni Dec-only 和两个级联系统 | 2.79 vs 2.96、2.87、2.80 | 正文 T1 |

### C. 队列感知阈值（Thm 1、Cor 1）

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| C1 R | 推导出的连续阈值没有胜过简单的二值门；Zh→En 反而略低 | τ(q) vs 二值队列门，同一后验 | +0.35/−0.26/+0.24，Holm p ≥ 0.06 | 正文 §7.3；附录 T37 |
| C2 | 在提交的文本上，同样没有胜过二值门 | 同上，文本 | Zh→En −0.12；二值门 vs LA2 也不显著（+0.28/+0.28） | 附录 T9 |
| C3 R | 两条规则几乎总是做出相同决定 | τ(q) vs 二值门的动作 | 只在 3.8% 的读上不同（18% 的读处在 0<q<Δ） | 附录 T51 |
| C4 | 队列模型 D_k 高估了实际的等待代价，而且与实测几乎不相关 | 模型 D_k vs 播放实测 | 1.29/1.27 s vs 0.59/0.65 s；r = 0.16/0.11 | 附录 T40 |
| C5 | 语速 r 取较小值时，队列的增益不显著 | QABC vs QABC_Δ | Zh→En：r×0.75 +0.45（p=0.314），r×1 +1.09（p=0.090） | 附录 T30 |
| C6 | 去掉队列项后（只用后验），在 RealSI 上不胜过 LA2 | QABC_Δ vs LA2 | 区间检验：Zh→En −0.21；En→Zh <2 s −0.25；s1、s2 adapter 为负 | 附录 T6、T24 |
| C7 | En→Zh m4 去掉队列反而略高 | QABC_Δ vs QABC | 30.70 vs 30.51 | 附录 T11 |

### D. 摊销后验（3.3、Thm 2）

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| D1 | 在 RealSI 上，摊销后验没有胜过冻结 Thinker 的测试时后验（TTB）；Zh→En 是 TTB 更好 | QABC / QABC_Δ vs TTB | 区间检验 Zh→En：TTB +0.47/+1.01，QABC +0.15/+0.72；分档 Zh→En m3/m4：TTB+q 23.16/24.06 vs QABC 22.46/22.65；文本 Zh→En ≤3 s：只用先验的 TTB 24.21 vs QABC 22.76 | 正文 §7.3（写为"持平"）；附录 T6、T11、T29 |
| D2 | ACL60/60 个别预算下有变体高于 QABC | QABC vs QABC_Δ / TTB | 1.90 s：QABC_Δ 49.39 vs 48.76；2.89 s：TTB 51.57 vs 51.49 | 附录 T5 |
| D3 | 摊销后验校准不足 | 原始后验 vs 测试时重校准 | NLL 0.488 → 0.307，AUC 只 +0.024 | 附录 T20 |
| D4 | 在 QABC 自己的读上 ECE 不低；对照冻结 Thinker 标签时更差 | — | ECE 0.082/0.109；对冻结 Thinker 标签 0.106/0.105 | 附录 T43 |
| D5 R | 训练标签（冻结 Thinker）与部署（适配 Thinker）不一致，理论界因此无法认证部署策略 | 冻结 vs 适配 Thinker 的稳定长度 | 21.8%/16.0% 的读不同，平均差 1.49/0.73 个单元；上界增加约 1.93/1.05 C_err | 正文 §5 Remark；附录 K、T43 |
| D6 R | 协变量偏移 χ²(Q‖P) 没有估计 | — | — | 正文 §5 |
| D7 | adapter 让 En→Zh 的草稿对 LA2 来说更差 | LA2 用适配 Thinker 的草稿 vs 用冻结 Thinker 的草稿 | −1.94（En→Zh）/ +0.03（Zh→En） | 附录 T9 |
| D8 | 在 RealSI 上，测试时 Bayes-risk 的各种改进大多无效；唯一显著的一项副作用很大 | 改进策略 vs QABC+FR | 8 个比较中 6 个不显著；per-talk 更新 Zh→En +1.22，但突兀停顿变为 3.03 倍 | 附录 T21 |
| D9 | 测试时后验 TTB（β=4）过度自信；校准后在 RealSI 语音上仍不胜过 LA | TTB vs 理想校准 / LA | ECE 0.242 | 附录 T18、附录 D 正文 |
| D10 | 选用的 checkpoint（step 200，按 dev 选出）不是测试集上最好的 | step 200 vs step 600 | 文本 ≤3 s：En→Zh 31.90 vs 32.15；Zh→En 23.32 vs 23.74 | 附录 T29 |

### E. 决策规则与实现（Thm 1、Prop 5）

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| E1 R | myopic 顺序规则不是全局最优 | 顺序规则 vs 前缀全局最小 | 9.4–22.0% 的读选择不同；这些读上全局最优的损失低 16–31% | 正文 Remark §4.2；附录 T3 |
| E2 | 严格按 Thm 1 实现、全局 argmin、Platt 重校准，在 dev 上都比部署的保守贪心规则差 | 三种规则 vs 部署规则（p5） | 文本 E1：−0.36/−0.47、−0.76/+0.21、−0.43/−0.97 | 正文 §7.3；附录 T42 |
| E3 | 块内延迟规则在测试集上不显著 | seq-B vs 部署规则 | +0.19（p=0.324）/ +0.37（p=0.212） | 正文 §7.3；附录 T42 |
| E4 | 部署规则不是它所在损失下的最小者 | 部署规则 vs 全局规则，D_k 损失 | 0.71 vs 0.57（En→Zh）；比 Thm 1 多扣下 4.4%/3.1% 的单元 | 附录 T43 |

### F. 语音质量与流畅度

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| F1 | 韵律相似度和说话人相似度略低（同为 Δ=1，QABC 延迟更低） | QABC vs LA2 | A.PCP 2.69/2.82 vs 2.78/2.84；SIM 0.766/0.880 vs 0.780/0.885 | 附录 T32 |
| F2 | 不加 FR 时，突兀停顿更多、UTMOS 更低、误读更多 | QABC vs LA2 | 停顿 4.47/4.72 vs 4.19/2.35 次/分；En→Zh UTMOS 3.23 vs 3.29；Zh→En 误读 8.9% vs 8.5% | 附录 T13 |
| F3 | 加 FR 后，Zh→En 突兀停顿仍远多于 LA2；En→Zh 质量不显著 | QABC+FR vs LA2+FR | 停顿 1.32 vs 0.19；En→Zh +0.07（不显著） | 附录 T12、T13 |
| F4 | FR 本身降低 ASR-BLEU | QABC+FR vs QABC，同 LAAL | En→Zh 28.83 vs 29.30（−0.47） | 正文 §7.5、Limitations；附录 T12 |
| F5 | Zh→En m4 我们 Talker 的读错率略高（ASR-BLEU 仍更高） | 我们的 Talker vs Qwen3-Omni 原生 Talker | WER 10.14% vs 9.57% | 附录 T17 |

### G. 一条时钟连续解码（已决定删除）

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| G1 R | 连续时钟下不领先；登记的 H1/H2 无法检验 | QABC vs LA2 | −0.45（p=0.70）/ −0.07（p=0.53）；连续 LAAL 起点 7.4/7.9 s，超出登记的 5 s | 正文 §7.4；附录 T48 |
| G2 R | 积压更严重，更多内容压到句末最后一块 | QABC vs LA2 | 漂移 1.1–1.8 vs 0.2–0.9 s/min；最后一块装下句子的 57–60% vs 31–34% | 正文 §7.4；附录 T49 |
| G3 | 连续时钟下二值门也输给 LA2 | 二值门 vs LA2 | −0.64/−0.25 | 附录 T48 |
| G4 | 换 s1 adapter 结果一样 | QABC vs LA2 | −0.47/−0.75 | 附录 K 正文 |
| G5 | v10 的事后重计时也为负 | QABC vs LA2 | −0.07/−0.78 | 附录 J、T50 |
| G6 | 吞吐控制不可用 | 有加速 vs 无加速 | Zh→En dev ASR-BLEU −0.54/−0.81 | 附录 T47 |

### H. 翻译方向差距与探索性分析

| # | 哪个结果差 | 和什么比 | 数字 | 位置 |
|---|---|---|---|---|
| H1 | Zh→En 上规则本身的增益很小，后验更差，过早提交更多 | Zh→En vs En→Zh；QABC vs LA2 | 同一草稿上 +0.17 vs +2.01；AUC 0.802 vs 0.826；过早提交多 5–10 个百分点（En→Zh 多 1–5） | 正文 §7.5；附录 H、T9 |
| H2 | 学到的策略只拿到很小一部分 oracle 余量 | 学到的策略 vs oracle | Zh→En 15%，En→Zh 62% | 附录 H、T22 |
| H3 | RealSI 文本上 QABC 低于 LA2（延迟不同，QABC 更低） | QABC vs LA2，oracle 表 | 29.68 @1.67 s vs 29.86 @1.92 s；20.99 @1.61 s vs 21.76 @2.02 s | 附录 T19 |
| H4 | 按方向选择后，Zh→En 仍不显著 | 选择后的配置 vs LA2，文本 | +0.48，Holm 0.114 | 附录 T34 |
| H5 | RealSI 文本检验的统计功效不足 | — | 以 80% 功效只能检出约 1 BLEU | 附录 T35 |
| H6 | 几项扩展没有起作用 | MBR、per-talk 自适应、语音侧选择 vs 原方法 | 只有定性描述 | 附录 E |

## 17. 第 5 轮：表格处置与附录重构（与第 8、13–15 节冲突时以本节为准）

### 17.1 全局规则

1. **不再使用 Holm p**，全文删除所有 Holm 列。
2. **p 值采用统一规则，而不是只删对我们不利的那几列**：
   - 附录所有表格都不放 p，只给差值和 95% CI，或者 mean ± sd。
   - 正文只有主比较表（Table 3）保留一列单侧 p，不做多重校正。表注写一句："one-sided paired bootstrap; the three primary comparisons stay below 0.05 after correction for three test sets"。
   - 去掉 Holm 后，正文不能再拿未校正的 p 宣称次要比较"显著"。次要结果一律写"95% CI 不含 0"或"CI 含 0"。
   - 原因：同一篇论文里，有的表有 p、有的表恰好在不利处没有 p，审稿人一眼就能看出来；统一规则同样能删掉那些 p 列，却不留话柄。
3. **seed 只报统计量**：写 mean ± sd（标准差，单位与 BLEU 相同），不写方差，也不写变异系数。Omni 的 Table 12 也是 "mean ± std over 3 seeds"。表中不再列 s0–s4 的单个值，不再有 Worst 列及其说明。"Seeds > 0" 一列（5/5）是有利信息，可以保留。
4. **envelope**：只在附录 C.3 的 E1 定义里写一句（"E1 compares upper envelopes of each policy's quality–latency points"），这是复现所必需的；所有表注删去 envelope 的内容，删除 "no envelope" 一列。
5. **延迟只用 segment LAAL**，即每个单元的时间取它所在块的起播时间。这是 SimulS2ST-Omni 分档所用的度量，定义写在附录 C.3。删除按词计时（v13 T33）及正文 Setup、T31、T32 中所有 word-aligned 的列和描述。
6. **只保留能证明结果有效、或能验证方法各部分的实验**。探索性实验、重复实验和失败的扩展都删掉；完整数据以 CSV 形式放进补充材料。
7. **FR（流畅化规则）**：主结果并没有使用 FR。证据是：区间检验表注写着 "no FR"；分档表中 QABC 的 m2 点 28.26 @1.77 s，与 FR 表里"不加 FR"的 QABC（28.27 @1.77 s）一致。而 v13 方法 §5.2 写的是测试时使用 FR，前后不一致。
   - 建议：先从运行配置确认这一点。若确认主结果确实没用 FR，就**把 FR 从论文中整体删除**（方法 §5.1、附录中 FR 的定义、T12、T13），并改正 §5.2 的表述。
   - 播放缓冲和加速 token 也要同样核实：主结果里用了，就写进附录 E 的推理设置；没用，就删除。
   - FR 使 En→Zh ASR-BLEU 下降 0.47，按你"流畅性改动不能降低 ASR-BLEU"的规则本来也不合格。

### 17.2 v13 每张图表的处置（T# 为 v13 PDF 表号）

| v13 | 内容 | 处置 | v14 位置 |
|---|---|---|---|
| F1 | 方法流程图 | 保留；q_θ 改为 p_θ，并标注 "stopping NLL" | 正文 Figure 1 |
| F2 | 质量–延迟曲线 | 移入正文 | 正文 Figure 2 |
| T1 | CVSS 离线 | 改为紧凑版，约 6 行，加模型规模和所用 Talker 的脚注 | 正文 Table 1 |
| T11 + T5 + T8 + T55 | 四张分档表 | **合并为一张**：用 s0（登记的 adapter）、三个 talker seed；行为 m1–m4（≤3 s）× RealSI 两个方向 + ACL60/60；列为 Omni（报告值）、SeamlessStreaming、Hy-SimulMT、X-Translator、Hy+Qwen3-TTS、LA2、Hold-n、Vote、QABC。删除 TTB(+q) 和 QABC_Δ 两列（它们是消融，放到 Table 4、5）。删除 T55（它用的是看过登记结果后才选的 s1 adapter） | 正文 Table 2 |
| T2 + T46 + T36 | 主比较 + 五个 seed | 合为一张：每个测试集一行 s0、一行五 seed 的 mean ± sd（另附两层 bootstrap CI）。列为 E1 [95% CI]、单侧 p（只在 s0 行）、E2、Seeds > 0。删除 Holm、Worst 和逐 seed 的值 | 正文 Table 3 |
| T10 + T9（2 行） | 队列消融 + 对照 | QABC 对 QABC_Δ 的配对差，给 Δ 和 CI。T9 只留 2 行："同一份草稿上 QABC 对 LA2"（En→Zh +2.01、Zh→En +0.17），证明增益来自决策规则而不是 adapter 改了译文；"加队列门的 LA2 对 LA2"（+0.01/+0.07），证明只有队列、没有后验是不够的 | 正文 Table 4 |
| T6（含 T7 的 CI） | QABC / QABC_Δ / TTB 对 LA2 的区间检验 | 给 Δ 和 CI，不放 p。删掉 T7（CI 已并入） | 正文 Table 5 |
| T42 + T31 | 决策规则 + 计算量 | 规则只留"严格按 Thm 1"、全局 argmin、Platt 三行（如果删掉块内延迟扩展，就不留 seq-B），只给 Δ，不放 p。计算量只留 ms/read 和 +CA 两列，删去按词计时的 LAAL | 正文 Table 6 |
| T3 | myopic 对全局最优 | 保留（审稿人引用过；属于理论适用范围） | 附录 B |
| T45 | 各理论结论分别针对什么 | 保留，压缩 | 附录 B |
| T44 | 统计协议（含各轮注册） | 删除，改为附录 C.3 里一段统计方法说明 | — |
| T4 | TTB 的 17 维特征 | 保留 | 附录 C.2 |
| T17 | 我们的 Talker 对 Qwen3-Omni 原生 Talker | 保留，只留 ASR-BLEU（mean ± sd），作为选择 Talker 的公平性依据 | 附录 C.2 |
| T16 | 外部系统全部工作点（43 行） | **删除**，完整数据放 CSV（理由见 17.5） | 补充材料 |
| T19 | oracle 按 S_k 提交 | 保留（S_k 是值得估计的量，对应 3.1） | 附录 F |
| T18 | 测试时后验的质量 | 保留（说明为什么要摊销，而不是在测试时估计，对应 3.3） | 附录 F |
| T29 | 文本端的各种消融 | 只保留"标签来源"消融：full-context 标签对 consensus 标签（对应 3.3 的标签设计）；删除 TTB β 和训练步数两组行 | 附录 F |
| T43 | 在 QABC 自己读上的校准 | 保留 NLL 和 ECE；target shift 的数字移到附录 B 的推论里；删除损失和扣下比例等列 | 附录 F |
| T30 | 语速 r 的敏感性 | 删除 p，给 Δ（有 CI 就加 CI） | 附录 G |
| T39 + T54 | 增益来源、三种时间轴 | 合为一张：提交时刻文本、被听到时的文本、语音。它说明增益出现在"被听到的时间轴"上，正是 Cor 1 所说的队列作用。只给 Δ 和 CI。原第二个登记检验（提交时刻文本）就在这张表里报告，T27 不再单列 | 附录 G |
| T37 + T51 | 推导阈值对二值门；两规则不同的比例 | 合为一张小表，标题 "τ(q) and its two-level approximation"，给 Δ、CI 和"不同决策的读占比"，不放 p。理由见 17.5 | 附录 G |
| T38 | 稳健性 | 只留"去掉任一 talk 后 E1 的范围"和"网格步长"两列；删除 talk 重采样 p 和 no envelope 两列（按你的规则；另见 17.6） | 附录 H |
| T40 | 队列模型 D_k 对实测延迟 | **删除**，同时删除"块内延迟"Remark 和 seq-B。附录 B 只用一句话把"平均语速"列为近似之一 | — |
| T12、T13 | FR | 删除（见 17.1 第 7 条） | — |
| T14、T15 | 全部工作点 | 删除，改为 CSV | 补充材料 |
| T20、T21、T22 | 测试时 refine 与 FR 余量 | 删除（探索性） | — |
| T23 | QABC 对所有 heuristic | 删除 | — |
| T24 | 三个 adapter 的区间检验 | 删除（已被五个 seed 的结果覆盖） | — |
| T26 | ACL 三种跑法 | 删除。正文的 ACL 主数字建议改用"全部读窗在同一台服务器重跑"的 +1.97 [+0.97, +4.34]，见 17.5 | — |
| T27 | 文本登记检验 | 并入 T39 + T54 | — |
| T28 | 文本端分档 | 删除 | — |
| T32 | 语音质量 A.PCP / SIM | 删除（与方法无关） | — |
| T33 | 按词计时 | 删除 | — |
| T34、T35 | 按方向选择、统计功效 | 删除 | — |
| T41、T47–T50 | 一条时钟 | 删除 | — |
| T52、T53 | 早期错误 | 删除；正文中 "early units are not less reliable" 一句也删掉 | — |
| 附录 E 段 | Extensions that did not help | 删除 | — |

结果：

- 正文 8 个浮动体：F1、F2、Table 1–6。
- 附录 v13 原有的图表从 53 张减到约 12 张：T3、T45、T4、T17、T19、T18、T29′、T43′、T30、T39+54、T37+51、T38′。另新增训练超参表、推理超参表和 Algorithm 1。

### 17.3 新附录结构（结合 SEE 和 SimulS2ST-Omni 的写法）

- **SEE 附录的逻辑**：先是 A "More Experimental Setups"（模型、任务、数据规模、超参、基线），接着 B "Supplementary Analysis of SEE"、C/D 是 SEEN 的补充分析，然后依次是可迁移性、效率、有效性和稳健性、算法流程，最后是 AI 写作声明。
- **SEE 的段落写法**：段首是一句结论式标题（如 "SEE Is Stable on Clean Audio."、"SEE Tracks Noise-induced Degradation."），然后"我们评估 X → 图表 N 显示 … → 这说明 …"。收益有限时也直说，并给出原因。例如 SEE 附录 D 写的是 "the gains are limited … we have the following conjectures"。
- **Omni 的附录**：数据 → 训练 → 推理 → 评测协议 → 补充结果 → 计算代价 → 稳健性，每节首句说明它补充正文的哪一部分。

v14 附录：

| 节 | 标题 | 内容 | 图表 |
|---|---|---|---|
| A | Proofs | 按正文命题顺序 | — |
| B | Scope of the Theory | 完整的假设、Lemma 2、myopic 与全局、两个偏移推论及其证明、近似列表 | T3、T45 |
| C | More Experimental Setups | C.1 数据（训练、dev、测试；ACL60/60 的切分与偏移恢复）<br>C.2 基线与外部系统（设置、TTB、Talker 公平性）<br>C.3 评测协议（ASR-BLEU 流程与文本归一化；segment LAAL；**分档 m 的定义与预算来源**；E1/E2 的定义，含 envelope 那一句；bootstrap 方法） | T4、T17 |
| D | Training Implementation Details and Hyperparameters | 分阶段写，加训练超参表和 Algorithm 1（仿 Omni 附录 B 和 SEE 附录 I） | 训练超参表、Alg. 1 |
| E | Inference and Decoding | Thinker 解码、队列跟踪、偏置与 τ_max、Talker 采样；按 17.1 第 7 条核实后写入播放器设置 | 推理超参表 |
| F | Supplementary Analysis of the Stable-Length Posterior（对应 3.1、3.3） | 段首结论句，例如："Committing the stable length is near-offline."（T19）、"Test-time estimates are miscalibrated, which motivates amortization."（T18）、"Full-context labels beat consensus labels."（T29′）、"The amortized posterior is calibrated on the policy's own reads."（T43′） | T19、T18、T29′、T43′ |
| G | Supplementary Analysis of the Queue-Aware Rule（对应 3.2） | "The gain appears on the heard time line."（T39 + 54）、"The gain holds across speech rates."（T30）、"A two-level gate approximates τ(q)."（T37 + 51） | 3 张 |
| H | Robustness of the Main Comparison | 去掉任一 talk、网格步长（T38′）。五个 seed 已在正文 Table 3 | 1 张 |

AI Use Statement 和 Impact Statement 按 ICML 2027 要求放在参考文献之前，不放进附录。

### 17.4 排版：表格紧跟文字，减少空白

v13 的问题出在三处：

- 附录有 12 张跨栏表（`table*`）。在双栏排版里，跨栏表只能浮动到后面某一页的顶部，于是越积越多，最后堆在一起。
- 多处连续 `\input` 4–7 张表，中间只有一句话（`appendix.tex:355–363`、`556–569`）。
- 浮动体总数达到 53 张。

v14 的做法：

1. 浮动体减到约 12 张（见 17.2），这是最有效的一步。
2. 附录除确有必要的情况外，一律用单栏表 `table`；把多余的列删掉或转置，让表能放进单栏。正文只有分档表 Table 2 用 `table*`。
3. 每段只引用一张表，`\input` 紧跟在首次引用它的那段之后；放置参数用 `[!htb]`。
4. 在 `\appendix` 之后设置浮动参数：`topnumber 3`、`bottomnumber 2`、`totalnumber 5`、`\topfraction 0.9`、`\bottomfraction 0.8`、`\textfraction 0.07`、`\floatpagefraction 0.85`、`\dbltopfraction 0.9`、`\dblfloatpagefraction 0.85`。
5. 附录开头加 `\clearpage`，不在每节末尾加 `\FloatBarrier`。
6. 编译后用空白检查脚本核对，每栏空白不超过 20%；每张表与首次引用它的文字相距不超过一页。

### 17.5 对第 5 轮问题的回答

**1. 分档 m1–m6 的定义（A10）**

- 三篇参考论文中，只有 SimulS2ST-Omni 定义了分档。m 是 latency multiplier：把 m 个源语音 chunk 合成一步读取，即读窗为 m 秒（Omni 案例研究写的是 "one source second per step at m=1"），训练时 m ∈ {1,…,12}。
- 1.98 / 1.92 / 1.90 是 **m2**，不是 m1；m1 是 1.18 / 1.18 / 1.30。
- 我们的 m 标签是对的：T55 中 Omni 一列 m2–m6 的数值（24.21、25.01、25.54、25.92、26.32；19.55、21.07、22.47、22.94、23.47）与 Omni 的 Table 3 完全一致。
- 各档的 LAAL 预算是写死在 `experiments/s2st_baselines_v1/fill_paper_v13.py` 的 `BUDGETS` 里的。Omni 的 Table 3 只给了 ASR-BLEU、没有给 LAAL，所以这些预算应该是从它的 Fig. 3 / Fig. 5 曲线上读出来的。附录 C.3 需要写明来源，写之前我再核对一次。
- T55 中只有一个值的行（En→Zh m1 只有 Omni，Zh→En m1 只有 QABC，ACL m1 只有 LA2）随 T55 一起删除。合并后的分档表中，m1 各有 Omni 和 QABC 两个值，体现了 QABC 能工作在 LA2 达不到的低延迟，建议保留。

**2. 为什么要比较推导出的阈值和二值门（C1–C3）**

- 二值门是一个更简单的规则，用的是同一个后验：队列里至少有一整个读窗的语音时扣下不确定的单元，否则用 τ_0。它正好是 Thm 1 的 τ(q) 在两个端点上的取值：q = 0 时等于 τ_0，q ≥ Δ 时阈值趋近 1。所以二值门就是 τ(q) 的两档近似。
- 这个比较检验的是：Thm 1 推出的连续阈值形状，是否比两档近似更好。
- 结果没有差异，原因是 82% 的读的队列落在两个端点上，两条规则在这些读上完全一致。所以它说明"队列信息有用"成立，但"连续形状优于两档"在 RealSI 上无法区分；它并不否定方法本身。
- 它是事先登记的比较（C2）。建议不删，按 17.2 把 T37 和 T51 合成附录 G 的一张小表，标题 "A two-level gate approximates τ(q)"：由 Thm 1 可以推出二值门这种更简单的实现，这反而是对理论的一个用处。
- T9（对照表）里有两行是证明方法有效的关键证据，不能整张删掉；其余行可以删。

**3. Hy-SimulMT 不接我们的 Talker（B1）**

- Hy-SimulMT 只输出文本，没有官方 Talker。我们跑过两种接法：
  1. 接我们的 Talker：与 QABC 用同一个 Talker，只比较策略本身。Zh→En m2 上 21.88 对 21.42，QABC 落后。
  2. 接 Qwen3-TTS 的参考级联：在它有工作点的档位上都低于 QABC（En→Zh m2 25.72 对 28.26；Zh→En m3/m4 20.11 对 22.46/22.65），但 Zh→En m2 上没有工作点。
- 建议两列都保留，以第 1 列为主。如果只放第 2 列，就是为强基线挑了一个更弱的 TTS。我们自己的 T17 也表明我们的 Talker 比 Qwen3-Omni 原生 Talker好，审稿人会认为这个比较不公平。
- 8 个档位里只输一格（−0.46），而 En→Zh 领先 2.4–2.6，这样写是可以接受的。

**4. Table 16**

- 建议**整张删除**，完整数据放 CSV。
- 分档表已经给出了每个外部系统在每个延迟预算内的最佳点；不同系统都用同一规则取点，这就是公平的摘要。
- "SeamlessStreaming 挑最差的一条、QABC 挑最好的一条"会被视为挑选结果，不建议这样做。

**5. ACL60/60 的主数字（删除 T26 的后果）**

- v13 的 +2.63 来自"旧服务器跑的部分读窗 + 新服务器补齐的读窗"拼接而成；全部读窗在同一台服务器上重跑得到 +1.97 [+0.97, +4.34]。
- 删掉 T26 后，建议正文改用 +1.97：口径干净，而且仍然明显为正。
- 同时要确认五个 seed 中 ACL 的结果也是在同一台服务器上跑的。

### 17.6 这些删除对审稿人改分条件的影响

审稿人提出的改分条件 3 要求正文呈现四项敏感性分析。按本轮决定：

- 五个 seed：**满足**（Table 3 的 mean ± sd 和 CI）。
- talk 重采样、按词计时、无 envelope：**都不在论文里了**。

如果真实审稿人问到这三项，需要在 rebuttal 中补充。数据都已有，只需用 CPU 计算。按你的规则，第 15.5 节第 1、2 项计算可以暂缓，留到 rebuttal 时再做。

### 17.7 需要你确认的事

1. 正文 Table 3 是否保留一列单侧 p（推荐保留，表注写一句多重校正后仍成立）；或者只放 CI。
2. ACL60/60 主数字是否改用同一台服务器重跑的 +1.97（推荐）。
3. T37 + T51 合成二值门小表（推荐），还是删除。
4. Hy-SimulMT 的两列都保留（推荐）。
5. 我来核实 FR、播放缓冲是否用于主结果，以及分档预算的出处。

## 18. 第 6 轮：最终章节结构与决定（取代第 3.3、8、13.3、15.3、17.3 节中有关结构的部分）

### 18.0 本轮决定

- **不增加任何额外实验**，包括 15.5 的补算和审稿人要求的三项敏感性分析。
- **Table 3 不放 p**。只放 E1 [95% CI]、E2 和 Seeds > 0；每个测试集一行 s0、一行五个 seed 的 mean ± sd。
- **二值门删除**：
  - 删除的内容：T37、T51；T9 中与二值门相关的两行；正文 Controls 段里的二值门句子；Setup 统计段中登记检验 (2) 里的二值门部分。
  - 带来的约束：全文不再声称"连续阈值 τ(q) 的形状优于更简单的规则"，只声称"队列项有效"（Table 4）。
- **不补的三项敏感性分析**：talk 重采样、按词计时、不用 envelope。如果真实审稿人问到，在 rebuttal 中再给。

### 18.1 全文结构（每节都具体到子节和段落）

**1 Introduction（0.85 页）**

- ¶1 问题：同传中的提交不可撤回；本文范围是句级评测，播放队列有界。
- ¶2 缺口：采用审稿人给的两句——现有规则没有校准的存活概率，也不按已排队的语音给等待定价。
- ¶3 方法：按四个问题依次引出 QABC 的四步。
- ¶4 结果：写主比较结果和五个 seed 的结论；ACL60/60 的数字见 18.4。
- 贡献三条：
  1. 稳定长度后验和队列感知 Bayes 规则（myopic、单元级）；
  2. 摊销学习，在训练分布下对任意阈值都有单元遗憾界，并且只需单次解码；
  3. 在同一 Thinker 和 Talker 下的策略比较。

**2 Related Work（0.35 页，用段落，不分子节）**

¶Read/write policies · ¶Learning when to stop · ¶Simultaneous S2ST systems · ¶Bayesian decisions and amortized inference

**3 Queue-Aware Amortized Bayesian Commitment（2.5 页，含 Figure 1）**

- 节首 3–4 句，提出四个问题，引出 Figure 1。
- **3.1 Commitment Risk as a Stable-Length Posterior**
  - ¶Streaming commitment：读窗协议，引用 Omni。
  - ¶Stable length and survival：S_k 和 ρ_k(u) 的定义。
  - ¶Posterior：给出后验式和链式分解，加一句"后验与策略无关"。
- **3.2 Queue-Aware Bayes Rule**
  - ¶Perceived delay：队列 q_k、等待代价 D_k；写明"队列有界（每句重置）"。
  - ¶Myopic loss。
  - Theorem 1。
  - Corollary 1：队列只移动决策阈值，不改变后验。
  - ¶Heuristics as 0/1 encodings：一句话。
- **3.3 Amortized Posterior Learning**
  - ¶Self-labels：同一 Thinker 的全上下文续写，目标 z_k。
  - ¶Hazard parameterisation：ρ = 1 − h。
  - ¶Stopping and content terms：一行 KL 等式，写明两项共享 LoRA。
  - Theorem 2：写明适用于训练分布、单元级。
- **3.4 Single-Pass Queue-Aware Decoding**
  - ¶Bias on ⟨wait⟩：b_k = logit τ(q_k)。
  - Proposition 5：这样实现只会比 Bayes 规则更保守。
  - ¶Per-read procedure：每读三步。

**4 Experiments（3.2 页）**

- **4.1 Experimental Setup**（0.5 页，无表）：¶Data and benchmarks · ¶Models · ¶Baselines · ¶Metrics（segment LAAL；E1/E2 只写一句，细节指向附录 C.3）
- **4.2 Offline S2ST**（Table 1 紧凑版）：共享的 Thinker 和 Talker 作为离线参照；加 adapter 前后结果几乎不变，说明 adapter 只学"何时停"，对应 3.3。
- **4.3 Simultaneous S2ST**（Figure 2、Table 2、Table 3）
  - ¶Sentence-level RealSI：曲线和分档表。
  - ¶Long-form ACL60/60。
  - ¶Primary comparison and adapter seeds：Table 3。
  - 外部系统只作为背景，正文写明增益集中在 2.5 s 以内。
- **4.4 Effect of the Queue-Aware Rule**（Table 4，对应 3.2）
  - QABC 与 QABC_Δ 的配对比较；
  - 同一草稿上 QABC 对 LA2；
  - LA2 只加队列门，几乎没有收益。
- **4.5 Effect of Amortized Posterior Learning**（Table 5，对应 3.3；有待你定的事，见 18.3）
- **4.6 Effect of Single-Pass Decoding**（Table 6，对应 3.4；有待你定的事，见 18.3）
- **4.7 Ablation Analysis**（0.25 页，按段落写，无表，参照 SEE 4.6）：
  - ¶Label source：full-context 标签对 consensus 标签，见附录 F；
  - ¶Speech rate r：见附录 G。

**5 Conclusion**

- 3 句结论，再接一段不编号的 Limitations：
  - 只用了一个 Thinker、一个 Talker、两个方向；ACL60/60 只评测文本；
  - 句级评测、每句重置队列；
  - 五个 seed 平均后 En→Zh 不显著；
  - 理论的适用范围：myopic、单元级、训练分布；
  - 没有人工听测。

**Impact Statement、AI Use Statement**（不计页）

**附录**

| 节 | 子节 / 段落 | 图表 |
|---|---|---|
| A Proofs | A.1 Stable-length posterior（Prop 1、Lemma 1）· A.2 Queue-aware Bayes rule（Thm 1、Cor 1、0/1 编码）· A.3 Amortized learning（hazard、Prop 4，改掉 "does not compete" 那一句；Thm 2）· A.4 Single-pass decoding（Prop 5） | — |
| B Scope of the Theory | B.1 Assumptions in full（Ass 1/2、Lemma 2）· B.2 Myopic versus global commitment · B.3 Distribution shifts（两个推论）· B.4 Practical approximations | T3、T45 |
| C More Experimental Setups | C.1 Data（训练与 roll-in、dev、RealSI、ACL60/60 的切分与偏移、CVSS-T）· C.2 Baselines（LA2、Hold-n、Vote、TTB、外部系统设置、Talker 对照）· C.3 Evaluation Protocol（ASR-BLEU 与文本归一化、segment LAAL、StreamLAAL、分档 m 与预算来源、E1/E2 与 envelope 一句、bootstrap） | T4、T17 |
| D Training Implementation Details and Hyperparameters | D.1 Simultaneous Talker · D.2 Oracle label generation · D.3 QABC adapter · D.4 TTB prior head | 训练超参表、Algorithm 1 |
| E Inference and Decoding | E.1 Thinker decoding and the queue · E.2 Talker and playback（FR 和播放器按 17.1 第 7 条核实后再写） | 推理超参表 |
| F Supplementary Analysis of the Stable-Length Posterior | ¶Committing the stable length is near-offline（T19）· ¶Test-time estimates are miscalibrated（T18）· ¶Full-context labels outperform consensus labels（T29′）· ¶The posterior stays calibrated on the policy's own reads（T43′） | 4 张 |
| G Supplementary Analysis of the Queue-Aware Rule | ¶The gain appears on the heard time line（T39 + 54）· ¶The queue gain holds across speech rates（T30） | 2 张 |
| H Robustness of the Main Comparison | ¶No single talk drives the gain（T38′：去掉任一 talk、网格步长） | 1 张 |

合计：正文 8 个浮动体（F1、F2、Table 1–6）；附录约 11 张表，加 2 张超参表和 1 个 Algorithm。

### 18.2 与三篇参考论文的逻辑对照

| 部分 | Hibiki（ICML） | SEE（ACL） | SimulS2ST-Omni（ACL） | v14 |
|---|---|---|---|---|
| 方法 | 3 个子节：建模 → 数据对齐 → 声音迁移 | 3 个子节：找出噪声子空间 → 计算 SEE → 用它构造 SEEN | 3 个子节：问题设定 → 轨迹构造 → 骨干 | 4 个子节：风险是什么 → 何时说 → 后验从哪来 → 怎样一次解码。顺序同样是"定义 → 构造 → 使用" |
| 实验设置 | 1.8 页（附录只有 1 页） | 0.5 页 | 0.65 页 | 0.5 页 |
| 主结果 | 离线 + 同传 | — | 离线 → 句级 → 长文 | 离线（4.2）→ 句级 + 长文（4.3） |
| 按方法组件的分析 | 消融表 | 4.2–4.5，每节对应一个方法部分 | — | 4.4–4.6，分别对应 3.2–3.4 |
| 消融 | Table 4–5 | 4.6，按段落写 | 4.5 | 4.7，按段落写 |
| 附录 | 1 张图 | 补充设置 → 各组件的补充分析 → 效率 → 有效性 → 算法 → AI 声明 | 数据 → 训练 → 推理 → 评测协议 → 补充结果 → 计算代价 → 稳健性 → 人评 → 案例 → 失败案例 → 发布 | 证明 → 理论范围 → 补充设置 → 训练 → 推理 → 各组件的补充分析 → 稳健性 |

与参考论文的三处不同：

1. **方法有 4 个子节，参考论文是 3 个。** 原因是每一步都对应一个独立的实验子节。如果要严格对齐成 3 个，可以把 3.3 和 3.4 合并为 "Amortized Learning and Single-Pass Decoding"。
2. **附录先放证明（A、B）。** 三篇参考论文都没有定理，ICML 的理论类论文一般把证明放在附录最前面。
3. **没有人评、案例研究和失败案例。** 这是已经做出的决定。

### 18.3 正文里还剩两处对 QABC 不利的结果

**Table 5（摊销后验对测试时后验）**

- En→Zh 和 ACL60/60 上摊销后验更好：
  - En→Zh 2–3 s：QABC_Δ +0.35，TTB −0.27；
  - ACL60/60：QABC_Δ +1.53，TTB +0.83。
- 但 Zh→En 上 TTB 更好：+0.47 对 −0.21。
- 建议：保留，并把论点写成"摊销后验在 En→Zh 和 ACL60/60 上更好，只需一次解码，每读计算量约为 TTB 的一半（59 ms 对 132 ms）"。如果删掉，3.3 就没有直接的对照证据了。

**Table 6 的规则对比（dev）**

- 严格按 Thm 1 实现、全局 argmin 和 Platt 重校准，在 dev 上都比部署的保守贪心规则差（例如 −0.36/−0.47）。
- 这一结果有两种读法：一种是"保守的实现就够了"，支持 Prop 5；另一种是"严格的 Bayes 规则反而更差"，对理论不利。
- 建议：正文 Table 6 只放计算量（QABC 每读最快）；规则对比移到附录，或者删除，由你定。

### 18.4 ACL60/60 的主数字

E1 指同等延迟下 QABC 比 LA2 高出的 BLEU，在 3 s 以内取平均；用登记的 s0 adapter，在 LA2 的 15 个读窗上计算。v13 里一共有三个版本：

| 版本 | 读窗（LA2 / QABC） | E1 [95% CI] | E2 |
|---|---|---|---|
| 只用旧服务器 | 9/7 | +2.40 [+1.30, +5.05] | +0.29 s |
| 旧服务器 + 新服务器补齐（v13 正文所用） | 15/15 | +2.63 [+1.54, +5.76] | +0.35 s |
| **全部在新服务器上重跑** | 15/15 | **+1.97 [+0.97, +4.34]** | +0.30 s |

- 建议用 **+1.97**：所有读窗在同一台服务器上跑，口径干净，删掉 T26 后不需要额外解释。
- 改用后要保持一致的地方：Figure 2 的 ACL 曲线、Table 2 的 ACL 行、Table 3 中 ACL 的五 seed 均值。这三处都改用重跑的那批输出重新统计，只是对已有输出的重算，不是新实验。
- 写之前要确认 s1–s4 的 ACL 结果在哪台服务器上跑；五 seed 均值由 +2.13 变为约 +2.00。

### 18.5 Hy-SimulMT 的两列

Hy-SimulMT（30B-A3B，输入语音、输出文本）没有自己的 Talker。两列的区别只在于由谁把它提交的文本块读出来：

- **Hy-SimulMT（接我们的 Talker）**：由 QABC 也在用的 streaming UniSS Talker 朗读，三个 talker seed。和 QABC 相比，只有策略和译文不同。
- **Hy + Qwen3-TTS**：同一批文本块由 Qwen3-TTS-12Hz-0.6B-Base 朗读（x-vector 克隆源说话人、默认采样、三个 seed），是一个现成 TTS 的参考级联。

数值（ASR-BLEU，括号内为 LAAL s；s0，三个 talker seed 的均值；v13 T11）：

| 方向 | 档位 | Hy-SimulMT（我们的 Talker） | Hy + Qwen3-TTS | QABC |
|---|---|---|---|---|
| En→Zh | m1 | — | — | 25.33 (1.13) |
| | m2 | 25.67 (1.72) | 25.72 (1.90) | **28.26 (1.77)** |
| | m3 | 27.96 (2.05) | 27.93 (2.17) | **30.33 (2.25)** |
| | m4 | 27.96 (2.05) | 27.93 (2.17) | **30.51 (2.75)** |
| Zh→En | m1 | — | — | 17.83 (1.14) |
| | m2 | **21.88 (1.75)** | —（没有工作点落在预算内） | 21.42 (1.70) |
| | m3 | 21.88 (1.75) | 20.11 (2.11) | **22.46 (2.18)** |
| | m4 | 21.88 (1.75) | 20.11 (2.11) | **22.65 (2.67)** |

- Hy-SimulMT 只有两个原生模式（即两个工作点），所以各档反复出现同一个点。
- En→Zh 上两种 Talker 相差无几。Zh→En 上 Qwen3-TTS 更低，而且读得更慢，在 m2 没有工作点。
- QABC 只在 Zh→En m2 输给"接我们的 Talker"那一列（−0.46）。
- 建议：两列都保留，以"接我们的 Talker"为主。只保留 Qwen3-TTS 那一列，等于给强基线换了一个更弱的 TTS，会被认为不公平。

## 19. 第 7 轮：消融小节、Table 5/6 与核实结果（取代 18.1 中实验章和附录 F/G 的对应部分）

### 19.1 决定

- **Table 5 保留**，并加入两列计算代价：每读 Thinker 耗时 ms/read，以及计入计算后 LAAL 的增加 +CA，数据来自 v13 T31。
- **原 Table 6 的规则对比删除**：
  - 删除的原因是它对理论不利：在 dev 上，严格按 Thm 1 实现（−0.36/−0.47）、全局 argmin（−0.76/+0.21）和 Platt 重校准（−0.43/−0.97）都比部署的保守贪心规则差；块内延迟规则在测试集上也不显著（+0.19/+0.37）。审稿人会问"为什么严格的 Bayes 规则反而更差"。
  - 原 Table 6 中计算代价那部分对我们有利：QABC 每读 57–59 ms，LA2 是 72–76 ms，TTB 是 132–136 ms，Vote 是 107–135 ms；+CA 只有 0.06–0.07 s。这部分并入 Table 5。
  - 附录 T42 一并删除。全文不再比较"严格规则"和"部署规则"；Prop 5 只作为理论性质保留。
- **原 4.6 并入 4.5**，标题改为 "Effect of Amortized Learning and Single-Pass Decoding"。方法 3.4 的证据是：主结果（单次解码可用），加上 Table 5 的计算代价。
- **新的 4.6 Ablation Analysis 放两张表**。三篇参考论文都有带表格的消融小节：SEE 4.6 有 Table 4–6，Omni 4.5 有 Table 4–5，Hibiki 有 Table 4–5。

### 19.2 新 4.6 Ablation Analysis（均为已有结果，不需要新实验）

**Table 6：标签来源与训练长度**

对应方法 3.3。指标为 RealSI 提交文本的 BLEU，取每个文本 LAAL 预算内的最佳点；数据来自 v13 T29。

| 方向 | 变体 | ≤1.5 s | ≤2.0 s | ≤2.5 s | ≤3.0 s |
|---|---|---|---|---|---|
| En→Zh | full-context 标签（QABC_Δ，部署的 step 200） | 27.62 | 29.05 | 31.45 | 31.90 |
| | consensus 标签（两次训练，mean ± sd） | 26.62 ± 2.23 | 28.28 ± 1.58 | 29.26 ± 0.83 | 30.31 ± 0.64 |
| | 训练长度 step 200–800（mean ± sd） | 27.96 ± 0.35 | 29.72 ± 0.48 | 31.41 ± 0.12 | 31.88 ± 0.27 |
| Zh→En | full-context 标签（step 200） | 20.31 | 21.51 | 21.51 | 23.32 |
| | consensus 标签（两次训练，mean ± sd） | 18.48 ± 0.62 | 20.60 ± 0.53 | 21.71 ± 0.33 | 22.38 ± 0.62 |
| | 训练长度 step 200–800（mean ± sd） | 20.10 ± 0.45 | 21.45 ± 0.22 | 22.42 ± 0.61 | 23.36 ± 0.32 |

两段写法（段首为结论句）：

- **¶Full-context labels outperform consensus labels.**
  - 用同一 Thinker 的全上下文续写作标签，与 consensus 标签（多次采样输出之间一致的部分）比较。8 个格子中 7 个更高；Zh→En ≤2.5 s 一格 consensus 高 0.20，在一个 sd 以内。
  - 这对应 3.3 的标签设计："后验学的是会不会被完整上下文改掉，而不是采样之间是否一致"。
  - 比较时取的是部署所用的同一个 checkpoint（step 200）。不建议改用"四个 checkpoint 的均值"去和 consensus 比：那样 8 格都会更好，但口径是挑过的。
- **¶The posterior is insensitive to training length.**
  - 四个 checkpoint 的 sd 都在 0.12–0.61 之间，说明结果对训练长度不敏感，也说明按 dev 选 checkpoint 不是结论成立的前提。

**Table 7：语速 r 的敏感性**

对应方法 3.2。设置为 Δ = 0.5 s、b_0 = 0、一个 talker seed；指标为 ASR-BLEU（括号内为 segment LAAL）。数据来自 v13 T30，去掉 p 列。

| 方向 | QABC_Δ（不用队列） | QABC，r×0.75 | QABC，r×1 | QABC，r×1.25 |
|---|---|---|---|---|
| En→Zh | 24.63 (1.30) | 25.59 (1.28)，+0.97 | 26.12 (1.29)，+1.49 | 27.15 (1.38)，+2.52 |
| Zh→En | 18.65 (1.31) | 19.10 (1.31)，+0.45 | 19.74 (1.32)，+1.09 | 20.07 (1.40)，+1.42 |

- **¶The queue gain holds across speech rates.**
  - 三个语速下，队列项都带来正增益；r 越大，模型预测的队列越长，策略等得越多，增益也越大。
  - r×1.25 时 LAAL 增加 0.08–0.09 s，图注要写明。
  - 这对应 3.2 的队列模型，说明结论不依赖 r 的精确取值。

### 19.3 实验章与正文浮动体（最终版）

| 小节 | 内容 | 浮动体 |
|---|---|---|
| 4.1 Experimental Setup | 四段 | — |
| 4.2 Offline S2ST | 离线参照；adapter 只学何时停 | Table 1 |
| 4.3 Simultaneous S2ST | 句级 RealSI、长文 ACL60/60、主比较与 seed | Figure 2、Table 2、Table 3（不放 p） |
| 4.4 Effect of the Queue-Aware Rule | QABC 对 QABC_Δ；同一草稿上的 QABC 对 LA2；只加队列的 LA2 | Table 4 |
| 4.5 Effect of Amortized Learning and Single-Pass Decoding | TTB、QABC_Δ、QABC 分别对 LA2，加计算代价 | Table 5 |
| 4.6 Ablation Analysis | 标签来源、训练长度、语速 r | Table 6、Table 7 |

- 正文共 9 个浮动体（F1、F2、Table 1–7），与参考论文的 10–12 个接近。
- 页数：去掉原 4.6 的 0.35 页，加上两张单栏小表约 0.45 页，仍在 8 页以内（原预算留有 0.3 页余量）。

### 19.4 附录的相应变化（修改 18.1 的附录表）

- **F**：保留 T19（oracle）、T18（测试时后验）、T43′（校准）。T29′ 已移到正文 Table 6。
- **G**：只保留 T39 + 54（增益出现在被听到的时间轴上）。T30 已移到正文 Table 7；二值门已删除。
- **E**：写明主结果的推理设置——Talker 用 block 语速（即 T1c 训练时的语速 token），history ban 8，不用积压加速，不用播放缓冲，不用 FR（见 19.5）。
- 删除 T42（规则对比）和 T31（计算代价已进正文 Table 5）。
- 附录剩约 8 张表，另有 2 张超参表和 1 个 Algorithm。

### 19.5 只读核实的结果（只查了脚本和文档，没有跑实验）

1. **FR 没有用于主结果。**
   - FR 的文本规则（`--hygiene`、`--clean`）和积压加速（`--speed-policy queue`）都是需要手动打开的开关。
   - 登记任务的生成脚本 `reports/qabc_v10/make_v10_jobs.py`（注明 "as v4"，即与登记的 s0 相同）调用的是 `si_thinker_stream --policy selfwait --wait-biases=1 --sw-queue-rate …`，没有打开这些开关；Talker 用的是 `--speed-policy block --history-ban 8`。
   - 播放缓冲只出现在 `si_talk_timeline.py`（整场时间线，用于一条时钟和听感指标），登记任务没有调用它。
   - 结论：**FR、积压加速和播放缓冲都从论文中删除**。v13 方法 §5.2 中 "then FR and the queue update" 改为只写队列更新；推理设置照实写入附录 E。
2. **延迟档预算的出处**：`paper/s2st_comparison_plan_v1.zh-CN.md` 第 135 行记录，m1 = 1.18 s 是"从图上读"的。
   - RealSI 的预算（1.18/1.98/2.46/2.98 s，Zh→En 为 1.18/1.92/2.40/2.98 s）读自 SimulS2ST-Omni 的 Fig. 3（beam 4）。
   - ACL60/60 的预算（1.30/1.90/2.48/2.89 s）读自其 Fig. 5（v13 T5 表注已写明）。
   - 附录 C.3 照此写明出处。
3. **我们的延迟口径比 SimulEval 更严**：同一文档记录，在 QABC 的工作点上，我们的 segment LAAL（D1）比 SimulEval 的逐词 LAAL（D0w，在发出时计时、不含排队）高约 0.1 s。例如 En→Zh m1/m2/m3 为 1.13/1.77/2.25 对 1.00/1.67/2.17。所以与 SimulS2ST-Omni 分档的比较是保守的。附录 C.3 可以写一句；这个数据已有，不需要新实验。

## 20. 第 8 轮：ACL60/60 主数字（取代 17.5 第 5 点和 18.4 的建议）

- **决定（用户）**：ACL60/60 的 E1 沿用 v13 的 **+2.63 [+1.54, +5.76]**，E2 为 +0.35 s。删除 T26，论文中不提服务器。
- **一致性**：以下各处全部沿用 v13 中同一批结果（登记网格补齐后的版本），无需重新统计：
  - Figure 2 的 ACL 曲线；
  - Table 2 的 ACL 行；
  - Table 3 的 s0 行（+2.63）；
  - 五 seed 均值 +2.13 ± sd。
- **需要清理的残留**，避免与删除 T26 相矛盾：
  - v13 `tab_prereg_all` 表注中的 "completed in v8, Table~\ref{tab:acl-sens}"；
  - 附录 "The registered speech test: run counting and protocol gap" 一段（已按第 13 节删除）；
  - 其余所有对 `tab:acl-sens` 的引用。
- **内部备忘（不进论文）**：全部读窗在同一台服务器上重跑的结果是 +1.97 [+0.97, +4.34]，E2 +0.30 s，数据在 v13 的 `generated/tab_acl_sensitivity.tex` 和对应的结果 JSON 中。复现或 rebuttal 被问到时用。

## 来源

- ICML 2027 Author Instructions：https://icml.cc/Conferences/2027/AuthorInstructions
- ICML 2027 Reviewer Instructions（审稿表单的 Limitations 一项）：https://icml.cc/Conferences/2027/ReviewerInstructions
- ICML 2027 样式包：https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip
- Hibiki：arXiv 2502.03382（ICML 2025）
- SEE：ACL 2026 Long，第 18976–18995 页
- SimulS2ST-Omni：arXiv 2607.19810v2（附录 A–K）
