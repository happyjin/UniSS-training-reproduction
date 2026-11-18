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
| N6 | 推导出的连续阈值对二值队列门：+0.35/−0.26/+0.24，Holm p ≥ 0.06；两者只在 3.8% 的读上不同 | 2.4 P1、Q3 | **正文** §7.3 Controls 段；附录 Table 37、51 | 不建议删：这是事先登记的比较（C2） | 主张拆成两层：(i) 队列有用，证据强（Table 4）；(ii) 精确的 τ(q) 是否优于二值门，本数据无法区分，因为 96% 的读队列要么为空、要么满一个读窗，两条规则在这些读上一致。正文写一句，详表放附录 H.3。可选做分桶分析（见 15.5） |
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

## 来源

- ICML 2027 Author Instructions：https://icml.cc/Conferences/2027/AuthorInstructions
- ICML 2027 Reviewer Instructions（审稿表单的 Limitations 一项）：https://icml.cc/Conferences/2027/ReviewerInstructions
- ICML 2027 样式包：https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip
- Hibiki：arXiv 2502.03382（ICML 2025）
- SEE：ACL 2026 Long，第 18976–18995 页
- SimulS2ST-Omni：arXiv 2607.19810v2（附录 A–K）
