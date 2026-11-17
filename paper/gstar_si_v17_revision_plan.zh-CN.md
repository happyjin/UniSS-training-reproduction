# MIST 论文 v17 修改方案：回应 v16 模拟审稿意见（待确认）

2026-11-17（布拉格时间）。依据：用户提供的 `MIST_v16_ACL_Review_CN.pdf`（模拟审稿，Weak Reject 3/5）和 v16（`paper/gstar_si_v16/`，commit 9f843334d）。
reviewer 引用的数字都已核对，与 v16 Table 5 一致：R1→R2 +1.56 / +0.24 / +0.87，PA Talker +0.29 / +0.02 / +0.33，best-of-two +0.18 / +0.30 / +0.26；EndOffset −0.69 / −0.40 / −0.94 与 −0.18 / −0.15 / −0.23。

---

## 0. 总体判断

| reviewer 意见 | 是否接受 | 改法 | 是否需要新实验 |
|---|---|---|---|
| P0 CRF 是否必要（需要无未来代价、无 CRF、λ=0 三个对照） | **接受，这是最关键的一条** | 见 §1：先报告已有的 dev 对照；建议补做一组严格对照 | 可选，约 1 天 GPU |
| P0 时延不含计算时间，simultaneous 不等于实时 | 接受 | 正文明确写 “computation-unaware”，并给出实时性的事实 | 否 |
| P1 人类听感测试 | 部分接受 | 你此前决定不做听测；可补一个自动的说话人辨认代理指标（可选） | 可选，几小时 |
| P1 收窄 “first” 声明，与 DiariST / G-STAR / X-Translator 明确区分 | 接受 | Related Work 加一段区分；Intro 的措辞需要你批准 | 否 |
| 指标措辞：说成“改编”；tcpASR-BLEU 的时间约束是对齐**源语音时间**，不是“何时被听到” | 接受 | §3.6 和 Table 1 caption 各加一句 | 否 |
| CRF 的定位：不是新的 CRF 理论，创新在于对播放队列建模，以及用 β 把未来代价带进当前选择 | 接受 | 3.4 加一句定位；突出 w_i ∝ exp φ · β_{i+1} | 否 |
| P2 改标题 | 接受，取决于 §1 的结果 | 见 §6 | 否 |

---

## 1. P0：CRF 是否必要（最关键）

### 1.1 已有证据（我在报告里找到的，此前没有用在论文中）

**(a) BoN-SFT 消融，即“无格”（dev，已预登记）**：`reports/mist_w3po_v1/talker_crf_results_v2.zh-CN.md`，预登记文件 `talker_crf_prereg_v2.zh-CN.md`。
- 唯一改动是候选权重：q(c) ∝ exp(0.5·ℓ̄ + 5·f)，每块独立。**没有延迟项、没有 λ、没有后向消息。**
- 它等价于 **λ = 0 的 CRF**：没有延迟项时，后验按块分解。所以它同时就是 reviewer 要的 “Without posterior regularization (λ=0)” 对照，也是最严格的 “Without future costs” 对照（连当前块的延迟项也去掉了）。
- 训练步数、起点、推理（含 best-of-2、积压语速）都和 PA Talker 相同。
- 结果（全 dev，两个 TTS 种子，Zh→En，A′ 的文本）：
  - **质量持平**：BoN − PA = −0.14 ASR-BLEU，P(Δ≤0) = 0.91，未达到登记的显著线。
  - **每条流 LAAL 持平**：+0.021 s。
  - 登记的结论措辞：**“PL-EM 在每条流设定下的收益可由 best-of-N 蒸馏解释”**。
  - 未登记的补充：BoN 的 **EndOffset 晚 0.10 s，单通道 LAAL 晚 0.44 s，两个种子方向一致**。也就是说，CRF 的作用主要体现在缩短播放、减少排队，而不在质量。

**(b) R3′，语速匹配规则（test，v11 预登记主检验）**：与 MIST 用同样的 best-of-2，并把规则的语速在 dev 上手调到与 MIST 相同。
- MIST − R3′ 的 tcpASR-BLEU 为 +0.11 / +0.05 / +0.23，**只有 AliMeeting 显著**（p_Holm = 0.007）；LAAL 相同。
- 你此前决定论文中不放这一结果，但它正好回答 reviewer 的问题。

**解读**：两项证据都说明，在我们的数据上，**CRF 相对于“只看质量”或“手调语速”的独立贡献主要是适度的时延收益**（dev EndOffset −0.10 s），质量收益只在 OOD 的 AliMeeting 上显著。reviewer 建议的对照很可能复现这一结论。论文的表述需要据此校准，否则会被严格的审稿人抓住。

### 1.2 方案

**方案 A：不做新实验**（约半天，只改写作）
- 在 4.5 Ablation Analysis 增加一段 *Is the CRF necessary?*：报告 BoN（λ=0）的 dev 对照，即质量持平、EndOffset 晚 0.10 s、两个种子一致。
- 结论如实写成：**CRF 的作用是用预算确定工作点、并为未来排队定价，从而带来额外的时延收益；质量收益主要来自积压语速和保真度选择。**
- R3′ 是否一并报告由你决定（§7 D2）。
- 标题采用 §6 的方案 C。

**方案 B（建议）：补一组严格控制变量的对照，结果出来后再决定标题和措辞**
- 在**最终管线**上做：使用 B 的 deadline Thinker 文本、同一批候选、同样的积压语速规则、同样的训练步数和超参、同样的 best-of-2 解码，**只改候选权重**：

| 变体 | 候选权重 | 对应 reviewer 的哪一项 |
|---|---|---|
| **V0 = MIST**（现有） | $w_i(c)\propto e^{\phi^{\lambda^\star}_i(\tilde h_i,c)}\beta_{i+1}(g_i(\tilde h_i,c))$ | — |
| **V1 无延迟项（λ=0）** | $w_i(c)\propto e^{\phi^{0}_i(c)}$，即 BoN | Without posterior regularization |
| **V2 只有局部延迟、没有未来代价** | $w_i(c)\propto e^{\phi^{\lambda^\star}_i(\tilde h_i,c)}$，去掉 β_{i+1}，λ* 不变 | Without future costs；也覆盖 “Without CRF：局部、考虑时长的加权” |

- **流程**（按项目惯例先登记再跑）：
  1. 写预登记：两个比较 V0−V1、V0−V2；主指标 tcpASR-BLEU 和 EndOffset；按会议做 sign-flip 检验，Holm 校正。
  2. 用 CPU 生成两种权重的训练行。
  3. 两个变体 × 两个方向各训练一个 Talker。每次 1,200 步，在 4 张卡上约 25 分钟。
  4. 先在 dev 上过门槛。
  5. 在 3 个 test 集上各解码一次，两个 TTS 种子，然后打分。
- **耗时**：约 1 天。8 张卡目前都由占位训练（`train_mist_workload.py`）占着，按约定可以按 PID 停掉，把卡让给实验。
- **按结果写论文**：
  - 若 V0 在 EndOffset 或质量上**稳定优于 V2**（两个种子同号且检验通过）：论文可以说“未来代价有用”，标题用方案 A，保留 Playback-Queue CRF。
  - 若 **V0 ≈ V2**：说明作用来自局部的时长定价，而不是后向消息。标题用方案 C，把 CRF 定位为“为局部延迟定价的有原则的方式，并且由预算自动确定工作点”。
  - 不论结果如何，都放进 4.5（正文加一行或一小段），完整结果放附录。

---

## 2. P0：时延不含计算时间（仅改写作）

已知事实：附录 C 写明，按实测计算时间回放，AliMeeting 上 Thinker 每秒音频需要 1.20 s 计算，块结束的中位额外延迟为 52.8 s；按目前的实现（batch-1、每次读都重新编码前缀）跟不上实时。

- **4.1 Metrics** 加一句：LAAL 和 EndOffset 都是 **computation-unaware**（对应 SimulEval 中不计计算时间的版本），评价的是同传**策略**本身，不是实时部署；按目前的实现，解码慢于实时（附录 C）。
- **Table 1 caption**：写成 “LAAL (↓, computation-unaware)”。
- **Limitations**：已有一句，保留。
- **标题**：去掉容易被读成“实时”的字样，或明确写任务名（见 §6）。

---

## 3. P1：人类听感测试

- 你在 v11 时（2026-11-15）决定不做听测。我**不建议为了这一条改变决定**：做盲测需要招募评审员，周期超出当前范围。
- **写作上**：正文 4.3 已把“停顿变多”写成代价，Limitations 也写了没有听测。可以在 Table 2 的 caption 或 4.3 里，把 AliMeeting 的停顿从 4.4 增加到 11.8 次/分钟作为数字点明，让代价更清楚。
- **可选的自动代理（几小时，不训练模型）**：用说话人验证模型（如 WavLM-SV），比较每条输出流与**说话人自己在原会议中的录音**的相似度（不再用 voice prompt）。这可以回答“听众能不能辨认是谁在说”，也同时去掉 Limitations 里“SIM 只对比了 voice prompt”这一条。

---

## 4. P1：收窄 “first” 声明，明确与已有工作的区别

- **Related Work**（不在受保护范围内）加 2–3 句：
  - **DiariST**：有说话人归属的**文本**同传，不输出语音；
  - **G-STAR**：有说话人归属的**识别**，不做翻译；
  - **X-Translator**：输出语音，但它的说话人追踪只用来选克隆音色，不为重叠的说话人分别输出（在我们的评测里它也需要借用我们 diarizer 的标签）。
  - 我们的区别：同传过程中按说话人分开输出语音流，包括重叠部分，并且在整场会议上评测。
- **Introduction 贡献 1**（受保护，需要你批准）。建议把 “the first SimulS2ST that produces separate, speaker-attributed spoken translations in each talker's voice, even during overlapping speech” 收窄为：
  > *To our knowledge, the first simultaneous S2ST system that outputs separate speaker-attributed speech streams, including overlapping speech, evaluated on whole meetings.*
- **AliMeeting 的重叠结果**：正文已经如实写了“部分迁移”。全文检查，不能有“重叠时所有文本指标都提高”之类的表述。

---

## 5. 指标措辞与 CRF 定位（仅改写作）

- **§3.6**：把 “we propose” 改为 “we **adapt** cpWER and tcpWER to translated speech as cpASR-BLEU and tcpASR-BLEU”。Intro 里的 “introduce” 是你写的，不改。
- **§3.6 增加一句**：tcpASR-BLEU 的时间约束针对的是**源语音时间**（MIST 用时间 token；没有时间 token 的系统用相邻两次输出之间读入的音频区间，代码 `build_e7.py` / `build_e8.py` 就是这样处理的），**不衡量译文何时被听到**；后者由 LAAL 和 EndOffset 衡量。附录 E 同步写明。
- **Table 1 caption**：“without / with **source-time** constraints”。
- **§3.4 增加一句定位**：*We do not propose a new CRF; the contribution is the model of the playback queue — states (backlog, rendering) with FIFO transitions — under which backward messages price the waiting a rendering imposes on later blocks.* 并在 §3.5 强调 $w_i\propto e^{\phi}\beta_{i+1}$ 这一机制（已有，再点明一次）。
- Abstract 和 Intro 若要突出 β 的机制，需要你自己改或批准（受保护）。

---

## 6. 标题（取决于 §1 的结果）

| 方案 | 标题 | 适用条件 |
|---|---|---|
| A（reviewer 首选） | **MIST: A Playback-Queue CRF for Speaker-Attributed Speech-to-Speech Translation in Meetings** | 方案 B 的实验显示 V0 稳定优于 V2 |
| B | Who Said What, and When It Is Heard: A Playback-Queue CRF for Speaker-Attributed Speech-to-Speech Translation | 同上，保留现有的叙事线，但较长 |
| C（稳妥） | **MIST: Playback-Aware Speaker-Attributed Speech-to-Speech Translation for Multi-Talker Meetings** | 不做新实验，或 V0 ≈ V2 |

- CRF 不需要在标题里写全称。Abstract 已经写了 “conditional random field (CRF)”。
- 关于 “Simultaneous”：reviewer 担心被读成“实时”。我的建议是：标题按 reviewer 的方案用 “Speech-to-Speech Translation”，在 Abstract 和 Setup 里明确是 simultaneous 策略、时延不计计算时间。若你希望标题保留 “Simultaneous”，可以写成 “… Simultaneous Speech-to-Speech Translation …”，配合 §2 的说明也站得住。

---

## 7. 需要你决定

| 编号 | 问题 | 我的建议 |
|---|---|---|
| D1 | CRF 对照：方案 A（只报告已有的 dev BoN 结果）还是方案 B（在最终管线上补做 V1、V2，约 1 天 GPU，需要停占位任务）？ | **B**，同时也报告已有的 dev BoN 结果 |
| D2 | R3′（语速匹配规则，test）：继续不放，还是放进附录 G？ | reviewer 正是在问这个，**建议至少放附录**，正文不夸大 CRF 的作用。最终由你定 |
| D3 | 听测：维持不做？要不要做自动的“说话人辨认”代理指标（几小时）？ | 维持不做听测；**做**代理指标 |
| D4 | Intro 贡献 1 的 “first” 是否按 §4 收窄（受保护文字）？ | 收窄 |
| D5 | 标题：等 D1 的结果再定（A 或 C）；标题是否保留 “Simultaneous”？ | 先按 C 准备，实验支持后再改成 A |

## 8. 执行步骤（确认后执行）

1. 新建 `paper/gstar_si_v17/`，从 v16 复制过去，v16 不做任何修改。
2. 改写作：§2、§4（Related Work）、§5 的各项，以及 4.5 新增的 CRF 必要性段落（先用已有的 dev BoN 结果）。
3. 若选 D1 = B：
   - 写预登记；
   - 按 PID 停掉占位任务；
   - 生成训练行，训练 V1、V2（两个方向）；
   - 先跑 dev 门槛，再在 test 上各解码一次并打分；
   - 结果写进 4.5 和附录 G，并按 §6 确定标题。
   - 所有新输出目录都按运行 ID 命名，已有文件不覆盖。
4. 若选 D3 做代理指标：只读已有的合成语音，计算与原说话人录音的说话人验证相似度，写进 Table 2 或附录。
5. 编译：正文 ≤ 8 页；没有溢出或未定义引用；全文 grep 不到 distil。打包 zip，commit 后 push。
