# 单说话人 QABC 论文 v17 修改方案（回应 v16 首读盲审，基于 bayes_simul_s2st_v16）

写于 2026-11-18（布拉格时间）。审稿报告：QABC_v16_First_Read_Blind_Review_CN.pdf，结论 Weak Reject 3/6（接近边界），各项分数为 Soundness 3、Presentation 3、Originality 3、Significance 2。

**原则（用户要求）**：不补新实验。v17 只改文字、表注和图注，用来收紧主张、讲清协议、修正理论表述。有几项数据其实早已跑过，只是之前按你的决定删掉了，是否恢复见第 3 节，由你决定。摘要和引言的改动只作为建议列出（第 4 节），需要你同意才动。

## 1. 逐条评估

| # | 审稿意见 | 是否合理 | 需要新实验吗 | v17 处理 |
|---|---|---|---|---|
| W1 | 句级评测、每句清空队列，不足以支撑连续播放场景的部署价值 | 合理。队列机制正是本文卖点 | 要做连续录音实验，就需要重新跑（旧结果是负面的，已删除） | 只改文字：在 §3.2、§4.1、Limitations 中写明 "sentence-level commitment under the tracked queue; continuous cross-sentence playback is a distinct deployment setting"（用审稿人给的句子 A）。不声称"未测试" |
| W2 | 没有在同一后验上比较连续 τ(q) 与简单门控 | 合理，而且审稿人列为 P0 | **数据已有**（v13 中二值门对比为 +0.35/−0.26/+0.24，两者只在 3.8% 的读上决策不同），但按你的决定已删除 | 正文 §4.4 改为审稿人的句子 D：只主张"队列感知优于固定延迟"，不主张"精确阈值优于所有更简单的队列规则"。是否把已有的门控对比放回附录，见 3.1 |
| W3 | 统计：没有精确的校正 p 值；没有按 talk 聚类的 bootstrap；没有区分主比较和 seed 稳健性 | 合理 | **数据已有**：按 talk 重采样的 CI 为 En→Zh [−0.20, +1.63]、Zh→En [+0.14, +1.77]、ACL [+1.85, +5.14]；校正后 p 为 0.030/0.030/<0.001 | §4.3 改为审稿人的句子 C（主比较与五 seed 分开陈述）；Table 3 表注写明检验方向和区间类型。是否在附录 C.3 写出精确 p、是否在附录 H 加 talk 聚类 CI，见 3.2 和 3.3 |
| W4 | 理论与部署之间的证明链没闭合 | 合理。附录 B 已披露，但正文表述还可以更严 | 估计 oracle 单元级遗憾需要新分析，不做 | §3.3 Theorem 2 之后改用审稿人的句子 B；贡献和结论中不出现"部署保证" |
| W5 | "听者感知延迟"是模型代理，没有听者数据 | 合理 | 预测时长 r·a 与实测的对比在 v13 中做过：D_k 约高估 2 倍，相关系数 ≤ 0.16，按你的决定已删除；人工听测没有做 | 只改文字：Definition 2 改为 "we model the delay the listener perceives as …"；Limitations 写明感知延迟由平均语速建模、未经听者验证 |
| W6 | 摊销后验与 TTB 的对比混杂了标签、容量和推断时间 | 合理，是个好的指正 | 匹配标签的 TTB 需要新实验，不做 | §4.5 改写为"完整后验之间的比较，不是对摊销这一单一因素的因果检验"，并指出 Table 6 在摊销模型内部分离了标签的作用 |
| W7 | 计算效率：单卡 H200 与 TP=2 是什么关系；没有 Talker 耗时和 RTF | 测量设置的澄清合理；RTF 需要新测量 | 不做新测量 | Table 5 表注和 §4.5 写明：计时为单路、单卡 H200、只计 Thinker；评测运行时用 TP=2 是为了吞吐；Talker 计算不计入，评测是 computation-excluded。§4.5 把 "cheapest" 改为 "lowest Thinker cost per read" |
| 理论 1 | Figure 2 写成 "stopping NLL = min E KL"，等号不对 | 正确，是笔误 | 否 | 改为 "stopping NLL $=\mathbb{E}\,\mathrm{KL}(p\|p_\theta)+H$" |
| 理论 2 | Proposition 3 不能说成"精确执行 Theorem 1" | 正确。v16 新加的 T7 句子 "stops decoding where Theorem 1 would" 说过了 | 否 | 改为：只有当首选内容词占满（π_k=1）时才与 Theorem 1 一致，否则更保守；实现是 Theorem 1 的保守版本 |
| 理论 3 | 实际训练还有 content loss，不能把整个训练损失等同于 Proposition 2 的 stopping NLL | 正确 | 否 | Proposition 2 之后加一句："the full training loss adds the content term, which shares the adapter; Proposition 2 concerns the stopping term only." |
| 理论 4 | "前面单元一旦错了，后面的错误不再收费"这个损失需要说明为什么合理 | 合理 | 否 | Definition 3 之后加一句理由（见 2.3），附录 B.2 已讨论它对全局最优的影响 |
| 证据 1 | 3.1%/5.5% 是选定工作点的结果，不是 Table 3 的 E1 | 合理 | 否 | 这句话在摘要和引言里，是否修改见第 4 节 |
| 证据 2 | Table 6 的标签对比只有两次训练，语速分析只有一个 talker seed | 合理 | 否 | §4.6 正文中注明 "(two runs)" 和 "(one talker seed)"，避免过度推广 |
| 原创性 | 贡献属于组合式创新，基础工具都是经典的 | 认可 | 否 | 不改，相关表述本来就不夸大 |
| 基线 | Table 2 和 Figure 3 中的外部系统只能作背景，不能作为策略层面的因果优势 | 合理 | 否 | §4.3 加一句：外部系统的模型规模、解码方式和语音生成都不同，Omni 的分档预算是从图上读的；策略之间的比较只看同一 Thinker 和 Talker 上的 LA2、Hold-n、Vote |
| P2-1 | 引言与摘要、贡献列表有重复 | 合理 | 否 | 只删不改，见第 4 节 |

## 2. v17 的文字修改（不需要任何新实验）

### 2.1 §3.2（W1、W5）

- Definition 2 中 "The delay the listener perceives from waiting is" 改为 "We model the delay the listener perceives from waiting as"。
- 队列模型段的最后一句改为：

  > Our speech evaluations reset the playback queue at sentence boundaries, so we assess sentence-level commitment under this tracked queue; continuous, cross-sentence playback is a distinct deployment setting.

### 2.2 §3.3（W4、理论 3）

- Proposition 2 之后补一句：

  > The full training loss adds the content term, which shares the adapter; Proposition~\ref{prop:kl} concerns the stopping term only.

- Theorem 2 之后的范围句，按审稿人的句子 B 改为：

  > Theorem~\ref{thm:regret} bounds the expected per-unit regret of plug-in decisions under the training distribution; it does not establish global prefix optimality, nor a guarantee for the adapted Thinker and the biased greedy decoder under deployment shift (Corollaries~\ref{cor:shift} and~\ref{cor:target}, Appendix~\ref{app:scope}), so we measure calibration on the policy's own reads instead (Appendix~\ref{app:posterior}).

### 2.3 Definition 3 之后（理论 4）

> Charging only the first wrong unit mirrors the stable length, which ends at the first divergence: once a revision is heard the listener has already met an error, so the loss counts whether a commitment introduces one rather than how many follow (Appendix~\ref{app:scope-global} discusses what this implies for whole prefixes).

### 2.4 §3.4（理论 2）

替换 v16 的 T7：

> Proposition~\ref{prop:bias} is what makes \qabc{} a single greedy pass: a bias of $\mathrm{logit}\,\tau(q_k)$ on $\wait$ stops decoding exactly where Theorem~\ref{prop:threshold} would when the top content token takes all non-wait mass ($\pi_k{=}1$), and otherwise only waits longer, so the implemented rule is a conservative version of the Bayes rule rather than an exact one.

### 2.5 Figure 2（理论 1）

训练框中改为 "stopping NLL $=\mathbb{E}\,\mathrm{KL}(p\,\|\,p_\theta)+H$ (Prop.~2)"。

### 2.6 §4.1（W1）

Data 段中 "at the sentence level as SimulS2ST-Omni does" 补为 "at the sentence level as SimulS2ST-Omni does, with the playback queue reset at every sentence"。

### 2.7 §4.3（W3、基线）

- Latency tiers 段之后补一句：

  > External systems differ in model size, decoding and speech generator, and SimulS2ST-Omni's budgets are read from its figures, so Table~\ref{tab:realsi-main} places \qabc{} among them rather than isolating policies; the policy comparison is with LA2, Hold-$n$ and Vote on our Thinker and Talker.

- Primary 段按审稿人的句子 C 改写：

  > The pre-specified single-adapter comparison improves equal-latency quality on all three test sets (Table~\ref{tab:primary}; one-sided tests corrected for the three test sets, Appendix~\ref{app:protocol}). Across five training seeds the mean gains remain positive, but the interval includes zero for RealSI En$\rightarrow$Zh, so the evidence for that direction is sensitive to adapter training.

- Table 3 表注补一句：

  > E1 and E2 test \qabc{} $>$ LA2 one-sided; the intervals are two-sided 95\% percentile intervals of the same bootstrap.

### 2.8 §4.4（W2）

段末按审稿人的句子 D 补充：

> On the learned posterior, the queue-aware threshold improves over a constant-delay threshold in every matched setting we tested; the comparison with queue-gated LocalAgreement does not by itself show that this exact threshold beats every simpler queue-aware threshold on the same posterior.

### 2.9 §4.5（W6、W7）

- 补一句：

  > This compares complete posteriors rather than amortization alone: \ttb{} also differs in its labels (agreement among samples rather than full-context stable lengths) and its capacity (an MLP prior on 17 features); within amortized adapters, Table~\ref{tab:ablation} isolates the labels.

- "The amortized policy is also the cheapest" 改为 "\qabc{} also has the lowest Thinker cost per read"。
- Table 5 表注改为：

  > Thinker computation per read, measured with one stream on one H200 (the evaluation runs serve the Thinker with tensor parallelism 2 for throughput); Talker computation is not included …

### 2.10 §4.6（证据 2）

正文中 consensus 标签处注明 "(two training runs)"，语速处注明 "(one talker seed)"。

### 2.11 Limitations（W1、W5）

在现有 3 句中并入：

> Evaluation resets the playback queue at every sentence, and perceived delay is modelled from a mean speech rate rather than measured with listeners.

## 3. 需要你决定：已有数据，只是此前删掉了，不需要新实验

审稿人把下面三项列为 P0。数据都现成，恢复只是改附录。不恢复也可以，正文按第 2 节收紧主张即可。需要提醒的是：如果真实审稿人在 rebuttal 里问到，这些结果都存在，回答时必须如实说明。

### 3.1 同一后验上的二值门对比（W2 / P0-2）

- **数据**：推导阈值相对二值门 +0.35/−0.26/+0.24；两者只在 3.8% 的读上决策不同（82% 的读队列为空或满一个读窗，两条规则在这些读上一致）。
- **做法**：附录 G 加一小表，标题 "A two-level gate approximates τ(q)"，写明二值门就是 τ(q) 在两个端点上的取值，在这批数据上两者无法区分。
- **推荐恢复**。正文已经按句子 D 收窄了主张，这张表不会削弱主张，反而直接回答了审稿人最关心的问题。

### 3.2 按 talk 聚类的 bootstrap（W3 / P0-3）

- **数据**（adapter seed 0）：En→Zh [−0.20, +1.63]、Zh→En [+0.14, +1.77]、ACL [+1.85, +5.14]。
- **做法**：附录 H 的稳健性表加一列 "talks resampled, 95% CI"。
- **推荐恢复**，并在正文 Limitations 写一句：En→Zh 在按 talk 聚类时区间含 0。
- 如果不恢复：Limitations 至少要写明 "sentence-level resampling ignores clustering by talk"。

### 3.3 精确的校正 p 值（W3）

- **数据**：0.030 / 0.030 / <0.001（单侧，Holm 校正三个测试集）。
- **做法**：只写进附录 C.3 那一句，不放进表格。
- **推荐恢复**。摘要（你写的）说 "significant"，论文里需要有可核对的依据。如果你坚持全文不出现 p 值，正文和摘要就要避免 "significant" 一词，改成区间表述。

## 4. 摘要和引言：建议的改动（需要你同意）

1. **摘要（证据 1）**：把 "At an approximately 2-second latency budget, QABC achieves 3.1%/5.5% relative ASR-BLEU gains" 改为 "At selected operating points near a two-second budget, the relative ASR-BLEU gains are 3.1%/5.5%"（审稿人的句子 F），以免和事先固定的主比较混淆。
2. **引言（只删不改）**：
   - 删去第 3 段末尾的 "while also reducing latency in both directions"。这只在选定工作点成立，Table 3 中等质量节省的延迟（E2）在 RealSI 上并不显著。
   - 删去第 2 段与第 1 段重复的 "Existing policies often rely on heuristics … playback queue" 一句（审稿人 P2-1）。

这两处删除还能省出约 4 行，正好补偿第 2 节新增的篇幅。

## 5. 不做的事（都需要新实验），以及 rebuttal 时的应对

| 审稿人的建议 | 现状 |
|---|---|
| 连续录音、同一时钟的 S2ST，报告队列轨迹和 p50/p95 等待 | 做过一版，结果负面：QABC 不领先，backlog 持续积累，已删除。不重跑；rebuttal 被问到时如实说明 |
| 线性门控、D_k(u) 等阈值变体 | 块内延迟变体和精确阈值规则只在 dev 上做过，已删除。不新增 |
| 按队列分组、按位置的 oracle 单元级遗憾 | 没做，不做 |
| 实际语音时长与 r·a 的误差 | 做过：D_k 约高估 2 倍，已删除。不恢复；正文措辞改为"建模的感知延迟" |
| 人工盲听 | 没做；已写入 Limitations |
| 端到端 RTF、Talker 的 p50/p95 | 没测；表注写明 Talker 计算不计入 |
| 匹配标签的 TTB、只用 stopping 或只用 content loss 的消融 | 没做；§4.5 改为不作因果归因 |

## 6. 页数预算

v16 正文正好占满 8 页，没有余量。

- **新增**：第 2 节约 20 行（约 0.17 页）。
- **补偿**（按顺序执行，压够即停）：
  1. 第 4 节引言的两处删除：约 4 行（需你同意）；
  2. Table 4 下半部分的两行对照改为 §4.4 正文中的一句，数字照写：约 0.05 页；
  3. Related Work 第 2 段压缩：约 3 行；
  4. Figure 1 图注再缩一行，§4.3 Latency tiers 段压缩 1–2 行；
  5. 如果仍超页：§4.5 新增的两句合并为一句。

定义、命题、定理和 v16 新加的说明句都不删。

## 7. 执行步骤

1. 用 `cp -a` 把 v16 复制为 `paper/bayes_simul_s2st_v17/`，v16 不动。
2. 按第 2 节修改正文、图注和表注；生成脚本复制为 `fill_paper_v17_body.py`，修改 Table 3、Table 5 的表注，以及 Table 4（视补偿需要）。
3. 按你在第 3、4 节的决定处理附录和摘要、引言。
4. 检查：
   - 编译后没有未定义的引用，没有 Overfull；
   - 正文不超过 8 页，第 9 页从 Impact Statement 开始；
   - grep 正文中 "deployment guarantee"、"exactly where" 等过强表述；
   - 摘要和引言除你同意的改动外逐字不变。
5. 写 CHANGELOG，提交并推送。

## 8. 需要你决定

1. 3.1 二值门对比：恢复到附录 G（推荐），或不恢复。
2. 3.2 按 talk 聚类的 CI：恢复到附录 H，并在 Limitations 写一句（推荐）；或只在 Limitations 写"未做聚类"。
3. 3.3 精确的校正 p 值：写进附录 C.3 一句（推荐，因为摘要说了 "significant"）；或全文去掉 "significant"。
4. 第 4 节摘要那一句的改写，以及引言的两处删除：同意（推荐）或保持原样。
