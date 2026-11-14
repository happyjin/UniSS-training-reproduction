# MIST v10 修改方案：把 mist_w3po_v1 的方法和结果写进论文（2026-11-14）

- 起点稿：`paper/gstar_si_v9/`（ACL 格式，11-07 暂停时的状态）。v10 在新目录 `paper/gstar_si_v10/` 里改，**v9 目录一个字节都不动**。
- 依据：`reports/mist_w3po_v1/` 下的 LOG、`final_test_results_v1`、`joint_baseline_vs_method_20261114`、`aprime_plem_crf_formal_and_latency`、`crf_formal_and_next`、`serial_v2p_train`、`latency_chunk_protocol_table_v1`、`multispeaker_baselines_plan_v1`。
- 本文只是方案：不改任何代码或实验，不动测试集。

---

## 0. 结论

### 0.1 δ = −1 可以固定

可以，而且应该固定。论文里这样处理：

- **写法**：δ 是解码时的一个常数，地位和“贪心解码”“读周期 1 s”一样，写进方法和超参数表。正文不出现 δ 的扫描曲线，也不把它当可调旋钮来卖。
- **来源**：附录用一句话加一张小表，说明 δ 只在 dev5 上定过一次。规则是“文本 LAAL 最接近严格基线”，候选 {−1, −2, −3, −5}，δ = −1 在完整 dev 上确认。测试集只跑 δ = −1 一次（正在跑）。
- **两方向同一个值**：zh→en 和 En→Zh 都是 −1，不需要按方向分开。
- **我之前建议的“基线也扫 δ 画曲线”不再需要**。只保留一个可选对照：严格基线 + δ = −1（dev5 文本，1 次解码，约 1 小时，不是扫参）。它回答审稿人最可能问的一句“偏置是否对基线同样有效”。是否做由你决定（§7）。

### 0.2 v10 的定位

v9 的主张是“谁、何时、说了什么”：说话人归属的格式、重叠区评测、和级联的对比。这些全部保留。

v10 在它上面加一条新主线：**把“何时写”和“何时说”都当成隐结构，用精确的前向-后向做 EM 来学**。
- **Thinker**：SerialCRF v2′，写出时刻和说话人顺序是隐变量。
- **Talker**：PL-EM，播放积压格上的后验，带延迟预算。
- 再加一个固定的解码常数 δ。

动机来自 v9 自己暴露的问题：
- 语音延迟约 4 s；
- 56% 的词在窗口最终读才写出，话语层面接近交替传译；
- 写出标签是固定 1 s lag 的规则；
- talker 是规则式排队。

**整体篇幅原则：不让论文变杂**。新增只有“两个隐结构模块 + 一个常数”这一条线。下面这些都只是系统细节，写进实验设置或附录，不列为贡献：
- 读级块合并（rl）；
- C2 关窗 gap 0.6 s；
- chunk 延迟协议；
- 联合双向训练。

---

## 1. v9 → v10：系统和口径的变化

| 项 | v9（MIST-Simul） | v10（MIST） | 论文里怎么写 |
|---|---|---|---|
| Thinker 写出训练 | deadline 目标：标点块按字符比例定截止，+1 s lag，SFT | 第 1 段：同样的 deadline SFT（联合双向，作为严格基线）；第 2 段：SerialCRF v2′ EM（3500 步） | 方法新增一节；deadline 目标保留，作为初始化和基线 |
| 写出决策 | 贪心到 `<wait>` | 同上，`<wait>` 的 logit 加常数 δ = −1 | 方法里一个公式 |
| 方向 | 每个方向一个 Simul 模型 | 一个联合双向模型（zh→en + En→Zh） | 实验设置 |
| talker | UniSS talker 块级微调，规则排队（式 layout） | PL-EM：播放积压格上的 CRF 后验 + 后验正则延迟预算；推理时每块采样候选、CTC 挑选 | 方法新增一节 |
| 读级合并 | 无 | 同一读、同一说话人的块合并后送 talker（rl） | 实验设置一句 + 附录 |
| 在线关窗 | 1 s 无活动 | 0.6 s（C2） | 实验设置一句 |
| 主延迟 | 文本 LAAL + 听众侧滞后中位数（v8 排队模型） | **每条流的语音 LAAL（逐单元，词按块内播放展开），chunk 协议**；EndOffset 补充；单通道 LAAL 只作补充 | 指标一节重写 |
| 主质量 | tcpBLEU（参考窗口和整场）、cpBLEU | 整场 tcpBLEU all / ov / 1spk（官方 v8 overlap_score）+ 语音 ASR-BLEU（两个 TTS 种子） | 基本不变 |
| 统计 | 精确置换检验 + Holm、带解码噪声的 bootstrap | 新比较：按会议配对 bootstrap（gate_2seed，10k），P(差 ≤ 0) | 统计一段补一句 |

**新的主比较（测试集，只跑一次）**：
- 严格基线 = 联合 v8 SFT s6000 + 冻结的 T1c talker；
- MIST v10 = 联合 v2′ s3500 + δ = −1 + rl + PL-EM talker（zh→en 用 `talker_plem_20261111T001759Z`；En→Zh 用 `talker_plem_en2zh_20261114T045019Z` i0400）。

---

## 2. 贡献（引言最后一段的新写法）

保留 v9 的两条，新增两条：

1. （v9）说话人优先、带时间戳的联合解码格式，以及读入分离器流的融合。
2. （v9）重叠感知的评测和对照：tcpBLEU-ov、并发召回、级联与重标对照、预登记检验。
3. **（新）写出时刻的隐结构训练（SerialCRF v2′）**。每个译文块允许在标签读或晚一读写出，同一读内的说话人顺序也是隐变量。后验的配分函数精确，用前向-后向求，EM 训练。再配一个固定的解码常数 δ，纠正 EM 的晚写漂移。
4. **（新）播放时间线上的说话训练（PL-EM）**。talker 每块的候选和积压状态构成线性链 CRF，势函数里直接包含语音时间线的“秒 × 词”延迟。延迟预算按 Ganchev et al. (2010) 的后验正则求对偶 λ*。

摘要和引言中对应的结果句，等 §3 的 E1/E2 跑完再填数字。

---

## 3. 实验：增加、替换、移走

### 3.1 必做

| 编号 | 实验 | 为什么必须做 | 成本 | 状态 |
|---|---|---|---|---|
| E1 | v10 主系统测试集：v2′ s3500 + δ = −1 + rl + PL-EM；AMI zh→en、AliMeeting、AMI En→Zh；文本 + 语音（两个种子），全部指标 | 主表 | 8 卡叠加约 2–3 小时 | **正在跑**（`wb_test.sh`，21:18 起） |
| E2 | **严格基线 Thinker + rl + PL-EM talker**（dev 和测试集） | 现在的主比较同时换了 Thinker 和 talker。dev zh→en 的 ASR-BLEU +4.3（19.24 → 23.50）里有多少来自 talker，现在说不清，审稿人一定会问。有了这一行，增益就能拆成“Thinker 部分 + talker 部分” | 基线文本已解码，只需读级合并 + 语音链：dev 约 1 小时，测试约 2 小时（叠加） | 未做，需要新驱动（复制 `wb_test.sh` 的 speech 部分） |
| E3 | 显著性：E1 对严格基线、E1 对 E2、E2 对严格基线，按会议配对 bootstrap + 精确符号翻转置换检验，Holm 校正 | v9 的统计口径；新主张要有同样的检验 | CPU | 未做 |
| E4 | v10 的计算成本：每读耗时、RTF（沿用 v8 的排队模型口径），以及 PL-EM 推理时多候选采样的额外 talker 耗时 | v9 的主张之一是“跟得上音频”；v10 的 talker 推理变贵了，必须交代 | 从解码日志统计，CPU | 未做 |

### 3.2 建议做

| 编号 | 实验 | 作用 | 成本 |
|---|---|---|---|
| E5 | DiariST-AliMeeting 的 SAgBLEU / SAtBLEU（我们的文本和 ASR 文本） | 唯一能和已发表多说话人数字直接对上的外部对照；只用评测脚本，不跑 DiariST（符合 11-03 的决定） | CPU，约半天；先确认我们的 Ali 音频对应 IHM-MIX |
| E6 | 严格基线 + δ = −1（dev5 文本，1 次） | 回答“偏置对基线是否同样有效”；不是扫参 | 约 1 小时 |
| E7 | SeamlessStreaming 直接喂混合音频（多说话人测试集） | 一个公开 S2ST 系统在重叠会议上的表现，补“外部基线为零”的缺口（`multispeaker_baselines_plan_v1` 的 B-sim-2） | 约 10 卡·时 |

### 3.3 不做（写进 Limitations）

- 第 3 条思路（对等待决策加延迟奖励或 RL）：放进未来工作，一句话。
- 人工听测、COMET / MetricX：按既定决定不做。
- 方案 B 的更多 ρ 取值：只报 ρ = 0.5 一个预登记值。
- v9 暂停时未完成的 P0-B 固定滞后全集、P1-4 speaker-blind 补种子：它们针对的是旧系统。已完成的部分（Lag2 全集、D2 子集等）放附录；没完成的写进 Limitations，不再补跑。

### 3.4 表格的去留

| v9 表 | 内容 | v10 处理 |
|---|---|---|
| `tab_v7_main`（Table 2） | 参考窗口上的离线 MIST 对 4 个级联 | **保留**在正文，可压缩列数。这是归属贡献的证据，系统不变 |
| `tab_v8_ovmain` + `fig_v8_overlap` | 重叠区分解 | **保留**在正文 |
| `tab_v8_latency` | v8 系统的 RTF 和听众侧滞后 | **替换**为新主表 `tab_v10_main`；原表移到附录“v9 口径” |
| `tab_v7_online`、`tab_v8_sched`、`tab_v8_reg`、`tab_v7_reg` | v8 在线运行和预登记检验 | 移到附录，标题注明“v9 系统（deadline SFT）、v9 口径” |
| 新增 `tab_v10_main`（正文） | 测试集主表：严格基线、E2（基线 + PL-EM）、MIST v10；三个测试集 × {tcpBLEU all/ov/1spk，ASR-BLEU，语音 LAAL，EndOffset} | 正文核心表 |
| 新增 `tab_v10_ablation`（正文，dev） | 逐步加组件，见 §3.5 | 正文 |
| 新增 `tab_v10_calib`（附录） | δ 的 dev5 标定（5 行 × 2 方向） | 附录 |
| 新增 `tab_v10_drift`（附录） | 晚写漂移的证据（§5.4） | 附录 |
| 新增 `tab_v10_m2laal`（附录） | 单说话人口径的语音 LAAL（块开始时刻），方便和单说话人论文对照 | 附录 |
| 新增 `tab_v10_dup`（附录） | Ali 的重复率、BP 与精度分解、去重 BLEU（诊断用） | 附录 |
| 新增 `tab_v10_cost`（附录） | E4 | 附录 |
| 新增 `tab_v10_diarist`（附录或正文小表） | E5 | 视结果定 |

### 3.5 正文消融表 `tab_v10_ablation`（dev，两个方向）

每一行只改一个东西。现有数字来源都写在表后，空格的要补跑。

| 行 | Thinker | 解码 | talker | 现有数据 |
|---|---|---|---|---|
| 1 | deadline SFT（严格基线） | δ = 0 | T1c | 有（joint dev `jdev_b*`） |
| 2 | deadline SFT | δ = 0 | rl + PL-EM | **E2 补** |
| 3 | + SerialCRF v2′ | δ = 0 | rl + PL-EM | 有（`jdev_m*_s03500`） |
| 4 | + SerialCRF v2′ | **δ = −1** | rl + PL-EM | 有（`*_wbzm1em1`） |
| 5 | v2′ + E 步延迟预算 ρ = 0.5（方案 B） | δ = 0 | — | dev5 文本已有；作为“训练时控延迟”的对照，只列文本 |
| 6 | 允许提前写的格 [e−1, e+1]（PMC） | δ = 0 | — | 已有（分开的 zh→en 系统，dev：延迟 −0.79 s，但语音 −1.70、文本 −1.92） |

talker 的单独消融（分开的 zh→en 系统，AMI 测试集，已有，进附录）：
- 规则 talker 20.52 / 3.998；
- PL-EM 20.64 / 3.919；
- rl + PL-EM 20.68 / 3.904。

（ASR-BLEU / chunk 语音 LAAL）

已有的 dev 主数字（joint，完整 dev）：

| 方向 | 行 1 严格基线 | 行 3 v2′ δ=0 | 行 4 v2′ δ=−1 |
|---|---|---|---|
| zh→en 文本 / ASR / 语音 LAAL / EndOffset | 27.93 / 19.24 / 3.973 / 4.142 | 27.75 / 23.33 / 3.957 / 3.430 | 28.31 / 23.50 / 3.605 / 3.176 |
| En→Zh 文本 / ASR / 语音 LAAL / EndOffset | 41.07 / 31.07 / 3.121 / 3.450 | 41.90 / 31.72 / 3.255 / 3.162 | 41.35 / 31.65 / 2.962 / 2.932 |

### 3.6 必须如实交代的三件事（写进正文或附录，不能省）

1. **δ 是看过第一次测试结果之后才引入的**。
   - 第一次测试（δ = 0，11-14 08:18 UTC 完成）显示方法起步更晚（Ali 语音 LAAL 4.277 → 4.848），之后才提出方案 A。
   - δ 只在 dev 上选定，测试集只跑一次。但“加这个旋钮”这个决定本身受了测试结果影响。
   - 正文要写明，并在附录同时给出 δ = 0 的测试结果。这不是扣分项，隐瞒才是。
2. **第 2 段检查点选的是 3500，不是规则选出的 4000**。
   - 用户 11-13 21:31（布拉格）只看 dev 数据时改的，依据是“BLEU 优先于小幅延迟超出”。当时方法行的 s4000 测试编排器还没开始任何作业。
   - 3500 的 En→Zh 延迟比参考多 +0.137 s，超出预登记的 +0.1 s 可行线。
   - 写进预登记偏离一节，同时报告 4000 步的 dev5 数字（LOG 当时已约定）。
3. **AliMeeting 上方法的 tcpBLEU 低于基线**（δ = 0 时 18.99 对 21.43）。
   - 已查明主要来自 BP：基线的跨窗口重复译文更多（重复词 27.7k 对 12.7k），把长度撑高了。
   - 去重后方法反而高 2.33。
   - 正文主指标仍是官方 tcpBLEU，照实报；去重 BLEU 和 BP/精度分解放附录，作诊断，不当主指标。
   - 如果 E1（δ = −1）在 Ali 上仍低，摘要和结论要收窄到“AMI 两个方向”。

---

## 4. LaTeX 修改：逐文件方案

### 4.0 目录与构建

```bash
# 只复制，不改 v9
cp -r paper/gstar_si_v9 paper/gstar_si_v10
# 新表格脚本（新文件），读取下列 json，生成 generated/macros_v10.tex 和 tab_v10_*.tex
paper/gstar_si_v10/scripts/make_tables_v10.py
# 构建
cd paper/gstar_si_v10 && python scripts/make_tables_v10.py && tectonic main.tex
```

`make_tables_v10.py` 的数据来源（只读）：

| 用途 | 文件 |
|---|---|
| 测试主表（δ = −1） | `eval_outputs/mist_w3po_v1/joint/wb_test_summary_v1.json`（E1 完成后） |
| 测试（δ = 0、严格基线） | `eval_outputs/mist_w3po_v1/joint/test_summary_v1.json` |
| dev 主数字 | `eval_outputs/mist_w3po_v1/decode/jdev_*/score_*.json`，`crf/t_eval/jdev_*/lat_chunk_*.json` |
| δ 标定 | `logs/mist_w3po_v1/joint/wb/summary.tsv` + stage-2 dev5 代理 `s2*_dev5_joint_v2p_*_s03500` |
| 方案 B | `eval_outputs/mist_w3po_v1/joint/joint_v2pl_20261114T130043Z/selection.json`、训练日志的 `lat_*` / `exp_defer` / `q_label` |
| talker 消融 | `eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/ami/table_v1.json` |
| 晚写漂移 | `reports/mist_w3po_v1/joint_baseline_vs_method_20261114.zh-CN.md` 的统计脚本输出 |

**v9 遗留问题，顺手修**：
- `main.tex` 没有 `\input{generated/macros_v9}`，但 `results.tex` 用到了 `v9:` 宏；
- `app:v9` 未定义（`main.log` 有 undefined reference 警告）；
- `tab:related8` 要确认有定义（相关工作表现在只有 `tab:related7`）。

### 4.1 `main.tex`

- 标题保留 “Who Said What and When”，副标题改成：

```latex
\title{Who Said What and When: Speaker-Attributed Simultaneous Meeting Interpretation\\
with Latent Write and Playback Alignment}
```

- 增加 `\input{generated/macros_v9}` 和 `\input{generated/macros_v10}`。
- 新宏：

```latex
\newcommand{\bias}{\delta_{\mathrm{w}}}   % 写偏置；不要用 \delta，正文 \delta 已是 80 ms 帧长
```

（v9 的 `\delta` 已经表示 80 ms 帧长，所以写偏置用 `\delta_{\mathrm{w}}`，避免符号冲突。）

### 4.2 `sections/abstract.tex`（重写，约 180 词）

结构：
1. 任务一句；
2. v9 的格式与归属结果一句（数字保留 v7/v8 宏）；
3. **新**：写出时刻与说话顺序作隐变量的 EM，talker 在播放时间线上的 EM；
4. **新**：测试集主结果一句：相对 deadline SFT 基线，tcpBLEU / ASR-BLEU 和语音 LAAL 的变化，用 `v10:` 宏，等 E1/E2/E3；
5. 局限一句：单通道仍跟不上；真实普通话上的结论（视 E1 的 Ali 结果）。

### 4.3 `sections/introduction.tex`

- 第 3 段（“We present MIST …”）后加一段动机：

  > A deadline-trained decoder writes most words only when a window closes (56\% at its last read) and a rule-based
  > speech layer queues them, so a listener hears a segment about 4\,s after its source. We treat both decisions as
  > latent structure: …

- 贡献段按 §2 的四条重写。
- 删掉 v9 里“text lag does not grow with overlap”之类针对旧系统的结果句；保留归属部分的数字句。

### 4.4 `sections/related.tex`

在同传段补三条，每条一句：
- 隐读写或隐对齐的同传训练（HMT、单调注意力族、CTC 式隐对齐）：我们的不同点是**集合状态 + 说话人顺序**的格；
- 后验正则（Ganchev et al., 2010）：我们把语音时间线延迟作为期望约束；
- 测试时延迟控制（阈值类策略）：δ 是同一类常数，但只取一个固定值。

### 4.5 `sections/method.tex`

保留 §Task 到 §Training 不变。修改如下。

**(a) §Simultaneous interpretation**：
- “Deadline targets”一段保留，改为“初始化和基线”，并说明联合双向：一个模型，两个方向的行交错。
- 后面新增两段。

```latex
\paragraph{Latent write alignment.} Deadline targets fix when each chunk is written. We let the data decide within
one read: chunk $i$ of speaker $k_i$, due at read $d_i$ (Eq.~\ref{eq:due}), may be written at $d_i$ or at $d_i+1$.
The latent plan is $z=(A_{1:R},o_{1:R})$, where $A_r$ is the set of chunks written by read $r$ and $o_r$ the order
in which the speakers of read $r$ are written, with
\begin{equation}
\{i:d_i\le r-1\}\subseteq A_r\subseteq\{i:d_i\le r\},\qquad A_R=I,
\label{eq:lattice}
\end{equation}
and $A_r$ prefix-closed within each speaker. With $W_r^{o}(A',A)$ the tokens a read writes when it moves from $A'$ to
$A$ in order $o$, and $e_r$ its stop token,
\begin{equation}
\psi_{r,o}(A'\!\to\!A)=\log p_\theta\big(W_r^{o}(A',A)\,e_r\mid \bm{x}_{\le t_r},\bm{c},A'\big),\quad
q(z)=\frac{\exp\sum_r\psi_{r,o_r}(A_{r-1}\!\to\!A_r)}{Z}.
\label{eq:post}
\end{equation}
$Z$ and every edge marginal are exact by forward--backward over the set states. The M-step minimises the
cross-entropy of the read rows weighted by the tempered posterior $q_\tau$ ($\tau=0.5$, no gradient through $q$):
\begin{equation}
\mathcal{L}_{\mathrm{W}}(\theta)=-\sum_{r}\sum_{A'\to A,\,o}q_\tau(r,A'\!\to\!A,o)\,
\log p_\theta\big(W_r^{o}(A',A)\,e_r\mid\cdot\big).
\label{eq:mstep}
\end{equation}

\paragraph{Write bias.} Hearing one more read makes the same translation more likely, so EM moves posterior mass
towards the later write and the trained decoder waits longer (\S\ref{sec:res-drift}). At decoding we add one constant
to the wait logit,
\begin{equation}
\tilde{\ell}_u(\wait)=\ell_u(\wait)+\bias,\qquad \bias=-1,
\label{eq:bias}
\end{equation}
at every step of a read where waiting is allowed (never at the last read). $\bias$ was set once on development
meetings, as the value whose text lag was closest to the deadline-trained model's, and is not tuned per direction or
test set (Appendix~\ref{app:calib}).
```

**(b) §Speaking in each speaker's voice**：
- 保留“Voices and layout”中的音色提示部分。
- 式 `eq:layout` 保留，用于单通道。
- 新增一段 PL-EM：

```latex
\paragraph{Speaking on the playback timeline.} Each speaker's stream is a sequence of blocks $n=1,\dots,N$ that
become available at $a_n$; for each block the Talker samples a candidate set $C_n$ (8 samples at two speaking rates)
with durations $d(c)$. The backlog $b_n\in\{0,\Delta_b,\dots,B\}$ evolves deterministically,
\begin{equation}
b_{n+1}=\big\lceil\max(0,\,a_n+b_n+d(c_n)-a_{n+1})\big\rceil_{\Delta_b},
\label{eq:backlog}
\end{equation}
and a block's words, spread over its playback, are heard $L_n(b,c)=W_n b+\tfrac{W_n+1}{2}d(c)$ word-seconds after
$a_n$. A linear-chain CRF over $(b_n,c_n)$ scores
\begin{equation}
\psi_n(b,c)=\tau\,\bar\ell(c)+\beta f(c)-\lambda L_n(b,c),
\label{eq:plpot}
\end{equation}
with $\bar\ell$ the Talker's mean token log-probability, $f$ a CTC fidelity term, $\tau=0.5$, $\beta=5$. $\lambda$ is the
dual of the posterior-regularised problem $\min_q\mathrm{KL}(q\Vert q_0)$ s.t.\ $\mathbb{E}_q[\sum_nL_n]/\sum_nW_n\le
\text{budget}$~\citep{ganchev2010posterior}, found by bisection. The Talker is trained on the candidates weighted
by the posterior at the backlog it actually reached; at inference it samples candidates and keeps the one with the best
CTC fidelity.
```

- 训练权重、语速 token 损失、KL 正则：写进附录 `app:plem`，正文只留一句。

**(c) §Running on a whole meeting**：
- “closes after 1 s” 改为 0.6 s。
- 加一句读级合并：同一读、同一说话人的块合并后再送 talker。

### 4.6 `sections/setup.tex`

- **Models**：新增联合训练的描述。
  - 第 1 段：联合 deadline SFT，960k 行，两个方向交错，dev5 规则选 6000 步。
  - 第 2 段：联合 v2′，dev5 规则选 4000，按 BLEU 优先改选 3500（偏离，见附录）。
  - PL-EM talker：每个方向一个；En→Zh 由 dev5 规则选第 400 次迭代。
- **Baselines**：
  - 新增“严格基线”（第 1 段 + T1c talker）和 E2 行；
  - v9 的级联、重标、LA2 原样保留；
  - 说明后者属于 v9 系统和 v9 口径。
- **Metrics**：重写延迟部分，给出定义。

```latex
\emph{Speech LAAL} (our primary latency) is LAAL~\citep{papi2022over} over reference units with one delay per target
word: the moment the word is heard, with the words of a synthesised block spread evenly over its playback; blocks are
released no earlier than the online window detector would release them (\emph{chunk protocol},
Appendix~\ref{app:chunk}). EndOffset is the delay from the end of a source unit to the end of its last heard word.
```

- **Statistics**：
  - 新比较的口径：按会议配对 bootstrap，两个 TTS 种子合并；
  - 精确符号翻转置换检验 + Holm（E3）；
  - 门槛规则：点估计 ≥ 参照 − 0.2，且 P(差 ≤ 0) < 0.95。

### 4.7 `sections/results.tex`

- §Offline、§Overlapped speech：保留，只删去指向 `tab_v8_latency` 的句子。
- §Commit policy and latency、§Whole meetings：压缩成一段“v9 系统的在线结果（附录）”。
- 新增 §“Learning when to write and when to speak”（`sec:res-v10`），约 0.8 页：
  1. **主表** `tab_v10_main`：三个测试集。先报质量（tcpBLEU all/ov、ASR-BLEU），再报延迟（语音 LAAL、EndOffset）。
  2. **增益拆分**（E2）：Thinker 部分和 talker 部分各占多少。
  3. **消融** `tab_v10_ablation`（dev）：
     - v2′ 不加 δ：质量升、起步晚；
     - 加 δ：延迟回到基线以下；
     - 方案 B：训练时预算只减轻漂移；
     - PMC：允许提前写会伤质量。
  4. AliMeeting 的 BP 现象一句话，详见附录。
- 新增 §“Why latent alignment writes late”（`sec:res-drift`），约 0.3 页。数字都已有：
  - 后验中每窗口推迟的块数约 8（方案 B 降到约 4.2）；
  - 标签路径后验 0.57–0.60（方案 B 0.71）；
  - 有输出的读入占比下降（dev zh→en 45.9% → 36.2%）；
  - δ = −1 把它拉回来。

### 4.8 `sections/analysis.tex`

- 保留 v9 的错误分析。
- 加一段“话语层面接近交替传译”的分解（来自 `aprime_plem_crf_formal_and_latency` §2.1）：
  - 语音 LAAL = 原始提交 + 检测器夹紧 + 播放；
  - 56% 的词在最终读写出。
  
  这一段同时解释了为什么 v10 降延迟只能降一部分，并引出未来工作（句内增量写出）。

### 4.9 `sections/conclusion.tex`

- 结论两句：
  - 归属：v9 结论；
  - 时序：隐结构 EM + 固定 δ 在不降质量的前提下降低语音延迟（按 E1 的实际结果写）。
- **Limitations** 增删：
  - 增：
    - δ 是全局常数，且在第一次测试之后引入（如实）；
    - 语音 LAAL 仍约 3–4 s，话语层面接近交替传译；
    - PL-EM 推理时多候选采样有额外计算（E4）；
    - 单通道仍跟不上；
    - 只训练一个种子；
    - 方案 B 只试了 ρ = 0.5；
    - 对等待决策的延迟奖励（RL）未做。
  - 删：针对 v9 已不成立的说法，例如“Text LAAL is computation-unaware”这类表述，改为新口径。

### 4.10 `sections/appendix.tex` 和新附录

新增附录节（建议放到新文件 `sections/appendix_v10.tex`，在 `appendix.tex` 末尾 `\input`）：

| 标签 | 内容 |
|---|---|
| `app:v10train` | 两段联合训练的细节、早停规则、检查点选择与偏离（3500 对 4000） |
| `app:crf` | SerialCRF v2′ 的完整定义：格的构造、前向-后向、与暴力枚举对拍的误差、最终读全后验 |
| `app:plem` | PL-EM 完整定义：训练权重、语速 token 损失、KL 正则、预算与 λ* = 0.497 |
| `app:calib` | δ 的 dev5 标定表 `tab_v10_calib` + δ 引入时间的说明 |
| `app:chunk` | chunk 延迟协议、语音 LAAL 定义、单说话人口径（块开始时刻）对照表 `tab_v10_m2laal` |
| `app:drift` | 晚写漂移证据 `tab_v10_drift`；方案 B 的训练曲线（lat_E0 / lat_E / λ） |
| `app:dup` | Ali 的重复、BP、去重 BLEU 诊断 `tab_v10_dup` |
| `app:cost` | E4 成本表 |
| `app:v9proto` | v9 系统在 v9 口径下的在线结果、RTF、预登记检验（原 `tab_v8_latency`、`tab_v7_online` 等） |
| `app:v9` | 修复 v9 遗留的未定义引用（AliMeeting 人工参考的对齐细节） |

`appendix_v8.tex` 保持原样，作为 `app:v9proto` 的一部分。

### 4.11 `references.bib` 新增

- Ganchev et al., 2010, *Posterior Regularization for Structured Latent Variable Models*（JMLR）；
- HMT（Zhang & Feng, 2023, *Hidden Markov Transformer for Simultaneous Machine Translation*，ICLR）；
- 单调注意力族的一篇代表作（如 Arivazhagan et al., 2019, MILk）；
- Sarawagi & Cohen, 2004（semi-Markov CRF；只在讨论格状态时需要）。

写进论文前逐条核对原文出处，不引用审稿中的论文。

---

## 5. 正文新图表的具体样式

### 5.1 `tab_v10_main`（测试集，双栏宽）

列：

| Set | System | tcpBLEU all | ov | 1spk | ASR-BLEU | Speech LAAL ↓ | EndOffset ↓ |

行：每个测试集 3 行（严格基线、基线 + PL-EM、MIST v10）。

### 5.2 `tab_v10_ablation`（dev，单栏）

- 只列 zh→en 和 En→Zh 的 tcpBLEU、ASR-BLEU 和语音 LAAL。
- 行 5、6 只有文本，语音列写 “–”。

### 5.3 不加新正文图

- δ 的标定只放附录表，不画曲线。
- 延迟分解用一句话加附录表，节省正文篇幅。

### 5.4 `tab_v10_drift`（附录）

| 模型 | 每窗口推迟块数（后验） | 标签路径后验 | 有输出的读入占比（dev zh→en） | 文本 LAAL（dev5，chunk） |
|---|---|---|---|---|
| deadline SFT | — | — | 45.9% | 3.065 |
| v2′ | 约 8 | 0.57–0.60 | 36.2% | 3.540 |
| v2′ + E 步预算 ρ = 0.5 | 约 4.2 | 约 0.71 | 待统计 | 3.445 |
| v2′ + δ = −1 | — | — | 待统计 | 3.155 |

---

## 6. 顺序与时间

| 步骤 | 内容 | 依赖 | 时间 |
|---|---|---|---|
| 1 | E1 测试结果（正在跑） | — | 今晚 |
| 2 | E2 驱动（新文件，bash -n + DRY_RUN），dev + 测试语音链 | 你确认 | 约 3 小时 GPU |
| 3 | E3 统计、E4 成本、`tab_v10_drift` 待统计项 | E1、E2 | CPU 半天 |
| 4 | 复制 v9 → v10，写 `make_tables_v10.py`，生成表格和宏 | 3 | 半天 |
| 5 | 改正文（§4.1–4.9）和附录（§4.10） | 4 | 1–2 天 |
| 6 | 可选：E5（CPU）、E6（1 小时）、E7（约 10 卡·时） | 你决定 | 并行 |
| 7 | 构建、交叉检查所有 `\num{}` 宏无缺失、无未定义引用，导出 Overleaf zip（新文件名） | 5 | 半天 |

---

## 7. 需要你决定的事

1. E2（严格基线 + PL-EM talker）现在开跑吗？我认为必须做。
2. E6（基线 + δ = −1 的 dev5 对照，一次解码）做不做？
3. E7（SeamlessStreaming 外部基线）做不做？
4. 副标题用 “with Latent Write and Playback Alignment” 是否合适，还是保留 v9 原标题？
5. 如果 E1 在 AliMeeting 上 tcpBLEU 仍低于基线，是否同意把主张收窄到 AMI 两个方向，Ali 只报延迟和诊断？
