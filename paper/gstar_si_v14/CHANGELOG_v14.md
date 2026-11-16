# v14 修改说明（2026-11-17）

## 用户要求

- **Table 1**：放不同方法的多说话人语音到语音（S2ST）同传基线，消融实验移到另一张表。
- **语音质量指标**：从 ASR-BLEU 换成 cpASR-BLEU 和 tcpASR-BLEU，这两个指标能衡量说话人是否分对。
- **补跑**：还没跑过的基线在三个测试集上跑完。
- 结果写进新版 LaTeX。

预登记：`reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md`，含修订 1（cp 的计分范围）和修订 2（X-Translator 的输入电平）。

## Table 1（`tab:main`，通栏表）：多说话人 S2ST 系统对比

**系统**：
- SeamlessStreaming（E7，零样本）；
- X-Translator（E8，零样本，官方 xt-default 配置）；
- 同训练数据的级联：说话人无关的 MIST-Simul，事后用因果重标分配说话人，再由基础 Talker 合成（E9）；
- B；
- MIST。

**指标**：每个测试集列出 cpASR-BLEU、tcpASR-BLEU、LAAL_block、EndOffset。
- 粗体是每列最好的值。
- \* 表示 MIST 与该系统的差异显著：BLEU 按 cp、tcp 两族分别做 Holm 校正；延迟用 sign-flip 检验，p < 0.05。

**结果**：MIST 在 3 个集合上的 cp 和 tcp 都是最高，全部 p_Holm < 0.001。下表格式为 cp / tcp，LAAL / EndOff：

| 系统 | AMI Zh→En | AMI En→Zh | AliMeeting |
|---|---|---|---|
| Seamless | 12.10 / 8.48，2.73 / 2.53 | 13.79 / 11.31，2.50 / 3.04 | 13.59 / 9.42，3.24 / 2.72 |
| X-Translator | 12.05 / 8.64，2.69 / 3.44 | 23.31 / 19.96，2.74 / 3.32 | 19.46 / 13.77，2.77 / 3.80 |
| 级联 | 24.88 / 18.37，3.51 / 4.55 | 35.07 / 30.97，2.80 / 3.72 | 21.81 / 16.16，3.55 / 4.87 |
| B | 24.72 / 18.07，3.12 / 4.08 | 35.83 / 31.65，2.54 / 3.48 | 22.60 / 16.73，3.28 / 4.57 |
| **MIST** | **26.84 / 20.35**，3.03 / 3.13 | **36.49 / 32.23**，**2.43 / 2.91** | **24.04 / 18.08**，2.95 / 3.23 |

**X-Translator 的输入电平（修订 2）**：
- 在两个 AMI 集合上，每个集合用一个常数增益，由 dev 窗口确定（×1.81 / ×2.16）。
- AliMeeting 用原始输入。
- 不加增益的原始运行放附录 G.4：AMI Zh→En 为 6.48 / 4.53，AMI En→Zh 为 12.54 / 10.86。这一行是探索性结果，不计入 Holm。

## 消融表（`tab:ablation`，单栏）

- 内容为 R0 B → R1 → R2 → R3 PA Talker → R4 = MIST。
- 指标为 tcpASR-BLEU 和 EndOffset，数值与 v13 相同。
- 播放层不改变说话人标签，所以每一行的说话人归属都相同，消融不需要 cp 指标，也不需要 DER。

## 正文

- **摘要**：加一句"MIST 在每个集合的说话人归属语音 BLEU 上都领先 SeamlessStreaming、X-Translator 和同数据级联"。"Added one at a time" 一句改为明确指播放层的组成部分。
- **引言**：
  - 改正对 X-Translator 的描述。它是语音到语音系统，为每个跟踪到的说话人克隆音色，但只识别一路音频流；DiariST 只输出文本。
  - 贡献 (iii) 改为"与公开及同数据的多说话人 S2ST 系统对比，并相对同文本的截止基线逐组件消融"。
  - 贡献 (i) 改短。
- **实验设置**：
  - Systems 一段加入 X-Translator（含电平设置）和级联，各自指向附录；
  - Metrics 一段定义 cpASR-BLEU（会议级、cpWER 式）和 tcpASR-BLEU；
  - Statistics 一段加入 Table 1 的 Holm 校正族（在新增系统解码之前就已固定）；
  - Models、Data 两段压缩。
- **§5.1**：改写为与多说话人 S2ST 系统的对比。数值均已按 table1_v1.json 核对：
  - 公开系统的 tcpASR-BLEU 为 MIST 的 35–76%；
  - Seamless 落后 8.7–20.9，X-Translator 落后 4.3–12.3；
  - 级联在语音上与 B 持平（−0.7 至 +0.3），开始说话晚 0.3–0.4 s；MIST 比它高 1.3–2.0，LAAL 低 0.4–0.6 s。
- **结论**：加上"领先公开及同数据的多说话人 S2ST 系统"。
- **Limitations**：新增 "External systems" 一段：外部系统零样本运行在我们的检测窗口上、由我们的 diarizer 分配说话人；X-Translator 的增益由 dev 校准；外部系统只有一次 TTS 运行。

## 附录

- **G.4 X-Translator**：协议、能量门问题、dev 校准增益、原始运行的数值。
- **G.5 同数据级联**：
  - 不重标时文本 tcpBLEU 只有 10.9 / 24.4 / 11.4，说明说话人归属来自 diarizer；
  - 重标后，在重叠区间的文本低于 B：15.7 对 20.0，30.9 对 36.2。
- 已有的 cpASR-BLEU 定义和 DiariST 官方指标对比（v14 早先版本）保留。
- **DiariST 表**：加入 X-Translator 语音一行（test 上 SAg / SAt 为 16.04 / 13.66，MIST 语音为 20.25 / 18.61）。

## 版面

- 正文到第 8 页为止：结论在第 8 页 y=618，Limitations 从第 9 页开始。
- 全文 22 页，没有未定义的引用，也没有 overfull。

## 数据与代码

- `eval_outputs/mist_w3po_v1/joint/table1_v14/table1_v1.json`，由 `experiments/mist_w3po_v1/evaluation/table1_analysis.py` 生成，X-Translator 行按修订 2，`~` 前缀的行是探索性行。
- 表格：`scripts/make_tables_v14.py` 生成 `generated/v14/`：`tab_systems`、`macros_v14`、`switches` 等。
