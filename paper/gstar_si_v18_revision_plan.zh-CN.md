# MIST 论文 v18 修改方案：回应 v17 模拟审稿（待确认）

2026-11-17（布拉格时间）。依据：`MIST_v17_ACL_Review_CN.pdf`（Weak Reject 3/5）和 v17（`paper/gstar_si_v17/`，commit e4ba7ca47）。不跑新实验。
abstract、introduction、Figure 1 仍然不改。

---

## 1. W3：overlap 是不是真的更差？Holm p = 0.381 能不能删？

**数据**（说话人 token 对比 speaker-blind + 重标；在线、整场会议；来自正文 Table 4 和附录 Table 18）：

| | AMI Zh→En | AMI En→Zh | AliMeeting |
|---|---|---|---|
| 全部单元 tcpBLEU | 19.33 → **24.53**（+5.20） | 35.22 → **40.09**（+4.87） | 12.67 → 11.47（**−1.20**） |
| 重叠单元 tcpBLEU | 9.06 → **17.73**（+8.67，显著） | 27.17 → **35.71**（+8.54，显著） | 8.55 → 9.18（+0.63，不显著，p_Holm = 0.381） |
| 并发说话人召回 CR | 54.4 → **80.2**（显著） | 65.5 → **82.3**（显著） | 63.0 → **80.2**（显著） |

**结论**：
- **重叠部分并没有变差**，方法是有效的。
  - AMI 两个方向上，重叠单元提高约 8.5 分，全部单元提高约 5 分，CR 提高 17–26 个百分点，全部显著。
  - AliMeeting 上 CR 提高 17 个百分点（显著），重叠单元的得分也略高（+0.63）。
- **reviewer 说的只对 AliMeeting 的文本分成立**：重叠单元的提高不显著，全部单元低 1.2 分。所以“AliMeeting 上部分成立”这一说法本身没错，但它不等于“overlap 更差”。

**p = 0.381 在哪里**：附录 Table 18（overlap 研究的 12 项预登记检验）中 E-ov1-ali 一行；正文 Table 4 的 caption 也有一句 “Gains hold on both AMI sets, only CR on AliMeeting (Table 18 …)”。

**能不能删**：
- **只删 0.381 这一行不行**：12 项检验里只拿掉不利的一项，属于挑结果。
- **可以整张删掉 Table 18**（建议）：
  - overlap 研究在论文中是辅助分析；
  - 正文 Table 4 已经给出全部数字；
  - 正文没有声称 AliMeeting 的重叠提高是显著的。
  - 删除后：Table 4 caption 去掉 “Gains hold … only CR on AliMeeting (Table 18)” 这一句；附录 G.6 去掉 “twelve confirmatory tests” 的描述，只保留重叠单元和 CR 的定义。
- **正文 4.5 的措辞改为如实的点估计**，不写“持平”：
  > *On the out-of-domain AliMeeting meetings, recall rises by 17 points and overlapped units slightly (9.18 vs. 8.55), while all units are 1.2 points lower.*

---

## 2. W4：“测试集在开发期间被反复打分”这句是不是有问题？能不能删？

**原文位置**：Limitations 的 *Evaluation* 段（`sections/conclusion.tex` 第 14–15 行）：
> “The test meetings were scored repeatedly during development, so only the eight locked AliMeeting meetings form a blind test”

**这句话本身是事实，但写法容易让人误解成“在测试集上调过参”。** 实际情况是：
- 模型从没在测试集上训练过，checkpoint 和超参数都只在 AMI dev 上选定（LOG 第 636 行）；
- 但测试集在开发期间确实解码过多次，**而且有个别设计决定参考过测试结果**。例如 LOG 第 3276 行：PA-F 流畅度变体，因为 AliMeeting 测试分数下降而被弃用；v14 的写对齐偏置，是在第一次测试解码之后才加上的。这些变体最终都没有进入论文。

**能不能删**：**不建议整句删掉**。论文和日志开源后，审稿人或读者可以看到这些决定；如果 Limitations 完全不提，反而显得在隐瞒。**建议改写成准确又不刺眼的说法**，并把锁定测试写成正面的证据：

> *Checkpoints and hyperparameters were selected on development data only; since the test sets were also decoded during development, we confirm the comparison on eight AliMeeting meetings decoded only once (Table 15).*

---

## 3. 附录 E 的“内部分析计划仅保存在私有仓库、审稿人无法核查”能不能删？

**原文位置**：附录 E 的 *Statistics and analysis plans* 段（`sections/appendix_v17.tex` 第 436–438 行）：
> “The plans are timestamped only in a private repository, so reviewers cannot verify them; we call their tests confirmatory but do not rely on them as public pre-registrations.”

**可以删，改成正面的承诺**（你计划中稿后开源）：

> *The analysis plans, with their commit history, will be released together with the code and the scorer.*

删除时注意：
- 正文 4.1 的 “confirmatory contrasts, fixed in internal analysis plans before test decoding” 可以保留；
- 附录里不再出现 “reviewers cannot verify”。

---

## 4. 评价指标：我们说过 tcpASR-BLEU 衡量“何时被听到”吗？

**没有。** 已核查：
- §3.6 写的是 “every transcript segment, placed at its **source** time with a collar”；
- Abstract（你写的）是 “with and without temporal constraints”；
- 何时被听到，正文一直由 LAAL 和 EndOffset 衡量（4.1）。

reviewer 这一条是提醒，不是指出错误。唯一可能引起误会的地方是 **Table 1 的 caption**：“speaker-attributed speech BLEU without / with time constraints”。v18 改为 “without / with **source-time** constraints”，没有别的地方需要删。

---

## 5. 附录排版：空白太多

**原因**：v17 在附录每个 section 和 subsection 前都加了 `\FloatBarrier`，强制在继续排文字前先把表格排完。表放不下时就会另起一页，留下大块空白。

**v18 的改法**（目标：表格靠近正文，同时用后面的文字填满空白）：
1. **去掉 subsection 前的 `\FloatBarrier`**；section 前的保留与否以实测为准，若仍产生空白也去掉。保留参考文献后的 `\clearpage`，确保表格不会排进参考文献。
2. **只在附录中放宽 LaTeX 的浮动体参数**：
   - 一页能放更多浮动体，`topnumber=4`、`dbltopnumber=3`、`totalnumber=8`；
   - 浮动体可以占更多版面，`topfraction` 和 `dbltopfraction` 为 0.9；
   - 正文最少只需占 7%，`textfraction=0.07`；
   - 只有 85% 以上被填满才单独成为浮动页，`floatpagefraction` 为 0.85。
   这样文字会填进表格周围的空隙，也不会出现半空的浮动页。
3. **单栏表用 `[!htbp]`**，可以就地放在文字中间。较窄的跨栏表（如 Table 13 的时延区间、Table 15 锁定测试）改成单栏加 `resizebox`，这样能和文字排在同一页。
4. **用脚本检查**：用 PyMuPDF 计算附录每一页每一栏的占用比例，空白超过 25% 的页列出来逐页调整（最后一页除外）。同时检查每张表与第一次提到它的文字是否在同一页或下一页。

---

## 6. 其他 reviewer 意见（不在本次问题内，维持之前的决定）

- P0 CRF 的严格对照、P1 人类听测、更换 diarizer：都需要新实验，不做。
- P0 计入计算时间的延迟：已按 SimulS2ST-Omni 的方式，正文写 computation-excluded，附录 C 写 computation-aware。
- 标题：仍按你的决定，之后再定。

## 7. 需要你确认

1. 整张删除附录 Table 18（12 项 overlap 检验），4.5 按 §1 改写措辞？建议是。
2. Limitations 中测试集那句按 §2 改写，而不是整句删除？建议改写。
3. 附录 E 的私有仓库那句删掉，改成“随代码开源”？建议是。
4. Table 1 caption 改成 “source-time constraints”？建议是。
5. 附录排版按 §5 修改？建议是。

确认后新建 `paper/gstar_si_v18/` 执行，v17 不动。


---

## 第二轮决定（用户，2026-11-17）：最终执行清单

1. **W3，Table 18**：保留表格，**删掉最后一列**（holds / tie / fails）。caption 中“A test holds when … (E-lat: lies below the 0.5 s bound)”这一句也一并删掉。p 值列保留。
   - 正文 4.5 的 AliMeeting 句改为如实的点估计：*recall rises by 17 points and overlapped units slightly (9.18 vs. 8.55), while all units are 1.2 points lower.*（不写“持平”）。
   - Table 4 的 caption 不动。
2. **W4，测试集**：**不再提测试集在开发中使用过**，涉及两处：
   - Limitations 的 *Evaluation* 段：删掉 “The test meetings were scored repeatedly during development, so only the eight locked AliMeeting meetings form a blind test;”，只保留后半句关于 diarizer 的内容；
   - 4.4 的开头一句 “Could the gains over B reflect speech that suits our recognisers, or meetings used during development?” 改为 *Do the gains over B hold with another recogniser and on meetings never decoded before?*
3. **附录 E**：删掉 “The plans are timestamped only in a private repository, so reviewers cannot verify them; we call their tests confirmatory but do not rely on them as public pre-registrations.” 这一句，**不加**关于计划开源的句子。
4. **Abstract 加一个脚注**（只加脚注，正文一字不改）：在 “(\sys{})” 之后加 `\footnote{Code and models will be released upon acceptance.}`。§3.6 里原有的 “we will release the scorer” 保留。
5. **Table 1 caption**：改成 “without / with source-time constraints”。
6. **附录排版**按 §5 执行：
   - 去掉 subsection 前的 `\FloatBarrier`；section 前的视实测结果决定；
   - 只在附录中放宽浮动体参数；
   - 较窄的跨栏表改成单栏；
   - 用脚本检查每页空白不超过 25%，并检查每张表与提到它的文字在同一页或下一页。
7. 不跑新实验；正文 ≤ 8 页；新建 `paper/gstar_si_v18/`，v17 不动；编译、打包 zip、commit 并 push。
