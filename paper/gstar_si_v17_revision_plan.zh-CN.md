# MIST 论文 v17 修改方案（第二轮，回应用户对 v16 模拟审稿的决定）

2026-11-17（布拉格时间）。基于 v16（`paper/gstar_si_v16/`，commit 9f843334d）。**不跑新实验。**
abstract、introduction、Figure 1 仍然不改；“first” 的声明不改，也不收窄。

---

## 1. 把最能证明 CRF 有效的结果写进正文（用户已同意）

**证据**：已有的 test 析因实验（`factorial_test_v1.json`），在同一积压语速规则下比较 PA Talker 和 base Talker：

| 对比 | tcpASR-BLEU（Z→E / E→Z / Ali） | EndOffset（s） |
|---|---|---|
| 各只采样一个（R3b − R2） | +0.29 / +0.02 / +0.33 | −0.18 / −0.15 / −0.22 |
| 都做二选一（R4 − R3a） | +0.07 / −0.03 / +0.27 | −0.18 / −0.14 / −0.22 |

- EndOffset 在 6 个对比中**全部 p<0.001**。
- tcpASR-BLEU 没有任何一处显著变差；AliMeeting 两处都显著提高，AMI Zh→En 在只采样一个时也显著提高。

**修改**：
- **Table 5（ablation）**：在 R2 之后加一行 “R3a R2 + best of two（base Talker）”，数字为 20.28 / 32.26 / 17.81 与 3.30 / 3.04 / 3.44（取自同一 JSON，四舍五入）。表格成为语速规则固定下的 2×2：{base, PA} Talker × {一个样本, 二选一}。
  - 每列加粗最好值时，En→Zh 一列的 tcpASR-BLEU 最好值是 R3a（32.26，高于 MIST 0.03，不显著）。如实加粗。
- **Table 5 caption** 加一句：*With the rate rule and selection fixed, the PA Talker lowers EndOffset on every set ($p<0.001$; Table~\ref{tab:sig5})*。显著性只写在 caption 里，不在单元格里加星号。附录新增一张小表，列出这 6 个配对检验（差值、95% 区间、p），数据来自同一 JSON 的 `paired`。
- **4.5 Playback-layer components 段**，改写为：
  > *With the rate rule and the selection fixed, replacing the base Talker by the CRF-trained PA Talker ends speech 0.14–0.22 s earlier on every set, with one sample or with best of two, and never lowers tcpASR-BLEU; on AliMeeting it also raises it. Best-of-two selection, in turn, adds quality without changing EndOffset: the CRF supplies the latency gain that quality-based selection does not, as the backward messages of Eq.~(11) predict.*
  - 措辞是 “the CRF-trained Talker”，不写 “the CRF is necessary”。这组对比证明的是“优于 base Talker”，而不是“优于其他微调方式”。

---

## 2. 计算时间（1.20 s / 52.8 s）：SimulS2ST-Omni 有没有类似问题，它怎么处理的

### 2.1 SimulS2ST-Omni 的做法（读了论文 v2 和开源代码 `github.com/hasaki321/SimulS2ST-Omni`；README 写明已被 EMNLP 2026 Main 接收，论文正文中没有写会议名）

- **用词**：标题、README 用 “Streaming”（README 中 streaming 14 次，simultaneous 1 次）。论文正文两种都用（streaming 82 次，simultaneous 27 次），小节标题就叫 *Sentence-Level Simultaneous Translation*、*Long-form Simultaneous Translation*、*Computation-Aware Simultaneous Evaluation*。也就是说，它把 streaming 和 simultaneous 当作同义词。
- **它也有类似问题，而且没有回避**：
  - 正文的主结果用的是**不计计算时间**的 LAAL / StreamLAAL，正文用一句话指向附录：*“This persists under CA/RTF analysis … (App. F)”*。
  - LiveInterpret 2.0 的时延用它论文报告的 LAAL，并加脚注说明这是为了避开服务端排队。
  - **附录 F “Computation-Aware Simultaneous Evaluation”** 报告了：
    - 计算感知的 LAAL_CA 和 StreamLAAL_CA：计算时间平均让时延再增加约 2 s；
    - 实际计算的 RTF：**在最低时延档上 RTF > 1**（En→Zh m1 为 1.54 / 1.86，即跑不了实时），并明确写出“从 m4 档开始 RTF < 1，才可用于真实场景”。
  - 开源代码中评测命令直接带 `--computation-aware`（SimulEval 同时输出不计计算时间和计计算时间两种时延），另外提供了实时的 WebSocket 演示。
- **结论**：它的情况和我们相同（主结果不计计算时间，部分设置跑不了实时），处理方式是**正文用不计计算时间的时延，并用一句话交代；附录如实给出计算感知的时延和 RTF，并说明在哪些设置下能实时**。它没有删这些结果，仍然被接收。

### 2.2 v17 的做法（按 SimulS2ST-Omni 的先例，建议不删数字）

- **4.1 Metrics**：*We report computation-excluded (non-computation-aware) latency, as SimulEval does by default: LAAL and EndOffset count audio and playback time but not computation; a computation-aware replay is in Appendix~C.*
- **Table 1 caption**：写 “LAAL (↓, computation-excluded)”。
- **附录 C**：*Computation* 段改名为 *Computation-aware analysis*。保留数字，但写成 SimulS2ST-Omni 的风格：
  - 每秒音频的计算时间（RTF）：Thinker 0.65 / 0.80 / 1.20，PA Talker 0.18 / 0.26 / 0.31；
  - 原因：我们的实现每次读都重新编码前缀，没有 KV cache；
  - 结论：AMI 上的 Thinker 快于实时；AliMeeting 的会议很长，前缀重新编码使它慢于实时；实时部署需要增量编码，留作后续工作。
- **Limitations**：一句话，*Latencies exclude computation; with prefix re-encoding, our implementation is slower than real time on long meetings (Appendix C).*
- **若你仍然要删数字**：可以删掉附录 C 的具体数值，只保留 4.1 和 Limitations 中各一句。但 SimulS2ST-Omni 的先例说明，**保留反而更稳**：审稿人看到作者主动给出计算感知分析，通常比自己去追问更认可。

---

## 3. 说话人相似度代理指标：不做（用户决定）

正文没有把“保留每位说话人的音色”作为主要优势来突出，所以不做这项指标。Table 2 中现有的 SIM 列（与 voice prompt 比较）保持不变。

---

## 4. 停顿：删除（用户决定）

需要删掉的位置：
- **正文 Table 2**：删掉 pauses 列；
- **4.3**：删掉末句 “the cost is more pauses between blocks …”；
- **Limitations 的 *Method* 段**：删掉 “Faster speech brings more pauses …”；
- **附录 Table 12**：删掉 “pauses / min” 下的两列（≥0.5 s、mid-sent.）；
- **附录 G.2**：删掉 “pauses become two to three times as frequent”。

（我此前建议保留一句不带数字的说明，记录在 v17 证据文件 §3 中；这里按你的决定执行。）

---

## 5. 附录 Table 18 的 offline 两行：为什么可以删，以及删后的连带改动

**理由**：
1. offline Thinker 不属于 MIST，MIST 是同传系统；
2. Table 19 的 12 项预登记检验全部针对 online 行，offline 行不是任何检验的对象；
3. offline 行在参考窗口上解码，与论文其他部分整场会议在线解码的协议不同；
4. Table 18 的 online 两行与正文 Table 4 **完全相同**。

**删后的连带改动**：删掉 offline 两行（包括 offline 的 “Cascade, same data”）后，Table 18 就只剩与 Table 4 重复的内容。所以：
- **整张 Table 18 都删掉**，保留 Table 19（12 项检验）和 G.6 中的指标定义；
- G.6 中分析 offline 重标的那段（“Relabelling the offline Thinker's own output …”）一并删掉。

**这不是必须的**：放在附录里不占正文。删掉的理由是范围（与论文要证明的系统无关），而不是因为结果不利。若你想保留预登记研究的完整性，也可以不删。

---

## 6. 附录的图表排版（新问题）

**现象**：Table 6 排进了参考文献中间；很多附录表堆到了最后几页，不在引用它们的文字旁边。

**原因**：
- 浮动体用的是 `[t]`，LaTeX 会把它推到下一个能放下的页面顶部；
- `table*`（跨两栏）只能放在下一页的顶部；
- 附录开始前没有分页，所以附录的表会漂进参考文献。

**改法**：
1. **在 `\appendix` 前加 `\clearpage`**：附录从新的一页开始，表格不会再进入参考文献。
2. **用 `placeins` 宏包**，在附录每个 `\section` 结束处加 `\FloatBarrier`：每节的表必须在本节内排完，不会漂到后面的节。
3. **放置参数**：单栏表改为 `[!htbp]`；宽表能缩到单栏的就缩成单栏（加 `resizebox`），实在需要跨栏的用 `table*[!t]`，并放在引用它的段落之前。
4. **每个 `\input` 放在第一次引用它的段落之后**；比如 Table 6 放在 A 节第一段之后。
5. **编译后逐个检查**：每张附录表都与第一次提到它的文字在同一页或下一页；没有表跑到参考文献里。

---

## 7. 页数

- **增加**：
  - Table 5 新增一行，caption 加一句；
  - 4.5 的改写；
  - 4.1 加一句；
  - 合计约 +4 行。
- **减少**：
  - 删掉 4.3 和 Limitations 中关于停顿的句子，约 −3 行；
  - Table 2 少一列（行数不变）。
- 预计正文仍在 8 页左右。若超出，再精简 4.5 或 caption 的措辞。

## 8. 执行步骤（确认后执行）

1. 新建 `paper/gstar_si_v17/`，从 v16 复制过去，v16 不做任何修改。
2. 依次修改：
   - 第 1 节：Table 5 加一行；附录新增配对检验表；改写 4.5；
   - 第 2 节：4.1、Table 1 caption、附录 C、Limitations；
   - 第 4 节：删除停顿；
   - 第 5 节：若确认，删除 Table 18 和相应段落；
   - 第 6 节：附录排版。
3. 生成脚本 `make_tables_v17.py`：所有数字仍然从 JSON 读取并核对。
4. 编译，检查：
   - 正文 ≤ 8 页；
   - 没有未定义引用、没有溢出；
   - 附录每张表都在引用处附近；
   - 全文 grep 不到 distil 和 pauses。
5. 打包 zip，commit 后 push。
