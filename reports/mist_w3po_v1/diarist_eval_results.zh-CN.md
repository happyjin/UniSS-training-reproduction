# DiariST 官方评测（SAgBLEU / SAtBLEU）：我们的 AliMeeting 输出

- 2026-11-16 18:30 布拉格，只用 CPU。登记：`multispk_s2st_baselines_prereg_v2.zh-CN.md` §5。
- 代码：`experiments/mist_w3po_v1/evaluation/diarist_eval.py`。
- 结果：`eval_outputs/mist_w3po_v1/diarist_eval/diarist_eval_v1.json`。参考、假设 TSV 和 eval.py 日志在同一目录下。

## 1. 官方代码与参考

- **代码**：DiariST 仓库 github.com/Mu-Y/DiariST，commit `2a95bfbdbcd0722606c5f126f4af711c0b6602d5`（2023-09-19），只读克隆在 `research_sources/diarist_20261116/DiariST`。
- **打分**：用官方 `diarist/scoring/eval.py`，代码不改。
  - 环境是 venvs/s2st_baselines（sacreBLEU 2.6.0，含 fire）。仓库和论文锁定的是 2.3.1，本机没有，按项目规则也不安装。
  - 用 uniss-eval 的 sacreBLEU 2.5.1 抽查了 4 个运行，两位小数完全相同。
- **参考的构建**：
  1. 用官方 `generate_data_json.py`（不改）读两份文件：DiariST 人工英文译文 `data/diarist_alimeeting_refs_v1/{test,dev}.txt`（sha256 与 SOURCE*.md 一致），以及官方小会话列表 `diarist/data/{test,dev}.json`。
  2. 每个小会话的参考 JSON 按 `generate_diarist_alimeeting.gen_ihm_mix` 写出：fix_timing，文件名为 `{session}-{start}-{end}.json`。只是不写音频，因为打分用不到。
  3. 三种条件（IHM-MIX、IHM-CAT、SDM）的参考文本和顺序相同。
- **核对**：
  - **dev**：本机有 AliMeeting Eval 原始近讲音频。直接运行官方 `gen_ihm_mix`（不改），得到的 76 个 JSON 与我们的**逐字节相同**。
  - **test**：本机没有 Test_Ali 原始包，不过参考不依赖音频，用的是同一条代码路径。
  - **规模**：
    - test 有 195 个小会话，来自 20 场会议，共 16,358 句。每句恰好落在一个小会话里，没有句子丢失或重复。按空白切分共 138,524 词。小会话长 180.0–358.7 s，合计 10.77 h，其中相邻小会话最多重叠 4.15 s。
    - dev 有 76 个小会话，来自 8 场会议，共 6,454 句、42,136 词，小会话长 180.1–352.5 s，合计 4.20 h。
    - 与论文一致：20 场 / 10 h / 195 个，8 场 / 4 h / 76 个，每个 3–6 分钟。README 写的 "Found 195 files" 也对得上。仓库和论文都没有给句数或词数。
  - **管线自检**：把参考本身当作假设输入，test 和 dev 的 SAgBLEU 与 SAtBLEU 都是 100.00。
  - **音频**：官方 IHM-MIX dev 的 76 段，与我们 near_mix 会议音频同一时段的相关系数都是 1.0000，长度也相同。唯一的差别是每场会议有一个整体增益：我们的音频峰值超过 0.9 时会整体缩小，两边的增益比在 1.00–1.16 之间。

## 2. 假设的转换（规则事先写定）

- **时间**：每段的会议时钟 = 窗口 start_s + 段 start_s，与 cp_asr_bleu 和 score_ext 相同。
- **排序与说话人标签**：按会议时钟的起点排序，起点相同时保持原顺序。说话人标签用会议级的 `spk<k>`。
  - Thinker 的说话人 token 由 B 和 MIST 共用；Seamless 的标签是因果重标。
- **TSV 格式**：每段一行，列为 speaker、start、end、text，时间用小会话时钟。
- **小会话归属**：每段归入**起点所在的小会话**。官方相邻小会话有重叠；起点落在重叠区时归后一个，即按 [start_i, start_{i+1}) 划分。
  - 不丢任何段。起点晚于最后一个小会话结束的段归入最后一个（只有 Seamless 有 1 段）。
  - 落在重叠区的段：test 文本 65 / 32,923，语音约 37 / 7,500，Seamless 54 / 5,933；dev 文本 37，语音 24。
  - 敏感性：改为"归前一个"后，各系统的变化都 ≤ 0.09 BLEU（见 JSON 的 `earlier`）。
- **空小会话**：每个小会话都写一个文件，没有段就写空文件，所以 195 / 76 个全部参与计分。

## 3. 结果（IHM-MIX，人工参考）

- 语音行是两个 TTS 种子的 ASR 转写取均值，括号内是两个种子各自的值。外部系统只有一次运行。

**test（195 个小会话）**

| 系统 | 输出 | SAgBLEU | SAtBLEU |
|---|---|---|---|
| B / MIST（Thinker 文本，二者相同） | 文本 | **22.38** | **21.06** |
| MIST（B Thinker + 播放层，rlplem） | 语音转写 | 20.25（20.24 / 20.27） | 18.61（18.62 / 18.60） |
| B（T1c Talker） | 语音转写 | 19.02（19.07 / 18.96） | 17.38（17.45 / 17.32） |
| SeamlessStreaming（零样本，因果重标） | 语音转写 | 9.28 | 7.97 |
| X-Translator | 语音转写 | 打分时还没有输出（见 §5） | |
| *DiariST 论文：t-SOT ST + t-vec（流式，DiariST）* | 文本 | *16.62* | *14.76* |
| *DiariST 论文：t-SOT ST + d-vec* | 文本 | *16.62* | *13.85* |
| *DiariST 论文：1spk ST（+d-vec / +t-vec）* | 文本 | *16.15* | *13.39 / 13.22* |
| *DiariST 论文：Whisper-small 离线 ST→SD* | 文本 | *13.68* | *12.10* |
| *DiariST 论文：Whisper-small 离线 SD→ST* | 文本 | *10.40* | *9.31* |

**dev = AliMeeting Eval（76 个小会话）**

| 系统 | 输出 | SAgBLEU | SAtBLEU |
|---|---|---|---|
| B / MIST（Thinker 文本） | 文本 | 15.26 | 13.74 |
| MIST（rlplem） | 语音转写 | 14.54（14.46 / 14.61） | 12.78（12.72 / 12.83） |
| B（T1c） | 语音转写 | 12.50（12.55 / 12.45） | 10.95（11.03 / 10.86） |
| *DiariST 论文：t-SOT ST + t-vec* | 文本 | *16.05* | *13.91* |
| *DiariST 论文：t-SOT ST + d-vec* | 文本 | *16.05* | *13.15* |
| *DiariST 论文：1spk ST（+d-vec / +t-vec）* | 文本 | *14.73* | *12.01 / 12.00* |
| *DiariST 论文：Whisper-small ST→SD / SD→ST* | 文本 | *14.48 / 10.71* | *11.96 / 9.21* |

- 论文 dev 数字的来源：arXiv:2309.08007v2 HTML 版 Table 1/2。test 数字与已核对的数字逐项一致。dev 数字引用前请再对一次 PDF。
- **解读**：
  - **test**：我们的 Thinker 文本比 DiariST 高 5.8 SAgBLEU、6.3 SAtBLEU。MIST 的**语音**转写也比 DiariST 的**文本**高 3.6 / 3.9。
  - **dev**：Thinker 文本低于 DiariST（SAgBLEU 15.26 对 16.05，SAtBLEU 13.74 对 13.91），MIST 语音也更低。
  - 所以不能写"全面超过 DiariST"，只能说"在 DiariST-AliMeeting test 上更高，dev 上相当或略低"。
  - 我们自己的 tcpBLEU 也是 dev 明显低于 test（文本 14.58 对 21.43），dev 本身更难。
  - MIST 语音在两个集合上都高于 B 语音：test +1.2 / +1.2，dev +2.0 / +1.8。方向与论文 ASR-BLEU 的结论一致。

## 4. 可比性说明（引用时必须写明）

1. **解码单位不同**：我们对整场会议做在线解码（按检测到的语音窗口，跨小会话保留上下文），事后按段起点把输出切进官方小会话；DiariST 是逐个小会话独立解码。
2. **输出与延迟**：DiariST 只输出文本，没有报告延迟。我们的文本行是同传的 Thinker 输出；语音行是合成语音的 ASR 转写，比文本多一层 TTS 和 ASR 的损失。
3. **相同的部分**：
   - 输入都是 IHM-MIX（各近讲话筒之和），与官方逐样本核对过，只差一个整体增益；
   - 参考都是 DiariST 人工英文译文，包括语气词句（如 "Um."）。我们的 tcp/cp 口径会排除语气词句，这里不排除；
   - 打分都用官方 eval.py（sacreBLEU 默认 13a，小会话内做说话人置换，语料级 BLEU）。
4. **训练数据**：DiariST 在 AliMeeting 训练集（GPT-4 译文）上训练，属于域内；我们从未在 AliMeeting 上训练或调参，它只作测试。两边模型规模也不同。
5. **sacreBLEU 版本**：2.6.0，官方锁定的是 2.3.1；2.5.1 与 2.6.0 结果相同。

## 5. 待补

- **X-Translator**：解码目录 `jtest_ali_xtranslator_default` 在打分时（18:28 布拉格）还不存在。有了之后执行：
  ```
  python -m experiments.mist_w3po_v1.evaluation.diarist_eval extra --sub test --key X-Translator:asr \
      --run jtest_ali_xtranslator_default --hyp hyp_asr.jsonl
  ```
  结果写到 `diarist_eval_v1_extra_jtest_ali_xtranslator_default.json`，不会覆盖已有文件。
- **附带结果**（`extra` 的功能测试）：MIST Talker 的输入是合并后的读出段（`_rl`），同一份文本在 test 上为 22.51 / 21.06。与未合并文本的 22.38 / 21.06 只有拼接顺序上的差别。
