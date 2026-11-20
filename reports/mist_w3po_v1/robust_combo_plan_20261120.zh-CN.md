# 鲁棒单说话人同传：完整系统组合方案（2026-11-20 16:45 CET 起草，写在任何组合数字之前）

本文件只定组合阶段的做法、组合矩阵、自检和判定草案，并定第十五波发现的"CV-LA 增益是否只是晚提交"的等延迟对照。脚本已写好，只做过 `bash -n` 和 DRY_RUN，没有跑任何 GPU 作业。第十四到十七波的判定以各自预登记为准，本文件不改它们。

- **启动条件：** 某个组件进不进组合，要等它那一波按预登记的两条主判定判完。
- **待定项：** 第 5 节列出的决定点在启动前确认；确认后在本文件末尾追加"启动记录"，写在任何组合数字之前。

## 1. 组件与记号
| 记号 | 组件 | 来源 | 状态（16:45） |
|---|---|---|---|
| L | UniST-ERM-WiSE Thinker：UniST LoRA，α = 0.5 | 第十四波 | 训练中 |
| N | NCWI：按每条输入前 1 s 估出的 b̂ 落在哪个箱，选 α*（LoRA 缩放），取代固定的 α = 0.5；以 L 为前提 | 第十六波 | 等第十四波 |
| C | CV-LA：增强视角作为一致性见证的提交规则 | 第十五波 | 组 1（底座）通过；组 2（unist）等第十四波 |
| E | 感知层 LoRA（enc_wise05） | 第十七波 | 训练中 |
| V | V1 音色 | 第三、五波 | 已是主系统组件 |
| T | T1c export_1500 Talker | 基线 | 固定 |

## 2. 新代码与自检
### 2.1 新代码
都在 `experiments/mist_w3po_v1/voicecond/`。

- **`combo_si_thinker.py`：** 逐行复制 wave15_si_thinker_cvla.py（它本身是 si_thinker_stream la2 / w1 的逐行复制），两个原文件都不改。在此基础上加了两个互相独立的开关。
  - `--second-view none | enh | same | budget`：
    - none：纯 LA2；
    - enh：CV-LA，与第十五波 `--view1 enh` 相同；
    - same：自检，与第十五波 `--view1 same` 相同；
    - budget：等延迟对照，见第 4 节。
  - `--assign <wave16 assign_<dir>.jsonl> --lora-root <wave16/lora>`：NCWI 的 vLLM 多 LoRA，LLM 参数与 wave16_si_thinker_ncwi.py 相同（enable_lora、max_loras 8、rank 16、qkv_proj/o_proj）。
    - 第十六波按"音频前 1 s 的 sha1"查 α*。这里改为**按条目 id 查 α\***，因为同一条的增强视角音频不同，按音频查会查不到。
    - 结果是同一条的两个视角用同一个适配器；α* = 0 时不挂 LoRA。
  - 两个开关都不开时，就是在 `--thinker` 指定的模型上跑 LA2；该模型可以是底座，也可以是合并后的副本。
  - 内置断言：
    - d1 == d0 时，提交量等于 LA2 的提交量；
    - budget ≥ LA2 一致前缀时，提交量等于 LA2 的提交量。
- **`combo_selfcheck.py`：** 口径与 wave15_selfcheck、wave16_tools agree 相同。
  - 统计量：逐条 blocks（write_ms、text、final）完全一致的比例。
  - 门槛：第五波三次底座运行两两一致率中的最小值，由脚本重算（英→中 0.9035，中→英 0.9095）。
  - 已在现有文件上复算第十五波的自检：英→中 0.9286、中→英 0.9303、PASS，与 `wave15/selfcheck.json` 一致。
- **`combo_tools.py`：** 如果第十六波最终走的是合并模型回退路径（NCWI_ROOT = eval_ncwi_merged），用它的 `groups.txt` 拆分作业（每个 α* 一个作业），跑完后用 `assemble` 按 S1 顺序拼回。
- **`combo_orch.sh`：** 每次调用跑一个臂，链条与第五波、第十四波的 combo 相同：
  1. Thinker：新跑，或者用 THINKER_FROM 复用已有输出；
  2. 原音色 Talker：新跑，或者用 RAW_FROM 复用；
  3. V1 token 加 V1 Talker：T1c，4 个 shard，种子 20261021；
  4. 论文协议的 ASR；
  5. SQUIM PESQ 和 TitaNet 相似度。
  - 其余约定：
    - 每个臂有自己的 PIDS、STOP 文件和 watcher（`combo_noplaceholder.sh`）；
    - 每次启动前，先按 PID 停掉该卡上的占位；
    - 不碰 guard 2726951、gpu_watch 52045、sync 1222453；
    - 臂根目录已存在时拒绝运行。
  - 可选开关：THINKER_ONLY=1 用于自检臂；NO_V1=1 只跑原音色 Talker 和 ASR，用于对照臂。
- **`combo_score.py`：** 统计量、单元和 bootstrap 与 wave3_v_score.compare 相同；另加 Thinker text BLEU，用同样的单元 bootstrap。
  - 已验证：把第五波的 ew_v1 冒充成 full 臂，复算出的 ew_v1 对 B 的 60 个数字（ASR、PESQ、相似度、D1、D2）与 `wave5/M/combo/summary.json` 逐个相同。
  - cvla_base 的 text 差也与第十五波 summary_g1 相同，例如中→英 a 带噪 +2.998 [+1.308, +4.783]。

### 2.2 自检臂
只跑 Thinker，每个约 5 分钟 × 2 卡；在全系统臂之前跑。

| 臂 | 配置 | 对照 | 何时跑 |
|---|---|---|---|
| SC-N | none + NCWI LoRA | wave16/eval_ncwi | 仅当第十六波走了 LoRA 路径 |
| SC-C | enh，`--thinker wave14/merged_unist_wise05` | wave15/eval_cvla_unist | 第十五波组 2 完成后 |

- 两项都 PASS（达到上面的门槛），才跑全系统臂。
- 有一项 FAIL 时，先跑 SC-0（none，底座，对照三次底座运行）定位问题，不改门槛。

## 3. 组合矩阵（dev：S1 (a)/(b) × 干净/带噪，两个方向）
| 臂 | Thinker | 音色 | 来源 | 新增计算 |
|---|---|---|---|---|
| **full** | N + C | V1 | 新跑 `SV=enh NCWI=1` | Thinker、原音色 Talker 和 V1 Talker、ASR、指标 |
| full_b（−V1） | N + C | 原音色 | 同上（full 臂内） | — |
| noN（−N） | L(0.5) + C | V1 | 复用 wave15/eval_cvla_unist | V1 Talker、ASR、指标 |
| noC（−C） | N | V1 | 复用 wave16/`$(cat NCWI_ROOT)` | 同上 |
| noL（−L，因而也 −N） | 底座 + C | V1 | 复用 wave15/eval_cvla_base | 同上 |
| L（只有 L + V1） | L(0.5) | V1 | 复用 wave14/eval_unist_wise05 | 同上 |
| ctrl_budget_base | 底座 + 等延迟对照 | 原音色 | 新跑 `SV=budget BUDGET_FROM=wave15/eval_cvla_base NO_V1=1` | Thinker、原音色 Talker、ASR |
| 参照 B | 底座 LA2 | 原音色 | wave2_S1 | — |
| 参照 ew_v1 | ERM-WiSE | V1 | wave5/M/combo | — |

- L 臂的链条与 `wave14_combo_orch.sh` 完全相同，只换输出目录。**建议用 `combo_orch.sh ARM=L` 代替 `wave14_combo_orch.sh`**，这样所有臂的目录结构和打分一致；第十四波预登记的"组合"判定改用 combo_score 的 L_v1 对 B / 对 ew_v1。属于决定点 D5。
- 启动示例：
  ```
  ARM=full SV=enh NCWI=1 CARDS="0 1 2 3" setsid nohup bash experiments/mist_w3po_v1/voicecond/combo_orch.sh > logs/mist_w3po_v1/combo/full_orch.log 2>&1 < /dev/null 8>&- 9>&- &
  ARM=noN THINKER_FROM=eval_outputs/mist_w3po_v1/wave15/eval_cvla_unist RAW_FROM=eval_outputs/mist_w3po_v1/wave15/eval_cvla_unist CARDS="4 5" ...
  ARM=ctrl_budget_base SV=budget BUDGET_FROM=eval_outputs/mist_w3po_v1/wave15/eval_cvla_base NO_V1=1 CARDS="4 5" ...
  ```
- 打分：
  ```
  combo_score --arm full=... --arm noN=... --arm noC=... --arm noL=... --arm L=... --arm ctrl_budget_base=... --arm cvla_base=eval_outputs/mist_w3po_v1/wave15/eval_cvla_base --ablate noN --ablate noC --ablate noL --ablate L --pair cvla_base_b:ctrl_budget_base_b --primary full
  ```

### 3.1 卡时估计
按今天的日志估：第十五波 cvla_base 链 15:56–16:21，第五波 combo 06:27–06:52。

| 臂 | 卡 | 墙钟 | 卡时 |
|---|---|---|---|
| SC-N、SC-C（各） | 2 | 约 5 min | 约 0.2 |
| full（Thinker 约 5 min；原音色和 V1 两轮 Talker 约 16 min；ASR 约 14 min；指标约 4 min） | 4 | 约 40 min | 约 2.7 |
| noN、noC、noL、L（各；V1 Talker 约 7 min，ASR 约 14 min，指标约 4 min） | 2–4 | 约 26 min | 约 1.0–1.5 |
| ctrl_budget_base（Thinker 约 3 min，Talker 约 8 min，ASR 约 14 min） | 2 | 约 25 min | 约 0.85 |
| 打分（CPU，每次） | — | 约 2–5 min | — |

- 合计约 9 卡时。8 张卡并行时约 1 小时墙钟（full 在 0–3，其余臂在 4–7）。
- NCWI LoRA 路径的 Thinker 若明显变慢，以第十六波 eval_ncwi 的实际用时为准。
- ctrl_budget_base 只依赖已完成的第十五波组 1，卡 4–5 一空出就可以跑，不必等组合。

## 4. CV-LA 的等延迟对照（预先写定，一个设置，不扫参）
### 4.1 问题
- 第十五波组 1 中，CV-LA 的增益伴随 D1 增加 0.25–0.60 s。
- LA2 本身就有"更晚提交、BLEU 更高"的曲线。论文 results.json 里 RealSI 上 LA2 读窗 m 的曲线：
  - 英→中：m = 1.0 为 29.75 / 2.51 s，1.1 为 30.14 / 2.76 s，1.25 为 30.15 / 3.03 s；
  - 中→英：m = 1.0 为 23.19 / 2.70 s，1.25 为 24.05 / 3.03 s。
- 这条曲线在测试集上，只作背景，不用于任何选择。
- 按这条曲线，+0.25–0.5 s 约值 +0.4–0.9 BLEU，小于 CV-LA 带噪单元的 +1.7–2.9。但它是干净的测试数据，dev 带噪上没有对应曲线，所以必须在 dev 上做一个对照。

### 4.2 对照：按预算截断的 LA2（`--second-view budget`）
- **规则：** 同一个 Thinker、同样的读窗，不加第二个视角。第 t 次读时提交 d0[n : min(A_t, B_t)]。
  - A_t 是 LA2 自己的一致前缀；
  - B_t 是 CV-LA 参照运行（wave15/eval_cvla_base）在同一条、同一次读之后累计已提交的单元数，取其 drafts 里 cv_new 的累计和；
  - 读到句尾时提交 v0 的完整草稿，与 LA2 相同。
- **延迟为什么可比：**
  - 对照每次读累计提交的单元数不超过 CV-LA，提交时刻的分布由同一份预算决定；
  - 唯一的差别是在哪里截：CV-LA 截在跨视角不一致处（有信息），对照截在预算用完处（无信息）；
  - 对照还可能因自身一致前缀更短而提交得更少，所以它的延迟只会等于或略高于 CV-LA。这让对照偏保守：CV-LA 若在延迟不高于对照时仍显著更好，就支持"跨视角信息"。
- **为什么不用别的方案：**
  - LA3（三次草稿的公共前缀）系统性地多等一个读窗（约 1 s），延迟与 CV-LA 不可比；
  - 加大读窗 m 要先选 m，等于扫参；
  - 伪第二视角（v0 截短或加抖动后作见证）有自由参数（截多少、抖多大）。
  - 预算对照没有参数，一次运行，只依赖已完成的第十五波组 1。
- **实现：** `load_budget()` 已在现有的 cvla_base 输出上检查过：英→中 1274 条、中→英 2012 条，每条的非末次读与 range(1000, dur, 1000) 一一对应。

### 4.3 判定（写定）
比较 cvla_base_b 对 ctrl_budget_base_b，单元同第十五波。

- **"跨视角信息"成立，需要同时满足：**
  1. 至少一个带噪单元 text 差的 CI 下界 > 0；
  2. 这些单元的 D1 差（CV-LA − 对照）≤ +0.10 s；
  3. 没有任何单元的 text 或 ASR 点估计 < −0.3 且 CI 上界 < 0。
- 否则记为"CV-LA 的增益主要由更晚提交解释"。CV-LA 仍可按第十五波的判定进入组合，但论文里不能把增益归因于跨视角信息。
- 辅助：两臂提交后被参考否定的比例，用 wave15_commit 的口径。

## 5. 需要确认的决定点
- **D1 第十七波通过时：** 底座改为 enc_wise05，即只在音频编码器加 LoRA 的合并副本，并用 `BASE=<该副本>`。
  - NCWI 的文本层 LoRA 照常按 id 挂在上面，沿用第十六波拟合的 α*，不重新拟合；
  - 这一组合只在 LoRA 路径下有定义，合并回退路径下脚本拒绝运行；
  - α* 是在不含编码器 LoRA 的端点上拟合的，这是一个预先写定的近似，要在论文中说明；
  - 备选：不叠加，把 E 单独作为另一个 full 变体臂，多约 2.7 卡时。
  - **建议叠加**，另跑一个 −E 臂（即本矩阵的 full），作为消融。
- **D2 第十六波不通过：** full = L + C + V，Thinker 直接复用 wave15/eval_cvla_unist，即矩阵里的 noN；不跑 noC 和 SC-N。
- **D3 第十五波组 2 不通过：** full = N + V，复用 wave16 的输出；不跑 noN、noL 和 SC-C。等延迟对照仍跑，因为它只属于论文分析。
- **D4 比较对象：** 主判定对 B（第五波全系统规则）。对 ew_v1 只报告不退化，不作门槛。是否要求 full 不劣于最强的单组件臂（L、noC、noN 中最好的一个），需要确认。草案：只报告，不作门槛。
- **D5：** 用 combo_orch 的 L 臂代替 `wave14_combo_orch.sh`，理由见第 3 节。

## 6. 判定规则草案（沿用主线方案和第五波全系统规则，一个设置）
- **主判定：** full_v1 对 B，两条。
  1. 每个方向 × {a, b} × 条件的 ASR-BLEU 不退化：差 ≥ −0.3 且 CI 上界 ≥ 0；
  2. 每个方向 × {a, b_noxt} 的带噪 PESQ 提升：配对中位数 > 0 且 CI 下界 > 0。
- **报告：**
  - full_v1 对 ew_v1 的不退化检查；
  - 消融表：full − 去掉某个组件的臂，给出 ASR、text、D1、D2 及 CI；
  - −V1：full_v1 对 full_b。
- **选择规则：** 按"BLEU 优先于 LAAL"选择。full 的 LAAL 若比最好的单组件臂明显更差（D2 > +0.5 s），要在报告中单独讨论，但不推翻 BLEU 上的结论。
- **测试：** 组合判完后，再在 `robust_test_prereg_20261120.zh-CN.md` 里补写系统列表和测试判定，写在任何测试数字之前。

## 启动记录（只追加，写在任何组合数字之前）
- **2026-11-20 16:42 CET 决定点（uniss-de，按最优选择，不扫参）：**
  - D1：第十七波若通过，底座换 enc_wise05，NCWI 沿用已拟合的 α*（只在 LoRA 路径下）；另跑不含 E 的 full 臂作消融。
  - D2 / D3：第十六波或第十五波组 2 不通过，就把对应组件从组合里去掉；等延迟对照照常跑。
  - D4：full 是否不劣于最强单组件臂只报告，不作门槛。
  - D5：用 `combo_orch.sh ARM=L` 代替 `wave14_combo_orch.sh`；第十四波的组合判定改用 combo_score 的 L_v1 对 B / 对 ew_v1。
- **16:42 CET 启动 ctrl_budget_base**（卡 4–5）：它只依赖已完成的第十五波组 1；第十五波组 2 要等第十四波合并（约 17:40 之后）才用卡 4–5，本臂预计约 25 min 跑完。

## 结果：CV-LA 的等延迟对照（17:12 CET，只追加）
- **运行：** ctrl_budget_base，卡 4–5，16:43–17:10 CET。打分用 `combo_pair_score.py`（只算配对：沿用 combo_score 的读取和统计，判定照 4.2 节逐字实现）。输出 `eval_outputs/mist_w3po_v1/combo/ctrl_budget_base/pair_cvla_vs_budget.json`。
- **判定："跨视角信息"不成立。** 没有任何带噪单元 text 差的 CI 下界 > 0，也没有显著退化单元。
- Δ = cvla_base − ctrl_budget_base，[95% CI]：

| 单元 | Δ text | Δ ASR-BLEU | Δ D1 (s) |
|---|---|---|---|
| 英→中 a 带噪 | −0.05 [−0.39, +0.24] | +0.38 [−0.14, +0.90] | −0.021 |
| 英→中 a 干净 | −0.19 [−0.41, −0.01] | −0.05 | −0.017 |
| 英→中 b 带噪 | +0.23 [−0.01, +0.49] | −0.00 | −0.018 |
| 英→中 b 干净 | −0.03 | +0.75 [+0.03, +1.59] | −0.018 |
| 中→英 a 带噪 | −0.09 [−0.36, +0.06] | −0.31 [−1.03, +0.36] | −0.004 |
| 中→英 a 干净 | +0.10 | +1.79 [−0.25, +5.79] | −0.008 |
| 中→英 b 带噪 | +0.04 [−0.06, +0.14] | −0.04 | −0.014 |
| 中→英 b 干净 | +0.03 | +0.15 | −0.016 |

- **解读：** 只要每次读后的累计提交量相同，截在两个视角不一致处还是截在提交量用完处，译文质量几乎一样（text 差都在 ±0.25 以内）。第十五波组 1 的 +1.7–2.9 BLEU 全部可以由"少提交、晚提交"解释。增强视角真正的作用是给出一个随噪声变化的保守程度：带噪单元被截短的读更多（12–18%，干净为 7–11%）。
- **对论文的含义：** CV-LA 不能写成"跨视角信息带来增益"，只能写成"无参数、随噪声自适应的提交保守度"。它是否优于固定的保守度，要和一个不随噪声变化、平均延迟相同的对照比较，本波没有做。

## 追加对照：不随噪声变化的等量预算（17:13 CET 预登记，写在本臂任何数字之前）
- **动机：** 上一节显示 CV-LA 的增益来自"少提交、晚提交"。剩下要回答的是：CV-LA 的保守程度随噪声变化（带噪截得多、干净截得少），这种自适应本身有没有价值。
- **臂 ctrl_pool_base：** 与 ctrl_budget_base 相同，只把预算换成不随噪声变化的版本。同一句话的干净版和带噪版（meta 里 set 和 pair 都相同，读次一一对应）用同一份预算：B_t = ceil((B_t^clean + B_t^noisy) / 2)，B 取自 wave15/eval_cvla_base。这样每对句子的总提交量与 CV-LA 相同（只差 ceil 取整），但不再依赖噪声条件。没有自由参数。
  - 实现：`combo_si_thinker.py --budget-pool-meta`，编排用 `BUDGET_POOL=1`。在现有文件上核对过：末次读的预算和，英→中 CV-LA 干净 / 带噪为 4831 / 4053，pooled 为 4531 / 4531；中→英 4675 / 3695，pooled 为 4323 / 4323。
- **判定（写定）："随噪声自适应的保守度"成立，需要同时满足：**
  1. cvla_base − ctrl_pool_base 至少在一个带噪单元 text 差的 CI 下界 > 0，且该单元的 D1 差 ≤ +0.10 s；
  2. 两种条件合并后，没有任何方向的 D1 差 > +0.10 s；
  3. 没有任何单元（a / b × 两种条件）的 text 或 ASR 差 < −0.3 且 CI 上界 < 0。
  - 预期代价：干净单元上 CV-LA 可能略差（它在干净时提交得更多），这一项只报告。
- **卡：** 需要两张能放下 0.85 显存 Thinker 的卡（约 25 min）。卡 4–5 要留给第十五波组 2，所以等哪两张卡先空出来就用哪两张，不和别的 Thinker 叠在同一张卡上。
- **18:04 CET 启动 ctrl_pool_base（卡 4–5）：** 第十五波组 2 已在 18:02 交还卡 4–5。第十六波只在卡 4–5 有 ≥ 60 GB 空闲时才叠上去，本臂的 Thinker 先占卡，不会冲突。
