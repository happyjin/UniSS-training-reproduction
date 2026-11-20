# 鲁棒单说话人线：最终测试预登记（2026-11-20 16:20 CET 起草；数据部分，写在任何测试数字之前）

## 0. 状态
- 本文件目前只定下**测试数据**（第 1–5 节；第 3、4 节于 2026-11-20 补上最终条目数）。参加比较的系统、主判定和报告格式，等 dev 上的完整系统组合和逐项消融做完后补写在第 6 节，**仍然写在任何测试数字之前**。
- 到写本文件为止，没有任何系统在下面任何一个测试集上跑过（RealSI 干净版只有论文和第二波 S1 复现检查里的基线数字，见 1.3）。
- 所有测试集只用于测试：不训练、不拟合、不选检查点、不调任何参数。每个测试集、每个系统只跑一次。

## 1. RealSI 干净版
### 1.1 评测单元
- 与论文不含 QABC 的 LA2 基线完全相同：`si_infer.realsi_items(<dir>)`。英→中 346 句，中→英 431 句，每个方向 10 段录音（`data/raw/realsi/RealSI_data/data/<dir>/wav/<vid>.wav`）。
- 第 k 句 = 录音 `[start_time, end_time)` 毫秒的样本，双声道取平均；参考译文 = `trg_text`。
- 中→英有 3 对相邻句在时间上重叠，英→中有 1 句短于 1 s。这些都照原样保留，与基线相同。

### 1.2 本次的输入文件
- `data/open_voicecond_v1/realsi_noisy_v1/items_clean_<dir>.jsonl`，即 `--items-file` 格式 `{id, audio, dur_ms, translation, src_text}`，音频在 `audio/<dir>/clean/<id>.wav`。
- 这些音频是按 realsi_items 的切法存下的 FLOAT wav。核对结果：与 realsi_items 送进模型的数组**逐位相同**，346/346 和 431/431。
- 干净版和带噪版用同一条管线、同样的切句，只换音频，因此两者是逐条成对的比较。
- Talker 的音色 prompt（音频取到 max(首次写出, 1 s)）和 V1 token 都取自切出的句子本身。用 items 文件和直接用 realsi_items，Talker 的输入也完全相同。

### 1.3 已有的参照数字（不是本次测试的结果）
- 论文 `results.json` realsi LA2 m=1：英→中 ASR-BLEU 29.75、D1 2.51 s；中→英 23.19、2.70 s。
- 第二波 S1 复现检查用论文存下的块，重新跑了 Talker 和打分，得到 29.78 / 2.514 s 和 23.15 / 2.701 s。
- Thinker 重跑不是逐位一致的：单种子端到端为 29.30 / 22.58（见 `robust_wave2_diag_prereg.zh-CN.md`）。所以测试表里的底座**用本次重跑的数字**，论文数字只作对照列。

## 2. RealSI 带噪版（`data/open_voicecond_v1/realsi_noisy_v1/`，已生成，只生成这一次）
### 2.1 构建方式
- 脚本：`experiments/mist_w3po_v1/voicecond/realsi_noisy_build.py`，sha1 e12ab2ec…。build_info.json 记录了脚本 sha1、生成时的 git HEAD、种子、背景池和许可证。只用 CPU，拒绝覆盖。
- **逐条对应：** id、切点、长度、参考都与 1.2 的干净版相同，只是源音频加了背景。

### 2.2 背景
- 只用 OpenBroadcast-SI 的 id2 留出池（`bench/bgsrc/pools.json`），共三类：
  - 音乐 = 留出的 MUSAN music（bed 层）；
  - 人群 = 留出的 FSD50K crowd（amb 层）；
  - 音效 = 留出的 FSD50K sfx 加 MUSAN noise（sfx 层）。
- 每段录音生成一条连续的背景时间线：`bench_build.bg_schedule`，种子为 `OpenBroadcast-SI-v1|realsi:bg:<dir>:<vid>`；用 `render_bg` 渲染，每层分别渲染，以便记录逐句各层的能量占比；丢掉前 2.5 s 引入段（bed 层的淡入）。三层叠加与 OpenBroadcast-SI id2 的做法相同。
- 实际用到 294 个背景文件，清单和许可证见 build_info.json。许可证有 MUSAN（CC BY 4.0）、FSD50K 各条的 CC BY / CC BY-NC / CC0 / Sampling+。

### 2.3 电平
- 每句一个 BVR 目标值，取 OpenBroadcast-SI 的 BVR 统计（`bench/meta/stats.json` 的 bvr_db.values，与 `bench_build.plan` 用的是同一份）在分层分位点上的值：
  - q_i = 0.05 + 0.90 (perm_i + 0.5) / n；
  - perm 是 range(n) 的随机排列，用 `random.Random("OpenBroadcast-SI-v1|realsi:bvr:<dir>")` 打乱。
- alpha_i = sqrt(10^(BVR/10) · E(干净句) / E(背景段))，与 `bench_build.mix` 的 alpha 相同，只是能量在整句上计算。
- 不做 ducking：一句只覆盖一个说话段，ducking 只会在句首多出一个 50 ms 的起音斜坡。
- 限幅器与 bench 相同：混合后峰值超过 0.99 时，语音和背景一起整体缩放到 0.99 以下。

### 2.4 为什么逐句切、逐句定电平，而不是生成一条整段的带噪录音
- 基线每句单独评测：Thinker 只听切出的这一句，音色 prompt 也来自这一句。所以按 realsi_items 原样切干净录音，就能保证时间轴和每个读窗完全不变。
- 逐句取 BVR 时，每个方向有 346 / 431 次抽样贴合目标分布；如果按录音取，只有 10 次。
- 背景时间线仍然按整段录音连续生成，所以相邻句的背景是同一条真实的连续背景。

### 2.5 核对结果（`verify.json`，PASS）
| | 英→中 | 中→英 |
|---|---|---|
| 条目数 | 346 = realsi_items | 431 = realsi_items |
| id、参考、两份 items 文件之间的对应 | 一致 | 一致 |
| 干净切句与 realsi_items 逐位相同 | 346/346 | 431/431 |
| 干净与带噪长度相同 | 346/346 | 431/431 |
| 从文件反算 BVR 与目标的最大偏差 | 5e-5 dB | 6e-5 dB |
| 启用限幅的句数 | 2 | 24 |
| 峰值 > 0.99 | 0 | 0 |
| 主导层：音乐 / 人群 / 音效 | 303 / 41 / 2 | 369 / 57 / 5 |
| 平均能量占比：音乐 / 人群 / 音效 | 0.80 / 0.14 / 0.06 | 0.80 / 0.14 / 0.06 |

- 中→英限幅较多，是因为 RealSI 中→英原始录音本身就很响（干净句的峰值常在 0.9 以上）。
- BVR 分位数（dB）：

| 分位 | 5 | 10 | 25 | 50 | 75 | 90 | 95 |
|---|---|---|---|---|---|---|---|
| 目标，英→中 | −32.79 | −27.65 | −21.02 | −14.28 | −11.05 | −7.72 | −6.77 |
| 目标，中→英 | −32.78 | −27.69 | −21.03 | −14.29 | −11.04 | −7.72 | −6.77 |
| OpenBroadcast-SI 统计（q 0.05–0.95 段） | −32.54 | −27.52 | −20.99 | −14.29 | −11.05 | −7.72 | −6.79 |

- KS 统计量 0.003（p = 1.0）。按语音活跃帧（帧能量 > 峰值 − 30 dB）计的 BVR 约低 1 dB（中→英中位数 −15.45 dB）。
- 抽查 8 + 8 条：没有削波（|y| > 0.98 的样本比例 ≤ 1.3e-5），带噪句的活跃 RMS 与干净句相近。

### 2.6 背景留出检查（构建和核对时都做了断言）
- id2 池的全部 324 个文件都在 `HELDOUT_BACKGROUNDS*.txt` 里，而且都满足 sha1 留出规则。
- 它们与 Idea 2 背景索引（`data/open_voicecond_v1/idea2/index/backgrounds.jsonl`）的 train（1393）、dev（165）、test（187）三个划分的交集都是 **0**。
- 本线所有训练的噪声都来自这个索引的 train 划分：CoVoST 加噪 ERM、第五波、第十四波 UniST、第十七波，都是 `wave1_data.bg_pools("train")`。dev (a) 带噪用的是 dev 划分。V1 音色编码器（Idea 1）的背景池在构建时就排除了留出文件。
- 语音侧：第十四波 UniST 训练数据已按文本、文件名排除了与 RealSI 重叠的 5 条（中→英）和与 OOD-3、OpenBroadcast-SI bench 重叠的行（`wave14/data/stats.json`）。

## 3. 真实噪声的分布外测试：OOD-3 的单说话人片段（已生成，只生成这一次）
- **位置：** `data/open_voicecond_v1/ood_single_v1/`，与第 4 节在同一个目录。
  - 脚本：`experiments/mist_w3po_v1/voicecond/ood_single_build.py`。build_info.json 记录了脚本 sha1 和生成时的 git HEAD。只用 CPU，拒绝覆盖。
  - items 文件是 `items_<dir>.jsonl`（`--items-file` 格式）。按子集拆开的文件在 `items/<set>_<cond>_<dir>.jsonl`。
  - meta 文件是 `meta_<dir>.jsonl`，格式为 `{id, set, cond, pair, stream, ovl, dur_ms}`，口径同 `wave2_S1/data/meta_*.jsonl`；stream = 会议:说话人，作为 bootstrap 的分块单位。
  - 逐条来源在 `manifest_<dir>.jsonl`。
- **来源：** `data/open_voicecond_v1/idea2/ood3/<set>_ref.jsonl`，即 Idea 2 OOD-3 子集全部参考窗口里的全部 utterance：
  - AMI test 的 EN2002d 和 IS1009c；
  - AliMeeting test 的 R8009_M8021 和 R8009_M8024。
- **取法：** 同 `wave2_s1_prep.py` 的 (b)。每个参考 utterance 按 `[begin_s, end_s)` 从窗口音频中切出，不加边。只有同时满足下面四条的才保留：
  1. 不是犹豫词；
  2. 时长 ≥ 1 s；
  3. 参考非空，且是目标语言：英→中要含汉字，中→英要含拉丁字母。这一条用来去掉 AMI 里原样抄进 zh 字段、没有翻译的片段，例如 "WHAT RENA"、"OOH"；
  4. **在整场会议的完整时间线上**，没有任何其他说话人的 utterance（包括犹豫词）与它重叠。
- **完整时间线的来源：**
  - en 用 AMI source_ihm 索引（两场会议共 1827 条）；
  - ali 用 `alimeeting_si_v1/utts`（1632 条）；
  - ami 用配音版的 `windows_test_fullref_v9`（1420 条）。
  - 构建时断言：每条参考的绝对时间都与完整时间线一致（误差 < 10 ms）。

| 子集 | 方向 | 声学 | 参考 | utterance | 去掉：犹豫 / < 1 s / 非目标语言 / 串音 | **保留** | **时长** |
|---|---|---|---|---|---|---|---|
| en（AMI 原声） | 英→中 | 真实会议 | AMI zh | 715 | 56 / 218 / 2 / 254 | **185** | **0.19 h** |
| ali（AliMeeting 远场） | 中→英 | 真实远场 | DiariST 人工译文 | 1602 | 15 / 304 / 0 / 430 | **853** | **0.58 h** |
| ami（中文配音的 AMI） | 中→英 | 合成语音 + 会议场景 | AMI 英文原文 | 1201 | 0 / 468 / 0 / 339 | **394** | **0.39 h** |

- **ami 子集单列**：中文语音是配音生成的，不是真实录音。它不进"真实噪声"的主结论，只作补充。
- en 有 15 句峰值达到 1.0，是 AMI 原始录音本身的削波，保留原样，不修复；ali 和 ami 没有削波。

## 4. OpenBroadcast-SI test 的单说话人片段（已生成，只生成这一次）
- **来源：** `data/open_voicecond_v1/bench/{clean,id2,ood1}/<dir>/refs_windows.jsonl`，split = test。每个参考窗口恰好是一个 utterance。
- **取法：** 同 `wave2_s1_prep.py` 的 (b)，按窗口从会话混音里切出。
- **只保留没有串音的轮次**：在会议的 utts 里，没有其他说话人的轮次与它在时间上重叠。
- **三个条件严格一一对应：** 构建时断言三个条件的轮次、时间和参考完全相同。pair = 会议号_序号，每个 pair 恰好包含 clean / id2 / ood1 三份，时长和参考都相同（核对通过）。

| 方向 | 参考窗口 | 去掉：串音 | **保留（每个条件）** | **时长（每个条件）** |
|---|---|---|---|---|
| 中→英 | 1326 | 648 | **678** | **0.97 h** |
| 英→中 | 1283 | 698 | **585** | **0.85 h** |

- **条件：**
  - clean：干净的孪生版本；
  - id2：留出的 MUSAN / FSD50K 背景，与 RealSI 带噪版是同一个池；
  - ood1：FMA、DEMAND 和 ESC-50，这三个语料整体保留给 benchmark，任何训练和 dev 都没有用过，是背景分布外的条件。
- 源语音是 CoVoST 2 TEST 中留出的说话人（`HELDOUT_SPEECH.txt`）。
- id2 和 ood1 的峰值都 ≤ 0.99（bench 的限幅器）。英→中 clean 有 2 句峰值为 1.0，是源 clip 本身的削波，保留原样。

## 4a. 第 3、4 节与训练数据的重叠检查（`ood_single_v1/verify.json`，PASS）
- **比较对象：** 第五波 CoVoST ERM 的训练行（即 ERM-WiSE 主系统的训练数据）和第十四波 UniST 的训练行（第十六、十七波也用它），两个方向都查。
- **比较内容：** 源文本和参考译文的归一化文本（与 `wave14_data.norm` 相同）是否完全相同；对 OBSI 还比较 CoVoST clip id 和说话人。

| | 第五波 CoVoST ERM | 第十四波 UniST |
|---|---|---|
| OOD-3 三个子集：源文本 / 参考 | 0 / 0 | 0 / 0 |
| OBSI 中→英：源文本 / 参考 / clip / 说话人 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| OBSI 英→中：源文本 / 参考 / clip / 说话人 | 3 / 6 / 0 / 0 | 0 / 1 / 0 / 0 |

- OBSI 英→中的命中：
  - 6 次和 1 次参考命中都短于 10 个字符，属于偶然相同的短句；
  - 3 次源文本命中是**同一个 pair `e031_0004` 的三个条件**（"Media specialist and former Right On!"）。Common Voice 中同一句文本会由不同说话人朗读，这里 clip 和说话人都不同，所以不是同一段录音。
- **决定（写定）：** pair `e031_0004` 不进主表，即 OBSI 英→中主表为 584 个 pair。数据不重新生成，排除在打分时按这个 pair 号进行，含它的结果只在附录给出。

## 5. 可选：ACL 60/60 英→中（干净、领域分布外）
- 数据：`data/acl6060_dev_v1`，5 个演讲，468 句参考。items 文件按 VAD 切成 633 段，参考需要用 mwerSegmenter 重新对齐，与 MIST 主线以前的 ACL 协议相同。
- 只有英→中一个方向。如果时间允许就跑；不跑也不影响主判定。

## 6. 系统列表、主判定、报告格式
**待 dev 组合完成后补写，写在任何测试数字之前。** 至少要写定以下几项：
- 比较的系统：底座 LA2；ERM-WiSE；最终组合；以及逐项去掉一个组件的消融，是否进测试待定。
- 每个测试集和条件的主指标：ASR-BLEU、text BLEU、speech LAAL（主延迟），EndOffset 只作补充。
- bootstrap 的单位：句子，或按录音分块。
- TTS 种子：只用一个种子 20261021，与探索阶段相同。
- 判定规则：沿用 dev 的统一规则，或者另外写定。
- OOD-3 和 OpenBroadcast-SI 单说话人测试集已于 2026-11-20 生成并核对（第 3、4、4a 节）。到写入时，还没有任何系统在它们上面跑过。

## 6. 系统列表、主判定、报告格式（2026-11-20 19:45 CET 补写；写在任何测试数字之前，到此时没有任何系统在任何测试集上跑过）
### 6.1 比较的系统（事先写定；每个系统在每个测试集上只跑一次）
| 名称 | Thinker | 提交策略 | 音色 | Talker | 来源（dev 上的对应） |
|---|---|---|---|---|---|
| **B** | 基础 Qwen3-Omni-30B-A3B-Instruct | LA2，读窗 1 s（`si_thinker_stream --policy la2 --windows 1`） | 原音色（本句源音频前 max(首次写出, 1 s)） | T1c export_1500 | wave2 S1 底座协议 |
| **EW_v1** | ERM-WiSE（`wave5/M/merged_erm_wise05`） | LA2 w1（`si_thinker_stream`） | V1 token（`wave5_m_v1tokens.py`） | T1c | 第五波全系统 ew_v1 |
| **C_v1**（全系统） | ERM-WiSE | CV-LA（`combo_si_thinker --second-view enh`，DeepFilterNet3 因果增强视角，`wave15_enh_views.py` 同一实现） | V1 | T1c | 组合臂 C |
| EW_b、C_b（消融，同次运行顺带产出） | 同上 | 同上 | 原音色 | T1c | ew_b、C_b |

- Talker 的参数与 dev 完全相同：`si_infer --window 1 --text-policy file --speed-policy block --gpu-mem 0.1`，`SI_KV_CACHE_GIB=7.73`，4 个分片，种子 20261021（只用一个 TTS 种子，与探索阶段相同）。
- **条件系统 CE**：Thinker 换成第十七波 b 的 enc_cv_wise05，其余同 C_v1。只有同时满足以下两条才测：(1) 第十七波 b 两条主判定都通过；(2) dev 上 CE 组合臂通过全系统规则（与臂 C 相同的规则 1）。满足时用本节完全相同的协议，每个测试集只跑一次，并在本文件追加启动记录，写在 CE 的任何测试数字之前；不满足就不测，也不在测试集上做任何 CE 相关的计算。
- 不测的：NCWI、UniST 底座、第十七波 enc（都在 dev 上没通过）；CV-LA 的两个 dev 对照（ctrl_budget、ctrl_pool）只是机理分析，不进测试。

### 6.2 测试集与单元
- 每个方向合并成一个 items 文件，评测一次：`eval_outputs/mist_w3po_v1/test_robust/data/items_<dir>.jsonl` 和 `meta_<dir>.jsonl`。
  - RealSI 干净 / 带噪：id 加前缀 `rsc__` / `rsn__`（原 id 相同，必须区分），set = realsi，cond = clean / noisy，pair = 原 id，rec = 录音号。
  - OOD-3 与 OBSI：沿用 `ood_single_v1` 的 id 和 meta（set = ood3_en / ood3_ali / ood3_ami / obsi，cond = real / clean / id2 / ood1）。
  - 条目数：英→中 346 × 2 + 185 + 585 × 3 = 2632；中→英 431 × 2 + 853 + 394 + 678 × 3 = 4143。
- **单元（cell）：** 方向 × {realsi/clean, realsi/noisy, obsi/clean, obsi/id2, obsi/ood1, ood3_en（英→中）, ood3_ali（中→英）, ood3_ami（中→英，单列）}。
  - OBSI 英→中的主表排除 pair e031_0004（第 4a 节），含它的结果放附录。
- **bootstrap 单位：**
  - RealSI 按句（stream = 句 id）。理由：基线协议里每句单独评测，句子之间没有任何状态传递；每个方向只有 10 段录音，按录音分块只剩 10 个块，CI 没有意义。按录音分块的 CI 作为辅助一并报告。
  - OBSI 和 OOD-3 按 meta 的 stream（会议:说话人）分块。
  - 2000 次重采样，seed 0；ASR-BLEU 和 text BLEU 用单元内充分统计量求和（同 `wave3_v_score.compare` / `combo_score.text_compare`）。

### 6.3 指标
- **主指标：** ASR-BLEU（论文协议：`reports/uniss_si_v4/speech_only.py` → `evaluation.asr_transcribe --batch-size 4`）。
- **辅助：** Thinker text BLEU；语音 LAAL（主延迟，逐单元；D1 / D2 用 `laal_defs.sentence_defs`，与 dev 相同）；PESQ（SQUIM，前 20 s）；说话人相似度（TitaNet-large，对同 pair 的干净孪生版本；OOD-3 没有干净孪生，对本句源音频）。
- RealSI 干净版另列论文基线对照列（29.75 / 2.51 s，23.19 / 2.70 s），表中 B 用本次重跑的数字作主。

### 6.4 主判定（写定）
1. **C_v1 相对 B，不退化：** 所有主表单元（ood3_ami 除外）× 两个方向，ASR-BLEU 差的点估计 ≥ −0.3 且 CI 上界 ≥ 0。
2. **C_v1 相对 B，带噪提升：** 带噪单元（realsi/noisy、obsi/id2、obsi/ood1）中，至少一个方向的一个单元 ASR-BLEU 差的 CI 下界 > 0。
- 两条都满足，记为"全系统在测试集上成立"。
- **报告但不作门槛：** C_v1 相对 EW_v1（CV-LA 的测试增量）；EW_v1 相对 B（ERM-WiSE + V1 的测试增量）；C_v1 相对 C_b（V1 的贡献）；各系统的 LAAL 差、PESQ 配对中位数差、相似度差；ood3_ami；含 e031_0004 的 OBSI 英→中。
- 判定只用 6.2 定义的单元和统计，不另挑子集，不改阈值。

### 6.5 实现与资源
- 编排：`experiments/mist_w3po_v1/voicecond/test_orch.sh`（combo_orch.sh 的副本，换成测试数据根；watcher POLL = 1；launch 前等占位离开，最多 120 s）。打分：`test_score.py`。
- 卡：B 用卡 0–1，EW 用卡 2–3，C 用卡 4–5；卡 6–7 留给第十七波 b 训练，不碰。

### 6.6 启动记录（只追加；写在任何测试数字之前）
- **19:47 CET：** 合并测试数据已生成（`test_data_build.py`，`test_robust/data/build.json`：英→中 2632 条 4.18 h，中→英 4143 条 5.47 h）。三个系统同时启动，每个系统只跑这一次：
  - B：`ARM=B TK=stream RUNS=b`，卡 0–1；
  - EW：`ARM=EW TK=stream BASE=wave5/M/merged_erm_wise05 RUNS="b v1"`，卡 2–3（得到 EW_b、EW_v1）；
  - C：`ARM=C TK=combo SV=enh BASE=wave5/M/merged_erm_wise05 RUNS="b v1"`，卡 4–5（先在 CPU 上生成增强视角；得到 C_b、C_v1）。
- 为测试数据根加的改动（默认行为不变）：`wave5_m_v1tokens.py`、`wave15_enh_views.py`、`wave3_v_spksim.py` 读环境变量 S1_DATA；spksim 在 S1_DATA 下对没有干净孪生的条目（OOD-3）用本句源音频作参照（见 6.3）。
- 卡时估计：每个系统约 1.5–2 h 墙钟 × 2 卡（Thinker 约 10–15 min，Talker 每个 run 约 25 min，ASR 约 30 min，PESQ / 相似度约 15 min），合计约 10–12 卡时；预计约 21:45–22:15 CET 全部完成。
