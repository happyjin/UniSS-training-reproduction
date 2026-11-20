# 鲁棒单说话人线：最终测试预登记（2026-11-20 16:20 CET 起草；数据部分，写在任何测试数字之前）

## 0. 状态
- 本文件目前只定下**测试数据**。参加比较的系统、主判定和报告格式，等 dev 上的完整系统组合和逐项消融做完后补写在第 6 节，**仍然写在任何测试数字之前**。
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

## 3. 真实噪声的分布外测试：OOD-3 的单说话人片段（可行，尚未生成）
- **来源：** `data/open_voicecond_v1/idea2/ood3/<set>_ref.jsonl`，即 Idea 2 OOD-3 子集的参考窗口：
  - AMI test 的 EN2002d 和 IS1009c，两个方向都有；
  - AliMeeting test 的 R8009_M8021 和 R8009_M8024，中→英，参考译文经过人工审校。
- **取法（与 `wave2_s1_prep.py` 的 (b) 相同）：**
  - 每个参考 utterance 按 `start_s + begin_s .. start_s + end_s` 从窗口音频中切出，作为一句单说话人输入；
  - 保留的条件：不是犹豫词，时长 ≥ 1 s，窗口内没有其他说话人与它在时间上重叠；
  - 有重叠的句子打上 `ovl` 标记，单独报告，不进主表。
- **粗计数**（只检查了窗口内的重叠；生成时还要对全会议的 utts 再检查一次，数量可能略少）：

| 子集 | 方向 | 参考窗口 | utterance | 无串音、≥ 1 s | 时长 |
|---|---|---|---|---|---|
| ami（中文配音的 AMI） | 中→英 | 386 | 1201 | 394 | 0.39 h |
| en（AMI 原声） | 英→中 | 256 | 715 | 187 | 0.19 h |
| ali（AliMeeting 远场） | 中→英 | 844 | 1602 | 853 | 0.58 h |

- ami 子集的中文语音是配音生成的，不是真实录音，在表中单列，并注明"合成语音 + 真实会议声学"。en 和 ali 是真实的会议声学环境。

## 4. OpenBroadcast-SI test 的单说话人片段（可行，尚未生成）
- **来源：** `data/open_voicecond_v1/bench/{id2,ood1,clean}/<dir>/refs_windows.jsonl`，split = test。每个参考窗口恰好是一个 utterance（中→英 1326 个，英→中 1283 个）。
- **取法：** 同 `wave2_s1_prep.py` 的 (b)（S1 dev 用的就是 bench_dev 的这套切法），按参考时间从会话混音里切出。
- **只保留没有串音的轮次**：会议内没有其他说话人的参考轮次与它在时间上重叠。中→英 678 句（0.97 h），英→中 585 句（0.85 h）。三个条件（clean / id2 / ood1）的句子一一对应。
- **条件：**
  - clean：干净的孪生版本；
  - id2：留出的 MUSAN / FSD50K 背景，与 RealSI 带噪版是同一个池；
  - ood1：FMA、DEMAND 和 ESC-50，这三个语料整体保留给 benchmark，任何训练和 dev 都没有用过，是背景分布外的条件。
- 源语音是 CoVoST 2 TEST 中留出的说话人（`HELDOUT_SPEECH.txt`），本线训练和 dev 都没有用过。

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
- OOD-3 和 OpenBroadcast-SI 单说话人测试集的生成，要在第 6 节写定之后、任何系统评测之前完成，并在生成后先核对。
