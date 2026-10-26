# 多说话人同声传译(AMI 英→中)方案

把 SCOPE(`tagspeech/gstar-nemotron` 分支的最好系统)的说话人条件化和贝叶斯解码,接到我们当前最好的同传系统(m2)上。

- 日期:2026-10-26
- 适用仓库:UniSS(同传侧)、Auden `tagspeech/gstar-nemotron`(多说话人侧,只读借鉴)
- 状态:方案。除第 2、3 节中标注"已核实"的事实外,其余都是待实施的设计。

---

## 0. 一页总结

**要做什么**
- 输入:一段多人会议的单通道录音。
- 输出:边听边出的中文译文和中文语音,并且:
  - 每句译文标明是谁说的;
  - 每个人的译音保持他自己的音色;
  - 延迟和单人同传在同一量级(m2 档约 1.7–1.8 s)。

**数据**
- AMI 英→中的逐句翻译和合成语音已经在 Lance 表里,但 md 示例用的快照 67 里没有,要用 **v239**(当前最新)。
  - IHM 子集:train / validation / test = 104,004 / 12,730 / 12,177 条。
  - 还有同样规模的 SDM 子集。
- 这些数据是**逐句、单说话人、离线翻译**的。表里没有会议号和时间戳,需要和处理层的索引 join,再按会议把句子摆回时间轴。这样才能得到多说话人同传的训练样本。

**可借鉴的东西**
SCOPE 在 AMI 上是目前最好的流式多说话人识别:cpWER 平均 19.72,按 Whisper 归一化,AMI-SDM 19.78 / AMI-IHM 13.03,都低于已发表的 SSA-v2。可直接搬过来的部件有:
1. Nemotron-3 流式说话人分离(8 个槽位,1.04 s 前瞻);
2. 从后验算出的四状态目标说话人置信度 STNO;
3. 按置信度逐帧调节编码器 FFN 奇异值的 BSSC(只有几千个参数);
4. 说话人活动 token(sd_proj)和特征划分(DSS);
5. 逐说话人的条件链解码;
6. 三个贝叶斯部件:束搜索重排、逐词或逐块归属、序贯注册。

**能不能把 SCOPE 的 ASR 分支换成我们的同传模型**
可以,这也是推荐的主线(方案 B,"SCOPE-SI")。

- 关键的维度是对得上的:
  - Qwen3-Omni Thinker 的音频编码器(AuT,32 层,d=1280,FFN 5120)输出 2048 维;
  - 这和 SCOPE 里 Qwen3-ASR-1.7B 的 LLM 嵌入维度相同;
  - 所以 sd_proj 和 DSS 的形状不用改。
- BSSC 只需按 32 层重新做 SVD,参数量 64 × 4 × 32 = 8,192。
- 需要改的地方:
  - 识别窗口要改成同传的"每 Δ 秒读一次";
  - 输出从"带时间的转写"改成"带提交的译文";
  - 依赖时间 token 的贝叶斯部件要改成块级;
  - Talker 的音色记忆要按说话人槽位分开。

**推荐路线**
1. 先做零训练基线 B1:现有 m2 + Nemotron 分离 + 块级贝叶斯归属 + 按槽位的音色记忆。这一步验证数据、指标和端到端流程。
2. 再训练方案 B 的说话人条件化模块。
3. 最后在多说话人数据上重训提交策略(ABC),并让 SDR 的语音队列由所有说话人共享。

各方案都有对照基线和消融,见第 4–6 节。

---

## 1. 需求整理

| # | 需求 | 本文对应位置 |
|---|---|---|
| R1 | 读懂 `gstar-nemotron` 分支最好的实验结果、报告和模型 | §2.1 |
| R2 | 读懂 md(read-s2st-lance.md)如何读取翻译后生成的数据,并用给定的 AMI filter(ihm 子集为例)取数据 | §3.1–3.5 |
| R3 | 理解同传数据与 HF 原始 AMI(`edinburghcstr/ami`, ihm)的关系 | §3.3–3.4 |
| R4 | 做一个支持**多说话人**的同传模型 | §4 |
| R5 | SCOPE(G-STAR 式)架构里有什么可借鉴 | §4.1 |
| R6 | 能否把 SCOPE 的 ASR 分支换成我们当前最好的同传模型 | §4.3(方案 B),§2.3 |
| R7 | 特征如何融合 | §4.6 |
| R8 | 最终结果如何得到(推理流程和输出) | §4.7 |
| R9 | 完整方案:数据获取、架构、训练、测试;有多个设计选择时分别写出 | §3–§6,方案 A–E |

**沿用的约束**(来自之前的工作约定):
- 不做任何删除或覆盖已有实验的操作。
- 训练用 Megatron,数据完全 shuffle,GPU 保持满载。
- 训练数据放在 `data/` 下。
- 不额外安装训练环境。读取 Lance 需要的 `pylance==9.0.1` 须单独确认后,装在隔离目录里。
- 不挤占正在运行的 GPU 任务。

---

## 2. 现状盘点

### 2.1 `tagspeech/gstar-nemotron` 分支:当前最好系统 SCOPE(已核实)

来源:
- `examples/tagspeech/gstar_nemotron/REPORT.md`(2,396 行);
- `examples/tagspeech/paper_icml_v2/`;
- 分支最新提交 `6c248a8`(2026-10-26)。

#### 演进主线

| 阶段 | 测试 cpWER |
|---|---|
| TagSpeech(Qwen2.5-7B,双编码器) | 44.16 |
| FireRed 线 | 28.17 |
| Qwen3-ASR-1.7B 线(G-STAR 式训练 + Nemotron) | 19.77(ihm-mix,流式) |
| 换成 AMI-SDM + NOTSOFAR 联合训练,三测试集口径 | — |
| 贝叶斯全系统 E1 + E2 | 24.31(三集平均) |
| 重叠区专项:条件链 + BSSC = SCOPE | 19.72(三集平均) |

#### SCOPE 的组成(论文 §3)

1. **前端**:Nemotron-3-Diarization 流式分离器。
   - 后验 P_dia ∈ [0,1]^{T×8},帧长 80 ms,前瞻 1.04 s。
   - 用 AOSC 说话人缓存在整场会议里保持槽位一致。
   - 因果切窗:窗口最长 8 s,遇到 1 s 停顿就关闭。
2. **底座**:Qwen3-ASR-1.7B。
   - 编码器 24 层 × 1024(FFN 1024→4096→1024),每秒约 13 个 token。
   - 投影到 LLM 的 2048 维,LLM 28 层。
   - 编码器加 LoRA r=32,LLM 加 LoRA r=64。
3. **sd_proj**:后验插值到 token 中心,经过 Conv5 + MLP,得到 2048 维的"说话人活动 token"。每 5 个音频 token 后插 1 个。
4. **STNO 置信度**:先做 Platt 校准,再算目标 k 的四状态分布:
   - b_t^(k) = [(1−p̂_k)q̄, p̂_k q̄, (1−p̂_k)(1−q̄), p̂_k(1−q̄)];
   - 其中 q̄ = ∏_{j≠k}(1−p̂_j);
   - 四个状态依次是静音、独自说、他人在说、重叠。
5. **BSSC**:每个编码器 FFN 权重做截断 SVD(r=32),逐 token 用 g_t = Gᵀb_t 调节前 32 个奇异值。
   - G ∈ R^{4×32} 零初始化,24 层 × 2 个矩阵共 6,144 个参数;
   - 效果和 FiLM(98.3 万参数)相当,显著好于随机正交基。
6. **DSS**:按活跃槽位对 2048 维特征做 softmax 软划分。目标拥有的特征最多放大 K 倍。后验种子 ω=1 必不可少,全零初始化会卡在驻点。
7. **条件链解码**:窗口内活跃槽位按活动量降序逐个解码。
   - 第 i 轮的 prompt = 窗口前最后 30 个词 + 前 i−1 轮的结果;
   - 训练时以 0.1 的概率给"弱活动、无参考词"的槽位,目标为空(学会停);
   - 提示词以 0.1 的概率丢弃,后验以 0.2 的概率扰动;
   - 损失是加权交叉熵:时间 token 1.5,说话人 token 2。
8. **贝叶斯解码**(无训练参数,几个标量在 dev 上估):
   - **束搜索重排**:束宽 4,打分 log p_θ(h) + (λ/τ)·Σ_{f∈F(h)} logit p_{f,k}。λ=1,τ 在 AMI 上为 5.1,NSF 上为 4.6。
   - **逐词归属**:2S 状态的 HMM 做前向–后向,得到精确后验,包括 LLM 标签的混淆矩阵。
   - **序贯注册**:对新槽位做序贯概率检验,拒绝的槽位的词重新归属。

#### 训练配方

| 项 | 设置 |
|---|---|
| 训练数据 | AMI-SDM train(133 场,79.4 h)+ NOTSOFAR-1 train(526 场,54.3 h),**没有模拟重叠** |
| 初始化 | E2a |
| 步数和批量 | 20k 步,batch 16 turn,4 GPU |
| 优化器 | AdamW,峰值 lr 2e-5,warm-up 500 步,cosine 衰减 |
| 分组学习率 | 编码器适配器和 DSS 为 5 倍 lr,门控 G 和 ζ 为 50 倍 lr |
| 可训练参数 | 95.5M / 2.04B(4.7%) |

#### 主结果(test,cpWER 无 collar,三集依次为 AMI-SDM / NSF / AMI-IHM)

| 系统 | cpWER(保留填充词) | Whisper 归一化 | 延迟 |
|---|---|---|---|
| SOT-LLM+ctx(强基线) | 22.61 / 29.94 / 15.66 | 21.76 / 29.07 / 14.81 | 7.4 s |
| **SCOPE(S12c + 束宽 4)** | **20.63 / 24.79 / 13.75**(平均 19.72) | **19.78 / 23.63 / 13.03** | 7.3 s |
| SSA-v2(已发表,流式) | — | 20.33 / — / 13.38 | 1.04 s 块 |

- 对 SOT-LLM+ctx 的差值是 −3.02,95% 区间 [−3.29, −2.72]。
- 192 个测试会话中 187 个变好。改进幅度和重叠比例的相关系数为 −0.69 到 −0.82。
- 三个贝叶斯部件在最终模型上都显著有效:

  | 部件 | 平均差 |
  |---|---|
  | 束搜索重排 | −0.53 |
  | 严格归属 | −0.18 |
  | A6b + 注册 | −0.04 |

  注册还把 NSF 说话人数完全正确率从 66.3% 提高到 75.0%。

#### 已知短板(对同传最重要)

1. **延迟**:SCOPE 是窗口级解码,平均定稿约 7 s。
   - 窗口 ≤6 s 时平均 cpWER +0.78,≤4 s 时 +2.45;
   - 同传需要的约 1–2 s 延迟,不能直接沿用它的窗口。
2. **重叠区**:
   - 重叠区错误率约为单人区的 3 倍;
   - 次说话人错误率 42 / 43 / 32%;
   - 重叠中 ≤2 词的短附和最难;
   - 分离器以漏检为主(AMI-SDM DER 19.6,其中漏检 15.8)。
3. **重叠帧上两人的 STNO 几乎对称**:L1 距离中位数约 0.1。分支最新一轮(`robust/STNO_IMPROVE_PLAN_CN.md`)正在试:
   - 8 维带人数和主导度的置信度 b6d;
   - 自注册的说话人向量 spk;
   - 漏检模拟。

#### 模型和代码位置

| 内容 | 位置 |
|---|---|
| checkpoint、后验 dump、日志 | VUT 集群,`examples/tagspeech/experiments/gstar_nemotron/`(gitignored)。环境 `/mnt/matylda6/qjin/anaconda3/envs/{auden_tagspeech_ami,nemotron312}` |
| Qwen3 模型封装 | `gstar_nemotron/qwen3/qwen_gstar.py` |
| 条件链、BSSC、DSS | `gstar_nemotron/robust/{chain_x.py,train_chain_x.py,chain_decode_x.py}` |
| 在线驱动 | `gstar_nemotron/qwen3/online/` |
| 贝叶斯部件 | `gstar_nemotron/{beam,bayes2}/` |

本机(这台 8 卡机器)上没有 SCOPE 的 checkpoint,也没有 NeMo 环境。

### 2.2 我们当前最好的同传系统:m2(已核实)

**组件**
- **Thinker**:Qwen3-Omni-30B-A3B,加 ABC 适配器(顶层文本层注意力的 LoRA r=32,已合并为 `thinker_merged_i200`)。
  - 策略是 selfwait:wait 标记加偏置 b0;
  - SDR 语音队列风险:q_{k+1} = max(0, q_k + r·a_k − Δ)。
- **Talker**:UniSS T1c@1500,BiCodec 编码,逐块发声。
- **音色**:贝叶斯音色记忆,累计 30 s 后冻结。
- **一键脚本**:`experiments/uniss_qwen3_omni_v1/scripts/run_stream_s2st.sh`。
- **部署文档**:`docs/deploy/stream_s2st_bayes_deploy.zh-CN.md`。

**效果**

| 测试集 | 方向 | 指标 |
|---|---|---|
| RealSI | 英→中 | ASR-BLEU 28.26,LAAL 1.77 s |
| RealSI | 中→英 | ASR-BLEU 21.42,LAAL 1.70 s |
| ACL60/60-dev | 英→中 | 文本 BLEU 47.33,StreamLAAL 1.52 s |

单次读取的 Thinker 计算延迟约 57–59 ms。

**多说话人短板(已核实)**
- 音色记忆在单人长音频上更好;
- 在双人访谈上更差,因为没有触发重置;
- 目前没有任何"谁在说"的信息进入模型。

### 2.3 两边能否对接:关键维度对照

| 项 | SCOPE(Qwen3-ASR-1.7B) | 我们的 Thinker(Qwen3-Omni) | 结论 |
|---|---|---|---|
| 音频编码器 | AuT 24 层 × 1024,FFN 4096 | AuT 32 层 × 1280,FFN 5120 | BSSC 重新做 SVD;参数 32×2×4×32 = 8,192 |
| 编码器输出 / LLM 嵌入 | 2048 | 2048(`output_dim`=2048,text hidden 2048) | sd_proj、DSS、目标嵌入 u_k **形状不变** |
| 音频 token 率 | 约 13 个/s | AuT 约 12.5 个/s(80 ms;实现时以代码为准核对) | Conv5 每 token 对应约 0.4 s,不变 |
| LLM | 1.7B dense,28 层 | 30B-A3B MoE,48 层 | LoRA 只加在注意力上(沿用 ABC 做法) |
| 解码粒度 | ≤8 s 窗口,整窗解码 | 每 Δ=0.5–1 s 读一次,前缀续写加提交 | 窗口改为"块内读取"(§4.3) |
| 输出 | 时间 token + 文字 + 说话人 token | 译文 + wait 标记 | 时间相关的贝叶斯部件改为块级(§4.6) |

---

## 3. 数据获取与构造

### 3.1 md 的读取方法要点(read-s2st-lance.md)

1. **固定快照**:每个实验固定一个 Lance 版本,用 `lance.dataset(URI, version=V)` 打开,不要用"最新"。
2. **读取环境**:Python 3.11,`pylance==9.0.1`(8.0.1 读嵌套的 codec 列会出错),`soundfile`。
   - 本机目前没有装 pylance。需要确认后装在隔离目录里,例如 `/opt/dlami/nvme/jasonleeeli/tools/python_lance/`,不动现有训练环境。
3. **两步读取**:
   - 第一步,`ds.scanner(columns=[元数据列], filter=..., with_row_id=True)` 只扫元数据。
   - 第二步,用该批的 `_rowid` 调 `ds.read_blobs("source_audio_payload" / "target_audio_payload", ids=row_ids, preserve_order=True)` 取音频字节,再用 `soundfile` 解码。
   - 不要对整表 `to_table()`(TB 级音频)。
4. **主要字段**:
   - `target_codec_codes`:int16,形状 [帧, codebook],是 Qwen3-TTS-Tokenizer-12Hz 的 codec ID;
   - 质量证据可能为空,空表示未知,不是 0:`target_audio_whisper_wer/cer`、`target_audio_utmos_score`、`target_audio_ecapa_similarity`、`target_audio_quality_flags`;
   - 表里没有统一的 split 列,用 `_curion_contribution_id` 区分。

### 3.2 AMI 在哪个快照(已核实,2026-10-26)

**md 示例用的 v67 没有 AMI。** 发布标记目录 `.../_meta/curated/task=s2st/table=data/releases/` 中:
- 最新是 `lance_version=239`,2026-10-24 发布;
- 81,359,642 行,39 列;
- 包含全部 18 个 AMI 贡献。

| 子集 | 方向 | 贡献 ID(train / validation / test) | 加入时的版本 |
|---|---|---|---|
| ihm | en→zh | `ami_ihm_en_zh_{train,validation,test}` | 201 / 207 / 213 |
| ihm | en→ja、en→ko | `ami_ihm_en_{ja,ko}_*` | 203–217 |
| sdm | en→zh | `ami_sdm_en_zh_{train,validation,test}` | 219 / 225 / 231 |
| sdm | en→ja、en→ko | `ami_sdm_en_{ja,ko}_*` | 221–235 |

**建议固定 v239。** 提交标记:

```text
s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/_meta/curated/task=s2st/table=data/releases/lance_version=239/_COMMITTED.json
```

**条数**(处理层 manifest 的 `record_count`,即进入 Lance 的行数):

| 子集 | split | 源句数(HF) | 成功生成 | 保留率 |
|---|---|---|---|---|
| ihm | train | 108,496 | 104,004 | 95.9% |
| ihm | validation | 13,098 | 12,730 | 97.2% |
| ihm | test | 12,643 | 12,177 | 96.3% |
| sdm | train | 107,313 | 102,861 | 95.9% |
| sdm | validation | 13,098 | 12,730 | 97.2% |
| sdm | test | 12,643 | 12,177 | 96.3% |

### 3.3 这批 AMI 同传数据是怎么生成的(已核实,来自处理层 manifest)

链路名是 `speech_chain_all_local_en_zh_v1`,运行号 `ami-identity-20261021-001`:

```
HF edinburghcstr/ami (ihm / sdm,revision 46f28f25,逐句切好的音频)
  → clean_16k_mono_v1:统一为 16 kHz 单声道 PCM
  → 机器翻译 3 个候选:Hy-MT2-30B-A3B-FP8 / NiuTrans LMT-60-4B / MiLMMT-46-4B
  → XCOMET-XL、xlm-roberta-xl 打分,chrF 交叉指标 → 选 1 条(mt_selection)
  → VoxCPM2 合成(音色克隆,clone_mode=controllable,参考是该句源音频)
  → ECAPA 相似度、UTMOS、faster-whisper-large-v3 可懂度 → 选 1 条(tts_selected,选择指标 ecapa_sim)
  → Qwen3-TTS-Tokenizer-12Hz 编码(feature_version qwen3_tts_12hz_v1)
  → 进入 Lance(schema curion.speech.translation_pair@5.0.0)
```

和 HF 原始数据的对应:
- HF `ihm/train` 共 108,502 行。用户给的 row 34 是:

  ```text
  meeting_id=EN2001a  audio_id=AMI_EN2001a_H04_MEO069_0275384_0275512
  speaker_id=MEO069   microphone_id=H04
  begin_time=2753.84  end_time=2755.12
  text="ARE THEY SPOKEN NUMBERS"
  ```

- 处理层源索引对每一句都保留了这些字段。join 键 `record_id` 的形式是 `ami-ihm-test-00002-of-00004-00000000`,`original_record_id` 就是 HF 的 `audio_id`。

### 3.4 每行是什么、缺什么

**是什么**
- 一行 = AMI 的一句话(一个说话人的一个片段)。
- 包括:
  - 源音频:ihm 是该人的头戴麦,sdm 是远场单麦的同一时间段,里面会混入别人的声音;
  - 英文转写;
  - 选中的中文译文;
  - 克隆该人音色的中文合成语音;
  - Qwen3-TTS 12 Hz codec;
  - 质量指标。

**缺什么**(对多说话人同传很关键)
1. **Lance 行里没有会议号和时间。**
   - v239 manifest 的 39 列中有 `source_sample_id` 和 `source_speaker_ref`,没有 `meeting_id`、`begin_time`、`end_time`;
   - 需要和处理层索引 join,索引里有这些字段。
2. **翻译是逐句、离线、整句的**,不是同传式的增量译文。
   - 同传需要的"边听边出"的部分译文,要么自己构造(§3.6),要么沿用我们已有的 ABC 自标注方式(§5)。
3. **每句的合成语音各自克隆音色**,同一个人不同句子的音色不完全一致。短句的 ECAPA 相似度可能很低,例如 "OKAY" 只有 0.19,要按质量过滤。
4. **目标 codec 是 Qwen3-TTS 12 Hz**,而 UniSS Talker 用 BiCodec。用 UniSS Talker 时,需要把目标 wav 重新编码成 BiCodec(§3.6 第 6 步)。
5. **AMI 只有英语源。** 要做多说话人中→英,需要用同一条链路处理中文会议数据(例如 AliMeeting,TagSpeech 已有 AliMeeting 模型)。

### 3.5 读取 AMI 的脚本(按 md 的写法改造;待装好 pylance 后先跑小样核对)

```python
"""AMI ihm en->zh: metadata scan (no audio), join with the processed index for meeting/speaker/time,
then fetch audio blobs batch by batch. Read-only on S3; writes go under data/ami_multispk_si_v1/."""
import io

import lance
import pyarrow as pa
import pyarrow.dataset as pads
import soundfile as sf

URI = "s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/curated/task=s2st/data.lance/"
VERSION = 239                      # the first snapshot that holds every AMI contribution; pin it
SO = {"region": "ap-southeast-3"}
FILTER = """dataset = 'ami' AND src_lang = 'en' AND tgt_lang = 'zh'
AND _curion_contribution_id IN ('ami_ihm_en_zh_train', 'ami_ihm_en_zh_validation', 'ami_ihm_en_zh_test')"""
META = ["id", "source_sample_id", "_curion_contribution_id", "source_transcript", "target_text",
        "duration_ms", "target_duration_ms", "source_speaker_ref", "target_speaker_ref", "speaker_relation",
        "target_audio_whisper_cer", "target_audio_ecapa_similarity", "target_audio_utmos_score",
        "target_audio_quality_flags"]

ds = lance.dataset(URI, version=VERSION, storage_options=SO)
meta = ds.scanner(columns=META, filter=FILTER, with_row_id=True).to_table()   # ~129k rows, metadata only
print(meta.num_rows)            # expected 104,004 + 12,730 + 12,177 = 128,911

# meeting / speaker / time live in the processed index of the same chain (join key: record_id)
IDX = ("s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/processed/index/dataset=ami/"
       "process_version=speech_chain_all_local_en_zh_v1/source_subset=ihm/")
idx = pads.dataset(IDX, format="parquet", partitioning="hive").to_table(
    columns=["record_id", "original_record_id", "meeting_id", "speaker_id", "microphone_id", "begin_time", "end_time"])
# ASSUMPTION to check on the first rows: Lance source_sample_id == processed record_id
joined = meta.join(idx, keys="source_sample_id", right_keys="record_id", join_type="inner")
assert joined.num_rows == meta.num_rows, "join key differs -- inspect source_sample_id vs record_id / audio_id"

# audio: per batch of row ids, as the md shows
row_ids = joined.column("_rowid").to_pylist()[:2]
src = ds.read_blobs("source_audio_payload", ids=row_ids, preserve_order=True)
tgt = ds.read_blobs("target_audio_payload", ids=row_ids, preserve_order=True)
for (_, s), (_, t) in zip(src, tgt):
    print(sf.read(io.BytesIO(s))[1], sf.read(io.BytesIO(t))[1])   # check both sample rates; do not assume 16 kHz
```

**首次运行要核对的三件事**
1. `source_sample_id` 与处理层 `record_id` 是否同一形式。
   - 如果不是,改用 `original_record_id`(即 HF 的 `audio_id`)。
   - `audio_id` 本身就编码了会议、麦克风、说话人和起止时间(单位 10 ms),可以直接解析:`AMI_EN2001a_H04_MEO069_0275384_0275512` → 2753.84–2755.12 s。
2. 目标音频的采样率(VoxCPM2 输出)。
3. 三个 split 的行数是否等于上表。

**SDM 子集**:把贡献 ID 换成 `ami_sdm_en_zh_*`,索引路径的 `source_subset` 换成 `sdm`。

### 3.6 从逐句数据重建会议级多说话人同传样本

产物全部写到 `data/ami_multispk_si_v1/`。先写新文件,不覆盖已有的。

**1. 会议级源音频**
- **SDM**:本机已有 169 场完整会议,`/opt/dlami/nvme/jasonleeeli/datasets/auden_ami_sdm/wav_db/{meeting}/audio/{meeting}.Array1-01.wav`(另有 2 场缺失)。
- **IHM-mix**:各人头戴麦相加。需要 AMI 原始头戴麦音频(Lhotse `ami` recipe 的 `ihm-mix`),或从逐句 IHM 拼回去。拼回去只覆盖有转写的片段,会丢掉背景,不推荐作为主要来源。
- 两者都用:SDM 作为主要训练和测试条件(和 SCOPE 一致),IHM-mix 作为测试条件(SCOPE 训练时没见过)。

**2. 时间轴**
- 每场会议按 `begin_time` 排序,得到监督列表:说话人、起止时间、英文、中文、目标语音。
- 词级时间用本机的 AMI 官方标注 `ami_public_manual_1.6.2.zip`(`words/*.words.xml` 共 688 个文件,带词级起止时间)。不需要再做强制对齐。

**3. 切成训练片段**
- 采用 TagSpeech 和 Lhotse 的做法:`trim_to_supervision_groups(max_pause=0)`,把互相重叠或紧邻的句子并成一组。
- 一组超过 20 s 时,在说话人换人处切开。
- **片段内每个说话人的句子都是完整的**,和逐句译文一一对应,不会切断句子。

**4. 片段内每个说话人的目标**
- 该人在片段内各句译文按时间拼接 = 这个说话人的整段目标。
- 每句译文都带源时间,用来:
  - 构造块级的源时间标签(§4.6 F7);
  - 给前缀截断样本提供对齐。

**5. 块级对齐**:给"读到时刻 t 时,应当已经说出多少译文"提供参考。
- 做法 a:沿用 `multi_speaker_pipeline_share` 的 LLM 双语切块(chunk_segmentation)。
- 做法 b:用词对齐(英文词时间 + 中英词对齐)按单调块合并。
- 产物:每个块的 (源结束时间, 译文块)。
- 用途:前缀截断训练(§5 P2)和同传参考布局(第 6 步)。

**6. 目标语音**
- **UniSS Talker 路线**:把 Lance 的目标 wav 用 BiCodec 重新编码,得到语义 token 和 32 个全局 token。
  - 全局 token 不用目标 wav 的,改用"该说话人在源会议里的贝叶斯音色记忆",让同一个人在整场里音色一致。
- **Qwen3-TTS codec 路线**(方案的可选分支):直接用 `target_codec_codes`,需要配套的 12 Hz Talker 或 decoder。
- **会议级参考布局**(用于试听和 Talker 时间轴训练):沿用 `multi_speaker_pipeline_share/layout.py` 的 δ 摆放。
  - 块放在"源结束 + δ × 源时长";
  - 超出预算就加速,保持不变调;
  - 目标块之间严格不重叠,即单通道译员。
  - 输出 `alignment[]`、`mix.wav`、`overlay.wav`。

**7. 质量过滤**(初始阈值,在 validation 上看分布后再定):
- Whisper CER ≤ 0.2;
- UTMOS ≥ 3.0;
- ECAPA ≥ 0.3;
- 任何一项为空时不过滤,但打标记,在消融里单独看。

**8. 短附和词的策略**(需要决定,作为消融):AMI 有大量 OKAY、YEAH、UM 这类短句。
- 策略 i:全部翻译;
- 策略 ii:≤2 词的附和不翻译(译员常见做法);
- 策略 iii:保留但损失权重 0.3。

默认用 ii 训练、两种口径都评测,因为 SCOPE 的数据显示重叠中的短附和最难。

**9. 分离后验**
- 本机没有 NeMo,不安装。
- AMI 各场会议的 Nemotron 流式后验在 VUT 集群上已有 dump(`gstar_nemotron/nemotron/`)。把 `.npy` 或 RTTM 拷贝到 `data/ami_multispk_si_v1/diar/`。
- 训练时用这些真实后验,并按 SCOPE 的方式扰动(概率 0.2)。
- 另备一份参考分离(0/1 后验),用于上限对照。

**10. 清单**
- `manifests/{train,dev,test}_{sdm,ihmmix}.jsonl`,每行一个片段:

  ```text
  meeting, start, dur, audio, speakers:[{id, slot, utts:[{begin,end,en,zh,tgt_audio,blocks}]}], diar
  ```

- 训练前全量 shuffle,用固定种子,并记录在 manifest 头。

**规模估计**:AMI train 约 80 h 会议音频,按句组切片约 2–3 万个片段。每个片段平均 1.5–2 个说话人,所以条件链样本约 4–6 万个。

---

## 4. 架构设计

### 4.1 SCOPE 中可借鉴的部件与迁移方式

| SCOPE 部件 | 作用 | 搬到同传后的形态 | 需要训练? |
|---|---|---|---|
| Nemotron-3 流式分离 + AOSC | 谁在说、全局槽位 | 原样使用(1.04 s 前瞻,与同传的读取窗口相容) | 否 |
| 因果切窗(1 s 停顿关窗,≤8 s) | 分段 | 同传"片段"分段器(≤20 s,带跨片段上下文);片段内每 Δ 秒读一次 | 否 |
| Platt 校准 + STNO 置信度 | 目标与他人的关系 | 原样(2 个标量/数据集,在 AMI dev 上拟合) | 否 |
| BSSC | 编码器"只听目标" | AuT 32 层 × 2 矩阵,r=32,G 零初始化 | 是(8,192 参数) |
| sd_proj 活动 token | LLM 知道谁在说 | 原样形状(Conv5+MLP → 2048) | 是 |
| DSS 特征划分 | 投影后特征按人分 | 原样形状(2048),ω=1 种子 | 是 |
| 目标嵌入 u_k | 指定要译谁 | 原样(2048) | 是 |
| 条件链 | 后译的人可以"解释掉"已分配的词 | 每次读取时按活动量排序,逐个说话人续写译文 | 是(LoRA) |
| 跨窗口上下文(30 词) | 长对话连贯 | 片段前最近 30 个**译文**词 + 该说话人已提交的前缀 | 是 |
| 束搜索 + D(h) 重排 | 识别 × 分离联合 | 对提交候选重排:D 用候选所覆盖源帧的 log-odds | 否(τ、λ 在 dev 上估) |
| 逐词 HMM 归属 | 纠正说话人标签 | 改为**逐块**归属:块 = 一次提交;似然用块对应源帧 | 否 |
| 序贯注册 | 去掉误开的槽位 | 原样;被拒槽位的译文块重新归属,未播放的块取消 | 否 |
| (最新)b6d、spk 自注册 | 重叠帧区分 A 和 B | 作为方案 B 的扩展消融 | 是 |

### 4.2 方案 A:级联基线(作参照,不作主线)

```
mixture → Nemotron → SCOPE(说话人归属转写,约 7 s 定稿)
        → 按说话人的文本同传或 MT(Qwen3 文本模式 / Hy-MT2)
        → 按槽位音色的 TTS(UniSS Talker 或 VoxCPM2)
```

- **优点**:
  - 现成部件最多,重叠区识别直接继承 SCOPE 的最好水平;
  - 可作为译文质量的"高延迟参考"。
- **缺点**:
  - 延迟约 7 s + MT + TTS,不是同传;
  - SCOPE 的 checkpoint 在 VUT 集群,需要在那边跑识别,再把结果拷过来。
- **用途**:质量上限参照,也用于做错误分析(识别错还是翻译错)。

### 4.3 方案 B(推荐主线):SCOPE-SI,把 SCOPE 的 ASR 分支换成同传 Thinker

```
           ┌────────────── Nemotron-3 流式分离(冻结)──────────────┐
mixture ──►│ P_dia ∈ [0,1]^{T×8}  →  Platt 校准  →  STNO b_t^(k)    │
           └──────────────┬───────────────────────┬────────────────┘
                          │                       │
                          ▼                       ▼
  AuT 编码器(32 层,冻结 + LoRA r=32)    sd_proj:Conv5+MLP → 2048 维活动 token
  每个 FFN:BSSC  g_t = Gᵀ b_t^(k)             │
                          │                       │
                          ▼                       │
  投影 2048 维 → DSS(按活跃槽位软划分,突出 k) ─┤
                          │                       │
                          ▼                       ▼
  Qwen3-Omni Thinker LLM(冻结 + ABC LoRA)  输入 = [prompt: 跨片段 30 词译文;
                                              本次读取里已轮过的说话人 "spk j: 已提交译文";
                                              说话人 k 已提交的前缀] + 音频 token(每 5 个插 1 个活动 token) + u_k
                          │
                          ▼
  对每个活跃说话人 k(本次读取内按活动量降序,条件链):
    续写草稿(束宽 4)→ D 重排 → ABC 提交判定(ρ 与阈值;SDR 用**共享**语音队列)
                          │
                          ▼
  已提交块 {k, 源时间, 文字} → 块级贝叶斯归属 / 序贯注册 → Talker
  (按槽位的贝叶斯音色记忆,证据只取 b^T 高的帧)→ 输出调度器 → 译音流
```

**关键设计点**

1. **读取与分段**
   - 片段由因果分段器给出,沿用我们长音频同传的切法(停顿切,≤20 s,带上下文)。
   - 片段内每 Δ 读一次,沿用 m2 的 Δ:英→中 0.5 s。
   - 每次读取时重新编码片段内全部已听音频,和现在的 selfwait 一样。
2. **活跃说话人**
   - 片段内到目前为止活动 ≥0.2 s 的槽位,即 SCOPE 的开轮门槛。
   - 一次读取里,只对"上次读取以来有新活动"的槽位续写。其余槽位跳过,省算力。
3. **每个说话人一条独立的提交状态**:已提交前缀、ρ、等待计数。
   - ABC 的提交判定逐说话人做;
   - SDR 的队列 q 是**全体说话人共享**的,因为单通道译员只有一张嘴;
   - 队列满时,所有人都更倾向于等待,这和真人译员压缩、省略次要发言的行为一致。
4. **条件链**:本次读取里先轮到的说话人的**已提交**译文写进后面说话人的 prompt。只写已提交的,不写草稿,避免把不稳定的文字传下去。这让次说话人不重复主说话人已经译过的内容。
5. **计算量**
   - 每次读取的 Thinker 前向数 = 本次有新活动的说话人数 K_r。AMI 上大部分时间 K_r = 1,重叠时为 2–3。
   - 按单人约 58 ms 估算,K_r=3 时约 0.18 s,低于 Δ=0.5 s。
   - 同一读取的 K_r 个说话人可以拼成一个 batch:音频相同,BSSC 的置信度不同,所以编码器也要逐说话人算。AuT 只有约 0.6B,开销可以接受。
6. **对现有 m2 的兼容**
   - 当分离器只给出一个槽位时,STNO 退化为"独自说或静音"。
   - G 零初始化、DSS 门控 ζ=0,所以初始模型和 m2 完全一样。
   - 单人测试集(RealSI、ACL)用来检验不退化。

### 4.4 方案 C:序列化多说话人同传(SOT-SI)

- **做法**:单路解码,输出 `<spk_k> 译文块` 和 wait 标记的交错序列。
  - 输入同样有 sd_proj 活动 token;
  - 编码器不能按单个目标调节,改用说话人无关的条件(例如 STNO_IMPROVE 的 b6d 中"人数、主导度"那部分)。
- **优点**:
  - 每次读取只有一次前向,计算量和 m2 相同;
  - 天然是单通道译员的输出顺序。
- **缺点**:SCOPE 的经验是,SOT 在重叠区明显不如条件链加 BSSC(cpWER 差约 3 个点),次说话人容易被吞掉。
- **用途**:低算力部署的备选,也是方案 B 的直接对照(相当于 SCOPE 论文里的 SOT-LLM 基线)。

### 4.5 方案 D:混合(SOT 主流 + 重叠时触发条件链补译)

- **做法**:平时跑方案 C 的单路。当分离器在最近 Δ 内检测到两个以上的槽位都有显著活动时(例如 b^O > 0.5 持续 ≥0.3 s),对次说话人追加一次方案 B 式的目标说话人续写。prompt 里放主流已提交的译文,让它"解释掉"已译的内容。
- **优点**:平均计算量接近 C,重叠区能力接近 B。
- **风险**:两套解码状态的一致性(同一说话人在主流和补译里都出现时如何合并)。合并规则:同一说话人以补译为准,主流中与补译时间重叠的该人片段撤销(未播放时)。

### 4.6 特征融合:逐项说明

记号:
- 片段 w 的 AuT token 数 N_w(约 12.5/s);
- 槽位数 S=8;
- 目标说话人 k。

| 编号 | 融合位置 | 形式 | 维度 | 说明 |
|---|---|---|---|---|
| F1 | 输入 token 层 | sd_proj:P_tok = 𝓘(P_dia) ∈ [0,1]^{N_w×8},Z = MLP(Conv5(P_tok)) | ⌈N_w/5⌉ × 2048 | 每 5 个音频 token 后插 1 个活动 token,LLM 看到所有人什么时候在说 |
| F2 | 编码器内部(每层 FFN) | BSSC:y_t = (W+ΔW_LoRA)x_t + U_r diag(σ_r ⊙ g_t) V_rᵀ x_t,g_t = Gᵀ b_t^(k) | G ∈ R^{4×32},每个矩阵一个,共 64 个 | 逐 token 按"目标独自说、他人在说、重叠、静音"放大或衰减奇异方向 |
| F3 | 投影后 | DSS:e'_t = e_t ⊙ (1 + tanh(ζ)(K M_{t,k} − 1)),M_{t,·,c} = softmax_j(A e_t + η_j + p̃_{t,j} ω)_c | A ∈ R^{2048×2048},ω 初始化为 1 | 投影后的 2048 维特征按人软划分,突出目标 |
| F4 | 输入末尾 | 目标嵌入 u_k(可学习,每槽位一个) | 2048 | 告诉 LLM 这一轮译谁 |
| F5 | prompt 文本 | 跨片段 30 词译文 + 本次读取已轮过说话人的已提交译文("说话人 j: …")+ 目标已提交前缀 | 文本 | 条件链,避免重复和遗漏 |
| F6 | 编码器内部(扩展) | 自注册说话人向量 e_k:第 8 层上取 b^T>0.5 的 token 求均值;后面各层 g_t += (b^T+b^O)·tanh(A_s e_k) | A_s ∈ R^{32×1280} | 重叠帧上区分 A 和 B(STNO_IMPROVE X2),作为消融 |
| F7 | 输出侧 | 每个提交块附带源时间 token ⟨t_a⟩⟨t_b⟩(80 ms 网格,可选) | 251 个时间 token | 让 D 重排和块级归属知道这块译的是哪段源音频。训练目标来自 AMI 词级时间 |
| F8 | Talker | 按槽位的贝叶斯音色记忆 θ_j,证据权重 w = Σ b^T_{t,j}·0.08 s(只用目标独自说的帧),30 s 后冻结 → BiCodec 32 个全局 token | 32×128 潜变量 | 每个人的译音保持其本人音色,重叠帧不污染音色 |
| L1 | 后期(贝叶斯) | 提交候选重排:argmax_h log p_θ(h) + (λ/τ) Σ_{f∈F(h)} logit p_{f,k} | 标量 λ、τ | F(h) 来自 F7;没有 F7 时用"上次提交到本次读取之间的帧" |
| L2 | 后期(贝叶斯) | 块级归属 HMM:隐变量为块的槽位 z_b,似然为 (1/τ)Σ_{f∈F_b} logit p_{f,j},标签模型为混淆矩阵 C,转移 ε | 2S 状态前向–后向 | 方案 C、D 必需;方案 B 用来纠正分离器的换位 |
| L3 | 后期(贝叶斯) | 序贯注册:新槽位为真的后验概率一旦离开 [α, 1−α] 就判定 | α、κ | 防止误开槽位把译文发给"幽灵说话人" |

融合顺序的建议(也是训练的增量顺序):

1. F1 + F4 + F5,也就是条件链本身;
2. 加 F2(BSSC);
3. 加 F3(DSS);
4. 加 F7(时间 token);
5. 加 F6,作为扩展。

每加一项都在 dev 上做配对 bootstrap。F8 和 L1–L3 不需要训练,可以和 1–5 并行做。

### 4.7 最终结果如何得到(推理流程)

每次读取(时刻 t,间隔 Δ):

1. **分离更新**:Nemotron 读入新音频,输出到 t+1.04 s 为止的后验。算 Platt 校准后的 p̂ 和每个槽位的 STNO 置信度。
2. **分段**:当前片段是否关闭(1 s 停顿或达到 20 s)。关闭时,各说话人的剩余草稿走"片段末尾强制提交"。
3. **活跃说话人**:片段内活动 ≥0.2 s,且上次读取以来有新活动的槽位,按片段内活动量降序排列。
4. **逐说话人续写**:对每个 k,依次:
   - 编码(BSSC 用 b^(k),DSS 突出 k);
   - 拼 prompt(F5);
   - 束宽 4 续写草稿;
   - L1 重排;
   - ABC 计算 ρ 并与阈值比较。阈值由 SDR 根据**共享队列** q 和 Δ 决定。
   - 若提交,得到块 {槽位 k, 源时间 F7, 文字, 提交时刻 t}。
5. **注册与归属**:
   - 新槽位进入序贯检验,未注册槽位的块先缓存不播;
   - L2 在片段关闭时对该片段的块做一次前向–后向,纠正明显的换位;
   - 已经播出的块不撤回,只在文本记录里更正。
6. **Talker**:每个提交块用该槽位的音色记忆 θ_k 合成。音色记忆用 b^T 加权的源音频更新,30 s 后冻结。
7. **输出调度**:
   - **单通道译员**(默认):提交顺序的 FIFO 队列,块的播放时刻 = max(提交时刻, 上一块结束时刻);换说话人时插入 0.2 s 停顿。队列积压由 SDR 反馈到下一次提交判定。
   - **多轨配音**(可选):每个槽位一轨,允许和源一样重叠,最后混音或分轨输出。
8. **最终产物**(每场会议):
   - `transcript.json`:每块的 {槽位, 注册后的全局说话人, 源时间, 提交时刻, 译文, 播放开始, 时长, 音色相似度};
   - `translation.wav`:单通道译音;
   - `stereo.wav`:左原声、右译音;
   - 可选的分轨 `spk_{j}.wav`;
   - 输出说话人的 RTTM。
   - 同时给出 fluency 提交格式,和今天交付的 m2_fluency 同格式,便于评测停顿。

### 4.8 方案对比

| | A 级联 | **B SCOPE-SI(推荐)** | C SOT-SI | D 混合 | E 双分支(可选) |
|---|---|---|---|---|---|
| 延迟 | 约 8–10 s | 约 1.7–2.2 s(m2 + 分离 1.04 s 前瞻部分重叠) | 约 1.7 s | 约 1.7–2.0 s | 译音约 2 s;字幕约 7 s |
| 每次读取的 Thinker 前向 | — | K_r(多为 1,重叠时 2–3) | 1 | 平均约 1.1 | K_r + SCOPE |
| 重叠区能力 | 继承 SCOPE | 强(BSSC + 条件链) | 弱 | 中到强 | 强 |
| 新增训练 | 无 | BSSC、sd_proj、DSS、u_k、LoRA、ABC 重训 | sd_proj、LoRA、ABC | B 和 C 都要 | B |
| 工程量 | 低(要跨集群) | 高(AuT 前向要接 BSSC,vLLM 要自定义) | 中 | 最高 | 最高 |
| 主要风险 | 延迟不达标 | vLLM 里逐请求传入置信度 | 吞掉次说话人 | 两套状态合并 | 资源 |

**方案 E(双分支)**:SCOPE 的 ASR 分支保留,并行运行(在 VUT 或另一张卡上)。它给出说话人归属的源语转写,用于:
1. 会议纪要和字幕;
2. 作为后续片段的跨片段上下文,替换 F5 里的译文上下文,也就是"源语言上下文";
3. 离线复核和打分。

同传分支照方案 B 运行。两个分支共享 Nemotron 前端和 STNO 计算,编码器不共享(分别是 Qwen3-ASR 和 Qwen3-Omni 的 AuT)。

---

## 5. 训练方案

总原则:
- 用 Megatron(沿用 `pretrain_thinker_sd.py` 和 `run_thinker_sd.sh` 的 Thinker 适配器训练入口;Talker 用 `pretrain_e2` 系列)。
- 数据全量 shuffle,GPU 保持满载。
- 每条训练线用新的 RUN_ID 和新目录,不覆盖已有实验。
- 每一阶段都有通过判据,不通过就停。

### P0 数据与零训练基线(无训练)

1. **读取数据**:pylance 隔离安装需确认;§3.5 脚本读 v239 的 AMI ihm 和 sdm en→zh,join 处理层索引,核对三件事。
2. **构造会议级清单**:§3.6 的 1–10 步,产出 `data/ami_multispk_si_v1/`。
3. **分离后验**:从 VUT 拷贝 AMI 各 split 的 Nemotron 流式后验,同时生成参考分离 0/1 后验。
4. **实现评测指标**(§6.2):cpBLEU、cpCOMET、块级归属准确率、按人音色相似度和音色混淆率、分重叠和单人的指标。
5. **零训练基线**:
   - **B0**:现有 m2 直接跑整段混音,不分说话人,用单一音色记忆。
   - **B1**:m2 + Nemotron + L2 块级归属 + L3 注册 + F8 按槽位音色记忆。模型不变,只加后处理和音色。
   - **B1-oracle**:B1 用参考分离,作为分离误差的上限对照。

**判据**:B1 相对 B0 的说话人归属准确率和音色混淆率明显改善,cpBLEU 不降。这一步本身就是第一个可交付的多说话人同传系统。

### P1 会议领域适配(单人,逐句 IHM)

- **数据**:AMI ihm train 逐句(单说话人,源为头戴麦),经 §3.6 第 7 步过滤,约 10 万句。
- **训练**:
  1. 在 AMI 上重新生成 ABC 的自标注。稳定长度 S_k = lcp(读前缀时的草稿, 完整句译文);"完整句译文"取 Thinker 自己的整句译文,和训练 m2 时一致。Lance 的 `target_text` 作为对照标签源。
  2. 继续训练 ABC 的 LoRA,只训适配器,和 m2 相同。
- **判据**:AMI ihm test 逐句同传的 BLEU 和 LAAL 曲线不劣于 m2;RealSI、ACL 回归测试不降超过 0.3。

### P2 说话人条件化(方案 B 的核心)

- **数据**:AMI SDM train 片段(§3.6),条件链样本按 SCOPE 方式构造:
  - 每个样本是一轮:随机取片段和轮次 i,前 i−1 个说话人的参考译文放进 prompt;
  - 以 0.1 的概率取弱活动、无参考的槽位,目标为空;
  - prompt 词以 0.1 的概率丢弃,后验以 0.2 的概率扰动;
  - **前缀截断增强**:以 0.5 的概率把片段音频截到某个读取时刻 t,目标截到 §3.6 第 5 步给出的、t 之前应已说出的块。让模型在前缀上的草稿也是"该人的"译文。
- **可训练模块**:
  - BSSC(G,64 × 4 × 32),sd_proj,DSS(A、η、ω、ζ),u_k;
  - AuT 编码器 LoRA r=32;
  - LLM 注意力 LoRA:沿用 ABC 的层范围,r=32。
- **起点**:P1 的模型。G=0、ζ=0 时与 P1 等价,训练前先核对。
- **超参起点**(取自 SCOPE,按我们的 batch 调):
  - lr 2e-5,warm-up 500,cosine;
  - 编码器 LoRA 和 DSS 为 5 倍 lr,G 和 ζ 为 50 倍 lr;
  - 损失:译文 token 1,时间 token 1.5(若用 F7),wait 标记沿用 ABC 的权重;
  - 步数先 5k,看 dev 曲线再决定是否到 20k。
- **增量消融顺序**(各自独立 RUN_ID,dev 上配对 bootstrap):
  1. 条件链(F1+F4+F5);
  2. + BSSC;
  3. + DSS;
  4. + F7;
  5. + F6(spk)或 b6d。
- **判据**:相对 B1,dev 上重叠区 cpBLEU 提高 ≥2 BLEU,单人区变化在 ±0.3 以内;RealSI、ACL 回归不降超过 0.3。

### P3 多说话人提交策略(ABC 重训 + SDR 共享队列)

- 在 P2 的模型上,对每个说话人的前缀草稿重新生成稳定长度自标注,训练 ABC。
- SDR 改为共享队列,速率 r 用英→中的 0.191 s/字。
- 扫描 Δ ∈ {0.5, 1},b0 ∈ {0, 1, 2},在 dev 上画质量–延迟曲线。
- **判据**:在相同 StreamLAAL 下,cpBLEU 不低于 P2 的固定阈值版本,并且曲线整体支配 B1。

### P4 贝叶斯解码与输出调度(无训练)

- 在 AMI dev 上拟合以下参数:
  - L1 的 τ、λ;
  - L2 的 ε 和 C;
  - L3 的 α 和 κ;
  - Platt 的 a_d 和 c_d。
- 逐项消融(test 配对 bootstrap):束搜索、D 重排、块级归属、注册、按槽位音色、单通道和多轨。

### P5(可选)Talker 会议适配

- **数据**:Lance AMI 目标语音用 BiCodec 重编码。全局 token 用按槽位的音色记忆,不用逐句克隆音色,以去掉句间不一致。
- 小步数微调 T1c(Megatron,`pretrain_e2` 入口)。
- **判据**:按人音色相似度提升、ASR-CER 不升,RealSI 回归不降。

### 资源估计

| 阶段 | 资源 | 耗时 |
|---|---|---|
| P0 数据 | CPU 为主 | 1–2 天 |
| P1 | 8 卡 | 约 0.5 天(与 m2 的 ABC 训练同量级) |
| P2 | 8 卡,每条消融线 0.5–1 天 | 约 3–5 天,共 5 条线,可两两并行 |
| P3 | 8 卡 | 1 天 |
| P4 | CPU 和少量 GPU 推理 | 1–2 天 |
| P5 | 8 卡 | 1 天 |

---

## 6. 测试方案

### 6.1 测试集

| 测试集 | 用途 |
|---|---|
| AMI test(16 场会议,标准 full-corpus 划分),SDM 和 IHM-mix 两种条件 | 主测试 |
| AMI dev(18 场) | 调参和选 checkpoint |
| AMI ihm test 逐句(12,177 句) | 单说话人条件下的领域内同传 |
| RealSI 英→中、ACL60/60-dev 英→中 | 单说话人回归 |

需要注意:
- SCOPE 报告指出,早期 ihm-mix test 24 场里有 8 场属于 AMI train,Nemotron 训练时见过。本方案只用标准的 16 场。
- SCOPE 本身的训练集(AMI-SDM train + NSF train)和 AMI test 不重叠。

**参考**:
- 每个说话人的参考译文 = 该人在会议里各句的 Lance `target_text`,按时间排列,带句级时间;
- 英文源参考 = AMI 转写。

### 6.2 指标

| 类别 | 指标 | 定义 |
|---|---|---|
| 带说话人的译文质量 | **cpBLEU / cpchrF / cpCOMET** | 和 cpWER 同思路:系统槽位和参考说话人用 Hungarian 一一匹配(按时间重叠,或按匹配后总 chrF 最大);每对把系统流用 mwerSegmenter 对参考句重切分后计入语料级 BLEU 和 COMET。没匹配上的系统槽位与空参考配对,计入精度惩罚;没匹配上的参考与空输出配对 |
| 不分说话人的译文质量 | BLEU / COMET + StreamLAAL | 整场输出对全部参考做 mwerSegmenter 重切分,和我们 ACL60 长音频口径相同(`si_streamlaal.py`) |
| 语音质量 | ASR-BLEU(cp 和不分说话人两种) | paraformer-zh 在纯语音块上转写,避免长静音导致 ASR 幻觉 |
| 说话人归属 | 块级归属准确率;说话人数完全正确率;输出 DER(可选) | 块的源时间(F7 或读取窗口)与参考说话人时间重叠最大者为真值 |
| 音色 | 按人 SIM;音色混淆率 | SIM = WavLM-TDNN 余弦(输出块,该人的 IHM 参考音);混淆率 = 输出块的音色离别人比离本人更近的比例;也报会话级质心相似度 |
| 延迟 | StreamLAAL(文本,按人和整体);首包和尾包偏移;队列滞后 | 也分重叠区和单人区报告 |
| 重叠专项 | 重叠区和单人区的 cpBLEU;次说话人覆盖率(该人参考的 chrF 召回);短附和(≤2 词)的覆盖率 | 对应 SCOPE 的已知短板 |
| 流畅度 | fluency 提交包(audio、results.jsonl、asr_lat.jsonl) | 与今天交付的 m2_fluency 格式一致 |

**显著性**
- 按会话的配对 bootstrap,1000 次。
- 会话只有 16 场,另外做"会话内按片段分层"的 bootstrap 作为补充,两者都报。

### 6.3 对照与消融矩阵(test)

| 组 | 系统 |
|---|---|
| 基线 | B0(m2 原样)、B1(m2 + 分离后处理 + 按槽位音色)、A(级联,质量参照)、C(SOT-SI) |
| 主系统 | B = SCOPE-SI(P2 + P3 + P4 全系统) |
| 条件化消融 | −BSSC、−DSS、−条件链(改为独立的目标说话人解码)、BSSC 随机基、F6 或 b6d 扩展 |
| 贝叶斯消融 | −束搜索、λ=0、−块级归属、−注册 |
| 输出消融 | 单通道和多轨;按槽位音色和全局音色;短附和策略 i、ii、iii |
| 上限 | 参考分离(0/1 后验);参考分段 |
| 回归 | RealSI、ACL60 上 B 与 m2 的差 |

### 6.4 通过判据(最终)

1. AMI test(SDM 和 IHM-mix)上,B 的 cpBLEU 显著高于 B1,并且平均 StreamLAAL 增加 ≤0.3 s。
2. 重叠区 cpBLEU 提升 ≥2,次说话人覆盖率提升 ≥5 个百分点。
3. 音色混淆率显著低于 B0;按人 SIM 不低于单人场景的 bayes freeze30 水平(约 0.83)。
4. RealSI、ACL60 上与 m2 的差在 ±0.3 BLEU 以内。

---

## 7. 风险与开放问题

| 风险 | 影响 | 应对 |
|---|---|---|
| pylance 未安装 | 无法按 md 读取 | 确认后装在隔离目录;备选:直接读处理层 parquet 索引 + WDS tar(索引里有 shard_uri、byte_offset、byte_length) |
| `source_sample_id` 与 `record_id` 不同形 | join 失败 | 改用 `original_record_id` 或 `audio_id`,解析会议和时间 |
| NeMo 和 Nemotron 不在本机 | 拿不到分离后验 | 在 VUT 集群算好并拷贝后验;不在本机装 NeMo |
| SCOPE checkpoint 在 VUT 集群 | 方案 A、E 需跨集群 | A 只作参照;B 只借鉴设计和代码,在本机从头训练模块 |
| vLLM 中 AuT 前向要按请求传入置信度(BSSC) | 部署工程量 | 先用 HF 或 Megatron 前向做离线评测;再写 vLLM 自定义模型类,把置信度作为多模态附加输入 |
| 逐句克隆的目标语音音色不一致 | Talker 训练有噪声 | 用按槽位的音色记忆生成全局 token,并按 ECAPA 过滤 |
| 逐句离线译文不是同传参考 | 前缀目标难定 | 沿用 ABC 自标注;块级对齐只用于前缀截断增强 |
| 分离器漏检(AMI-SDM 漏检 15.8%) | 漏译 | 报告参考分离上限;F7 和 L1 用 log-odds 宽容弱活动;考虑 STNO_IMPROVE 的漏检模拟 |
| 计算量随说话人数增长 | 实时性 | 只对有新活动的槽位续写;batch;方案 D 作为降级 |
| 只有英→中(AMI 只有英语源) | 中→英多说话人缺数据 | 请数据组对 AliMeeting 跑同一条 speech_chain |
| 测试会话少(16 场) | 显著性弱 | 会话和片段两层 bootstrap;同时报 SDM、IHM-mix 两个条件 |

需要你决定的开放问题:
1. 短附和词的翻译策略(§3.6 第 8 步),默认 ii。
2. 默认输出形式:单通道译员,还是多轨配音。默认单通道。
3. 是否需要 F7 源时间 token(提升贝叶斯部件的精度,但增加输出长度)。
4. 是否同时做 en→ja、en→ko(数据已在 v239 里)。

---

## 8. 里程碑

| 里程碑 | 内容 | 交付 |
|---|---|---|
| M0 | P0 全部 + B0、B1 在 AMI dev 和 test 的结果 | `data/ami_multispk_si_v1/`、评测脚本、B1 多说话人同传(无需训练即可演示) |
| M1 | P1 会议领域适配 | ABC-AMI 模型、单人回归表 |
| M2 | P2 条件化 5 条消融线 | SCOPE-SI 模型、dev 消融表 |
| M3 | P3 + P4 全系统 test | 主表、消融表、bootstrap |
| M4 | P5(可选)+ 试听页 + fluency 包 | Talker 适配、试听网页、报告 |

---

## 附录 A:路径清单

| 内容 | 路径 |
|---|---|
| Lance 表 | `s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/curated/task=s2st/data.lance/`(v239) |
| v239 提交标记 | `.../_meta/curated/task=s2st/table=data/releases/lance_version=239/_COMMITTED.json` |
| AMI 处理层索引(会议、时间、译文) | `s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/processed/index/dataset=ami/process_version=speech_chain_all_local_en_zh_v1/source_subset={ihm,sdm}/split={train,validation,test}/source_lang=en/target_lang=zh/` |
| AMI 源索引(16 kHz 单声道) | `.../processed/index/dataset=ami/process_version=clean_16k_mono_v1/source_subset={ihm,sdm}/split=*/` |
| AMI 处理链提交 | `.../_meta/commits/layer=processed/dataset=ami/process_version=speech_chain_all_local_en_zh_v1/` |
| 本机 AMI SDM 整场音频 | `/opt/dlami/nvme/jasonleeeli/datasets/auden_ami_sdm/wav_db/{meeting}/audio/{meeting}.Array1-01.wav`(169 场) |
| 本机 AMI 官方标注(词级时间) | `/opt/dlami/nvme/jasonleeeli/datasets/auden_ami_sdm/ami_public_manual_1.6.2.zip` |
| 多说话人数据管线参考 | `/opt/dlami/nvme/jasonleeeli/projects/multi_speaker_pipeline_share/` |
| SCOPE 代码(只读) | `github.com/happyjin/Auden`,分支 `tagspeech/gstar-nemotron`,目录 `examples/tagspeech/gstar_nemotron/`(robust/、qwen3/、beam/、bayes2/) |
| SCOPE 论文 | 同分支 `examples/tagspeech/paper_icml_v2/` |
| 我们的 m2 同传 | `docs/deploy/stream_s2st_bayes_deploy.zh-CN.md`、`experiments/uniss_qwen3_omni_v1/scripts/run_stream_s2st.sh` |
| 新数据和产物 | `data/ami_multispk_si_v1/`(训练数据);`eval_outputs/uniss_qwen3_omni_v1/ami_multispk_*`(评测) |

## 附录 B:本文已核实和未核实的事实

**已核实**(2026-10-26,直接读取 S3 元数据、处理层 manifest 和 parquet、HF datasets-server、分支源码和报告):
- v239 含 18 个 AMI 贡献,以及各 split 的行数;
- 生成链路的模型;
- 处理层索引里有会议号、说话人、起止时间;
- Lance 列中没有这些字段;
- Qwen3-Omni AuT 的层数和维度;
- SCOPE 的结果和配方。

**未核实,实施时先验证**:
- Lance 的 `source_sample_id` 与处理层 `record_id` 是否同形;
- 目标音频的采样率;
- AuT 实际的 token 率(按 12.5/s 设计);
- VUT 集群上 AMI 的 Nemotron 后验 dump 是否覆盖 train、dev、test 三个 split。
