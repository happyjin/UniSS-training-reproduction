# 多说话人语音到语音翻译完整方案 v5

**Nemotron-3 说话人分离 + 同传大模型,端到端,先离线后同传**

- 日期:2026-10-26(v1–v4 保留不动)
- 数据:AMI 英→中;只用 IHM-mix 通道;只用审查通过的数据
- 不用 SCOPE 的贝叶斯推理;需要装包时只装进一个专用环境

---

## 0. 需求和对应的设计

| # | 你的需求 | 本方案的做法 |
|---|---|---|
| 1 | 必须接 Nemotron-3 说话人分离,用它得到"谁在什么时候说话"的时间戳 | Nemotron-3-Diarization 作为模型的**说话人分离支**(冻结)。它输出:<br>① 80 ms 一帧的说话人后验(8 个槽位,整场会议槽位一致);<br>② "谁在什么时候说话"的片段(RTTM)。<br>后验作为 token 进大模型;片段用来切窗和对齐时间 |
| 2 | 另一支是目前最好的同传大模型 | m2 的 Qwen3-Omni Thinker(ABC + SDR)作为**同传支**:AuT 编码器给出音频 token,LLM 负责翻译和"何时开口" |
| 3 | 两支一起经过大模型,输出与 GitHub 分支(G-STAR)相同格式 | 说话人 token 和音频 token 按 K:1 交错(和分支 G-STAR 完全一样)进同一个 Thinker。输出 `<\|t起\|>译文<\|t止\|><\|spk k\|>`;会议级结果导出成分支同款的 XML / JSON |
| 4 | 端到端 | 一个模型图:分离支(冻结子网络)→ SD 投影器 → 交错 → LLM → Talker。谁、何时、说什么、现在说不说都由网络输出。没有"转写→翻译"的文字接口,没有贝叶斯后处理 |
| 5 | 多说话人的语音到语音翻译 | Talker 按说话人槽位用各自的音色合成译音,再合成单通道(同传)或按源时间排布(离线配音) |
| 6 | 先给 baseline | baseline = **G-STAR-SI**。先做离线版(阶段 O),再转同传(阶段 S),两个阶段同一结构、同一格式 |

---

## 1. 系统总览

```
                    ┌──────────────── 说话人分离支(冻结)────────────────┐
                    │ Nemotron-3-Diarization(Streaming Sortformer)        │
 会议单通道音频 ──┬►│  离线配置:整场一次   /   流式配置:1.04 s 前瞻       │──► 后验 P ∈ [0,1]^{T×8}(80 ms)
 (IHM-mix 16 kHz) │ │  输出 ① 后验  ② RTTM:谁在什么时候说话(整场槽位一致) │──► RTTM:切窗 + 时间参照
                  │ └──────────────────────────────────────────────────────┘
                  │                                 │ 后验插值到音频 token 网格
                  │                                 ▼
                  │                     SD 投影器(Conv1d k=5 s=5 + MLP)→ 说话人 token s_j ∈ R^2048
                  │ ┌──────────────── 同传大模型支 ────────────────────────┐        │
                  └►│ m2 Thinker 的 AuT 编码器(32 层,冻结)→ 投影到 2048    │──► 音频 token a_i(约 13/s)
                    └──────────────────────────────────────────────────────┘        │
                                                                          K:1 交错 ▼
                        <|audio_start|> a0 a1 a2 a3 a4 s0 a5 … a9 s1 … <|audio_end|> + 指令 + 上文
                                                     │
                                  Thinker LLM(m2 底座 + LoRA,48 层 MoE)
                                                     │
                      输出:<|t起|>译文<|t止|><|spk k|> … [同传时每次读取后输出 <wait>]
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
   会议记录(XML / JSON):谁 · 何时 · 原文时间段 · 译文         UniSS Talker(冻结):每个槽位一条流,用该说话人的音色
                                                                     → 离线:按源时间排布(配音)/ 同传:单通道按提交顺序
```

### 时间戳从哪里来

1. **Nemotron-3 是"谁在什么时候说话"的来源。**
   - 它的后验以说话人 token 的形式进模型;
   - 它的 RTTM 片段决定切窗位置。
   - 槽位编号整场一致,就是最终输出里的说话人编号。
2. **大模型输出每段译文的起止时间和说话人。**
   - 时间 token 在 80 ms 网格上,相对窗口起点;
   - 训练时,目标时间来自 AMI 参考,参考说话人按与 Nemotron 槽位的帧重叠做 Hungarian 匹配,所以模型学到的是"跟着 Nemotron 的槽位和时间走"。
   - 这和分支 G-STAR 的做法相同。
3. **会议记录同时附上 Nemotron 原始的 RTTM**,方便核对。评测时从模型输出片段直接算 DER,和分支同口径。

---

## 2. 模型规格(G-STAR-SI)

| 模块 | 规格 | 训练 |
|---|---|---|
| Nemotron-3-Diarization | `nvidia/Nemotron-3-Diarization`(公开,openmdw-1.1,.nemo 199 MB);8 槽位,80 ms。<br>离线配置:spkcache 264 / fifo 40 / chunk 340 / 右上下文 40 / 更新 300。<br>流式配置(1.04 s):264 / 264 / 9 / 4 / 222 | 冻结 |
| SD 投影器 | 照搬分支 `SDProjector`:Conv1d(8→1024, k=5, s=5) → GELU → Linear(1024→2048) → GELU → Linear(2048→2048),约 6.3M | 训练(最后一层零初始化) |
| AuT 编码器 | m2 Thinker 自带:32 层 × 1280,FFN 5120,输出 2048,约 13 token/s | 冻结(消融:LoRA r32) |
| Thinker LLM | m2(`thinker_merged_i200`,Qwen3-Omni-30B-A3B,已合并 ABC) | 注意力 LoRA r32,全部 48 层 |
| 新 token | 251 个 `<\|t0\|>…<\|t250\|>`(80 ms,≤20 s)+ 8 个 `<\|spk1\|>…<\|spk8\|>` | 只训练这 259 行的嵌入和输出头。放在词表空闲行(词表 151,676 个 token,矩阵 152,064 行,空 388 行),不扩词表,合并后 vLLM 仍可加载 |
| 位置编码 | 只有音频和文本时,Qwen3-Omni 的 M-RoPE 等价于一维递增;交错后直接用 arange(分支在 Qwen3-ASR 上同样处理) | — |
| Talker | UniSS T1c@1500;每个槽位一条流;音色 = 该槽位"独自说话"帧(p_k > 0.5 且其他槽位 < 0.5)上的 BayesVoiceMemory | 冻结 |

### 输入模板

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>[交错的 a / s token]<|audio_end|>
[指令:翻译成中文,并按格式标注每段的源时间和说话人][上文:前一窗口最后约 30 个译文字,带说话人 token]<|im_end|>\n
<|im_start|>assistant\n[同传时:本窗口已提交的输出]
```

### 输出格式

**离线(阶段 O)**:一次输出窗口内全部片段,按源开始时间排序,每个片段是一句话:

```
<|t3|>或者说，这要求过高了吗？<|t18|><|spk2|><|t16|>是的，嗯，我不是指它本身，而是说它是整个事情的一部分。<|t81|><|spk1|>…<|im_end|>
```

**同传(阶段 S)**:每次读取只输出新提交的片段(以译文块为单位,时间是这块对应的源语音),然后输出 `<|box_end|>`(wait)。段尾输出剩余部分和 `<|im_end|>`。

**会议级结果**:所有窗口的片段换算到整场时间,导出分支同款格式:

```xml
<text>
0.24-1.44>或者说，这要求过高了吗？
1.28-6.48>是的，嗯，我不是指它本身，而是说它是整个事情的一部分。
</text>
<speaker>
<spk id="2" t="0.24-1.44"/>
<spk id="1" t="1.28-6.48"/>
</speaker>
```

JSON 版本另加每段的提交时刻和译音文件。

---

## 3. 技术路线:先离线(阶段 O),再同传(阶段 S)

| 理由 | 说明 |
|---|---|
| 数据天然是离线的 | AMI 的译文是整句翻译;离线目标准确,同传目标需要构造时间表 |
| 快速试结构 | 离线每窗口 1 行(约 2.6–5 万行),训练约 1 h;同传约 17 万行、约 2 h,推理也贵约 7 倍 |
| 好诊断 | 能分清问题出在"说话人和翻译"还是"流式和延迟" |
| 直接复用 | 离线模型是同传阶段的初始化,也是质量上限;同传的提交策略可以用离线整句译文做自蒸馏(m2 就是这样训的) |
| 分支先例 | G-STAR 先离线后流式:离线 19.20,流式 19.77 |

**两个阶段的工程完全共用**:离线就是"只在窗口末尾读一次"的同传特例。数据管线、模型类、推理脚本、打分都一份代码。

---

## 4. 数据 pipeline

已有、已审查(`data/ami_multispk_si_v1/`):IHM-mix 171 场(100 h),译文和译音分片,审查结果;train / dev / test 分别 25,676 / 3,566 / 2,815 段。

| 步骤 | 内容 | 产物 |
|---|---|---|
| D1 | 唯一的新环境:`venvs/nemo_diar`。用现有 Python 3.12 建 venv,不动已有环境。<br>装 torch 2.6.0 + NeMo(按分支 `setup_env.sh`);下载 Nemotron-3。<br>脚本:`scripts/setup_nemo_diar.sh`(**已启动**,脱离会话运行) | `venvs/nemo_diar`、`pretrained_models/Nemotron-3-Diarization/` |
| D2 | 171 场整场导出:离线配置和流式配置(1.04 s)各一份,每场得到后验 [T, 8] 和 RTTM。<br>脚本:`data_clean/dump_nemotron_posteriors.py`(参考分支 `nemotron/dump_posteriors.py`) | `data/ami_multispk_si_v1/diar/nemotron_{offline,stream}/{meeting}.{post.pt,rttm}` |
| D3 | 参考说话人活动 → 0/1 后验(AMI 词级时间),用于上限实验 | `.../diar/oracle/{meeting}.npy` |
| D4 | 每场:参考说话人 → Nemotron 槽位的 Hungarian 映射(按帧重叠,同分支 `map_speakers_to_slots`);离线、流式各一份 | `.../diar/slot_map_{offline,stream}.json` |
| D5 | 窗口:<br>① 参考句组(已审查,≤20 s);<br>② 按 Nemotron 活动切窗:1 s 无人说话就关窗,≤20 s。<br>窗口里有任何一句没通过审查,就整窗剔除 | `.../e2e_gstar_v1/windows_{split}_{ref,diar}.jsonl` |
| D6 | 离线目标:窗口内每句 → `<\|t起\|>译文<\|t止\|><\|spk 槽位\|>`,按开始时间排序;纯犹豫词不出片段 | `.../e2e_gstar_v1/offline_{split}.jsonl` |
| D7 | 同传目标:每句按标点切块,用 AMI 词级时间定每块的源起止;提交时刻 = 源结束 + 1 s;每(窗口, 读取时刻)一行,同 v0 的格式 | `.../e2e_gstar_v1/stream_{split}.jsonl` |
| D8 | 训练行:<br>阶段 O = 离线行(两种窗口混合,和分支 "mix" 一样)+ 15% 单人整句行;<br>阶段 S = 同传读取行 + 20% 离线行 + 15% 单人同传行 | `.../e2e_gstar_v1/rows_{O,S}_{train,val}.jsonl` |

---

## 5. 训练

**共同配置**
- Megatron,8 卡;GBS 64,MBS 8(实测是加速关键);数据全量 shuffle,固定种子。
- 用 `setsid nohup` 脱离会话运行。
- 可训练部分:SD 投影器、259 个新 token 的行、Thinker 注意力 LoRA r32。
- 学习率:LoRA 1e-4;投影器和新 token 5e-4;warm-up 100,cosine。
- 损失:交叉熵;时间 token ×1.5、说话人 token ×2(分支的分层权重);wait 和结束标记 ×1。
- 扰动:后验以 0.2 的概率加噪或平移 ±1 帧;上文以 0.1 的概率丢词。
- 产物:LoRA 合进 m2;新 token 行写入空闲行;分词器加 token;投影器另存。

| 阶段 | 初始化 | 数据 | 后验 | 步数和时间 |
|---|---|---|---|---|
| **O(离线)** | m2 Thinker | 离线行 | Nemotron **离线**配置 | 约 2 遍,约 1,600 步,约 1–1.5 h |
| **S(同传)** | 阶段 O 的模型 | 同传读取行为主 + 20% 离线行 | Nemotron **流式**配置,按读取时刻截断(最后约 1 s 用暂定值) | 约 1 遍,约 2,700 步,约 2 h |

新文件:
- `training/thinker_gstar_model.py`:交错输入、新 token 行、投影器、分层损失;
- `training/pretrain_thinker_gstar.py`;
- `scripts/run_thinker_gstar.sh`;
- `training/merge_thinker_gstar.py`。

---

## 6. 推理

### 阶段 O(离线)

```
整场音频 → Nemotron(离线配置)→ 后验 + RTTM → 按 RTTM 切窗(≤20 s)
每个窗口:a = AuT(窗口音频);s = 投影器(窗口后验);一次生成全部片段
→ 片段换算到整场时间 → XML / JSON
Talker:每个槽位用整场"独自说话"的帧建音色;每段译音按源开始时间排布(超长就加速,同 multi_speaker_pipeline 的 δ 排布)→ 配音式译音
```

### 阶段 S(同传)

```
音频流 → Nemotron(流式 1.04 s)持续出后验 → 因果切窗(1 s 无活动关窗,≤20 s)
窗口内每 1 s 读一次:交错(音频, 截至此刻的后验)+ 已提交前缀 → 输出新片段 + <wait>(窗口末尾输出结束标记)
新片段立刻进对应槽位的 Talker 流(音色记忆按槽位在线累积,30 s 后冻结)
→ 单通道:片段在 max(提交时刻, 上一段结束) 播放,换人时停 0.2 s;同时写 XML / JSON
```

**推理引擎**:交错输入需要自定义嵌入,baseline 用 HF 批量生成(`evaluation/gstar_si_stream.py`,离线和同传共用,同一轮的所有窗口拼一个 batch)。之后可以写 vLLM 插件提速。

---

## 7. 实验

### 阶段 O(离线)

| 编号 | 系统 | 目的 |
|---|---|---|
| C0 | **非端到端的级联参照**:按 Nemotron RTTM 切出每人片段 → m2 逐段离线翻译 → 标上 Nemotron 槽位 | 端到端是否优于"分离后逐段翻译",尤其在重叠区 |
| **O-G1** | **G-STAR-SI 离线 baseline**(两支 + 交错 + G-STAR 格式) | 主系统 |
| O-nosd | O-G1,说话人 token 置零,格式不变 | 分离支的贡献 |
| O-oracle | O-G1 用参考 0/1 后验 | 分离误差上限 |
| O-enc | O-G1 + AuT LoRA r32 | 编码器适配(重叠区) |
| O-add | 说话人 token 相加到音频 token,不交错 | 融合方式 |
| O-local | 段内按首次出现编号的局部槽位(分支 stage 2) | 局部 vs 全局槽位 |

### 阶段 S(同传)

| 编号 | 系统 | 目的 |
|---|---|---|
| B0 | m2 原样(不分说话人) | 同传、不做多说话人的参照 |
| **S-G1** | **G-STAR-SI 同传 baseline**(从 O-G1 初始化) | 主系统 |
| S-noinit | 从 m2 直接训同传,不经过阶段 O | 先离线是否有用 |
| S-oracle | 参考后验 | 上限 |
| S-Δ | 读取间隔 0.5 s 和 1 s | 质量和延迟的权衡 |
| S-single | RealSI 英→中、ACL60 英→中 | 单人能力是否退化 |

另外,当前已经在训的 E0(不接分离支)训完即保留,作为"没有分离支"的参考,不再投入新的资源。

### 判据

| 对比 | 要求 |
|---|---|
| O-G1 对 C0 | 整场 cpBLEU 不低于 C0,重叠段显著更好;DER 不劣于 Nemotron 自身太多(Nemotron 在干净 16 场上约 9–12%) |
| O-G1 对 O-nosd | cpBLEU、DER、说话人数正确率显著更好(证明分离支有效) |
| S-G1 对 O-G1 | cpBLEU 损失 ≤3 BLEU;StreamLAAL 在 2.5 s 以内(含 1.04 s 前瞻) |
| S-G1 对 B0 | cpBLEU 显著更高;不分说话人的 BLEU 不降超过 1 |
| S-single | 与 m2 的差在 ±0.5 BLEU 以内 |

---

## 8. 评测指标

| 类别 | 指标 |
|---|---|
| 带说话人的译文 | cpBLEU / cpchrF:整场口径,槽位按整场 Hungarian 匹配参考说话人;也报窗口内口径 |
| 谁在什么时候说话 | 从输出片段算 DER(collar 0,和分支同口径);同时报 Nemotron 自身的 DER;说话人数正确率;片段归属准确率 |
| 不分说话人 | 整场 BLEU(mwerSegmenter 重切分) |
| 语音 | cp-ASR-BLEU(paraformer 在纯语音上);按人音色相似度(WavLM,与该人 IHM 原声比);音色串的比例 |
| 延迟(阶段 S) | LAAL 和 StreamLAAL(含分离前瞻);首包和尾包偏移;fluency 提交包 |
| 分组 | 单人窗口 / 多人窗口 / 重叠窗口 / 严格审查子集 |
| 显著性 | 按会话的配对 bootstrap 1000 次,加窗口分层 bootstrap |

---

## 9. 工程清单(全部新文件,已有文件不改)

| 文件 | 状态 |
|---|---|
| `scripts/setup_nemo_diar.sh` | ✅ 已写,已启动 |
| `data_clean/dump_nemotron_posteriors.py` | 待写(D2) |
| `data_clean/build_ami_gstar_si_rows.py` | 待写(D3–D8;复用 `build_ami_ms_rows.py` 的审查和时间表) |
| `training/thinker_gstar_model.py`、`pretrain_thinker_gstar.py`、`merge_thinker_gstar.py`、`scripts/run_thinker_gstar.sh` | 待写 |
| `evaluation/gstar_si_stream.py` | 待写(HF 推理,离线和同传共用) |
| `evaluation/e2e_ms_post.py` | 扩展:整场 cp 指标、DER、XML / JSON 导出;Talker 拆流和合并已有 |
| `evaluation/e2e_ms_voice.py` | ✅ 已有(音色指标) |

---

## 10. 时间线(布拉格时间)

| 时间 | 内容 |
|---|---|
| 10-26 晚 | D1 装环境(进行中);写 D2 导出脚本;E0 训完(约 20:50),其测试队列约 21:45 结束,之后 GPU 空出 |
| 10-26 夜到 10-27 上午 | D2 导出 171 场后验(离线和流式);D3–D8 构造数据;写模型类和 HF 推理;2 卡冒烟 |
| 10-27 下午 | 阶段 O 训练(约 1.5 h)→ O-G1 和 C0 测试;接着 O-nosd、O-oracle |
| 10-28 | O-enc、O-add、O-local;选出离线最好的结构 → 阶段 S 训练(约 2 h)→ S-G1 测试 |
| 10-29 | S-noinit、S-oracle、S-Δ、单人回归;结果表和报告 |

## 11. 需要你确认

1. 先离线(阶段 O)再同传(阶段 S)的路线。
2. 输出用分支同款特殊 token(不扩词表)。
3. 加入 C0(Nemotron 切段 + m2 逐段翻译)作为非端到端的参照。它只用于比较,不作为交付的系统。
