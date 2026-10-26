# 多说话人同声传译方案 v4:G-STAR 式双支路端到端 baseline(AMI 英→中)

- 日期:2026-10-26(v1–v3 保留不动)
- 需求(用户原话的整理):
  1. 一支是说话人分离(diarization);
  2. 另一支是我们目前最好的同传大模型;
  3. 两支一起经过大模型,输出和 GitHub `tagspeech/gstar-nemotron` 分支模型训练、推理一样格式的结果(时间 + 说话人 + 文字),只是文字换成译文;
  4. 给出如何搭建 pipeline、如何训练和推理;
  5. 必须是端到端结构;
  6. 先给出一个能做多说话人同传的 baseline。
- 沿用的决定:只用 IHM-mix;只用审查通过的数据;不用 SCOPE 的贝叶斯推理;需要装包时统一装进一个环境。

---

## 0. 一页总结

```
                        ┌──────────── 说话人分离支 ────────────┐
 会议单通道音频流 ──┬──►│ Nemotron-3 流式分离(冻结,8 槽位,     │──► 说话人 token s_j(每 5 个音频 token 一个)
 (IHM-mix 16 kHz)  │    │ 1.04 s 前瞻)→ 后验 → SD 投影器      │                     │
                   │    └──────────────────────────────────────┘                     │ K:1 交错
                   │    ┌──────────── 同传大模型支 ────────────┐                     ▼
                   └───►│ m2 的 Qwen3-Omni Thinker:             │──► 音频 token a_i ──► [a0..a4 s0 a5..a9 s1 …]
                        │ AuT 编码器 → 投影 2048               │                     │
                        └──────────────────────────────────────┘                     ▼
                         同一个 Thinker LLM(m2 + LoRA):边听边输出 G-STAR 格式
                         <|t起|>译文块<|t止|><|spk k|> … <wait>
                                                     │
                         UniSS Talker(冻结),按槽位 k 用该说话人的音色 → 单通道中文译音
```

- **结构**:与分支里 Qwen 线的 G-STAR 模型(报告中的 Q3:流式整段 cpWER 19.77,而纯端到端的 TagSpeech 是 44.16)**一一对应**,只把"ASR 大模型"换成"我们的同传大模型",把"转写"换成"译文"。
- **输出格式**:完全沿用分支的 SOT 格式。`<|t起|>`、`<|t止|>` 是 80 ms 网格上的 251 个时间 token,`<|spk k|>` 是 8 个说话人 token。新 token 放进 Qwen3-Omni 词表的 388 个空闲行,不扩词表,vLLM 仍能加载。
- **同传化**:每 Δ = 1 s 读一次,模型只输出本次可以提交的片段,然后输出 wait 标记,沿用 m2 的 selfwait 机制。段尾输出剩余内容和结束标记。
- **端到端**:一个模型图。分离支是冻结的子网络(和分支里"speaker branch"的地位相同),它的输出作为 token 进入大模型;谁说的、说什么、什么时候说都由大模型输出。没有转写到翻译的文字接口,也没有贝叶斯后处理。
- **最终结果**:每场会议输出:
  - 带时间和说话人的双语记录,可以导出成和分支相同的 XML / JSON;
  - 单通道中文译音,每个说话人保持自己的音色。

---

## 1. 参照:分支里 G-STAR 模型(Qwen 线)的结构

来源:
- `examples/tagspeech/gstar_nemotron/qwen3/qwen_gstar.py`
- `gstar/targets.py`、`gstar/model.py`
- `qwen3/train.py`、`qwen3/decode_global.py`
- `REPORT.md`

| 部件 | 分支里的做法 |
|---|---|
| 说话人支 | Nemotron-3-Diarization 流式分离(冻结):后验 [T, 8],80 ms/帧,1.04 s 前瞻,AOSC 缓存使槽位整场一致 |
| 识别支 | Qwen3-ASR-1.7B 的 AuT 编码器(冻结)→ A ∈ [N, 2048],约 13 token/s |
| SD 投影器 | 后验重采样到音频 token 网格 → Conv1d(kernel 和 stride 都是 5)+ 两层 MLP → S ∈ [⌈N/5⌉, 2048] |
| 融合 | 在 `<\|audio_start\|> … <\|audio_end\|>` 内按 K:1 交错:a0..a4, s0, a5..a9, s1, … |
| 大模型 | Qwen3-1.7B + LoRA(r64);位置编码直接用 arange |
| 输出格式 | 每段 `<\|t起\|>文字<\|t止\|><\|spk k\|>`,按开始时间排序;251 个时间 token(80 ms,≤20 s)+ 8 个说话人 token,放在单独的嵌入和输出头里,原词表冻结 |
| 损失 | 分层交叉熵:时间 token ×1.5,说话人 token ×2 |
| 说话人标签 | 参考说话人用 Hungarian(按帧重叠)匹配到分离槽位,目标里写匹配到的槽位号。<br>stage 2 为局部模式:槽位按段内首次出现重排;<br>stage 3 为全局模式:直接用整场槽位 |
| 训练阶段 | stage 2:≤20 s 的段,局部槽位;<br>stage 3:全局槽位 + VAD 切窗(与推理一致,流式提升 1.5) |
| 推理 | 按分离活动做因果切窗(1 s 停顿关窗,≤20 s),每窗整段解码,解析出段 → 会议级 XML / JSON |
| 结果 | 流式整段 test cpWER **19.77**(离线 19.20);同一流程零训练只有 26.75;纯端到端的 TagSpeech 是 44.16 |

我们要改的只有三处:
1. 识别支换成同传大模型;
2. 段内文字换成译文;
3. 整窗解码换成边听边提交。

---

## 2. Baseline 模型:G-STAR-SI

### 2.1 模块

| 模块 | 来源 | 形状 / 参数 | 训练 |
|---|---|---|---|
| 分离支 | `nvidia/Nemotron-3-Diarization`(HuggingFace 公开,已查) | 后验 P ∈ [0,1]^{T×8},80 ms | 冻结 |
| SD 投影器 | 分支 `gstar/model.py` 的 `SDProjector` | 后验插值到 AuT token 网格 → Conv1d(8→1024, k=5, s=5) → GELU → Linear(1024→2048) → GELU → Linear(2048→2048);约 6.3M | 训练,最后一层零初始化 |
| 同传支编码器 | m2 Thinker 的 AuT(32 层 × 1280,FFN 5120) | 输出 a_i ∈ R^2048,约 13/s | 冻结(消融里加 LoRA r32) |
| 大模型 | m2 Thinker LLM(Qwen3-Omni-30B-A3B,48 层 MoE,已合并 ABC) | — | 注意力 LoRA r32(全部 48 层) |
| 新 token | 251 个 `<\|t0\|>…<\|t250\|>` + 8 个 `<\|spk1\|>…<\|spk8\|>` | 放进词表空闲行 151676–151934(共 388 行空闲) | 只训练这 259 行的嵌入和输出头 |
| Talker | UniSS T1c@1500 | — | 冻结 |

### 2.2 输入格式(每次读取)

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n
<|audio_start|> a0 a1 a2 a3 a4 s0 a5 … a9 s1 … <|audio_end|>
[同传多说话人指令] [上文:前一段最后约 30 个已提交译文字,带说话人 token]
<|im_end|>\n<|im_start|>assistant\n
[本段已提交的输出(作为回复前缀)]
```

- **音频**:当前段从开始到读取时刻 t 的音频。
- **后验**:截至 t。分离器的前瞻是 1.04 s,最后约 1 s 用暂定值,训练时按同样方式截断。
- **槽位**:全局槽位(整场一致),与 G-STAR stage 3 相同。

### 2.3 输出格式(训练目标和推理输出相同)

```
<|t3|>或者说，这要求过高了吗？<|t18|><|spk2|><|t16|>是的，<|t24|><|spk1|><|t28|>我想是的<|t36|><|spk3|> … <wait>
```

- 每个片段是一个可提交的译文块:时间是这块译文对应的源语音起止时间(80 ms 网格,相对段首),说话人 token 是对应的分离槽位。
- 片段按提交顺序排列:源语音说完 + 1 s(和 v0 的时间表相同)。
- 每次读取只输出本次新提交的片段,然后输出 wait 标记 `<|box_end|>`。段尾那次读取输出剩余片段和 `<|im_end|>`。
- 纯犹豫词不出片段。

**最终结果的格式**:把所有片段拼起来,转成分支同款的会议级 XML / JSON:

```xml
<text>
0.24-1.44>或者说，这要求过高了吗？
1.28-1.92>是的，
</text>
<speaker>
<spk id="2" t="0.24-1.44"/>
<spk id="1" t="1.28-1.92"/>
</speaker>
```

### 2.4 语音

- 每个槽位一条 Talker 流,只包含该槽位的片段。
- 音色参考 = 该槽位"独自说话"的帧(p_k > 0.5 且其他槽位都 < 0.5),截至该槽位第一次提交;用已有的 BayesVoiceMemory 按槽位累积,30 s 后冻结。
- 各流的译音按提交顺序合并成单通道:片段在 max(提交时刻, 上一段结束) 开始播放,换人时停 0.2 s。

### 2.5 为什么是端到端

- 分离支是冻结的子网络,它的输出以 token 形式进入大模型,地位和分支 G-STAR 的 speaker branch 相同。
- "谁说的"(说话人 token)、"什么时候、说什么"(时间 token + 译文)、"现在说不说"(wait 或结束)全部由大模型一次前向输出。
- 没有转写到翻译的文字接口。没有贝叶斯重排、HMM 归属、序贯注册。

---

## 3. Pipeline 搭建

### 3.1 数据准备(一次性,CPU 和 GPU)

| 步骤 | 内容 | 产物 |
|---|---|---|
| D1 | 建 NeMo 环境(唯一新装的环境)。下载 Nemotron-3-Diarization。复用分支 `nemotron/setup_env.sh` 的安装步骤 | `conda_envs/nemo_diar`、`pretrained_models/Nemotron-3-Diarization` |
| D2 | 对 171 场 IHM-mix **整场**按流式配置(1.04 s)导出后验。参考分支 `nemotron/dump_posteriors.py` | `data/ami_multispk_si_v1/diar/nemotron_stream/{meeting}.npy`,[T, 8] |
| D3 | 参考说话人活动(AMI 词级时间)→ 0/1 后验,用于上限实验 | `.../diar/oracle/{meeting}.npy` |
| D4 | 每场会议:参考说话人 → 分离槽位的 Hungarian 映射,用分支 `targets.map_speakers_to_slots` 的整场版 | `.../diar/slot_map.json` |
| D5 | 段落:沿用已审查的段(train 25,676、dev 3,566、test 2,815)。另生成**分离器因果切窗**(1 s 无活动关窗,≤20 s),用于流式训练和测试 | `.../e2e_gstar_v1/pieces_*.jsonl` |
| D6 | 目标序列:每段的译文块 → `<\|t起\|>译文<\|t止\|><\|spk槽位\|>`,按提交时间排序 | 同上 |
| D7 | 训练行:每(段, 读取时刻)一行 = 音频截至 t + 后验截至 t + 已提交前缀 → 本次新片段 + wait / 结束;另 20% 整段行(t = 段长,一次输出全部);再混入 15% 单人同传行 | `.../e2e_gstar_v1/rows_{train,val}.jsonl` |

### 3.2 训练 pipeline(Megatron,8 卡,完全 shuffle)

| 阶段 | 数据 | 对应分支 | 目的 |
|---|---|---|---|
| T1 | 参考段落 + **局部**槽位(段内按首次出现重排),整段行和读取行混合 | stage 2 | 先学会格式和融合 |
| T2 | 从 T1 继续:**全局**槽位 + 分离器切窗(和推理一致),读取行为主 | stage 3 | 整场一致的说话人和流式一致性 |

baseline 先只训 T2,从 m2 直接起步。T1 → T2 的两阶段作为消融,看分阶段是否必要。

| 配置 | 值 |
|---|---|
| 入口 | `training/pretrain_thinker_gstar.py`,模型类 `training/thinker_gstar_model.py` |
| 可训练 | SD 投影器、259 个新 token 的嵌入和输出头、Thinker 注意力 LoRA r32;消融里加 AuT LoRA r32 |
| 损失 | 交叉熵:时间 token ×1.5,说话人 token ×2,wait 和结束标记 ×1(分支的分层权重 + m2 的结束标记) |
| 扰动 | 后验以 0.2 的概率扰动(高斯噪声、时间平移 ±1 帧);上文以 0.1 的概率丢词 |
| 批量 | GBS 64,MBS 8(实测提速关键) |
| 学习率 | LoRA 1e-4;SD 投影器和新 token 5e-4;warm-up 100,cosine |
| 步数 | 约 1 遍数据(约 2,700 步,约 2 h) |
| 运行方式 | `setsid nohup` 脱离会话,不再被会话结束带走 |
| 合并 | LoRA 合进 m2;新 token 的行写进嵌入和输出头的空闲行;分词器加入 259 个 token;SD 投影器另存 |

### 3.3 推理 pipeline(严格流式)

```
for 每场会议(流式):
    分离支持续产出后验(1.04 s 前瞻)
    因果切窗:有人说话就开段,1 s 无活动或满 20 s 就关段
    for 段内每个读取时刻 t(每 1 s):
        a = AuT(音频[段首:t]);s = SD 投影器(后验[段首:t]);x = 交错(a, s)
        y = Thinker 贪心续写(prompt + x + 已提交前缀),遇到 wait 停;段尾遇到结束标记停
        解析 y → 片段 (t起, t止, spk, 译文),立即交给对应槽位的 Talker 流
    关段:上文 = 本段最后约 30 个译文字,带说话人 token
每个槽位的 Talker 流 → 单通道合并 → 输出 XML / JSON 记录 + 译音
```

**推理引擎**:交错输入需要自定义嵌入,vLLM 的 Qwen3-Omni 不支持直接传入。baseline 先用 HF 批量生成(新写 `evaluation/gstar_si_stream.py`):同一读取轮次内所有段拼成一个 batch。test 约 2.8k 段,8 卡并行预计 0.5–1 h。之后再写 vLLM 插件提速,不影响结果。

---

## 4. 实验

### 4.1 系统和消融

| 编号 | 系统 | 目的 |
|---|---|---|
| B0 | m2 原样(只有同传支,不分说话人) | 不做多说话人的参照(排队中) |
| E0 | v0:只有同传支,模型自己编号说话人(纯端到端,无分离支) | "去掉分离支"的对照(训练中) |
| **G1** | **G-STAR-SI baseline**:两支 + K:1 交错 + G-STAR 输出格式 + 全局槽位 + 流式读取 | 本方案的 baseline |
| G1-nosd | G1,但说话人 token 全置零(格式不变) | 分离支本身的贡献 |
| G1-oracle | G1 用参考 0/1 后验 | 分离误差的上限 |
| G1-2stage | T1(局部)→ T2(全局)两阶段 | 分支的分阶段训练是否必要 |
| G1-enc | G1 + AuT LoRA r32 | 编码器适配(SCOPE 里重叠区 −0.6 到 −1.2) |
| G1-add | 说话人 token 改为相加到音频 token,不交错 | 融合方式 |
| G1-single | RealSI 英→中、ACL60 英→中 | 单人能力是否退化 |

### 4.2 之后的改进(不在 baseline 范围内)

v3 方案里 SCOPE 的部件可以按顺序加到 G1 上:
1. STNO + BSSC;
2. 逐说话人条件链;
3. DSS;
4. 跨窗上下文。

这些部件的加法顺序和要回答的问题见 v3 第 6 节(那里以 E 编号的系统为起点,这里的起点是 G1)。

### 4.3 判据

**G1 对 E0**:
- 整场 cpBLEU 显著提高;
- 说话人数正确率和归属准确率明显提高;
- 不分说话人的 BLEU 不降超过 0.5。

**G1 对 B0**:
- cpBLEU 显著提高;
- 不分说话人的 BLEU 不降超过 1;
- StreamLAAL 增加 ≤ 0.5 s(其中包含分离前瞻)。

**G1-single**:与 m2 的差在 ±0.5 BLEU 以内。

---

## 5. 测试与指标

| 类别 | 指标 |
|---|---|
| 带说话人的译文 | **cpBLEU / cpchrF**:整场口径,槽位按整场 Hungarian 匹配参考说话人;也报段内口径 |
| 说话人(新增,和分支同口径) | 输出片段带源时间和说话人,可以直接算 **DER**(collar 0,参考为 AMI 时间);说话人数正确率;片段级归属准确率 |
| 不分说话人 | 整场 BLEU + StreamLAAL(mwerSegmenter 重切分) |
| 语音 | cp-ASR-BLEU(paraformer 在纯语音上);按人音色相似度;音色串的比例 |
| 延迟 | LAAL 和 StreamLAAL(含分离前瞻);首包和尾包偏移 |
| 分组 | 单人段 / 多人段 / 重叠段 / 严格审查子集 |
| 显著性 | 按会话的配对 bootstrap 1000 次,加段落分层 bootstrap |
| 回归 | RealSI、ACL60 |

---

## 6. 工程清单(全部新文件,不改已有文件)

| 文件 | 作用 |
|---|---|
| `data_clean/dump_nemotron_posteriors.py` | D2,在 NeMo 环境运行 |
| `data_clean/build_ami_gstar_si_rows.py` | D3–D7(复用 `build_ami_ms_rows.py` 的审查和时间表) |
| `training/thinker_gstar_model.py` | 交错输入、新 token 行、SD 投影器、分层损失、arange 位置 |
| `training/pretrain_thinker_gstar.py`、`scripts/run_thinker_gstar.sh` | 训练入口和启动(setsid nohup) |
| `training/merge_thinker_gstar.py` | LoRA + 新 token 行合并,分词器加 token,SD 投影器另存 |
| `evaluation/gstar_si_stream.py` | HF 批量流式推理,输出片段和 XML / JSON |
| `evaluation/e2e_ms_post.py`(扩展) | 整场 cp 指标、DER、XML 导出;Talker 按槽位拆分和合并沿用现有代码 |

---

## 7. 时间线(布拉格时间)

| 时间 | 内容 |
|---|---|
| 10-26 晚 | E0 训完,B0 / E0 自动测试(约 21:45 出结果)。并行:D1 建环境和下载模型(不占 GPU) |
| 10-27 上午 | D2 导出 171 场后验(GPU 约 1 h),D3–D7 构造数据;写模型类、训练入口、推理脚本;2 卡冒烟 |
| 10-27 下午 | G1 训练(8 卡约 2 h)→ 测试(约 1 h) |
| 10-28 | G1-nosd、G1-oracle、G1-enc 等消融,单人回归;结果表和报告 |

---

## 8. 需要你确认

1. **结构**:分离支 + 同传大模型支,K:1 交错进同一个 Thinker,输出 G-STAR 的 `<|t|>译文<|t|><|spk|>` 格式,这就是你要的 baseline 吗?
2. **输出格式**:用分支同款特殊 token(放在词表空闲行,不扩词表)。另一种是纯文本标签,不用改词表,但每个片段头要多占约 11 个 token。默认用特殊 token。
3. **训练阶段**:先只训 T2(全局槽位),两阶段作为消融,可以吗?
