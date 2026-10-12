# 换 Qwen3-Omni-30B-A3B:完整阶段计划

**日期** 2026-10-11
**动机** 翻译这一轴只由基座决定,而我们的训练从未推高过它
**结论先行** 这是**同拓扑换件**,不是架构改造。UniSS 全部非大模型模块原样保留。

---

## 〇、为什么做这件事

| 系统 | Text-BLEU en→zh / zh→en | ASR-BLEU |
|---|---|---|
| 0.5B Phase3 | 24.15 / 15.33 | 23.58 / 12.05 |
| **Qwen2.5-Omni-3B 零样本** | **30.56 / 26.65** | — |
| Stage 1(冻结 3B Thinker) | 30.45 / 26.82 | 28.70 / 24.82 |
| Stage 2(LoRA) | 29.91 / 23.77 | 23.09 / 18.73 |

换基座 0.5B→3B 带来 **+6.41 / +11.32**。此后**我们所有训练都没能再推高一分**:
Stage 1 冻结 Thinker(≈ 零样本),Stage 2 的 LoRA 把 zh→en 推低了 3 分。

**基座是唯一被证明能移动 Text-BLEU 的杠杆。**

而 ASR-BLEU 近似是 Text-BLEU 的 1:1 函数 —— cascade 路径下 Talker 只损耗
**2.00–2.82** BLEU(实测:Stage 1 的 2.00/1.75,Stage 2 Talker 的 3.75/2.82)。
译文每涨一分,语音跟涨约一分。

旁证:IWSLT 2026 的 CUHKSZ 系统用 Qwen3-Omni-30B-A3B 做同传,但**只输出文本**
(LoRA 只加在 LLM 上,不涉及 Talker/codec)。它支持的是换 Thinker,不是换 Talker。

## 一、替换什么、保留什么

### 替换(一个零件)

```
Thinker:  Qwen2.5-Omni-3B (36 层, hidden 2048)
       →  Qwen3-Omni-30B-A3B (48 层, hidden 2048, MoE 30B/3B 激活)
```

### 丢弃

Qwen3-Omni 自带的 **Talker(16 quantizers × 16 code groups 多码本)和 code2wav**
整个删掉 —— 和现在删 Qwen2.5-Omni 的 `token2wav` 是同一个动作。
它的多码本设计与我们无关,因为那条路径被整体丢弃。

### 原样保留(UniSS 侧,全部)

| 模块 | 说明 |
|---|---|
| BiCodec 声码器 / 分词器 | `decode_tokens_to_audio` 里 `tokens[:32]` 是说话人,`tokens[32:]` 是内容 |
| **32 个 global token(4,096 码本)** | 从源音频提取,**直通声码器,不经任何语言模型** —— A.PCP 2.925 全表最高的来源 |
| 8,192 单流语义码 @ 50 Hz | 不变 |
| wav2vec2-large-xlsr-53 | BiCodec 内部 |

### 原样保留(我们的桥接件,全部)

**因为两边 Thinker 的 hidden_size 都是 2048**,维度一个都不用改:

| 模块 | 维度 |
|---|---|
| `global_embed` | `Embedding(4096, 2048)` |
| `thinker_to_talker_proj` | `[896, 2048]`,是 **Talker 的属性**,随 Talker 留下 |
| `stage0_assembly.build_text_stream` | 在 2048 维上工作 |
| `talker_surgery` / `talker_transfer` | 手术对象是 Qwen2.5-Omni 的 Talker,它留下了 |

### 唯一要新建的

一个约 50 行的组装模块,同进程加载两个模型:

```python
thinker = Qwen3Omni...from_pretrained("Qwen3-Omni-30B-A3B").thinker   # ~60 GB
talker  = Qwen2_5Omni...from_pretrained("Qwen2.5-Omni-3B").talker      # ~0.8 GB
# 删掉 Qwen3 的 talker/code2wav，删掉 Qwen2.5 的 thinker
```

transformers ≥4.57 里两个类并存,没有冲突。

## 二、与上一次替换的工作量对比

| | UniSS 0.5B → Omni 3B(已完成) | Omni 3B → Omni 30B(本计划) |
|---|---|---|
| 拓扑 | **1 个模型 → 2 个模型** | 不变 |
| 新建模块 | **5 个** | **1 个**(~50 行) |
| 码本对齐 | 必须做,依赖 8,192 == 槽位 0–8,191 的巧合 | **不涉及** |
| 桥接维度 | 从零设计 | **2048 == 2048,零改动** |
| UniSS 音色通路 | 需重新接上 | **原样保留** |
| 重训范围 | Stage 0/1/2/3 全部 | **Stage 1/2** |

## 三、阶段计划

### Phase 0 — 环境与落地(不占 GPU,半天)

```
新建 /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni-eval
  transformers 从源码装（≥4.57，官方推荐 5.2.0）
  torchvision 一并装（新环境无历史包袱，装了反而省掉我们的去视频 shim）
uniss-train / uniss-eval 一个字节不动
下载 Qwen3-Omni-30B-A3B ≈ 70 GB
```

**门**:模型能加载,单条音频前向产出通顺译文。

**要实测的两件事**(不能猜):
1. Qwen3-Omni 的音频编码器是 AuT 不是 Whisper 式 —— 有没有我们修过的
   「300 秒 mel 填充」那个坑(当时它让 Stage 2 慢了 17 倍)
2. 25 秒音频的真实显存(官方只给了 15 秒视频的 78.85 GB)

### Phase A — 零样本 S2TT 决策门(**无训练**,2 小时)

与当初给 3B 做的 Step 1 门同一套协议、同一个脚本
([omni_zeroshot_s2tt.py](experiments/uniss_omni_s2st_v1/evaluation/omni_zeroshot_s2tt.py),
`--model` 本来就是参数),只改模型类和处理器两处。

全量 CVSS-T test,两个方向各 4,897 对。

**门:Text-BLEU 对比 30.56 / 26.65**
- zh→en **+3 以上** → 继续
- +1.5 以下 → **停**,基座不是杠杆,省下后面全部成本

zh→en 是弱方向(26.65 对 en→zh 的 30.56),头部空间最大,所以门设在它上面。

### Phase B — cascade,测 ASR-BLEU(**无训练**,2 小时)

Phase A 的译文 jsonl → 现有 Talker → BiCodec → 完整打分 harness。

一个实际约束决定它必须分两进程:Qwen3-Omni 要 transformers ≥4.57,
我们的 Talker 跑在 4.53 上。所以 A 在新环境产文本,B 在老环境产语音。
这正是 cascade 的天然形态。

需要一个小脚本(1–2 小时):读译文 jsonl → `generate_codes` → 声码器 → 打分。

**门:ASR-BLEU > 28.70 / 24.82(Stage 1 基线)。**

**到这里你的问题就完整回答了,而且一行训练代码都没跑。**
只有 A+B 都过门,才值得做下面的。

### Phase C — Stage 0 组装与空操作验证(1 天)

新建 `modeling/qwen3_assembly.py`,复用现有 `stage0_forward_check.py` 的验证:

- 维度对齐:thinker 2048 → proj [896,2048] → talker 896
- **因果性探针**:改动第 t 步之后的输入,第 t 步输出必须**精确为零**变化
- 单步前向跑通

**门**:因果性探针精确为零;形状全对。

### Phase D — Stage 1 重跑:Talker 学读新 Thinker(待实测)

**这是必须的** —— 维度相同不等于表示相同。Qwen3 和 Qwen2.5 的 2048 维隐状态
是两个模型的两个空间,数值含义毫无关系。Talker 读惯了前者,换后者就是读天书。

配方同 Stage 1:TTS 任务,Thinker 冻结,Talker 全量 SFT。

**先导实验(必做,2×2,000 步)**:两个起点各跑一次,按 dev 下降快慢选:
- (i) 从 UniSS phase-3 权重起(与 Stage 1 相同)
- (ii) **从我们现有的 Stage 1 iter 40000 起** —— Talker 已会产 BiCodec 码,
  只需重映射输入空间,可能快得多

**成本未知,必须实测**:60 GB 冻结 Thinker 会逼小 micro-batch。
Stage 1 当初是 40,000 步 × GBS 384。不猜。

**门**:TTS 任务(给定文本)的 ASR-BLEU ≥ Stage 1 水平。

### Phase E — Stage 2 联合训练(待实测)

带上[退步分析](reports/uniss_omni_s2st_v1/stage2/HOW_TO_FIX.zh-CN.md)的全部修正:

| 项 | 值 | 理由 |
|---|---|---|
| s2st 份额 | 1.5 → **3.0** | 联合条件化欠训是 Stage 2 退步的主因 |
| 步数 | **按 Phase D 实测的步时定** | 目标联合样本 ≥ 240 万 |
| `MIN_LR` | 2e-5 → **2e-6** | Stage 1 退火到 3.5e-6,Stage 2 停在 5.7 倍高处 |
| param group | **LoRA 与 Talker 分开** | 低秩适配器要高 LR,已收敛的 Talker 要低 |
| **周期性监控** | **声学探针,不是 CE** | 见下 |

**监控必须换成声学的。** Stage 1 的 dev CE 全程**上升**(7.159→7.301)而音频变好,
Stage 2 的 dev CE **下降**(4.432→4.108)而音频变差 —— 两个阶段方向都是反的。
改为:固定随机 200 条,每 2,000 步生成 + ASR-BLEU,占总时长约 3%。

**门:ASR-BLEU 不得劣于 Phase B 的 cascade 数字。**

### Phase F — Stage 3 流式轨迹微调

计划不变(合并 adapter,挂新 LoRA r8,块轨迹微调)。
**门**:A.PCP ≥ 2.7889 / 2.9073,UTMOS en→zh ≥ 3.8855。

## 四、决策门汇总

| 阶段 | 门 | 不过怎么办 |
|---|---|---|
| 0 | 模型加载 + 产出通顺译文 | 停,查环境 |
| **A** | **Text-BLEU zh→en 比 26.65 高 3 以上** | **停,基座不是杠杆** |
| **B** | **ASR-BLEU > 28.70 / 24.82** | **停,收益不过 Talker 这一关** |
| C | 因果性探针精确为零 | 组装有误 |
| D | TTS ASR-BLEU ≥ Stage 1 | Talker 学不会新表示 |
| E | **不劣于 Phase B** | 回退,不得用翻译收益解释掉 |
| F | A.PCP ≥ 2.7889 / 2.9073 | 回退 |

**每一阶段在全部 Table 1 指标上不得劣于上一阶段**,比较必须用全量测试集或
固定种子的**随机抽样** —— 不得用排序后的前 N 条(这条教训见退步分析第七节)。

## 五、资源与冲突

| 项 | 需求 | 现状 |
|---|---|---|
| 模型下载 | ~70 GB | 磁盘 1.46 TiB ✓ |
| 推理显存 | 78.85 GB(15 秒,bf16+FA2);25 秒待测 | H200 140 GB,**单卡单进程** |
| Phase A 全量推理 | 9,794 条 / 8 卡 | 约 1–1.5 小时 |
| 训练显存 | 60 GB 冻结权重 + 激活 | micro-batch 要大幅下调,待测 |

**两处冲突要先拍板:**
1. **磁盘**:Phase A 的 70 GB 与「拉全量 2 TiB 音频」共享那 1.46 TiB
2. **GPU**:本计划与 Stage 2 修复抢同一批卡。
   Phase A/B 回答「基座能否提高上限」;Stage 2 修复回答「现有基座的潜力有没有
   兑现」(cascade 已证明 24.41/27.89 vs 联合路径 19.64/23.49,有 5 分没兑现)

## 六、不做的事

- **不换 Talker / codec**(CosyVoice 等)。UniSS 音色通路目前完好
  (SIM-O ±0.01,A.PCP ±0.08),是我们唯一全表第一的指标
- **不动 `uniss-train` / `uniss-eval`**。新环境必须是全新建的
- **不在未修复的 Stage 2 上起 Stage 3**

## 七、未知项(要测,不猜)

1. Qwen3-Omni 音频预处理有没有「300 秒 mel 填充」那类坑
2. 25 秒音频的真实显存
3. MoE 路由在冻结推理下是否稳定
4. Talker 能否学会新 Thinker 的表示,以及从哪个起点更快
5. 30B 冻结 Thinker 下的真实步时
