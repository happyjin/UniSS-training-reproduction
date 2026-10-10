# 换 Omni 基座:具体方案

2026-09-21。决定依据:当前 0.5B 的翻译文本质量比 Omni 低 9–10.7 BLEU,
继续在 Instruct 基座上调的上限已经看到(DPO 与 MBR 两条独立路径都停在同一个天花板)。

---

## 〇、先看清 Omni 强在哪、弱在哪

SimulS2ST-Omni Table 1(CVSS-T):

| 系统 | 参数 | **Text-BLEU** en→zh / zh→en | **ASR-BLEU** en→zh / zh→en | A.PCP |
|---|---|---:|---:|---:|
| **我们 Phase3** | 0.5B | 24.15 / 15.33 | **23.58** / 12.05 | **2.79 / 2.91** |
| Qwen-Omni2.5(现成) | 7B | **34.85 / 24.39** | **8.04** / 22.66 | 1.90 / 1.92 |
| Ours(Dec-only) | 3B | 31.59 / 24.90 | 27.12 / 23.41 | 2.96 / 2.75 |
| **Ours(Thinker–Talker)** | **3B** | **33.73 / 26.41** | **31.12 / 25.18** | 2.70 / 2.64 |

**两个方向都要看清:**

* **Omni 的文本翻译确实强得多**(+10.70 / +9.06)—— 这是换它的理由;
* **但现成 Omni 的语音输出很差**:en→zh ASR-BLEU 只有 **8.04**(比我们低 15.5),
  A.PCP 1.90/1.92 是全表最低。

**结论:要的是 Omni 的 Thinker(文本),不是它的语音生成。**
而 SimulS2ST-Omni 的 **3B 双流**在文本上已接近现成 7B(33.73 vs 34.85),
zh→en 还更高(26.41 vs 24.39),语音上则高出 23 个 ASR-BLEU。

**所以选 Qwen2.5-Omni-3B + 双流配方,而不是直接用 7B 现成模型。**

---

## 一、数据:障碍已经消失 —— S3 上有现成的真实音频

### 1.1 已核实的内容

```
s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/
    opt/dlami/nvme/zeyangsong/data/unist/
      raw/                                  210 个 parquet(208 个训练输入)
      converted/unist_qwen3_s2s_v2/
        manifests/manifest-train-*.jsonl    198 个,149.6 GB
        speech/unist_bicodec_qwen3_v2/      662 个 indexed-WDS tar,2.01 TiB
        rejects/  state/  logs/  _SUMMARY.json
```

`_SUMMARY.json` 的关键字段:

| 字段 | 值 |
|---|---|
| `rows` | **19,227,252**(原 19,826,439,质量过滤剔除 599,187) |
| `source_audio_bytes` | 2,211,796,760,762 ≈ **2.01 TiB** |
| `qwen_codec_frames` | 1,704,324,507 @ 12 Hz ≈ **39,450 小时** |
| `qwen_tokenizer` | **Qwen3-TTS-Tokenizer-12Hz**(16 码本,manifest 自带的目标码) |
| `bicodec_model` | **Spark-TTS-0.5B/BiCodec** ← **正是我们在用的** |
| `license` | **CC-BY-NC-4.0**(非商用,需注意) |

**源音频总量 ≈ 39,450 小时,是论文所用 2,000 小时的约 20 倍。**

### 1.2 已端到端验证

**音频可取用**:manifest 的 `audios` 是 `wds://...tar?offset=&length=&format=flac`,
按字节范围直取即可:

```
offset=512 length=19216  →  19,216 字节  →  FLAC 可解码:1.24 s / 16 kHz / 单声道
与 manifest 的 source_duration_ms: 1240 完全吻合 ✅
```

**回链可用**:`provenance` 带 `source_parquet` + `source_row_index`,实测

```
manifest: train-00000.parquet 第 249 行,target_bicodec_len 54,bicodec_global_len 32
本地核对: id=NCSSD_R_EN_0000000666,转录/译文一致,
          target_bicodec 长度 54 ✅,bicodec_global 长度 32 ✅
```

**所以可以同时拿到:真实源音频(给 Omni 编码器)+ 我们的 BiCodec 目标码与 32 个
global token(保住 A.PCP 第 1 名的说话人通路)。**

### 1.3 目标码用哪一套 —— 这是必须做的选择

| 选项 | 来源 | 代价 |
|---|---|---|
| **BiCodec 8,192**(推荐) | 经 `provenance` 回链到本地 parquet 的 `target_bicodec` + `bicodec_global` | 训练时要同时读 manifest(音频)与本地 parquet(目标码) |
| Qwen3-TTS-12Hz 16 码本 | manifest 自带 `target_codec_codes` | 省事,但**放弃 32 个 global token 通路 = 放弃 A.PCP 优势**,且要换声码器 |

**选 BiCodec。** 我们唯一进 Table 1 前列的两个指标(A.PCP 第 1、UTMOS 第 1)都依赖这条通路。

### 1.4 落地策略:先取子集,不必拉全量

2.01 TiB 对 2.4 TB 余量太满。按论文规模取子集:

```
39,450 小时 / 198 个 train shard ≈ 199 小时/shard
2,000 小时  →  约 10 个 shard  →  约 101 GB
```

**先拉 10 个 shard(~2,000 小时,101 GB)**,与论文规模对齐;
论文还证明砍到 10%(200 小时)仍鲁棒,所以这个量是充裕的。
不够再增量拉取,**不必一次性落地 2 TiB**。

---

## 二、保留什么、替换什么

| 组件 | 现在 | 换 Omni 后 |
|---|---|---|
| 文本主干 | Qwen2.5-0.5B(词表 180,480) | **Omni Thinker**(3B) |
| 音频编码器 | WhisperVQ + GLM bridge | **Omni 原生编码器(冻结)** |
| 码生成 | 与文本共用 AR 头 | **Omni Talker**,输出头换成 BiCodec **8,192** |
| **语义码本** | **BiCodec** | **不换**(论文换 DualCodec,我们不跟) |
| **说话人** | **源音频 32 个 global token** | **不变** —— 绕过 LLM,是我们 A.PCP 第 1 名的来源 |
| **声码器** | **BiCodec** | **不变** |
| 评测链 | `cvss_t_zh_en_phase3_v1/` | **不变**,保证与历史可比 |

**必须接受的代价(写在前面)**:换掉音频编码器会使 **p2st 流式级联、Stage A/B、
本轮全部 DPO 成果失效** —— 级联的 ASR prompt 建在 `bridge_projection` +
`GLM_SEMANTIC_OFFSET` 上,WhisperVQ 被六个以上实验脚本依赖。
这些检查点与报告**保留不动**,但新线不继承它们。

---

## 三、替换方式与训练过程

### 3.1 一张图:数据如何流进新架构

```
S3 tar (FLAC, offset/length)          本地 parquet (provenance 回链)
        │                                      │
        │ 字节范围读 → FLAC 解码                 │ target_bicodec (8,192 码本)
        │                                      │ bicodec_global (32 token)
        ▼                                      ▼
  Omni 音频编码器(冻结)                   ┌──────────────┐
        │                                │              │
        ▼                                │              │
   Omni Thinker ──→ 目标文本 Y^text ──────┤              │
        │                                │   Talker φ   │
        └── 隐状态 H_θ ───────────────────┤  (头=8,192)   │
                                         └──────┬───────┘
                                                ▼
                                       BiCodec 语义码 Y^code
                                                │
                              + 32 个 global token(源音频,绕过 LLM)
                                                ▼
                                    BiCodec 声码器 → 目标语音
```

**源侧**:真实 FLAC → Omni 原生编码器(**不再用 WhisperVQ + GLM bridge**)。
**目标侧**:文本由 Thinker 出,语义码由 Talker 出,**说话人仍走 32 个 global token**。

### 3.2 目标函数

块分解目标不变(论文式 1、式 2),离线是 `C=1, g₁=|X|` 的特例:

```
C_c = ( X_{1:g_c},  Y_{<c}^text,  Y_{<c}^code )

log p(Y^text, Y^code | X, τ) = Σ_c log p(Y_c^text, Y_c^code | C_c)
```

**参数化从单头换成双流(式 4):**

```
现在   p_θ(Y_c^text, Y_c^code | C_c)              一个 AR 头,180,480 路 softmax
                                                  实测:码占 95.2% 梯度,文本只占 4.8%

换后   p_θ(Y_c^text | C_c) · p_φ(Y_c^code | C_c, Y_c^text, H_θ)

       L = Σ_c [ L_text(c) + λ · L_code(c) ]

       L_text(c) = −Σ_i log p_θ(y_i | C_c, y_<i)                  # Thinker,文本词表
       L_code(c) = −Σ_j log p_φ(z_j | C_c, Y_c^text, H_θ, z_<j)   # Talker,8,192 路
```

**Talker 不预测文本** —— 它在文本已定之后才生成码,这就是保护文本规划的机制。
`λ` 按两项损失各自 token 数归一后等权(论文正文未给此值)。

### 3.3 要新建的组件

| 组件 | 说明 |
|---|---|
| **WDS 音频 resolver** | 解析 `wds://` URI → 本地 tar 的 `offset`/`length` → FLAC 字节 → 解码。README 给了映射公式:`${WDS_AUDIO_ROOT}/${process_version}/${后续相对路径}` |
| **双源数据集** | 同时读 manifest(音频 + 文本)与本地 parquet(`target_bicodec` + `bicodec_global`),按 `provenance.source_row_index` 对齐 |
| **Talker 模块** | Omni talker,codec embedding 与输出头换成 BiCodec 8,192 + 4 个特殊 token |
| **交叉条件通路** | `H_θ` → Talker;Omni Thinker 与 Talker 维度不同,需投影层 |
| **双头 loss** | 两个交叉熵 + λ 加权 |

### 3.4 训练过程(三阶段,全程 LoRA)

```
Stage 0  手术与空操作验证                                    【新增,论文没有】
    组装 Thinker + Talker + 交叉通路,不训练。
    验收:用 gold 文本喂 Talker,前向不报错、维度对齐、
          BiCodec 码空间输出分布合理。
    成本:零训练。

Stage 1  Talker warmup                                  【抄 SimulS2ST-Omni】
    冻结 Omni 音频编码器与 Thinker,只训 Talker 与交叉通路。
    数据:TTS 任务(目标文本 → BiCodec 码),2 epoch。
    目的:让全新的 8,192 输出头在我们的码空间上收敛。
    验收:TTS 重建的 A.PCP 不低于当前管线。

Stage 2  联合预训练                                      【混合配方】
    Thinker 与 Talker 各挂 LoRA;音频编码器保持冻结。
    混合比以论文的 ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5 为起点,
    其中 S2ST 那一份**只用 Quality,不要 Performance**(2026-10-10 决定,理由见下)。
    验收:CVSS-T Text-BLEU > 24.15 / 15.33。

    ── 只保留 Quality 的依据 ────────────────────────────────────
    Q 与 P 都是**离线**模式,区别只在目标序列的构成,与流式无关:
        Quality      (SLOW_MODE)     转写 → 译文 → 语义码
        Performance  (BALANCE_MODE)  译文 → 语义码(跳过转写)

    1. Q 实测更好。项目自己的 UniST test 全量(n=14,232 / 9,110):
           eng→cmn  ASR-BLEU   quality 46.31   performance 39.00   (+7.31)
           cmn→eng  ASR-BLEU   quality  1.75   performance  1.48   (该方向两者皆坏)
    2. 项目自己的流式线早已 quality-first。
       experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1..v9
       与 uniss_stagea_quality_first_joint_grpo_v1 全部如此,
       same_prefix_teacher.py 用的是 TOKEN_SLOW_MODE。
       用 performance 的 subsecond_v2 是更早的一条线。
    3. 去掉 P 不影响流式。流式是 Stage 3 的块轨迹,与 Q/P 正交;
       P 省下的只是首声延迟,而块轨迹已在更低层面解决延迟。
    4. Omni 具备 ASR,Quality 的转写段可行 —— 实测中文源语音字错率约 9.9%
       (6 条 CVSS-T dev,两条完全正确,错误集中在专名)。
    5. Q/P 本来就不是 Omni 的东西。它们是 UniSS 在 180,480 词表里自定义的
       两个控制 token;换到 Omni 后由我们自己定义,少定义一个没有兼容负担。

    P 的配额并入 Quality。

Stage 3  流式轨迹微调                                    【抄 SimulS2ST-Omni】
    合并 Stage-2 adapter,挂新 LoRA,冻结 embedding 与预测头,
    在块轨迹上微调(ASR/S2TT/MT/S2ST)。
    基建已有:NIR 单调性过滤、固定块重分箱、轨迹池均已实现验证。
    这一步是 A.PCP 反超的来源,不可省。
    验收:A.PCP ≥ 2.7889 / 2.9073,UTMOS en→zh ≥ 3.8855。
```

### 3.5 执行顺序与决策门

```
第 1 步  现成 Omni 在我们的测试集上到底强多少          1 天,纯推理
    下载 Qwen2.5-Omni-3B(以及 7B 做规模对照)
    在 CVSS-T 上只跑 S2TT,用我们的 ASR 协议
    对照我们的 Text-BLEU 24.15 / 15.33
    门:差距远小于论文的 +10 → 停下,差距不在基座

第 2 步  拉 10 个 train shard                        半天,约 101 GB
    ≈ 2,000 小时,与论文规模对齐
    同时拉对应的 manifest

第 3 步  Stage 0 手术与空操作验证                      1 天
    门:前向打通、维度对齐

第 4 步  Stage 1 → 2 → 3                             按论文,约 Dec-only 全量的一半 GPU 小时
    每阶段按 3.4 的验收门把关

第 5 步  CVSS-T 全量评测                              1 天
    复用 experiments/evaluation/cvss_t_zh_en_phase3_v1/run_full_evaluation.sh
    沿用 whisper-large-v3-attention-mask-v2 协议,同时报告泄漏审计
```

---

## 四、验收门

| 阶段 | 必须满足 | 失败处理 |
|---|---|---|
| 1 | Omni Text-BLEU ≫ 24.15 / 15.33 | 停,差距不在基座 |
| 2 | Talker 在 BiCodec 码本上正常收敛 | 码本对齐有问题 |
| 3 | 重建音频 UTMOS 接近源音频 | 改走路径 B |
| 4 Stage 2 | CVSS-T Text-BLEU > 24.15 / 15.33 | 回退 |
| **4 Stage 3** | **A.PCP ≥ 2.7889 / 2.9073,UTMOS en→zh ≥ 3.8855** | **回退,不得用翻译收益解释掉** |

---

## 五、成本

| 步骤 | 时间 |
|---|---|
| 1 现成 Omni 对照 | 1 天 |
| 2 Talker 手术 | 2–3 天 |
| 3 重建 2k 小时音频 | 2–3 天 |
| 4 三阶段 LoRA 训练 | 按论文,约 Dec-only 全量的一半 GPU 小时 |
| 5 CVSS-T 评测 | 1 天 |

---

## 六、风险

1. ~~重建音频的 codec 痕迹~~ —— **已消除**:S3 上是真实 FLAC,不需要回解;
2. **BiCodec 与 Omni Talker 的架构匹配** —— 论文换的是 DualCodec,我们的码本更小(8,192 vs 16,384),
   理论上更容易,但需第 2 步验证;
3. **流式成果作废** —— 已接受,现有检查点与报告保留不动;
4. ~~**Q/P 双模式**~~ —— **已决定只保留 Quality**(2026-10-10)。Q 在项目自己的
   全量测试上比 P 高 7.31 ASR-BLEU,项目的流式线 v1..v9 本就是 quality-first,
   且 Q/P 与流式正交 —— 去掉 P 只放弃首声延迟的捷径,而那由 Stage 3 的块轨迹解决。
   详见 Stage 2 条目下的依据。
5. **磁盘** —— 余量 2.4 TB;先拉 10 个 shard 约 101 GB(~2,000 小时),不必落地全量 2.01 TiB。
6. **许可** —— 该数据集标注 **CC-BY-NC-4.0(非商用)**,任何对外发布前须确认合规。

---

## 七、不做的事

* 不删除任何现有数据、检查点、报告;
* 不修改现有 UniSS Phase1–3 与流式线的脚本;
* 不在第 1 步决策门之前启动任何训练。
