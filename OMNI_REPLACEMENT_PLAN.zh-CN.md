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

## 一、最大的障碍:我们没有训练用的原始音频

已核实:

```
UniST parquet 字段:source_glm / source_bicodec / target_glm / target_bicodec
                    / bicodec_global / transcription / translation / ...
                    —— 全部是 token 化表示,没有任何音频字段
磁盘原始音频:RealSI 777 条(测试集)+ data/raw 20 条,其余为 0
余量:2.4 TB
```

**Omni 的音频编码器吃原始波形,不吃我们的 `glm_semantic` token。**
现有 1.47 TB 打包数据对 Omni **完全不可用**,必须重建音频。

### 三条路,按成本排

| 路径 | 做法 | 存储 | 风险 |
|---|---|---|---|
| **A. BiCodec 回解**(推荐先试) | `source_bicodec + bicodec_global` → BiCodec 解码 → 波形 | **~2k 小时 ≈ 230 GB** | 编解码重建有损,Omni 编码器要吃"带 codec 痕迹"的音频 |
| B. 取回原始语料 | UniST 来源为 LibriTTS-R / GigaSpeech / CommonVoice / WenetSpeech4TTS / MagicData / NCSSD / DailyTalk / HiFi-TTS | TB 级下载 + 与 UniST 行重新对齐 | 工程量大,但音频是真的 |
| C. 全量回解 | 19.8M 条全部解码 | **~3.2 TB > 余量 2.4 TB** | **不可行** |

**选 A,且只做 ~2k 小时** —— 这正是 SimulS2ST-Omni 的数据规模
(*"only ~2,000 hours of high-quality Chinese-English paired S2ST data"*),
论文还证明砍到 10% 仍鲁棒。

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

## 三、实施顺序

### 第 1 步:现成 Omni 在**我们的**测试集上到底强多少(1 天,纯推理)

**这一步不能跳。** 34.85 是论文在他们的协议下测的;
我们要知道的是 Omni 在 **CVSS-T + 我们的 ASR 协议**下相对 24.15 / 15.33 的真实差距。

```
下载   Qwen2.5-Omni-3B(以及 7B,用于判断规模是否必要)
验证   离线可加载;Thinker / Talker / 音频编码器三部分可分离
测试   RealSI 777 条真实音频 + CVSS-T,只跑 S2TT(语音→文本)
对照   我们的 Text-BLEU 24.15 / 15.33
```

**决策门**:
* Omni 3B 的 Text-BLEU 显著高于 24.15 / 15.33 → 继续;
* 若差距远小于论文的 +10 → 说明差距来自协议或数据而非基座,**停下重新评估**。

### 第 2 步:Talker 手术与码本对齐(2–3 天)

```
初始化  Omni Talker
替换    codec embedding 与输出头 → BiCodec 语义码本 8,192
        + 4 个特殊 token(codec_bos / codec_eos / codec_pad / codec_mask)
        (论文对 DualCodec 16,384 做的是同一件事)
验证    用真实 (文本, BiCodec 码) 对做教师强制,确认 Talker 能收敛到我们的码空间
```

**决策门**:Talker 在 BiCodec 码本上的教师强制困惑度是否正常收敛。

### 第 3 步:重建 ~2k 小时训练音频(2–3 天)

```
选样    从 UniST 198 shard 按 NIR 难度/长度分层抽样(nir_score.py / nir_stratify.py 已实现)
解码    BiCodec(source_bicodec, bicodec_global) → 16 kHz 单声道波形
        8 卡并行,复用 evaluation/decode_audio.py 的批量解码
产出    ~230 GB,新目录,不动任何现有数据
抽检    随机 200 条人工试听 + UTMOS,确认重建音频可用作编码器输入
```

**决策门**:重建音频的 UTMOS 若显著低于源音频,说明 codec 痕迹太重,
改走路径 B(取回原始语料)。

### 第 4 步:三阶段训练

```
Stage 1  Talker warmup
         冻结 Thinker 与音频编码器,只训 Talker,仅用 TTS 数据,2 epoch

Stage 2  联合预训练
         ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5
         Thinker 与 Talker 各挂 LoRA;音频编码器保持冻结
         ⚠ 保留 UniSS 的 Quality / Performance 双模式任务定义 ——
           论文配方里没有 Q/P,照抄会丢掉我们 Q 比 P 高 3.57–5.06 BLEU 的能力

Stage 3  流式轨迹微调
         合并 Stage-2 adapter,挂新 LoRA,冻结 embedding 与预测头
         在块轨迹上微调 —— A.PCP 反超发生在这一步,不可省
```

**全程 LoRA,不做全参数预训练。** 论文在 8×A800 上即可完成。

### 第 5 步:评估

复用 `experiments/evaluation/cvss_t_zh_en_phase3_v1/run_full_evaluation.sh`,
沿用 `whisper-large-v3-attention-mask-v2` 协议,同时报告泄漏审计。

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

1. **重建音频的 codec 痕迹** —— 第 3 步的决策门专门挡这个;
2. **BiCodec 与 Omni Talker 的架构匹配** —— 论文换的是 DualCodec,我们的码本更小(8,192 vs 16,384),
   理论上更容易,但需第 2 步验证;
3. **流式成果作废** —— 已接受,现有检查点与报告保留不动;
4. **Q/P 双模式** —— 论文配方里没有,Stage 2 必须自己保住;
5. **磁盘** —— 2.4 TB 余量,2k 小时音频约 230 GB,可行;全量回解不可行。

---

## 七、不做的事

* 不删除任何现有数据、检查点、报告;
* 不修改现有 UniSS Phase1–3 与流式线的脚本;
* 不在第 1 步决策门之前启动任何训练。
