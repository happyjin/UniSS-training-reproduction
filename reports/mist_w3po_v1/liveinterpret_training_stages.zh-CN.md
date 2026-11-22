# LiveInterpret（streaming_si_4b）训练全流程：每个阶段训练什么、优化什么

写于 2026-11-22（布拉格时间）。

**依据（均为本机读码和读配置的结果）：**
- `third_party/liveinterpret_training_pipelines/` 下的 `pipelines/streaming_si/README.md`、`train_*.sh`、`ktrain/ktrainjob_streaming_si_*.yaml`；
- `custom/streaming_si/model/model_si.py`、`custom/streaming_si/data/s4_streaming_dataset.py`，以及两个桥接验收脚本；
- 我们实际使用的 checkpoint 的 `args.json`、`trainer_state.json`、`s4_metadata.json`。

**标注约定：**
- 仓库里没写明、只能从名字推断的地方，标为“推断”。
- 文中阶段编号采用仓库的新编号 S1–S5。源项目中对应的是 S1-0 / S2 / S1-1w / S1-1a / S3-w / S3 / S4，见 README 第 3 节。

---

## 0. 模型由哪几块组成

```
源语音 ─► Voxtral 流式编码器（冻结，80 ms/帧）─► audio_proj ─► Thinker（Qwen3.5-4B）─► 译文 token / 读写决策
                                                                   │ 隐状态
                                                                   ▼
                       talker_text_embed（冻结的原始 embedding 副本）─► text_proj ─┐
                       thinker 隐状态 ─► thinker_proj（GatedBridge）──────────────┴► fusion（门控残差）
                                                                                      ▼
                                                     Talker（Qwen3-0.6B 结构，28 层）─► CosyVoice3 FSQ speech token（25 Hz）
                                                                                      ▼
                                                                  CosyVoice3 flow + HiFT（不训练）─► 24 kHz 波形
```

| 模块 | 参数量 | 作用 | 训练情况 |
|---|---|---|---|
| Voxtral 编码器 | – | 声学特征 | **始终冻结** |
| `audio_proj` | 小 | 把音频特征投到 Thinker 输入空间 | S2、S4 训练 |
| Thinker | 4.21B | 理解源语音、生成译文；S5 起还负责读写决策 | S2、S4 全参训练；S5 用 LoRA |
| `thinker_proj` + `fusion` | 约 1,000 万 | 把 Thinker 隐状态（“说什么”的语义）接入 Talker | S3-w、S4-w 单独预热；S3、S4、S5 中都训练 |
| `text_proj` | 小 | 把译文 token 的 embedding 接入 Talker | S1、S3 和 S4 中随 Talker 一起训练 |
| Talker | 0.47B | 自回归生成语音 token（“怎么说”） | S1、S3、S4 全参训练；S5 用 LoRA |
| `talker_text_embed` | 0.64B | Thinker embedding 的冻结副本 | **始终冻结**（原因见 §2 S2） |
| 声码器 | – | speech token → 波形 | 不训练（推理时以最近 2 s 源音作为音色 prompt） |

**Talker 的三路输入：**
1. 经过桥接的 Thinker 隐状态（cond 路）；
2. 译文 token 的 embedding（text 路）；
3. 已生成的 speech token（自回归输入）。

前两路由 `fusion` 融合：`x = cond + σ(gate)·proj([cond; text])`，再经过 RMSNorm。cond 路是主干，text 路只是带门控的补充。

---

## 1. 流程总览

```
导出基座 ─► S1（TTS，talker，text_only）──┐
        └─► S2（ST，audio_proj+thinker）──┴─► 合并 ─► S3-w（桥接预热）─► S3（TTS，talker，full）
              ─► S4-w（跨语桥接预热）─► S4（离线 S2ST，全部放开）─► S5（流式读写，LoRA）─► S5 续训若干轮
```

**设计思路：**
- 先把“说什么”（Thinker 翻译）和“怎么说”（Talker 发声）分开训练（S1 ∥ S2）。
- 再用桥接把两者接起来（S3-w / S3）。
- 然后端到端联合训练离线 S2ST（S4-w / S4）。
- 最后只用 LoRA 学流式读写时机（S5）。
- 每次接入一条新的信息路时，都先单独训练桥接（“-w” 阶段），通过验收后再放开主干。

---

## 2. 各阶段

### S1：TTS 预训练（Talker 学“怎么说”）

| 项 | 内容 |
|---|---|
| 数据 | Emilia 的 FSQ v3 TTS 数据（`TTS_FSQ_FILTERED`，2,060 个 parquet，20.5 GiB）：文本、目标 speech token、参考 speech token |
| 可训练 | 整个 Talker，包括 `text_proj`（以及 fusion 中处理 text 路的部分） |
| 关键设置 | `cond_mode=text_only`：cond 路整段置零，Thinker 的输出全部丢弃；文本 loss 权重 0，语音 loss 权重 1 |
| 超参 | 每卡 32 × 2 epoch，lr 1e-4（4 节点、全局 512 时放大到 1.4e-4，约 7.4 万步），`MAX_SPEECH_LEN` 750（约 30 s），参考音 drop 0.5 |
| **优化目标** | speech token 的自回归 CE |

S1 学的是“看到文本就能说出流畅语音”，相当于一个以 FSQ token 为单位的语音语言模型，同时学会读音和声码器 token 的分布。参考音 drop 0.5 让模型在有参考音和没有参考音时都能说话。

这一阶段完全不依赖 Thinker，所以可以和 S2 并行。

### S2：语音→文本翻译（Thinker 学“说什么”）

| 项 | 内容 |
|---|---|
| 数据 | UniST P2 加上 1031 版数据：10 个翻译方向（zh/en/ja/ko）、2 个噪声目录、4 个同语言对目录（en2en、zh2zh、ja2ja、ko2ko）、3 个 machine 分片，约 9,600 万行，10.9 TB |
| 可训练 | `audio_proj` + 整个 Thinker（全参） |
| 关键设置 | 文本 loss 权重 1，语音 loss 权重 0；音频 0.5–30 s，译文 ≤256 token，超长的直接丢弃 |
| 超参 | 每卡 8 × 1 epoch，lr 1e-4；全局 1,536 时约 5.7 万步（实际用过的是 ckpt56924） |
| **优化目标** | 给定源语音，译文 token 的 CE。即把音频接到 LLM 上，训练成离线语音翻译 |

**副作用及应对：** Qwen 的 embedding 和 lm_head 是绑定的。全参训练 Thinker 后，embedding 会变：实测 S2 前后逐 token 余弦只有 0.37，接近正交。这样 S1 在原始 embedding 上学到的 text→speech 映射就会失效。因此模型另外保留了一份**冻结的原始 embedding 副本** `talker_text_embed`，专门给 Talker 的 text 路使用。

### 合并

`merge_stage_checkpoints.py` 取 S2 的 `audio_proj` 和 Thinker、S1 的 Talker，作为 S3-w 的起点。

### S3-w：桥接预热（只训练 `thinker_proj` + `fusion`）

| 项 | 内容 |
|---|---|
| 数据 | 与 S1 相同的 Emilia 数据，只用约 1/10 |
| 可训练 | 只有 `talker.thinker_proj` 和 `talker.fusion`（约 1,000 万参数） |
| 关键设置 | `cond_mode=full`，Thinker 隐状态开始进入 Talker；去掉 prompt 中重复的原文；不使用音色 prompt |
| 超参 | 每卡 32，lr 3e-4，步数在 yaml 中设定 |
| **优化目标** | speech CE，但只更新桥接。目的是让刚接进来的 Thinker 语义“对上” Talker，不破坏 S1 学到的发声能力 |
| 验收（`s3w_bridge_gate.py`） | 在同一批样本上，full 的 speech CE ≤ text_only。合并后、未预热时实测为 text_only 3.51 / full 4.95，此时接入 Thinker 反而更差；预热的目的就是让 full 至少不差于 text_only |

### S3：TTS 正式阶段（Talker 学会依赖 Thinker 语义）

| 项 | 内容 |
|---|---|
| 数据 | Emilia，3,810 万行，1 epoch |
| 可训练 | 整个 Talker，包括桥接和 `text_proj` |
| 关键设置 | `cond_mode=full`；去掉 prompt 中重复的原文，迫使 Talker 用 Thinker 的隐状态，而不只是照着文本念 |
| 超参 | 全局 2,048，约 1.86 万步，lr 1e-4 |
| **优化目标** | speech CE。Talker 主干适应“Thinker 隐状态 + 文本 embedding”联合条件 |

这里送进 Thinker 的仍是 TTS 文本，不是翻译任务。所以 S3 只是让 Talker 学会读 Thinker 的隐状态，还没有接触过跨语言数据。

### S4-w：跨语言桥接预热

| 项 | 内容 |
|---|---|
| 数据 | 与 S2 相同的 P2 + 1031 数据（S2ST：源语音 → 目标 FSQ） |
| 可训练 | 只有 `thinker_proj` 和 `fusion` |
| 关键设置 | 只计语音 loss；默认 2,500 步，lr 3e-4，warm-up 3% |
| 超参 | 4 节点全局 512 × 2,500 步，约 128 万样本 |
| **优化目标** | speech CE。S3 里 Thinker 处理的是 TTS 文本，而 S4 里 Thinker 处理的是源语音翻译，两者隐状态的分布不同（“跨池”）。这一步先只让桥接适应翻译场景下的隐状态 |
| 验收（`s4w_bridge_gate.py`，门 A） | 每个语言方向都要满足 full − text_only < −0.005，即 Thinker 路确实提供了文本路之外的信息。差值落在 ±0.005 内，说明 Thinker 路被 fusion 门控关掉了；差值为正，说明 Thinker 路是负资产 |

### S4：离线 S2ST 联合训练（全部放开）

| 项 | 内容 |
|---|---|
| 数据 | 与 S2 相同，包括同语言对。同语言对用来教“端到端语音直通”，把源语音的韵律和音色传到输出 |
| 可训练 | `audio_proj` + Thinker + Talker（含桥接），全参 |
| 关键设置 | 文本 loss 权重 1.5、语音 loss 权重 1；max_grad_norm 5.0；warm-up 3% |
| 超参 | 每卡 32，lr 5e-5，1 epoch。我们用的底座是 `S4_1031v2_ckpt93767`（93,767 步） |
| **优化目标** | 译文 CE + speech CE，端到端离线 S2ST：一整句源语音进来，先出完整译文，再出完整语音 |

**文本 loss 权重的调参经过（yaml 注释中的实测）：**
- 第一版 1:1 时，语音 CE 约是文本 CE 的 5 倍。Thinker 的梯度主要来自语音 loss，文本 loss 从 0.58 一路涨到 0.82，翻译能力被侵蚀。
- 随后把文本权重提到 2.5。
- 梯度探针显示：文本和语音两路梯度近乎正交；在 clip=1 下，权重 2.5 会让 Talker 的有效步长只剩 53%。
- 最终改为 1.5，并把 clip 放宽到 5.0。

### S5：流式读写（chunk-as-turn，LoRA）

**训练样本格式**（`si-rw-m1-v0` 记录，640 ms 一块）：

```
<|chunk|> audio_pad×n_0 [<|src_end|>] <|wait|>
<|chunk|> audio_pad×n_1               <|W|>  译文增量_1   （之后 Talker 立刻合成这段增量）
...
<|chunk|> audio_pad×n_K ... [<|src_end|>] <|wait|> <eos>
```

| 项 | 内容 |
|---|---|
| 数据 | `si_s4_rechunk_D` 中的 en2zh / zh2en 流式轨迹记录（640 ms，共约 48 GiB 的 jsonl，音频 5.4 TiB）。每条记录给出每一块之后应当等待还是写、写什么；目标语音为 `target_codec_codes`。验证集按源文本指纹分组切出 |
| 可训练（默认 `scope=joint`） | Thinker 和 Talker 的 LoRA（默认 r8/α16）：全注意力层的 q/k/v/o，线性注意力层的 in_proj_qkv / in_proj_z / out_proj，**以及 MLP 的 gate/up/down**；4 行流式 token（wait / W / chunk / src_end，embedding 与 lm_head 共享）；`thinker_proj` + `fusion` 全量。`audio_proj`、`text_proj` 和编码器冻结 |
| 损失 | `loss = text_w·split_total + speech_w·speech_CE`，其中 `split_total = (Σ译文CE + w_dec·(Σ决策CE + Σ步边界CE)) / (n_txt + w_dec·(n_dec + n_bnd))`。决策 = wait/W，步边界 = 每步结尾的 `<|chunk|>` |
| 超参（首训） | lr 3e-5 → 2e-6 余弦，每卡 48 × 1 epoch，`MAX_LENGTH` 1024，最长 30 s 源音频；用 `dec_macro_f1` 选 best（决策类别很不平衡，用准确率会把“永远写”选成 best） |
| 采样 | 按句长分层加权；延迟倍数 m（`SI_S4_LATENCY_MULTIPLIER`，默认 1，可随机取 1–4）；实验臂 prefix-drop（裁掉前缀上下文），只训练过 default 臂 |
| **优化目标** | ① 读写时机：每 640 ms 决定等还是说；② 在部分源音频上生成译文增量；③ Talker 对短增量逐段合成。它学的是离线 S4 能力的**流式化**，不教新的翻译知识 |

**S5 首训结果**（2026-10-13，旧版 zh/en 底座，3 节点 24 卡，17,309 步，约 6.4 小时）：
- dec_macro_f1：第 2000 步 0.775，末段 0.786；
- text_loss：0.886 → 0.785。

结论是读写时机在前 12% 的步数里就基本学完了，后面主要在提升译文和语音。

### S5 的后续续训（我们 E10 / v3 的起点）

| 轮次 | checkpoint | 已知信息 |
|---|---|---|
| S5 首训 | `s5_simul_default_640/checkpoint-16000`（best） | r8，640 ms，zh/en，旧底座 |
| S5 “off_mchunk_ft” | `s5_enzhjako_off_mchunk_ft_lora_r64_20261106/checkpoint-10000` | 换到 `S4_1031v2_ckpt93767` 底座，LoRA r64，en/zh/ja/ko。**推断**：名字中的 off / mchunk 指加入离线样本、多种块长（同事的推理脚本支持 640/1280/1920/2560 ms 和两段式块长），仓库里没有写明数据配方 |
| S5 续训（p4mix / “ours10”） | `p4mix_2n_20261117/checkpoint-29773` | 从上一行加载 adapter 后重新开始（不是 resume）；joint scope，r64/α128，lr 1.2e-5 → 2e-6，24 卡 × 16，1 epoch，29,773 步（约 1,140 万条）。eval 结果：dec_macro_f1 第 2000 步 0.840，末步 0.848；text_acc 0.743 → 0.749；speech_acc10 0.756 → 0.760。**推断**：p4mix 指混入第 4 批数据，配方不在仓库里 |

续训期间各项指标几乎不动，说明这一轮主要是在新数据混合上保持原有能力，而不是学新能力。

---

## 3. 一张表看清“谁在什么时候被优化”

| 阶段 | Voxtral | audio_proj | Thinker | 桥接 thinker_proj + fusion | text_proj | Talker 主干 | 流式 token | 文本 / 语音 / 决策 loss |
|---|---|---|---|---|---|---|---|---|
| S1 | 冻 | – | 前向被跳过 | cond 路置零 | 训 | 训 | – | 0 / 1 / – |
| S2 | 冻 | 训 | 全参 | – | – | – | – | 1 / 0 / – |
| S3-w | 冻 | 冻 | 冻 | **只训这里** | 冻 | 冻 | – | 0 / 1 / – |
| S3 | 冻 | 冻 | 冻 | 训 | 训 | 训 | – | 0 / 1 / – |
| S4-w | 冻 | 冻 | 冻 | **只训这里** | 冻 | 冻 | – | 0 / 1 / – |
| S4 | 冻 | 训 | 全参 | 训 | 训 | 训 | – | 1.5 / 1 / – |
| S5 | 冻 | 冻 | LoRA（attn + MLP） | 全量 | 冻 | LoRA | 4 行 | 1 / 1 / 1 |
| E10（上一版，我们） | 冻 | 冻 | LoRA r64（从 S5 继续）+ 12 行 token + `e10_sd_proj` | 冻 | 冻 | 冻 | – | 1 / 0 / 1 |
| v3（我们） | 冻 | 冻 | 新 LoRA r32，只加在 attn；8 个说话人标签行 + SDProjector | 冻 | 冻 | 冻 | – | 1 / 0 / 1（离线和单说话人记录的决策 loss 屏蔽） |

---

## 4. 对我们多说话人工作的含义

1. **翻译知识来自 S2 和 S4，S5 只是把它流式化。**
   - S2/S4 是约 9,600 万行的全参训练，S5 在其上只用 LoRA。
   - 我们的 E10 和 v3 也是在 S5 之上加 LoRA，所以翻译能力的上限基本由 S4 决定。
   - B 的 Thinker 是 Qwen3-Omni-30B-A3B。v3 中→英 BLEU 仍低于 B，主要是底座差距，难以靠在 S5 之上加 LoRA 补回来。
2. **读写策略是在 S5 中学会的，规律简单、学得很快**（第 2000 步就基本收敛）。
   - 所以 E10 的数据只要把读写轨迹定义错（离线行被展开成“一直等、最后一次写完”），很快就会被学坏。
   - v3 把这类样本的决策 loss 屏蔽后，读写行为就恢复了正常。
3. **Talker 从没见过多说话人、长窗口和带说话人切换的输入。**
   - S1、S3 是单人 TTS；S4、S5 是单说话人、≤30 s 的句子。
   - 音色由声码器 prompt（最近 2 s 源音）决定，Talker 本身没有说话人条件。
   - E10 和 v3 都冻结了 Talker，所以多说话人时的音色区分只能靠按说话人分路、各自换参考音。听感上的分离度和停顿上限由这一点决定。
4. **S4 的同语言对数据是专门用来教“韵律和音色直通”的。**
   - 我们冻结了 Talker，也冻结了桥接，所以保留了这部分能力。
   - 但多说话人的韵律仍完全依赖 S4/S5 在单说话人数据上学到的东西。
5. **我们与 S5 的训练部位不同。**
   - S5 训练了 Talker LoRA、桥接和 MLP LoRA。
   - v3 为了和 B 对齐，只在 Thinker 的注意力上加 LoRA，Talker 和桥接都冻结。
   - 因此 v3 的 Thinker 隐状态一旦漂移，桥接没有机会跟着适应。如果以后要改善语音质量和停顿，可以考虑放开 `thinker_proj` + `fusion`，即 S4-w 式的桥接微调。这一项未实验。
