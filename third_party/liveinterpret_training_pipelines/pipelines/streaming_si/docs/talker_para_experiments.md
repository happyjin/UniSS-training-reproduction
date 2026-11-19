# 增强 talker 的副语言与音色建模：三个实验设计

> 配套：`paralinguistic_preservation.md`（通路现状与探针实测）、
> `timbre_and_paralinguistic_ablation.md`（推理侧消融：音色三通路、token 携带说话人信息）。

## 0. 为什么现在要做（前提已变）

两个此前的结论被推翻，使得"改 talker 结构"从"暂缓"变成"值得做"：

| 旧结论 | 现状 | 影响 |
|---|---|---|
| 「数据里情感很少，目标 TTS 压平了情感，模型学不到」 | **错**。目标音频用**源说话人的 `bicodec_global`** 解码（`p2_unist_to_fsq.py` L22），音色与风格条件同源；旧结论依据的 emotion2vec ESim delta（+0.02~0.03）来自一个我们已独立证明近乎无区分度的指标（对**所有**系统 delta 都是 0.01~0.03）。实测见 §1 | 数据侧有信号 → 瓶颈在**模型通路**，结构实验有意义 |
| 「训练 talker 不会提升 SIM-o，音色全由 flow 条件决定」 | **错**。源自身 token + 两路条件全关仍得 SIM-o 0.748 / delta 0.130（vs talker token 的 0.487 / 0.005）：speech token 本身携带说话人信息 | 改 talker 可同时提升**音色与副语言** |

另有一条旧实测需要正确解读，不能当作"换层无用"的依据：

> **实验 4（换 talker 取层到 L4/L8）无收益**，但那是在 **S4-w scope（只训 `thinker_proj` + `fusion`，10.5M 参数）**
> 下测的。文档自己写明「探针上 L4~L8 的 R² 更高，但 **talker 只训桥接时用不上这部分差距**」
> （`paralinguistic_preservation.md` L30-31）。本轮三个实验都**重训 S3/S4**，talker 主干与
> thinker 一起更新，与实验 4 不是同一结论域。

## 1. 数据侧事实核验（实验的前提）

脚本：`custom/streaming_si/eval/check_p2_pair_para.py`

做法：从 P2 parquet 采样，用 **S3 字节范围 GET** 精确取出 `source_wav` 与 `target_wav`
（目标 tar 不在训练节点，但在 S3 同前缀下，7GB tar 只取需要的几十 KB），然后测
源↔目标的 SIM-o / AutoPCP / ESim（都带随机配对基线），并用**不依赖神经指标**的
逐句声学量（F0 均值与标准差、能量标准差、停顿占比、时长）做交叉验证。

同时按「源语音表现力」（F0 std + energy std 的 z-score 之和，不用情感分类器，避免
emotion2vec 的中立偏置）排序导出 wav 对，供人工评审。

### 1.1 实测结果（841 对：machine_0 400 对 eng→cmn + machine_1 441 对混合方向）

| 指标 | machine_0 配对 / 随机 / delta | machine_1 配对 / 随机 / delta |
|---|---|---|
| **SIM-o** | 0.850 / 0.621 / **+0.229** | 0.920 / 0.669 / **+0.251** |
| **AutoPCP** | 3.264 / 2.485 / **+0.779** | 2.972 / 2.263 / **+0.709** |
| ESim | 0.848 / 0.822 / +0.026 | 0.898 / 0.885 / +0.013 |

**不依赖神经指标的逐句声学相关（源 vs 目标）**：

| 量 | machine_0 | machine_1 |
|---|---|---|
| F0 均值 | **+0.894** | **+0.968** |
| 时长 | **+0.898** | **+0.952** |
| 停顿占比 | +0.804 | +0.670 |
| 有声段占比 | +0.683 | +0.642 |
| 能量标准差 | +0.465 | +0.563 |
| F0 标准差 | +0.385 | +0.612 |

**结论**：

1. **音色高度对齐**（SIM-o delta +0.23~0.25），与「源/目标共用 `bicodec_global`」一致。
   作为参照：**我们 S4 生成语音的 SIM-o delta 只有 0.162**，即**数据给的上限比模型当前做到的高**。
2. **韵律显著迁移**：F0 均值与时长相关 0.89~0.97，停顿 0.67~0.80，能量与 F0 起伏 0.39~0.61。
   AutoPCP delta +0.71~0.78，与我们 S4 的 0.782 相当 —— 说明**数据侧韵律已不是瓶颈**。
3. **ESim delta 仍只有 +0.013~0.026**。但这不能解读为"数据里没情感"：
   同一指标在 RealSI/ACL 评测上对**所有 12 个系统**的 delta 都是 0.01~0.03
   （`README.md` 观察第 8 条），**该指标在本任务上近乎无区分度**，不能用其小 delta 反推数据性质。
   8 条小样本上曾看到 +0.115，扩到 841 对后回落，进一步说明它方差大、不可依赖。
   情感是否真的保留，**以人工评审为准**。

**人工评审样本**（各 12 组，按源语音表现力 = F0 std + energy std 的 z-score 之和排序，
即情感/语调最夸张的样本；**刻意不用情感分类器排序**，以避开 emotion2vec 的中立偏置）：

- `/fsx/stevehguo/eval/p2chk/m0/review/REVIEW.md`（eng→cmn）
- `/fsx/stevehguo/eval/p2chk/m1/review/REVIEW.md`（混合方向）

每组一个子目录，含 `src.wav`（源）与 `tgt.wav`（目标），`REVIEW.md` 附源文、译文、各指标。

> **注意**：AutoPCP 已修正为官方 v2 口径（`pick_layer=9` + `symmetrize=true`），
> 旧数值（layer 7 / 单向）不可与本表比。

## 2. 三个实验的共同设置

### 2.1 现状基线（要改的东西）

本节及全文的结构与层号**一律按 `streaming_si_4b`**（`model/variants.py` L28-29）：

```
Voxtral(32L, 1280d) --adapter--> 3072 @12.5Hz --audio_proj--> 2560
                                                                  │  替换 <|audio_pad|> 位置
                                                                  ▼
                        thinker = Qwen3.5-4B（32L / 2560d，3:1 Gated DeltaNet : 门控全注意力）
                                                                  │
                                 ┌────────────────────────────────┤ 只取 answer_mask==1
                                 │                                │ 即**译文位置**、**最后一层**
                                 ▼                                ▼
                    talker.thinker_proj                   talker.text_proj
                    = GatedBridge(2560 → 1024)            = Linear(2560 → 1024)
                                 └──────── TalkerFusion ──────────┘
                                                │
         talker = Qwen3-0.6B 结构（28L / 1024d，16/8 头，head_dim 128，ffn 3072）
                                                │
                                   llm_decoder → CV3 FSQ 6565（6561 + 4）
```

> **不要套用 0.6B 变体的规格。** `streaming_si`（Thinker Qwen3-0.6B 28L/1024d）配的是
> **0.1B talker：8 层 / 1024d / 16-4 头**（`variants.py` 的 `0.1b` preset = `QWEN3_TALKER_8L`，
> `base_model.py` L256 标为 `weight_tag=talker8`）。本轮三个实验全部在 4B 上做，
> talker 是 **28 层**而非 8 层——两者参数量差约 6 倍，涉及"加一路条件要不要重新预热""显存够不够"
> 的判断时不可混用。

关键事实（`model/model_si.py`）：
- `pick_bridge_hidden` L841-847：`talker_bridge_layer < 0`（默认 -1）取末层；`>=0` 取 `hidden_states[layer+1]`——**取层已可配置**；
- `gather_answer_hidden` L849-858：按 `answer_mask==1` 抽取 → **只有译文位置**进 talker，音频帧位置不进；
- `si_encode.py` L127：`answer_mask = [0]*(n_frames+len(instr)) + [1]*len(tgt)`；
- Voxtral `VoxtralEncoder.forward`（`voxtral_compat.py` L331-355）**没有** `output_hidden_states`，取 L4 需新增；
- Voxtral 与 `talker_text_embed` **始终冻结**，首步有 `_check_frozen_invariants` 强校验（`modeling_streaming_si.py` L184-199）。

#### 层号换算（4B = 32 层；配错一层就对不上探针）

`hidden_states[0]` 是 embedding 输出，`hidden_states[i]` 是第 *i* 层输出，而
`pick_bridge_hidden` 取 `hidden_states[layer+1]`，所以：

```
SI_TALKER_BRIDGE_LAYER = k   →   hidden_states[k+1]   →   探针口径的 L(k+1)
```

实验 4 实锤了这个换算：`取层 3 / 7` 在探针文档里记为 **L4 / L8**
（`paralinguistic_preservation.md` L401、L477-478）。

**与 Qwen3-Omni 同口径**：Omni 的 `accept_hidden_layer=N` 取第 *N* 层输出
（`custom/qwen_omnispeak/mcore_talker_loss.py` L171 `idx = accept - 1`）。本仓库
`qwen_omnispeak` 的实配是 **`accept_hidden_layer=24`、Thinker 共 48 层**
（Qwen3-Omni-30B-A3B，`QWEN_OMNISPEAK_MODEL_DESIGN.md` L8/L18/L43/L53），即 **24/48 = 恰好半深度**。

> 早先此处写「18/36」，两个数都不对（已改）。不影响结论：24/48 与 18/36 都是 0.50。

换到 4B thinker 的 **32 层**，半深度 = 第 16 层输出 = `SI_TALKER_BRIDGE_LAYER=15`。

### 2.2 统一的训练与评测口径

| 项 | 取值 |
|---|---|
| **变体** | **4B（`streaming_si_4b`）**，不走 0.6B |
| **起点** | S4 最终 checkpoint（`checkpoint-24980`，即评测用的 `s4_ck`） |
| **流程** | 重初始化桥接 → **S4-w 式预热**（2500 步，只训桥接）→ **S4 joint** |
| 对照 | 从同一 `checkpoint-24980` 出发、同样重初始化桥接、同样预热+joint，但**不改通路**（末层 + 译文位置）。种子 / 步数 / 数据 / LR 调度全部一致 |
| 主 KPI | **AutoPCP delta**（官方 v2 口径：layer 9 + symmetrize）、**SIM-o delta**；次：ASR-BLEU / COMET 不许掉超过 0.5 BLEU |
| 诊断指标 | 译文位置探针 R²（韵律/情感）、时长比、UTMOS |
| 评测集 | RealSI en↔zh + ACL 60/60，用既有 `run_worker2.sh` 流水线；**必须用留出样本**（当前探针样本在 S4 训练集内，`paralinguistic_preservation.md` 5.2 末尾） |

#### 为什么用 4B 而不是先 0.6B（修正早先的判断）

早先写「0.6B 先做、通过再上 4B」，核实后撤销：

- **评测与探针证据全部来自 4B**。`s4_ck/config.json`：`hidden_size=2560`、`num_hidden_layers=32`、
  `talker_num_hidden_layers=28` → `streaming_si_4b`（Qwen3.5-4B + talker 0.6b）；
  `paralinguistic_preservation.md` 5.2 的 checkpoint-6000 同属该链，表头的 **L32** 就是「talker 当前取的层」。
- **取层结论不可迁移**：探针峰值 L4~L8、末层只剩 L4 的 45%，都是 32 层 Qwen3.5-4B（含 3:1 Gated
  DeltaNet / 门控全注意力）上的分布；0.6B 是 28 层 Qwen3-0.6B、hidden 1024，层行为不同。
- **0.6B 没有可比基线**：所有已发布数值（S4 ASR-BLEU 26.48、AutoPCP delta 0.782 等）都是 4B。
  在 0.6B 上做要额外训一条完整对照链，比直接在 4B 上做更贵，且会制造新的口径差异
  （本轮已因口径差异踩过两次：prompt 静默失效、AutoPCP 参数错配）。

#### 为什么**不需要**重训 S3

S3 教的是 talker「**怎么说**」（Emilia 单语 TTS：文本 → FSQ token）。三个实验改的是
「talker **看什么条件**」，没有改 talker 的声学建模能力本身，而该能力已在 S4 checkpoint 内且仍有效。
为换条件通路而重训 S3 属于重复与改动无关的工作。真正必需的不是 S3，而是**桥接预热**（下节）。

#### 从 S4 续训的三条强制要求

1. **重初始化 `thinker_proj` + `fusion`，不要沿用 S4 的权重。**
   `train_tts.sh` L15 实测：「只放 `thinker_proj` 时，`fusion` 的门控还停在『cond 恒零』的解上」。
   同理，S4 的 `fusion` 门控是为「末层译文位置」调好的；换通路后若沿用，门控可能继续按旧解把
   新信息压掉 —— 结果是**测不出新通路的好处，但原因是门控没动，不是通路没用**，会重演实验 4 的陷阱。
2. **先 S4-w 式预热（2500 步、只训桥接、lr 3e-4）再进 joint。** 验收用门 A：
   `python custom/streaming_si/scripts/s4w_bridge_gate.py --model <ckpt> --data '<P2 glob>' --wav_dir <tar>`，
   要求每个方向 full 的 speech CE < text_only。
3. **LR 调度必须与对照臂逐字一致。** `common_args.sh` L159 是 `cosine_with_min_lr`；S4 跑满 1 epoch 后
   LR 已衰到底，续训等于热重启，有冲击。对照臂用同一套调度才可比。

#### 从 S4 续训会碰到的其它问题（与实验本身无关，但要盯）

| 问题 | 依据 | 应对 |
|---|---|---|
| 数据进入第二轮，样本重复、过拟合风险 | S4 是 1 epoch ≈ 24980 步，24980 正好一轮结束 | 盯 held-out 上的 speech CE；必要时换 machine_2 分片 |
| 翻译能力被侵蚀 | `train_s2st.sh` L43 实测：24 卡 / lr 8e-5 下文本 loss 4300 步内 0.58 → 0.82 | 主 KPI 里设「ASR-BLEU 不许掉超 0.5」；必要时调高 `SI_TEXT_LOSS_WEIGHT` |

#### 结论的适用范围（重要限制）

从 S4 续训得到的是「**在一个已按末层条件收敛的模型上换通路，有没有用**」，不等于
「从头训练时哪种通路更好」。因为 S4 的 thinker 表示已被「把副语言写进译文位置」这个压力塑造过
（5.2：训到 24% 时译文位置末层 R² 已从 0.024 → 0.092）；若实验 b 成立（talker 直接看音频位置），
thinker 其实**不再需要**把副语言挤进译文位置，从头训会得到不同的分工。
从头训（S1→S2→S3→S4）成本高一个量级，故**先用续训做筛选**，哪个臂有效再决定是否值得从更早阶段验证。

### 2.3 风险与门控

两级门控，避免把坏臂送进昂贵的 joint：

1. **门 A（预热后）**：`s4w_bridge_gate.py` 要求每方向 full 的 speech CE < text_only。
   不达标说明新通路没被 talker 用上，停下来查。
2. **早停门（joint 前 1/10 步）**：若 speech CE 比对照高 >5%，或文本 CE 涨幅超过对照 2 倍，停。

---

## 3. 实验 a：talker 改接 thinker 中间层

**改什么**：`talker_bridge_layer` 从 -1（末层）改为中间层。

**依据**（全部为 4B / S4 链的实测，`paralinguistic_preservation.md` 5.2、5.3）：
副语言信息集中在 thinker 译文位置的**浅中层**，韵律 R² 为 **L4 0.203 / L12 0.159 / 末层(L32) 0.092**
（同文 L201、L375），末层只剩 L4 的约 45%（韵律）/ 2/3（情感）。Qwen3-Omni 同样取半深度
（`accept_hidden_layer=24` / 共 48 层）而非末层。

**代价**：几乎为零——`SI_TALKER_BRIDGE_LAYER` 已是现成开关（`register_streaming_si.py` L81-83），
`forward_thinker` 会自动打开 `output_hidden_states`（L831-838）。只是会多一份 hidden 的显存。

**配置**（4B 的 thinker 共 **32** 层；取层 → 探针层号的换算见 §2.1）：

| 臂 | `SI_TALKER_BRIDGE_LAYER` | = 探针层 | 该层韵律 R²（S4@6000） | 说明 |
|---|---|---|---|---|
| a0（对照，三实验共用） | `-1` | L32（末层） | 0.092 | 现状 |
| a1 | `6` | **L7** | 介于 L4 0.203 与 L8 之间 | 探针峰值区（L4~L8）内 |
| a2 | `15` | **L16** | 约 0.14（L12 0.159 与 L32 0.092 之间） | 半深度，与 Omni 的 18/36 同口径 |

任务 yaml 由 `custom/streaming_si/scripts/gen_expa_yaml.py` 生成（勿手改），
它同时固化了节点白名单的三条硬约束：机型不能混（EFA 数不同）、EFA 可能被同节点他人任务占满、
P2 数据同步实占 **2.1 TB** 而驱逐阈值是 `nodefs.available<10%`。

> **两处口径修正**（原表有误，已改）：
> 1. 原表给 a1 标注的 R² 0.203 是**探针 L4** 的值，而 `取层=6` 落在**探针 L7**。a1 已按 `6` 启动
>    （2026-10-19），不回改取值，仅把标注改对——它仍在 L4~L8 峰值区内，依据成立。
> 2. 原表 a2 写 `16`，但那对应探针 **L17**（17/32 = 0.53），并非声称的"半深度"。
>    要与 Omni 严格同口径（`accept_hidden_layer=24` / 共 48 层 = 0.50）应取 `15`（→ 探针 L16）。
>    差一层对结果几乎无影响，但既然 a2 尚未启动，就按正确口径定为 `15`。

### 3.1 预热实测（2026-10-19）：换层**确实有收益**，推翻实验 4

a0 / a1 各跑 2500 步（25 分钟，16 卡）后用门 A 验收，**两臂两个方向全部 PASS**：

| 臂 | 方向 | text_only | full | full − text_only | 判定 |
|---|---|---|---|---|---|
| a0（末层 L32） | cmn→eng | 3.0490 | 2.8075 | −0.2415 | PASS |
| | eng→cmn | 3.2142 | 2.9485 | −0.2657 | PASS |
| **a1（取层 6 = 探针 L7）** | cmn→eng | 3.0684 | **2.7863** | **−0.2821** | PASS |
| | eng→cmn | 3.2357 | **2.8834** | **−0.3522** | PASS |

**a1 在两个方向上都优于 a0**：thinker 路增益多 0.041（cmn→eng）/ **0.087**（eng→cmn），
绝对 full CE 也都更低；训练 loss 同向（a0 3.355 / a1 **3.099**）。

与实验 4 对照，差别只在**是否重初始化桥接**：

| 轮次 | cmn→eng | eng→cmn | 结论 |
|---|---|---|---|
| 实验 4（**未**重初始化）`l7` vs `lc` | −0.2653 vs −0.2680 | −0.2319 vs −0.2380 | 换层**略差** |
| 本轮（**已**重初始化）`a1` vs `a0` | −0.2821 vs −0.2415 | −0.3522 vs −0.2657 | 换层**明显更好** |

这正是下方预注册所说的第三种情形：**实验 4 的"无收益"是 fusion 门控停在为末层调好的旧解上
造成的假象**，不是取层没用。§2.2 把"重初始化桥接"列为强制要求是对的，而且现在有了直接证据。

保留项：门 A 用训练集样本（该阶段无留出集），且 CE 只是代理量；最终判据仍是 joint 后的
AutoPCP / SIM-o delta。a2 的预热尚未完成。

**与实验 4 的区别（关键）**：实验 4 得出"换层无收益"，但那是在 **S4-w scope（只训
`thinker_proj`+`fusion`，10.5M 参数）** 下测的，文档自己写明「探针上 L4~L8 的 R² 更高，但
**talker 只训桥接时用不上这部分差距**」（L30-31）。本实验预热后**继续 joint**（thinker + talker
一起更新），与实验 4 不是同一结论域。

**预注册：预热阶段的门 A 不构成判据，别误读。** 本轮预热与实验 4 的 `l3`/`l7` 臂
**起点、scope、超参全部相同**（S4 最终 checkpoint-24980、冻结 thinker、只训桥接、
全局 256 / lr 3e-4 / 2500 步），唯一差别是本轮**重初始化了桥接**。实验 4 的实测是
（`paralinguistic_preservation.md` L484-520）：

| 实验 4 的组 | 门 A（full − text_only，两方向） | 韵律偏相关 vs 末层对照 |
|---|---|---|
| `l3`（探针 L4） | −0.2651 / −0.2313 | −0.013 ~ +0.005，95% 区间跨 0 |
| `l7`（探针 L8） | −0.2653 / −0.2319 | 同上 |
| `lc`（末层对照） | −0.2680 / −0.2380 | — |

三臂都过门 A，但换层比末层**少 0.003~0.007** 的 thinker 路收益。所以：

- a1/a2 预热后门 A 若**再次略差于 a0**，这只是**复现**实验 4，**不能**据此停掉该臂——
  §2.2 要求的桥接重初始化正是为排除"门控停在旧解"这一混淆，而预热阶段 thinker 冻结，
  探针测到的浅层优势按定义用不上。
- 真正的判据在 **joint 之后**的 AutoPCP delta / SIM-o delta。
- 反过来，若 a1/a2 预热后门 A **优于** a0，说明重初始化本身就解开了门控，
  这是超出实验 4 的新信息，值得单独记一笔。

**最终判读**：若 joint 后 a1/a2 的 AutoPCP delta 显著高于 a0，说明实验 4 的"无收益"确实是被
"只训桥接 + 未重初始化"掩盖的；若仍无差异，则副语言的损失不在"取哪层"，而在"只看译文位置"——
直接转向实验 b/c。

**成本**：预热 2500 步 + joint。a0 是三实验共用对照，故 a 的净增成本只有 a1（+ 可选 a2）。

---

## 4. 实验 b：改取源音频位置的 hidden（而非译文位置）

**改什么**：`gather_answer_hidden` 现在只取 `answer_mask==1`（译文位置）。改为取
**音频帧位置**（`answer_mask==0` 段里对应 `<|audio_pad|>` 的那些位置）的 hidden 作为
talker 的 thinker 路条件。

**依据**：这正是 `paralinguistic_preservation.md` 的"方案 A"，也是 Qwen2.5/3-Omni 的做法
（前缀含音频位置）。译文位置的 hidden 已被"预测下一个译文 token"这个目标压缩过，
副语言是副产物；音频位置的 hidden 直接承载声学信息。探针数据支持：
Voxtral L4 的韵律 R² 0.286 → thinker 输入 0.216 → **译文位置末层只剩 0.024**
（5.1 实验 2/3）——信息是在"进入译文位置"这一步丢掉的。

**实现要点**（需改代码，约 40 行）：
1. `si_encode.py` 额外输出 `audio_mask`（标记 `<|audio_pad|>` 位置），与 `answer_mask` 并列；
2. `model_si.py` 新增 `talker_cond_positions ∈ {answer, audio, audio+answer}` 配置：
   - `audio`：只取音频位置 hidden；
   - `audio+answer`：音频位置在前、译文位置在后拼接（信息最全，序列更长）；
3. **长度对齐**是关键问题：talker 的条件序列长度此前等于译文 token 数，与 speech token
   的交错逻辑（S5）绑定。音频位置是 12.5Hz、译文是 token 粒度，两者不等长。
   - S1/S4（前缀式）：直接拼接即可，不需要逐位对齐；
   - **S5（流式交错）会受影响**，需同步改 `build_stream_talker_condition`。
     建议本实验**先只做到 S4**，S5 等结论确定后再改。
4. text 路（`talker_text_embed`）保持按译文位置取，不动——保留"说什么"的信息。

**臂**：

| 臂 | `talker_cond_positions` | 取层 |
|---|---|---|
| b0（对照） | `answer` | -1 |
| b1 | `audio` | -1 |
| b2 | `audio+answer` | -1 |
| b3 | `audio+answer` | 6（与实验 a 的结论合并） |

**预期**：b 是三个实验里**最可能见效**的，因为它直接消除了"译文位置瓶颈"这个已被探针
定位的损失点。风险是条件序列变长、训练变慢，且音频位置含大量源语言内容信息，
可能干扰翻译（要盯 ASR-BLEU）。

---

## 5. 实验 c：Voxtral 中间层作为独立通路接入 talker

**改什么**：新增一条 `Voxtral L4 → talker` 的旁路，与现有 thinker 路、text 路三路融合。

**依据**：Voxtral L4 的韵律 R² 0.286 是全链路最高点（thinker 输入 0.216、译文位置末层 0.024）。
实验 b 取的是"thinker 内部的音频位置"，已经过 `audio_proj` 与若干层 thinker；实验 c 直接
从**声学编码器浅层**引线，绕过 thinker 的全部压缩。

**实现要点**（改动最大，约 120 行）：
1. **Voxtral 需要支持取中间层** —— 当前 `VoxtralEncoder.forward`（`voxtral_compat.py` L331-355）
   不收集 hidden，`VoxtralEncoderOutput`（L153-157）也没有该字段。两种做法：
   - 加 `output_hidden_states` 参数（干净，但要动 compat 层，注意 `padding_cache` 流式路径）；
   - 注册 forward hook 取第 4 层输出（侵入小，但流式与 CUDA Graph 下易出问题）。
     **建议走参数式**，并只在 `para_tap_layer >= 0` 时收集，避免常态显存开销。
2. 新增投影 `talker.para_proj = GatedBridge(1280 → 1024)`（Voxtral hidden 1280 → talker 1024）；
3. 帧率对齐：Voxtral L4 是 **50Hz**（adapter 前），talker 条件是译文 token 粒度。
   需要池化：按音频总长平均池化到固定长度 K（建议 K=16~32）作为**前缀**追加，
   或按注意力池化（可学，参数更多）。**建议先用固定 K 的平均池化**，最简且无新超参风险。
4. 融合：`TalkerFusion` 现在是二路（cond, text）。扩成三路有两种方式：
   - 串行：先 `fusion(cond, text)` 得 h，再 `fusion2(h, para)`；
   - 前缀：把 para 的 K 个位置拼在条件序列**最前面**，不改 fusion。
     **建议前缀式**，改动最小，且与 Omni 的"音频前缀"思路一致。
5. Voxtral 仍**全程冻结**，只训新增的 `para_proj`——注意 `_check_frozen_invariants`
   （`modeling_streaming_si.py` L184-199）会校验 encoder 冻结，新增参数不在其列，无需改校验。

**臂**：

| 臂 | 旁路 | 池化 K | 取层 |
|---|---|---|---|
| c0（对照，= a0） | 无 | — | -1 |
| c1 | Voxtral L4 前缀 | 16 | -1 |
| c2 | Voxtral L4 前缀 | 32 | -1 |

**预期**：信息上限最高（直取 R² 0.286 的表示）。

> **c 的预热是强制的，不是可选项。** `para_proj` 随机初始化，而
> `paralinguistic_preservation.md` L278 实测「随机桥接输出约为 text 路的 **40 倍**，
> CE 3.51 → 4.95」。把这样一个模块塞进已收敛的 S4 再以 lr 5e-5 跑 joint，等于
> 「拿已收敛的 talker 去适应噪声」（`train_tts.sh` L14 原话），thinker 与 talker 都会被带坏，
> 而 joint 很贵、坏了才发现代价高。**必须先冻结其它一切、只训 `para_proj`，过门 A 后才 joint。**

---

## 6. 执行顺序与资源

建议按**边际收益/成本比**排序，而不是按 a→b→c：

| 优先 | 实验 | 代码量 | 理由 |
|---|---|---|---|
| 1 | **a0（共用对照）+ a1**（取 L6） | 0（现成开关） | 零代码验证"实验 4 的无收益是否因只训桥接"，先拿这个结论；a0 同时是 b/c 的对照 |
| 2 | **b1/b2**（源音频位置） | ~40 行 | 直接消除已定位的瓶颈（Voxtral L4 R² 0.286 → thinker 输入 0.216 → 译文位置末层 **0.024**），最可能见效 |
| 3 | **c1**（Voxtral L4 旁路） | ~120 行 | 信息上限最高，但要改 compat 层 + 新模块强制预热 |

**a0 = b0 = c0**：三个实验共用同一个对照臂（现状通路，同起点同口径），所以总 run 数不是
3+4+3，而是 1 个对照 + 各实验的净增臂。

每个臂都必须**同口径**（同起点 `checkpoint-24980`、同样重初始化桥接、同 LR 调度、同种子、
同数据分片）——否则会重犯"拿不同口径的数相比"的错（本轮已踩两次：prompt 静默失效、
AutoPCP 参数错配）。

**资源与排期**（臂名后括注 `取层值 → 探针层`，避免与探针层号混淆）：

| 批次 | 臂 | 卡数 | 阶段 | 状态（2026-10-19 23:30） |
|---|---|---|---|---|
| 第 1 批（零代码） | a0 对照（`-1` → L32） | 16 | 预热 → joint | 预热**完成并过门 A**；joint **在跑**（`expa-a0j-rdb2`，p5en ×2） |
| | a1（`6` → L7） | 16 | 同上 | 预热**完成并过门 A**（优于 a0，见 §3.1）；joint 已提交（p5e ×2） |
| 第 2 批 | a2（`15` → L16） | 8 | 同上 | 预热**待起**：S3 CSI 挂载异常（下文） |
| | b1 / b2 | 各 16 | 同上 | 待 a 结论 + 代码 |
| 第 3 批 | b3 / c1 / c2 | 各 16 | c 需额外的 `para_proj` 单独预热 | — |

各臂机型可不同（a0 用 `ml.p5en.48xlarge`，a1/a2 用 `ml.p5e.48xlarge`），同为 8×H200、
仅 EFA 数不同（16 vs 32），**不影响数值可比性**；但 `per_device × 卡数 × GA` 必须相同。
a2 单节点 8 卡用 `GRADIENT_ACCUMULATION_STEPS=2` 补齐（预热全局 256 / joint 全局 512）。

**joint 时长**（基于历史 S4 joint `s3_s2st/v0-20261012-034209` 的 checkpoint 时间戳实测：
24 卡、全局 768、1.21 s/步 → **26.4 样本/秒/卡**）：1 epoch = 1928 万样本，
**16 卡约 12.7 小时，8 卡约 25.3 小时**。a2 单节点是关键路径。

### 6.1 集群运维坑（都实际踩过，提交前必查）

| 坑 | 症状 | 查法 |
|---|---|---|
| 机型混用 | worker 永久 Pending | 白名单节点的 `instance-type` 必须一致，且与 `efa_devices_per_worker` 匹配（p5e=32 / p5en=16） |
| **EFA 被他人占满** | GPU 空着却调度不上，报 `Insufficient vpc.amazonaws.com/efa` | 查 `vpc.amazonaws.com/efa` 实际占用，**不能只看 GPU** |
| **磁盘压力** | 节点带 `disk-pressure:NoSchedule`，或同步途中被驱逐 | P2 数据同步实占 **2.1 TB**，阈值 `nodefs.available<10%`；**可用空间 < 3 TB 的节点不要用** |
| 他人工作区占零散卡 | 整节点凑不满 8 卡 | 闲置 GPU 工作区是 Deployment，`scale --replicas=0` 可逆；但可能被对方改回 |
| **预热产物在本地盘** | 换节点提交 joint 时 `MODEL_PATH` 读不到 | 产物写在 master 节点的 hostPath `/opt/dlami/nvme`，**非共享存储**；跑完立刻 `aws s3 sync` 到 S3 |
| **`sync_code.sh` 不可单独调用** | 任务报 `No module named 'streaming_si.data'` 或 `error reading input file` | 它**先清空目标前缀再上传**，且 `CODE_PREFIX` 里的 `__OWNER__` 由 `submit.sh` 预渲染。单独调用会把代码传到字面 `__OWNER__/` 并清空真实前缀 —— **只能通过 `submit.sh` 间接触发** |
| S3 CSI 挂载时序 | Pod 久在 `ContainerCreating`，报 `Failed to find corresponding MountpointS3PodAttachment` | 附件与挂载 Pod 都在但 kubelet 侧找不到；删 Pod 重建即可 |

## 7. 待确认

- [ ] §1 全量 800 条的数据核验结果 + 人工评审（已在跑）
- [ ] 旧文档 ESim delta +0.02~0.03 与本次 +0.115 的口径差异来源
- [ ] S5 流式在实验 b/c 下的条件交错方案（本轮先只做到 S4）
- [ ] 留出样本集的构造（当前探针样本在 S4 训练集内）
