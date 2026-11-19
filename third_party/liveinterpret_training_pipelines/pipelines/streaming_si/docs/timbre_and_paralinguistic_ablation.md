# 音色与副语言：消融实验结论（2026-10-17 ～ 10-19）

> 回答一个问题：**S4 与 SimulS2ST-Omni 在 SIM-o / AutoPCP 上的差距，来自哪条通路，改哪里才有用？**
> 实验全部在 RealSI en→zh（346 句）上做，推理与打分同卡同脚本，译文 346/346 逐字不变
> （消融只改 flow 条件，不碰文本侧）。数值详见 FSX `/fsx/stevehguo/eval/abl/REPORT_timbre.md`。

## 0. 一句话结论

1. **音色（SIM-o）由三条通路共同决定**，且三条都不可被其他两条替代：
   ① CAM++ 说话人向量、② 源语音声学 prompt、③ **speech token 自身携带的残余说话人信息**。
   三条全关时 SIM-o delta 掉到 0.005（随机水平），任一条单独开启都能恢复到 0.08～0.13。
2. **talker 既影响副语言也影响音色。** 把 talker 产出的 token 换成源语音自身的 token，在任何
   条件组合下 SIM-o 都上升 0.17～0.26（§6）。所以**训练 talker 能同时改善 SIM-o 和 AutoPCP**，
   二者不是割裂的优化目标。
3. flow 推理参数（CFG 强度、ODE 步数、prompt 长度/位置）**已全部调到饱和**，推理侧没有免费的
   提升空间；换说话人编码器（ECAPA / WavLM）收益有限且必须连带重训 `spk_embed_affine_layer`。

> **勘误（2026-10-19）**：本文档初版第 0 节曾写「音色完全由 flow 条件决定，与 token 无关、
> 训练 talker 不会提升 SIM-o」。该结论源于只做了单向消融（固定 token 改条件）就外推，
> 已被反向实验推翻，详见 §6、§7。现结论以本版为准。

## 1. 链路结构与三条音色通路

```
talker ──speech tokens(6561, FSQ 语义码本, 说话人解耦不彻底)──► flow ──mel──► HiFT ──24k 波形
            │                                                    ▲
            └─③ token 残余说话人信息（隐式通路，见 §7）           │
                    ┌────────────────────────────────────────────┤
                    │ ① spk_emb：CAM++ 192 维向量 → spk_embed_affine_layer(80,192)
                    │ ② 声学 prompt：prompt_token 拼前缀 + prompt_feat 钉前 mel_len1 帧
                    │    （inpainting 式克隆，flow 输出时丢弃 prompt 段）
```

`source_prompt.py`（`custom/streaming_si/inference/`）实现通路 ②，设计动机对标 Omni 的
"源语音声学前缀续写"：只给 ① 时带到输出里的只有音色；② 额外带情感、发声方式、能量、录音质感。

通路 ③ 原以为不存在（FSQ 语义 token 的设计目标就是说话人解耦），§7 的实验证明解耦并不彻底。

## 2. 事故记录：prompt 通路曾整轮静默失效

**现象**：第一轮消融里 `p20`（prompt 20s）、`tail`（prompt 取结尾）与基线的 SIM-o 差值
精确为 0.0000。

**根因**：`run_timbre_ablation.sh` 漏了 `run_worker2.sh` 里的
`export COSYVOICE_SPEECH_TOKENIZER=...`；而 `build_source_prompt()` 找不到
`speech_tokenizer_v3.onnx` 时只打一条 WARN 就退回「只用说话人向量」。10 个臂全部
`source_prompt_tokens=0`，跑在无 prompt 状态。

**影响范围**：仅消融实验作废（留档 `/fsx/stevehguo/eval/abl_noprompt_bug/`）。
第二轮正式评测（`out2/`）worker 有该 export，日志确认 prompt 生效，**已发布报告不受影响**。
其中无 prompt 的 base/nospk 臂被复用为受控对照（`noprompt` / `nospk_noprompt`）。

**修复**（均已入库）：
- `build_source_prompt` 找不到 tokenizer 时 **SystemExit 硬失败**；确认不要 prompt 请显式
  `--no_source_prompt`；
- 消融脚本补 tokenizer export，且结束断言 `source_prompt_tokens > 0`（显式关闭的臂除外）。

## 3. 修复后的消融结果（prompt 全部生效）

| 臂 | 说明 | SIM-o | SIM-o delta | UTMOS | AutoPCP delta |
|---|---|---|---|---|---|
| base | 基线（ode5 / prompt 10s head / cfg 0.7） | 0.813 | 0.161 | 3.274 | 1.187 |
| noprompt | 只关 prompt ② | 0.775 | 0.142 | 3.267 | 0.982 |
| nospk | 只关 CAM++ ① | 0.715 | 0.084 | 3.429 | 0.968 |
| nospk_noprompt | 两路都关 | 0.487 | 0.005 | 3.464 | 0.635 |
| p20 | prompt 20s | 0.814 | 0.162 | 3.280 | 1.191 |
| tail | prompt 取结尾 | 0.813 | 0.162 | 3.278 | 1.188 |
| cfg10 / 15 / 20 | flow CFG 0.7→1.0/1.5/2.0 | 0.815/0.814/0.812 | ±0.002 内 | 单调劣化（2.0 时 −0.75） | 持平或降 |
| cfg15_ode10 | CFG 1.5 + ODE 10 | 0.819 | 0.163 | −0.246 | +0.035 |

**读法**（本节所有臂都用 talker 产出的 token，只改 flow 条件）：
- 两路都关时 SIM-o delta 崩到 0.005（≈随机水平）；两路各自单独关，另一路都能补回一部分。
  **①与②贡献相当，不可互相替代**（①对 SIM-o 略强，②对 AutoPCP 略强）。
  注意 0.487 这个下限是 **talker token** 下的下限，换成源自身 token 同样全关可达 0.748（§7）。
- prompt 长度（10s→20s）与位置（head→tail）**已饱和**，变化在 ±0.002 噪声内。
- CFG 调大只伤音质（UTMOS 单调下降）不涨音色，**保持 yaml 默认 0.7**。Omni VoiceBox 的
  cfg_scale=2.0 不可类比：两者 CFG 公式与条件构造不同。
- ODE 5→10 无收益。

## 4. 复制合成上界（关键定量）

`resynth_upper_bound.py`：把源音频编成**它自己的** speech token 过 flow/HiFT 重建，
完全排除 talker，量声码器通路的音色天花板（同一套 CAM++ / tokenizer / flow / ODE 5）。

| 臂 | 条件 | SIM-o | AutoPCP delta | UTMOS |
|---|---|---|---|---|
| rs_spk | 源自身 token + CAM++ | 0.963 | 1.817 | 3.473 |
| rs_prompt | 源自身 token + 仅 prompt | 0.971 | 2.182 | 3.263 |
| rs_both | 源自身 token + 两路全开 | 0.979 | 2.255 | 3.229 |
| （实际 S4-offline） | talker token + 两路全开 | 0.813 | 1.187 | 3.274 |

**判读**：
- 0.963～0.979 说明声码器链路本身（flow + HiFT + CAM++）有足够的音色还原能力，**不是瓶颈**。
- **但这个值不能当作 S2ST 的"音色天花板"**：`rs_*` 是把源音频重建回它自己，内容、语言、
  时长、语速完全一致；S4 输出是另一种语言。而 WavLM x-vector 对音素分布与时长并非完全不敏感，
  所以 0.979 准确的叫法是「**同内容同语言重建**的上界」。
- 曾猜测 0.166 的差距主要是"时长失配造成的指标伪影"，**已检验、不成立**：
  corr(SIM-o, |log 时长比|) 在 RealSI en→zh 上是 **+0.147**（方向相反），ACL 上 −0.006，
  只有 zh→en 上显著（S4 −0.436、Omni-L4 −0.257）。而 0.813 vs 0.979 正是在 en→zh 上测的，
  所以时长伪影解释不了这个差距。内容/语言差异的影响尚未单独定量（需要同说话人、
  同语言、不同内容的对照，目前缺数据）。
- AutoPCP delta 从 2.255 掉到 1.187，是 talker 影响副语言的直接体现。

## 5. 单向消融不能外推：一次推理错误的记录

`base` 与 `nospk` 两臂的 `speech_tokens` **346/346 逐 token 完全相同**，仅 flow 条件不同，
SIM-o 从 0.813 掉到 0.715（无 prompt 版本里 0.775→0.487）。

当时据此得出"音色完全由 flow 条件决定、与 token 无关"，**这是错的**。该实验固定 token、
只改条件，能证明的只有「条件是有效因果通路」；对「token 是否也是通路」它没有任何证据，
因为 token 从未作为自变量变化过。正确的做法是再做反方向实验（固定条件、改 token 来源），
结果见 §6、§7，与上述外推相反。

教训：单向消融只支持"所改变的因素有影响"，不支持"未改变的因素无影响"。

## 6. 反方向实验：固定 flow 条件，只换 token 来源

把 §3（talker token）与 §4（源自身 token）按**相同 flow 条件**配对：

| flow 条件 | talker token | 源自身 token | token 带来的差 |
|---|---|---|---|
| 仅 CAM++（无 prompt） | `noprompt` 0.775 | `rs_spk` 0.963 | **+0.188** |
| 仅 prompt（无 CAM++） | `nospk` 0.715 | `rs_prompt` 0.971 | **+0.256** |
| 两路全开 | `base` 0.813 | `rs_both` 0.979 | **+0.166** |

三种条件设置下，换 token 来源一律带来 +0.17～0.26，**与条件通路同一量级甚至更大**。
token 确实是一条音色通路。

## 7. 判别实验：token 携带说话人信息（通路 ③ 成立）

§6 的差距有两种可能机制，需要区分：
1. **token 本身携带残余说话人信息**（解耦不彻底）→ 训练 talker 能直接改善 SIM-o；
2. **token 的自然度调制条件的实现强度**（talker token 偏离真实语音统计分布，flow 的 `mu`
   出现分布外，decoder 向"平均音色"回退）→ 训练改善的是 token 自然度，SIM-o 只是副产物。

判别方法：**源自身 token + 两路条件全关**。若 token 不含说话人信息，SIM-o 应与
`nospk_noprompt`（talker token + 全关）相当。

| 臂 | token 来源 | flow 条件 | SIM-o | SIM-o delta | AutoPCP delta | UTMOS |
|---|---|---|---|---|---|---|
| `nospk_noprompt` | talker | 全关 | 0.487 | 0.005 | 0.635 | 3.464 |
| `rs_nospk_noprompt` | **源自身** | 全关 | **0.748** | **0.130** | 1.415 | 3.531 |

**结论：机制 1 成立。** flow 完全拿不到任何显式说话人信息时，仅靠 token 本身，
SIM-o 仍达 0.748 / delta 0.130——已超过 `nospk`（0.715，有 prompt 无 CAM++），
接近 `noprompt`（0.775，有 CAM++ 无 prompt）。CosyVoice3 的 6561 单码本 FSQ 语义 token
**并未彻底解耦说话人**，其音色承载能力与两条显式通路相当。

三条通路的完整图景（RealSI en→zh）：

| 开启的通路 | SIM-o | SIM-o delta |
|---|---|---|
| 无（talker token，两路全关） | 0.487 | 0.005 |
| 仅 ③ token | 0.748 | 0.130 |
| 仅 ① CAM++ | 0.775 | 0.142 |
| 仅 ② prompt | 0.715 | 0.084 |
| ① + ② | 0.813 | 0.161 |
| ③ + ① | 0.963 | 0.282 |
| ③ + ② | 0.971 | 0.269 |
| ① + ② + ③ | 0.979 | 0.293 |

读出两点：
- talker token 不只是"不带音色"，相对源自身 token 它在**主动稀释**音色（0.748 → 0.487 的反向差）。
- 强互补性：单开一条在 0.72～0.78 徘徊；③ 加任一显式通路即跃到 0.96+；而 ①+②（无 ③）
  只有 0.813。**③ 不可被显式条件替代。**

## 8. 换说话人编码器的评估（结论：不做）

- `spk_embed_affine_layer` 是 **(80, 192) 的已训练线性层**，拟合的是 CAM++ 嵌入空间。
  ECAPA-TDNN 同为 192 维只是巧合，空间基底完全不同，直接换是分布错配，预期退化；
  真换必须连带重训该层（乃至 flow），而 §4 已证明声码器链路不是瓶颈。
- `wavlm-base-plus-sv` 正是 SIM-o 的打分模型，拿它当编码器会自证循环，指标虚高，不可用于决策。

## 9. 后续方向（按优先级）

1. **训练 talker 是唯一同时推动音色与副语言的杠杆。** §7 证明 token 既携带说话人信息、
   也决定韵律，所以提升 token 质量（更接近真实语音统计分布）可同时改善 SIM-o 与 AutoPCP。
   两个指标一起监控即可，不必取舍。
2. **AutoPCP delta 仍是更灵敏的 KPI**（1.187，离同内容上界 2.255 差 1.07；SIM-o 的相对空间较小），
   手段：S4/S5 数据与训练目标里强化韵律/停顿/重音条件。
3. 推理侧已无空间（§3：CFG / ODE / prompt 长度位置全部饱和）。
4. 若要进一步定量「跨语种内容差异对 SIM-o 的固有惩罚」，需要构造同说话人、同语言、
   不同内容的对照（如同一 talk 内两个 segment 互算 SIM-o），当前数据未覆盖。
5. S5 空译文（极短句全程 WAIT）与 talker 超 StaticCache 失败，是流式策略问题，与音色无关，
   见 `README.md` 观察与问题表。

## 10. 产物位置

| 内容 | 路径 |
|---|---|
| 汇总表（§3 / §4 / §7 各表的来源） | FSX `/fsx/stevehguo/eval/abl/REPORT_timbre.md` |
| 无 prompt 事故数据（留档，见 §2） | FSX `/fsx/stevehguo/eval/abl_noprompt_bug/` |
| 消融脚本 / 打分脚本 / 汇总脚本 | `custom/streaming_si/eval/{run,score,report}_timbre_ablation*` |
| 复制合成上界与判别臂脚本 | `custom/streaming_si/eval/resynth_upper_bound.py`（`--no_speaker` / 不传 `--source_prompt` 即 §7 的全关臂） |
| prompt 通路实现与设计动机 | `custom/streaming_si/inference/source_prompt.py`、本文档同目录 `paralinguistic_preservation.md` 第 7 节 |

## 附：AutoPCP 数值怎么读

AutoPCP 是 comparator **未过激活的 logit**（`odim=5` / `output_act=False`，取第 0 维，
官方 `compare_audio_pairs` 无任何刻度映射；`_pool` 里的 softmax 只是注意力池化权重）。
**不是 1～4 / 1～5 的量表分**，值域无界、随机基线可为负（如 −0.25、−0.36）。
只能同列相对比较、对照随机基线看 delta，不可按「满分 4 分」解读，也不可跨数据集比。
