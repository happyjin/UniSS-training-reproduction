# 翻译质量问题的根因:eng→cmn 的语序重排失败,不是 ASR

## 1. 逐样本分解

| 样本 | 方向 | ASR 错误率 | ASR chrF | **译文 BLEU** | **译文 chrF** | 生成/源 |
|---|---|---:|---:|---:|---:|---:|
| `emilia_zh_0003927983` | cmn→eng | 0.032 | 76.6 | 15.31 | **47.8** | 1.07 |
| `emilia_zh_0003929091` | cmn→eng | 0.065 | 81.2 | 10.79 | **47.4** | 0.96 |
| `emilia_zh_0003980703` | cmn→eng | 0.085 | 77.7 | 5.37 | **43.9** | 0.68 |
| **`LibriSpeech_0000174311`** | **eng→cmn** | **0.093** | **84.5** | 4.19 | **9.8** | 0.85 |
| `emilia_zh_0003980934` | cmn→eng | 0.105 | 73.4 | 44.32 | **74.2** | 1.00 |
| `LibriSpeech_0000079616` | eng→cmn | 0.211 | 78.3 | 18.06 | 19.6 | 0.60 |
| `LibriSpeech_0000213211` | eng→cmn | 0.391 | 75.1 | 8.70 | 15.9 | 0.69 |
| `LibriSpeech_0000104521` | eng→cmn | 0.425 | 76.3 | 9.97 | 14.8 | 0.82 |

| 方向 | ASR 错误率 | ASR chrF | 译文 BLEU | **译文 chrF** |
|---|---:|---:|---:|---:|
| cmn→eng | 0.072 | 77.2 | 18.95 | **53.3** |
| eng→cmn | 0.280 | **78.6** | 10.23 | **15.0** |

**关键异常:两个方向的 ASR chrF 几乎相同(77.2 vs 78.6),译文 chrF 却差 3.6 倍。**
而 `LibriSpeech_0000174311` 的 ASR 错误率只有 **0.093**(全部里第 4 好),
译文 chrF 却是 **9.8**(全部里最差)。

**所以 eng→cmn 的翻译问题不是 ASR 传播造成的。**

## 2. 真正的根因:逐词直译,没有语序重排

`LibriSpeech_0000174311`,ASR 基本正确:

| | |
|---|---|
| 源 gold | `But they do not engross the whole discourse so to themselves during their meals that the younger may not put in for a share On the contrary they engag[ed]…` |
| 模型 ASR | `But they do not grow so whole discourse so to themselves during the their meals that they younger may not put in for a share on the contrary thing gag` |
| 参考译文 | 但他们并不会在用餐时完全占据谈话,不让年轻人有机会加入。相反,他们会邀请年轻人参与交谈… |
| **模型译文** | 但他们确实没有那么发展到这种程度以至于在他们的餐食中他们年轻时可能不会把进去吃一顿分享**在相反的事情上订婚了**他们去谈话那所以他们可能会做出那样的免费方式的对话… |

逐处对照:

| 英文片段 | 参考译法 | 模型译法 | 问题 |
|---|---|---|---|
| `during their meals` | 在用餐时 | 在他们的餐食中 | 直译 |
| `the younger may not put in for a share` | 不让年轻人有机会加入 | 他们年轻时可能不会把进去吃一顿分享 | **逐词直译,语义碎裂** |
| `on the contrary they engaged` | 相反,他们会邀请 | **在相反的事情上订婚了** | **`engage` 取错义项** |

**模型在每个 160ms 块上翻译新到的几个词,而不重排。** 英译中需要大量语序调整
(SVO → 主题-述题、关系从句前置),逐块直译必然产出这种词串。

字数也印证:参考 63 字,模型 **100 字**(+59%)—— 直译比意译长得多。

而 cmn→eng 方向,中文的语序与英文相对接近,逐块翻译可用(chrF 43.9–74.2)。

## 3. 这正是 CSSEL-P2P 论文点名的问题

> "SimulST requires incremental translation under strict latency constraints,
> yet remains challenging for decoder-only LLM systems due to limited context
> and **cross-lingual reordering**."

它的解法就是**用 teacher 标注的 prefix-to-prefix 目标 + bounded waiting** ——
让数据告诉模型"这个源前缀该吐多少、该怎么组织",而不是逐块直译。

**而本项目的 `incremental_mt_event` 家族已经是这个形态**
(`_mt_request` 的 target 就是 `event.target_text_delta`,support 滞后 −160ms),
**只是在旧目标函数里被 boundary 权重 0.10 饿着,而且这次训练也没有针对它加强。**

## 4. 三个独立问题,不要混为一谈

| 问题 | 表现 | 根因 | 解法 |
|---|---|---|---|
| **① eng→cmn 翻译差** | chrF 15.0 vs 53.3 | **逐词直译不重排** | 训练:强化 prefix-to-prefix 家族(CSSEL-P2P 路线) |
| ② 英文 ASR 差 | 错误率 0.280 vs 0.072 | Stage-A 英文侧弱 | 训练:S3 英文流式 ASR |
| **③ 卡顿割裂** | 生成/源 0.68–0.85 | **重复惩罚压掉了合法持续音** | 推理:窗口 64→8;根治要训练片段长度监督 |

**①和③互相独立**:`LibriSpeech_0000174311` 生成/源 0.85(卡顿轻)但 chrF 9.8
(翻译最差);`emilia_zh_0003980703` 生成/源 0.68(卡顿重)但 chrF 43.9(翻译好)。

## 5. 优先级

**① 是最大阻塞。** 英文 ASR 从 0.396 已经改善到 0.280,但 eng→cmn 的 chrF 仍是
15.0 —— 就算 ASR 做到完美,逐词直译也产不出可用译文。

推理侧对 ① **无能为力** —— 它是"怎么翻"的问题,不是"何时翻"的问题。
