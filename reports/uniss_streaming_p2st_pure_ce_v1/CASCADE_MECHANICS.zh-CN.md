# ⑤a 级联机械验收 —— 外部切换规则跑通了

**判定 `pass`,三项判据 2/2。这一步不测质量。**

运行的 checkpoint(B′ `iter_0001132`)**从未在隔离的 prefix-to-prefix 序列上训练过**,
所以译文会很差 —— 那是预期,不是判据,和 ④ 里 loss 数值不是判据一样。

## 三项判据

| 判据 | 结果 |
|---|---|
| 会话实际跑的任务序列 == `rule_trace` 从它观测到的 delta 复现出的序列 | **2/2** |
| 生成的任何输出里都不含 `WAIT_READ` / `WRITE_GENERATE` | **2/2** |
| 每个片段都被放在"证明它的那段音频"之后,不早于 | **2/2** |

**第一条是核心:任务顺序来自规则,不来自模型发出的任何 token。** 它不是断言,
是把会话自己的 trace 和规则的输出逐项比对出来的。

`switch_rule.next_task` 另有 11 项穷举单测(纯整数,不需要模型),
覆盖了 bounded wait、级联顺序不可违反、每块每阶段最多跑一次,
以及"决策 token 在规则的词汇表里根本不可表示"。

## bounded wait 在真实运行里可见

```
NCSSD_R_EN_0000000083  eng->cmn  12 块 -> {asr: 12, mt: 2, tts: 1}
NCSSD_R_EN_0000000261  eng->cmn  12 块 -> {asr: 12, mt: 3, tts: 0}
```

**ASR 每块都跑,MT 只跑 2–3 次,TTS 0–1 次。** 这正是规则的形状:
转写没长就回去听,译文没长就不开口 —— 而"不开口"这个分支以前是模型要做的决策。

第二条样本 **12 块内一次都没开口**,`first_audible = None`。在旧的交织 runtime 里
这只能通过 δ 偏置去撬;这里它是规则的一个分支,而且是可解释的:
提交器没放出新的目标文本。

## 两个具体的、可行动的失败(⑥ 要修的东西)

**① MT 把源文抄了下来而不是翻译。** `src: "I think a"` → `tgt: "I think a"`,
而方向是 eng→cmn,目标应该是中文。隔离形式的 MT prompt
(`[TASK_S2T_TRANSLATION, STREAMING_MODE, lang, START_CONTENT 源 END_CONTENT START_CONTENT 目标]`)
对这个 checkpoint 是全新的结构。

**② TTS 不终止。** `sem = 384` 正好撞上 `max_semantic_tokens`,
`terminator_rate = 0.93`(15 个阶段里 14 个正常终止,没终止的那个就是 TTS)。
**隔离形式下的 `END_SEMANTIC` 是未训练的** —— 这就是 `natural_eos` 问题,
但现在它被隔离到了一个干净的单任务里,而不是和 WAIT/WRITE 挤在一个桶里。

## 一个我发现的、应该在 ⑥ 之前修的不一致

ASR 那一路输出了中文(`"石雕了现在"`)去转写英文音频。原因很可能是 prompt 头不匹配:

```
Stage-A 训练用的头:  [TASK_STREAMING_ASR, STREAMING_MODE, lang, *wrap_global_tokens(speaker)]
C 现在的头:          [TASK_STREAMING_ASR, STREAMING_MODE, lang, START_GLM, ...]
```

**C 少了 speaker 块。** `TASK_STREAMING_ASR` 之所以是三个 task token 里唯一被训练过的,
就是因为 Stage-A 用了它 —— 而 C 没有沿用它的完整头部结构,等于把那份训练白放掉了一部分。

这和 task token 那次是同一类问题(训练格式没有对齐它本该继承的那个格式),
修法一样便宜,但要再走一轮 ①→②→④。
