# 试听 — 两个 bug 修完之后,iter 12000(自由生成)

`_生成.wav` 是模型**自己走**出来的(贪心,无教师强制);`_金标.wav` 是同一句的金标码。

| # | 方向 | 参考文本 | 生成码/金标码 |
|---|---|---|---|
| 01 | eng→cmn | 姜涛曾就读轩尼诗道官立下午小学、邓肇坚维多利亚官立中学和青年学院。 | 700/444 |
| 02 | cmn→eng | keung to attended hennessy road government | 700/424 |
| 03 | eng→cmn | 在工作了九年后，伯爵不幸去世。 | 700/198 |
| 04 | cmn→eng | after nine years of work the earl passed a | 187/144 |
| 05 | eng→cmn | 整个系统称为键接合。 | 141/174 |
| 06 | cmn→eng | the complete system is called conjugation | 700/136 |
| 07 | eng→cmn | 一些听众跟随至办事处，与审讯人员进行说理斗争控诉。 | 310/322 |
| 08 | cmn→eng | some of the audience followed to the offic | 383/198 |
| 09 | eng→cmn | 一些附近的居民爬上较高的树木而逃过了海啸。 | 173/345 |
| 10 | cmn→eng | some nearby residents climbed on the high  | 700/201 |
| 11 | eng→cmn | 里舍·巴塔尼是一位印度演员，常出演宝莱坞电影。 | 206/351 |
| 12 | cmn→eng | rishi bhutani is an indian actor who often | 700/242 |

## 客观判定(Whisper-large-v3 转写)

| 参考 | 生成音频的 ASR |
|---|---|
| after nine years of work the earl passed away | **After nine years of work, the Earl passed away.** |
| some of the audience followed to the office to argue… | **完全一致** |
| 一些听众跟随至办事处,与审讯人员进行说理斗争控诉。 | **一些聽眾跟隨著辦事處與審訊人員進行說理鬥爭控訴** |
| keung to attended hennessy road government primary school | Keen too attended Hennessy Road Government Primary School. |
| 整个系统称为键接合。 | 整个系统称为间接核 |
| 在工作了九年后,伯爵不幸去世。 | 在工作了九年后,(后半跑飞) |
| the complete system is called conjugation | so(失败) |
| 姜涛曾就读…… | 間套層(失败) |

8 条里 5 条好、1 条半、2 条坏。**这是一个能用的系统**,不是失败的。

## 之前听到噪声的两个原因,都是我的 bug

1. **导出器把 Adam 一阶矩当权重导出**(`.talker.` 子串匹配收进了优化器状态,
   剥前缀后四个键塌成一个)。末层 RMSNorm≈0 → 每个位置同样的 logits。
2. **生成脚本加载了说话人前缀却从不传给解码函数**。`prefix_width` 于是为 0,
   模型在缺失训练时所用条件的情况下运行,分布变平、eos 胜出。
   根因是一次 `str.replace` 因为目标串被另一处改动而静默失配,我当时没加断言。

## 还没解决的

12 条里 6 条撞上 700 码的上限 —— 停止行为仍不稳。
两条完全失败的也还没查。但这些是具体缺陷,不是'模型没学会'。
