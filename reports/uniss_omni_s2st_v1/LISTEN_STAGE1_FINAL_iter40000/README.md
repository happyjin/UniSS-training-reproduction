# Stage 1 最终试听 — iter 40,000(级联 S2ST)

链路:源语音 → Omni Thinker 翻译 → Talker(+源说话人 32 个音色 token)→ BiCodec 声码器

`_源语音.wav` 是输入,`_翻译语音.wav` 是输出。音色取自源说话人,译文保留原嗓音。

| # | 方向 | 参考译文 | 模型译文 | 输出语音的 ASR |
|---|---|---|---|---|
| 01 | en→zh | 姜涛曾就读轩尼诗道官立下午小学、邓肇坚维多利亚官立中学和青年学院 | 金钟曾就读于Hennassy Road政府小学、Teng Shu | 金钟曾就读于Hennessy Road的政府小学Tanchuki |
| 02 | zh→en | keung to attended hennessy road  | Zhang Tao studied in the Xuanli  | Zhang Tao studied in the Shunli  |
| 03 | en→zh | 在工作了九年后，伯爵不幸去世。 | 九年后，伯爵去世了。 | 九年后伯爵去世了 |
| 04 | zh→en | after nine years of work the ear | After nine years of work, the Ea | After nine years of work, the Ea |
| 05 | en→zh | 整个系统称为键接合。 | 完整的系统被称为共轭。 | 完整的系统被称为共乐 |
| 06 | zh→en | the complete system is called co | The whole system is called the i | The whole system is called the i |
| 07 | en→zh | 一些听众跟随至办事处，与审讯人员进行说理斗争控诉。 | 一些观众跟随到办公室与审问者争论。 | 一些觀眾跟隨到辦公室與審問者爭論 |
| 08 | zh→en | some of the audience followed to | Some listeners followed the duty | Some listeners followed the duty |
| 09 | en→zh | 一些附近的居民爬上较高的树木而逃过了海啸。 | 附近的一些居民爬上了高大的树木，从海啸中逃生。 | 附近的一些居民爬上了高大的树木从海啸中逃生 |
| 10 | zh→en | some nearby residents climbed on | Some nearby residents climbed up | Some nearby residents climbed up |
| 11 | en→zh | 里舍·巴塔尼是一位印度演员，常出演宝莱坞电影。 | Rishi Butani是一位印度演员，经常出现在宝莱坞电影中。 | Rishi Bhattani是一位印度演员,经常出现在保莱坞电影 |
| 12 | zh→en | rishi bhutani is an indian actor | Rishabh Taneja is an Indian acto | Vishal Tanisha is an Indian acto |
| 13 | en→zh | 后被选中为庶吉士，散馆授编修。 | 后来，他被选为侍从，被任命为《 Sangwan》的编辑。 | 后来他被选为侍从,被任命为Sun One的编辑 |
| 14 | zh→en | later he was selected as a shuji | Later, he was selected as the go | Later, he was selected as the go |
| 15 | en→zh | 许多法律制度对大多数或所有的案件，或者是某些特定类型的案件会使用 | 许多法律体系都采用法官处理大多数或所有案件，或者处理某些类型的案 | 许多法律体系都采用法官处理大多数或所有案件,或者处理某些类型的案 |
| 16 | zh→en | many legal systems adopt judges  | Many legal systems use judges to | Many legal systems use judges to |

## 120 条上的指标

| 方向 | n | 文本 BLEU(Thinker,冻结) | 语音 ASR-BLEU(端到端) | Talker 的代价 |
|---|---|---|---|---|
| zh→en | 60 | 36.36 | **34.43** | −1.93 |
| en→zh | 60 | 38.16 | **34.30** | −3.86 |

## 训练过程中的改善

| 检查点 | zh→en | en→zh |
|---|---|---|
| iter 12,000 | 29.69 | 33.50 |
| iter 24,000 | 33.78 | 33.92 |
| **iter 40,000** | **34.43** | **34.30** |

增益在收窄(zh→en +4.09 → +0.65),Stage 1 接近饱和。

文本 BLEU 全程不变(Thinker 冻结),所以全部改善都来自 Talker。
