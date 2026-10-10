# 试听 — 翻译后的音频(级联 S2ST)

链路:**源语音 → Omni Thinker 翻译 → Talker 用源说话人的 32 个音色 token 说出来 → BiCodec 声码器**

每条两份:`_源语音.wav` 是输入,`_翻译语音.wav` 是输出。
音色 token 取自**源说话人**,所以译文应保留原说话人的嗓音。

| # | 方向 | 参考译文 | 模型译文 | 输出语音的 ASR |
|---|---|---|---|---|
| 01 | en→zh | 姜涛曾就读轩尼诗道官立下午小学、邓肇坚维多利亚官立中学和青年学院。 | 金钟曾就读于Hennassy Road政府小学、Teng Shu K | 金钟曾就读于Hanasi Road政府小学Tanshu King维多 |
| 02 | zh→en | keung to attended hennessy road go | Zhang Tao studied in the Xuanli Ci | John Tao studied in the Shenley Ci |
| 03 | en→zh | 在工作了九年后，伯爵不幸去世。 | 九年后，伯爵去世了。 | 九年后,伯爵去世了。 |
| 04 | zh→en | after nine years of work the earl  | After nine years of work, the Earl | After nine years of work, the earl |
| 05 | en→zh | 整个系统称为键接合。 | 完整的系统被称为共轭。 | 完整的系统被称为共恶 |
| 06 | zh→en | the complete system is called conj | The whole system is called the ind | The whole system is called the ind |
| 07 | en→zh | 一些听众跟随至办事处，与审讯人员进行说理斗争控诉。 | 一些观众跟随到办公室与审问者争论。 | 一些觀眾跟隨到辦公室與審問者爭論 |
| 08 | zh→en | some of the audience followed to t | Some listeners followed the duty r | Some listeners followed the duty r |
| 09 | en→zh | 一些附近的居民爬上较高的树木而逃过了海啸。 | 附近的一些居民爬上了高大的树木，从海啸中逃生。 | 附近的一些居民爬上了高大的树木,从海啸中逃生。 |
| 10 | zh→en | some nearby residents climbed on t | Some nearby residents climbed up t | So nearby residents climbed up a t |
| 11 | en→zh | 里舍·巴塔尼是一位印度演员，常出演宝莱坞电影。 | Rishi Butani是一位印度演员，经常出现在宝莱坞电影中。 | Rishi Bhattani是一位印度演员,经常出现在保莱坞电影中。 |
| 12 | zh→en | rishi bhutani is an indian actor w | Rishabh Taneja is an Indian actor  | Rishabh Tanjai is an Indian actor  |
| 13 | en→zh | 后被选中为庶吉士，散馆授编修。 | 后来，他被选为侍从，被任命为《 Sangwan》的编辑。 | 后来他被选为侍从,被任命为三管的编辑。 |
| 14 | zh→en | later he was selected as a shujish | Later, he was selected as the gove | Later, he was selected as the gove |
| 15 | en→zh | 许多法律制度对大多数或所有的案件，或者是某些特定类型的案件会使用法官 | 许多法律体系都采用法官处理大多数或所有案件，或者处理某些类型的案件。 | 许多法律体系都采用法官处理大多数或所有案件,或者处理某些类型的案件 |
| 16 | zh→en | many legal systems adopt judges fo | Many legal systems use judges to t | Many legal systems use judges to t |

## 120 条上的客观指标

| 方向 | n | 文本 BLEU(Thinker) | 语音 ASR-BLEU(端到端) |
|---|---|---|---|
| zh→en | 60 | 36.36 | **29.69** |
| en→zh | 60 | 38.16 | **33.50** |

对照本项目现有 Phase3 系统在 CVSS-T **test** 上的 ASR-BLEU:**23.58 (en→zh) / 12.05 (zh→en)**。
注意测试集不同(这里是 dev 子集 60 条/方向),且这是**级联、非流式**,不能直接当作同条件对比。

文本 BLEU 与语音 ASR-BLEU 之间的差(6.7 / 4.7)就是 Talker 这一段引入的损失。
