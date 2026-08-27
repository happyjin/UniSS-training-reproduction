# Phase A 修复与 long-episode RL：训练长样本和外部长音频联合评估

## 1. 评估问题与声明边界

本报告回答两个不同问题。第一组是正式 RL rollout 真正见过的 8 条长训练 episode，用于判断 Runtime v2 修复以及 RL iter15/30/45 是否学会训练目标；这组结果是 **train-seen/in-domain**，不能用于宣称泛化。第二组是之前反复试听的 4 条 Wikimedia 外部长音频，用于继续观察模型在非训练音频上的结构表现；因为它们没有与当前协议匹配的人工 reference，不能伪造 BLEU、chrF、WER 或 CER。两组结果始终分表讨论。

固定推理条件：640 ms decision chunk、160 ms 物理声学 block、24 s acoustic ring、相同 Phase A speaker token、相同 Runtime v2。所有样本均输出连续译音、全局时间轴和左源右译立体声。

8 条训练 episode 是由 6–13 个真实 15-shard 训练短句以约 160 ms 间隔拼接而成，时长约 70.8–79.0 秒。协议同时审计 episode 音频哈希和 component sample ID，确认不与 validation 重叠。

## 2. Train-seen 长 episode：正式 reference 指标

### 2.1 ASR 与 MT 内容质量

| 系统 | 中→英中文 CER↓ | 英→中英文 WER↓ | 中→英 BLEU/chrF↑ | 英→中 BLEU/chrF↑ | LCS 文本覆盖↑ | 4-gram 重复率↓ |
|---|---:|---:|---:|---:|---:|---:|
| Phase A iter381 + Runtime v2（修复阶段） | 0.763 | 0.689 | 2.36/21.38 | 9.55/17.19 | 0.196 | 0.004 |
| RL iter15 + Runtime v2 | 0.763 | 0.689 | 2.41/21.51 | 7.32/15.34 | 0.184 | 0.007 |
| RL iter30 + Runtime v2 | 0.763 | 0.689 | 2.38/21.52 | 9.33/17.32 | 0.193 | 0.003 |
| RL iter45 + Runtime v2 | 0.763 | 0.689 | 2.53/21.72 | 8.80/16.60 | 0.195 | 0.003 |

CER/WER 衡量流式 ASR 编辑错误；BLEU/chrF 衡量最终增量译文与 teacher translation 的匹配；LCS 文本覆盖是 hypothesis 与 reference 的最长公共子序列召回，只用于判断漏译趋势，不等同于语义指标；4-gram 重复率用于暴露循环扩写。

### 2.2 WRITE、TTS、覆盖与运行效率

| 系统 | 首次发声 p50/p95 ms↓ | WRITE gap p95/max ms↓ | 最大内部静音 mean/max ms↓ | 译音/源音时长比 | WRITE 总数 | pending/TTS失败 | pre-final率 | WAV健康 | RTF↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase A iter381 + Runtime v2（修复阶段） | 4480/12512 | 24064/44640 | 26875/40400 | 0.395 | 86 | 0/0 | 1.000 | 1.000 | 5.717 |
| RL iter15 + Runtime v2 | 4480/12512 | 29750/44640 | 28038/40100 | 0.377 | 84 | 0/0 | 1.000 | 1.000 | 5.261 |
| RL iter30 + Runtime v2 | 4480/12512 | 22400/44640 | 23575/40400 | 0.396 | 89 | 0/0 | 1.000 | 1.000 | 5.269 |
| RL iter45 + Runtime v2 | 4480/12512 | 23936/44640 | 26338/41100 | 0.405 | 87 | 0/0 | 1.000 | 1.000 | 5.300 |

### 2.3 逐训练样本试听路径

#### episode_000006_cmn_eng（cmn→eng，76.04s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000006_cmn_eng.wav`
- reference transcription：为什么这么说呢？比方说你用treasure,那你用treasure,你至少你把你的私钥脱离了你的电脑和你的手机。带大家去看一种生活的可能性，而不是让大家来喜欢我。马宁远这才犹犹豫豫的将那只装着山参的红木盒拿到胸前。你还没听到我的第三条秘诀呢，嗯那是什么呢？爷爷，我的第三条秘诀就是。每个人都是人类相似性和差异性的独特结合，没有两个人能够拥有同样的指纹。然而，我们又如此相似。如果你用普通来形容的话，我觉得是ok我误会了吗。这个笑话用了一个双关的手法，就是用生理上的心脏位置偏移双关。父母偏心，偏爱众多孩子中的一个。呃，我在想有可能对我们的这位题主来说，想象已经成为他习惯性的对现实生活的一种调剂。所以他习惯性的想到了孩子的未来会是怎样啊，假若我接近了生命的终点的时候，我要做怎样怎样的事情。因为我想他想象的是非常的有细节感。
- reference translation：Why is that? For example, if you use treasure, then by using treasure, you at least take your private key away from your computer and your phone. To show everyone a possibility of life, rather than to make everyone like me. Ma Ningyuan hesitantly brought the redwood box containing the ginseng to his chest. You haven't heard my third secret yet, what is it? Grandpa, my third secret is. Everyone is a unique combination of human similarities and differences; no two people can have the same fingerprints. Yet, we are so similar. If you describe it as ordinary, I think it's okay. Did I misunderstand. This joke uses a pun, referring to the physical displacement of the heart. Parents being partial, favoring one among many children. Well, I think it's possible that for this questioner, imagination has become a habitual way of seasoning real life. So, he habitually thinks about how his child's future will be, what he would do if he were approaching the end of his life. Because I think his imagination is very detailed.
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.796，chrF=20.88，LCS覆盖=0.119，4-gram重复=0.000，首次发声=14080 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.796，chrF=20.88，LCS覆盖=0.119，4-gram重复=0.000，首次发声=14080 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.796，chrF=19.95，LCS覆盖=0.102，4-gram重复=0.000，首次发声=14080 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.796，chrF=21.02，LCS覆盖=0.124，4-gram重复=0.000，首次发声=14080 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000006_cmn_eng/episode_000006_cmn_eng/stereo_left_source_right_translation.wav`

#### episode_000028_cmn_eng（cmn→eng，72.20s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000028_cmn_eng.wav`
- reference transcription：到了这个阶段，这些地方，乡绅家族除了能够给自己家孩子多请几个好老师之外，已经做不了更多的事情。呃，刚才还一直沉浸在赵超所说的这种。把他塞进了床上的颈被之下，然后吧嗒一声，门锁锁上了。我是从那期节目开始之后呢，就录转粉儿，每期开始关注。嚯，还能有这事儿。这不好意思的，你人来就行了，怎么拿那么多的东西呢。门槛儿还挺高的，这个这不是人普通家庭就能够轻易去承受的。如果结果还可以的话，那你就，可以半年一去。这样的例子以前也举不胜举啊。当他和大明诸侯那些高官们一起在宫殿上朝时。必须趁自己还没有明显腐败的时候，赶紧让自己被吃掉。那么我们今天高举碳达峰啊，碳中和的这样一个义旗啊，有利于把我们在全世界主要经济体中间主要政治玩家中间的朋友搞得多多的啊敌人对手搞得少少的啊。
- reference translation：At this stage, in these places, the gentry families can do little more than hire a few more good teachers for their children. Oh, I was just immersed in what Zhao Chao was saying. He was pushed under the neck pillow on the bed, and then the door clicked as it was locked. I became a fan after that episode and started following every episode since then. Wow, that's something. "Oh, you shouldn't have, you could just come over, why did you bring so many things?". The threshold is quite high, this is not something an ordinary family can easily afford. If the results are satisfactory, then you can go every six months. There have been countless examples of this before. When he attended the court sessions in the palace with the high officials of Daming. One must be eaten while one is not yet obviously corrupt. So, by upholding the banners of peak carbon emissions and carbon neutrality today, we can make more friends among the major economies and political players in the world and have fewer enemies or opponents.
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.588，chrF=30.29，LCS覆盖=0.232，4-gram重复=0.000，首次发声=5120 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.588，chrF=30.78，LCS覆盖=0.221，4-gram重复=0.000，首次发声=5120 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.588，chrF=31.73，LCS覆盖=0.232，4-gram重复=0.000，首次发声=5120 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.588，chrF=31.50，LCS覆盖=0.232，4-gram重复=0.000，首次发声=5120 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000028_cmn_eng/episode_000028_cmn_eng/stereo_left_source_right_translation.wav`

#### episode_000004_cmn_eng（cmn→eng，71.50s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000004_cmn_eng.wav`
- reference transcription：嗯，没错，但是我对他有些害怕。父亲带我去的避难地，是现在富士屋饭店所在的高地。我觉得吧有总比没有强啊。再说了，想了解剧情的朋友们，你还可以去看动画嘛，对吧？接下来呢再说说游戏的操作啊。在创业者决定雇用某家公司之前。往外走时，一只脚踏在门槛上。他们已在这个世界的快乐和痛苦中耗干了自己。他尴尬的笑了笑，再重新打火。奥运会的成功的神话，是如何被反复延缩塑造的。你现在能想想，你确实咱们从合法的角度来讲，基本没有什么是咱们弄不到的了，对吧。应该是指捐赠扣除钱的应纳税所得额。好的。然后导演呢是有来自韩国，有来自日本和美国的。你会失败，而他会跟你站在一起，教会你如何面对失败。当它一旦蔓延到整个经济体系当中时，人们便开始将手中的资产转换为流动性更高的资产。
- reference translation：Well, yes, but I'm a bit afraid of him. The shelter my father took me to is the high ground where the Fujiya Hotel now stands. I think having something is better than having nothing. Besides, if you want to know the plot, you can always watch the animation, right? Next, let's talk about the game controls. Before an entrepreneur decides to hire a company. When going out, one foot steps on the threshold. They have exhausted themselves in the joy and suffering of this world. He gave an awkward smile and tried to restart the engine. How is the myth of the success of the Olympics repeatedly constructed and shaped. You can think about it, from a legal perspective, there's basically nothing we can't get, right. It should refer to the taxable income after donation deductions. Okay. Then the directors are from South Korea, Japan, and the United States. You will fail, and he will stand by your side, teaching you how to face failure. When it spreads throughout the entire economic system, people begin to convert their assets into more liquid assets.
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.883，chrF=12.02，LCS覆盖=0.125，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.883，chrF=12.02，LCS覆盖=0.125，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.883，chrF=12.02，LCS覆盖=0.125，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.883，chrF=12.02，LCS覆盖=0.125，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000004_cmn_eng/episode_000004_cmn_eng/stereo_left_source_right_translation.wav`

#### episode_000002_cmn_eng（cmn→eng，70.82s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000002_cmn_eng.wav`
- reference transcription：作为领导，适当的给手下增加压力是好的。与其结婚与否无关，但由于已经父亲去世，吴广军拿不出证据证明该套房产是对他个人的赠与。另外，在香港注册的投资公司是吴广军，在婚前就登记注册的，婚后也一直由他自己打理。妻子一直没有对此过问，凭什么离婚时却要来拿走一半。因为经常以前也遇到过啊，在微信上你跟人沟通的时候。我觉得就是你这些华人家族在不同国家的影响力是不一样的，对吧？但是肯定不是这个，就是整个国家的，就是政治势力全部做个排序。什么叫家，一个宝盖头底下就是有很多家庭成员组成的，爷爷奶奶父，爸爸妈妈孩子这些才叫家。那在这个过程当中呢，车企的利润恐怕会继续压薄啊，尽管有销量，但是业绩啊也就勉勉强强可能不太会不会太好看。Conversation在一场对话的中途啊，我我们只是听到他们在说一些什么事情，是是通过我们自己脑补到底是怎么回事。当然情景就是说他太太。
- reference translation：As a leader, it is good to appropriately increase pressure on subordinates. Regardless of whether he is married or not, due to his father's passing, Wu Guangjun cannot provide evidence to prove that the property was a personal gift to him. Additionally, the investment company registered in Hong Kong was registered by Wu Guangjun before the marriage and has always been managed by him alone. His wife has never inquired about it, so why should she take half of it during the divorce. Because I've often encountered this before when communicating with people on WeChat. I think the influence of these Chinese families in different countries is not the same, right? But it's definitely not a ranking of the entire country's political forces. What is a family? Under one roof, there are many family members, such as grandparents, parents, and children; that is what makes a family. In this process, the profit margins of car manufacturers are likely to continue to be squeezed. Although there may be sales volume, the performance might just barely manage and may not look too good. In the middle of a conversation, we just hear them talking about something, and we fill in the details with our own imagination. Of course, the context is about his wife.
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.781，chrF=21.87，LCS覆盖=0.142，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.781，chrF=21.87，LCS覆盖=0.142，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.781，chrF=21.87，LCS覆盖=0.142，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.781，chrF=21.87，LCS覆盖=0.142，4-gram重复=0.000，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000002_cmn_eng/episode_000002_cmn_eng/stereo_left_source_right_translation.wav`

#### episode_000007_eng_cmn（eng→cmn，78.96s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000007_eng_cmn.wav`
- reference transcription：All you control is when they say action and you have a handful of lines to say. And besieged Aristobulus therein, the people still supporting Hyreanus and assisting him in the siege while none. but the priests continued with Aristobulus. So Aretas united the forces of the Arabians and of the Jews together and pressed on the siege vigorously. Voting strategy under approval is guided by two competing features of approval voting. You are more likely to be successful when you are successful. You are more likely to feel satisfied. However, there is another level to consider in the long run. If you continue to advance and improve any area can become challenging at some point. His eye lost every particle of lustre and seemed to sink back and down. The chairman of the committee stated the point he had in view. Mister. Tilden asked him to restate it once or twice, made curious and inconsequential remarks. I can zoom out and I can also use the wide angle lens. You can also use portrait mode when you take a photo. So if there's someone in front of you, it will blur out the background. I am holding a peace sign and the background is blurry. And then I can also take a panorama. So this is just a very long photo.
- reference translation：你控制的只是他们说开始时，你有几句台词要说。并围困了阿里斯托布洛斯，人民仍然支持海雷努斯并协助围攻，而只有祭司们继续支持阿里斯托布洛斯。于是，阿雷塔斯联合了阿拉伯人和犹太人的力量，加紧了围攻。在批准投票下的投票策略受到批准投票的两个相互竞争的特征的指导。当你成功时，你更有可能取得成功。你更有可能感到满意。然而，从长远来看，还有另一个层面需要考虑。如果你继续进步和提高，任何领域在某个时候都可能变得具有挑战性。他的眼睛失去了所有的光彩，似乎向后下沉。委员会主席陈述了他所关注的问题。蒂尔登先生请他重复一两次，然后发表了奇怪而不重要的评论。我可以使用广角镜头，也可以拉远。拍照时你也可以使用人像模式。所以如果有人在你面前，背景会变得模糊。我正在比和平手势，背景是模糊的。然后我还可以拍摄全景照片。所以这是一张很长的照片。
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.797，chrF=13.29，LCS覆盖=0.162，4-gram重复=0.024，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.797，chrF=14.27，LCS覆盖=0.165，4-gram重复=0.024，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.797，chrF=13.29，LCS覆盖=0.162，4-gram重复=0.024，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.797，chrF=13.58，LCS覆盖=0.171，4-gram重复=0.023，首次发声=4480 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000007_eng_cmn/episode_000007_eng_cmn/stereo_left_source_right_translation.wav`

#### episode_000033_eng_cmn（eng→cmn，76.96s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000033_eng_cmn.wav`
- reference transcription：Sure, a string per this definition is a charr star as a programmer would say, what does that mean? A string is quite simply a variable that contains the address of a character. It is a very popular festival and is Ikeda's main event of the year. You can bring him back in six or eight months. This convention's gonna cost plenty. Both dialects have three subdialects. Human knowledge can also be a productive force. Of course, I go to the theater or to the movies with my wife and I go to the bar with my friends. Will you do this. There he will build a wonderful new home for the family. Altogether, it is clear that here or nowhere is that i in as ideal hero. But i have never had a bus boy like you. Harry almost said something apologetic to Percy, but caught himself just in time. Just listen to what other people are talking about. Go yourself and then write down the questions that people ask each other. Like, how was your day? Did you have a good day today? Do you like cats? Do you like dogs? Uh, and then prepare answers in advance for some of those. It's not a perfect solution, uh, but certainly.
- reference translation：好的，根据这个定义，字符串就是程序员所说的字符指针，这是什么意思呢？字符串简单来说就是一个包含字符地址的变量。这是一个非常受欢迎的节日，是池田市一年中的主要活动。你可以在六到八个月后带他回来。这个大会将会花费不少。两种方言各有三个次方言。人类知识也可以是一种生产力。当然，我和我的妻子去看戏剧或电影，我和我的朋友去酒吧。你会做这个吗。他将在那里为家人建造一个美妙的新家。总而言之，这里或任何地方都不是那个理想中的英雄所在之处。但我从来没有遇到过像你这样的服务员。哈利差点对珀西说了一些道歉的话，但及时收住了口。只需听别人在谈论什么。自己去，然后把人们互相问的问题记下来。比如，你今天过得怎么样？你今天过得愉快吗？你喜欢猫吗？你喜欢狗吗？然后提前为其中一些问题准备好答案。这虽然不是完美的解决方案，但确实有所帮助。
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.448，chrF=28.54，LCS覆盖=0.404，4-gram重复=0.000，首次发声=9600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.448，chrF=22.37，LCS覆盖=0.333，4-gram重复=0.031，首次发声=9600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.448，chrF=28.64，LCS覆盖=0.410，4-gram重复=0.000，首次发声=9600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.448，chrF=26.26，LCS覆盖=0.377，4-gram重复=0.000，首次发声=9600 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000033_eng_cmn/episode_000033_eng_cmn/stereo_left_source_right_translation.wav`

#### episode_000023_eng_cmn（eng→cmn，70.88s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000023_eng_cmn.wav`
- reference transcription：A lead appleletes were great to work with because they had only one interest winning. This crisis underpins department-wide staffing challenges, and having it addressed will be a key component in expanding the number of patrol deputies deployed on our streets throughout our region. By the time Leopold and Berthold came back, it was too late. Tower. I'm not sure what the cool places in your city are. Uhm, so be ready. If you're talking to tourists, be ready to talk about your city because that's what they will be interested in. Uh, they'll want to know a little bit about the city that they are visiting. Also be ready to talk about your local food. Draw a picture and write about a time you went to visit the doctor. Social Studies. Make sure that you always incorporate positivity or you try to shift your consciousness, especially if you find yourself getting a little bit wrapped up in negativity or being so overwhelmed by something or you're becoming emotional or mentally stressed out. So pay attention to those things, alright.
- reference translation：领先的运动员很棒，因为他们只有一个目标——获胜。这场危机凸显了整个部门的人员配备挑战，解决这一问题将是扩大我们地区街道上巡逻警员数量的关键组成部分。等到利奥波德和贝特霍尔德回来时，已经太晚了。塔。我不确定你所在城市里有哪些有趣的地方。嗯，所以要做好准备。如果你和游客交谈，准备好谈论你的城市，因为这是他们感兴趣的。呃，他们想知道一些他们正在访问的城市的情况。还要准备好谈论你们当地的食物。画一幅画并写一篇关于你去看医生的文章。社会研究。请确保你始终保持积极的态度，或者试着调整你的意识，特别是当你发现自己被消极情绪所困扰，或者被某些事情压得喘不过气，或者情绪化或精神压力过大时。所以，请注意这些事情，好吗。
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.606，chrF=19.46，LCS覆盖=0.313，4-gram重复=0.007，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.606，chrF=18.02，LCS覆盖=0.291，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.606，chrF=19.86，LCS覆盖=0.295，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.606，chrF=19.41，LCS覆盖=0.313，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000023_eng_cmn/episode_000023_eng_cmn/stereo_left_source_right_translation.wav`

#### episode_000035_eng_cmn（eng→cmn，70.88s）

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_phasea_stateful_longepisode_rl_v1/train/audio/episode_000035_eng_cmn.wav`
- reference transcription：So obviously a bug, at least if I want to tolerate uppercase and lowercase, which is kind of reasonable. This soluble compound is then washed away with the rainwater. Believe that they long for this amount of money should be provided to the project by all of us. In my guidelines for saving the environment, I suggest modest changes, like eating vegetarian meals two days a week. You know, this gap between on the one hand, great gains and getting more kids into school, and on the other hand, in most countries, not really seeing a lot of improvement in learning. Once shut up with you in your loft or wherever you sleep. Of which each extreme angle formed one centre. The whole garden was treated in one harmonious colouring of full yellow, orange and orange brown, half hardy annuals, such as French and African marigolds. He stepped over to the window and shouted through it at the top of his voice that the vacancy was filled. The vacancy has been filled. Richard and Lawless could not move. At nine o'clock the next morning, there were a lot of people in the church. Richard and Lawless were there too.
- reference translation：这显然是一个错误，至少如果我要容忍大小写的话，这是相当合理的。这种可溶性化合物随后被雨水冲走。相信他们渴望这笔钱应该由我们所有人提供给这个项目。在我的环保指南中，我建议做出一些小的改变，比如每周有两天吃素食。你知道，一方面，取得了巨大进展，让更多孩子入学，而另一方面，在大多数国家，学习效果并没有明显改善之间的差距。一旦和你一起被关在阁楼或你睡觉的地方。其中每个极端角度形成一个中心。整个花园采用了和谐的全黄色、橙色和橙棕色的配色，使用了半耐寒的法国和非洲万寿菊等一年生植物。他走到窗前，用最大的声音喊道空缺已经有人填补了。空缺已经填补了。理查德和洛瓦斯动弹不得。第二天早上九点钟，教堂里有很多人。理查德和洛瓦斯也在那里。
- Phase A iter381 + Runtime v2（修复阶段）：CER/WER=0.899，chrF=5.12，LCS覆盖=0.074，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/phasea_iter381_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：CER/WER=0.899，chrF=5.12，LCS覆盖=0.074，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter15_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：CER/WER=0.899，chrF=5.12，LCS覆盖=0.074，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter30_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：CER/WER=0.899，chrF=5.12，LCS覆盖=0.074，4-gram重复=0.000，首次发声=3200 ms。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/rl_iter45_runtime_v2/parts/episode_000035_eng_cmn/episode_000035_eng_cmn/stereo_left_source_right_translation.wav`

## 3. 之前反复试听的四条外部长音频

这四条分别是 Helen Keller、Shimon Peres、新加坡—越南关系和张河桥乡。它们不属于训练数据。本节复用已经在相同 checkpoint、相同 Runtime v2 和相同 640 ms 配置下完成的正式结果，并重新独立读取 WAV 检查采样率、声道、finite、RMS、peak 和非静音比例；不重复消耗 GPU 做完全相同的推理。

| 系统 | 首次发声 mean[min,max] ms↓ | 译音/源音覆盖 | 最大内部静音 mean/max ms↓ | WRITE | pending/TTS失败 | WAV健康 | RTF↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase A iter381 + Runtime v2（修复阶段） | 32640[10240,75520] | 0.472 | 77525/131700 | 204 | 1/1 | 4/4 | 4.468 |
| RL iter15 + Runtime v2 | 32480[10240,74880] | 0.667 | 44450/48900 | 344 | 0/0 | 4/4 | 5.196 |
| RL iter30 + Runtime v2 | 32480[10240,74880] | 0.347 | 53975/60000 | 177 | 0/0 | 4/4 | 4.290 |
| RL iter45 + Runtime v2 | 32480[10240,74880] | 0.685 | 47450/50800 | 315 | 1/4 | 4/4 | 5.463 |

### 3.1 外部长音频试听矩阵

#### long_en_helen_keller_full

- Phase A iter381 + Runtime v2（修复阶段）：首次发声=75520 ms，覆盖=0.316，最大静音=61800 ms，WRITE=42，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：首次发声=74880 ms，覆盖=0.365，最大静音=43000 ms，WRITE=50，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：首次发声=74880 ms，覆盖=0.288，最大静音=60000 ms，WRITE=38，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：首次发声=74880 ms，覆盖=0.338，最大静音=50800 ms，WRITE=44，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav`

#### long_en_shimon_peres_full

- Phase A iter381 + Runtime v2（修复阶段）：首次发声=18560 ms，覆盖=0.170，最大静音=68400 ms，WRITE=33，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：首次发声=18560 ms，覆盖=0.172，最大静音=47400 ms，WRITE=32，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：首次发声=18560 ms，覆盖=0.169，最大静音=47400 ms，WRITE=32，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：首次发声=18560 ms，覆盖=0.181，最大静音=47100 ms，WRITE=32，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav`

#### long_zh_singapore_vietnam_full

- Phase A iter381 + Runtime v2（修复阶段）：首次发声=10240 ms，覆盖=0.387，最大静音=131700 ms，WRITE=48，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：首次发声=10240 ms，覆盖=1.192，最大静音=38500 ms，WRITE=152，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：首次发声=10240 ms，覆盖=0.492，最大静音=59100 ms，WRITE=63，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：首次发声=10240 ms，覆盖=1.043，最大静音=42700 ms，WRITE=124，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav`

#### long_zh_zhangheqiao_full

- Phase A iter381 + Runtime v2（修复阶段）：首次发声=26240 ms，覆盖=1.014，最大静音=48200 ms，WRITE=81，pending/TTS失败=1/1。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav`
- RL iter15 + Runtime v2：首次发声=26240 ms，覆盖=0.939，最大静音=48900 ms，WRITE=110，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch1_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav`
- RL iter30 + Runtime v2：首次发声=26240 ms，覆盖=0.441，最大静音=49400 ms，WRITE=44，pending/TTS失败=0/0。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav`
- RL iter45 + Runtime v2：首次发声=26240 ms，覆盖=1.179，最大静音=49200 ms，WRITE=115，pending/TTS失败=1/4。
  - 连续译音：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_continuous.wav`
  - 时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_global_timeline.wav`
  - 左源右译：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch3_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav`

## 4. 逐样本变化审计：RL 到底改了什么

当前严格级联协议对 ASR prompt 强制关闭 RL adapter，只在 MT、semantic TTS 和 control prompt 上启用。因此 ASR 不是 RL 可学习路径。下面使用逐样本精确字符串比较和 WAV 文件 SHA256，而不是仅比较四舍五入后的指标。`唯一数=1` 表示四个 arm 完全相同；大于 1 表示至少一个 RL checkpoint 真正改变了输出。

| 样本 | ASR文本唯一数 | MT文本唯一数 | 连续WAV唯一数 | 时间轴WAV唯一数 | 立体声WAV唯一数 |
|---|---:|---:|---:|---:|---:|
| episode_000006_cmn_eng | 1 | 3 | 4 | 4 | 4 |
| episode_000028_cmn_eng | 1 | 4 | 4 | 4 | 4 |
| episode_000004_cmn_eng | 1 | 1 | 4 | 4 | 4 |
| episode_000002_cmn_eng | 1 | 1 | 4 | 4 | 4 |
| episode_000007_eng_cmn | 1 | 3 | 4 | 4 | 4 |
| episode_000033_eng_cmn | 1 | 4 | 4 | 4 | 4 |
| episode_000023_eng_cmn | 1 | 4 | 4 | 4 | 4 |
| episode_000035_eng_cmn | 1 | 1 | 4 | 4 | 4 |

| RL checkpoint 相对 Phase A | ASR文本相同 | MT文本相同 | 连续WAV相同 | 时间轴WAV相同 | 立体声WAV相同 |
|---|---:|---:|---:|---:|---:|
| RL iter15 + Runtime v2 | 8/8 | 4/8 | 0/8 | 0/8 | 0/8 |
| RL iter30 + Runtime v2 | 8/8 | 4/8 | 0/8 | 0/8 | 0/8 |
| RL iter45 + Runtime v2 | 8/8 | 3/8 | 0/8 | 0/8 | 0/8 |

审计结论：ASR 在全部 8 条样本、全部 checkpoint 上逐字完全相同，确认 CER/WER 一致不是显示精度导致。iter15/30 各改变 4/8 条最终 MT，iter45 改变 5/8 条；其余样本最终译文未变。所有 WAV 均发生变化，因为 adapter 在 semantic TTS route 上启用，即使最终可见 MT 文本相同，声学 token 仍可改变。固定随机种子后仍观察到这种差异，所以不能把 WAV SHA256 差异误判为结构或内容必然改善。

## 5. 结果结论与当前问题

- **train-seen 最均衡 checkpoint 是 iter30，但提升很小。** 相比 Phase A，平均最大内部静音从 26.88 s 降到 23.58 s，WRITE gap p95 从 24.06 s 降到 22.40 s，英→中 chrF 从 17.19 微升到 17.32，4-gram 重复率从 0.004 降到 0.003；同时 LCS 覆盖从 0.196 降到 0.193，中→英 BLEU 也略升但幅度不足以构成稳定联合收益。
- **iter15 在 train-seen 上整体退化。** 英→中 BLEU/chrF、文本覆盖、WRITE gap 与内部静音都比 Phase A 差；它只是在无 reference 的四条外部长音频上显示出较好的结构覆盖和较少失败，因此不能据此宣称语义质量更好。
- **iter45 是混合收益。** 中→英 BLEU/chrF 与音频覆盖最高，但英→中质量下降，外部长音频仍有 pending/TTS failure，不适合作为统一最佳 checkpoint。
- **首次 WRITE 没有被 RL 改善。** 四个 arm 的 train-seen 首次发声均为 p50=4480 ms、p95=12512 ms；外部长音频仍约 10.24–75.52 s。当前系统不能声称低延迟实时同传。
- **ASR 是首要瓶颈。** 中→英中文 CER=0.763、英→中英文 WER=0.689，且 RL 路由根本不更新 ASR。上游错误和漏识别直接限制增量 MT，后续 reward 无法补回没有进入文本上下文的源内容。
- **训练信号仍偏弱且域不匹配。** 当前仅 64 条 episode、45 次 update；group-relative rollout 候选之间的差异有限。episode 由多个短句以约 160 ms 间隔拼接，虽然长度约一分钟，但不等价于自然连续讲话的停顿、共发音和话题连续性。
- **RTF 只适合本轮 arm 间近似比较。** 正式运行每个 arm 同时启动 8 个模型进程，四个 arm 顺序执行；表中 RTF 包含并发 GPU contention 与解码开销，不应当解释为单流部署 RTF。

下一版应优先让训练目标与推理路由一致：若 reward 包含 ASR 项，就必须让可训练参数实际进入 ASR route，或明确把 ASR reward 移出 policy 优化并单独固定强 ASR。随后扩大自然连续长语音 episode，并直接优化 first-WRITE、WRITE-gap、覆盖和双向 MT 质量的联合 Pareto reward。

## 6. 可复现文件

- 固定 train-seen 协议：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/uniss_phasea_rl_trainlong_eval_v1/evaluation/protocol_train_seen_long8.json`
- 每个 arm 的完整 reference、文本、event、WAV audit 和逐样本指标位于：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_rl_trainlong_eval_v1/<arm>/SCORED.json`
- 外部长音频旧联合报告：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/REPORT.zh-CN.md`
