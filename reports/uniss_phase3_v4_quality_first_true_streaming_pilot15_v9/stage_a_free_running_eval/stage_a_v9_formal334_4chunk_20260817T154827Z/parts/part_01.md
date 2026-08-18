# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8822**
- Weighted CTC blank ratio: **0.1886**
- Weighted streaming WER/CER: **0.4824**
- Weighted causal-full WER/CER: **0.2952**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042263 | 0.1421 | 10 | There remained remained the yes were spent and port | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000042263 | 0.1684 | 11 | Every main. of the yes where was bent in port | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000042263 | 0.1947 | 11 | That 缅因州 of the yes where was spent in port | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.1579 | 10 | That 缅因州 of the yes where was spent in port | wer | 0.4444 |
| 160 | causal_full_asr | CommonVoice_EN_0000069954 | 0.1804 | 19 | P dividends also suggest that the plant was present decades before its first collection | wer | 0.2308 |
| 320 | causal_full_asr | CommonVoice_EN_0000069954 | 0.1867 | 17 | The defence also suggests that the plant was present decades before its first collection | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000069954 | 0.2089 | 17 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.2184 | 23 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000237791 | 0.1311 | 10 | I this Ben to but a stone old | wer | 0.8750 |
| 320 | streaming_asr | CommonVoice_EN_0000237791 | 0.1803 | 10 | The does he Ben to but a stone old | wer | 0.8750 |
| 640 | streaming_asr | CommonVoice_EN_0000237791 | 0.1749 | 10 | The does he bunch to but a stone old | wer | 0.8750 |
| 1280 | streaming_asr | CommonVoice_EN_0000237791 | 0.2678 | 14 | The dusty bunch to but a stone all | wer | 0.6250 |
| 160 | streaming_asr | CommonVoice_EN_0000352398 | 0.1993 | 15 | He concluded he I school form lam al ball high guy | wer | 0.7273 |
| 320 | streaming_asr | CommonVoice_EN_0000352398 | 0.2320 | 19 | He complicated he high school form lam al ball high scu | wer | 0.6364 |
| 640 | streaming_asr | CommonVoice_EN_0000352398 | 0.2222 | 17 | He completed he high school from lam a all high scrooge | wer | 0.4545 |
| 1280 | streaming_asr | CommonVoice_EN_0000352398 | 0.2680 | 22 | He complicated he high school room ram all ball high scu | wer | 0.5455 |
| 160 | causal_full_asr | CommonVoice_EN_0000471648 | 0.1822 | 13 | Wood is best for making toys and blacks | wer | 0.1250 |
| 320 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2356 | 13 | Wood is best for making toys and blacks | wer | 0.1250 |
| 640 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2578 | 14 | Wood is best for making toys and blacks | wer | 0.1250 |
| 1280 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2933 | 15 | Wood is best for making toys and blacks | wer | 0.1250 |
| 160 | streaming_asr | CommonVoice_EN_0000519794 | 0.1595 | 16 | Give pass from conclusion is list of Then the on this course | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000519794 | 0.2371 | 19 | You by of in the equation is list of Then that on this goal | wer | 1.0000 |
| 640 | streaming_asr | CommonVoice_EN_0000519794 | 0.2672 | 21 | You By of in a transition is list of Then the on this call | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000519794 | 0.2457 | 18 | You've by of in conclusion is list of Then the and goes | wer | 0.8333 |
| 160 | streaming_asr | HQ-Conversations_0000041144 | 0.2094 | 20 | 对和谢谢人处这个极端什么其实我也有点这种焦虑 | cer | 0.4348 |
| 320 | streaming_asr | HQ-Conversations_0000041144 | 0.1986 | 20 | 对后下身处这个阶段嘛其实我也有点这种焦虑 | cer | 0.2174 |
| 640 | streaming_asr | HQ-Conversations_0000041144 | 0.1769 | 20 | 对后下人处这个阶段嘛其实我也有点这种焦虑 | cer | 0.2609 |
| 1280 | streaming_asr | HQ-Conversations_0000041144 | 0.1913 | 18 | 对和下身处这个阶段嘛其实我这有点这种焦虑 | cer | 0.3043 |
| 160 | streaming_asr | LibriSpeech_0000104521 | 0.1659 | 58 | And the drew up beside polis steps And age wink dressed the unform of Silver cloth came forward to sister a to light said the girl could to he personage showed was said one to master emperor | wer | 0.5500 |
| 320 | streaming_asr | LibriSpeech_0000104521 | 0.1730 | 57 | And the drew out up beside polis steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed was said one to master emperor | wer | 0.5000 |
| 640 | streaming_asr | LibriSpeech_0000104521 | 0.1671 | 62 | And the drew out up beside police steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed was said one to master emperor | wer | 0.5000 |
| 1280 | streaming_asr | LibriSpeech_0000104521 | 0.1635 | 61 | And the drew out of beside police steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed where said one to master emperor | wer | 0.5250 |
| 160 | causal_full_asr | LibriSpeech_0000214472 | 0.0882 | 25 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared. He spoke eagerly and anxiously. | wer | 0.1200 |
| 320 | causal_full_asr | LibriSpeech_0000214472 | 0.1059 | 25 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared. He spoke eagerly and anxiously | wer | 0.0800 |
| 640 | causal_full_asr | LibriSpeech_0000214472 | 0.0985 | 28 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared he spoke eagerly and anxiously | wer | 0.0400 |
| 1280 | causal_full_asr | LibriSpeech_0000214472 | 0.0956 | 27 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far certainly disappeared. He spoke eagerly and anxiously | wer | 0.1200 |
| 160 | streaming_asr | LibriSpeech_0000265196 | 0.1239 | 34 | I she be real great it you say nothing about this there or some in the had house and neighbourhood who son not there's is you state here and the you you out for kind go to bed read here or ex | wer | 0.4667 |
| 320 | streaming_asr | LibriSpeech_0000265196 | 0.1531 | 39 | I she by real great a you say nothing about this They or some in the had how and neighbourhood who son not fancy dies you state here and the you you out for kind go to bed read here are ex | wer | 0.5333 |
| 640 | streaming_asr | LibriSpeech_0000265196 | 0.1676 | 45 | I show by real great if you say nothing. about this They or some in the had how and neighbourhood who sincerely not fancy dies you state here and the you you a for kind of go to bed read here are ex | wer | 0.5333 |
| 1280 | streaming_asr | LibriSpeech_0000265196 | 0.1458 | 48 | I she by real great a you say nothing about this they or some in the had how and neighbourhood who sincerely in a fancy dies you state here and the you you a for kind of go to bed read here are books | wer | 0.5111 |
| 160 | streaming_asr | emilia_zh_0004003103 | 0.2185 | 54 | 所以我就的这些可能父母个有的影响对他会影响我但是吵吵没有没有一套观念说我是必须得吃反抗什么都去去争取我要行的我窗外是觉得说我可以<\|write_generate\|><\|cmn\|><\|start_content\|>顺其自然自然而然是得到想要懂<\|write_generate\|><\|cmn\|><\|start_content\|>你就 | cer | 1.2184 |
| 320 | streaming_asr | emilia_zh_0004003103 | 0.2328 | 57 | 所以我就的之前可能父母个有的影响对他会影响但是吵吵没有没有一套观念说我是必须得是反抗什么都去去争取我要性的我从而是觉得说我可以可以顺其自然自然是得到想要等<\|write_generate\|><\|cmn\|><\|start_content\|>你就 | cer | 0.8046 |
| 640 | streaming_asr | emilia_zh_0004003103 | 0.2328 | 60 | 所以我就都这些可能父母我我的影响对他会影响但是吵了没有没有一套观念说我是必须得是反抗什么都去争取我要等行的我从是觉得说可以可以顺其自然自然是得到想要懂<\|write_generate\|><\|cmn\|><\|start_content\|>嗯 | cer | 0.7586 |
| 1280 | streaming_asr | emilia_zh_0004003103 | 0.2107 | 60 | 所以我的这些可能父母我我的影响对他会影响但是我没有没有一套观念说我是必须得是反抗什么都去争取我要的行的我从是觉得说我可以可以顺其自然自然是得到想要的<\|write_generate\|><\|cmn\|><\|start_content\|>嗯 | cer | 0.7126 |
| 160 | streaming_asr | emilia_zh_0004176058 | 0.1087 | 8 | 她传发明改变的美国 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0004176058 | 0.1565 | 12 | 新一代传发明改变的美国 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004176058 | 0.1609 | 12 | 新一代传发明改变的美国 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004176058 | 0.1565 | 11 | 新一代传发明改变的美国 | cer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0004213341 | 0.1374 | 17 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004213341 | 0.1450 | 16 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004213341 | 0.1450 | 18 | 世界上每天都有积极的发生这些奇迹大都来源于精神的力量 | cer | 0.1154 |
| 1280 | causal_full_asr | emilia_zh_0004213341 | 0.1298 | 16 | 世界上每天都有奇迹的发生这些奇迹大都来源于精神的力量 | cer | 0.0385 |
| 160 | streaming_asr | emilia_zh_0004422761 | 0.2606 | 15 | This let many tire people to set to and hounds and Cities | wer | 0.5455 |
| 320 | streaming_asr | emilia_zh_0004422761 | 0.2500 | 14 | This let many tire people to set the and hounds and Cities | wer | 0.5455 |
| 640 | streaming_asr | emilia_zh_0004422761 | 0.1755 | 14 | This led many tire people to set to and hounds and Cities | wer | 0.4545 |
| 1280 | streaming_asr | emilia_zh_0004422761 | 0.1702 | 14 | This let many tired people to set to and hounds and Cities | wer | 0.5455 |
| 160 | causal_full_asr | emilia_zh_0004665564 | 0.1591 | 35 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled an iron hand in a velvet political glove | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004665564 | 0.1316 | 34 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled an iron hand in a velvet political glove. | wer | 0.2857 |
| 640 | causal_full_asr | emilia_zh_0004665564 | 0.1316 | 34 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled with an iron hand in a velvet political glove | wer | 0.2571 |
| 1280 | causal_full_asr | emilia_zh_0004665564 | 0.1297 | 37 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled with an iron hand in a velvet political glove | wer | 0.2286 |
| 160 | streaming_asr | emilia_zh_0004706082 | 0.2174 | 8 | The quality have looking clever the and was | wer | 0.4444 |
| 320 | streaming_asr | emilia_zh_0004706082 | 0.2236 | 10 | But quality have looking clever or the and was | wer | 0.5556 |
| 640 | streaming_asr | emilia_zh_0004706082 | 0.1429 | 9 | The quality have looking clever the and was | wer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0004706082 | 0.1429 | 10 | The quoted have looking clever the and was | wer | 0.5556 |
| 160 | streaming_asr | emilia_zh_0004797638 | 0.2105 | 10 | I Thank Qiao five take could not pickled chop. | wer | 0.8889 |
| 320 | streaming_asr | emilia_zh_0004797638 | 0.2158 | 10 | Hua Thank Q find take could not picly twist | wer | 0.8889 |
| 640 | streaming_asr | emilia_zh_0004797638 | 0.2368 | 10 | Fang Thank Qiao find take could not quickly chuckled | wer | 0.7778 |
| 1280 | streaming_asr | emilia_zh_0004797638 | 0.2632 | 12 | Fang Thank kills find take could not picly chop. | wer | 0.8889 |
| 160 | streaming_asr | emilia_zh_0004874290 | 0.2271 | 17 | The book from miss Song ten arrive that's evening with the kind not find | wer | 0.6923 |
| 320 | streaming_asr | emilia_zh_0004874290 | 0.1992 | 18 | The book from miss song and arrives that's evening was the kind not find | wer | 0.6154 |
| 640 | streaming_asr | emilia_zh_0004874290 | 0.1873 | 18 | The book from miss <\|glm_semantic_14290\|>one arrives that's evening was the kind not find | wer | 0.5385 |
| 1280 | streaming_asr | emilia_zh_0004874290 | 0.1713 | 18 | The book from miss <\|glm_semantic_14290\|>onement arrives that's evening was the kind not find | wer | 0.5385 |
| 160 | streaming_asr | emilia_zh_0005000497 | 0.2357 | 23 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005000497 | 0.2500 | 23 | 街区企业所有的债务之后所得的的就是谢价值 | cer | 0.2857 |
| 640 | streaming_asr | emilia_zh_0005000497 | 0.2357 | 22 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005000497 | 0.2393 | 22 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 160 | causal_full_asr | emilia_zh_0005184596 | 0.1212 | 10 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 320 | causal_full_asr | emilia_zh_0005184596 | 0.1515 | 11 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 640 | causal_full_asr | emilia_zh_0005184596 | 0.1566 | 11 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 1280 | causal_full_asr | emilia_zh_0005184596 | 0.1515 | 13 | That can't be a criticism of your hands. | cer | 1.7368 |
| 160 | streaming_asr | emilia_zh_0005370483 | 0.1876 | 23 | 录有的就这就就是是是是一个什么感觉就是愚弄和他但错他不的啊 | cer | 0.6098 |
| 320 | streaming_asr | emilia_zh_0005370483 | 0.2174 | 27 | 读说的就就就就是是是是一个什么感觉就是鱼龙混的他但错好的不的啊 | cer | 0.4878 |
| 640 | streaming_asr | emilia_zh_0005370483 | 0.1991 | 28 | 读有的就就就就是是是是一个什么感觉就是鱼龙混的他天错他的不的。 | cer | 0.5122 |
| 1280 | streaming_asr | emilia_zh_0005370483 | 0.2174 | 28 | 读有的就就就就是是是是一个什么感觉就是鱼龙混的他但错他的不的好 | cer | 0.4878 |
| 160 | streaming_asr | emilia_zh_0005601476 | 0.3030 | 16 | 但是情绪中不如把心情找我 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0005601476 | 0.3273 | 16 | 外情绪中不如把心情找我 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005601476 | 0.3333 | 18 | 外情绪中不如把心情叫我 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005601476 | 0.3273 | 17 | 外情绪中不如把心情叫我 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005852724 | 0.2602 | 23 | 这个算重要的这这位妈妈我一次希望我的两个女儿可以被 | cer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0005852724 | 0.2677 | 22 | 这个是很重要的这身为妈妈我一次希望我的两个女儿可以被 | cer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0005852724 | 0.2342 | 19 | 这个是很重要的这身为妈妈我一次希望我的两个女儿可以被 | cer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0005852724 | 0.2268 | 20 | 这个是很重要的所以身为妈妈我一直在希望我的两个女儿可以被 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0005853036 | 0.2254 | 26 | 做这个就如果我想长的话我排斥啊家里你介绍的也事儿呼吸没啥 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005853036 | 0.2159 | 28 | 做这个就如果我然后长的话我排斥者嗯家里就介绍的也事儿父亲没是 | cer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0005853036 | 0.2190 | 32 | 做这个就如果不是然后长的话我排斥嗯家里就介绍的也事儿会没是 | cer | 0.5312 |
| 1280 | streaming_asr | emilia_zh_0005853036 | 0.2000 | 28 | 的这个就如果不是然后找的话我不排斥嗯家里新介绍的也哎呼吸没是 | cer | 0.5312 |
| 160 | streaming_asr | emilia_zh_0006099379 | 0.1398 | 14 | 啊都了第一四居然个这样然后然后分开使用 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006099379 | 0.1613 | 17 | 啊都了你四转到个这样然后然后分开使用 | cer | 0.5789 |
| 640 | streaming_asr | emilia_zh_0006099379 | 0.1470 | 15 | 哦都了你四居然个量然后他分开使用 | cer | 0.3684 |
| 1280 | streaming_asr | emilia_zh_0006099379 | 0.1398 | 15 | 哦都了你四居然个量然后他分开使用 | cer | 0.3684 |
| 160 | causal_full_asr | emilia_zh_0006270577 | 0.2842 | 22 | 用直接用三份盖其实可以的啊这个也不能哪里我们并不是所有的人都需要打 | cer | 0.3333 |
| 320 | causal_full_asr | emilia_zh_0006270577 | 0.2945 | 23 | 用直接用三份盖其实可以的啊这个也不能理我们并不是所有的人都需要的 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0006270577 | 0.2603 | 22 | 用直接用三份盖其实可以的啊这个也不能拿礼物并不是所有的人都是需要的 | cer | 0.2727 |
| 1280 | causal_full_asr | emilia_zh_0006270577 | 0.2637 | 23 | 用直接用三份盖其实可以的啊这个也不能理我并不是所有的人都需要的 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0006304027 | 0.1643 | 19 | 如果按照我们刚刚所讲的这个规律来看的话这我们对的龙啊人是哪更呢 | cer | 0.2286 |
| 320 | streaming_asr | emilia_zh_0006304027 | 0.1902 | 28 | 如果按照什么刚刚所讲的这个规律来看的话这我们对应的龙啊赢是哪更呢 | cer | 0.2571 |
| 640 | streaming_asr | emilia_zh_0006304027 | 0.1729 | 23 | 如果按照什么刚刚所讲的这个规律来看的话这里我们对应的龙啊因是哪个呢 | cer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0006304027 | 0.1614 | 22 | 如果按照我们刚刚所讲的最大的规律来看的话最我们对应到龙啊因是哪个呢 | cer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0006430098 | 0.2280 | 23 | Just was offered Several good job But he want it wait and the coron | wer | 0.4286 |
| 320 | streaming_asr | emilia_zh_0006430098 | 0.2443 | 24 | Zhang Was offer Several good job But he want it wait and but coron. | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006430098 | 0.1889 | 19 | Zhang Was offered several good job But he want it wait and the corrot and | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0006430098 | 0.1498 | 18 | John Was offered Several good job But he want it wait and the corral and | wer | 0.4286 |
| 160 | streaming_asr | emilia_zh_0006503627 | 0.2255 | 35 | 昔日繁华都会情怀圣经经历如此肆无忌惮的烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0006503627 | 0.2276 | 31 | 昔日繁华都会情怀圣经经历如此肆无忌惮的烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0006503627 | 0.2276 | 32 | 昔日繁华都会情怀圣经经历如此肆无忌惮烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0006503627 | 0.2296 | 35 | 昔日繁华多会情怀圣经经历如此肆无忌惮烧杀抢后被摧毁打击成为一片废墟 | cer | 0.1944 |
| 160 | streaming_asr | emilia_zh_0006725267 | 0.3540 | 13 | 了出去留待也是帮助部分讨论院门 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006725267 | 0.3416 | 12 | 布兰登就是留着也是帮助部分讨论院门 | cer | 0.6316 |
| 640 | streaming_asr | emilia_zh_0006725267 | 0.3292 | 13 | 布莱顿就是留着也是帮助部分讨论院们 | cer | 0.6842 |
| 1280 | streaming_asr | emilia_zh_0006725267 | 0.3478 | 15 | 柏林出榴弹也是帮部分讨论院们 | cer | 0.5789 |
| 160 | causal_full_asr | emilia_zh_0006884293 | 0.2039 | 34 | 不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0286 |
| 320 | causal_full_asr | emilia_zh_0006884293 | 0.2187 | 33 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0006884293 | 0.2162 | 30 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0006884293 | 0.2138 | 29 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 25 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 320 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 28 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 640 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 28 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 1280 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 29 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 160 | streaming_asr | emilia_zh_0007170191 | 0.2204 | 13 | 这对犹太人来说肯定是不能接受等 | cer | 0.0667 |
| 320 | streaming_asr | emilia_zh_0007170191 | 0.2043 | 16 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007170191 | 0.2204 | 15 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007170191 | 0.2151 | 15 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007524507 | 0.2009 | 16 | 好我在来看又在找的这就用去杠杆他的颜色 | cer | 0.5652 |
| 320 | streaming_asr | emilia_zh_0007524507 | 0.2366 | 18 | 好我在来又在找的这个就是。去杠杆他的的颜色 | cer | 0.6087 |
| 640 | streaming_asr | emilia_zh_0007524507 | 0.2143 | 19 | 好我在来看就散着的这个就是用去更改的的颜色 | cer | 0.4348 |
| 1280 | streaming_asr | emilia_zh_0007524507 | 0.2054 | 18 | 好我在来看就散着的这个就是。去更改他的颜色 | cer | 0.4348 |
| 160 | streaming_asr | emilia_zh_0007721270 | 0.2981 | 13 | 那我不管家公司时我通常就两件的 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0007721270 | 0.3043 | 15 | 当我不管家公司时我通常就两件是 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0007721270 | 0.2981 | 17 | 当我关于家公司时我通常就两件是 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0007721270 | 0.2795 | 14 | 当我有关家公司时我通常就两件的 | cer | 0.3889 |
| 160 | streaming_asr | EN_B00097_S03849_W000000 | 0.1925 | 32 | You you and defe don't and lined Then though the American boisage democracy And Capitalistic Civilization of was enamis live and progressed | wer | 0.6296 |
| 320 | streaming_asr | EN_B00097_S03849_W000000 | 0.1863 | 34 | He for now defe dom and line Then though the American boisage democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6667 |
| 640 | streaming_asr | EN_B00097_S03849_W000000 | 0.1718 | 31 | Here for now defe don't and line Then no the American boys. democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00097_S03849_W000000 | 0.1429 | 31 | It for not defe dom and line then no the American boys. democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6296 |
| 160 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2500 | 14 | 因为他们正在迅速消耗这些资源 | wer | 1.0000 |
| 320 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2849 | 15 | 因为他们正在迅速消耗这些资源 | wer | 1.0000 |
| 640 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2326 | 13 | 他们正在迅速消耗这些资源 | wer | 1.0000 |
| 1280 | causal_full_asr | EN_B00043_S01557_W000006 | 0.1919 | 15 | 他们正在迅速消耗这些资源 | wer | 1.0000 |
| 160 | streaming_asr | EN_B00064_S08593_W000000 | 0.1382 | 25 | He Teachers the pass does not exist a fact which longed to as sphere of knowledge And which before before one and the world and a | wer | 0.3704 |
| 320 | streaming_asr | EN_B00064_S08593_W000000 | 0.1451 | 28 | He teach of the pass does not exist a fact which belongs to to a sphere of knowledge And which before before one and the world and a | wer | 0.3704 |
| 640 | streaming_asr | EN_B00064_S08593_W000000 | 0.1297 | 27 | He teacher the pass does not exist a fact which belongs to with the sphere from lodged And which before before one and the world and a | wer | 0.4074 |
| 1280 | streaming_asr | EN_B00064_S08593_W000000 | 0.1195 | 23 | He teach the pass does not exist a fact which belongs to with the sphere of knowledge And which before the one and the world and a | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00048_S03599_W000216 | 0.2086 | 22 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 320 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1771 | 20 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 640 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1600 | 18 | So for those different categories of use you'll find that the model of a verb is used slightly differently. | wer | 0.3529 |
| 1280 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1571 | 17 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 160 | streaming_asr | EN_B00048_S05961_W000019 | 0.1844 | 29 | I'll just I'll just use this there to explained orge of the vast variety of caryotic organisms | wer | 0.5625 |
| 320 | streaming_asr | EN_B00048_S05961_W000019 | 0.1671 | 25 | I just I use this there to explain orge of the vast variety of carriotic organisms | wer | 0.4375 |
| 640 | streaming_asr | EN_B00048_S05961_W000019 | 0.1556 | 29 | biologists I use this there to explained orge of the vast variety of carriotic organisms | wer | 0.3750 |
| 1280 | streaming_asr | EN_B00048_S05961_W000019 | 0.1326 | 25 | I'll just I'll just use this there to explained orge of the vast variety of carriotic organisms | wer | 0.5625 |
| 160 | streaming_asr | EN_B00058_S03808_W000018 | 0.1573 | 49 | We can also say I had gone having to functions in there to to a pass perfect as well was the perfect hands with the for so and she can see what inflation does is it changers it actually change the word | wer | 0.3023 |
| 320 | streaming_asr | EN_B00058_S03808_W000018 | 0.1418 | 48 | We can also say I had gone having to functions in there to indication a pass perfect as well was the perfect hands with the for so a she can see what inflation does is it changers it actually change the word | wer | 0.3023 |
| 640 | streaming_asr | EN_B00058_S03808_W000018 | 0.1251 | 47 | We you also say I had gone having to functions in there to in the cave a pass perfect as well was the perfect lengths with the for so a she can see what inflation does is it changers it actually change the word | wer | 0.3721 |
| 1280 | streaming_asr | EN_B00058_S03808_W000018 | 0.1263 | 46 | We you also say I had gone having to inflictions in there to in the cave a pass perfect as well was the perfect lengths with the for so and she can see what in fluctuation does is it changers it actually change the word | wer | 0.3953 |
| 160 | streaming_asr | EN_B00091_S08343_W000001 | 0.1361 | 16 | Not and mommy came to serveys my at shiveness the gravations seremony | wer | 0.7500 |
| 320 | streaming_asr | EN_B00091_S08343_W000001 | 0.1492 | 20 | Both and mommy came to serveys my at shiveness the gravations severnity | wer | 0.7500 |
| 640 | streaming_asr | EN_B00091_S08343_W000001 | 0.1649 | 18 | But and moment came to serveys my at sh improvements The gravations severnese | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00091_S08343_W000001 | 0.1545 | 17 | But and Moments came to severage my at sh improvements The gravations thermony | wer | 0.7500 |
| 160 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1828 | 22 | These are not variables of algebra. A variable is any characteristic | wer | 0.2308 |
| 320 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1690 | 20 | These are not the variables of algebra. A variable is any characteristic | wer | 0.1538 |
| 640 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1345 | 20 | These are not the variables of algebra. A variable is any characteristic | wer | 0.1538 |
| 1280 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1310 | 21 | And these are not the variables of algebra. A variable is any characteristic | wer | 0.0769 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
