# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9152**
- Weighted CTC blank ratio: **0.8791**
- Weighted streaming WER/CER: **0.2827**
- Weighted causal-full WER/CER: **0.1058**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042263 | 0.8421 | 16 | There remained of the year was spent in court | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000042263 | 0.8053 | 21 | There remained of the year was spent in port | wer | 0.2222 |
| 640 | streaming_asr | CommonVoice_EN_0000042263 | 0.7947 | 23 | There remained of the year was spent in port | wer | 0.2222 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.8000 | 20 | There remained of the year was spent in port | wer | 0.2222 |
| 160 | causal_full_asr | CommonVoice_EN_0000069954 | 0.8354 | 31 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000069954 | 0.8418 | 34 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000069954 | 0.7943 | 39 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.7595 | 44 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000237791 | 0.8251 | 19 | The dusty bench to by let stone all | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000237791 | 0.8251 | 19 | The dusty bench to by it stone all | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000237791 | 0.8197 | 24 | The dusty bench to by it stone all | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000237791 | 0.8361 | 20 | The dusty bench to by let stone old | wer | 0.3750 |
| 160 | streaming_asr | CommonVoice_EN_0000352398 | 0.9575 | 8 | He complicated his I school form ram L all high school | wer | 0.4545 |
| 320 | streaming_asr | CommonVoice_EN_0000352398 | 0.9575 | 7 | He conflicted his I school formal lamb L all high school | wer | 0.5455 |
| 640 | streaming_asr | CommonVoice_EN_0000352398 | 0.9248 | 13 | He confiliated he high school from lam L all high school | wer | 0.4545 |
| 1280 | streaming_asr | CommonVoice_EN_0000352398 | 0.9444 | 9 | He confiliated he high school from lamb al ball high school | wer | 0.4545 |
| 160 | causal_full_asr | CommonVoice_EN_0000471648 | 0.9156 | 13 | Wood is best for making chosen blacks | wer | 0.3750 |
| 320 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8889 | 13 | Wood is best for making chosen blacks | wer | 0.3750 |
| 640 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8933 | 13 | Wood is best for making chosen blacks | wer | 0.3750 |
| 1280 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8844 | 13 | Wood is best for making toys and blacks | wer | 0.1250 |
| 160 | streaming_asr | CommonVoice_EN_0000519794 | 0.8922 | 17 | Give pass of in acquisition is listed Then is underclass | wer | 0.5833 |
| 320 | streaming_asr | CommonVoice_EN_0000519794 | 0.8750 | 18 | If prized of acquisition is unless to then is underscarce | wer | 0.5833 |
| 640 | streaming_asr | CommonVoice_EN_0000519794 | 0.8664 | 21 | You priced of in acquisition is listed then is under scares | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000519794 | 0.8664 | 19 | You've priced of acquisition is unless to then is un discounts | wer | 0.7500 |
| 160 | streaming_asr | HQ-Conversations_0000041144 | 0.9892 | 3 | 对然后下身处这个阶段的其实我也有点这种焦虑 | cer | 0.1739 |
| 320 | streaming_asr | HQ-Conversations_0000041144 | 0.9675 | 9 | 对然后下身处这个极端吗其实我也有点这种焦虑 | cer | 0.2609 |
| 640 | streaming_asr | HQ-Conversations_0000041144 | 0.9711 | 7 | 对然后下身处这个极端吧其实我也有点这种焦虑 | cer | 0.2174 |
| 1280 | streaming_asr | HQ-Conversations_0000041144 | 0.9675 | 9 | 对然后下身处这个极端吧其实我也有点这种焦虑 | cer | 0.2174 |
| 160 | streaming_asr | LibriSpeech_0000104521 | 0.8223 | 71 | And the drew a up beside the palace steps and aged winky dressed the uniforms of Silver cloth came forward to sists some to light said the carrot to his personage show was set once to master emperor | wer | 0.4000 |
| 320 | streaming_asr | LibriSpeech_0000104521 | 0.7903 | 85 | And the drew a up beside the palace steps and aged winky dressed the uniform of Silver cloth came forward to sists some to light said the carrot to his personage show was at once to master emperor | wer | 0.3500 |
| 640 | streaming_asr | LibriSpeech_0000104521 | 0.7701 | 94 | And the drew a up beside the palace steps and aged winky dressed the uniform of Silver cloth came forward to sists some to light said the carrot to his personage show was at once to master emperor | wer | 0.3500 |
| 1280 | streaming_asr | LibriSpeech_0000104521 | 0.7784 | 89 | And the drew a up beside the palace steps and aged winky dressed the uniform of Silver cloth came forward to sists some to light said the carrot to his personage show was at once to master emperor | wer | 0.3500 |
| 160 | causal_full_asr | LibriSpeech_0000214472 | 0.8279 | 59 | By common consent the subject is never mentioned between us The better irony of his tone thus far suddenly disappeared he spoke eagerly and anxiously | wer | 0.0400 |
| 320 | causal_full_asr | LibriSpeech_0000214472 | 0.8176 | 57 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far Suddenly disappeared he spoke eagerly and anxiously | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000214472 | 0.8118 | 60 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared he spoke eagerly and anxiously | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000214472 | 0.8015 | 62 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared he spoke eagerly and anxiously | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000265196 | 0.8017 | 68 | I show be really great if will say nothing about this there are some in the had house and neighbourhood who surely not enough fancy is You stay here and do few do not for the kind to go a bed read Here a books | wer | 0.3333 |
| 320 | streaming_asr | LibriSpeech_0000265196 | 0.7974 | 70 | And show be really great told if you will excite nothing about this there are some in the had how and neighbourhood who surely in a fancy is You stay here and if few do not for the kind to go a bed read here a books | wer | 0.4000 |
| 640 | streaming_asr | LibriSpeech_0000265196 | 0.7945 | 72 | I show be really great if you will excite nothing about this there are some in the had how and neighbourhood who sincerely in a fancy is You stay here and if you do not for kind of go a bed read Here a books | wer | 0.3333 |
| 1280 | streaming_asr | LibriSpeech_0000265196 | 0.7974 | 70 | I show be really great if you will excite nothing about this there are some in the had how and neighbourhood who Certainly in a fancy is You stay here and if you do not for kind of go a bed read Here are books | wer | 0.3111 |
| 160 | streaming_asr | emilia_zh_0004003103 | 0.9636 | 23 | 所以我就的这些可能父母给我的影响会他会影响我但是我穷了没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西新的我从来都就是觉得说我可以很顺其自然自然而然是得到我想要懂嗯 | cer | 0.1494 |
| 320 | streaming_asr | emilia_zh_0004003103 | 0.9636 | 22 | 所以我觉得的这些可能父母给我的影响会他会影响我但是我穷了没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西新的我从来都就是觉得说我可以很顺其自然自然而是得到我想要懂想嗯 | cer | 0.1494 |
| 640 | streaming_asr | emilia_zh_0004003103 | 0.9636 | 23 | 所以我觉得的这些可能父母给给我的影响会他会影响我但是我穷了没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西新的我从来都就是觉得说我可以很顺其自然自然而然是得到我想要懂想嗯 | cer | 0.1494 |
| 1280 | streaming_asr | emilia_zh_0004003103 | 0.9688 | 21 | 所以我就觉得这些可能父母给我的影响会他会影响我但是成了没有没有一套观念说我是必须得是反抗什么都去去争取我要东西新的我从来都就是觉得说我可以很顺其自然自然而然是得到我想要懂想嗯 | cer | 0.1609 |
| 160 | streaming_asr | emilia_zh_0004176058 | 0.9870 | 3 | 新一代传奇发明改变的美国 | cer | 0.0833 |
| 320 | streaming_asr | emilia_zh_0004176058 | 0.9913 | 2 | 新一代传奇发明改变的美国 | cer | 0.0833 |
| 640 | streaming_asr | emilia_zh_0004176058 | 0.9870 | 2 | 新一代传奇发明改变的美国 | cer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0004176058 | 0.9870 | 2 | 新一代传奇发明改变的美国 | cer | 0.0833 |
| 160 | causal_full_asr | emilia_zh_0004213341 | 0.9771 | 6 | 世界上每天都有机器在发生这些奇迹大都来源于精神的力量 | cer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004213341 | 0.9580 | 10 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004213341 | 0.9580 | 9 | 世界上每天都有积极的发生这些奇迹大都来源于精神的力量 | cer | 0.1154 |
| 1280 | causal_full_asr | emilia_zh_0004213341 | 0.9580 | 9 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004422761 | 0.8351 | 17 | This let many twarry people to settle int towns and cities | wer | 0.2727 |
| 320 | streaming_asr | emilia_zh_0004422761 | 0.7872 | 20 | This led many twarry people to settle and towns and cities | wer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0004422761 | 0.7553 | 23 | This led many twired people to settle and towns and cities | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0004422761 | 0.7979 | 22 | This led many twired people to settle and towns and cities | wer | 0.1818 |
| 160 | causal_full_asr | emilia_zh_0004665564 | 0.6719 | 101 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acid of features He rolled with an iron hand and a velvet political glove | wer | 0.2571 |
| 320 | causal_full_asr | emilia_zh_0004665564 | 0.6365 | 109 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acidific features He rolled with an iron hand and a velvet political glove | wer | 0.2286 |
| 640 | causal_full_asr | emilia_zh_0004665564 | 0.6523 | 98 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acidific features He rolled with an iron hand and enveloped political ground | wer | 0.2571 |
| 1280 | causal_full_asr | emilia_zh_0004665564 | 0.6660 | 90 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acidific features He rolled with an iron hand and enveloped political love | wer | 0.2571 |
| 160 | streaming_asr | emilia_zh_0004706082 | 0.8509 | 11 | The quality of looking clever the he was | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0004706082 | 0.7826 | 15 | The quality of looking clever the he was | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004706082 | 0.8261 | 15 | The quality of looking clever the he was | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004706082 | 0.8199 | 13 | The quality of looking clever the he was | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004797638 | 0.9895 | 2 | I take Q five take cute not quickly try | wer | 0.7778 |
| 320 | streaming_asr | emilia_zh_0004797638 | 0.9895 | 2 | Why thank cue find that you not quickly try | wer | 0.5556 |
| 640 | streaming_asr | emilia_zh_0004797638 | 0.9895 | 2 | What Thank kills find take you not quickly try | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0004797638 | 0.9263 | 8 | Flying thank kill find that you not quickly try | wer | 0.5556 |
| 160 | streaming_asr | emilia_zh_0004874290 | 0.8327 | 27 | The book from mister phone turned arrived that's evening with the kind noted find | wer | 0.5385 |
| 320 | streaming_asr | emilia_zh_0004874290 | 0.7769 | 35 | The book from mister phone turned arrives that evening with the kind note in five | wer | 0.5385 |
| 640 | streaming_asr | emilia_zh_0004874290 | 0.7769 | 33 | The book from mister phone to an arrives that evening with the kind note find | wer | 0.5385 |
| 1280 | streaming_asr | emilia_zh_0004874290 | 0.7809 | 33 | The book from miss thorn and arrive that evening with the kind note in five | wer | 0.6154 |
| 160 | streaming_asr | emilia_zh_0005000497 | 0.9643 | 7 | 坚持企业所有的债务之后所得的的就是企业价值 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005000497 | 0.9607 | 8 | 鉴于企业所有的债务之后所得的的就是企业价值 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005000497 | 0.9536 | 9 | 减去企业所有的债务之后所得的的就是企业价值 | cer | 0.0952 |
| 1280 | streaming_asr | emilia_zh_0005000497 | 0.9571 | 8 | 减去企业所有的债务之后所得的的就是企业价值 | cer | 0.0952 |
| 160 | causal_full_asr | emilia_zh_0005184596 | 0.9848 | 2 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005184596 | 0.9798 | 3 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005184596 | 0.9798 | 3 | 那也不能对你两只手指这样的姿势有什么批评 | cer | 0.0526 |
| 1280 | causal_full_asr | emilia_zh_0005184596 | 0.9798 | 3 | 那也不能对你两只手指这样的姿势有什么批评 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0005370483 | 0.9725 | 7 | 读了一些呢就这这就就是是是是一个什么感觉就是于龙很大但是从而好像的不的啊 | cer | 0.4878 |
| 320 | streaming_asr | emilia_zh_0005370483 | 0.9680 | 9 | 读了一些呢就就就就就是是是是一个什么感觉就是鱼龙混杂但是从而好像的不的 | cer | 0.3659 |
| 640 | streaming_asr | emilia_zh_0005370483 | 0.9703 | 9 | 读了一些的就就就就是是是是一个什么感觉啊就是鱼龙混搭但是村好像的不的啊 | cer | 0.4390 |
| 1280 | streaming_asr | emilia_zh_0005370483 | 0.9588 | 14 | 独立一些呢就就就就是是是是一个什么感觉呢就是鱼龙混搭但是村好像了不的好 | cer | 0.3171 |
| 160 | streaming_asr | emilia_zh_0005601476 | 1.0000 | 0 | 把继续中不如把心情找古法 | cer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0005601476 | 1.0000 | 0 | 外情绪中不如把心情找古法 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005601476 | 0.9879 | 2 | 外情绪中不如把心情找无法 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005601476 | 0.9939 | 1 | 外情绪中不如把心情找无法 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005852724 | 0.9777 | 4 | 这个算重要的所以所以妈妈我一次希望我的两个女儿可以被 | cer | 0.2143 |
| 320 | causal_full_asr | emilia_zh_0005852724 | 0.9740 | 6 | 这个很重要对所以因为妈妈我一次希望我的两个女儿可以被 | cer | 0.1786 |
| 640 | causal_full_asr | emilia_zh_0005852724 | 0.9665 | 8 | 这个是很重要的所以因为妈妈我一直在希望我的两个女儿可以被 | cer | 0.0714 |
| 1280 | causal_full_asr | emilia_zh_0005852724 | 0.9628 | 9 | 这个是很重要的所以身为妈妈我一直在希望我的两个女儿可以被 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0005853036 | 0.9524 | 11 | 做这个就是如果我想长的话我不排斥啊家里亲戚介绍也也重新没想 | cer | 0.4062 |
| 320 | streaming_asr | emilia_zh_0005853036 | 0.9524 | 12 | 做这个就是如果我想长的话我不排斥啊价值经济介绍点事儿重新没想 | cer | 0.4688 |
| 640 | streaming_asr | emilia_zh_0005853036 | 0.9556 | 10 | 有这个就如果我想长的话我不排斥啊加入进去介绍点事儿不是没想 | cer | 0.4375 |
| 1280 | streaming_asr | emilia_zh_0005853036 | 0.9587 | 10 | 有这个就如果我想找的话我不排斥啊焦虑情绪介绍点哎培训没想 | cer | 0.4688 |
| 160 | streaming_asr | emilia_zh_0006099379 | 0.9821 | 4 | 哦都了你四居然个量然后他他分开使用 | cer | 0.4211 |
| 320 | streaming_asr | emilia_zh_0006099379 | 0.9749 | 5 | 哦都了你四居然个量然后他他分开使用 | cer | 0.4211 |
| 640 | streaming_asr | emilia_zh_0006099379 | 0.9749 | 5 | 哦都了你一次捐捐到一个量然后他分开使用 | cer | 0.1053 |
| 1280 | streaming_asr | emilia_zh_0006099379 | 0.9713 | 5 | 哦懂了你一次捐捐到一个量然后他分开使用 | cer | 0.0526 |
| 160 | causal_full_asr | emilia_zh_0006270577 | 0.9384 | 16 | 用直接用散粉盖其实可以的啊这个研不哪里我们并不是所有的人都需要的 | cer | 0.2121 |
| 320 | causal_full_asr | emilia_zh_0006270577 | 0.9486 | 12 | 用直接用散粉盖其实可以的啊这个研部哪里有并不是所有的人都需要的 | cer | 0.1818 |
| 640 | causal_full_asr | emilia_zh_0006270577 | 0.9486 | 12 | 用直接用散粉盖其实可以的啊这个研部哪里有并不是所有的人都需要的 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006270577 | 0.9418 | 13 | 用直接用散粉盖其实可以的啊这个研部哪里有并不是所有的人都需要的 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006304027 | 0.9452 | 15 | 如果按照我我们刚刚所讲的这个规律来看的话这里我们对应的罗马也是哪一个跟呢 | cer | 0.1143 |
| 320 | streaming_asr | emilia_zh_0006304027 | 0.9395 | 17 | 如果按照我我们刚刚所讲的这个规律来看的话这里我们对应的罗马因为是哪一个跟呢 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0006304027 | 0.9424 | 14 | 如果按照我我们刚刚所讲的这个规律来看的话这里我们对应的罗网引是哪一个哥呢 | cer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0006304027 | 0.9337 | 19 | 如果按照我我们刚刚所讲的这个规律来看的话这里我们对应的罗网引是哪一个哥呢 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0006430098 | 0.8925 | 18 | John was offered Several good job But he wanted to wait and look around | wer | 0.0714 |
| 320 | streaming_asr | emilia_zh_0006430098 | 0.9023 | 18 | Zhang was offered Several good job But he wanted to wait and look around | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0006430098 | 0.8664 | 19 | John was offered Several good job But he wanted to wait and look around | wer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0006430098 | 0.8469 | 21 | John was offered Several good job But he wanted wait and looked around | wer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0006503627 | 0.9749 | 8 | 昔日繁华多会情怀盛进经历如此肆无忌惮的杀杀抢掠后被摧毁打击成为一片废墟 | cer | 0.1944 |
| 320 | streaming_asr | emilia_zh_0006503627 | 0.9791 | 7 | 昔日繁华多会情怀盛敬经历如此肆无忌惮的杀杀抢掠后被摧毁打击成为一片废墟 | cer | 0.1944 |
| 640 | streaming_asr | emilia_zh_0006503627 | 0.9770 | 8 | 昔日繁华多会情怀盛敬敬礼如此肆无忌惮的稍稍强忍后被摧毁打击成为一片废墟 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0006503627 | 0.9791 | 7 | 昔日繁华多会情怀盛敬经历如此肆无忌惮的杀杀抢掠后被摧毁打击成为一片废墟 | cer | 0.1944 |
| 160 | streaming_asr | emilia_zh_0006725267 | 0.9565 | 6 | 无论都是留着也是帮助部分掏出原本 | cer | 0.6842 |
| 320 | streaming_asr | emilia_zh_0006725267 | 0.9503 | 7 | 我来住小榴弹也是帮助部分逃脱员们 | cer | 0.5789 |
| 640 | streaming_asr | emilia_zh_0006725267 | 0.9565 | 6 | 我打算住留的也是帮助部分逃脱员们 | cer | 0.6842 |
| 1280 | streaming_asr | emilia_zh_0006725267 | 0.9689 | 4 | 我来住小刘的也是帮助部分逃脱渔民 | cer | 0.6842 |
| 160 | causal_full_asr | emilia_zh_0006884293 | 0.9607 | 11 | 我不明白这城的继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0857 |
| 320 | causal_full_asr | emilia_zh_0006884293 | 0.9607 | 13 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0006884293 | 0.9558 | 14 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0006884293 | 0.9681 | 11 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007017041 | 0.9883 | 2 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007017041 | 0.9912 | 3 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007017041 | 0.9912 | 3 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007017041 | 0.9883 | 4 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007170191 | 0.9892 | 2 | 这对犹太人来说肯定是不能接受等 | cer | 0.0667 |
| 320 | streaming_asr | emilia_zh_0007170191 | 0.9892 | 2 | 这对犹太人来说肯定是不能接受等 | cer | 0.0667 |
| 640 | streaming_asr | emilia_zh_0007170191 | 0.9946 | 1 | 这对犹太人来说肯定是不能接受等 | cer | 0.0667 |
| 1280 | streaming_asr | emilia_zh_0007170191 | 0.9946 | 1 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007524507 | 0.9955 | 1 | 好我们在来看又在找的这就是用去更改的颜色 | cer | 0.4348 |
| 320 | streaming_asr | emilia_zh_0007524507 | 0.9777 | 5 | 好我们在like can用在的这个就用去更改他颜色 | cer | 0.6957 |
| 640 | streaming_asr | emilia_zh_0007524507 | 0.9688 | 7 | 好我们在来看又在找的这个这中去更改他的颜色 | cer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0007524507 | 0.9777 | 5 | 好我们在来看又下早的这个所龙去更改它颜色 | cer | 0.3478 |
| 160 | streaming_asr | emilia_zh_0007721270 | 0.9938 | 1 | 当我接管家公司是我通常会做两件事儿 | cer | 0.1111 |
| 320 | streaming_asr | emilia_zh_0007721270 | 0.9814 | 3 | 当我接管家公司时我通常会做两件事儿 | cer | 0.0556 |
| 640 | streaming_asr | emilia_zh_0007721270 | 0.9876 | 2 | 当我不管家公司时我通常会做两件事儿 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0007721270 | 0.9938 | 1 | 当我不管家公司时我通常会做两件事儿 | cer | 0.1111 |
| 160 | streaming_asr | EN_B00097_S03849_W000000 | 0.7288 | 74 | If you not death don't and lying then you though the that the American Borswad democracy And Capitalistic civilization other worsened enemies of live and progress | wer | 0.4074 |
| 320 | streaming_asr | EN_B00097_S03849_W000000 | 0.6977 | 81 | If you not death dom and lying then you know the that the American Borswad democracy And Capitalistic civilization other words enemies of live and progress | wer | 0.3704 |
| 640 | streaming_asr | EN_B00097_S03849_W000000 | 0.6894 | 80 | If you not death don't and lying then you know the that the American Borswad democracy and Capitalistic civilization other words enemies of live and progress | wer | 0.3704 |
| 1280 | streaming_asr | EN_B00097_S03849_W000000 | 0.6915 | 79 | If you not death dom and lying then you know the that the American Boiswad democracy And Capitalistic civilization other words enemies of live and progress | wer | 0.3704 |
| 160 | causal_full_asr | EN_B00043_S01557_W000006 | 0.9012 | 12 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 320 | causal_full_asr | EN_B00043_S01557_W000006 | 0.8953 | 14 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 640 | causal_full_asr | EN_B00043_S01557_W000006 | 0.8895 | 13 | Because they were rapidly using up this resource | wer | 0.1250 |
| 1280 | causal_full_asr | EN_B00043_S01557_W000006 | 0.9012 | 12 | Because they were rapidly using up this resource | wer | 0.1250 |
| 160 | streaming_asr | EN_B00064_S08593_W000000 | 0.8549 | 38 | He teacher that the past does not exist Have fact which belongs to to the the sphere of knowledge and which therefore now no one in the world him out | wer | 0.2593 |
| 320 | streaming_asr | EN_B00064_S08593_W000000 | 0.8567 | 40 | He teacher that the path does not exist a fact which belongs to to the the sphere of knowledge And which Therefore not no one in the world can no | wer | 0.2222 |
| 640 | streaming_asr | EN_B00064_S08593_W000000 | 0.8498 | 40 | He teacher that the past does not exist of fact which belongs to to the the sphere of knowledge And which Therefore now one in the the world him no | wer | 0.2963 |
| 1280 | streaming_asr | EN_B00064_S08593_W000000 | 0.8379 | 46 | He teacher that the past does not exist of fact which belongs to to the the sphere of knowledge And which therefore now one in the the world him no | wer | 0.2963 |
| 160 | causal_full_asr | EN_B00048_S03599_W000216 | 0.8286 | 30 | So if for those different categories of use you'll find that the mobile device used slightly differently | wer | 0.2353 |
| 320 | causal_full_asr | EN_B00048_S03599_W000216 | 0.8143 | 36 | So for those different categories of use you'll find that the mobile verbs are used slightly differently | wer | 0.0588 |
| 640 | causal_full_asr | EN_B00048_S03599_W000216 | 0.8029 | 37 | So for those different categories of use you'll find that the mobile apps are used slightly differently | wer | 0.1176 |
| 1280 | causal_full_asr | EN_B00048_S03599_W000216 | 0.8086 | 36 | So for those different categories of use you'll find that the mobile verbs are used slightly differently | wer | 0.0588 |
| 160 | streaming_asr | EN_B00048_S05961_W000019 | 0.8012 | 37 | I just Now use this theory to explain origins of vast of variety of caryotic organisms | wer | 0.4375 |
| 320 | streaming_asr | EN_B00048_S05961_W000019 | 0.7666 | 43 | I just Now use this Theory to explain origins of vast of variety of caryotic organisms | wer | 0.4375 |
| 640 | streaming_asr | EN_B00048_S05961_W000019 | 0.7579 | 43 | I'll just now use this theory to explain origins of vast of variety of caryotic organisms | wer | 0.4375 |
| 1280 | streaming_asr | EN_B00048_S05961_W000019 | 0.7810 | 41 | Alogist now use this theory to explain origins of vast of variety of caryotic organisms | wer | 0.3750 |
| 160 | streaming_asr | EN_B00058_S03808_W000018 | 0.8522 | 60 | We can also say I head gone having two functions in their to indicate a pass perfect as well was the perfect tense with third four So as each and see what inflation does is it changes it actually changes the word | wer | 0.2326 |
| 320 | streaming_asr | EN_B00058_S03808_W000018 | 0.8498 | 64 | We can also say I had gone having two functions in there to indicate a pass perfect as well was the perfect tense with the third four So as you and see what inflation does is it changes it actually changes the word | wer | 0.1395 |
| 640 | streaming_asr | EN_B00058_S03808_W000018 | 0.8367 | 66 | We can also say I had gone having two inflictions in there to indicate a pass perfect as well was the perfect tense with the third four So as you and see what inflation does is it changes it actually changes the word | wer | 0.1395 |
| 1280 | streaming_asr | EN_B00058_S03808_W000018 | 0.8486 | 64 | We can also say I had gone having two and inflictions in there to indicate a pass perfect as well was the perfect tense with the third four So as you and see what inflation does is it changes it actually changes the word | wer | 0.1628 |
| 160 | streaming_asr | EN_B00091_S08343_W000001 | 0.9005 | 24 | Not and mommy came to seremonized my atreatment the gravations sermony | wer | 0.6667 |
| 320 | streaming_asr | EN_B00091_S08343_W000001 | 0.8874 | 24 | But when moments came to seremonized my agreements the gravations ceremony | wer | 0.4167 |
| 640 | streaming_asr | EN_B00091_S08343_W000001 | 0.8822 | 24 | But when moments came to serveys my atculments the gravations ceremony | wer | 0.4167 |
| 1280 | streaming_asr | EN_B00091_S08343_W000001 | 0.8586 | 27 | But when moments came to serveys my atreatment the gravations ceremony | wer | 0.4167 |
| 160 | causal_full_asr | EN_B00083_S08368_W000007 | 0.7966 | 30 | These are not the variables of algebra A variable is any characteristic | wer | 0.0769 |
| 320 | causal_full_asr | EN_B00083_S08368_W000007 | 0.7931 | 31 | And these are not the variables of algebra and variable is any characteristic | wer | 0.0769 |
| 640 | causal_full_asr | EN_B00083_S08368_W000007 | 0.8069 | 30 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00083_S08368_W000007 | 0.7966 | 31 | And these are not the variables of algebra and variable is any characteristic | wer | 0.0769 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
