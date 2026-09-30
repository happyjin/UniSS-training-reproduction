# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9502**
- Weighted CTC blank ratio: **0.7008**
- Weighted streaming WER/CER: **0.1794**
- Weighted causal-full WER/CER: **0.0913**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042263 | 0.7526 | 35 | There remain a the year was spent in court | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000042263 | 0.7211 | 36 | There remain of the year was spent in port | wer | 0.2222 |
| 640 | streaming_asr | CommonVoice_EN_0000042263 | 0.6947 | 39 | That remained of the year was spent in port | wer | 0.2222 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.7053 | 38 | There remain of the year was spent in port | wer | 0.2222 |
| 160 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6867 | 75 | Epidance also suggests that the plant was present decades before its first collection | wer | 0.0769 |
| 320 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6519 | 78 | Epidance also suggests that the plant was present decades before its first collection | wer | 0.0769 |
| 640 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6361 | 83 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6297 | 82 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000237791 | 0.7650 | 32 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 320 | streaming_asr | CommonVoice_EN_0000237791 | 0.7486 | 33 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 640 | streaming_asr | CommonVoice_EN_0000237791 | 0.7322 | 35 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 1280 | streaming_asr | CommonVoice_EN_0000237791 | 0.7377 | 34 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 160 | streaming_asr | CommonVoice_EN_0000352398 | 0.8497 | 35 | He confided his high school from a ran lull all high school | wer | 0.4545 |
| 320 | streaming_asr | CommonVoice_EN_0000352398 | 0.8235 | 38 | He complicated his high school form of ran L all high school | wer | 0.5455 |
| 640 | streaming_asr | CommonVoice_EN_0000352398 | 0.7974 | 41 | He completed his high school from a ram lull all high school | wer | 0.2727 |
| 1280 | streaming_asr | CommonVoice_EN_0000352398 | 0.8301 | 35 | He completed his high school from a random life all high school | wer | 0.3636 |
| 160 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8267 | 26 | Wood is best from making chosen blacks | wer | 0.5000 |
| 320 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8000 | 28 | Wood is best for making chosen blacks | wer | 0.3750 |
| 640 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8133 | 29 | Wood is best for making chosen blacks | wer | 0.3750 |
| 1280 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8000 | 31 | Wood is best for making chosen blacks | wer | 0.3750 |
| 160 | streaming_asr | CommonVoice_EN_0000519794 | 0.6724 | 54 | To the press of unecrization is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000519794 | 0.6638 | 60 | Through the prize of unecrization is unless the dend is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000519794 | 0.6724 | 54 | Give the prize of unacquisition is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000519794 | 0.6595 | 56 | You've prized a in acquisition is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.7500 |
| 160 | streaming_asr | HQ-Conversations_0000041144 | 0.7942 | 52 | 对然后先伸出这个极端的其实我也有一点这种焦虑 | cer | 0.3043 |
| 320 | streaming_asr | HQ-Conversations_0000041144 | 0.7762 | 58 | 对然后下深处这个极端嘛其实我也有点这种焦虑 | cer | 0.3043 |
| 640 | streaming_asr | HQ-Conversations_0000041144 | 0.7762 | 56 | 对然后下深处这个极端嘛其实我也有一点这种焦虑 | cer | 0.2609 |
| 1280 | streaming_asr | HQ-Conversations_0000041144 | 0.7690 | 55 | 对然后下深处这个阶段嘛其实我也有一点这种焦虑 | cer | 0.1739 |
| 160 | streaming_asr | LibriSpeech_0000104521 | 0.6789 | 184 | Until they drew up beside the palace steps And age to wink dressed in the uniform of silver cloth came forward to a sister them to allight said the scarecrow who his personage showed a set once to master the emperor | wer | 0.3250 |
| 320 | streaming_asr | LibriSpeech_0000104521 | 0.6718 | 183 | Until they drew up the beside the palace steps And aged to wink dressed in the uniform of silver cloth came forward to a sister them to allight Said the scarecrow who his personage showed a that once to your master the emperor | wer | 0.3000 |
| 640 | streaming_asr | LibriSpeech_0000104521 | 0.6754 | 182 | Until they drew up the beside the palace steps and aged to wink dressed in the uniform of silver cloth came forward to a sister them to allight said the scarecrow who his personage showed a that once to master the emperor | wer | 0.3250 |
| 1280 | streaming_asr | LibriSpeech_0000104521 | 0.6730 | 178 | Until they drew up the beside the palace steps and aged to wink dressed in the uniform of silver cloth came forward to assist them to allight said the scarecrow who his personage showed us at once to master the emperor | wer | 0.2250 |
| 160 | causal_full_asr | LibriSpeech_0000214472 | 0.7765 | 111 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000214472 | 0.7500 | 113 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000214472 | 0.7206 | 117 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000214472 | 0.7191 | 119 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000265196 | 0.6429 | 175 | I shall be really great if you will say nothing about this There are some in the house and neighborhood who are sitting enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.1111 |
| 320 | streaming_asr | LibriSpeech_0000265196 | 0.6268 | 174 | Are she a really great if you will say nothing about this There are some in the house and neighborhood who are sitting enough and is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.2222 |
| 640 | streaming_asr | LibriSpeech_0000265196 | 0.6224 | 171 | I show be really great if you will say nothing about this there are some in the house and neighborhood who are silly enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.1111 |
| 1280 | streaming_asr | LibriSpeech_0000265196 | 0.6210 | 174 | I shall be really great if you will say nothing about this There are some in the house and neighborhood who are silly enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.0889 |
| 160 | streaming_asr | emilia_zh_0004003103 | 0.6905 | 211 | 所以我就觉得这些可能父母给我的影响会他会影响我但是我没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0805 |
| 320 | streaming_asr | emilia_zh_0004003103 | 0.6697 | 225 | 所以我就觉得之前可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得去反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0690 |
| 640 | streaming_asr | emilia_zh_0004003103 | 0.6580 | 228 | 所以我就觉得之前可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0805 |
| 1280 | streaming_asr | emilia_zh_0004003103 | 0.6632 | 230 | 所以我就觉得这些可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得去反抗什么都去去争取我要都信的我从来都是觉得说我可以很顺其自然自然而然是得到我想要懂行嗯 | cer | 0.1149 |
| 160 | streaming_asr | emilia_zh_0004176058 | 0.8522 | 32 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004176058 | 0.8478 | 35 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004176058 | 0.8391 | 35 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004176058 | 0.8478 | 33 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004213341 | 0.7176 | 67 | 世界上每天都有机器在发生这些奇迹大都来源于精神的力量 | cer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004213341 | 0.7061 | 69 | 世界上每天都有奇迹在发生这些奇迹大多来源于精神的力量 | cer | 0.0385 |
| 640 | causal_full_asr | emilia_zh_0004213341 | 0.6985 | 73 | 世界上每天都有积极的发生这些奇迹大都来源于精神的力量 | cer | 0.1154 |
| 1280 | causal_full_asr | emilia_zh_0004213341 | 0.6870 | 76 | 世界上每天都有奇迹的发生这些奇迹大多来源于精神的力量 | cer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0004422761 | 0.6809 | 44 | This led many twire people to settle into towns and cities | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0004422761 | 0.6489 | 45 | This led many twara people to settle into towns and cities | wer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0004422761 | 0.6543 | 44 | This led many twara people to settle into towns and cities | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0004422761 | 0.6543 | 44 | This led many twara people to settle in towns and cities | wer | 0.0909 |
| 160 | causal_full_asr | emilia_zh_0004665564 | 0.5639 | 155 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acid of features He rolled with an iron hand in a velvet political glove | wer | 0.2286 |
| 320 | causal_full_asr | emilia_zh_0004665564 | 0.5442 | 163 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitative features He ruled with an iron hand in a velvet political glove | wer | 0.2286 |
| 640 | causal_full_asr | emilia_zh_0004665564 | 0.5305 | 165 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitative features He ruled with an iron hand in a velvet political glove | wer | 0.2286 |
| 1280 | causal_full_asr | emilia_zh_0004665564 | 0.5599 | 164 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitating features He rolled with an iron hand in a velvet political glove | wer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0004706082 | 0.6708 | 35 | The quality of looking clever that he was | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0004706082 | 0.6211 | 38 | The quality of looking clever then he was | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004706082 | 0.6087 | 40 | The quality of looking clever then he was | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004706082 | 0.6398 | 38 | The quality of looking clever then he was | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004797638 | 0.8579 | 25 | Why can't kill fire Thank you not quickly Try | wer | 0.5556 |
| 320 | streaming_asr | emilia_zh_0004797638 | 0.8421 | 25 | Why thank you for thank you not quickly Try | wer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004797638 | 0.8263 | 27 | Fine thank you fine Thank you not quickly Try | wer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0004797638 | 0.8053 | 25 | Fine thank you thank thank you not quickly try | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004874290 | 0.6534 | 61 | The book from Mr Thornton arrived that evening with a kind note in Faid | wer | 0.3846 |
| 320 | streaming_asr | emilia_zh_0004874290 | 0.6096 | 64 | The book from Mr Thornton arrived that evening with a kind note in Faid | wer | 0.3846 |
| 640 | streaming_asr | emilia_zh_0004874290 | 0.6135 | 63 | The book from Mr Thornton arrived that evening with a kind note in Fied | wer | 0.3846 |
| 1280 | streaming_asr | emilia_zh_0004874290 | 0.6056 | 64 | The book from Mr Thornton arrived that evening with a kind note in Fied | wer | 0.3846 |
| 160 | streaming_asr | emilia_zh_0005000497 | 0.7679 | 58 | 鉴于企业所有的债务之后所得的就是企业价值 | cer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0005000497 | 0.7571 | 58 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005000497 | 0.7714 | 58 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005000497 | 0.7750 | 55 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005184596 | 0.7071 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005184596 | 0.7071 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005184596 | 0.7020 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005184596 | 0.7020 | 55 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005370483 | 0.7803 | 90 | 独立有些呢就就就就是是是是是一个什么感觉呢就是鱼龙混搭但是这村好像是不是个好 | cer | 0.2683 |
| 320 | streaming_asr | emilia_zh_0005370483 | 0.7551 | 97 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混的当然这村好像的不是个什么 | cer | 0.2439 |
| 640 | streaming_asr | emilia_zh_0005370483 | 0.7712 | 93 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混的当然这村好像的不是个什么 | cer | 0.2439 |
| 1280 | streaming_asr | emilia_zh_0005370483 | 0.7574 | 97 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混搭当然这村好像的不是个什么 | cer | 0.2439 |
| 160 | streaming_asr | emilia_zh_0005601476 | 0.7818 | 31 | 但情绪中不如把心情找过场 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005601476 | 0.7818 | 32 | 把情绪中不如把心情找过场 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005601476 | 0.7879 | 30 | 外情绪中不如把心情找个不好 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005601476 | 0.7879 | 30 | 外情绪中不如把心情找过场 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005852724 | 0.7546 | 59 | 这个算重要的这身为妈妈我一直希望我的两个女儿可以被 | cer | 0.1786 |
| 320 | causal_full_asr | emilia_zh_0005852724 | 0.7361 | 63 | 这个是很重要的所以作为妈妈我一直希望我的两个女儿可以被 | cer | 0.0714 |
| 640 | causal_full_asr | emilia_zh_0005852724 | 0.7212 | 67 | 这个是很重要的所以身为妈妈我一直希望我的两个女儿可以被 | cer | 0.0357 |
| 1280 | causal_full_asr | emilia_zh_0005852724 | 0.7212 | 68 | 这个是很重要的所以身为妈妈我一直希望我的两个女儿可以被 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0005853036 | 0.7397 | 77 | 这个这个就是如果我想找的话我不排斥啊价值经济介绍的也不是没有想 | cer | 0.5312 |
| 320 | streaming_asr | emilia_zh_0005853036 | 0.7079 | 83 | 作为这个就是如果我想找的话我不排斥啊家里亲戚介绍的人不是没有想 | cer | 0.4062 |
| 640 | streaming_asr | emilia_zh_0005853036 | 0.7270 | 76 | 有这个就如果我想找的话我不排斥啊家里亲戚介绍的人不是没想 | cer | 0.3438 |
| 1280 | streaming_asr | emilia_zh_0005853036 | 0.7175 | 80 | 呃这个就如果我想找的话我不排斥啊家里亲戚介绍的人不是没想 | cer | 0.3438 |
| 160 | streaming_asr | emilia_zh_0006099379 | 0.8100 | 47 | 啊等了你一次卷卷到这个量然后他分开使用 | cer | 0.2105 |
| 320 | streaming_asr | emilia_zh_0006099379 | 0.8029 | 50 | 啊等了你一次捐捐到这个量然后他发分开使用 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0006099379 | 0.7885 | 51 | 哦懂了你一次捐捐到这个量然后他把分开使用 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0006099379 | 0.7885 | 54 | 哦懂了你一次捐捐到这个量然后他他分开使用 | cer | 0.0526 |
| 160 | causal_full_asr | emilia_zh_0006270577 | 0.6884 | 84 | 用直接用三粉盖其实可以的啊这个演部哪里我们并不是所有的人都是需要的 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0006270577 | 0.6747 | 83 | 用直接用三份盖其实可以的啊这个演部哪里用并不是所有的人都是需要的 | cer | 0.2121 |
| 640 | causal_full_asr | emilia_zh_0006270577 | 0.6849 | 81 | 用直接用三粉盖其实可以的啊这个演部哪里我们并不是所有的人都是需要的 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006270577 | 0.6712 | 83 | 用直接用三粉盖其实可以的啊这个演部哪里有并不是所有的人都是需要的 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006304027 | 0.7147 | 94 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应的的轮音是哪一个呢 | cer | 0.1143 |
| 320 | streaming_asr | emilia_zh_0006304027 | 0.7003 | 99 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的罗马音是哪一个呢 | cer | 0.0286 |
| 640 | streaming_asr | emilia_zh_0006304027 | 0.6830 | 100 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的轮轮是哪一个呢 | cer | 0.0857 |
| 1280 | streaming_asr | emilia_zh_0006304027 | 0.6945 | 96 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的轮轮是哪一个呢 | cer | 0.0857 |
| 160 | streaming_asr | emilia_zh_0006430098 | 0.7003 | 56 | John was offered several good jobs But he wanted to wait and look around | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 63 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 62 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 64 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006503627 | 0.7975 | 89 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的烧杀抢掠后被摧毁殆尽成为一片废墟 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0006503627 | 0.7871 | 93 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0006503627 | 0.7766 | 99 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 1280 | streaming_asr | emilia_zh_0006503627 | 0.7766 | 98 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 160 | streaming_asr | emilia_zh_0006725267 | 0.6770 | 48 | 老狼猪小刘蛋也是能帮助部分人逃脱怨闷 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006725267 | 0.6522 | 50 | 伯兰登住小刘旦也只能帮助部分人逃出院门 | cer | 0.3684 |
| 640 | streaming_asr | emilia_zh_0006725267 | 0.6584 | 52 | 伯大人出小刘旦也使得帮助部分人逃出院门 | cer | 0.4211 |
| 1280 | streaming_asr | emilia_zh_0006725267 | 0.6522 | 51 | 伯大人住小刘旦也只能帮助部分人逃出院门 | cer | 0.3684 |
| 160 | causal_full_asr | emilia_zh_0006884293 | 0.7543 | 93 | 我不明白这张他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0571 |
| 320 | causal_full_asr | emilia_zh_0006884293 | 0.7420 | 96 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0006884293 | 0.7322 | 96 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0006884293 | 0.7346 | 97 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 67 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007017041 | 0.7683 | 69 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 70 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 70 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007170191 | 0.7419 | 41 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007170191 | 0.7366 | 40 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007170191 | 0.7473 | 41 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007170191 | 0.7473 | 42 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007524507 | 0.7723 | 47 | 好我们在来看用骰子走的这个就是送去更改它的颜色 | cer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0007524507 | 0.7589 | 50 | 好我们在来看用下载的这个搜索我们去更改它的颜色 | cer | 0.2174 |
| 640 | streaming_asr | emilia_zh_0007524507 | 0.7188 | 57 | 好我们在来看用下角的这个这个送去更改它的颜色 | cer | 0.2609 |
| 1280 | streaming_asr | emilia_zh_0007524507 | 0.7098 | 59 | 好我们在来看用下角的这个这个我们去更改它的颜色 | cer | 0.1739 |
| 160 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 46 | 当我接管家公司时我通常会做两件事儿 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 45 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007721270 | 0.7081 | 44 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 44 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 160 | streaming_asr | EN_B00097_S03849_W000000 | 0.5963 | 138 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 320 | streaming_asr | EN_B00097_S03849_W000000 | 0.5735 | 146 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 640 | streaming_asr | EN_B00097_S03849_W000000 | 0.5756 | 140 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00097_S03849_W000000 | 0.5549 | 141 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6686 | 40 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 320 | causal_full_asr | EN_B00043_S01557_W000006 | 0.7035 | 39 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 640 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6453 | 39 | Because they were rapidly using up this resource | wer | 0.1250 |
| 1280 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6512 | 39 | Because they were rapidly using up this resource | wer | 0.1250 |
| 160 | streaming_asr | EN_B00064_S08593_W000000 | 0.7270 | 105 | He teaches that the past does not exist A fact which belongs to the the sphere of knowledge and which therefore no one in the world can not | wer | 0.0741 |
| 320 | streaming_asr | EN_B00064_S08593_W000000 | 0.6980 | 110 | He teaches that the path does not exist A fact which belongs to the the sphere of knowledge and which therefore no one in the world can know | wer | 0.0741 |
| 640 | streaming_asr | EN_B00064_S08593_W000000 | 0.6911 | 114 | He teaches that the past does not exist a fact which belongs to the the sphere of knowledge and which therefore no one in the what world can know | wer | 0.0741 |
| 1280 | streaming_asr | EN_B00064_S08593_W000000 | 0.6928 | 116 | He teaches that the past does not exist a fact which belongs to the the sphere of knowledge and which therefore no one in the one world can know | wer | 0.0741 |
| 160 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6800 | 83 | So for those different categories of use you'll find that that modal verbs are used slightly differently | wer | 0.0588 |
| 320 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6343 | 82 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6229 | 83 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6371 | 79 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S05961_W000019 | 0.6369 | 87 | Alogist now use this theory to explain origin of vast of variety of Eukaryotic organisms | wer | 0.2500 |
| 320 | streaming_asr | EN_B00048_S05961_W000019 | 0.6196 | 90 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 640 | streaming_asr | EN_B00048_S05961_W000019 | 0.6167 | 94 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 1280 | streaming_asr | EN_B00048_S05961_W000019 | 0.6023 | 94 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 160 | streaming_asr | EN_B00058_S03808_W000018 | 0.6913 | 177 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 320 | streaming_asr | EN_B00058_S03808_W000018 | 0.6675 | 185 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 640 | streaming_asr | EN_B00058_S03808_W000018 | 0.6544 | 189 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 1280 | streaming_asr | EN_B00058_S03808_W000018 | 0.6544 | 190 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 160 | streaming_asr | EN_B00091_S08343_W000001 | 0.7644 | 70 | But when the moment came to serenade my atreatment the gravations seremony | wer | 0.3333 |
| 320 | streaming_asr | EN_B00091_S08343_W000001 | 0.7382 | 71 | But when the moment came to serenade my at<\|glm_semantic_224\|>U<\|glm_semantic_4609\|>S there gravations seremony | wer | 0.4167 |
| 640 | streaming_asr | EN_B00091_S08343_W000001 | 0.7382 | 72 | But when the moment came to serenade my attempts the gravations seremony | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00091_S08343_W000001 | 0.7225 | 72 | But when the moment came to serenade my attempts the gravations seremony | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6897 | 66 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6586 | 70 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6828 | 65 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6862 | 63 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
