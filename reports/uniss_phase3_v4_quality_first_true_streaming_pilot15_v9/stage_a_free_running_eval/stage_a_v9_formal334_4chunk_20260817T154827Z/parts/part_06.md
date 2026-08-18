# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8871**
- Weighted CTC blank ratio: **0.1859**
- Weighted streaming WER/CER: **0.4287**
- Weighted causal-full WER/CER: **0.2053**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000159911 | 0.1408 | 14 | The Series see particular claimed during the am tired storyline | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000159911 | 0.1655 | 14 | The Serious severe particular claimed during the am tired storyline | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000159911 | 0.2113 | 17 | The seerious see particular claimed during the amper storyline | wer | 0.5556 |
| 1280 | streaming_asr | CommonVoice_EN_0000159911 | 0.1338 | 12 | The severe severe particular claimed during the amper storyline | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000311292 | 0.1660 | 15 | It had any exception qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000311292 | 0.1830 | 16 | It had any exceptional qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000311292 | 0.1872 | 14 | And had any exceptional qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000311292 | 0.2638 | 17 | A had any exception qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.5556 |
| 160 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2320 | 18 | The soul of the world is nourished by people's happiness. | wer | 0.1000 |
| 320 | causal_full_asr | CommonVoice_EN_0000312479 | 0.3315 | 18 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2762 | 19 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2431 | 17 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000434002 | 0.1074 | 5 | He idea frightened him | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000434002 | 0.1983 | 10 | He idea Frightened him | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000434002 | 0.2149 | 9 | He idea Frightened him | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000434002 | 0.1983 | 11 | He idea Frightened him | wer | 0.2500 |
| 160 | streaming_asr | HQ-Conversations_0000026067 | 0.2685 | 13 | 的还有很多零辜负的我这样的就是在那种打像的用 | cer | 0.5600 |
| 320 | streaming_asr | HQ-Conversations_0000026067 | 0.2824 | 16 | 的还很多灯笼鼓鼓的我这样就是在那种打现在的用 | cer | 0.6000 |
| 640 | streaming_asr | HQ-Conversations_0000026067 | 0.2963 | 15 | 的还很多电动鼓舞的我这样就是在那种大下来了的用 | cer | 0.6000 |
| 1280 | streaming_asr | HQ-Conversations_0000026067 | 0.3102 | 16 | 的还很多landlord辜负的我这样就是在那种大下来的用 | cer | 0.8000 |
| 160 | causal_full_asr | LibriSpeech_0000011154 | 0.1442 | 35 | Other villagers said it was a fine idea, so they stopped working for once and began to plan celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.1471 |
| 320 | causal_full_asr | LibriSpeech_0000011154 | 0.1473 | 41 | Out of the villagers said it was a fine idea. So they stopped working for once and began to plan celebration. They thought that there ought to be swimming races and treefilling contests | wer | 0.2059 |
| 640 | causal_full_asr | LibriSpeech_0000011154 | 0.1473 | 41 | All the villagers said it was a fine idea, so they stopped working for once and began to plan a celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.0882 |
| 1280 | causal_full_asr | LibriSpeech_0000011154 | 0.1240 | 38 | All the villagers said it was a fine idea, so they stopped working for once and began to plan a celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.0882 |
| 160 | streaming_asr | LibriSpeech_0000090398 | 0.1457 | 16 | How eleven how giant will was and of what find of and | wer | 0.5000 |
| 320 | streaming_asr | LibriSpeech_0000090398 | 0.2211 | 16 | How again how giant will was and of but find to and | wer | 0.5833 |
| 640 | streaming_asr | LibriSpeech_0000090398 | 0.2161 | 18 | How elephant how giant she was and of but find of and | wer | 0.5000 |
| 1280 | streaming_asr | LibriSpeech_0000090398 | 0.1859 | 18 | How elephant how gent will was and of but find could and | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000168081 | 0.2588 | 74 | Business getting gawkins disference by baw saw directors considerations of corporator Policy all of which infl the Political market in economist mat some world a use the 结果 careful They formal connosis | wer | 0.5588 |
| 320 | streaming_asr | LibriSpeech_0000168081 | 0.2576 | 72 | Business getting g arguments dissession by baw saw rettest considerations of corporal Policy all of which infl the Political market in economist mat of world a you the results careful They formal connosis | wer | 0.5294 |
| 640 | streaming_asr | LibriSpeech_0000168081 | 0.2336 | 72 | Business getting g arguments dissections by board saw Directors considerations of corporal Policy all of it influenced the Political market in economist met of well a used the results careful They formal connosis | wer | 0.5588 |
| 1280 | streaming_asr | LibriSpeech_0000168081 | 0.2361 | 73 | Business getting g arguments dissections by board saw directors considerations of corporal Policy all of it influence the Political market in economist met some world a used the results careful They formal connosis | wer | 0.5294 |
| 160 | streaming_asr | emilia_zh_0003918097 | 0.1226 | 28 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中丧来说的一个热点的就是就是甚至你们应该 | cer | 0.2553 |
| 320 | streaming_asr | emilia_zh_0003918097 | 0.1459 | 30 | 啊是然后前两天那这是就是他们这个话题成为一个大家中中三来说的一个热点的啊就是就是就是你们应该 | cer | 0.2979 |
| 640 | streaming_asr | emilia_zh_0003918097 | 0.1480 | 30 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中餐来说的一个热点啊就是就是其实你们应该 | cer | 0.2128 |
| 1280 | streaming_asr | emilia_zh_0003918097 | 0.1374 | 26 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中餐来说的一个热点啊就是就是其实你们应该 | cer | 0.2128 |
| 160 | causal_full_asr | emilia_zh_0003940788 | 0.1700 | 26 | 呃然后呢这个因为它更大能帮助注意力同时性行为被盗凶猛就是特有那种王者之气 | cer | 0.2973 |
| 320 | causal_full_asr | emilia_zh_0003940788 | 0.1854 | 29 | 呃然后呢这个因为它更大呃能帮助注意力同时性情也没到胸嘛就是特有那种王者之气 | cer | 0.2973 |
| 640 | causal_full_asr | emilia_zh_0003940788 | 0.1921 | 30 | 呃然后呢这个因为它杠杆大呃能帮助注意力同时性情也被调凶嘛就是特有那种王者之气 | cer | 0.2703 |
| 1280 | causal_full_asr | emilia_zh_0003940788 | 0.2009 | 27 | 呃然后呢这个因为它够大呃能帮助注意力同时性情也被调凶猛就是特有那种王者之气 | cer | 0.2432 |
| 160 | streaming_asr | emilia_zh_0004111570 | 0.1850 | 28 | 我的意思是没有啊地方或者维度是我们被永远因果其中的要那样的地方来干什么呢 | cer | 0.1081 |
| 320 | streaming_asr | emilia_zh_0004111570 | 0.1803 | 23 | 我的意思是没有啊地方或者维度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1351 |
| 640 | streaming_asr | emilia_zh_0004111570 | 0.1780 | 22 | 我的意思是没有啊地方或者维度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1351 |
| 1280 | streaming_asr | emilia_zh_0004111570 | 0.1780 | 24 | 我的意思是没有啊的地方或者度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1622 |
| 160 | streaming_asr | emilia_zh_0004344705 | 0.2339 | 12 | 啊和马克正被一根粗粗树枝缠绕着缠绕着 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0004344705 | 0.2615 | 10 | 啊和mark.正被一根粗粗树枝缠绕着缠绕着 | cer | 0.6471 |
| 640 | streaming_asr | emilia_zh_0004344705 | 0.2294 | 11 | 安娜和mark正被一根粗粗树枝缠绕着缠绕着 | cer | 0.4706 |
| 1280 | streaming_asr | emilia_zh_0004344705 | 0.2477 | 14 | 安娜和mark.正被一根粗粗树枝缠绕着缠绕着 | cer | 0.5294 |
| 160 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 18 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away | wer | 0.3500 |
| 320 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 22 | The parlor had once been two rooms and the floor was swayed back to where the partition had been cut away. | wer | 0.4000 |
| 640 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 24 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away. | wer | 0.4000 |
| 1280 | causal_full_asr | emilia_zh_0004633749 | 0.1928 | 20 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away. | wer | 0.4000 |
| 160 | streaming_asr | emilia_zh_0004633795 | 0.3054 | 24 | We he the I've game a position on the honor on in the flatt last time was the little 棵树 the basis | wer | 0.7500 |
| 320 | streaming_asr | emilia_zh_0004633795 | 0.3188 | 27 | We he the they game my position on the orner on the flatt last time was the little 棵树 the basis | wer | 0.7000 |
| 640 | streaming_asr | emilia_zh_0004633795 | 0.2617 | 24 | We here the they game my position on of corner on the flatt last time was the little 棵树 the basis | wer | 0.6500 |
| 1280 | streaming_asr | emilia_zh_0004633795 | 0.2450 | 25 | We here the I've game my pursuit on of honor a the slat last top with the little 棵树 the basis | wer | 0.7000 |
| 160 | streaming_asr | emilia_zh_0004754844 | 0.1987 | 8 | But stop is class see and and king the is | wer | 0.8182 |
| 320 | streaming_asr | emilia_zh_0004754844 | 0.2185 | 8 | But stop is class see and and king the Earth | wer | 0.7273 |
| 640 | streaming_asr | emilia_zh_0004754844 | 0.1987 | 11 | But stop is class see and and king the Earth | wer | 0.7273 |
| 1280 | streaming_asr | emilia_zh_0004754844 | 0.1921 | 11 | But stop is class see and and king the Earth | wer | 0.7273 |
| 160 | streaming_asr | emilia_zh_0004843191 | 0.1501 | 51 | And four not only European gerographer But European skolars garms all love else Knowledge began to draw maps with space left away a and They be and with the there was not perfect and the the were important things a the not no | wer | 0.5532 |
| 320 | streaming_asr | emilia_zh_0004843191 | 0.1475 | 50 | Hence four not don't European jographers But European skolars garms all love else Knowledge begun to draw maps with space left away a little and They be to with the the was not perfect in the the were important things the the not no | wer | 0.5957 |
| 640 | streaming_asr | emilia_zh_0004843191 | 0.1358 | 46 | And four not on European jographers But European skolars galmers all love else Knowledge begun to draw maps with space left away a little in They be to with the three was not perfect in the the were important things the the not no | wer | 0.5957 |
| 1280 | streaming_asr | emilia_zh_0004843191 | 0.1332 | 49 | And for not only European jographers But European skolars galmers all love else Knowledge begun to draw maps with space left away a and They be to with the three was not perfect in the the were important things the the not no | wer | 0.5745 |
| 160 | streaming_asr | emilia_zh_0004943305 | 0.2019 | 16 | 我们国家层面涟漪历来都如此满满湿包装水啊 | cer | 0.4762 |
| 320 | streaming_asr | emilia_zh_0004943305 | 0.2081 | 15 | 我们国家层面年龄历来都如此麻木施巴增税啊 | cer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0004943305 | 0.1863 | 16 | 我们国家层面年龄历来都如此麻木施巴增税啊 | cer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0004943305 | 0.1832 | 15 | 我们国家层面年龄历来都如此满脸湿吧，脏睡。啊 | cer | 0.5238 |
| 160 | causal_full_asr | emilia_zh_0005058847 | 0.1614 | 28 | 因为死心难起什么就是慈悲的慈悲纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 320 | causal_full_asr | emilia_zh_0005058847 | 0.1651 | 26 | 因为死心难起什么就是慈悲的慈悲纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 640 | causal_full_asr | emilia_zh_0005058847 | 0.1820 | 27 | 因为死心难起怎么就是慈悲的死纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 1280 | causal_full_asr | emilia_zh_0005058847 | 0.1857 | 28 | 因为雌性难企，他妈就是雌配的雌纵然发起男的酒停为什么呢人到自私 | cer | 0.4333 |
| 160 | streaming_asr | emilia_zh_0005244297 | 0.2857 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 320 | streaming_asr | emilia_zh_0005244297 | 0.2762 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 640 | streaming_asr | emilia_zh_0005244297 | 0.3048 | 19 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 1280 | streaming_asr | emilia_zh_0005244297 | 0.3000 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 160 | streaming_asr | emilia_zh_0005507553 | 0.2139 | 13 | William you walk it me the water garden he that softly | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005507553 | 0.2086 | 14 | Will you walk it me the water garden he that softly | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0005507553 | 0.2032 | 13 | Will you walk it me the water garden he that softly | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005507553 | 0.2086 | 13 | Will you walk it me the what garden he that softly | wer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005713628 | 0.2557 | 16 | 大家会连想到中国话大家会想到的是什么名山大川 | cer | 0.0870 |
| 320 | causal_full_asr | emilia_zh_0005713628 | 0.2466 | 16 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 640 | causal_full_asr | emilia_zh_0005713628 | 0.2329 | 14 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 1280 | causal_full_asr | emilia_zh_0005713628 | 0.2420 | 18 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 160 | streaming_asr | emilia_zh_0005781428 | 0.1728 | 12 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005781428 | 0.1885 | 13 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005781428 | 0.1780 | 10 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005781428 | 0.1675 | 10 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0006000255 | 0.2265 | 22 | 因为上次我在旅行天也得在介绍的是一个德国的路线然后我的两位小姐妹一个我说 | cer | 0.1892 |
| 320 | streaming_asr | emilia_zh_0006000255 | 0.2427 | 24 | 因为上次我在旅行天得家这的是一个德国的路线马然后我的两位小姐妹一个我说 | cer | 0.2162 |
| 640 | streaming_asr | emilia_zh_0006000255 | 0.2524 | 28 | 因为上次我在旅行天得家接着的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1892 |
| 1280 | streaming_asr | emilia_zh_0006000255 | 0.2492 | 25 | 因为上次我在旅行天给家接着的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1622 |
| 160 | causal_full_asr | emilia_zh_0006056200 | 0.1541 | 16 | 就你要还是要用一些就是能看得到的东西，要不然你这粪还大啊 | cer | 0.3636 |
| 320 | causal_full_asr | emilia_zh_0006056200 | 0.1601 | 19 | 就你要还是要用一些就是能看得到的东西，要不然你这氛围太大了 | cer | 0.3030 |
| 640 | causal_full_asr | emilia_zh_0006056200 | 0.1722 | 19 | 就你要还是要有一些就是能看得到多少东西，要不然画你这氛围太大了 | cer | 0.3030 |
| 1280 | causal_full_asr | emilia_zh_0006056200 | 0.1601 | 21 | 就你要还是要有一些就是能看得到多少东西要不然画你这氛围太大了 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0006212326 | 0.1359 | 28 | 嗯一员这个的正确是这小老虎就是他一直大家冒着这就是了各地上的王子隐藏的自己的真实身份。 | cer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006212326 | 0.1392 | 28 | 嗯一个人这个的正确是这小老虎就是他一直大家冒着这出了各地上的王子隐藏自的真实身份。 | cer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006212326 | 0.1748 | 33 | 如远远这个真正是这小老虎只是他一直大家冒着这出了个体上的王子隐藏自的真实身份是 | cer | 0.4500 |
| 1280 | streaming_asr | emilia_zh_0006212326 | 0.1845 | 37 | 无一个到的真正是这小老虎就是他一直大家冒着这出了的上的王子隐藏字你真实身份是 | cer | 0.5250 |
| 160 | streaming_asr | emilia_zh_0006366864 | 0.1786 | 15 | I want to continue to make be house a happy hong for him | wer | 0.2308 |
| 320 | streaming_asr | emilia_zh_0006366864 | 0.1696 | 13 | I want it to continue to make believe house a happy hong for him | wer | 0.3077 |
| 640 | streaming_asr | emilia_zh_0006366864 | 0.1429 | 13 | I Wanted to continue to make believe house a happy home home him | wer | 0.1538 |
| 1280 | streaming_asr | emilia_zh_0006366864 | 0.1473 | 14 | And want to continue to make believe house a happy home home him | wer | 0.3077 |
| 160 | streaming_asr | emilia_zh_0006447049 | 0.1837 | 49 | You part the in thing that what well it and contribute the society my take good not to you friend love me and I rode the of are could go to say this this was I to to realize after turn thirty not to on a | wer | 0.6000 |
| 320 | streaming_asr | emilia_zh_0006447049 | 0.1756 | 45 | You part of in thing like what well it and contribute but society my it good not the you friend love me and I rode to of are could go to say this this was day to to realize after turn thirty not to on a | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0006447049 | 0.1837 | 47 | You part of and thing that what well it and contribute but society my it good no the you friend love me and I rode the of are could go to say this this was die to to realize after turn thirty not to on a | wer | 0.6200 |
| 1280 | streaming_asr | emilia_zh_0006447049 | 0.1707 | 48 | You part of in think they what well could and contribute but society my it good not the you friend love me and I rode the of are could go to say this this was die to to realize after turn thirty not to on a | wer | 0.6000 |
| 160 | causal_full_asr | emilia_zh_0006598177 | 0.2366 | 25 | 阿叔这个女性受害人之前案件当中如果跟他的啊五十几岁都是这个亲密关系 | cer | 0.3226 |
| 320 | causal_full_asr | emilia_zh_0006598177 | 0.2390 | 27 | 啊说这个女性受害人这些案件当中，如果我跟她的啊五十几岁都是这个亲密关系 | cer | 0.2581 |
| 640 | causal_full_asr | emilia_zh_0006598177 | 0.2268 | 28 | 啊说这个女性受害人这些案件当中都敢跟他的啊五十几人都是这个亲密关系 | cer | 0.1935 |
| 1280 | causal_full_asr | emilia_zh_0006598177 | 0.2439 | 28 | 啊说这个女性受害人这些案件当中如果跟她的啊五十几岁都是这个亲密关系 | cer | 0.1935 |
| 160 | streaming_asr | emilia_zh_0006658157 | 0.2424 | 34 | 所以我觉得是这样说谁任何品牌在中国办活动他肯定也是因为这一个活动深或者就一个产品本身他在 | cer | 0.2200 |
| 320 | streaming_asr | emilia_zh_0006658157 | 0.2254 | 40 | 这我觉得是这样说所以任何品牌在中国办活动。肯定也是因为在一个活动深或者就也产品本身他在 | cer | 0.2800 |
| 640 | streaming_asr | emilia_zh_0006658157 | 0.2121 | 40 | 最我觉得是这样说所以任何品牌在中国办活动。肯定也是因为在一个活动深或者就也产品本身他在 | cer | 0.2800 |
| 1280 | streaming_asr | emilia_zh_0006658157 | 0.2140 | 37 | 最我觉得是这样说所以任何并在中国办活动<\|write_generate\|><\|cmn\|><\|start_content\|>肯定也是因为在一个活动深或者就也产品本身他在 | cer | 1.1400 |
| 160 | streaming_asr | emilia_zh_0006940399 | 0.1642 | 16 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0006940399 | 0.1343 | 12 | 虽然他们他人都想进去浅显从跌倒的地方爬起来 | cer | 0.2381 |
| 640 | streaming_asr | emilia_zh_0006940399 | 0.1343 | 12 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0006940399 | 0.1194 | 13 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 160 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 21 | 印度所说的空隙风险在一起用最好印度尼西亚也曾经年龄高风险 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007124543 | 0.2016 | 25 | 印度所说的空隙风险在一起用最高印度尼西亚也曾经面临高风险 | cer | 0.2143 |
| 640 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 23 | 印度所说的空风险在一起用最高印度尼西亚也曾经年龄高风险 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 27 | 印度所说的空风险在一起用最高印度尼西亚也曾经面临高风险 | cer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0007399687 | 0.1871 | 12 | 他在一千多年钱许下的落叶知识不去 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0007399687 | 0.1930 | 12 | 他在一千多年钱许下的落叶知识不去 | cer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0007399687 | 0.1871 | 13 | 他他一千多年钱许下的落叶知识不去 | cer | 0.4375 |
| 1280 | streaming_asr | emilia_zh_0007399687 | 0.1813 | 13 | 他他一千多年钱许下的落叶知识不去去 | cer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0007635686 | 0.2044 | 21 | 我们的工作仍然没有实质性进展呢抓来几个小错误 | cer | 0.1364 |
| 320 | streaming_asr | emilia_zh_0007635686 | 0.1887 | 19 | 我们的工作仍然没有实质性进展呢抓了几个小错误 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0007635686 | 0.2075 | 15 | 我的工作仍然没有实质性进展呢抓来几个小错误 | cer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0007635686 | 0.1950 | 16 | 我的工作仍然没有实质性进展呢抓了几个小错误 | cer | 0.1364 |
| 160 | streaming_asr | EN_B00083_S00689_W000013 | 0.1719 | 32 | You you this a using information intended the to then calculate the now in his is it body to for the apartment This the number what need keep low | wer | 0.6000 |
| 320 | streaming_asr | EN_B00083_S00689_W000013 | 0.1652 | 29 | You you this a using information intention the to then calculated the now in the using body to for the apartment This the number what need keep low | wer | 0.5667 |
| 640 | streaming_asr | EN_B00083_S00689_W000013 | 0.1585 | 28 | You you this a using information intitude the to then calculated the now in the using body to the for the apartment This the number and need keep low | wer | 0.6000 |
| 1280 | streaming_asr | EN_B00083_S00689_W000013 | 0.1362 | 29 | You you this the using the information intention the to then calculate the now in is using body to the for the apartment This the number the need keep low | wer | 0.5667 |
| 160 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1408 | 34 | In addition the journal covers signaling networks, synthetic biology systems biology, trying to discovery and computation and modeling of regulatory pathways. | wer | 0.2500 |
| 320 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1354 | 34 | In addition the journal covers signaling networks synthetic biology systems biology joining discovery and computation and modeling of regulatory pathways | wer | 0.0500 |
| 640 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1300 | 33 | In addition the journal covers signaling networks synthetic biology systems biology, joint discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 1280 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1137 | 29 | In addition the journal covers signaling networks, synthetic biology systems biology, joint discovery and computation and modeling of regulatory pathways | wer | 0.1500 |
| 160 | streaming_asr | EN_B00013_S05888_W000041 | 0.1269 | 31 | The I me and of of that I of I of the the to the excess of be big friend chat port ages and seeing in it more make in the a guy you could the go you be that of the | wer | 0.7551 |
| 320 | streaming_asr | EN_B00013_S05888_W000041 | 0.1746 | 37 | The I me and my it that I of I I do to to the besition of be big friend chat port ages and seeing in it more make in the go you could you go you be next of the | wer | 0.7551 |
| 640 | streaming_asr | EN_B00013_S05888_W000041 | 0.1779 | 40 | The I me and my it that I of I I the the to the besition of be big friend chat for ages and seeing in it more make in the guys you could you a guy a be next of the | wer | 0.7143 |
| 1280 | streaming_asr | EN_B00013_S05888_W000041 | 0.1763 | 42 | The I me and of it a I of I I love the to the besition of being big friend chat for ages and seeing in it more make in the a guy he a the a guy a be next of the | wer | 0.7143 |
| 160 | causal_full_asr | EN_B00058_S06426_W000037 | 0.2289 | 21 | It's like this curve here where F here is on the horizontal axis | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00058_S06426_W000037 | 0.2218 | 24 | And let's say this curve here where f here is on the horizontal axis | wer | 0.2143 |
| 640 | causal_full_asr | EN_B00058_S06426_W000037 | 0.1796 | 22 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S06426_W000037 | 0.1479 | 17 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S02307_W000043 | 0.0641 | 18 | Let's check He likes to eep it's a I don't want to you they sally That isn't my shake | wer | 0.3333 |
| 320 | streaming_asr | EN_B00048_S02307_W000043 | 0.0549 | 12 | Let's check He likes to eep each I don't want to he they sal That isn't my shake to | wer | 0.3333 |
| 640 | streaming_asr | EN_B00048_S02307_W000043 | 0.0604 | 16 | Let's check He likes to eep each I don't want to you they sal That Isn't my shake to | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00048_S02307_W000043 | 0.0623 | 18 | Let's check He likes to eep heaths I don't want to he they sally That Isn't my shake to | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1886 | 13 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S01107_W000030 | 0.2057 | 16 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1314 | 13 | The entire project was dropped in my lap after Jason resigned. | wer | 0.0909 |
| 1280 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1371 | 13 | The entire project was dropped in my lap after Jason resigned. | wer | 0.0909 |
| 160 | streaming_asr | EN_B00058_S01128_W000118 | 0.2270 | 9 | As tall for or China I stick out like saw the | wer | 0.4615 |
| 320 | streaming_asr | EN_B00058_S01128_W000118 | 0.2393 | 10 | As tall for or chinese I stick out like saw the | wer | 0.5385 |
| 640 | streaming_asr | EN_B00058_S01128_W000118 | 0.2393 | 9 | As tall for or China I stick out like saw some | wer | 0.4615 |
| 1280 | streaming_asr | EN_B00058_S01128_W000118 | 0.1779 | 10 | As tell for or China a stick out like saw the | wer | 0.6154 |
| 160 | streaming_asr | EN_B00058_S07483_W000027 | 0.2866 | 18 | And is this fix or high irons rogs | wer | 0.5556 |
| 320 | streaming_asr | EN_B00058_S07483_W000027 | 0.3439 | 15 | And use this Fake our high airwards rodden | wer | 0.5556 |
| 640 | streaming_asr | EN_B00058_S07483_W000027 | 0.2548 | 13 | And use this Fake a high airwards robin | wer | 0.4444 |
| 1280 | streaming_asr | EN_B00058_S07483_W000027 | 0.2357 | 12 | And use this face a high airwards robin | wer | 0.4444 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
