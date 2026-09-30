# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9526**
- Weighted CTC blank ratio: **0.6948**
- Weighted streaming WER/CER: **0.1834**
- Weighted causal-full WER/CER: **0.1397**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000159911 | 0.7711 | 44 | The series resears particular claimed during the vamire storylines | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000159911 | 0.7430 | 49 | The serious rec seizure particular claim during the vampire storylines | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000159911 | 0.7430 | 50 | The serious recused particular clame during the vampire storylines | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000159911 | 0.7148 | 54 | The serious re seizure particular claim during the the vampire storylines | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000311292 | 0.7064 | 50 | In head many exceptional qualities compared to previous aircraft | wer | 0.2222 |
| 320 | streaming_asr | CommonVoice_EN_0000311292 | 0.6723 | 53 | In head many exceptional qualities compared to previous aircraft | wer | 0.2222 |
| 640 | streaming_asr | CommonVoice_EN_0000311292 | 0.6809 | 52 | It had many exceptional qualities compared to previous aircraft | wer | 0.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000311292 | 0.6766 | 57 | It had many exceptional qualities compared to previous aircraft | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6906 | 41 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6519 | 46 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6243 | 48 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6133 | 52 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000434002 | 0.8099 | 17 | He idea frightened him | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000434002 | 0.7686 | 19 | He idea frightened him | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000434002 | 0.7603 | 19 | He idea frightened him | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000434002 | 0.7686 | 19 | He idea frightened him | wer | 0.2500 |
| 160 | streaming_asr | HQ-Conversations_0000026067 | 0.7546 | 45 | 对还有很多内容服务的内容你就是在那种打线上的内容嗯 | cer | 0.5600 |
| 320 | streaming_asr | HQ-Conversations_0000026067 | 0.7500 | 48 | 对还有很多内容古佛的那种印度就是在那种大山羊的那种嗯 | cer | 0.2800 |
| 640 | streaming_asr | HQ-Conversations_0000026067 | 0.7639 | 46 | 对还有很多内容鼓舞的内容印度就是在那种大山羊的运动呢 | cer | 0.5200 |
| 1280 | streaming_asr | HQ-Conversations_0000026067 | 0.6944 | 60 | 但还有很多那种古法的那种建筑就是在那种搭山羊的那种嗯 | cer | 0.2000 |
| 160 | causal_full_asr | LibriSpeech_0000011154 | 0.6419 | 158 | Other villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.1471 |
| 320 | causal_full_asr | LibriSpeech_0000011154 | 0.6202 | 167 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 640 | causal_full_asr | LibriSpeech_0000011154 | 0.6295 | 159 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 1280 | causal_full_asr | LibriSpeech_0000011154 | 0.6202 | 161 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 160 | streaming_asr | LibriSpeech_0000090398 | 0.6583 | 46 | How elegant how gentle she was and of what refined good manners the | wer | 0.0833 |
| 320 | streaming_asr | LibriSpeech_0000090398 | 0.6583 | 47 | How elegant how gentle she was and of what refined good manners the | wer | 0.0833 |
| 640 | streaming_asr | LibriSpeech_0000090398 | 0.6181 | 50 | How elegant how gentle she was and of what refined good manners | wer | 0.0000 |
| 1280 | streaming_asr | LibriSpeech_0000090398 | 0.6281 | 51 | How elegant how gentle she was and of what refined good manners | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000168081 | 0.6591 | 190 | Business getting arguments discernance by boards of directors considerations of corporate policy all of which influenced the political American and economic maps of the world I use really results of careful though in formal conversation | wer | 0.2353 |
| 320 | streaming_asr | LibriSpeech_0000168081 | 0.6187 | 209 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world I use the results of careful though in formal conversation | wer | 0.1471 |
| 640 | streaming_asr | LibriSpeech_0000168081 | 0.6149 | 209 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world a use real results of careful though in formal conversation | wer | 0.1765 |
| 1280 | streaming_asr | LibriSpeech_0000168081 | 0.5934 | 213 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world a usually the results of careful though in formal conversation | wer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0003918097 | 0.7019 | 126 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 320 | streaming_asr | emilia_zh_0003918097 | 0.7061 | 127 | 哦是然后前两天呢这是就是为什么这个话题成为了一个大家争争三来说的一个热点呢就是这是其实你们应该 | cer | 0.0638 |
| 640 | streaming_asr | emilia_zh_0003918097 | 0.7167 | 122 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 1280 | streaming_asr | emilia_zh_0003918097 | 0.7146 | 125 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 160 | causal_full_asr | emilia_zh_0003940788 | 0.7815 | 92 | 呃然后呢这个因为它很大能帮助注意力同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1892 |
| 320 | causal_full_asr | emilia_zh_0003940788 | 0.7726 | 95 | 呃然后呢这个因为它很大呃能帮助助猎同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1351 |
| 640 | causal_full_asr | emilia_zh_0003940788 | 0.7726 | 93 | 呃然后呢这个因为它很大呃能帮助助练同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1351 |
| 1280 | causal_full_asr | emilia_zh_0003940788 | 0.7638 | 94 | 呃然后呢这个因为它很大呃能帮助助练同时性情也比较凶猛就是特有那种王者之气 | cer | 0.1081 |
| 160 | streaming_asr | emilia_zh_0004111570 | 0.7541 | 101 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004111570 | 0.7377 | 102 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004111570 | 0.7400 | 101 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004111570 | 0.7377 | 102 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004344705 | 0.8257 | 34 | 啊和mark正被一根粗粗的树枝缠绕着 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0004344705 | 0.7982 | 42 | 啊和马克正被一根粗粗的树枝缠绕着 | cer | 0.1176 |
| 640 | streaming_asr | emilia_zh_0004344705 | 0.7844 | 43 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004344705 | 0.7661 | 47 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004633749 | 0.5743 | 82 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 320 | causal_full_asr | emilia_zh_0004633749 | 0.5060 | 87 | The parlor had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 640 | causal_full_asr | emilia_zh_0004633749 | 0.5382 | 84 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 1280 | causal_full_asr | emilia_zh_0004633749 | 0.5261 | 84 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004633795 | 0.5940 | 88 | We have a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3500 |
| 320 | streaming_asr | emilia_zh_0004633795 | 0.5436 | 86 | We have a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3500 |
| 640 | streaming_asr | emilia_zh_0004633795 | 0.5436 | 86 | We had a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3000 |
| 1280 | streaming_asr | emilia_zh_0004633795 | 0.5537 | 87 | We had a lively game of prosy ones of corner on the flat plustop with the little tree for basis | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004754844 | 0.6358 | 40 | Five stuff these clouds here and then came to the earth | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0004754844 | 0.6093 | 42 | Five stuff these clives here and then came to the earth | wer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0004754844 | 0.6291 | 40 | Five stuff these clouds here and then came to the earth | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0004754844 | 0.5960 | 43 | Found stuff these clouds here and then came to the earth | wer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0004843191 | 0.6110 | 209 | Hence forth not only European drug refers but European scholars almost all over else of knowledge began to draw maps with spaces left to hello and they began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.1702 |
| 320 | streaming_asr | emilia_zh_0004843191 | 0.5718 | 221 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello and they began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.1064 |
| 640 | streaming_asr | emilia_zh_0004843191 | 0.5718 | 222 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello in They began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.0851 |
| 1280 | streaming_asr | emilia_zh_0004843191 | 0.5483 | 228 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello in They began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.0851 |
| 160 | streaming_asr | emilia_zh_0004943305 | 0.8385 | 50 | 我们国家层面面临历来都如此勉马十八周岁 | cer | 0.3810 |
| 320 | streaming_asr | emilia_zh_0004943305 | 0.8230 | 54 | 我们国家层面面临历来都如此勉马十八周岁啊 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004943305 | 0.8230 | 55 | 我们国家层面面临历来都如此勉马十八周岁啊 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004943305 | 0.8292 | 50 | 我们国家层面面临历来都如此勉蛮十八周岁啊 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005058847 | 0.8443 | 74 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 320 | causal_full_asr | emilia_zh_0005058847 | 0.8386 | 79 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 640 | causal_full_asr | emilia_zh_0005058847 | 0.8330 | 81 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 1280 | causal_full_asr | emilia_zh_0005058847 | 0.8368 | 80 | 因为磁性难起磁嘛就是磁碑的磁纵然发起难得久停为什么呢人道自私 | cer | 0.2333 |
| 160 | streaming_asr | emilia_zh_0005244297 | 0.6857 | 63 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005244297 | 0.6714 | 66 | 总得做点不一样的事情吧打电话报警还报出了<\|write_generate\|><\|cmn\|><\|start_content\|>车牌号 | cer | 1.8261 |
| 640 | streaming_asr | emilia_zh_0005244297 | 0.6714 | 67 | 总得做点不一样的事情吧打电话报警还爆出了车牌号 | cer | 0.0435 |
| 1280 | streaming_asr | emilia_zh_0005244297 | 0.6762 | 66 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005507553 | 0.6524 | 41 | Will you walk with me in the water garden He said softly | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005507553 | 0.6203 | 44 | Will you walk with me in the water garden He said softly | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005507553 | 0.5829 | 49 | Will you walk with me in the water garden he said softly | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005507553 | 0.5829 | 49 | Will you walk with me in the water garden he said softly | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005713628 | 0.7123 | 61 | 大家会联想到中国的话大家会想到的是什么名山大川 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005713628 | 0.7032 | 61 | 大家会联想到中国的话大家会想到的是什么名山大川 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005713628 | 0.7123 | 59 | 大家会联想到中国化大家会想到的是什么名山大川 | cer | 0.0870 |
| 1280 | causal_full_asr | emilia_zh_0005713628 | 0.7032 | 60 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 160 | streaming_asr | emilia_zh_0005781428 | 0.7644 | 39 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005781428 | 0.7277 | 43 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005781428 | 0.7173 | 45 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005781428 | 0.7277 | 41 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0006000255 | 0.6667 | 93 | 因为上次我在旅行天天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.1081 |
| 320 | streaming_asr | emilia_zh_0006000255 | 0.6408 | 97 | 因为上次我在旅行天天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.1081 |
| 640 | streaming_asr | emilia_zh_0006000255 | 0.6278 | 102 | 因为上次我在旅行天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.0811 |
| 1280 | streaming_asr | emilia_zh_0006000255 | 0.6343 | 99 | 因为上次我在旅行天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.0811 |
| 160 | causal_full_asr | emilia_zh_0006056200 | 0.7553 | 69 | 就是你又还是要用一些就是能看不掉的东西要不然的话你这氛围太大了 | cer | 0.3030 |
| 320 | causal_full_asr | emilia_zh_0006056200 | 0.7311 | 78 | 就你又还是要用一些就是能看不掉的东西要不然的话你这复印太大了 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0006056200 | 0.7281 | 80 | 就你又还是要有一些就是能看不掉的东西要不然的话你这风险太大了 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006056200 | 0.7130 | 79 | 就你又还是要有一些就是能看的更多的东西要不然的话你的风险还大了 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0006212326 | 0.8220 | 104 | 嗯远远这个个真正是这小老虎只是他一直在冒着追逐了个上的王子隐藏着自己的真实身份 | cer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0006212326 | 0.8074 | 106 | 嗯远远这个个真正是这小老虎就是他一直在冒着追逐了个上的王子隐藏的自己的真实身份 | cer | 0.4250 |
| 640 | streaming_asr | emilia_zh_0006212326 | 0.8091 | 106 | 路远远这个个真正是这小老虎只是他一直带着帽子伸出了个头上的王子隐藏的自己的真实身份 | cer | 0.3250 |
| 1280 | streaming_asr | emilia_zh_0006212326 | 0.8155 | 99 | 如原来这个个真正是这小老虎只是他一直在冒着你这出的恶作剧上的王子隐藏的自己的真实身份 | cer | 0.4000 |
| 160 | streaming_asr | emilia_zh_0006366864 | 0.7277 | 41 | I wanted to continue to make bleak house I happy home for him | wer | 0.0769 |
| 320 | streaming_asr | emilia_zh_0006366864 | 0.7098 | 44 | I wanted to continue to make bleak house I happy home for him | wer | 0.0769 |
| 640 | streaming_asr | emilia_zh_0006366864 | 0.6652 | 46 | I wanted to continue to make bleak house a happy home for him | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006366864 | 0.6652 | 47 | I wanted to continue to make bleak house a happy home for him | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006447049 | 0.5772 | 189 | You know part of him thanks like what what look can I contribute to society and my like good enough Do my friends love me and I are wrote a mother very quick note say Just as what I take to realize after I turned 30 not too long ago | wer | 0.2800 |
| 320 | streaming_asr | emilia_zh_0006447049 | 0.5772 | 190 | You know part of him thinks like what else can I contribute the society and my like good enough Do my friends love me and I are wrote a very good quick note say this is a what I pay to realize after I turn 30 not through long ago | wer | 0.3200 |
| 640 | streaming_asr | emilia_zh_0006447049 | 0.5561 | 191 | You know part of him thinks like what else can I contribute to society and my good enough Do my friends love me and I are wrote a very good quick note to say this is a what I pay to realize after I turn 30 not too long ago | wer | 0.2600 |
| 1280 | streaming_asr | emilia_zh_0006447049 | 0.5593 | 190 | You know part of him thinks like what else can I contribute to society and my right good enough Do my friends love me and I are wrote a very very quick note say This is a what I take to realize after I turn 30 not too long ago | wer | 0.3000 |
| 160 | causal_full_asr | emilia_zh_0006598177 | 0.8000 | 73 | 他说这个女性受害人这些案件当中都百分之五十几都是这个亲密关系 | cer | 0.0968 |
| 320 | causal_full_asr | emilia_zh_0006598177 | 0.7780 | 78 | 而说这个女性受害人这些案件当中有百分之五十级都是这个亲密关系 | cer | 0.0968 |
| 640 | causal_full_asr | emilia_zh_0006598177 | 0.7829 | 80 | 啊说这个女性受害人这些案件当中有百分之五十级都是这个亲密关系 | cer | 0.0645 |
| 1280 | causal_full_asr | emilia_zh_0006598177 | 0.7854 | 80 | 啊说这个女性受害人这些案件当中有百分之啊五十级都是这个亲密关系 | cer | 0.0323 |
| 160 | streaming_asr | emilia_zh_0006658157 | 0.7292 | 127 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这一个产品本身它在 | cer | 0.0600 |
| 320 | streaming_asr | emilia_zh_0006658157 | 0.7254 | 126 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.0800 |
| 640 | streaming_asr | emilia_zh_0006658157 | 0.7197 | 129 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.0800 |
| 1280 | streaming_asr | emilia_zh_0006658157 | 0.7140 | 131 | 对我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.1200 |
| 160 | streaming_asr | emilia_zh_0006940399 | 0.8134 | 45 | 虽然他们爱人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0006940399 | 0.8172 | 46 | 虽然他们爱人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0952 |
| 640 | streaming_asr | emilia_zh_0006940399 | 0.7836 | 54 | 虽然他们二人都想进去浅显从跌倒的地方爬起来 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0006940399 | 0.7799 | 55 | 虽然他们二人都想进去浅显从跌倒的地方爬起来 | cer | 0.1905 |
| 160 | streaming_asr | emilia_zh_0007124543 | 0.7827 | 74 | 印度所受的恐吓风险在一期中最好印度尼西亚也曾经面临高风险 | cer | 0.1071 |
| 320 | streaming_asr | emilia_zh_0007124543 | 0.7775 | 73 | 印度所述的空隙风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007124543 | 0.7853 | 71 | 印度所受的恐袭风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007124543 | 0.7827 | 72 | 印度所受的恐袭风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007399687 | 0.7076 | 45 | 他在一千多年前许下的诺言知识不需 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0007399687 | 0.7135 | 44 | 他在一千多年前许下的落言知识不需 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0007399687 | 0.7193 | 44 | 他在一千多年前许下的落言真实不需 | cer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0007399687 | 0.7193 | 43 | 他在一千多年前许下的落言真实不需 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0007635686 | 0.7925 | 60 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007635686 | 0.7862 | 57 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007635686 | 0.7925 | 56 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007635686 | 0.7799 | 62 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 160 | streaming_asr | EN_B00083_S00689_W000013 | 0.5714 | 136 | You will see that using information entered the tool then calculates the net and as you use it budget all out for the apartment This of the number we need to keep below | wer | 0.3000 |
| 320 | streaming_asr | EN_B00083_S00689_W000013 | 0.5357 | 149 | You will say the you using information entered the tool then calculates the net energy used budget all around for apartment This of the number we need to keep below | wer | 0.3000 |
| 640 | streaming_asr | EN_B00083_S00689_W000013 | 0.5335 | 153 | You will see the you using information entered the tool then calculates the net energy usage budget allowed to for apartment This of the number we need to keep below | wer | 0.2333 |
| 1280 | streaming_asr | EN_B00083_S00689_W000013 | 0.5223 | 156 | You will see the you using information entered the tool then calculates the net energy usage budget allowed for apartment This of the number we need to keep below | wer | 0.2000 |
| 160 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6588 | 140 | In addition the journal covers signaling networks synthetic biology systems biology draw it discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 320 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6390 | 145 | In addition the journal covers signaling networks synthetic biology systems biology draw it discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 640 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6516 | 136 | In addition the journal covers signaling networks synthetic biology systems biology draw data discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 1280 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6606 | 134 | In addition the journal covers signaling networks synthetic biology systems biology draw data discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 160 | streaming_asr | EN_B00013_S05888_W000041 | 0.6952 | 142 | But I've mean I have a big friend I love I don't love to see each other and absession I've been big friend chad's fort ages and say he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3878 |
| 320 | streaming_asr | EN_B00013_S05888_W000041 | 0.6771 | 148 | That I've mean I'm a big person I love I don't love see each other and absession I've been big friend chad's fort ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3265 |
| 640 | streaming_asr | EN_B00013_S05888_W000041 | 0.6540 | 155 | The I've mean I'm a big child I love I don't love to see each other opsession I've been big friend chad's four ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3061 |
| 1280 | streaming_asr | EN_B00013_S05888_W000041 | 0.6639 | 148 | The I've mean I'm a big child I love I I love to see each other opinion I've been big friend chad's four ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3061 |
| 160 | causal_full_asr | EN_B00058_S06426_W000037 | 0.7218 | 53 | It looks like this curve here where f here is on the horizontal axis | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6972 | 52 | And let's say this curve here where f here is on the horizontal axis | wer | 0.2143 |
| 640 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6761 | 55 | And looks like this curve here where f here is on the horizontal axis | wer | 0.0714 |
| 1280 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6831 | 54 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S02307_W000043 | 0.8223 | 66 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 320 | streaming_asr | EN_B00048_S02307_W000043 | 0.8278 | 59 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 640 | streaming_asr | EN_B00048_S02307_W000043 | 0.8242 | 61 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 1280 | streaming_asr | EN_B00048_S02307_W000043 | 0.8168 | 65 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 160 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5943 | 57 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5714 | 55 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5943 | 52 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6171 | 51 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S01128_W000118 | 0.6135 | 45 | As a tough foreign or in China I stick out like a sort them | wer | 0.3846 |
| 320 | streaming_asr | EN_B00058_S01128_W000118 | 0.6012 | 47 | As a tall foreign or in China I stick out like a sort them | wer | 0.3077 |
| 640 | streaming_asr | EN_B00058_S01128_W000118 | 0.5828 | 46 | As a tall foreign and China I stick out like a sort them | wer | 0.3077 |
| 1280 | streaming_asr | EN_B00058_S01128_W000118 | 0.5828 | 45 | As a tall foreign and China I stick out like a sort them | wer | 0.3077 |
| 160 | streaming_asr | EN_B00058_S07483_W000027 | 0.8153 | 25 | And use the fix while high virulence problem | wer | 0.4444 |
| 320 | streaming_asr | EN_B00058_S07483_W000027 | 0.8089 | 25 | And use this to fix while high virulence problem | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S07483_W000027 | 0.7452 | 27 | And use this to fix a high variance problem | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00058_S07483_W000027 | 0.7197 | 31 | And use this to fix a high variance problem | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
