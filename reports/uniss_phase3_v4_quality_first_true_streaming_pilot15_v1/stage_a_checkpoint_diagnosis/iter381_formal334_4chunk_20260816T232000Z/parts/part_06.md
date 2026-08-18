# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9221**
- Weighted CTC blank ratio: **0.8623**
- Weighted streaming WER/CER: **0.2507**
- Weighted causal-full WER/CER: **0.1616**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000159911 | 0.7993 | 31 | The seers seers particular claimed during the bamfire storyline | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000159911 | 0.7887 | 33 | The Serious seize particular claimed during the bamfire storyline | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000159911 | 0.7887 | 29 | The serious seized prudicial claimed during the bamper storyline | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000159911 | 0.7887 | 32 | The serious seized prudicial claimed during the the bamper storyline | wer | 0.7778 |
| 160 | streaming_asr | CommonVoice_EN_0000311292 | 0.7915 | 26 | A had many exceptional qualities compared to previous aircraft | wer | 0.1111 |
| 320 | streaming_asr | CommonVoice_EN_0000311292 | 0.7745 | 29 | It had a many exceptional qualities compared to previous aircraft | wer | 0.1111 |
| 640 | streaming_asr | CommonVoice_EN_0000311292 | 0.7660 | 27 | It helped many exceptional qualities compared to previous aircraft | wer | 0.1111 |
| 1280 | streaming_asr | CommonVoice_EN_0000311292 | 0.7404 | 30 | It held many exceptional qualities compared to previous aircraft | wer | 0.1111 |
| 160 | causal_full_asr | CommonVoice_EN_0000312479 | 0.8122 | 17 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000312479 | 0.7680 | 20 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000312479 | 0.7569 | 19 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000312479 | 0.7459 | 22 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000434002 | 0.8926 | 7 | He idea frightened him | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000434002 | 0.8843 | 8 | He idea frightened him | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000434002 | 0.8843 | 8 | He idea frightened him | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000434002 | 0.8760 | 8 | He idea frightened him | wer | 0.2500 |
| 160 | streaming_asr | HQ-Conversations_0000026067 | 0.9444 | 9 | 的还有很多那种舞的我见就是在那种打想的用 | cer | 0.4800 |
| 320 | streaming_asr | HQ-Conversations_0000026067 | 0.9583 | 6 | 对还有很多内容古佛的我们印度就是在那种打下的用 | cer | 0.4800 |
| 640 | streaming_asr | HQ-Conversations_0000026067 | 0.8843 | 18 | 到还有很多那种舞服的我们家务就是在那种打想要的扭 | cer | 0.4800 |
| 1280 | streaming_asr | HQ-Conversations_0000026067 | 0.9398 | 10 | 到还有很多那种古腐的那种建筑就是在那种大下降的运用 | cer | 0.2400 |
| 160 | causal_full_asr | LibriSpeech_0000011154 | 0.7705 | 70 | Other villagers said it was a fine idea so they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.1471 |
| 320 | causal_full_asr | LibriSpeech_0000011154 | 0.7535 | 75 | Other villagers said it was a fine idea so they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.1471 |
| 640 | causal_full_asr | LibriSpeech_0000011154 | 0.7457 | 80 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 1280 | causal_full_asr | LibriSpeech_0000011154 | 0.7426 | 81 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 160 | streaming_asr | LibriSpeech_0000090398 | 0.7739 | 26 | How elephant how gentle shill was and of what find good manners | wer | 0.2500 |
| 320 | streaming_asr | LibriSpeech_0000090398 | 0.7789 | 25 | How eleventh how gentle she was and of what find good manners | wer | 0.1667 |
| 640 | streaming_asr | LibriSpeech_0000090398 | 0.7940 | 22 | How elephant how gentle she was and of what find good manners | wer | 0.1667 |
| 1280 | streaming_asr | LibriSpeech_0000090398 | 0.7588 | 26 | How eligant how gentle she was and of what find good manners | wer | 0.1667 |
| 160 | streaming_asr | LibriSpeech_0000168081 | 0.8081 | 91 | Business getting argments discisions by boards of directors considerations of corporate policy all of which influenced the Political American thought and economic met of world I you will results of careful They informal Conversation | wer | 0.3235 |
| 320 | streaming_asr | LibriSpeech_0000168081 | 0.7576 | 106 | Business getting arguments discisions by boards of directors considerations of corporate Policy all of which influenced the Political American thought and economic met of world I you the results of careful They informal Conversation | wer | 0.2647 |
| 640 | streaming_asr | LibriSpeech_0000168081 | 0.7500 | 107 | Business getting arguments discisions by boards of directors considerations of corporate Policy all of which influenced the Political American and economic met of world I you the results of careful though informal Conversation | wer | 0.2059 |
| 1280 | streaming_asr | LibriSpeech_0000168081 | 0.7424 | 111 | Business getting arguments dissection of by boards of directors considerations of corporat policy all of which influenced the Political American and economic met of world I you the results of careful though informal Conversation | wer | 0.2647 |
| 160 | streaming_asr | emilia_zh_0003918097 | 0.9302 | 30 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家中周三来说的一个热点呢就是就是其实你们应该 | cer | 0.0638 |
| 320 | streaming_asr | emilia_zh_0003918097 | 0.9408 | 24 | 方是然后前两天呢就是就是为什么这个话题成为了一个大家中中间来说的一个热点呢呢就是就是其实你们应该 | cer | 0.1064 |
| 640 | streaming_asr | emilia_zh_0003918097 | 0.9450 | 22 | 然后是然后前两天呢就是就是为什么这个话题成为了一个大家中中间来说的一个热点呢呢就是就是其实你们应该 | cer | 0.1277 |
| 1280 | streaming_asr | emilia_zh_0003918097 | 0.9387 | 23 | 啊是然后前两天呢就是就是为什么这个话题成为了一个大家中中间来说的一个热点呢就是就是其实你们应该 | cer | 0.0851 |
| 160 | causal_full_asr | emilia_zh_0003940788 | 0.9868 | 6 | 呃然后呢这个因为它更大呃能帮助独立同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1892 |
| 320 | causal_full_asr | emilia_zh_0003940788 | 0.9845 | 6 | 呃然后呢这个因为它大呃能帮助助恋同时性情也没掉凶猛就是特有那种王者之气 | cer | 0.1892 |
| 640 | causal_full_asr | emilia_zh_0003940788 | 0.9845 | 6 | 呃然后呢这个因为它个很大呃能帮助助恋同时性情也比较凶猛就是特有那种王者之气 | cer | 0.0811 |
| 1280 | causal_full_asr | emilia_zh_0003940788 | 0.9801 | 6 | 呃然后呢这个因为它个很大呃能帮助助恋同时性情也比较凶猛就是特有那种王者之气 | cer | 0.0811 |
| 160 | streaming_asr | emilia_zh_0004111570 | 0.9789 | 7 | 我的意思是没有拿地方或者维度是我们被永远因果其中的要那样的地方来干什么呢 | cer | 0.1081 |
| 320 | streaming_asr | emilia_zh_0004111570 | 0.9766 | 7 | 我的意思是没有哪个地方或者维度是我们被永远巩固其中的要那样的地方来干什么呢 | cer | 0.0541 |
| 640 | streaming_asr | emilia_zh_0004111570 | 0.9719 | 9 | 我的意思是没有哪个地方或者维度是我们被永远巩固其中的要那样的地方来干什么呢 | cer | 0.0541 |
| 1280 | streaming_asr | emilia_zh_0004111570 | 0.9742 | 10 | 我的意思是没有哪个地方或者维度是我们被永远巩固其中的要那样的地方来干什么呢 | cer | 0.0541 |
| 160 | streaming_asr | emilia_zh_0004344705 | 0.9817 | 4 | 啊和mark正被一根粗粗的树枝缠绕着 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0004344705 | 0.9771 | 4 | 啊和mark正被一根粗粗的树枝缠绕着 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0004344705 | 0.9862 | 2 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004344705 | 0.9817 | 3 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004633749 | 0.6145 | 51 | The parlor had once been two rooms and the floor was swayed back to where their partition had been cut away | wer | 0.3500 |
| 320 | causal_full_asr | emilia_zh_0004633749 | 0.5783 | 57 | The parlor had once been two rooms and the floor was swayed back to where their partition had been cut away | wer | 0.3500 |
| 640 | causal_full_asr | emilia_zh_0004633749 | 0.5582 | 55 | The parlor had once been two rooms and the floor was swayed back where their partition had been cut away | wer | 0.3000 |
| 1280 | causal_full_asr | emilia_zh_0004633749 | 0.5703 | 54 | The pilot had once been two rooms and the floor was swayed back where their partition had been cut away | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004633795 | 0.8020 | 35 | We had a lively game a prosy one a corner on the flatt left top with it little tree for basis | wer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0004633795 | 0.7651 | 36 | We have a lively game of prosy one a corner on the flatt blust top with it's little trues for basis | wer | 0.4000 |
| 640 | streaming_asr | emilia_zh_0004633795 | 0.7450 | 43 | We had a lively game of prosy one a corner on the flatt blust top with this little trues for basis | wer | 0.3500 |
| 1280 | streaming_asr | emilia_zh_0004633795 | 0.7517 | 47 | We had a lively game of prosy on a corner on the flap plus top with the little trues for basis | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004754844 | 0.8079 | 19 | I stop these clives here and and came to the Earth | wer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0004754844 | 0.7550 | 24 | Babs stop these clives here and and came to the Earth | wer | 0.3636 |
| 640 | streaming_asr | emilia_zh_0004754844 | 0.7417 | 23 | five stop these clads here and and king to the Earth | wer | 0.4545 |
| 1280 | streaming_asr | emilia_zh_0004754844 | 0.7616 | 23 | Bam stop these class here and and king to the Earth | wer | 0.4545 |
| 160 | streaming_asr | emilia_zh_0004843191 | 0.7415 | 100 | Hence forth not only 欧洲 Geographers but 欧洲 skars in almost all over else of knowledge began to draw maps with spaces left to allow in They begin to admit that their theories were not perfect in that they were important things that that did not no | wer | 0.2340 |
| 320 | streaming_asr | emilia_zh_0004843191 | 0.6854 | 119 | Hence forth not only 欧洲 Geographers but 欧洲 skars in almost all ever else of knowledge began to draw maps with spaces left to hello in They begin to admit that their theories were not perfect in that they were important things that they did not no | wer | 0.2128 |
| 640 | streaming_asr | emilia_zh_0004843191 | 0.6514 | 129 | Hence forth not only 欧洲 Geographers but 欧洲 skolars almost all ever else of knowledge began to draw maps with spaces left to hello in They begin into admits that their theories were not perfect in that they were important things that they did not no | wer | 0.2766 |
| 1280 | streaming_asr | emilia_zh_0004843191 | 0.6462 | 134 | Hence forth not only European Geographers but European skellers almost all ever else of knowledge began to draw maps with spaces left to hello in They begin into would admit that their theories were not perfect in that they were important things that that it not know | wer | 0.2553 |
| 160 | streaming_asr | emilia_zh_0004943305 | 0.9938 | 2 | 我们国家层面年龄历来都如此编码十八周岁啊 | cer | 0.2381 |
| 320 | streaming_asr | emilia_zh_0004943305 | 0.9907 | 3 | 我们国家层面年龄历来都如此编码十八周岁啊 | cer | 0.2381 |
| 640 | streaming_asr | emilia_zh_0004943305 | 0.9938 | 2 | 我们国家成员年龄历来都如此绵马十八周岁啊 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0004943305 | 0.9938 | 2 | 我们国家成员年龄历来都如此绵马十八周岁啊 | cer | 0.1905 |
| 160 | causal_full_asr | emilia_zh_0005058847 | 0.9812 | 7 | 因为死心难起怎么就是慈悲的词纵然发起难得久情为什么呢人到自私 | cer | 0.2000 |
| 320 | causal_full_asr | emilia_zh_0005058847 | 0.9794 | 8 | 因为磁性难起磁嘛就是磁碑的磁纵然发起难得久听为什么呢人到自私 | cer | 0.3000 |
| 640 | causal_full_asr | emilia_zh_0005058847 | 0.9794 | 8 | 因为私心难起什么就是慈悲的慈纵然发起难得久情为什么呢人到自私 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005058847 | 0.9831 | 7 | 因为磁性难起磁嘛就是磁碑的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 160 | streaming_asr | emilia_zh_0005244297 | 0.9667 | 6 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005244297 | 0.9571 | 8 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005244297 | 0.9524 | 8 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005244297 | 0.9476 | 8 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005507553 | 0.8235 | 20 | Will you walk it me the water garden he said softly | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0005507553 | 0.8396 | 17 | Will you walk with me the water garden he said softly | wer | 0.0833 |
| 640 | streaming_asr | emilia_zh_0005507553 | 0.8556 | 14 | Will you walk with me the water garden he said softly | wer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0005507553 | 0.8342 | 17 | Will you walk with me the water garden he said softly | wer | 0.0833 |
| 160 | causal_full_asr | emilia_zh_0005713628 | 0.9680 | 6 | 大家会联想到中国画大家会想到的是什么名山大川 | cer | 0.0870 |
| 320 | causal_full_asr | emilia_zh_0005713628 | 0.9543 | 8 | 大家会联想到中国画大家会想到的是什么名山大川 | cer | 0.0870 |
| 640 | causal_full_asr | emilia_zh_0005713628 | 0.9589 | 7 | 大家会联想到中国画大家会想到的是什么名山大川 | cer | 0.0870 |
| 1280 | causal_full_asr | emilia_zh_0005713628 | 0.9589 | 8 | 大家会联想到中国文化大家会想到的是什么名山大川 | cer | 0.0870 |
| 160 | streaming_asr | emilia_zh_0005781428 | 0.9895 | 2 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005781428 | 0.9791 | 4 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005781428 | 0.9843 | 3 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005781428 | 0.9843 | 3 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0006000255 | 0.9644 | 7 | 因为上次我在旅行片给大家介绍的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.0811 |
| 320 | streaming_asr | emilia_zh_0006000255 | 0.9547 | 9 | 因为下次我在旅行天得家介绍的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1622 |
| 640 | streaming_asr | emilia_zh_0006000255 | 0.9482 | 11 | 因为下次我在旅行天给家介绍的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1351 |
| 1280 | streaming_asr | emilia_zh_0006000255 | 0.9385 | 14 | 因为上次我在旅行天给家介绍的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1081 |
| 160 | causal_full_asr | emilia_zh_0006056200 | 0.9728 | 7 | 就你要还是要用一些就是能看的摸着东西要不然的话你直接放大了 | cer | 0.2727 |
| 320 | causal_full_asr | emilia_zh_0006056200 | 0.9637 | 9 | 就你有要还是要用一些就是能看的摸着东西要不然的话你这氛围太大了 | cer | 0.2424 |
| 640 | causal_full_asr | emilia_zh_0006056200 | 0.9728 | 7 | 就你要还是要用一些就是能看的摸着的东西要不然的话你这氛围太大了 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006056200 | 0.9728 | 7 | 就你要还是要有一些就是能看的摸着的东西要不然的话你这氛围太大了 | cer | 0.1515 |
| 160 | streaming_asr | emilia_zh_0006212326 | 0.9887 | 6 | 嗯远远这个个真卷是这小老虎只是他一只对冒着追逐了个东西上的王子隐藏的自己对真实身份 | cer | 0.4750 |
| 320 | streaming_asr | emilia_zh_0006212326 | 0.9887 | 6 | 嗯远远这个个真正是这小老虎只是他一只对冒着追逐了都上的王子隐藏的自己的真实身份是 | cer | 0.4500 |
| 640 | streaming_asr | emilia_zh_0006212326 | 0.9806 | 9 | 如远远这个个真正是这小老虎只是他一只对冒着追逐了哥丁上的王者隐藏的自己的真实身份是 | cer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0006212326 | 0.9903 | 5 | 如约了这个个真正是这小老虎只是他一直对冒着追逐了哥就上的王者隐藏的自己对真实身份是 | cer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0006366864 | 0.8795 | 11 | I want it to continue to make be house I happy home for him | wer | 0.3077 |
| 320 | streaming_asr | emilia_zh_0006366864 | 0.8750 | 12 | I wanted to continue to make bleed house I happy home for him | wer | 0.1538 |
| 640 | streaming_asr | emilia_zh_0006366864 | 0.8884 | 11 | I wanted to continue to make bleed house I happy home for him | wer | 0.1538 |
| 1280 | streaming_asr | emilia_zh_0006366864 | 0.8839 | 11 | I wanted to continue to make bleed house I happy home for him | wer | 0.1538 |
| 160 | streaming_asr | emilia_zh_0006447049 | 0.7220 | 89 | You apart of and think like what what to and contribute was society my like good nough do my friends love me and I rode to of her quick note say This is what i pay into realize after turn thirty not to long a girl | wer | 0.4600 |
| 320 | streaming_asr | emilia_zh_0006447049 | 0.7154 | 96 | In Apart of and think like what what to a contribute was society my like good nough do more friends love me and I rode to of her quick notice say This is what i pay into realize after turn thirty not through long a girl | wer | 0.5200 |
| 640 | streaming_asr | emilia_zh_0006447049 | 0.6959 | 100 | In opart of and things like what what to a contribute but society my like good nough do my friends love me and I rode to of her quick notice say this is what I pay into realize after turn thirty not through long a girl | wer | 0.5200 |
| 1280 | streaming_asr | emilia_zh_0006447049 | 0.6927 | 106 | In apart of in things like what what can a contribute for society my like good enough do my friends love me and I rode to of her quick note say this is what I pay into realize after turn thirty not through long a girl | wer | 0.4600 |
| 160 | causal_full_asr | emilia_zh_0006598177 | 0.9829 | 5 | 阿叔这个女性受害人就像案件当中我如果跟他们啊五十几岁都是这个亲密关系 | cer | 0.3548 |
| 320 | causal_full_asr | emilia_zh_0006598177 | 0.9854 | 5 | 而说这个女性受害人就像案件当中如果跟他们那啊五十几页都是这个亲密关系 | cer | 0.3226 |
| 640 | causal_full_asr | emilia_zh_0006598177 | 0.9829 | 5 | 啊说这个女性受害人就像案件当中如果敢跟真的啊五十几人都是这个亲密关系 | cer | 0.2903 |
| 1280 | causal_full_asr | emilia_zh_0006598177 | 0.9854 | 5 | 啊说这个女性受害人就像案件当中如果敢跟真的啊五十几人都是这个亲密关系 | cer | 0.2903 |
| 160 | streaming_asr | emilia_zh_0006658157 | 0.9773 | 11 | 所以我觉得是这样子说所以任何品牌在中国办活动他肯定也是因为最一个活动本身或者就一个产品本身它在 | cer | 0.1200 |
| 320 | streaming_asr | emilia_zh_0006658157 | 0.9773 | 8 | 所以我觉得是这样子说所以任何品牌在中国伙伴活动他肯定也是因为最一个活动本身或者就一个产品本身他在 | cer | 0.1800 |
| 640 | streaming_asr | emilia_zh_0006658157 | 0.9697 | 11 | 所以我觉得是这样子说所以任何品牌在中国办活动他肯定也是因为最一个活动本身或者就一个产品本身他在 | cer | 0.1400 |
| 1280 | streaming_asr | emilia_zh_0006658157 | 0.9678 | 11 | 所以我觉得是这样子说所以任何品牌在中国办活动他肯定也是因为这一个活动本身或者就一个产品本身他在 | cer | 0.1200 |
| 160 | streaming_asr | emilia_zh_0006940399 | 0.9851 | 3 | 虽然他们二人都想进气前嫌从跌倒的地方爬起来 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0006940399 | 0.9813 | 3 | 虽然他们二人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0476 |
| 640 | streaming_asr | emilia_zh_0006940399 | 0.9851 | 3 | 虽然他们二人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0476 |
| 1280 | streaming_asr | emilia_zh_0006940399 | 0.9851 | 3 | 虽然他们二人都想进去前行从跌倒的地方爬起来 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007124543 | 0.9895 | 2 | 印度所述空隙风险在一起中最好印度尼西亚也曾经面临高风险 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0007124543 | 0.9895 | 3 | 印度所述空隙风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.1786 |
| 640 | streaming_asr | emilia_zh_0007124543 | 0.9895 | 2 | 印度所述恐吓危险在一起中最高印度尼西亚也曾经面临高风险 | cer | 0.2143 |
| 1280 | streaming_asr | emilia_zh_0007124543 | 0.9869 | 3 | 印度所述空隙风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.1786 |
| 160 | streaming_asr | emilia_zh_0007399687 | 0.9766 | 4 | 他他在一千多年前许下的落叶知识不需 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0007399687 | 0.9708 | 5 | 他在一千多年前许下的落叶知识不需 | cer | 0.3125 |
| 640 | streaming_asr | emilia_zh_0007399687 | 0.9649 | 6 | 他在一千多年前许下的落言真实不虚 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0007399687 | 0.9708 | 5 | 他在一千多年前许下的落言真实不虚 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0007635686 | 0.9717 | 7 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007635686 | 0.9748 | 7 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007635686 | 0.9717 | 7 | 我们的工作仍然没有实质性进展闹抓了几个小特务 | cer | 0.0455 |
| 1280 | streaming_asr | emilia_zh_0007635686 | 0.9748 | 6 | 我们的工作仍然没有实质性的进展闹抓了几个小特务 | cer | 0.0909 |
| 160 | streaming_asr | EN_B00083_S00689_W000013 | 0.7254 | 65 | You you say you using information entered the tool then calculated then net in his used it budget all out for the apartment This the number we need to keep low | wer | 0.4667 |
| 320 | streaming_asr | EN_B00083_S00689_W000013 | 0.7321 | 67 | You you so you using the information entered the tool then calculated the net energies used budget alliance for the apartment This the number we need to keep low | wer | 0.3000 |
| 640 | streaming_asr | EN_B00083_S00689_W000013 | 0.7433 | 66 | You you so a using information entered the too then calculated the now enjoy used budget alliance for the apartment This the number we need to keep low | wer | 0.4000 |
| 1280 | streaming_asr | EN_B00083_S00689_W000013 | 0.6920 | 79 | You will say the the using information entered that tool then calculated the now energies used budget alliance for the apartment This the number we need to keep low | wer | 0.4000 |
| 160 | causal_full_asr | EN_B00083_S00712_W000002 | 0.7726 | 65 | In addition the journal covers signaling networks synthetic biology systems biology try to discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 320 | causal_full_asr | EN_B00083_S00712_W000002 | 0.7401 | 72 | In addition the journal covers signaling networks synthetic biology systems biology generated discovery and computation and modeling of regulatory pathways | wer | 0.0500 |
| 640 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6949 | 82 | In addition the journal covers signaling networks synthetic biology systems biology generated discovery and computation and modeling of regulatory pathways | wer | 0.0500 |
| 1280 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6679 | 88 | In addition the journal covers signaling networks synthetic biology systems biology generated discovery and computation and modeling of regulatory pathways | wer | 0.0500 |
| 160 | streaming_asr | EN_B00013_S05888_W000041 | 0.8353 | 67 | That I me I a big part I of I out of to change that position I bit big friend chad port ages and saying you need more make him the Guy he a the guy you can be the next covey | wer | 0.5306 |
| 320 | streaming_asr | EN_B00013_S05888_W000041 | 0.8418 | 60 | That I me I of big far I of I i of to to that position of bit big friend chad port ages and saying you need more make him the Guy he could the guy you could be the next coverage | wer | 0.5102 |
| 640 | streaming_asr | EN_B00013_S05888_W000041 | 0.8402 | 62 | The I mean I a big kind I of I out of see each that obsession of bit big friend chad fort ages and saying he me more make him the Guy he a the guy you can be the next covey | wer | 0.5102 |
| 1280 | streaming_asr | EN_B00013_S05888_W000041 | 0.8336 | 62 | The I mean I of big kind I of I I love see each that obsession of been big friend chad four ages and saying he me more make him the Guy he a the guy he could be the next covey | wer | 0.4490 |
| 160 | causal_full_asr | EN_B00058_S06426_W000037 | 0.9049 | 12 | It looks like this curve here where f here is on the horizontal axis | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S06426_W000037 | 0.8768 | 16 | It looks like this curve here where f here is on the horizontal axis | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S06426_W000037 | 0.8415 | 20 | It looks like this curve here where f here is on the horizontal axis | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S06426_W000037 | 0.8521 | 20 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S02307_W000043 | 0.9414 | 15 | Let's check He likes to eat peter I don't want to eat this sal that isn't my shake | wer | 0.1111 |
| 320 | streaming_asr | EN_B00048_S02307_W000043 | 0.9432 | 16 | Let's check He likes to eat p Pizza I don't want to eat this sal that isn't my shake | wer | 0.1111 |
| 640 | streaming_asr | EN_B00048_S02307_W000043 | 0.9359 | 18 | Let's check He likes to eat pitsa I don't want to eat this sal that isn't my shake | wer | 0.1111 |
| 1280 | streaming_asr | EN_B00048_S02307_W000043 | 0.9414 | 18 | Let's check He likes to eat pizza I don't want to eat this sal that isn't my shake | wer | 0.0556 |
| 160 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6686 | 33 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6114 | 35 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6229 | 35 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6229 | 32 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S01128_W000118 | 0.7730 | 17 | As a tall for an or China I stick count like a sword them | wer | 0.4615 |
| 320 | streaming_asr | EN_B00058_S01128_W000118 | 0.7301 | 22 | As a tall for an or China I stick count like a sword them | wer | 0.4615 |
| 640 | streaming_asr | EN_B00058_S01128_W000118 | 0.7423 | 18 | As a tall four and are China I stick out like a sword them | wer | 0.3846 |
| 1280 | streaming_asr | EN_B00058_S01128_W000118 | 0.6933 | 21 | As a tall for an a China I stick out like a sore them | wer | 0.3077 |
| 160 | streaming_asr | EN_B00058_S07483_W000027 | 0.9554 | 4 | And you to fix I high earwings problem | wer | 0.4444 |
| 320 | streaming_asr | EN_B00058_S07483_W000027 | 0.9554 | 3 | And use this fakes I high airiness problem | wer | 0.4444 |
| 640 | streaming_asr | EN_B00058_S07483_W000027 | 0.8981 | 7 | And use this fix I high airiness problem | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00058_S07483_W000027 | 0.8790 | 8 | And use this fakes a high airiness problem | wer | 0.3333 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
