# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8768**
- Weighted CTC blank ratio: **0.1735**
- Weighted streaming WER/CER: **0.4917**
- Weighted causal-full WER/CER: **0.1552**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000188343 | 0.0866 | 5 | The of of just thoroughly | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000188343 | 0.1496 | 8 | They of a just thoroughly | wer | 0.6000 |
| 640 | streaming_asr | CommonVoice_EN_0000188343 | 0.1654 | 6 | Day of a just thoroughly | wer | 0.8000 |
| 1280 | streaming_asr | CommonVoice_EN_0000188343 | 0.1024 | 6 | They of a just Verily the | wer | 1.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000331841 | 0.1692 | 8 | Just did rel the me the asked time | wer | 0.6250 |
| 320 | streaming_asr | CommonVoice_EN_0000331841 | 0.2462 | 11 | Just did rode the me the asked time | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000331841 | 0.2692 | 11 | Just did rode the me the asked time | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000331841 | 0.2692 | 11 | You did rode the me the asked time | wer | 0.5000 |
| 160 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1339 | 21 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 320 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1726 | 20 | Teams from Topeka, Kansas and Wichita, Kansas, joined from the Western Association | wer | 0.2500 |
| 640 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1964 | 22 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 1280 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1726 | 25 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 160 | streaming_asr | CommonVoice_EN_0000471209 | 0.1301 | 5 | The boy begun begun you can to the groom | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000471209 | 0.2260 | 13 | The boy begun begun you can to the jun | wer | 0.7500 |
| 640 | streaming_asr | CommonVoice_EN_0000471209 | 0.1986 | 13 | The boy begun begun a can the jun | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000471209 | 0.1918 | 11 | The boy begun begun a can to the jun | wer | 0.7500 |
| 160 | streaming_asr | HQ-Conversations_0000028308 | 0.1613 | 6 | 我兄弟咋了？ | cer | 0.6000 |
| 320 | streaming_asr | HQ-Conversations_0000028308 | 0.2097 | 7 | 我行了咋了 | cer | 0.8000 |
| 640 | streaming_asr | HQ-Conversations_0000028308 | 0.2258 | 5 | 我行了咋了 | cer | 0.8000 |
| 1280 | streaming_asr | HQ-Conversations_0000028308 | 0.1935 | 6 | 我行。咋了？ | cer | 1.0000 |
| 160 | causal_full_asr | LibriSpeech_0000011649 | 0.1460 | 43 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000011649 | 0.1711 | 48 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000011649 | 0.1770 | 52 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000011649 | 0.1593 | 51 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000090820 | 0.1173 | 35 | which hempt me in he house Nearly to wigs The pace kitchens see heavenly say from war the was days like to little boat no winter see the and were out and the fields all day hasking cork and and can in new | wer | 0.5625 |
| 320 | streaming_asr | LibriSpeech_0000090820 | 0.1280 | 39 | which hempt me in he of Nearly to weeks The pace kitchens see Heaven say from war the was days like to little boat no winter see the and where out and the fields all day hasking cork and and can in new | wer | 0.6042 |
| 640 | streaming_asr | LibriSpeech_0000090820 | 0.1307 | 41 | Which hipped in he of Nearly to wigs The pace kitchens see heavenly say from war the was days like to little boat in winter see the and were out and the heals all day hasking coral. and and can in noon | wer | 0.5833 |
| 1280 | streaming_asr | LibriSpeech_0000090820 | 0.1253 | 40 | Which hipped in he house Nearly to weeks The pace kitchens see heavenly say from war the was days like to little boat in winter see the and were how and the heals all day asking cork and and can in new | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000215121 | 0.1652 | 45 | And to but you to me which your normsman of or I immediate counted him over of for to ban nuts one christ not the the his head the to no fest some But that is not all continued don't | wer | 0.6111 |
| 320 | streaming_asr | LibriSpeech_0000215121 | 0.1681 | 44 | And turn but you to me which your dormant of or I immed it counted can't over of forty about nuts on christ not the the his head the to no fest some But that is that all continued blind. | wer | 0.6389 |
| 640 | streaming_asr | LibriSpeech_0000215121 | 0.1768 | 46 | And turn but you to me which your dormancy of or I immediate counted can over of forty ban not on christ not the the his head the to no some But that is not all continue blind. | wer | 0.5833 |
| 1280 | streaming_asr | LibriSpeech_0000215121 | 0.1739 | 47 | And turn but you to me which your dormancy off or I me to recounted can over of forty ban not on christ not the the his head the to no some But that is not all continue blind. | wer | 0.6667 |
| 160 | streaming_asr | emilia_zh_0003918326 | 0.2115 | 21 | 啊我我什么还有一个点听意思这是这种现在你人在<\|write_generate\|><\|cmn\|><\|start_content\|>这些人他们点互相关联 | cer | 1.5278 |
| 320 | streaming_asr | emilia_zh_0003918326 | 0.1987 | 20 | 啊我我什么呢还有一个点听意思这是这个现在了人在。这些人他们眼互相关联 | cer | 0.4444 |
| 640 | streaming_asr | emilia_zh_0003918326 | 0.2212 | 21 | 啊我我什么呢还有一个变听意思这是这个是了人在就这些人他们眼互相关联 | cer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0003918326 | 0.2276 | 23 | 啊我我什么呢还有一个变听意思这是这段是人人起来<\|write_generate\|><\|cmn\|><\|start_content\|>这些人他们眼互相关联 | cer | 1.5556 |
| 160 | causal_full_asr | emilia_zh_0003942539 | 0.2257 | 40 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0175 |
| 320 | causal_full_asr | emilia_zh_0003942539 | 0.2178 | 38 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很慌恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0003942539 | 0.2119 | 39 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0351 |
| 1280 | causal_full_asr | emilia_zh_0003942539 | 0.1960 | 39 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0351 |
| 160 | streaming_asr | emilia_zh_0004129851 | 0.1852 | 8 | 之间我两个人做到地方喝酒能 | cer | 0.5714 |
| 320 | streaming_asr | emilia_zh_0004129851 | 0.2037 | 12 | 之间都两个人做到的地方喝酒能 | cer | 0.5714 |
| 640 | streaming_asr | emilia_zh_0004129851 | 0.2222 | 11 | 直接都两个人做的地方喝酒能 | cer | 0.5714 |
| 1280 | streaming_asr | emilia_zh_0004129851 | 0.1852 | 12 | 之间都两个人做的地方喝酒能 | cer | 0.5714 |
| 160 | streaming_asr | emilia_zh_0004358307 | 0.1673 | 33 | You was up to him him prove himself the here six to make the and ground of and and to music without the fate idea a how prayer were already | wer | 0.4333 |
| 320 | streaming_asr | emilia_zh_0004358307 | 0.1653 | 29 | You was up to him he prove himself the here six to make the and ground of and and to music without the face idea a how praud of were already | wer | 0.4333 |
| 640 | streaming_asr | emilia_zh_0004358307 | 0.1514 | 30 | You was up to him he prove himself of the here six to make the and road of him and he music without the face idea a how praud of were already | wer | 0.4333 |
| 1280 | streaming_asr | emilia_zh_0004358307 | 0.1315 | 33 | You was up to him he prove himself of the air six to make the and ride of him and he music without the face idea a how praud of were already | wer | 0.4333 |
| 160 | streaming_asr | emilia_zh_0004659190 | 0.1583 | 24 | See the the when was close granting a he repressed when you on love you in Anything creation | wer | 0.5238 |
| 320 | streaming_asr | emilia_zh_0004659190 | 0.1847 | 24 | Seeing the the when was close granting a he repressed When you and love you in Anything creation | wer | 0.4762 |
| 640 | streaming_asr | emilia_zh_0004659190 | 0.1873 | 24 | Seeing the the when was close granting a he Microsoft When you one love you in Anything creation | wer | 0.4762 |
| 1280 | streaming_asr | emilia_zh_0004659190 | 0.1530 | 24 | Seeing the the when was close granting he requests when you one love you and Anything creation | wer | 0.4762 |
| 160 | causal_full_asr | emilia_zh_0004659501 | 0.1404 | 15 | And for all you just write nine or ten and then where do, where to | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004659501 | 0.1633 | 18 | And for old you just write nine or ten and then where do, where to | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0004659501 | 0.1576 | 19 | And for all you just write nine or ten and then where do, where to | wer | 0.2857 |
| 1280 | causal_full_asr | emilia_zh_0004659501 | 0.1547 | 17 | And for all you just write nine or ten and then where to where to | wer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0004776929 | 0.1789 | 22 | You evening They thousand down for a being families in a This was it | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0004776929 | 0.1474 | 16 | in evening They stessel down for a big families in a This was it | wer | 0.3571 |
| 640 | streaming_asr | emilia_zh_0004776929 | 0.1298 | 15 | in evening They sessed down for a big family in a This was it | wer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0004776929 | 0.1263 | 19 | In even They thessel down for of big family in a This was it | wer | 0.4286 |
| 160 | streaming_asr | emilia_zh_0004843272 | 0.1444 | 35 | You principled tended the that economic grow is supreme the or right please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even appiness of the pen don't money grew of | wer | 0.6129 |
| 320 | streaming_asr | emilia_zh_0004843272 | 0.1648 | 38 | It principled tennet the that economic growth is supreme de过的 or right please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even appiness of the pen don't con dominant grew fried | wer | 0.6129 |
| 640 | streaming_asr | emilia_zh_0004843272 | 0.1481 | 39 | It's principled tenanted the that con dominant growth is this supreme the or or please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even ap<\|glm_semantic_5487\|>per upper the pen don't con dominant grew fried | wer | 0.6452 |
| 1280 | streaming_asr | emilia_zh_0004843272 | 0.1537 | 39 | It's principled tenanted the that con dominant growth is supreme good or or please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even ap<\|glm_semantic_5487\|>per upper the pen don't money grew fried | wer | 0.5806 |
| 160 | streaming_asr | emilia_zh_0004943713 | 0.1718 | 40 | 等因我是的他清出了你这这上面三的信仰可以成尝试问从业者黑洞租户他们你的情感的鱼场 | cer | 0.5116 |
| 320 | streaming_asr | emilia_zh_0004943713 | 0.2177 | 43 | 但是因我做到他清楚了你这这上面三的信仰可成尝试问从业者奋斗活跃。他们那个情感的愚蠢 | cer | 0.4186 |
| 640 | streaming_asr | emilia_zh_0004943713 | 0.2109 | 44 | 但是因我知道他清楚了你在在上面三的信仰可成尝试问从业者奋斗维护啊你的情感的愚蠢 | cer | 0.3023 |
| 1280 | streaming_asr | emilia_zh_0004943713 | 0.1973 | 42 | 但是是英我知道他清楚了你就这上面三对的信仰可以成尝试温从业者奋斗维护他你的情感的愚蠢 | cer | 0.3023 |
| 160 | causal_full_asr | emilia_zh_0005070101 | 0.1468 | 32 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005070101 | 0.1315 | 35 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005070101 | 0.1437 | 35 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005070101 | 0.1391 | 37 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0005313494 | 0.1962 | 10 | 对不能都乐来吧应该后面没有一个应该平 | cer | 0.5500 |
| 320 | streaming_asr | emilia_zh_0005313494 | 0.2025 | 11 | 不能都了来吧一个后面没有一个应该拼 | cer | 0.4500 |
| 640 | streaming_asr | emilia_zh_0005313494 | 0.2152 | 11 | 他就不能读了来吧一个后面没有一个应该平 | cer | 0.3500 |
| 1280 | streaming_asr | emilia_zh_0005313494 | 0.2215 | 11 | 他就不能读了来吧一个后面没有一个应该拼 | cer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0005578304 | 0.1507 | 12 | 对素也回来就好奇的问 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0005578304 | 0.1507 | 12 | 对数年回来就好奇的问 | cer | 0.4167 |
| 640 | streaming_asr | emilia_zh_0005578304 | 0.1945 | 15 | 对所以回来就孩子的问 | cer | 0.5833 |
| 1280 | streaming_asr | emilia_zh_0005578304 | 0.1726 | 18 | 道数年回来就孩子的问 | cer | 0.5833 |
| 160 | causal_full_asr | emilia_zh_0005714451 | 0.1818 | 16 | 什么撞到你进来你可以看到其实前边的三五表 | cer | 0.5200 |
| 320 | causal_full_asr | emilia_zh_0005714451 | 0.1888 | 18 | 虽然赚大宁接下来你可以看到其实前边的三部表 | cer | 0.4800 |
| 640 | causal_full_asr | emilia_zh_0005714451 | 0.1958 | 17 | 生产到宁接下来你可以看到其实前边的桑木表 | cer | 0.4400 |
| 1280 | causal_full_asr | emilia_zh_0005714451 | 0.1923 | 19 | 身份状态应接下来你可以看到其实前面的项目表 | cer | 0.4400 |
| 160 | streaming_asr | emilia_zh_0005818033 | 0.1287 | 10 | 我一个那个造成了啊这个他们这边 | cer | 0.5333 |
| 320 | streaming_asr | emilia_zh_0005818033 | 0.2047 | 12 | 报告一个那个造成啊这个他们这边 | cer | 0.4667 |
| 640 | streaming_asr | emilia_zh_0005818033 | 0.1520 | 11 | 报告一个那个过程啊这个咱们这边 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005818033 | 0.1287 | 9 | 报告一个那个过程啊这个咱们这边 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0006041629 | 0.2203 | 27 | 做他来谁学习后觉得现在去过去里其实创伤来但是整体来说我觉得这演员他他眼里在躲避有趣 | cer | 0.5417 |
| 320 | streaming_asr | emilia_zh_0006041629 | 0.1683 | 23 | 做看来谁信息后觉得下去过去里些创伤啊但是整体来说我觉得这演员他他眼里在特别有趣 | cer | 0.4167 |
| 640 | streaming_asr | emilia_zh_0006041629 | 0.1757 | 22 | 就是看像谁信息以后觉得现在就够里些创伤来但是整体来说我觉得这演员他在眼里在别有趣 | cer | 0.3542 |
| 1280 | streaming_asr | emilia_zh_0006041629 | 0.1782 | 21 | 就是看来谁信息后觉得这样就够里些创伤来但是整体来说我觉得这演员他他眼里在别有趣 | cer | 0.3958 |
| 160 | causal_full_asr | emilia_zh_0006099356 | 0.2531 | 18 | 就整个这个留下还不知道但是你非认主你确定是能拿到钱的什么 | cer | 0.3704 |
| 320 | causal_full_asr | emilia_zh_0006099356 | 0.2573 | 19 | 就整个这个留下还不着的我但是你非认主你确定是能拿到钱的什么 | cer | 0.4074 |
| 640 | causal_full_asr | emilia_zh_0006099356 | 0.2448 | 19 | 就整个这个留下还不知道但是你虽然这种你确定是拿多少钱那是吗 | cer | 0.4815 |
| 1280 | causal_full_asr | emilia_zh_0006099356 | 0.2365 | 19 | 就整个这个留下还不知道但是你非认着你确定是拿刀钱的什么 | cer | 0.4444 |
| 160 | streaming_asr | emilia_zh_0006270122 | 0.1567 | 18 | 你这样都是带的他他一个完整的这个原因啊他就是一 | cer | 0.3600 |
| 320 | streaming_asr | emilia_zh_0006270122 | 0.1604 | 23 | 你这样都是对的他他一个完整的这个原因啊他就是一 | cer | 0.3200 |
| 640 | streaming_asr | emilia_zh_0006270122 | 0.1679 | 18 | 你这样主是对的他是一个完整的这个原因啊他就是一 | cer | 0.2800 |
| 1280 | streaming_asr | emilia_zh_0006270122 | 0.1716 | 20 | 你这样读是对的他他一个完整的这个原因要他就是一 | cer | 0.3200 |
| 160 | streaming_asr | emilia_zh_0006379722 | 0.1551 | 18 | But This couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0006379722 | 0.1320 | 18 | But This couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0006379722 | 0.1287 | 16 | But the couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0006379722 | 0.1287 | 16 | But the couldn't lead the there two He must take then will him all the way | wer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0006464698 | 0.2249 | 29 | I one of ways could for example He one me to tell and my big count pass were and he can take photographed so for me from different dangles | wer | 0.5556 |
| 320 | streaming_asr | emilia_zh_0006464698 | 0.2227 | 32 | I one of ways could for ex stumble he one me to tell in my big count pass were and he can take photographed so for me from different dangles | wer | 0.6296 |
| 640 | streaming_asr | emilia_zh_0006464698 | 0.2271 | 30 | I one for has could for ex<\|glm_semantic_3040\|>bal he one me to tell in my being count pass were and he can take photographed so for me from different dangles | wer | 0.6667 |
| 1280 | streaming_asr | emilia_zh_0006464698 | 0.2249 | 32 | I one of ways clear for ex gamble he one me to tell in my being count pass were and he can take photographed so me from different dangles | wer | 0.5926 |
| 160 | causal_full_asr | emilia_zh_0006610442 | 0.2575 | 15 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0006610442 | 0.2934 | 15 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 640 | causal_full_asr | emilia_zh_0006610442 | 0.2934 | 16 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 1280 | causal_full_asr | emilia_zh_0006610442 | 0.2695 | 16 | 嗯我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 160 | streaming_asr | emilia_zh_0006713619 | 0.2237 | 17 | 不管那些心慢慢没要护寡赶紧护寡起来在给你三分钟时间互关玩发布了啊 | cer | 0.5135 |
| 320 | streaming_asr | emilia_zh_0006713619 | 0.2303 | 19 | 不管啊那些心男人你要护光赶紧护光起来在给你你三分钟时间互关玩啊，不了啊 | cer | 0.4595 |
| 640 | streaming_asr | emilia_zh_0006713619 | 0.2368 | 19 | 不管啊那些心们你要护光赶紧护光起来在给你三分钟时间互关玩啊，不了啊 | cer | 0.4324 |
| 1280 | streaming_asr | emilia_zh_0006713619 | 0.2368 | 19 | 不管啊那些心们你要护冠赶紧护冠起来在给你散恨事情互关玩下播了啊 | cer | 0.5135 |
| 160 | streaming_asr | emilia_zh_0006940509 | 0.3218 | 17 | 宏伟都是的街道很快变不了红雨 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0006940509 | 0.3276 | 16 | 宏伟都是的街道很快被不了红雨 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0006940509 | 0.2931 | 15 | 红都是的街道很快被不了红雨 | cer | 0.4667 |
| 1280 | streaming_asr | emilia_zh_0006940509 | 0.2529 | 14 | 红都是的街道很快被不了红雨 | cer | 0.4667 |
| 160 | streaming_asr | emilia_zh_0007124790 | 0.1328 | 12 | 你好请问就是去老自的路吧男人看了解决也 | cer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0007124790 | 0.1441 | 11 | 你好请问这是去老自的路吧男人看了表情也 | cer | 0.3182 |
| 640 | streaming_asr | emilia_zh_0007124790 | 0.1130 | 12 | 你好请问这个是去了自的路吧男人看了调军也 | cer | 0.4091 |
| 1280 | streaming_asr | emilia_zh_0007124790 | 0.1130 | 13 | 你好请问这个是去老自的路吧男人看了决定也 | cer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0007461662 | 0.1224 | 23 | 吧那么这个是导师新的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0007461662 | 0.1204 | 23 | 吧那么这个是导师信息的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0007461662 | 0.1367 | 25 | 把了吗这个是导师信息的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1944 |
| 1280 | streaming_asr | emilia_zh_0007461662 | 0.1204 | 26 | 把那么这个是导师感兴趣的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0556 |
| 160 | streaming_asr | emilia_zh_0007690753 | 0.1797 | 36 | 可能市场在行的的是仅仅了一年搞了二零零四年最漫长的下跌再发生上升宗旨再出线越线五连云的悲惨场面 | cer | 0.2549 |
| 320 | streaming_asr | emilia_zh_0007690753 | 0.1881 | 40 | 可能市场这。的的是仅仅了一年搞老二零零四年最漫长下跌再法式上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0007690753 | 0.1898 | 43 | 可能市场这。的的是仅仅了一年搞老二零零四年最漫长下跌再发生上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3137 |
| 1280 | streaming_asr | emilia_zh_0007690753 | 0.1898 | 41 | 可能市场这行的的是仅仅了一年搞老二零零四年最漫长下跌再发生上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3137 |
| 160 | streaming_asr | EN_B00083_S02942_W000001 | 0.1897 | 33 | It is going to be of restorative that then be a many course for any writer so want to learn how to break and a list are own serious Stories | wer | 0.4516 |
| 320 | streaming_asr | EN_B00083_S02942_W000001 | 0.1897 | 33 | It is going to be be re sourced that thing be a many course for any writer so want to learn how to right and a list are own serious Stories | wer | 0.4516 |
| 640 | streaming_asr | EN_B00083_S02942_W000001 | 0.1874 | 34 | It is going to be be restors that singing be a many course for any writer so want to learn how to right and a list are own Series Stories | wer | 0.4516 |
| 1280 | streaming_asr | EN_B00083_S02942_W000001 | 0.1663 | 33 | It is going to be be restores that singing being many course for any writer his want to learn had to right and huddled here owning Series Stories | wer | 0.5484 |
| 160 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1465 | 58 | We don't learn from our successes and we don't learn from our failures in a way that allows the impact to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success. I've tried to do that throughout my relationship with them. | wer | 0.0877 |
| 320 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1424 | 60 | We don't learn from our successes and we don't learn from our failures in a way that allows the impressed to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1228 |
| 640 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1445 | 61 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1053 |
| 1280 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1445 | 62 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1053 |
| 160 | streaming_asr | EN_B00013_S06799_W000009 | 0.2260 | 29 | And We that the can how better model is sure safe fearness and help and user to have better reliety appropriation trial | wer | 0.5238 |
| 320 | streaming_asr | EN_B00013_S06799_W000009 | 0.2374 | 33 | And We that that can had better model is true Safety fearness and help and user to have better reliety appropriation trial | wer | 0.4762 |
| 640 | streaming_asr | EN_B00013_S06799_W000009 | 0.1963 | 30 | And We that that can had better model is true Safety fearness and help and user to have better reliety appropriation trial | wer | 0.4762 |
| 1280 | streaming_asr | EN_B00013_S06799_W000009 | 0.1758 | 30 | And We That was can had better model is true Safety fearness and help and user to have better reularity appropriation trial | wer | 0.4762 |
| 160 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1227 | 25 | In Tibetan Buddhism, models are mainly used to count mantras. These mantras can be recited for different purposes linked to working with mind. | wer | 0.1739 |
| 320 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1227 | 21 | Intubent Buddhism models are mainly used to count monsters. These monsters can be recited for different purposes linked to working with mind. | wer | 0.2609 |
| 640 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1044 | 23 | Intubent Buddhism models are mainly used to count montras These montras can be recited for different purposes linked to working with mind | wer | 0.2174 |
| 1280 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1097 | 25 | Interpenent Buddhism models are mainly used to count montras. These montras can be recited for different purposes linked to working with mind. | wer | 0.2609 |
| 160 | streaming_asr | EN_B00048_S03599_W000339 | 0.2156 | 19 | And Such round I my shriek froze the blood of Every one close by | wer | 0.4615 |
| 320 | streaming_asr | EN_B00048_S03599_W000339 | 0.2062 | 21 | I turn to round I my shriek froze the blood of Everyone close by | wer | 0.3077 |
| 640 | streaming_asr | EN_B00048_S03599_W000339 | 0.2000 | 22 | I turned round I my shriek froze the blood of Everyone close by | wer | 0.1538 |
| 1280 | streaming_asr | EN_B00048_S03599_W000339 | 0.1437 | 19 | And turned around I my screek froze the blood of Everyone close by | wer | 0.2308 |
| 160 | streaming_asr | EN_B00058_S03125_W000010 | 0.1943 | 43 | We from come some count with base each qing is it's of from yeah to red indicating to supposition solution is a base at was I termor extinct turns red was a of and contacted with any kind of base | wer | 0.6190 |
| 320 | streaming_asr | EN_B00058_S03125_W000010 | 0.1833 | 43 | We tremor come some contemplate with base each change is it's of from yeah to red indicating to so be solution is a base at was I termor extinct turns red was a of some contacted with in can of base | wer | 0.6667 |
| 640 | streaming_asr | EN_B00058_S03125_W000010 | 0.1833 | 44 | I trumbery come some content with base each change is it's of from yeah to red indicating to supposition solution is bas. That was I termor extinct turns red was and of some contacted with in can of base | wer | 0.6905 |
| 1280 | streaming_asr | EN_B00058_S03125_W000010 | 0.1643 | 42 | I term come some content with base each change is it's of from yeah to red indicating to supposition solution is bas. That was I termor extinct turns red when and of and contacted with in can of base | wer | 0.6667 |
| 160 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1633 | 25 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. He said that's a place in September. | wer | 0.2692 |
| 320 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1658 | 23 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that took place in September | wer | 0.1154 |
| 640 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1633 | 27 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that to place in September | wer | 0.1538 |
| 1280 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1429 | 27 | So you original meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that your place in September | wer | 0.1154 |
| 160 | streaming_asr | EN_B00058_S07511_W000000 | 0.2513 | 20 | Getting everywhere yeah the house the orrying can you really of special the or to school | wer | 0.5789 |
| 320 | streaming_asr | EN_B00058_S07511_W000000 | 0.2256 | 17 | Getting everywhere yeah the house the ringing can you really of special the or to school | wer | 0.5789 |
| 640 | streaming_asr | EN_B00058_S07511_W000000 | 0.2308 | 19 | Again everywhere yeah the house the arning can you really of special the or to school | wer | 0.6316 |
| 1280 | streaming_asr | EN_B00058_S07511_W000000 | 0.2103 | 20 | Getting everywhere yeah the house the turning can you really of special the or to school | wer | 0.5789 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
