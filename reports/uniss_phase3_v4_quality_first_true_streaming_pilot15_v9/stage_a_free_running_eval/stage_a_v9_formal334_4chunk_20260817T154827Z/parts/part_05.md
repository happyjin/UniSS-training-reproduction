# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8904**
- Weighted CTC blank ratio: **0.1924**
- Weighted streaming WER/CER: **0.3922**
- Weighted causal-full WER/CER: **0.2534**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000126367 | 0.2071 | 15 | You some always some grids to tish for Exodusism and airage in the Poor | wer | 0.6923 |
| 320 | streaming_asr | CommonVoice_EN_0000126367 | 0.2357 | 17 | You you always some grids your this for Expedity and airage it the Poor | wer | 0.6923 |
| 640 | streaming_asr | CommonVoice_EN_0000126367 | 0.2607 | 16 | You you always some grids some this for Exicism and airage in the Poor | wer | 0.6923 |
| 1280 | streaming_asr | CommonVoice_EN_0000126367 | 0.2250 | 15 | You you always that grids your tish a for Expedicism and airage it the Poor | wer | 0.7692 |
| 160 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1600 | 17 | It is across the Columbia River from whence Yamen, Washington | wer | 0.2000 |
| 320 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1964 | 18 | It is across the Columbia River from whence Yamman, Washington | wer | 0.2000 |
| 640 | causal_full_asr | CommonVoice_EN_0000262462 | 0.2145 | 17 | It is across the Columbia River from West Hamen, Washington | wer | 0.2000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1673 | 15 | It is across the Columbia River from Wiesam in Washington | wer | 0.2000 |
| 160 | streaming_asr | CommonVoice_EN_0000286138 | 0.1310 | 11 | The port reminded the old man the he it said something about hidden dresses | wer | 0.2857 |
| 320 | streaming_asr | CommonVoice_EN_0000286138 | 0.1470 | 13 | The port reminded the old man the he it said something about hidden dress | wer | 0.2857 |
| 640 | streaming_asr | CommonVoice_EN_0000286138 | 0.1661 | 13 | The port reminded the old man the he it said something about hidden trasured | wer | 0.2857 |
| 1280 | streaming_asr | CommonVoice_EN_0000286138 | 0.1661 | 14 | The port remind of the old man the he it said something about hidden trasured | wer | 0.4286 |
| 160 | streaming_asr | CommonVoice_EN_0000433992 | 0.1882 | 15 | Nobody want the disgust how you all and got trap in a John learnt box | wer | 0.7143 |
| 320 | streaming_asr | CommonVoice_EN_0000433992 | 0.2288 | 20 | Nobody want the Discuss how you all and got trap in a John learnt box | wer | 0.6429 |
| 640 | streaming_asr | CommonVoice_EN_0000433992 | 0.2768 | 24 | Nobody want the Discuss how you all and got trap in a kind learnt box | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000433992 | 0.3210 | 22 | Nobody want the Discuss how you old in the got trap in a kind learnt box | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000593898 | 0.2143 | 19 | In We've seen here's seen George's the spent be in a mad wholes hole roots | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000593898 | 0.2536 | 21 | In We've here's seen George's the spent be in a mad wholesore roots | wer | 0.8333 |
| 640 | streaming_asr | CommonVoice_EN_0000593898 | 0.3000 | 24 | In was here's seen George's the spent be in a mad wholes cold roots | wer | 0.9167 |
| 1280 | streaming_asr | CommonVoice_EN_0000593898 | 0.3179 | 18 | In was here's saint George's the spent be in a mad class hall roots | wer | 0.8333 |
| 160 | causal_full_asr | DailyTalk_0000010084 | 0.2031 | 8 | Good morning sir, what can I do for you? | wer | 0.3333 |
| 320 | causal_full_asr | DailyTalk_0000010084 | 0.2109 | 9 | The morning sir, what can I do for you? | wer | 0.4444 |
| 640 | causal_full_asr | DailyTalk_0000010084 | 0.1797 | 6 | The morning sir, what can I do for you? | wer | 0.4444 |
| 1280 | causal_full_asr | DailyTalk_0000010084 | 0.1719 | 8 | The morning sir, what can I do for you? | wer | 0.4444 |
| 160 | streaming_asr | LibriSpeech_0000068284 | 0.1303 | 32 | Is not a were to facts but only of the meaning of facts It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 320 | streaming_asr | LibriSpeech_0000068284 | 0.1285 | 30 | Is not a world the facts but only of the meaning of fact It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 640 | streaming_asr | LibriSpeech_0000068284 | 0.1250 | 30 | Is not a world the facts but only of the meaning a facts It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 1280 | streaming_asr | LibriSpeech_0000068284 | 0.1373 | 34 | Is not a world the facts but only of the meaning of facts It see point to view for judging facts It appertains to it Different ology | wer | 0.1786 |
| 160 | streaming_asr | LibriSpeech_0000158773 | 0.1383 | 44 | I men do it blessings blessings berry a live that I redealize my capable my I a a conquer in I and when on you on security relaying my thrift the judgment and my not of world I chose is being some prefer of all there's | wer | 0.6327 |
| 320 | streaming_asr | LibriSpeech_0000158773 | 0.1383 | 48 | I men do it blessings blessings berry a believe that I redealizing my capable my I make a conquer in I and when on you on security relined my thrift the judgment and my not of world I chose is business some prefers of of the | wer | 0.6122 |
| 640 | streaming_asr | LibriSpeech_0000158773 | 0.1597 | 52 | I men joyed blessings blessings berry a live that I do you do my capable my my a little can in a I and when money on security relined my thrift the judgment and my knowledge of world I chose is business some prefers of a there's | wer | 0.6327 |
| 1280 | streaming_asr | LibriSpeech_0000158773 | 0.1289 | 44 | I men joyed blessings of every a believe that I you do like my capable my I make little conquer in I again when money on security relined my thrift the judgment and my knowledge of world I chose is business some prefers of of the | wer | 0.5918 |
| 160 | causal_full_asr | VCTK_0000006143 | 0.0886 | 9 | Being captain of this club's fantastic | wer | 0.2857 |
| 320 | causal_full_asr | VCTK_0000006143 | 0.1266 | 12 | Being captain of this club's fantastic | wer | 0.2857 |
| 640 | causal_full_asr | VCTK_0000006143 | 0.1266 | 12 | Being kept in of this club's fantastic | wer | 0.5714 |
| 1280 | causal_full_asr | VCTK_0000006143 | 0.1329 | 12 | Being kept in of this club's fantastic | wer | 0.5714 |
| 160 | streaming_asr | VCTK_0000029362 | 0.1000 | 8 | I about is exhausted | wer | 0.5000 |
| 320 | streaming_asr | VCTK_0000029362 | 0.1313 | 8 | I about is exhausted | wer | 0.5000 |
| 640 | streaming_asr | VCTK_0000029362 | 0.1125 | 8 | I body is sustained | wer | 0.5000 |
| 1280 | streaming_asr | VCTK_0000029362 | 0.1125 | 8 | I body is exhausted | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004111317 | 0.2418 | 54 | 我我主意我们为什么部挖坑潮水啊哦是的的我相信如果我们啊的足够神我们可以到水的东西这我们权责一个这一点呢他开始我办法 | cer | 0.3929 |
| 320 | streaming_asr | emilia_zh_0004111317 | 0.2467 | 58 | 我我主意我们为什么部挖坑找水啊哦是的的我相信如果我们啊的足够神我们可以到水的对这我们选择一个这一点看开始我爸爸 | cer | 0.3036 |
| 640 | streaming_asr | emilia_zh_0004111317 | 0.2503 | 63 | 我我主意我们为什么部挖坑潮水啊哦湿润的的我相信如果我们啊的足够神我们可以到水的堆热我们选择一个这一点那个开始我爸爸 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0004111317 | 0.2588 | 61 | 我有个主意我们为什么部挖坑找水呢哦湿润的的我相信如果我们啊的足够神我们可以到水的堆热我们选择一个这一点那个开始的爸爸 | cer | 0.3036 |
| 160 | streaming_asr | emilia_zh_0004270182 | 0.2292 | 23 | 找到从前埋藏的已经很久很久的古的弹走崩掉螃蟹的老鼠的 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0004270182 | 0.2411 | 26 | 找到从前埋藏的已经很久很久的古德弹走工具螃蟹老鼠们 | cer | 0.2593 |
| 640 | streaming_asr | emilia_zh_0004270182 | 0.2649 | 27 | 找到从前埋藏的已经很久很久的古道挡走工具螃蟹老鼠们 | cer | 0.2593 |
| 1280 | streaming_asr | emilia_zh_0004270182 | 0.2500 | 27 | 找到从前埋藏的已经很久很久的古德挡走工具螃蟹老鼠们 | cer | 0.2593 |
| 160 | streaming_asr | emilia_zh_0004621436 | 0.2066 | 18 | But more red and asked you they good possibly come is paying Guess | wer | 0.5714 |
| 320 | streaming_asr | emilia_zh_0004621436 | 0.2254 | 20 | The mother road and asked he they could possibly come is pay Guess | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004621436 | 0.1972 | 21 | The mother road and asked he they could possibly come is pay g guests | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0004621436 | 0.1502 | 19 | The mother road and asked you they could positively come is paying g guests | wer | 0.5000 |
| 160 | causal_full_asr | emilia_zh_0004621493 | 0.1905 | 14 | Jesus and the tribe to be on man who steals a little bit here and there | wer | 0.6429 |
| 320 | causal_full_asr | emilia_zh_0004621493 | 0.2143 | 14 | Just an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 640 | causal_full_asr | emilia_zh_0004621493 | 0.1524 | 15 | He's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 1280 | causal_full_asr | emilia_zh_0004621493 | 0.1333 | 12 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0004754634 | 0.2222 | 13 | Perhaps here of take a joke suggested jack do | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0004754634 | 0.2157 | 14 | Perhaps here of can jerk suggested jack all | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0004754634 | 0.2157 | 14 | Perhaps every of take a jerk suggested jack do | wer | 0.6000 |
| 1280 | streaming_asr | emilia_zh_0004754634 | 0.1961 | 14 | Perhaps here of take a jerk suggested jack do | wer | 0.6000 |
| 160 | streaming_asr | emilia_zh_0004841011 | 0.2326 | 17 | And all the she's way challenging and resurrecting He the friend count | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0004841011 | 0.3023 | 18 | And all the she's with challenging and resurrecting He the from count | wer | 0.6923 |
| 640 | streaming_asr | emilia_zh_0004841011 | 0.2279 | 16 | And all the use the with churning and resurrecting had the from count | wer | 0.6154 |
| 1280 | streaming_asr | emilia_zh_0004841011 | 0.1767 | 14 | And all the use the with churning and resurrecting had the from count | wer | 0.6154 |
| 160 | causal_full_asr | emilia_zh_0004841554 | 0.2023 | 16 | I seek to use me, but can you tell us whether purpose just come down from London? | wer | 0.4118 |
| 320 | causal_full_asr | emilia_zh_0004841554 | 0.2062 | 15 | I say excuse me, but can you tell us where the purpose just come down from London? | wer | 0.2941 |
| 640 | causal_full_asr | emilia_zh_0004841554 | 0.2140 | 22 | I say excuse me, but can you tell us whether purpose just come down from London? | wer | 0.2353 |
| 1280 | causal_full_asr | emilia_zh_0004841554 | 0.1751 | 18 | I say excuse me, but can you tell us whether purpose just come down from London? | wer | 0.2353 |
| 160 | streaming_asr | emilia_zh_0004927721 | 0.1568 | 12 | 一个人只有在城府于一个能量时才能了解他啊 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0004927721 | 0.1653 | 14 | 一个人只有在臣服于于一个能量时才能了解他啊 | cer | 0.2105 |
| 640 | streaming_asr | emilia_zh_0004927721 | 0.1822 | 14 | 一个人持有在臣服于就这个能量时才能了解他啊 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0004927721 | 0.1737 | 13 | 一个人持有在臣服就这个能量时才能了解他啊 | cer | 0.2105 |
| 160 | streaming_asr | emilia_zh_0005181378 | 0.1932 | 11 | 不是客气你阶段我的基本上其实你带回去 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0005181378 | 0.2443 | 14 | 不是客气你记得我的基本上其实你带回去 | cer | 0.3889 |
| 640 | streaming_asr | emilia_zh_0005181378 | 0.2273 | 14 | 不是客气你记得我的基本上说其实你带回去 | cer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0005181378 | 0.2386 | 14 | 不是客气你姐姐我的基本上其实你带回去 | cer | 0.3889 |
| 160 | streaming_asr | emilia_zh_0005507035 | 0.2086 | 13 | 那么政府还会收回一个决定了的都跟心 | cer | 0.3810 |
| 320 | streaming_asr | emilia_zh_0005507035 | 0.2246 | 12 | 那么本政府还会收回一个决定了大都跟相信 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005507035 | 0.2353 | 14 | 那么本政府还会收回一个决定了大的的安心 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005507035 | 0.2299 | 15 | 那么日本政府还会收回一个决定了大都。安心 | cer | 0.2381 |
| 160 | causal_full_asr | emilia_zh_0005600573 | 0.2045 | 20 | 参与的话是一千多然后直播间里面是十人 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005600573 | 0.2268 | 18 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005600573 | 0.2416 | 17 | 参与的话是一千多然后直播间里面是十人 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005600573 | 0.2268 | 16 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 160 | streaming_asr | emilia_zh_0005749601 | 0.2584 | 32 | 是说刚才说的那个五点啊就五点配套的在稍微把他总结一下滑第一点就是认知蒙蒙不足 | cer | 0.2955 |
| 320 | streaming_asr | emilia_zh_0005749601 | 0.2713 | 31 | 就什么刚才说的那个无底啊就无点配套的在稍微把他东西一下滑低点优势认知懵不足 | cer | 0.5227 |
| 640 | streaming_asr | emilia_zh_0005749601 | 0.2765 | 32 | 就什么刚才说的那个无底啊就五点配套的在稍微把他东西一下花低点优势认知懵不足 | cer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0005749601 | 0.2894 | 32 | 就什么刚才是的那个无底啊就五点配套的在稍微把他东西一下花第一点就是认知功能不足 | cer | 0.3864 |
| 160 | streaming_asr | emilia_zh_0005999475 | 0.2225 | 31 | 嗯这生活上你没有感觉特别的过因为我听说今年全球的费神之下然后现在现在变异常的过也 | cer | 0.4118 |
| 320 | streaming_asr | emilia_zh_0005999475 | 0.2454 | 36 | 然后社会生活你没有感觉特别的过因为我听说今年全球的费身之下然后现在现在变异常的国也 | cer | 0.4706 |
| 640 | streaming_asr | emilia_zh_0005999475 | 0.2500 | 35 | 嗯在生活中你没有感觉特别的过因为我听说今年全球的飞升之下然后现在现在变异常的过也 | cer | 0.4118 |
| 1280 | streaming_asr | emilia_zh_0005999475 | 0.2294 | 37 | 嗯在生活中你没有感觉特别的过进入我听说今年全球对飞升之下然后现在现在变异常的过也 | cer | 0.4706 |
| 160 | causal_full_asr | emilia_zh_0006041799 | 0.1982 | 13 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 320 | causal_full_asr | emilia_zh_0006041799 | 0.2070 | 15 | 在这个委托发生的时间前两个人就合伙开了这家面馆 | cer | 0.1250 |
| 640 | causal_full_asr | emilia_zh_0006041799 | 0.1630 | 12 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 1280 | causal_full_asr | emilia_zh_0006041799 | 0.1674 | 14 | 在这个委托发生的时间前两个人就合伙开了这家面馆 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0006212201 | 0.2609 | 17 | 没有办法一直坚持别说的动所以买通那个忽然 | cer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0006212201 | 0.2609 | 14 | 他没有办法一直坚持别说的动所以买通你个胡人 | cer | 0.3043 |
| 640 | streaming_asr | emilia_zh_0006212201 | 0.2464 | 17 | 没有办法一直坚持结束的移动作为买通的个胡人 | cer | 0.4783 |
| 1280 | streaming_asr | emilia_zh_0006212201 | 0.2464 | 15 | 没有办法一直坚持别说的移动所以买通的个胡人 | cer | 0.3478 |
| 160 | streaming_asr | emilia_zh_0006366492 | 0.1111 | 30 | He Just no want say some gaw his great cherish He want to return and they with men These Choose will give may great power | wer | 0.4231 |
| 320 | streaming_asr | emilia_zh_0006366492 | 0.1412 | 29 | He Just no want say some gaw his great cherish He want to return and let with men These chews will give may great power | wer | 0.4231 |
| 640 | streaming_asr | emilia_zh_0006366492 | 0.1281 | 28 | He Just no want say some gaw his great <\|bicodec_global_2703\|><\|eng\|><\|start_content\|>she He want to return and let with men These retudes will give may great power | wer | 0.4231 |
| 1280 | streaming_asr | emilia_zh_0006366492 | 0.1036 | 24 | He the not want to say some gaw his great treasures He want to return and let with men These retches will give may great power | wer | 0.3462 |
| 160 | streaming_asr | emilia_zh_0006446583 | 0.1923 | 27 | Now will can use am alright imaging see would is Actually hopping insight join what some one cracks nuckles | wer | 0.6500 |
| 320 | streaming_asr | emilia_zh_0006446583 | 0.1813 | 27 | Now will can use am alright imaging see would is actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0006446583 | 0.1758 | 26 | Now will can use am sorry. imaging see would is Actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 1280 | streaming_asr | emilia_zh_0006446583 | 0.1703 | 24 | Now will can use am sorry. imaging see would is Actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 160 | causal_full_asr | emilia_zh_0006502797 | 0.2517 | 39 | 我觉得这个委托人可我觉得那你说古代的时候和他的上头那古人都应该挖到比如说这个明代开始就要扔核桃嗯你你翻不出头 | cer | 0.4590 |
| 320 | causal_full_asr | emilia_zh_0006502797 | 0.2628 | 35 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就挖进核桃嗯你你发明出套 | cer | 0.3279 |
| 640 | causal_full_asr | emilia_zh_0006502797 | 0.2762 | 37 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就把这套嗯给发明出套了 | cer | 0.3443 |
| 1280 | causal_full_asr | emilia_zh_0006502797 | 0.2539 | 41 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就把这手套嗯你你发明手套 | cer | 0.3279 |
| 160 | streaming_asr | emilia_zh_0006610357 | 0.2480 | 27 | 今天觉得是非常确实关于非常只想但是似乎默默的这种做法他其实也并没有什么错觉得 | cer | 0.2791 |
| 320 | streaming_asr | emilia_zh_0006610357 | 0.2693 | 31 | 今天觉得是非常确实关键性非常之前但是似乎模模的这种做法他其实也并没有什么错我觉得 | cer | 0.2093 |
| 640 | streaming_asr | emilia_zh_0006610357 | 0.2693 | 31 | 今天觉得是非常确实关键性非常之前但是似乎模模的这种做法他其实也并没有错我觉得 | cer | 0.2558 |
| 1280 | streaming_asr | emilia_zh_0006610357 | 0.2507 | 31 | 这觉得是非常确实关键信息非常之前但是似乎做梦的这种做法他其实也并没有错我觉得 | cer | 0.2558 |
| 160 | streaming_asr | emilia_zh_0006883085 | 0.2453 | 16 | 在于终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 320 | streaming_asr | emilia_zh_0006883085 | 0.2075 | 17 | 在与终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0006883085 | 0.2217 | 17 | 在与终于停下几我都已经有点昏昏欲睡来了 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0006883085 | 0.2028 | 16 | 在与终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 160 | streaming_asr | emilia_zh_0007121845 | 0.1728 | 25 | 那么我们会也许只是依赖于一个技术比个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 320 | streaming_asr | emilia_zh_0007121845 | 0.1640 | 24 | 那么我们会也许只是依赖于一个技术bigger方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.1463 |
| 640 | streaming_asr | emilia_zh_0007121845 | 0.1693 | 26 | 那么我们会也许只是一烂鱼一个技术bigger方法一个优势或者是资源一些本地我们就去对抗市场 | cer | 0.2683 |
| 1280 | streaming_asr | emilia_zh_0007121845 | 0.1675 | 26 | 那么我们会也许只是一烂鱼一个技术比个方法一个优势或者是资源一些本地我们就去对抗市场 | cer | 0.1463 |
| 160 | streaming_asr | emilia_zh_0007399682 | 0.1637 | 22 | 有些上甚至一更加严厉的态度精神是我们咬认清生命的脆弱告诉我没一个人 | cer | 0.2353 |
| 320 | streaming_asr | emilia_zh_0007399682 | 0.1615 | 24 | 有些上甚至一更加而言的态度精神我们要认清生命的脆弱告诉我没一个人 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0007399682 | 0.1438 | 23 | 有些上甚至一更加而言的态度精神是我们要认清生命脆弱告诉我没一个人 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0007399682 | 0.1460 | 22 | 有些上甚至一更加眼里的态度精神是我们要认清生命脆弱告诉我没一个人 | cer | 0.2941 |
| 160 | streaming_asr | emilia_zh_0007635379 | 0.2727 | 15 | 所以来比是对于果然来说是一个很重要的是 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0007635379 | 0.2929 | 13 | 所以来比是对于德国人来说是一个很重要的是 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0007635379 | 0.2727 | 14 | 所以来比是对于中国人来说是一个很重要是 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007635379 | 0.2828 | 17 | 所以来比是对于中国人来说是一个很重要是 | cer | 0.2857 |
| 160 | causal_full_asr | emilia_zh_0007761003 | 0.1673 | 16 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0007761003 | 0.2007 | 17 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 640 | causal_full_asr | emilia_zh_0007761003 | 0.1673 | 18 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0007761003 | 0.1636 | 19 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0007790200 | 0.1376 | 34 | 特别就是了现在是小心私营企业卖厂的最佳时期大型所有实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007790200 | 0.1570 | 33 | 特别就是了现在是小心私营企业卖的最佳时机大型。实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2045 |
| 640 | streaming_asr | emilia_zh_0007790200 | 0.1499 | 32 | 特别就是了现在是小心私营企业卖的最佳时机大型。实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2045 |
| 1280 | streaming_asr | emilia_zh_0007790200 | 0.1570 | 34 | 特别指了现在是小心私营企业卖的最佳时机大型实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.1818 |
| 160 | streaming_asr | EN_B00013_S05834_W000745 | 0.2356 | 28 | First the then like could still stay stay was straw resilience the broken bone in it's chests with make make lose any build a to five to | wer | 0.6800 |
| 320 | streaming_asr | EN_B00013_S05834_W000745 | 0.2356 | 28 | The if then like could still stay for was strong was resilience the broken bone in it's chast which make make lose any build a to five to | wer | 0.6400 |
| 640 | streaming_asr | EN_B00013_S05834_W000745 | 0.2270 | 31 | First the from like could still stay through was strong <\|glm_semantic_298\|>ions the broken bone in it's chest which make and lose any build a to five to | wer | 0.6000 |
| 1280 | streaming_asr | EN_B00013_S05834_W000745 | 0.1897 | 31 | Even in the like could still stay through was strong was resilience the broken bone some is chest which make and lose any build a to five to | wer | 0.5600 |
| 160 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2248 | 15 | Ride your skateboard, said his dad, it's two slipboards at Adam. | wer | 0.5455 |
| 320 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2431 | 14 | Riding a skateboard set is dead. It's two slipboards at Adam. | wer | 0.8182 |
| 640 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2294 | 15 | Ride your skateboard, said his dad. It's two slippery slides at that. | wer | 0.5455 |
| 1280 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2064 | 15 | Ride your skateboard, said his dad, it's two slippers and Adam | wer | 0.4545 |
| 160 | streaming_asr | EN_B00048_S02289_W000002 | 0.1036 | 9 | This kind of mussel is Mostly connected to my bones | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S02289_W000002 | 0.1143 | 14 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 640 | streaming_asr | EN_B00048_S02289_W000002 | 0.1071 | 15 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 1280 | streaming_asr | EN_B00048_S02289_W000002 | 0.1036 | 14 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 160 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1401 | 28 | Every two months my coworkers and I would come together to discuss the news of my schedule. Our meetings were usually held in the staff room at our institute | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1419 | 36 | Every two months my coworkers and I would come together to discuss the news semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0714 |
| 640 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1505 | 35 | Every two months my coworkers and I would come together to discuss the new semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 1280 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1401 | 32 | Every two months my coworkers and I would come together to discuss the new semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 160 | streaming_asr | EN_B00048_S09662_W000003 | 0.1157 | 25 | Um here again So what don't you listen to the .log board the first time that listen how presented hasn't heard has goodbye and and could back and the but words | wer | 0.5625 |
| 320 | streaming_asr | EN_B00048_S09662_W000003 | 0.1550 | 30 | I'll pay here good so was don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm the that's listen how President hasn't known has goodbye. and and could back and the but words | wer | 0.6250 |
| 640 | streaming_asr | EN_B00048_S09662_W000003 | 0.1550 | 29 | I'll pay here great so what don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm that's listen how President hasn't had has goodbye. and and could back and the could words | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00048_S09662_W000003 | 0.1426 | 32 | I'll pay I'll pay good so what don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm that's listen how President hasn't had has goodbye. and and could back and the could words | wer | 0.6250 |
| 160 | streaming_asr | EN_B00058_S06165_W000019 | 0.1800 | 56 | We are you you the short for where you add seem like a colon after the constructor in the in you use This Western text in now the first argument to pass to to west constructor will be store in the question text properity | wer | 0.4634 |
| 320 | streaming_asr | EN_B00058_S06165_W000019 | 0.1825 | 56 | Where you use the short form where you add seem like a colon after the constructor in in you use this question text in now the first argament to pass to to west constructor will be store in the question text properity | wer | 0.3659 |
| 640 | streaming_asr | EN_B00058_S06165_W000019 | 0.1886 | 55 | Or you use the short form where you at seemingly cold after the constructor in the in you use this question text in now the first argament to pass to to west constructor will be store in the quest text properity | wer | 0.3902 |
| 1280 | streaming_asr | EN_B00058_S06165_W000019 | 0.1837 | 58 | Or you use the short form where you at seemingly cold and after the constructor in the in you use This question text in now the first argment to pass to to west constructor will be store in the quest text properity | wer | 0.4146 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
