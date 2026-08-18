# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8887**
- Weighted CTC blank ratio: **0.1851**
- Weighted streaming WER/CER: **0.4058**
- Weighted causal-full WER/CER: **0.5066**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000068601 | 0.1958 | 25 | The all high all high leaving collects documents on the his he of the mice we were released | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000068601 | 0.2264 | 28 | The all high all high leaving with ease collects 文档 on the his three he all the eyes we were released | wer | 1.4167 |
| 640 | streaming_asr | CommonVoice_EN_0000068601 | 0.2500 | 34 | The a of leaving a collects dominals on the he's free he are the mice we were released | wer | 1.0833 |
| 1280 | streaming_asr | CommonVoice_EN_0000068601 | 0.2288 | 31 | The a of leaving a collects domnance on the his three he of organized Liberalize | wer | 0.6667 |
| 160 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1136 | 13 | Then he's an old man. He's going to spend a month in Africa | wer | 0.1538 |
| 320 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1364 | 13 | Then he's an old man, he's going to spend a month in Africa | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1773 | 16 | Then he's an old man. He's going to spend a month in Africa | wer | 0.1538 |
| 1280 | causal_full_asr | CommonVoice_EN_0000126435 | 0.2091 | 16 | When he's an old man he's going to spend a month in Africa | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000263325 | 0.0830 | 7 | He as one under brother travert re- nelson | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000263325 | 0.1037 | 10 | He as one under brother travert re- nelson | wer | 0.5000 |
| 640 | streaming_asr | CommonVoice_EN_0000263325 | 0.1162 | 12 | He as one younger brother travert re- nelson | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000263325 | 0.1369 | 11 | He was one young brother travert re- Nelson | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000381462 | 0.1767 | 17 | Girard spent to this of s life with mysticism | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000381462 | 0.2241 | 15 | Girard spent the this of s life with Mr. Sis | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000381462 | 0.1940 | 18 | Girard spent to that of s life with Mr. Sis | wer | 0.5556 |
| 1280 | streaming_asr | CommonVoice_EN_0000381462 | 0.1509 | 14 | Girard spent the this of s life with Mr. Sis | wer | 0.4444 |
| 160 | streaming_asr | CommonVoice_EN_0000555807 | 0.0991 | 12 | I color <\|glm_semantic_14383\|>ence of as as stop right | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000555807 | 0.1651 | 15 | The color intensifies as as stop right | wer | 0.4286 |
| 640 | streaming_asr | CommonVoice_EN_0000555807 | 0.1745 | 16 | The color intensifies is this stop brightens | wer | 0.4286 |
| 1280 | streaming_asr | CommonVoice_EN_0000555807 | 0.1981 | 19 | The colour intensifies as this stop brightens | wer | 0.4286 |
| 160 | causal_full_asr | CommonVoice_EN_0000593818 | 0.1286 | 15 | If it wasn't difficult, get one be a problem | wer | 0.3333 |
| 320 | causal_full_asr | CommonVoice_EN_0000593818 | 0.1429 | 17 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 640 | causal_full_asr | CommonVoice_EN_0000593818 | 0.2429 | 24 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 1280 | causal_full_asr | CommonVoice_EN_0000593818 | 0.2857 | 21 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 160 | streaming_asr | LibriSpeech_0000035237 | 0.1218 | 35 | In at country very very the great dear of mass it covers the ground just just grass does here But most interesting f interesting about is slamings is the wave in migrated | wer | 0.4688 |
| 320 | streaming_asr | LibriSpeech_0000035237 | 0.1269 | 34 | In at country <\|glm_semantic_2425\|> res the great dear of must it covers the ground just just rest does here But most interesting f interesting about one slamings is the wave in migrated | wer | 0.5000 |
| 640 | streaming_asr | LibriSpeech_0000035237 | 0.1168 | 34 | In at country <\|glm_semantic_2425\|> re the great dear of mass it covers the ground just just rest does here But most interesting f interesting about it slamings is the wave in my great | wer | 0.5312 |
| 1280 | streaming_asr | LibriSpeech_0000035237 | 0.1168 | 36 | In at country <\|glm_semantic_2425\|> re the great dear of mass it covers the ground just just rest does here But most interesting they about it slamming is the wave a my great | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000124551 | 0.1567 | 46 | You platter fellow as a and the salad were but to begin breakfast who and was discovered that was of it's member's was missing hendry was the absent one at first the was but the not take in of circumstance | wer | 0.5128 |
| 320 | streaming_asr | LibriSpeech_0000124551 | 0.1681 | 52 | You platter families some and sail were both to begin breakfast who and was discovered the was of it member's was missing hendry was the absent one at first the was but let not take in of circumstance | wer | 0.5385 |
| 640 | streaming_asr | LibriSpeech_0000124551 | 0.1695 | 51 | You planchers found as simple and the salad were both to begin breakfast who and was discovered the was a but member was missing hendry was the absent one I first the was but let not take in of circumstance | wer | 0.5897 |
| 1280 | streaming_asr | LibriSpeech_0000124551 | 0.1638 | 53 | The Plants found assembled and the salad were both to begin Breakfast who and was discovered that was a but member was missing hendry was the absent one I first the was but let not take in of circumstance | wer | 0.4872 |
| 160 | causal_full_asr | LibriSpeech_0000263750 | 0.1424 | 32 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk ready | wer | 0.2000 |
| 320 | causal_full_asr | LibriSpeech_0000263750 | 0.1524 | 36 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1500 |
| 640 | causal_full_asr | LibriSpeech_0000263750 | 0.1407 | 37 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1500 |
| 1280 | causal_full_asr | LibriSpeech_0000263750 | 0.1407 | 35 | A magnificent person with powdered hair, breeches and silk stockings presented himself, Lord Reginald Sedley he announced in Walk Reggie | wer | 0.2000 |
| 160 | streaming_asr | VCTK_0000006134 | 0.1294 | 11 | This influence the the dinned in that can air | wer | 0.6667 |
| 320 | streaming_asr | VCTK_0000006134 | 0.1471 | 12 | The influence the a ins in that can air | wer | 0.8889 |
| 640 | streaming_asr | VCTK_0000006134 | 0.1529 | 12 | They influence to the to dinned and that can arrow | wer | 0.6667 |
| 1280 | streaming_asr | VCTK_0000006134 | 0.1588 | 11 | This influence to the to int and that can arrow | wer | 0.5556 |
| 160 | streaming_asr | emilia_zh_0004064583 | 0.2181 | 24 | 我相信没因巧合都是一道自信一线索告诉男 | cer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0004064583 | 0.2150 | 22 | 我相信没一些巧合都是一道资讯一线索告诉男人 | cer | 0.2273 |
| 640 | streaming_asr | emilia_zh_0004064583 | 0.2150 | 24 | 我相信没一些巧合都是一道自信疑线索告诉难 | cer | 0.3636 |
| 1280 | streaming_asr | emilia_zh_0004064583 | 0.1963 | 23 | 我相信没一次巧合都是一道自信一现告诉难 | cer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0004212211 | 0.1376 | 18 | 这里暴露的显而易见的挫败感感觉源于孩儿的个人经历而不是政治信念本身 | cer | 0.1613 |
| 320 | streaming_asr | emilia_zh_0004212211 | 0.1590 | 20 | 这里表路的写一件的挫败感感觉源于卡尔的个人经历而不是政治新闻本身 | cer | 0.3226 |
| 640 | streaming_asr | emilia_zh_0004212211 | 0.1621 | 21 | 这里表率的显而易见的挫败来源于卡尔的个人经历而不是政治性面本身 | cer | 0.1613 |
| 1280 | streaming_asr | emilia_zh_0004212211 | 0.1529 | 18 | 这里表率的显而易见的挫败来源于卡尔的个人经历而不是政治性面本身 | cer | 0.1613 |
| 160 | causal_full_asr | emilia_zh_0004422836 | 0.1544 | 37 | 精神讯犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行接戏在四大中失守之命案当时编曲的法令 | cer | 0.3137 |
| 320 | causal_full_asr | emilia_zh_0004422836 | 0.1728 | 39 | 精神训犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行接戏在四大中失守之命案当时编剧的法令 | cer | 0.3333 |
| 640 | causal_full_asr | emilia_zh_0004422836 | 0.1483 | 39 | 精神训犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行且戏在私打中失守致命案当时编剧的法令 | cer | 0.2745 |
| 1280 | causal_full_asr | emilia_zh_0004422836 | 0.1606 | 40 | 精神讯犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行且戏在私打中失守致命案当时编剧的法令 | cer | 0.2549 |
| 160 | streaming_asr | emilia_zh_0004519522 | 0.2671 | 10 | 这这个兄王的客人那呃是弱点来 | cer | 0.4667 |
| 320 | streaming_asr | emilia_zh_0004519522 | 0.2857 | 12 | 就这个兄王的客人那呃是昨天来 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004519522 | 0.2609 | 15 | 这个兄王的客人那呃是昨天来 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004519522 | 0.2298 | 12 | 这个姓王的客人那呃是昨天来 | cer | 0.2667 |
| 160 | causal_full_asr | emilia_zh_0004705832 | 0.1820 | 45 | These words are automatically no longer visualized which is why it takes kids longer to think because they're still visualizing the words it's not a mechanic anymore which also explains why kids live on kids | wer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0004705832 | 0.1758 | 46 | These words are automatic we no longer visualize it, which is why it takes kids a longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1667 |
| 640 | causal_full_asr | emilia_zh_0004705832 | 0.1774 | 45 | These words are automatic we no longer visualize it, which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0004705832 | 0.1575 | 47 | These words are automatic and no longer visualize it, which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1944 |
| 160 | streaming_asr | emilia_zh_0004724727 | 0.1714 | 27 | is it ren't you going to you all but a the so for Today the new things is a a with a a All all a new | wer | 0.4783 |
| 320 | streaming_asr | emilia_zh_0004724727 | 0.1844 | 25 | Is it redeemant yeah all but a the some for Today the new things is a a with a All all new | wer | 0.4348 |
| 640 | streaming_asr | emilia_zh_0004724727 | 0.1974 | 26 | is it redeemable yeah how but a the some for Today the new things is a a with a All all a new | wer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0004724727 | 0.1866 | 26 | Is it raditional yeah how but a the some for Today the new things is a a with a All the a new | wer | 0.3913 |
| 160 | streaming_asr | emilia_zh_0004804632 | 0.2184 | 14 | It this of the a are side when Several live these of some more gether | wer | 0.6429 |
| 320 | streaming_asr | emilia_zh_0004804632 | 0.2069 | 14 | It is after a are side when Several of these of the more gether | wer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0004804632 | 0.1724 | 13 | It is after a red side when Several of these of so more gether | wer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0004804632 | 0.1552 | 14 | It is upt a are side when seventh of these of so more gether | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004880227 | 0.2326 | 21 | I is always member the hours I spend who the master the the house some busher | wer | 0.4375 |
| 320 | streaming_asr | emilia_zh_0004880227 | 0.2558 | 20 | I is always member the our I spent who some master the the house some busher | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004880227 | 0.2442 | 22 | I is always remember the hours I spent who the mass the the house some Thus | wer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0004880227 | 0.2132 | 18 | I it always remember the hours I spent who the master the the how s Thus | wer | 0.3750 |
| 160 | streaming_asr | emilia_zh_0005094293 | 0.2322 | 16 | 对于托德过程来说已经是一很难做到的事情 | cer | 0.2857 |
| 320 | streaming_asr | emilia_zh_0005094293 | 0.2607 | 17 | 对于错工程学来说已经是以及很难做到的事情 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005094293 | 0.2749 | 16 | 都脱工程学来说已经是以及很难做到的事情的 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0005094293 | 0.2701 | 17 | 都吐工程学来说已经是以及很难做到的事情的 | cer | 0.2857 |
| 160 | causal_full_asr | emilia_zh_0005313767 | 0.2308 | 22 | 你有五秒钟的时间阅读第一小题带有相关内容 | cer | 0.1053 |
| 320 | causal_full_asr | emilia_zh_0005313767 | 0.2344 | 24 | You need to read the first question in five seconds. | cer | 2.2632 |
| 640 | causal_full_asr | emilia_zh_0005313767 | 0.2198 | 22 | You have five seconds to read the relevant content. | cer | 2.2632 |
| 1280 | causal_full_asr | emilia_zh_0005313767 | 0.2125 | 23 | You have five seconds to read the relevant content. | cer | 2.2632 |
| 160 | streaming_asr | emilia_zh_0005420605 | 0.1633 | 10 | 就刚查是考一这这大一遍过太行的 | cer | 0.6471 |
| 320 | streaming_asr | emilia_zh_0005420605 | 0.1429 | 10 | 有的港口是靠一这这的一遍过太行的 | cer | 0.4706 |
| 640 | streaming_asr | emilia_zh_0005420605 | 0.1531 | 9 | 有的光卡是靠一这这的一遍过太行的 | cer | 0.4118 |
| 1280 | streaming_asr | emilia_zh_0005420605 | 0.1684 | 12 | 有的光卡纸靠一这这到一遍过太行的 | cer | 0.4118 |
| 160 | streaming_asr | emilia_zh_0005670671 | 0.1906 | 25 | 时至今日切在人们社会的各主中所办的角度意见无代替其的生 | cer | 0.3939 |
| 320 | streaming_asr | emilia_zh_0005670671 | 0.1931 | 24 | 时至今日切在人们社会的各主中所败的角度已经无代替企业的生 | cer | 0.3636 |
| 640 | streaming_asr | emilia_zh_0005670671 | 0.2104 | 23 | 时至今日切在人们社会的各组主中所败的角度已经无代替企业的生 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005670671 | 0.2203 | 25 | 时至今日企业在人们社会的各组织中所败的角度一节无代替企业的生 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0005905391 | 0.1991 | 11 | 的这影响那感觉是什么了就是麦当劳 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005905391 | 0.1948 | 12 | 的是影响那感觉是是什么了就是麦当劳 | cer | 0.4375 |
| 640 | streaming_asr | emilia_zh_0005905391 | 0.1775 | 12 | 的之影响的感觉是是什么了就是麦当劳 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0005905391 | 0.1515 | 15 | 之影响的感觉是是什么了就是麦当劳 | cer | 0.3750 |
| 160 | causal_full_asr | emilia_zh_0005960185 | 0.2599 | 17 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 320 | causal_full_asr | emilia_zh_0005960185 | 0.2863 | 19 | 哦我知道一块但是那反正很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 640 | causal_full_asr | emilia_zh_0005960185 | 0.2643 | 21 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 1280 | causal_full_asr | emilia_zh_0005960185 | 0.2643 | 21 | 哦我知道一块但是那番很多人觉得不是玫瑰是天主葵 | cer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0006119250 | 0.2361 | 20 | 一就是呢我本来啊天蓬说朋友啊旁嘛旁然后还有点儿事儿 | cer | 0.4828 |
| 320 | streaming_asr | emilia_zh_0006119250 | 0.2262 | 24 | 一就是我呢我本来啊天蓬说吧朋友捧马捧然后还说点事儿 | cer | 0.4828 |
| 640 | streaming_asr | emilia_zh_0006119250 | 0.2426 | 23 | 一就是如果呢我本来啊肩膀说朋友啊捧捧捧啊，然后还有点儿事儿 | cer | 0.6207 |
| 1280 | streaming_asr | emilia_zh_0006119250 | 0.2393 | 22 | 一就是说那我本来啊天蓬说朋友啊捧捧捧啊，然后还说点事儿 | cer | 0.4828 |
| 160 | streaming_asr | emilia_zh_0006350396 | 0.1614 | 25 | I'm on found be position a the 'clock with spent a you okay And we found logins nabas they | wer | 0.8235 |
| 320 | streaming_asr | emilia_zh_0006350396 | 0.1582 | 20 | My on found be position and the cluck with spent a in okay And we found logins they are they are | wer | 0.8824 |
| 640 | streaming_asr | emilia_zh_0006350396 | 0.1424 | 21 | My I am found be position and the cluck with spend the and okay And we found logins they are they are | wer | 0.8824 |
| 1280 | streaming_asr | emilia_zh_0006350396 | 0.1234 | 19 | I'm I on found be position a a 'clock with spend the and okay And we found logins they are they are | wer | 0.8824 |
| 160 | causal_full_asr | emilia_zh_0006404958 | 0.2464 | 22 | The characters are like monkeys in winter sickening up withered wines, whitening water. | wer | 0.4667 |
| 320 | causal_full_asr | emilia_zh_0006404958 | 0.2429 | 24 | The characters are like monkeys in winter sickening up withered wines, watching in water. | wer | 0.4667 |
| 640 | causal_full_asr | emilia_zh_0006404958 | 0.2286 | 21 | The characters are like monkeys in winter sickening up withered wines, watching in water. | wer | 0.4667 |
| 1280 | causal_full_asr | emilia_zh_0006404958 | 0.1821 | 19 | The characters are like monkeys in winter, sickening up, rid of the wines, watching in water. | wer | 0.6000 |
| 160 | streaming_asr | emilia_zh_0006435274 | 0.1283 | 14 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0006435274 | 0.1460 | 17 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0006435274 | 0.1018 | 13 | I bank charges interest are brother doesn't charge interested | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0006435274 | 0.1195 | 15 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006544707 | 0.1421 | 22 | 这两人那也是在你说这个去见有登记以后呢现象而且这两两人其中有一个 | cer | 0.2647 |
| 320 | streaming_asr | emilia_zh_0006544707 | 0.1395 | 22 | 这了人那也是在你说这个区间有登记以后呢现象而且这两两人其中有一个 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0006544707 | 0.1342 | 22 | 这两个人那也是在你说的这个区间有经济结婚现象而且这两两人其中有一个 | cer | 0.1471 |
| 1280 | streaming_asr | emilia_zh_0006544707 | 0.1368 | 21 | 这两人那也是在你说的这个区间有登记结婚现象而这两两人其中有一个 | cer | 0.1471 |
| 160 | streaming_asr | emilia_zh_0006874664 | 0.2236 | 34 | 那其中他有说到他非常成的就是希望如我说这个饮水没如果是查沙怎么样在对这段要说里面有觉得不要真实 | cer | 0.3077 |
| 320 | streaming_asr | emilia_zh_0006874664 | 0.2257 | 37 | 那其中他有说到他非常成的就是希望如果说这个饮水没有如果是查沙怎么样在对这段要说里面有觉得比较真实 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0006874664 | 0.2278 | 37 | 那其中他有说到他非常成的就是希望如果说这个谁谁没如果是查沙怎么样在对这的要说里面有觉得比较真实 | cer | 0.2885 |
| 1280 | streaming_asr | emilia_zh_0006874664 | 0.2173 | 36 | 那其中他有说到他非常成的就是希望如果说这个谁谁没如果是查沙怎么样在对这的要说里面有觉得比较真实 | cer | 0.2885 |
| 160 | streaming_asr | emilia_zh_0007060532 | 0.3606 | 17 | 如果你对保格人偶间舞在就比如嘴说 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0007060532 | 0.3029 | 17 | 如果你对报告人偶件舞这样就比如嘴说 | cer | 0.5263 |
| 640 | streaming_asr | emilia_zh_0007060532 | 0.3029 | 15 | 如果你对表格人偶件是舞真爱就比如嘴说 | cer | 0.4211 |
| 1280 | streaming_asr | emilia_zh_0007060532 | 0.3125 | 17 | 如果你对表格人物件物真爱就别用嘴说 | cer | 0.2632 |
| 160 | causal_full_asr | emilia_zh_0007060544 | 0.2125 | 8 | If what you do is truly valuable to others | cer | 2.2667 |
| 320 | causal_full_asr | emilia_zh_0007060544 | 0.2125 | 10 | If what you do is truly valuable to others | cer | 2.2667 |
| 640 | causal_full_asr | emilia_zh_0007060544 | 0.1938 | 12 | If what you do is truly valuable to others | cer | 2.2667 |
| 1280 | causal_full_asr | emilia_zh_0007060544 | 0.1875 | 12 | If what you do is truly valuable to others | cer | 2.2667 |
| 160 | streaming_asr | emilia_zh_0007312674 | 0.1400 | 21 | 事实上轰隆也不需要你亲爱的啊他根本知道自己现在河地也认不出身边是谁和他大家在一起 | cer | 0.2051 |
| 320 | streaming_asr | emilia_zh_0007312674 | 0.1460 | 22 | 事实上轰隆也不需要你亲爱的啊他根本知道自己现在河地也认不出身边是谁谁他大家在一起 | cer | 0.2308 |
| 640 | streaming_asr | emilia_zh_0007312674 | 0.1643 | 25 | 事实上婚礼也不需要你亲爱的的他根本知道自己现在和地也认不出身边是谁和他大家在一起 | cer | 0.2051 |
| 1280 | streaming_asr | emilia_zh_0007312674 | 0.1542 | 26 | 事实上轰隆也不需要你亲爱的的他根本知道自己生在和地也认不出身边是谁和他带在一起 | cer | 0.1795 |
| 160 | streaming_asr | emilia_zh_0007551053 | 0.2913 | 25 | 七还更多工作是我律师是我所在又工作经验来前台前辈们进行 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007551053 | 0.2880 | 25 | 七更多工作是我律师所所在又工作经验的前台前辈们进行 | cer | 0.2857 |
| 640 | streaming_asr | emilia_zh_0007551053 | 0.3204 | 26 | 其啊更多的工作是有律师所所在有工作经验的前台前辈们进行 | cer | 0.1786 |
| 1280 | streaming_asr | emilia_zh_0007551053 | 0.3010 | 27 | 七啊更多的工作是有律师思索的有工作经验的前台前辈们进行 | cer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0007761299 | 0.2364 | 18 | 我觉得这可能不社会我我觉得我跟这的公共的图书出的开心 | cer | 0.3929 |
| 320 | streaming_asr | emilia_zh_0007761299 | 0.2509 | 18 | 我觉得这可能不适合我我觉得我跟这的公司的同事出的开心 | cer | 0.2143 |
| 640 | streaming_asr | emilia_zh_0007761299 | 0.2327 | 18 | 我觉得这行业。社会我我觉得我跟这的公的同事出的开心 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007761299 | 0.2364 | 20 | 我有这个行业。适合我我觉得我跟这的公是的同事出的开心 | cer | 0.2500 |
| 160 | streaming_asr | EN_B00043_S01954_W000033 | 0.1832 | 33 | Because was his more of the and rest in Motion it's this passity of Further and endlessly renov researched And not just and our private I this | wer | 0.7500 |
| 320 | streaming_asr | EN_B00043_S01954_W000033 | 0.1703 | 31 | Because love is more of the and rest in Motion it's this passity of Further and endlessly renov research And not just to not private I this | wer | 0.7083 |
| 640 | streaming_asr | EN_B00043_S01954_W000033 | 0.1466 | 30 | Because love is more of the and rest in Motion it's this passity of for and endlessly renov resourced a not is to our private I this | wer | 0.7500 |
| 1280 | streaming_asr | EN_B00043_S01954_W000033 | 0.1379 | 33 | Because love is more of the and rest in Motion it this passity of verb and endlessly renov resourced And not is to not private I this | wer | 0.7500 |
| 160 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2049 | 28 | Now we're now looking about VGG games or competent generated actors which looks tremendously realistic. | wer | 0.4286 |
| 320 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2294 | 28 | Now we're now thinking about video games or competing generated actors which looks tremendously realistic. | wer | 0.3571 |
| 640 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2049 | 27 | Now we're now talking about video games or computer generated actors which looks tremendously realistic. | wer | 0.2857 |
| 1280 | causal_full_asr | EN_B00043_S01957_W000002 | 0.1957 | 27 | Now we're now talking about video games or computer generated actors which looks tremendously realistic. | wer | 0.2857 |
| 160 | streaming_asr | EN_B00089_S03348_W000001 | 0.2551 | 29 | one the things that like do when each my introduction stronomy class this to be the cost the yes strawling picture rever the to | wer | 0.5517 |
| 320 | streaming_asr | EN_B00089_S03348_W000001 | 0.2346 | 27 | one the things that like do when each my introduce stronomy class his to be the class the yes strawling picture reve the a | wer | 0.5172 |
| 640 | streaming_asr | EN_B00089_S03348_W000001 | 0.2375 | 27 | one the things that like do when each my introduce stronomy class his to be the class the you strawling picture rever the they | wer | 0.5172 |
| 1280 | streaming_asr | EN_B00089_S03348_W000001 | 0.1760 | 26 | one the things that like do when each my introduction stronomy classes his to be then class the the strawling picture of the to | wer | 0.4483 |
| 160 | streaming_asr | EN_B00048_S07862_W000265 | 0.1835 | 27 | The me mean of crossing large Areas of water was in no sailing ship driven I the wind | wer | 0.2222 |
| 320 | streaming_asr | EN_B00048_S07862_W000265 | 0.1543 | 21 | The only means of crossing large Areas of water was in no sailing ship driven I the wind | wer | 0.1111 |
| 640 | streaming_asr | EN_B00048_S07862_W000265 | 0.1330 | 19 | The only main of crossing large Areas of water was in no selling ship driven I the wind | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00048_S07862_W000265 | 0.1410 | 20 | The only mean of crossing large Areas of water was it no sailing ship driven I the wind | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00048_S07870_W000001 | 0.2284 | 16 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 320 | causal_full_asr | EN_B00048_S07870_W000001 | 0.2099 | 13 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 640 | causal_full_asr | EN_B00048_S07870_W000001 | 0.1481 | 11 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 1280 | causal_full_asr | EN_B00048_S07870_W000001 | 0.1111 | 9 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 160 | streaming_asr | EN_B00058_S04429_W000016 | 0.1757 | 21 | The you see these they green leaves And this sping flower where I of second | wer | 0.4375 |
| 320 | streaming_asr | EN_B00058_S04429_W000016 | 0.1634 | 25 | The you see these they told green leagues And this being flower where I of second | wer | 0.5625 |
| 640 | streaming_asr | EN_B00058_S04429_W000016 | 0.1287 | 21 | The will see these they tell green leagues And this being flower where I of second | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00058_S04429_W000016 | 0.1262 | 20 | The you see these they tell green leagues And this being flower where I of second | wer | 0.5625 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
