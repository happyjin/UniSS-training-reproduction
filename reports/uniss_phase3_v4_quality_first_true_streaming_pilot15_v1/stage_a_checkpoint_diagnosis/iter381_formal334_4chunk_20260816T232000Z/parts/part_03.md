# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9254**
- Weighted CTC blank ratio: **0.8741**
- Weighted streaming WER/CER: **0.2598**
- Weighted causal-full WER/CER: **0.1316**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000068601 | 0.8986 | 18 | The Archival of livable ease collect domans on the his the organized obvious | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000068601 | 0.8892 | 21 | The Our of laborious collects domans on the heathery the almost obvious release | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000068601 | 0.8726 | 23 | The Archiving of liveries and collects dominence on the he three the organized obvious release | wer | 0.7500 |
| 1280 | streaming_asr | CommonVoice_EN_0000068601 | 0.8679 | 26 | The Archiving of leverals collect documents on the he three the almost Liberalise | wer | 0.6667 |
| 160 | causal_full_asr | CommonVoice_EN_0000126435 | 0.8409 | 21 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 320 | causal_full_asr | CommonVoice_EN_0000126435 | 0.8000 | 22 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000126435 | 0.7636 | 27 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 1280 | causal_full_asr | CommonVoice_EN_0000126435 | 0.7864 | 25 | When he's an old man he's going to spend the month in Africa | wer | 0.0769 |
| 160 | streaming_asr | CommonVoice_EN_0000263325 | 0.8506 | 18 | He has one under brother traveller read Nelson | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000263325 | 0.8589 | 20 | He has one under brother traveller read Nelson | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000263325 | 0.8631 | 20 | He as one young brother traveller read Nelson | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000263325 | 0.8382 | 22 | He was one younger brother traveller read Nelson | wer | 0.3750 |
| 160 | streaming_asr | CommonVoice_EN_0000381462 | 0.8103 | 23 | Jirard spent to this of this life with misters | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000381462 | 0.8017 | 22 | Chirrod Spent to this of this life would Mistresses | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000381462 | 0.8103 | 21 | Chir guides spent to just of this life would mystresses | wer | 0.7778 |
| 1280 | streaming_asr | CommonVoice_EN_0000381462 | 0.8060 | 23 | Jir tribes Spent to just of this life would Mistressus | wer | 0.7778 |
| 160 | streaming_asr | CommonVoice_EN_0000555807 | 0.9057 | 12 | The color intensifies as is stop right | wer | 0.4286 |
| 320 | streaming_asr | CommonVoice_EN_0000555807 | 0.9198 | 11 | The color intensifies as this star bright | wer | 0.2857 |
| 640 | streaming_asr | CommonVoice_EN_0000555807 | 0.9057 | 11 | The color intensifies as this star brightens | wer | 0.1429 |
| 1280 | streaming_asr | CommonVoice_EN_0000555807 | 0.8915 | 13 | The color intensifies as this star brightens | wer | 0.1429 |
| 160 | causal_full_asr | CommonVoice_EN_0000593818 | 0.9179 | 13 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000593818 | 0.9071 | 17 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000593818 | 0.9143 | 16 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000593818 | 0.9107 | 15 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000035237 | 0.8325 | 60 | In that country where res a great dear of mass it covers the ground just is grass does here But most interesting thing about these lemmings is the wave they migrated | wer | 0.2500 |
| 320 | streaming_asr | LibriSpeech_0000035237 | 0.8139 | 65 | In that country where res a great dear of mass it covers the ground just is grass does here But most interesting thing about the these lemmings is the wave they migrated | wer | 0.2812 |
| 640 | streaming_asr | LibriSpeech_0000035237 | 0.7868 | 73 | In that country where res a great dear of mass it covers the ground just is grass does here But most interesting thing about these lemmings is the wave they my great | wer | 0.2812 |
| 1280 | streaming_asr | LibriSpeech_0000035237 | 0.7800 | 73 | In that country where are a great deal of mass it covers the ground just is grass does here but most interesting thing about the these lemmings is the wave they my great | wer | 0.2812 |
| 160 | streaming_asr | LibriSpeech_0000124551 | 0.7350 | 96 | The platter family assemble and cell were about to begin breakfast We and it was discovered that what of it's member's was missing Henry was the absent one I first the was but little notice taking of circumstance | wer | 0.3590 |
| 320 | streaming_asr | LibriSpeech_0000124551 | 0.7251 | 101 | You platter family assemble in cell were about to begin breakfast When and it was discovered that what of it's member's was missing Henry was the absent one I first the was but little notice taking of the circumstance | wer | 0.3077 |
| 640 | streaming_asr | LibriSpeech_0000124551 | 0.7422 | 94 | Now plinters family assemble in cell were about to begin breakfast We're in it was discovered that what of it's member was missing Henry was the absent one I first their was but little notice taking of circumstance | wer | 0.3590 |
| 1280 | streaming_asr | LibriSpeech_0000124551 | 0.7336 | 96 | The plinters family assemble in cell were about to begin breakfast were in it was discovered that what of it's member's was missing Henry was the absent one I'd first their was but little notice taking of circumstance | wer | 0.3333 |
| 160 | causal_full_asr | LibriSpeech_0000263750 | 0.8241 | 50 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1500 |
| 320 | causal_full_asr | LibriSpeech_0000263750 | 0.8342 | 50 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1000 |
| 640 | causal_full_asr | LibriSpeech_0000263750 | 0.8191 | 52 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1000 |
| 1280 | causal_full_asr | LibriSpeech_0000263750 | 0.8141 | 52 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk ready | wer | 0.2000 |
| 160 | streaming_asr | VCTK_0000006134 | 0.7941 | 20 | There's influence to the the intend is like going to air | wer | 0.6667 |
| 320 | streaming_asr | VCTK_0000006134 | 0.8000 | 19 | This influence to the the intend is like going to arrow | wer | 0.4444 |
| 640 | streaming_asr | VCTK_0000006134 | 0.7588 | 21 | This influenced to the the int is like going arrow | wer | 0.4444 |
| 1280 | streaming_asr | VCTK_0000006134 | 0.7706 | 22 | This influenced to the the int is like can arrow | wer | 0.4444 |
| 160 | streaming_asr | emilia_zh_0004064583 | 0.9907 | 2 | 我下线没一次巧合都是一道咨询一个线索告诉我难 | cer | 0.2727 |
| 320 | streaming_asr | emilia_zh_0004064583 | 0.9813 | 3 | 我相信没一次巧合都是一道咨询一道线索告诉我难 | cer | 0.2273 |
| 640 | streaming_asr | emilia_zh_0004064583 | 0.9844 | 3 | 我相信没一次巧合都是一道自信一道线索告诉我难 | cer | 0.2273 |
| 1280 | streaming_asr | emilia_zh_0004064583 | 0.9907 | 3 | 我相信没一次巧合都是一道自信以线索告诉我难 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0004212211 | 0.9572 | 9 | 这里表露的显而易见的的挫败感感觉源于卡尔的个人经历而不是政治信念本身 | cer | 0.1290 |
| 320 | streaming_asr | emilia_zh_0004212211 | 0.9511 | 11 | 这里表露的显而易见的的挫败感感觉源于卡尔的个人经历而不是政治信念本身 | cer | 0.1290 |
| 640 | streaming_asr | emilia_zh_0004212211 | 0.9480 | 12 | 这里表露的显而易见的的挫败感感觉源于卡尔的个人经历而不是政治信念本身 | cer | 0.1290 |
| 1280 | streaming_asr | emilia_zh_0004212211 | 0.9572 | 10 | 这里表露的显而易见的的挫败感感觉源于卡尔的个人经历而不是政治信念本身 | cer | 0.1290 |
| 160 | causal_full_asr | emilia_zh_0004422836 | 0.9847 | 7 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能坦白交代罪行切系在斯巴达中失守致命按当时编区的法令 | cer | 0.1765 |
| 320 | causal_full_asr | emilia_zh_0004422836 | 0.9847 | 7 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能坦白交代罪行切系在斯大中失守致命按当时编区的法令 | cer | 0.1569 |
| 640 | causal_full_asr | emilia_zh_0004422836 | 0.9771 | 13 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在斯大中失守致命按当时编区的法令 | cer | 0.2157 |
| 1280 | causal_full_asr | emilia_zh_0004422836 | 0.9801 | 11 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能坦白交代罪行且戏在斯大中失守致命按当时编区的法令 | cer | 0.1569 |
| 160 | streaming_asr | emilia_zh_0004519522 | 0.9627 | 5 | 这这个凶王的客人呢呃是昨天来 | cer | 0.2667 |
| 320 | streaming_asr | emilia_zh_0004519522 | 0.9503 | 7 | 就这个姓王的客人呢呃是昨天来 | cer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0004519522 | 0.9379 | 7 | 这这个姓王的客人呢呃是昨天来 | cer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0004519522 | 0.9379 | 7 | 这这个姓王的客人呢呃是昨天来 | cer | 0.2000 |
| 160 | causal_full_asr | emilia_zh_0004705832 | 0.8135 | 74 | These words are automatic no longer visualized which is why texts kids are longer to think because they're still visualizing the words is not automatic anymore which also explains why kids live like kids | wer | 0.1111 |
| 320 | causal_full_asr | emilia_zh_0004705832 | 0.7752 | 85 | These words are automatic we no longer visualize it which is why texts kids are longer to think because they're still visualizing the words is not automatic anymore which also explains why kids live like kids | wer | 0.0278 |
| 640 | causal_full_asr | emilia_zh_0004705832 | 0.7508 | 90 | These words are automatic we no longer visualize it which is why texts kids are longer to think because they're still visualizing the words is not automatic anymore which also explains why kids live on kids | wer | 0.0556 |
| 1280 | causal_full_asr | emilia_zh_0004705832 | 0.7401 | 95 | These words are automatic we no longer visualize it which is why texts kids are longer to think because they are still visualizing the words is not automatic anymore which also explains why kids live on kids | wer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004724727 | 0.8655 | 30 | Is it's frenching the out but a the so for Today then new things is a a with a all the a new | wer | 0.4348 |
| 320 | streaming_asr | emilia_zh_0004724727 | 0.8590 | 35 | Is at original the out but a the sir for Today the new things is a a speaker with a a the a new | wer | 0.4783 |
| 640 | streaming_asr | emilia_zh_0004724727 | 0.8503 | 36 | Is at original and out but I think sir for Today the new things is a a speaker with a a the a new | wer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0004724727 | 0.8503 | 37 | Is at really shall you know out but I think sir for Today the new things is a a speaker with a all the a new | wer | 0.3913 |
| 160 | streaming_asr | emilia_zh_0004804632 | 0.6149 | 37 | It is often a pretty site when Several of these notes are more together | wer | 0.0714 |
| 320 | streaming_asr | emilia_zh_0004804632 | 0.5805 | 38 | It is often of pretty site when Several of these notes are more together | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004804632 | 0.5805 | 38 | It is often a pretty site when Several of these notes are more together | wer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0004804632 | 0.5920 | 41 | It is often a pretty site when several of these notes are more together | wer | 0.0714 |
| 160 | streaming_asr | emilia_zh_0004880227 | 0.8411 | 24 | I should always remember the hours i spent with the master of the house of usher | wer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0004880227 | 0.8721 | 18 | I is always remember the hours i spent with the master of the house so busher | wer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0004880227 | 0.7519 | 35 | I should always remember the hours i spent with the master of the house of usher | wer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0004880227 | 0.7442 | 33 | I show always remember the hours i spent with the master of the house so usher | wer | 0.1875 |
| 160 | streaming_asr | emilia_zh_0005094293 | 0.9479 | 7 | 对脱过程学来说已经是一件很难做到的事情 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005094293 | 0.9621 | 5 | 对于脱工程学来说已经是一件很难做到事情 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0005094293 | 0.9621 | 4 | 都是脱工程学来说已经是一件很难做到事情的 | cer | 0.2381 |
| 1280 | streaming_asr | emilia_zh_0005094293 | 0.9621 | 4 | 都是脱工程学来说已经是一件很难做到事情的 | cer | 0.2381 |
| 160 | causal_full_asr | emilia_zh_0005313767 | 0.9817 | 4 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005313767 | 0.9670 | 7 | 您有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0005313767 | 0.9707 | 7 | 您有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0526 |
| 1280 | causal_full_asr | emilia_zh_0005313767 | 0.9744 | 6 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005420605 | 0.9694 | 5 | 有的刚卡是靠一这只大一遍我太行的 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0005420605 | 0.9592 | 7 | 有的钢卡是靠一这只大一遍不太行的 | cer | 0.2941 |
| 640 | streaming_asr | emilia_zh_0005420605 | 0.9490 | 9 | 有的光卡是靠一这只大一遍不太行的 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0005420605 | 0.9439 | 9 | 有的光卡只是靠一这只大一遍如果太行的 | cer | 0.3529 |
| 160 | streaming_asr | emilia_zh_0005670671 | 0.9703 | 9 | 时至今日切在人们社会的各主流中所败的角色意见无代替企业的生 | cer | 0.3030 |
| 320 | streaming_asr | emilia_zh_0005670671 | 0.9703 | 9 | 时至今日企业在人们社会的各组织中所办的角色意见无可代替企业的生 | cer | 0.1515 |
| 640 | streaming_asr | emilia_zh_0005670671 | 0.9678 | 9 | 时至今日企业在人们社会的的各组织中所办的角色意见无代替企业的生 | cer | 0.2121 |
| 1280 | streaming_asr | emilia_zh_0005670671 | 0.9653 | 10 | 时至今日企业在人们社会的的各组织中所办的角色意境巫的企业的生 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0005905391 | 0.9697 | 6 | 的自影响的感觉是什么呢就是麦当劳 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0005905391 | 0.9654 | 5 | 的自明显的感觉是什么呢就是麦当劳 | cer | 0.1250 |
| 640 | streaming_asr | emilia_zh_0005905391 | 0.9654 | 6 | 呃最明显的感觉是是什么呢就是麦当劳 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005905391 | 0.9654 | 6 | 呃最明显的感觉是什么呢就是麦当劳 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005960185 | 0.9339 | 10 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天珠葵 | cer | 0.2083 |
| 320 | causal_full_asr | emilia_zh_0005960185 | 0.9515 | 9 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天珠葵 | cer | 0.2083 |
| 640 | causal_full_asr | emilia_zh_0005960185 | 0.9471 | 10 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005960185 | 0.9559 | 8 | 啊我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.2083 |
| 160 | streaming_asr | emilia_zh_0006119250 | 0.9705 | 7 | 一个就是说那我本来啊肩膀说朋友捧马捧马然后还有点事儿 | cer | 0.5517 |
| 320 | streaming_asr | emilia_zh_0006119250 | 0.9705 | 7 | 一个就是如果呢我本来啊肩膀说我朋友捧马捧马然后嗨折叠事儿 | cer | 0.6207 |
| 640 | streaming_asr | emilia_zh_0006119250 | 0.9803 | 5 | 一个就是说呢我本来啊肩膀说我朋友他捧马捧马然后还左脚事儿 | cer | 0.5517 |
| 1280 | streaming_asr | emilia_zh_0006119250 | 0.9770 | 6 | 意思就是说呢我本来啊天堂说我朋友的朋友朋友然后还折叠事儿 | cer | 0.4483 |
| 160 | streaming_asr | emilia_zh_0006350396 | 0.8386 | 29 | My on found be position as a clock with spenlo and and gorgans and we found loggings neither | wer | 0.4706 |
| 320 | streaming_asr | emilia_zh_0006350396 | 0.8101 | 34 | My on found be a position as a clock with spenlo and and gorgans and we found loggings Nearby | wer | 0.4706 |
| 640 | streaming_asr | emilia_zh_0006350396 | 0.8165 | 33 | My on found be a position as a clock with spenlo and and gorgans and we found loggings Nearby | wer | 0.4706 |
| 1280 | streaming_asr | emilia_zh_0006350396 | 0.7975 | 35 | My aunt found be a position as a clock with spenlo and and gorgans and we found loggings Nearby | wer | 0.4118 |
| 160 | causal_full_asr | emilia_zh_0006404958 | 0.8571 | 25 | Cobras are like monkeys in winter second up weaver the wines why drinking water | wer | 0.3333 |
| 320 | causal_full_asr | emilia_zh_0006404958 | 0.8321 | 26 | Carrivers are like monkeys in winter sickened up withered wines while drinking water | wer | 0.3333 |
| 640 | causal_full_asr | emilia_zh_0006404958 | 0.8357 | 24 | The characters are like monkeys in winter second up reeled at wines while drinking water | wer | 0.2000 |
| 1280 | causal_full_asr | emilia_zh_0006404958 | 0.8429 | 24 | The characters are like monkeys in winter sickening up withered wines while drinking water | wer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0006435274 | 0.8230 | 18 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0006435274 | 0.8274 | 17 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0006435274 | 0.8186 | 18 | I bank charges interest I brother doesn't charge interest | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0006435274 | 0.7965 | 20 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006544707 | 0.9447 | 16 | 这了人呢也是在你说这个区间有冬季结婚的现象而且这两万人其中有一个 | cer | 0.1765 |
| 320 | streaming_asr | emilia_zh_0006544707 | 0.9237 | 24 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两两万人其中有一个 | cer | 0.1471 |
| 640 | streaming_asr | emilia_zh_0006544707 | 0.9263 | 22 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两方人其中有一个 | cer | 0.1176 |
| 1280 | streaming_asr | emilia_zh_0006544707 | 0.9342 | 20 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两方的人其中有一个 | cer | 0.1471 |
| 160 | streaming_asr | emilia_zh_0006874664 | 0.9662 | 14 | 那其中他有说的到他非常成捆的就是希望如果说是饮水不要如果是叉叉怎么样在对这一段描述里面有觉得比较真实 | cer | 0.2308 |
| 320 | streaming_asr | emilia_zh_0006874664 | 0.9557 | 17 | 那其中他有说到他非常成捆的就是希望如果说就是金水不要如果是查查怎么样在对这一段描述里面有觉得比较真实 | cer | 0.1923 |
| 640 | streaming_asr | emilia_zh_0006874664 | 0.9599 | 14 | 那其中他有说到他非常诚恳的就是希望如果说的饮水机没有如果是查查怎么样在对这一段描述里面有觉得比较真实 | cer | 0.1731 |
| 1280 | streaming_asr | emilia_zh_0006874664 | 0.9599 | 14 | 那其中他有说到他非常诚恳的就是希望如果说的薪水没有如果是查沙怎么样在对这一段描述里面有觉得比较真实 | cer | 0.1731 |
| 160 | streaming_asr | emilia_zh_0007060532 | 0.9856 | 2 | 如果你对保存人物件是五是真爱就比如嘴说 | cer | 0.3684 |
| 320 | streaming_asr | emilia_zh_0007060532 | 0.9760 | 4 | 如果你对报个人某件事武士战就比如嘴说 | cer | 0.3684 |
| 640 | streaming_asr | emilia_zh_0007060532 | 0.9760 | 4 | 如果你对表格人物件是五是真爱就不用嘴说 | cer | 0.3158 |
| 1280 | streaming_asr | emilia_zh_0007060532 | 0.9663 | 6 | 如果你对保格人某件事武士真爱就别用嘴说 | cer | 0.2105 |
| 160 | causal_full_asr | emilia_zh_0007060544 | 0.9625 | 5 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0007060544 | 0.9625 | 5 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0007060544 | 0.9688 | 4 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0007060544 | 0.9563 | 5 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007312674 | 0.9959 | 2 | 事实上亨利也不需要你亲爱的的他根本知道自己身在何地也认不出身边是谁和他带在一起 | cer | 0.0769 |
| 320 | streaming_asr | emilia_zh_0007312674 | 0.9939 | 2 | 事实上红利也不需要你亲爱的的他根本知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0769 |
| 640 | streaming_asr | emilia_zh_0007312674 | 0.9899 | 4 | 事实上通力也不需要你亲爱的的他根本不知道自己现在何地也认不出身边是谁和他呆在一起 | cer | 0.1026 |
| 1280 | streaming_asr | emilia_zh_0007312674 | 0.9878 | 4 | 事实上通力也不需要你亲爱的的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0007551053 | 0.9838 | 4 | 其他更多的工作是有律师事务所在又工作经验的在前台前辈们进行了 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0007551053 | 0.9741 | 6 | 其他更多的工作是有律师事务所在有工作经验的的前台前辈们进行了 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007551053 | 0.9773 | 5 | 其他更多的工作是有律师事务所的有工作经验的的前台前辈们进行了 | cer | 0.1071 |
| 1280 | streaming_asr | emilia_zh_0007551053 | 0.9709 | 7 | 其他更多工作是有律师事务所的有工作经验的的前台前辈们进行了 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007761299 | 0.9127 | 20 | 我觉得这个行业不适合我我觉得我跟这个在员工的图书出的开心 | cer | 0.2857 |
| 320 | streaming_asr | emilia_zh_0007761299 | 0.9164 | 18 | 我觉得这个行业不适合我我觉得我跟这在公司的同事出的不开心 | cer | 0.0714 |
| 640 | streaming_asr | emilia_zh_0007761299 | 0.9236 | 16 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007761299 | 0.9200 | 18 | 我觉得这行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0357 |
| 160 | streaming_asr | EN_B00043_S01954_W000033 | 0.7866 | 50 | Because love is more of the and just in emotion it's a capacity of verb and endlessly renewable research And not just and our private I | wer | 0.3750 |
| 320 | streaming_asr | EN_B00043_S01954_W000033 | 0.7802 | 51 | Because love is more of the and just in Motion it's a capacity of verb and endlessly renovable research And not just in our private like is | wer | 0.4583 |
| 640 | streaming_asr | EN_B00043_S01954_W000033 | 0.7823 | 52 | Because love is more of the and just in Motion it's a capacity of verb and endlessly renovable research and not just in our private I is | wer | 0.4583 |
| 1280 | streaming_asr | EN_B00043_S01954_W000033 | 0.7974 | 46 | Because love is more of the and just in Motion it's to capacity of verb and endlessly renovable research and not just in our private lied is | wer | 0.5000 |
| 160 | causal_full_asr | EN_B00043_S01957_W000002 | 0.8807 | 27 | Now we're not going to build video games or a computer generated actors which looks tremendously realistic | wer | 0.4286 |
| 320 | causal_full_asr | EN_B00043_S01957_W000002 | 0.8257 | 33 | Now we're not going about video games or a computer generated actors which looks tremendously realistic | wer | 0.2857 |
| 640 | causal_full_asr | EN_B00043_S01957_W000002 | 0.7890 | 38 | Now we're not thinking about video games or a computer generated actors which looks dramatically realistic | wer | 0.3571 |
| 1280 | causal_full_asr | EN_B00043_S01957_W000002 | 0.7982 | 37 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 160 | streaming_asr | EN_B00089_S03348_W000001 | 0.7771 | 49 | one the things that like to do when each my interdoctoral stronomy Classes is to be in the class with the stronomy picture of that day | wer | 0.2759 |
| 320 | streaming_asr | EN_B00089_S03348_W000001 | 0.7390 | 57 | one the things out like to do when each my interductory stronomy Classes is to be in the class with the stronomy picture of the day | wer | 0.2759 |
| 640 | streaming_asr | EN_B00089_S03348_W000001 | 0.6862 | 62 | One of the things that like to do when each my interductory stronomy Classes is to be in the class with the stronomy picture of the day | wer | 0.2069 |
| 1280 | streaming_asr | EN_B00089_S03348_W000001 | 0.6804 | 68 | One of the things I like to do when each my interductory stronomy Classes is to be and class with the stronomy picture of the day | wer | 0.2759 |
| 160 | streaming_asr | EN_B00048_S07862_W000265 | 0.8511 | 25 | The only means of crossing large Areas of water was in a selling ship Driven by the wind | wer | 0.0556 |
| 320 | streaming_asr | EN_B00048_S07862_W000265 | 0.8165 | 31 | The only means of crossing large Areas of water was in a sailing ship driven by the wind | wer | 0.0000 |
| 640 | streaming_asr | EN_B00048_S07862_W000265 | 0.8298 | 29 | The only means of crossing large Areas of water was in a selling ship driven by the wind | wer | 0.0556 |
| 1280 | streaming_asr | EN_B00048_S07862_W000265 | 0.8564 | 28 | The only means of crossing large Areas of water was in a selling ship driven by the wind | wer | 0.0556 |
| 160 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7778 | 21 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7346 | 23 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7593 | 20 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7716 | 19 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S04429_W000016 | 0.9307 | 15 | The you see these it'll green lives And this sping flower where I this second know | wer | 0.5000 |
| 320 | streaming_asr | EN_B00058_S04429_W000016 | 0.9158 | 19 | Do you see these little green luses And this being flower where I a second know | wer | 0.3125 |
| 640 | streaming_asr | EN_B00058_S04429_W000016 | 0.9084 | 20 | The you see these little green ludes and this being flower where I a second know | wer | 0.3750 |
| 1280 | streaming_asr | EN_B00058_S04429_W000016 | 0.9134 | 18 | The you see these little green lusers and this being flower where I a second go | wer | 0.3750 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
