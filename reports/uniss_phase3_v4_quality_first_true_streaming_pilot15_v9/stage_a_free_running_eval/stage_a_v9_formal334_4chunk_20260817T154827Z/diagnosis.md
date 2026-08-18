# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 1336
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8849**
- Weighted CTC blank ratio: **0.1847**
- Weighted streaming WER/CER: **0.4384**
- Weighted causal-full WER/CER: **0.2486**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | causal_full_asr | CommonVoice_EN_0000069954 | 0.1804 | 19 | P dividends also suggest that the plant was present decades before its first collection | wer | 0.2308 |
| 320 | causal_full_asr | CommonVoice_EN_0000069954 | 0.1867 | 17 | The defence also suggests that the plant was present decades before its first collection | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000069954 | 0.2089 | 17 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.2184 | 23 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1015 | 14 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1015 | 13 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000116798 | 0.0985 | 15 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1108 | 14 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1136 | 13 | Then he's an old man. He's going to spend a month in Africa | wer | 0.1538 |
| 320 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1364 | 13 | Then he's an old man, he's going to spend a month in Africa | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000126435 | 0.1773 | 16 | Then he's an old man. He's going to spend a month in Africa | wer | 0.1538 |
| 1280 | causal_full_asr | CommonVoice_EN_0000126435 | 0.2091 | 16 | When he's an old man he's going to spend a month in Africa | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1561 | 12 | 强大的引擎或支援 | wer | 1.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000160093 | 0.2098 | 14 | A presence of my tangerine sir said Paul | wer | 0.7778 |
| 640 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1902 | 13 | A presence of my tangents are said Palt | wer | 0.8889 |
| 1280 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1610 | 15 | A presence of my tangents are said part | wer | 0.8889 |
| 160 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1600 | 17 | It is across the Columbia River from whence Yamen, Washington | wer | 0.2000 |
| 320 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1964 | 18 | It is across the Columbia River from whence Yamman, Washington | wer | 0.2000 |
| 640 | causal_full_asr | CommonVoice_EN_0000262462 | 0.2145 | 17 | It is across the Columbia River from West Hamen, Washington | wer | 0.2000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000262462 | 0.1673 | 15 | It is across the Columbia River from Wiesam in Washington | wer | 0.2000 |
| 160 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2320 | 18 | The soul of the world is nourished by people's happiness. | wer | 0.1000 |
| 320 | causal_full_asr | CommonVoice_EN_0000312479 | 0.3315 | 18 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2762 | 19 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000312479 | 0.2431 | 17 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1339 | 21 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 320 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1726 | 20 | Teams from Topeka, Kansas and Wichita, Kansas, joined from the Western Association | wer | 0.2500 |
| 640 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1964 | 22 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 1280 | causal_full_asr | CommonVoice_EN_0000352714 | 0.1726 | 25 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 160 | causal_full_asr | CommonVoice_EN_0000430515 | 0.0982 | 11 | Later years he occasionally put determined lower order batsmen | wer | 0.3636 |
| 320 | causal_full_asr | CommonVoice_EN_0000430515 | 0.1368 | 13 | In later years he occasionally put a determined lower order batsman | wer | 0.0909 |
| 640 | causal_full_asr | CommonVoice_EN_0000430515 | 0.1193 | 11 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000430515 | 0.0982 | 12 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000471648 | 0.1822 | 13 | Wood is best for making toys and blacks | wer | 0.1250 |
| 320 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2356 | 13 | Wood is best for making toys and blacks | wer | 0.1250 |
| 640 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2578 | 14 | Wood is best for making toys and blacks | wer | 0.1250 |
| 1280 | causal_full_asr | CommonVoice_EN_0000471648 | 0.2933 | 15 | Wood is best for making toys and blacks | wer | 0.1250 |
| 160 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1102 | 6 | Nothing personal in it | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1575 | 8 | Nothing personal on it | wer | 0.2500 |
| 640 | causal_full_asr | CommonVoice_EN_0000502585 | 0.2205 | 8 | Nothing personal in it | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1654 | 10 | Nothing personal on it | wer | 0.2500 |
| 160 | causal_full_asr | CommonVoice_EN_0000593818 | 0.1286 | 15 | If it wasn't difficult, get one be a problem | wer | 0.3333 |
| 320 | causal_full_asr | CommonVoice_EN_0000593818 | 0.1429 | 17 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 640 | causal_full_asr | CommonVoice_EN_0000593818 | 0.2429 | 24 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 1280 | causal_full_asr | CommonVoice_EN_0000593818 | 0.2857 | 21 | If it wasn't difficult, it wouldn't be a problem | wer | 0.1111 |
| 160 | causal_full_asr | DailyTalk_0000009768 | 0.1667 | 11 | Bring us a bottle of Remi Mortini and red wine | wer | 0.2000 |
| 320 | causal_full_asr | DailyTalk_0000009768 | 0.1882 | 12 | Bring us a bottle of Remi Mortini and red wine | wer | 0.2000 |
| 640 | causal_full_asr | DailyTalk_0000009768 | 0.1828 | 13 | Bring us the bottle of Remy Martinez and Red wine | wer | 0.2000 |
| 1280 | causal_full_asr | DailyTalk_0000009768 | 0.1828 | 12 | Bring us to bottle of Remy Martinez and Red wine | wer | 0.2000 |
| 160 | causal_full_asr | DailyTalk_0000010084 | 0.2031 | 8 | Good morning sir, what can I do for you? | wer | 0.3333 |
| 320 | causal_full_asr | DailyTalk_0000010084 | 0.2109 | 9 | The morning sir, what can I do for you? | wer | 0.4444 |
| 640 | causal_full_asr | DailyTalk_0000010084 | 0.1797 | 6 | The morning sir, what can I do for you? | wer | 0.4444 |
| 1280 | causal_full_asr | DailyTalk_0000010084 | 0.1719 | 8 | The morning sir, what can I do for you? | wer | 0.4444 |
| 160 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2248 | 15 | Ride your skateboard, said his dad, it's two slipboards at Adam. | wer | 0.5455 |
| 320 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2431 | 14 | Riding a skateboard set is dead. It's two slipboards at Adam. | wer | 0.8182 |
| 640 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2294 | 15 | Ride your skateboard, said his dad. It's two slippery slides at that. | wer | 0.5455 |
| 1280 | causal_full_asr | EN_B00013_S06748_W000028 | 0.2064 | 15 | Ride your skateboard, said his dad, it's two slippers and Adam | wer | 0.4545 |
| 160 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1784 | 17 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1878 | 15 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1315 | 14 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1033 | 13 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2500 | 14 | 因为他们正在迅速消耗这些资源 | wer | 1.0000 |
| 320 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2849 | 15 | 因为他们正在迅速消耗这些资源 | wer | 1.0000 |
| 640 | causal_full_asr | EN_B00043_S01557_W000006 | 0.2326 | 13 | 他们正在迅速消耗这些资源 | wer | 1.0000 |
| 1280 | causal_full_asr | EN_B00043_S01557_W000006 | 0.1919 | 15 | 他们正在迅速消耗这些资源 | wer | 1.0000 |
| 160 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1408 | 26 | I found all that hard to believe too and must have been terrible and there was nothing anyone could do about it. | wer | 0.0909 |
| 320 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1408 | 24 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it. | wer | 0.0455 |
| 640 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1529 | 28 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it. | wer | 0.0455 |
| 1280 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1456 | 26 | I found all that hard to believe to It must have been terrible and there was nothing anyone could do about it. | wer | 0.0909 |
| 160 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2049 | 28 | Now we're now looking about VGG games or competent generated actors which looks tremendously realistic. | wer | 0.4286 |
| 320 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2294 | 28 | Now we're now thinking about video games or competing generated actors which looks tremendously realistic. | wer | 0.3571 |
| 640 | causal_full_asr | EN_B00043_S01957_W000002 | 0.2049 | 27 | Now we're now talking about video games or computer generated actors which looks tremendously realistic. | wer | 0.2857 |
| 1280 | causal_full_asr | EN_B00043_S01957_W000002 | 0.1957 | 27 | Now we're now talking about video games or computer generated actors which looks tremendously realistic. | wer | 0.2857 |
| 160 | causal_full_asr | EN_B00043_S02699_W000000 | 0.1589 | 30 | Set the mesh. Just goes through the main door turn left walked down to the end of the corridor and it's the last door on the right | wer | 0.1923 |
| 320 | causal_full_asr | EN_B00043_S02699_W000000 | 0.1716 | 27 | This is a mesh just goes through the main door turn F to walk down to the end of the corridor and it's the last door on the right | wer | 0.2692 |
| 640 | causal_full_asr | EN_B00043_S02699_W000000 | 0.2309 | 26 | That may show just goes through the main door turn F to walk down to the end of the corridor and it's the last door on the right | wer | 0.2308 |
| 1280 | causal_full_asr | EN_B00043_S02699_W000000 | 0.2097 | 26 | Suddenly missed. That's good through the main door turned F to walk down to the end of the corridor and it's the last door on the right. | wer | 0.3077 |
| 160 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1593 | 12 | 温达你从饼干罐里偷了饼干 | wer | 1.0000 |
| 320 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1504 | 9 | 温达，你从饼干罐里偷了饼干 | wer | 1.0000 |
| 640 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1726 | 11 | Linda, you stole the cookies from the cookie jar. | wer | 0.2222 |
| 1280 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1283 | 7 | Linda, you stole the cookies from the cookie jar. | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00048_S03599_W000216 | 0.2086 | 22 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 320 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1771 | 20 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 640 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1600 | 18 | So for those different categories of use you'll find that the model of a verb is used slightly differently. | wer | 0.3529 |
| 1280 | causal_full_asr | EN_B00048_S03599_W000216 | 0.1571 | 17 | So for those different categories of use you'll find that the model verbs are used slightly differently. | wer | 0.1176 |
| 160 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1235 | 8 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1412 | 10 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1059 | 9 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07041_W000493 | 0.0765 | 8 | 也许他们就是不喜欢我，所以他们不想和我讲话 | wer | 1.0000 |
| 160 | causal_full_asr | EN_B00048_S07870_W000001 | 0.2284 | 16 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 320 | causal_full_asr | EN_B00048_S07870_W000001 | 0.2099 | 13 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 640 | causal_full_asr | EN_B00048_S07870_W000001 | 0.1481 | 11 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 1280 | causal_full_asr | EN_B00048_S07870_W000001 | 0.1111 | 9 | I've never heard of him. I doubt he's very powerful. | wer | 0.2000 |
| 160 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1441 | 13 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1525 | 14 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1525 | 13 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1144 | 10 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1401 | 28 | Every two months my coworkers and I would come together to discuss the news of my schedule. Our meetings were usually held in the staff room at our institute | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1419 | 36 | Every two months my coworkers and I would come together to discuss the news semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0714 |
| 640 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1505 | 35 | Every two months my coworkers and I would come together to discuss the new semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 1280 | causal_full_asr | EN_B00048_S09601_W000039 | 0.1401 | 32 | Every two months my coworkers and I would come together to discuss the new semester schedule. Our meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 160 | causal_full_asr | EN_B00052_S08813_W000015 | 0.2038 | 42 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans a cooperate and compromise | wer | 0.0385 |
| 320 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1942 | 40 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1808 | 42 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1635 | 40 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1886 | 13 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S01107_W000030 | 0.2057 | 16 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1314 | 13 | The entire project was dropped in my lap after Jason resigned. | wer | 0.0909 |
| 1280 | causal_full_asr | EN_B00058_S01107_W000030 | 0.1371 | 13 | The entire project was dropped in my lap after Jason resigned. | wer | 0.0909 |
| 160 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1633 | 25 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. He said that's a place in September. | wer | 0.2692 |
| 320 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1658 | 23 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that took place in September | wer | 0.1154 |
| 640 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1633 | 27 | So you originally meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that to place in September | wer | 0.1538 |
| 1280 | causal_full_asr | EN_B00058_S06163_W000045 | 0.1429 | 27 | So you original meeting that we did a bunch of episodes about of seeing people in the inner earth. You said that your place in September | wer | 0.1154 |
| 160 | causal_full_asr | EN_B00058_S06426_W000037 | 0.2289 | 21 | It's like this curve here where F here is on the horizontal axis | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00058_S06426_W000037 | 0.2218 | 24 | And let's say this curve here where f here is on the horizontal axis | wer | 0.2143 |
| 640 | causal_full_asr | EN_B00058_S06426_W000037 | 0.1796 | 22 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S06426_W000037 | 0.1479 | 17 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1227 | 25 | In Tibetan Buddhism, models are mainly used to count mantras. These mantras can be recited for different purposes linked to working with mind. | wer | 0.1739 |
| 320 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1227 | 21 | Intubent Buddhism models are mainly used to count monsters. These monsters can be recited for different purposes linked to working with mind. | wer | 0.2609 |
| 640 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1044 | 23 | Intubent Buddhism models are mainly used to count montras These montras can be recited for different purposes linked to working with mind | wer | 0.2174 |
| 1280 | causal_full_asr | EN_B00064_S09941_W000015 | 0.1097 | 25 | Interpenent Buddhism models are mainly used to count montras. These montras can be recited for different purposes linked to working with mind. | wer | 0.2609 |
| 160 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1408 | 34 | In addition the journal covers signaling networks, synthetic biology systems biology, trying to discovery and computation and modeling of regulatory pathways. | wer | 0.2500 |
| 320 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1354 | 34 | In addition the journal covers signaling networks synthetic biology systems biology joining discovery and computation and modeling of regulatory pathways | wer | 0.0500 |
| 640 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1300 | 33 | In addition the journal covers signaling networks synthetic biology systems biology, joint discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 1280 | causal_full_asr | EN_B00083_S00712_W000002 | 0.1137 | 29 | In addition the journal covers signaling networks, synthetic biology systems biology, joint discovery and computation and modeling of regulatory pathways | wer | 0.1500 |
| 160 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1465 | 58 | We don't learn from our successes and we don't learn from our failures in a way that allows the impact to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success. I've tried to do that throughout my relationship with them. | wer | 0.0877 |
| 320 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1424 | 60 | We don't learn from our successes and we don't learn from our failures in a way that allows the impressed to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1228 |
| 640 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1445 | 61 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1053 |
| 1280 | causal_full_asr | EN_B00083_S03017_W000001 | 0.1445 | 62 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1053 |
| 160 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1828 | 22 | These are not variables of algebra. A variable is any characteristic | wer | 0.2308 |
| 320 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1690 | 20 | These are not the variables of algebra. A variable is any characteristic | wer | 0.1538 |
| 640 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1345 | 20 | These are not the variables of algebra. A variable is any characteristic | wer | 0.1538 |
| 1280 | causal_full_asr | EN_B00083_S08368_W000007 | 0.1310 | 21 | And these are not the variables of algebra. A variable is any characteristic | wer | 0.0769 |
| 160 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1275 | 10 | You're unmute to a gun at cheery. Sorry good morning everybody | wer | 0.6667 |
| 320 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1235 | 11 | You're unmute we're gonna hear you. Sorry good morning everybody | wer | 0.5833 |
| 640 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1155 | 10 | You're unmute we're gonna hear you. Sorry good morning everybody | wer | 0.5833 |
| 1280 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1195 | 12 | You're unmute we're gonna carry you. Sorry good morning everybody. | wer | 0.7500 |
| 160 | causal_full_asr | LibriSpeech_0000011154 | 0.1442 | 35 | Other villagers said it was a fine idea, so they stopped working for once and began to plan celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.1471 |
| 320 | causal_full_asr | LibriSpeech_0000011154 | 0.1473 | 41 | Out of the villagers said it was a fine idea. So they stopped working for once and began to plan celebration. They thought that there ought to be swimming races and treefilling contests | wer | 0.2059 |
| 640 | causal_full_asr | LibriSpeech_0000011154 | 0.1473 | 41 | All the villagers said it was a fine idea, so they stopped working for once and began to plan a celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.0882 |
| 1280 | causal_full_asr | LibriSpeech_0000011154 | 0.1240 | 38 | All the villagers said it was a fine idea, so they stopped working for once and began to plan a celebration. They thought that there ought to be swimming races and tree felling contests | wer | 0.0882 |
| 160 | causal_full_asr | LibriSpeech_0000011649 | 0.1460 | 43 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000011649 | 0.1711 | 48 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000011649 | 0.1770 | 52 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000011649 | 0.1593 | 51 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 160 | causal_full_asr | LibriSpeech_0000192309 | 0.1924 | 23 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 320 | causal_full_asr | LibriSpeech_0000192309 | 0.1703 | 22 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 640 | causal_full_asr | LibriSpeech_0000192309 | 0.1451 | 20 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 1280 | causal_full_asr | LibriSpeech_0000192309 | 0.1104 | 17 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 160 | causal_full_asr | LibriSpeech_0000214472 | 0.0882 | 25 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared. He spoke eagerly and anxiously. | wer | 0.1200 |
| 320 | causal_full_asr | LibriSpeech_0000214472 | 0.1059 | 25 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared. He spoke eagerly and anxiously | wer | 0.0800 |
| 640 | causal_full_asr | LibriSpeech_0000214472 | 0.0985 | 28 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far suddenly disappeared he spoke eagerly and anxiously | wer | 0.0400 |
| 1280 | causal_full_asr | LibriSpeech_0000214472 | 0.0956 | 27 | By common consent the subject is never mentioned between us. The bitter irony of his tone thus far certainly disappeared. He spoke eagerly and anxiously | wer | 0.1200 |
| 160 | causal_full_asr | LibriSpeech_0000238204 | 0.1461 | 38 | What an entrepreneur those have a test why men who have a task who why it can only be speedy adventures a sort of person one reads of in books and it in needs a meal out | wer | 0.6875 |
| 320 | causal_full_asr | LibriSpeech_0000238204 | 0.1332 | 37 | Would answer one of those ever-taunts why men who ever-taunts who whyds can only be speedy adventurers, the sort person one reads of in books and knitting means a meal. | wer | 0.3750 |
| 640 | causal_full_asr | LibriSpeech_0000238204 | 0.1605 | 45 | What answer one of those evertized why men who evertized who was can only be speedy adventures, the sort person one reads of in books and it in needs and meal on | wer | 0.4375 |
| 1280 | causal_full_asr | LibriSpeech_0000238204 | 0.1676 | 43 | Would answer one of those I've a tussent why men who have a tussent who why it can only be speedy adventures the sort person one reads of in books and that it means a million | wer | 0.5312 |
| 160 | causal_full_asr | LibriSpeech_0000263750 | 0.1424 | 32 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk ready | wer | 0.2000 |
| 320 | causal_full_asr | LibriSpeech_0000263750 | 0.1524 | 36 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1500 |
| 640 | causal_full_asr | LibriSpeech_0000263750 | 0.1407 | 37 | A magnificent person with powdered hair, breeches and silk stockings presented himself Lord Reginald Sedley he announced in Walk Reggie | wer | 0.1500 |
| 1280 | causal_full_asr | LibriSpeech_0000263750 | 0.1407 | 35 | A magnificent person with powdered hair, breeches and silk stockings presented himself, Lord Reginald Sedley he announced in Walk Reggie | wer | 0.2000 |
| 160 | causal_full_asr | LibriSpeech_0000271146 | 0.2021 | 36 | But it's full of ant living mounds and enormous springs with the Indian such a beet some gigantic race for chilft in a past age | wer | 0.6522 |
| 320 | causal_full_asr | LibriSpeech_0000271146 | 0.2280 | 38 | But it's full of antiquity in the mounds and enormous bones with the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.6087 |
| 640 | causal_full_asr | LibriSpeech_0000271146 | 0.2228 | 39 | But it's full of untilving the mounds in enormous bones with the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.5652 |
| 1280 | causal_full_asr | LibriSpeech_0000271146 | 0.2202 | 35 | For it is full of until leaving the mounds enormous bones for the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.4348 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.1963 | 8 | That's true. What's about the mentality and that's | wer | 0.7143 |
| 320 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2699 | 10 | Just who's about the natty and that's | wer | 0.8571 |
| 640 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2577 | 11 | That's true. What about the mentality in bats | wer | 0.5714 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2577 | 13 | That's true. What about the mentality in bats | wer | 0.5714 |
| 160 | causal_full_asr | VCTK_0000006143 | 0.0886 | 9 | Being captain of this club's fantastic | wer | 0.2857 |
| 320 | causal_full_asr | VCTK_0000006143 | 0.1266 | 12 | Being captain of this club's fantastic | wer | 0.2857 |
| 640 | causal_full_asr | VCTK_0000006143 | 0.1266 | 12 | Being kept in of this club's fantastic | wer | 0.5714 |
| 1280 | causal_full_asr | VCTK_0000006143 | 0.1329 | 12 | Being kept in of this club's fantastic | wer | 0.5714 |
| 160 | causal_full_asr | emilia_zh_0003940788 | 0.1700 | 26 | 呃然后呢这个因为它更大能帮助注意力同时性行为被盗凶猛就是特有那种王者之气 | cer | 0.2973 |
| 320 | causal_full_asr | emilia_zh_0003940788 | 0.1854 | 29 | 呃然后呢这个因为它更大呃能帮助注意力同时性情也没到胸嘛就是特有那种王者之气 | cer | 0.2973 |
| 640 | causal_full_asr | emilia_zh_0003940788 | 0.1921 | 30 | 呃然后呢这个因为它杠杆大呃能帮助注意力同时性情也被调凶嘛就是特有那种王者之气 | cer | 0.2703 |
| 1280 | causal_full_asr | emilia_zh_0003940788 | 0.2009 | 27 | 呃然后呢这个因为它够大呃能帮助注意力同时性情也被调凶猛就是特有那种王者之气 | cer | 0.2432 |
| 160 | causal_full_asr | emilia_zh_0003942539 | 0.2257 | 40 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0175 |
| 320 | causal_full_asr | emilia_zh_0003942539 | 0.2178 | 38 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很慌恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0003942539 | 0.2119 | 39 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0351 |
| 1280 | causal_full_asr | emilia_zh_0003942539 | 0.1960 | 39 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他们问我们的时候啊，我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0351 |
| 160 | causal_full_asr | emilia_zh_0004036374 | 0.2377 | 33 | 所以有关共产主义运动的具体策略和结论的迅速来充当马克思主义哲学发展的内在逻辑迅速 | cer | 0.2250 |
| 320 | causal_full_asr | emilia_zh_0004036374 | 0.2523 | 31 | 所以有关共产主义运动的具体策略和结论的叙述来充当那马克思主义哲学发散的内在逻辑叙述 | cer | 0.1500 |
| 640 | causal_full_asr | emilia_zh_0004036374 | 0.2541 | 36 | 则有关共产主义运动的具体策略和结论的叙述来充当那马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0004036374 | 0.2559 | 37 | 则有关共产主义运动的具体策略和节目的叙述来充当那马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 160 | causal_full_asr | emilia_zh_0004213341 | 0.1374 | 17 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004213341 | 0.1450 | 16 | 世界上每天都有奇迹在发生这些奇迹大都来源于精神的力量 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004213341 | 0.1450 | 18 | 世界上每天都有积极的发生这些奇迹大都来源于精神的力量 | cer | 0.1154 |
| 1280 | causal_full_asr | emilia_zh_0004213341 | 0.1298 | 16 | 世界上每天都有奇迹的发生这些奇迹大都来源于精神的力量 | cer | 0.0385 |
| 160 | causal_full_asr | emilia_zh_0004343392 | 0.1476 | 13 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 320 | causal_full_asr | emilia_zh_0004343392 | 0.1550 | 14 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 640 | causal_full_asr | emilia_zh_0004343392 | 0.1587 | 16 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 1280 | causal_full_asr | emilia_zh_0004343392 | 0.1882 | 17 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 160 | causal_full_asr | emilia_zh_0004422836 | 0.1544 | 37 | 精神讯犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行接戏在四大中失守之命案当时编曲的法令 | cer | 0.3137 |
| 320 | causal_full_asr | emilia_zh_0004422836 | 0.1728 | 39 | 精神训犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行接戏在四大中失守之命案当时编剧的法令 | cer | 0.3333 |
| 640 | causal_full_asr | emilia_zh_0004422836 | 0.1483 | 39 | 精神训犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行且戏在私打中失守致命案当时编剧的法令 | cer | 0.2745 |
| 1280 | causal_full_asr | emilia_zh_0004422836 | 0.1606 | 40 | 精神讯犯罪嫌疑人对案情供认不会因私人军不满十八周岁尚能砍白胶带罪行且戏在私打中失守致命案当时编剧的法令 | cer | 0.2549 |
| 160 | causal_full_asr | emilia_zh_0004472296 | 0.1646 | 15 | 里面的不对啊这是造白书带有给你们念上念 | cer | 0.2632 |
| 320 | causal_full_asr | emilia_zh_0004472296 | 0.1677 | 15 | 明年的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 640 | causal_full_asr | emilia_zh_0004472296 | 0.1801 | 17 | 宁愿的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 1280 | causal_full_asr | emilia_zh_0004472296 | 0.1770 | 14 | 宁愿的不对啊这是照白书带我给你们念上念 | cer | 0.2105 |
| 160 | causal_full_asr | emilia_zh_0004621493 | 0.1905 | 14 | Jesus and the tribe to be on man who steals a little bit here and there | wer | 0.6429 |
| 320 | causal_full_asr | emilia_zh_0004621493 | 0.2143 | 14 | Just an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 640 | causal_full_asr | emilia_zh_0004621493 | 0.1524 | 15 | He's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 1280 | causal_full_asr | emilia_zh_0004621493 | 0.1333 | 12 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 160 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 18 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away | wer | 0.3500 |
| 320 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 22 | The parlor had once been two rooms and the floor was swayed back to where the partition had been cut away. | wer | 0.4000 |
| 640 | causal_full_asr | emilia_zh_0004633749 | 0.2369 | 24 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away. | wer | 0.4000 |
| 1280 | causal_full_asr | emilia_zh_0004633749 | 0.1928 | 20 | The pilot had once been two rooms and the floor was swayed back to where their partition had been cut away. | wer | 0.4000 |
| 160 | causal_full_asr | emilia_zh_0004659501 | 0.1404 | 15 | And for all you just write nine or ten and then where do, where to | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004659501 | 0.1633 | 18 | And for old you just write nine or ten and then where do, where to | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0004659501 | 0.1576 | 19 | And for all you just write nine or ten and then where do, where to | wer | 0.2857 |
| 1280 | causal_full_asr | emilia_zh_0004659501 | 0.1547 | 17 | And for all you just write nine or ten and then where to where to | wer | 0.2857 |
| 160 | causal_full_asr | emilia_zh_0004665404 | 0.2057 | 24 | If the wine be sweet I would drink it with him and if it be bitter I would drink it with him also was my answer | wer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004665404 | 0.2143 | 27 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004665404 | 0.1886 | 26 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004665404 | 0.1543 | 24 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004665564 | 0.1591 | 35 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled an iron hand in a velvet political glove | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004665564 | 0.1316 | 34 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled an iron hand in a velvet political glove. | wer | 0.2857 |
| 640 | causal_full_asr | emilia_zh_0004665564 | 0.1316 | 34 | But he ran extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled with an iron hand in a velvet political glove | wer | 0.2571 |
| 1280 | causal_full_asr | emilia_zh_0004665564 | 0.1297 | 37 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his excessive features. He rolled with an iron hand in a velvet political glove | wer | 0.2286 |
| 160 | causal_full_asr | emilia_zh_0004692595 | 0.1528 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004692595 | 0.1661 | 19 | Different equation and then later the differential equation interpreted in black diagram terms. | wer | 0.0769 |
| 640 | causal_full_asr | emilia_zh_0004692595 | 0.1362 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004692595 | 0.1362 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004705832 | 0.1820 | 45 | These words are automatically no longer visualized which is why it takes kids longer to think because they're still visualizing the words it's not a mechanic anymore which also explains why kids live on kids | wer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0004705832 | 0.1758 | 46 | These words are automatic we no longer visualize it, which is why it takes kids a longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1667 |
| 640 | causal_full_asr | emilia_zh_0004705832 | 0.1774 | 45 | These words are automatic we no longer visualize it, which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0004705832 | 0.1575 | 47 | These words are automatic and no longer visualize it, which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore, which also explains why kids live on kids | wer | 0.1944 |
| 160 | causal_full_asr | emilia_zh_0004732654 | 0.2531 | 19 | The quiet suddenly the fire had given a pale flicker and went down and the clocking ceased | wer | 0.2941 |
| 320 | causal_full_asr | emilia_zh_0004732654 | 0.2448 | 19 | But quite suddenly the fire hit gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004732654 | 0.2199 | 20 | But quite suddenly the fire had given a pale flicker and went down and the clocking ceased. | wer | 0.2353 |
| 1280 | causal_full_asr | emilia_zh_0004732654 | 0.1618 | 18 | But quite suddenly the fire ahead gave a pale flicker and went down and the clocking ceased. | wer | 0.1176 |
| 160 | causal_full_asr | emilia_zh_0004841554 | 0.2023 | 16 | I seek to use me, but can you tell us whether purpose just come down from London? | wer | 0.4118 |
| 320 | causal_full_asr | emilia_zh_0004841554 | 0.2062 | 15 | I say excuse me, but can you tell us where the purpose just come down from London? | wer | 0.2941 |
| 640 | causal_full_asr | emilia_zh_0004841554 | 0.2140 | 22 | I say excuse me, but can you tell us whether purpose just come down from London? | wer | 0.2353 |
| 1280 | causal_full_asr | emilia_zh_0004841554 | 0.1751 | 18 | I say excuse me, but can you tell us whether purpose just come down from London? | wer | 0.2353 |
| 160 | causal_full_asr | emilia_zh_0005058847 | 0.1614 | 28 | 因为死心难起什么就是慈悲的慈悲纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 320 | causal_full_asr | emilia_zh_0005058847 | 0.1651 | 26 | 因为死心难起什么就是慈悲的慈悲纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 640 | causal_full_asr | emilia_zh_0005058847 | 0.1820 | 27 | 因为死心难起怎么就是慈悲的死纵然发起难得酒停为什么呢人到自私 | cer | 0.2000 |
| 1280 | causal_full_asr | emilia_zh_0005058847 | 0.1857 | 28 | 因为雌性难企，他妈就是雌配的雌纵然发起男的酒停为什么呢人到自私 | cer | 0.4333 |
| 160 | causal_full_asr | emilia_zh_0005070101 | 0.1468 | 32 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005070101 | 0.1315 | 35 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005070101 | 0.1437 | 35 | 就可能使法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005070101 | 0.1391 | 37 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 160 | causal_full_asr | emilia_zh_0005070340 | 0.2054 | 15 | 曾是了一项名副其实的哲学工作，因为他继续着前人的努力 | cer | 0.1200 |
| 320 | causal_full_asr | emilia_zh_0005070340 | 0.2145 | 18 | 从事了一项名酷其实的哲学工作，因为他继续着前人的努力 | cer | 0.0800 |
| 640 | causal_full_asr | emilia_zh_0005070340 | 0.2326 | 17 | 从事了一下名酷其实的哲学工作因为他继续着前人的努力 | cer | 0.0800 |
| 1280 | causal_full_asr | emilia_zh_0005070340 | 0.2326 | 18 | 从事了一下名酷其实的哲学工作因为他继续着前人的努力 | cer | 0.0800 |
| 160 | causal_full_asr | emilia_zh_0005184596 | 0.1212 | 10 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 320 | causal_full_asr | emilia_zh_0005184596 | 0.1515 | 11 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 640 | causal_full_asr | emilia_zh_0005184596 | 0.1566 | 11 | That can't be a criticism of your two hands. | cer | 1.8947 |
| 1280 | causal_full_asr | emilia_zh_0005184596 | 0.1515 | 13 | That can't be a criticism of your hands. | cer | 1.7368 |
| 160 | causal_full_asr | emilia_zh_0005245611 | 0.2576 | 20 | 他提出预祝的商品只有再次革新才能更好的满足欧洲人的需求 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005245611 | 0.2508 | 23 | 他提出预注的商品只有再次革新才能更好的满足住人的需求 | cer | 0.1111 |
| 640 | causal_full_asr | emilia_zh_0005245611 | 0.2644 | 24 | 他提出预注的商品只有再次革新才能更好的满足住人的需求 | cer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0005245611 | 0.2576 | 27 | 他提出预注的商品只有再次革新才能更好的满足欧洲人的需求 | cer | 0.0370 |
| 160 | causal_full_asr | emilia_zh_0005313767 | 0.2308 | 22 | 你有五秒钟的时间阅读第一小题带有相关内容 | cer | 0.1053 |
| 320 | causal_full_asr | emilia_zh_0005313767 | 0.2344 | 24 | You need to read the first question in five seconds. | cer | 2.2632 |
| 640 | causal_full_asr | emilia_zh_0005313767 | 0.2198 | 22 | You have five seconds to read the relevant content. | cer | 2.2632 |
| 1280 | causal_full_asr | emilia_zh_0005313767 | 0.2125 | 23 | You have five seconds to read the relevant content. | cer | 2.2632 |
| 160 | causal_full_asr | emilia_zh_0005347375 | 0.2080 | 17 | 但我们却不知道他是什么只能确定他存在一定小野 | cer | 0.2727 |
| 320 | causal_full_asr | emilia_zh_0005347375 | 0.1726 | 16 | 但我们却不知道他是什么只能确定他存在一定小野 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0005347375 | 0.1858 | 17 | 但我们却不知道他是什么只能确定他存在以立小业 | cer | 0.2727 |
| 1280 | causal_full_asr | emilia_zh_0005347375 | 0.1814 | 15 | 但我们却不知道他是什么只能确定他存在以立小意 | cer | 0.2727 |
| 160 | causal_full_asr | emilia_zh_0005600573 | 0.2045 | 20 | 参与的话是一千多然后直播间里面是十人 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005600573 | 0.2268 | 18 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005600573 | 0.2416 | 17 | 参与的话是一千多然后直播间里面是十人 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005600573 | 0.2268 | 16 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 160 | causal_full_asr | emilia_zh_0005713628 | 0.2557 | 16 | 大家会连想到中国话大家会想到的是什么名山大川 | cer | 0.0870 |
| 320 | causal_full_asr | emilia_zh_0005713628 | 0.2466 | 16 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 640 | causal_full_asr | emilia_zh_0005713628 | 0.2329 | 14 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 1280 | causal_full_asr | emilia_zh_0005713628 | 0.2420 | 18 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 160 | causal_full_asr | emilia_zh_0005714451 | 0.1818 | 16 | 什么撞到你进来你可以看到其实前边的三五表 | cer | 0.5200 |
| 320 | causal_full_asr | emilia_zh_0005714451 | 0.1888 | 18 | 虽然赚大宁接下来你可以看到其实前边的三部表 | cer | 0.4800 |
| 640 | causal_full_asr | emilia_zh_0005714451 | 0.1958 | 17 | 生产到宁接下来你可以看到其实前边的桑木表 | cer | 0.4400 |
| 1280 | causal_full_asr | emilia_zh_0005714451 | 0.1923 | 19 | 身份状态应接下来你可以看到其实前面的项目表 | cer | 0.4400 |
| 160 | causal_full_asr | emilia_zh_0005780397 | 0.2148 | 47 | 工作对一定是最低的然后工作一定是消耗的所以再前两天呢就是我也做一丈嘛然后就跟一些小黄让做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.1538 |
| 320 | causal_full_asr | emilia_zh_0005780397 | 0.2148 | 49 | 工作对一定是对地的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友啊做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 640 | causal_full_asr | emilia_zh_0005780397 | 0.2199 | 50 | 工作对一定是对地的然后工作一定是消耗的所以在前两天呢就是我也做一件嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 1280 | causal_full_asr | emilia_zh_0005780397 | 0.2096 | 47 | 工作这一定是最低的然后工作一定是消耗的所以在前两天呢就是我也做HR然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 160 | causal_full_asr | emilia_zh_0005852724 | 0.2602 | 23 | 这个算重要的这这位妈妈我一次希望我的两个女儿可以被 | cer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0005852724 | 0.2677 | 22 | 这个是很重要的这身为妈妈我一次希望我的两个女儿可以被 | cer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0005852724 | 0.2342 | 19 | 这个是很重要的这身为妈妈我一次希望我的两个女儿可以被 | cer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0005852724 | 0.2268 | 20 | 这个是很重要的所以身为妈妈我一直在希望我的两个女儿可以被 | cer | 0.0357 |
| 160 | causal_full_asr | emilia_zh_0005926417 | 0.1906 | 24 | 没有办法就我自己没有办法去得出这样的判断或者被问出来横着呢是参考一些呃 | cer | 0.2500 |
| 320 | causal_full_asr | emilia_zh_0005926417 | 0.2072 | 23 | 没有办法就我自己没有办法去得出这样的判断或者得问出来横着呢是参考一些呃 | cer | 0.2500 |
| 640 | causal_full_asr | emilia_zh_0005926417 | 0.1961 | 20 | 没有办法就我自己没有办法去得出这样的判断或者得问出来横着呢是参考一些呃 | cer | 0.2500 |
| 1280 | causal_full_asr | emilia_zh_0005926417 | 0.2127 | 21 | 没有办法看就我自己没有办法去得出这样的判断或者得问出来横着能是参考一些呃 | cer | 0.2222 |
| 160 | causal_full_asr | emilia_zh_0005960185 | 0.2599 | 17 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 320 | causal_full_asr | emilia_zh_0005960185 | 0.2863 | 19 | 哦我知道一块但是那反正很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 640 | causal_full_asr | emilia_zh_0005960185 | 0.2643 | 21 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 1280 | causal_full_asr | emilia_zh_0005960185 | 0.2643 | 21 | 哦我知道一块但是那番很多人觉得不是玫瑰是天主葵 | cer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0005960324 | 0.1576 | 24 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005960324 | 0.1654 | 25 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005960324 | 0.1550 | 26 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005960324 | 0.1499 | 25 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006041799 | 0.1982 | 13 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 320 | causal_full_asr | emilia_zh_0006041799 | 0.2070 | 15 | 在这个委托发生的时间前两个人就合伙开了这家面馆 | cer | 0.1250 |
| 640 | causal_full_asr | emilia_zh_0006041799 | 0.1630 | 12 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 1280 | causal_full_asr | emilia_zh_0006041799 | 0.1674 | 14 | 在这个委托发生的时间前两个人就合伙开了这家面馆 | cer | 0.1250 |
| 160 | causal_full_asr | emilia_zh_0006056200 | 0.1541 | 16 | 就你要还是要用一些就是能看得到的东西，要不然你这粪还大啊 | cer | 0.3636 |
| 320 | causal_full_asr | emilia_zh_0006056200 | 0.1601 | 19 | 就你要还是要用一些就是能看得到的东西，要不然你这氛围太大了 | cer | 0.3030 |
| 640 | causal_full_asr | emilia_zh_0006056200 | 0.1722 | 19 | 就你要还是要有一些就是能看得到多少东西，要不然画你这氛围太大了 | cer | 0.3030 |
| 1280 | causal_full_asr | emilia_zh_0006056200 | 0.1601 | 21 | 就你要还是要有一些就是能看得到多少东西要不然画你这氛围太大了 | cer | 0.2727 |
| 160 | causal_full_asr | emilia_zh_0006099356 | 0.2531 | 18 | 就整个这个留下还不知道但是你非认主你确定是能拿到钱的什么 | cer | 0.3704 |
| 320 | causal_full_asr | emilia_zh_0006099356 | 0.2573 | 19 | 就整个这个留下还不着的我但是你非认主你确定是能拿到钱的什么 | cer | 0.4074 |
| 640 | causal_full_asr | emilia_zh_0006099356 | 0.2448 | 19 | 就整个这个留下还不知道但是你虽然这种你确定是拿多少钱那是吗 | cer | 0.4815 |
| 1280 | causal_full_asr | emilia_zh_0006099356 | 0.2365 | 19 | 就整个这个留下还不知道但是你非认着你确定是拿刀钱的什么 | cer | 0.4444 |
| 160 | causal_full_asr | emilia_zh_0006174175 | 0.2227 | 34 | 我正好来就记得老师在画画头头的讨论什么那大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.3478 |
| 320 | causal_full_asr | emilia_zh_0006174175 | 0.2133 | 34 | 我正好来就挤到老师在一块偷偷的讨论什么了大家就是老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.3043 |
| 640 | causal_full_asr | emilia_zh_0006174175 | 0.2251 | 34 | 我之后来就记得老师在一会儿偷偷的讨论什么了大家就是老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.2826 |
| 1280 | causal_full_asr | emilia_zh_0006174175 | 0.2133 | 34 | 我然后来就记得老师在一块偷偷的讨论什么呢大家其实老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.2391 |
| 160 | causal_full_asr | emilia_zh_0006270577 | 0.2842 | 22 | 用直接用三份盖其实可以的啊这个也不能哪里我们并不是所有的人都需要打 | cer | 0.3333 |
| 320 | causal_full_asr | emilia_zh_0006270577 | 0.2945 | 23 | 用直接用三份盖其实可以的啊这个也不能理我们并不是所有的人都需要的 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0006270577 | 0.2603 | 22 | 用直接用三份盖其实可以的啊这个也不能拿礼物并不是所有的人都是需要的 | cer | 0.2727 |
| 1280 | causal_full_asr | emilia_zh_0006270577 | 0.2637 | 23 | 用直接用三份盖其实可以的啊这个也不能理我并不是所有的人都需要的 | cer | 0.2424 |
| 160 | causal_full_asr | emilia_zh_0006330755 | 0.2212 | 16 | And now later they were standing in the graveyard of the old stone church. | wer | 0.3571 |
| 320 | causal_full_asr | emilia_zh_0006330755 | 0.1935 | 16 | An hour later they were standing in the graveyard of the old stone church. | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0006330755 | 0.1982 | 17 | An hour later they were standing in the graveyard of the old stone church. | wer | 0.2143 |
| 1280 | causal_full_asr | emilia_zh_0006330755 | 0.1797 | 16 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 160 | causal_full_asr | emilia_zh_0006404958 | 0.2464 | 22 | The characters are like monkeys in winter sickening up withered wines, whitening water. | wer | 0.4667 |
| 320 | causal_full_asr | emilia_zh_0006404958 | 0.2429 | 24 | The characters are like monkeys in winter sickening up withered wines, watching in water. | wer | 0.4667 |
| 640 | causal_full_asr | emilia_zh_0006404958 | 0.2286 | 21 | The characters are like monkeys in winter sickening up withered wines, watching in water. | wer | 0.4667 |
| 1280 | causal_full_asr | emilia_zh_0006404958 | 0.1821 | 19 | The characters are like monkeys in winter, sickening up, rid of the wines, watching in water. | wer | 0.6000 |
| 160 | causal_full_asr | emilia_zh_0006405437 | 0.1503 | 21 | The two little boys count of club tax rules are as clear as mud tax rules ours clear as mud | wer | 0.1000 |
| 320 | causal_full_asr | emilia_zh_0006405437 | 0.1640 | 25 | The two little boys kind of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.0500 |
| 640 | causal_full_asr | emilia_zh_0006405437 | 0.1435 | 25 | The two little boys can of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0006405437 | 0.1253 | 24 | The two little boys can of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.1000 |
| 160 | causal_full_asr | emilia_zh_0006502797 | 0.2517 | 39 | 我觉得这个委托人可我觉得那你说古代的时候和他的上头那古人都应该挖到比如说这个明代开始就要扔核桃嗯你你翻不出头 | cer | 0.4590 |
| 320 | causal_full_asr | emilia_zh_0006502797 | 0.2628 | 35 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就挖进核桃嗯你你发明出套 | cer | 0.3279 |
| 640 | causal_full_asr | emilia_zh_0006502797 | 0.2762 | 37 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就把这套嗯给发明出套了 | cer | 0.3443 |
| 1280 | causal_full_asr | emilia_zh_0006502797 | 0.2539 | 41 | 我觉得这个委托人可我觉得那你说古代的时候和他的手套那古人都已经挖了比如说这个明代开始就把这手套嗯你你发明手套 | cer | 0.3279 |
| 160 | causal_full_asr | emilia_zh_0006598177 | 0.2366 | 25 | 阿叔这个女性受害人之前案件当中如果跟他的啊五十几岁都是这个亲密关系 | cer | 0.3226 |
| 320 | causal_full_asr | emilia_zh_0006598177 | 0.2390 | 27 | 啊说这个女性受害人这些案件当中，如果我跟她的啊五十几岁都是这个亲密关系 | cer | 0.2581 |
| 640 | causal_full_asr | emilia_zh_0006598177 | 0.2268 | 28 | 啊说这个女性受害人这些案件当中都敢跟他的啊五十几人都是这个亲密关系 | cer | 0.1935 |
| 1280 | causal_full_asr | emilia_zh_0006598177 | 0.2439 | 28 | 啊说这个女性受害人这些案件当中如果跟她的啊五十几岁都是这个亲密关系 | cer | 0.1935 |
| 160 | causal_full_asr | emilia_zh_0006610442 | 0.2575 | 15 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0006610442 | 0.2934 | 15 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 640 | causal_full_asr | emilia_zh_0006610442 | 0.2934 | 16 | 我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 1280 | causal_full_asr | emilia_zh_0006610442 | 0.2695 | 16 | 嗯我觉得这这件事蛮有意思就是这种 | cer | 0.2778 |
| 160 | causal_full_asr | emilia_zh_0006659550 | 0.2885 | 16 | 中期的数据看，已经有例这个不时啊他们能做出两次错的判断 | cer | 0.4138 |
| 320 | causal_full_asr | emilia_zh_0006659550 | 0.2727 | 16 | 中期的数据看起来有利益这个不时他能做出两次错的判断 | cer | 0.3448 |
| 640 | causal_full_asr | emilia_zh_0006659550 | 0.2806 | 18 | 中期的数据看就有利于这个布什让他能做出两次错误的判断 | cer | 0.2414 |
| 1280 | causal_full_asr | emilia_zh_0006659550 | 0.2846 | 17 | 中期的数据看及有利这个不时难道还能做出两次错误的判断 | cer | 0.3448 |
| 160 | causal_full_asr | emilia_zh_0006884293 | 0.2039 | 34 | 不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0286 |
| 320 | causal_full_asr | emilia_zh_0006884293 | 0.2187 | 33 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0006884293 | 0.2162 | 30 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0006884293 | 0.2138 | 29 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006992873 | 0.1272 | 16 | 绿色的城堡大曹人首先警觉起来 | cer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0006992873 | 0.1623 | 18 | 绿色的城堡道草人首先警觉起来 | cer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0006992873 | 0.2105 | 18 | The green castle was first alerted by Cao Ren. | cer | 2.7143 |
| 1280 | causal_full_asr | emilia_zh_0006992873 | 0.2018 | 18 | The green castle was first alerted by Cao Ren. | cer | 2.7143 |
| 160 | causal_full_asr | emilia_zh_0007060544 | 0.2125 | 8 | If what you do is truly valuable to others | cer | 2.2667 |
| 320 | causal_full_asr | emilia_zh_0007060544 | 0.2125 | 10 | If what you do is truly valuable to others | cer | 2.2667 |
| 640 | causal_full_asr | emilia_zh_0007060544 | 0.1938 | 12 | If what you do is truly valuable to others | cer | 2.2667 |
| 1280 | causal_full_asr | emilia_zh_0007060544 | 0.1875 | 12 | If what you do is truly valuable to others | cer | 2.2667 |
| 160 | causal_full_asr | emilia_zh_0007353988 | 0.2085 | 36 | 两颗核弹都是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0426 |
| 320 | causal_full_asr | emilia_zh_0007353988 | 0.1956 | 36 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 640 | causal_full_asr | emilia_zh_0007353988 | 0.1900 | 36 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 1280 | causal_full_asr | emilia_zh_0007353988 | 0.1993 | 38 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 160 | causal_full_asr | emilia_zh_0007761003 | 0.1673 | 16 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0007761003 | 0.2007 | 17 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 640 | causal_full_asr | emilia_zh_0007761003 | 0.1673 | 18 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0007761003 | 0.1636 | 19 | 如果你像你的团队成员陪你走的更远那你就需要 | cer | 0.1818 |
| 160 | streaming_asr | CommonVoice_EN_0000042263 | 0.1421 | 10 | There remained remained the yes were spent and port | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000042263 | 0.1684 | 11 | Every main. of the yes where was bent in port | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000042263 | 0.1947 | 11 | That 缅因州 of the yes where was spent in port | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.1579 | 10 | That 缅因州 of the yes where was spent in port | wer | 0.4444 |
| 160 | streaming_asr | CommonVoice_EN_0000042530 | 0.1232 | 21 | It International exchanged under under is Multiple functions building of caffeeteria combination and as for | wer | 0.8462 |
| 320 | streaming_asr | CommonVoice_EN_0000042530 | 0.1469 | 18 | He International exchanged under under in Multiple functions building of capitaria combination and as for | wer | 0.9231 |
| 640 | streaming_asr | CommonVoice_EN_0000042530 | 0.1801 | 24 | He International exchanged under it in Multiple functions building of caffeeteria accommodation and asks from | wer | 0.8462 |
| 1280 | streaming_asr | CommonVoice_EN_0000042530 | 0.2062 | 28 | He International exchanged and and in Multiple functions building of capitory accommodation and asks from | wer | 0.8462 |
| 160 | streaming_asr | CommonVoice_EN_0000068601 | 0.1958 | 25 | The all high all high leaving collects documents on the his he of the mice we were released | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000068601 | 0.2264 | 28 | The all high all high leaving with ease collects 文档 on the his three he all the eyes we were released | wer | 1.4167 |
| 640 | streaming_asr | CommonVoice_EN_0000068601 | 0.2500 | 34 | The a of leaving a collects dominals on the he's free he are the mice we were released | wer | 1.0833 |
| 1280 | streaming_asr | CommonVoice_EN_0000068601 | 0.2288 | 31 | The a of leaving a collects domnance on the his three he of organized Liberalize | wer | 0.6667 |
| 160 | streaming_asr | CommonVoice_EN_0000116742 | 0.1311 | 11 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000116742 | 0.1516 | 11 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000116742 | 0.1557 | 12 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000116742 | 0.1230 | 11 | It This one after better now mountains in s Sweden | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000126367 | 0.2071 | 15 | You some always some grids to tish for Exodusism and airage in the Poor | wer | 0.6923 |
| 320 | streaming_asr | CommonVoice_EN_0000126367 | 0.2357 | 17 | You you always some grids your this for Expedity and airage it the Poor | wer | 0.6923 |
| 640 | streaming_asr | CommonVoice_EN_0000126367 | 0.2607 | 16 | You you always some grids some this for Exicism and airage in the Poor | wer | 0.6923 |
| 1280 | streaming_asr | CommonVoice_EN_0000126367 | 0.2250 | 15 | You you always that grids your tish a for Expedicism and airage it the Poor | wer | 0.7692 |
| 160 | streaming_asr | CommonVoice_EN_0000159911 | 0.1408 | 14 | The Series see particular claimed during the am tired storyline | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000159911 | 0.1655 | 14 | The Serious severe particular claimed during the am tired storyline | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000159911 | 0.2113 | 17 | The seerious see particular claimed during the amper storyline | wer | 0.5556 |
| 1280 | streaming_asr | CommonVoice_EN_0000159911 | 0.1338 | 12 | The severe severe particular claimed during the amper storyline | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000188343 | 0.0866 | 5 | The of of just thoroughly | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000188343 | 0.1496 | 8 | They of a just thoroughly | wer | 0.6000 |
| 640 | streaming_asr | CommonVoice_EN_0000188343 | 0.1654 | 6 | Day of a just thoroughly | wer | 0.8000 |
| 1280 | streaming_asr | CommonVoice_EN_0000188343 | 0.1024 | 6 | They of a just Verily the | wer | 1.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000189191 | 0.2043 | 10 | And The boy brought teeth orse boss | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000189191 | 0.2258 | 9 | And The boy brought teeth wharshed whose | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000189191 | 0.2258 | 9 | And The boy brought teeth wharshed wholeser | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000189191 | 0.3011 | 12 | And The boy brought teeth wharshed wholeser The | wer | 0.8333 |
| 160 | streaming_asr | CommonVoice_EN_0000237791 | 0.1311 | 10 | I this Ben to but a stone old | wer | 0.8750 |
| 320 | streaming_asr | CommonVoice_EN_0000237791 | 0.1803 | 10 | The does he Ben to but a stone old | wer | 0.8750 |
| 640 | streaming_asr | CommonVoice_EN_0000237791 | 0.1749 | 10 | The does he bunch to but a stone old | wer | 0.8750 |
| 1280 | streaming_asr | CommonVoice_EN_0000237791 | 0.2678 | 14 | The dusty bunch to but a stone all | wer | 0.6250 |
| 160 | streaming_asr | CommonVoice_EN_0000238037 | 0.1071 | 10 | You one of fire the of and but but it it | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000238037 | 0.1688 | 13 | You five of fire can all you of the it could the | wer | 0.7857 |
| 640 | streaming_asr | CommonVoice_EN_0000238037 | 0.1786 | 15 | You five of fire can all but you of the it could the | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000238037 | 0.2468 | 18 | You Find of fire can all the you for the the could the | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000263325 | 0.0830 | 7 | He as one under brother travert re- nelson | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000263325 | 0.1037 | 10 | He as one under brother travert re- nelson | wer | 0.5000 |
| 640 | streaming_asr | CommonVoice_EN_0000263325 | 0.1162 | 12 | He as one younger brother travert re- nelson | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000263325 | 0.1369 | 11 | He was one young brother travert re- Nelson | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000285524 | 0.1168 | 21 | The Townships pro proximity to the see favor moorie tried and fish f factories | wer | 0.5833 |
| 320 | streaming_asr | CommonVoice_EN_0000285524 | 0.1368 | 22 | The Townships pro proximity to of see favor moorie tried and fish factories | wer | 0.5833 |
| 640 | streaming_asr | CommonVoice_EN_0000285524 | 0.1396 | 20 | The Townships pro proximity to the see favor moorie tried and fish factories | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000285524 | 0.1567 | 20 | The Townships pro proximity to of see favor murray tried and fish Factories | wer | 0.5833 |
| 160 | streaming_asr | CommonVoice_EN_0000286138 | 0.1310 | 11 | The port reminded the old man the he it said something about hidden dresses | wer | 0.2857 |
| 320 | streaming_asr | CommonVoice_EN_0000286138 | 0.1470 | 13 | The port reminded the old man the he it said something about hidden dress | wer | 0.2857 |
| 640 | streaming_asr | CommonVoice_EN_0000286138 | 0.1661 | 13 | The port reminded the old man the he it said something about hidden trasured | wer | 0.2857 |
| 1280 | streaming_asr | CommonVoice_EN_0000286138 | 0.1661 | 14 | The port remind of the old man the he it said something about hidden trasured | wer | 0.4286 |
| 160 | streaming_asr | CommonVoice_EN_0000311292 | 0.1660 | 15 | It had any exception qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000311292 | 0.1830 | 16 | It had any exceptional qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000311292 | 0.1872 | 14 | And had any exceptional qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000311292 | 0.2638 | 17 | A had any exception qualities <\|glm_semantic_5661\|>pared previous aircraft | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000331841 | 0.1692 | 8 | Just did rel the me the asked time | wer | 0.6250 |
| 320 | streaming_asr | CommonVoice_EN_0000331841 | 0.2462 | 11 | Just did rode the me the asked time | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000331841 | 0.2692 | 11 | Just did rode the me the asked time | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000331841 | 0.2692 | 11 | You did rode the me the asked time | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000332324 | 0.1478 | 15 | And building post for can what where entirety d destroyed | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000332324 | 0.1652 | 15 | Many building post post can why where entirety d destroyed | wer | 0.8889 |
| 640 | streaming_asr | CommonVoice_EN_0000332324 | 0.1739 | 18 | Many building pose before can what where entirety d destroyed | wer | 0.8889 |
| 1280 | streaming_asr | CommonVoice_EN_0000332324 | 0.1348 | 14 | Many building post before can what were entirety d destroyed | wer | 0.7778 |
| 160 | streaming_asr | CommonVoice_EN_0000352398 | 0.1993 | 15 | He concluded he I school form lam al ball high guy | wer | 0.7273 |
| 320 | streaming_asr | CommonVoice_EN_0000352398 | 0.2320 | 19 | He complicated he high school form lam al ball high scu | wer | 0.6364 |
| 640 | streaming_asr | CommonVoice_EN_0000352398 | 0.2222 | 17 | He completed he high school from lam a all high scrooge | wer | 0.4545 |
| 1280 | streaming_asr | CommonVoice_EN_0000352398 | 0.2680 | 22 | He complicated he high school room ram all ball high scu | wer | 0.5455 |
| 160 | streaming_asr | CommonVoice_EN_0000381265 | 0.2105 | 13 | You vexisted int plane began | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000381265 | 0.3158 | 13 | You vexisted ins't plane began | wer | 0.8000 |
| 640 | streaming_asr | CommonVoice_EN_0000381265 | 0.3487 | 13 | You vexisted ins't plane megan | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000381265 | 0.1908 | 11 | You vexisted ins't plane megan | wer | 1.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000381462 | 0.1767 | 17 | Girard spent to this of s life with mysticism | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000381462 | 0.2241 | 15 | Girard spent the this of s life with Mr. Sis | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000381462 | 0.1940 | 18 | Girard spent to that of s life with Mr. Sis | wer | 0.5556 |
| 1280 | streaming_asr | CommonVoice_EN_0000381462 | 0.1509 | 14 | Girard spent the this of s life with Mr. Sis | wer | 0.4444 |
| 160 | streaming_asr | CommonVoice_EN_0000430128 | 0.1800 | 14 | Luo to can you test was capacity designed why rose bad | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000430128 | 0.2040 | 18 | Luo to can you taste was cap can see designed why rose bad | wer | 0.8571 |
| 640 | streaming_asr | CommonVoice_EN_0000430128 | 0.2280 | 18 | Luo who can you they was cap can see designed white rose bad | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000430128 | 0.2040 | 16 | Law who can you they was capancy designed white rose bad | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000433992 | 0.1882 | 15 | Nobody want the disgust how you all and got trap in a John learnt box | wer | 0.7143 |
| 320 | streaming_asr | CommonVoice_EN_0000433992 | 0.2288 | 20 | Nobody want the Discuss how you all and got trap in a John learnt box | wer | 0.6429 |
| 640 | streaming_asr | CommonVoice_EN_0000433992 | 0.2768 | 24 | Nobody want the Discuss how you all and got trap in a kind learnt box | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000433992 | 0.3210 | 22 | Nobody want the Discuss how you old in the got trap in a kind learnt box | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000434002 | 0.1074 | 5 | He idea frightened him | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000434002 | 0.1983 | 10 | He idea Frightened him | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000434002 | 0.2149 | 9 | He idea Frightened him | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000434002 | 0.1983 | 11 | He idea Frightened him | wer | 0.2500 |
| 160 | streaming_asr | CommonVoice_EN_0000471209 | 0.1301 | 5 | The boy begun begun you can to the groom | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000471209 | 0.2260 | 13 | The boy begun begun you can to the jun | wer | 0.7500 |
| 640 | streaming_asr | CommonVoice_EN_0000471209 | 0.1986 | 13 | The boy begun begun a can the jun | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000471209 | 0.1918 | 11 | The boy begun begun a can to the jun | wer | 0.7500 |
| 160 | streaming_asr | CommonVoice_EN_0000501889 | 0.1192 | 15 | We is part to motorcycle neet to the building | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000501889 | 0.1917 | 17 | We Usually part to motorcycle nein to the building | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000501889 | 0.1969 | 16 | We is part to motorcycle nein to the building | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000501889 | 0.1813 | 17 | We is part to motorcycle nay to the building | wer | 0.6667 |
| 160 | streaming_asr | CommonVoice_EN_0000519794 | 0.1595 | 16 | Give pass from conclusion is list of Then the on this course | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000519794 | 0.2371 | 19 | You by of in the equation is list of Then that on this goal | wer | 1.0000 |
| 640 | streaming_asr | CommonVoice_EN_0000519794 | 0.2672 | 21 | You By of in a transition is list of Then the on this call | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000519794 | 0.2457 | 18 | You've by of in conclusion is list of Then the and goes | wer | 0.8333 |
| 160 | streaming_asr | CommonVoice_EN_0000520146 | 0.1793 | 12 | Many chross supporter I you spring expanse | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000520146 | 0.2120 | 12 | Many chross supporter I leave spring suspension | wer | 0.7500 |
| 640 | streaming_asr | CommonVoice_EN_0000520146 | 0.1848 | 13 | Many rock's supporter I leave spere suspensions | wer | 0.7500 |
| 1280 | streaming_asr | CommonVoice_EN_0000520146 | 0.1739 | 13 | Many rocks supporter I leave spere suspensions | wer | 0.7500 |
| 160 | streaming_asr | CommonVoice_EN_0000555807 | 0.0991 | 12 | I color <\|glm_semantic_14383\|>ence of as as stop right | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000555807 | 0.1651 | 15 | The color intensifies as as stop right | wer | 0.4286 |
| 640 | streaming_asr | CommonVoice_EN_0000555807 | 0.1745 | 16 | The color intensifies is this stop brightens | wer | 0.4286 |
| 1280 | streaming_asr | CommonVoice_EN_0000555807 | 0.1981 | 19 | The colour intensifies as this stop brightens | wer | 0.4286 |
| 160 | streaming_asr | CommonVoice_EN_0000555853 | 0.1148 | 12 | Sam some he said from the the a | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000555853 | 0.1721 | 16 | Shaving water Sam some he said from then the a | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000555853 | 0.1680 | 13 | Shaving water Sam some he said from the the a | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000555853 | 0.1762 | 15 | Shaving water Sam He he said from the the a | wer | 0.6250 |
| 160 | streaming_asr | CommonVoice_EN_0000593898 | 0.2143 | 19 | In We've seen here's seen George's the spent be in a mad wholes hole roots | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000593898 | 0.2536 | 21 | In We've here's seen George's the spent be in a mad wholesore roots | wer | 0.8333 |
| 640 | streaming_asr | CommonVoice_EN_0000593898 | 0.3000 | 24 | In was here's seen George's the spent be in a mad wholes cold roots | wer | 0.9167 |
| 1280 | streaming_asr | CommonVoice_EN_0000593898 | 0.3179 | 18 | In was here's saint George's the spent be in a mad class hall roots | wer | 0.8333 |
| 160 | streaming_asr | DailyTalk_0000001997 | 0.2418 | 10 | The scotch please | wer | 0.5000 |
| 320 | streaming_asr | DailyTalk_0000001997 | 0.2527 | 9 | The scotch please | wer | 0.5000 |
| 640 | streaming_asr | DailyTalk_0000001997 | 0.2747 | 11 | Ding Scott please | wer | 0.7500 |
| 1280 | streaming_asr | DailyTalk_0000001997 | 0.2527 | 10 | Give scotch please | wer | 0.2500 |
| 160 | streaming_asr | EN_B00013_S05834_W000745 | 0.2356 | 28 | First the then like could still stay stay was straw resilience the broken bone in it's chests with make make lose any build a to five to | wer | 0.6800 |
| 320 | streaming_asr | EN_B00013_S05834_W000745 | 0.2356 | 28 | The if then like could still stay for was strong was resilience the broken bone in it's chast which make make lose any build a to five to | wer | 0.6400 |
| 640 | streaming_asr | EN_B00013_S05834_W000745 | 0.2270 | 31 | First the from like could still stay through was strong <\|glm_semantic_298\|>ions the broken bone in it's chest which make and lose any build a to five to | wer | 0.6000 |
| 1280 | streaming_asr | EN_B00013_S05834_W000745 | 0.1897 | 31 | Even in the like could still stay through was strong was resilience the broken bone some is chest which make and lose any build a to five to | wer | 0.5600 |
| 160 | streaming_asr | EN_B00013_S05888_W000041 | 0.1269 | 31 | The I me and of of that I of I of the the to the excess of be big friend chat port ages and seeing in it more make in the a guy you could the go you be that of the | wer | 0.7551 |
| 320 | streaming_asr | EN_B00013_S05888_W000041 | 0.1746 | 37 | The I me and my it that I of I I do to to the besition of be big friend chat port ages and seeing in it more make in the go you could you go you be next of the | wer | 0.7551 |
| 640 | streaming_asr | EN_B00013_S05888_W000041 | 0.1779 | 40 | The I me and my it that I of I I the the to the besition of be big friend chat for ages and seeing in it more make in the guys you could you a guy a be next of the | wer | 0.7143 |
| 1280 | streaming_asr | EN_B00013_S05888_W000041 | 0.1763 | 42 | The I me and of it a I of I I love the to the besition of being big friend chat for ages and seeing in it more make in the a guy he a the a guy a be next of the | wer | 0.7143 |
| 160 | streaming_asr | EN_B00013_S06799_W000009 | 0.2260 | 29 | And We that the can how better model is sure safe fearness and help and user to have better reliety appropriation trial | wer | 0.5238 |
| 320 | streaming_asr | EN_B00013_S06799_W000009 | 0.2374 | 33 | And We that that can had better model is true Safety fearness and help and user to have better reliety appropriation trial | wer | 0.4762 |
| 640 | streaming_asr | EN_B00013_S06799_W000009 | 0.1963 | 30 | And We that that can had better model is true Safety fearness and help and user to have better reliety appropriation trial | wer | 0.4762 |
| 1280 | streaming_asr | EN_B00013_S06799_W000009 | 0.1758 | 30 | And We That was can had better model is true Safety fearness and help and user to have better reularity appropriation trial | wer | 0.4762 |
| 160 | streaming_asr | EN_B00036_S05316_W000048 | 0.1617 | 20 | This Sinth floor see star has three for why arm spend at taste first see action | wer | 0.7059 |
| 320 | streaming_asr | EN_B00036_S05316_W000048 | 0.1848 | 20 | This seven flowers see star has three for wide arm spend at taste for see fortune | wer | 0.5882 |
| 640 | streaming_asr | EN_B00036_S05316_W000048 | 0.1452 | 16 | This SMFLO see star has three for why arm spend at taste first see fortune | wer | 0.6471 |
| 1280 | streaming_asr | EN_B00036_S05316_W000048 | 0.1617 | 20 | This Sunflower see start has three for why arm spend at taste first see reaching | wer | 0.6471 |
| 160 | streaming_asr | EN_B00043_S01954_W000033 | 0.1832 | 33 | Because was his more of the and rest in Motion it's this passity of Further and endlessly renov researched And not just and our private I this | wer | 0.7500 |
| 320 | streaming_asr | EN_B00043_S01954_W000033 | 0.1703 | 31 | Because love is more of the and rest in Motion it's this passity of Further and endlessly renov research And not just to not private I this | wer | 0.7083 |
| 640 | streaming_asr | EN_B00043_S01954_W000033 | 0.1466 | 30 | Because love is more of the and rest in Motion it's this passity of for and endlessly renov resourced a not is to our private I this | wer | 0.7500 |
| 1280 | streaming_asr | EN_B00043_S01954_W000033 | 0.1379 | 33 | Because love is more of the and rest in Motion it this passity of verb and endlessly renov resourced And not is to not private I this | wer | 0.7500 |
| 160 | streaming_asr | EN_B00043_S02661_W000018 | 0.1623 | 18 | Because opper frequent your mellowed dramatic not to say unrealistic | wer | 0.5556 |
| 320 | streaming_asr | EN_B00043_S02661_W000018 | 0.1887 | 19 | Because opper frequent your mellowed and not to say unrealistic | wer | 0.5556 |
| 640 | streaming_asr | EN_B00043_S02661_W000018 | 0.1472 | 16 | Big opper -frequency your mellowed and not to say unrealistic | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00043_S02661_W000018 | 0.1396 | 16 | Big opper -frequency are mellowed and not to say unrealistic | wer | 0.5556 |
| 160 | streaming_asr | EN_B00048_S01234_W000037 | 0.2485 | 14 | And no cases when that sure the and more going be it dog | wer | 0.6875 |
| 320 | streaming_asr | EN_B00048_S01234_W000037 | 0.2000 | 10 | And no cases were that sure the and more going to be it dog | wer | 0.6250 |
| 640 | streaming_asr | EN_B00048_S01234_W000037 | 0.1818 | 11 | And the cases run that sure the and going to be a dog | wer | 0.5625 |
| 1280 | streaming_asr | EN_B00048_S01234_W000037 | 0.1758 | 10 | And the cases run that sure the and going to be a dog | wer | 0.5625 |
| 160 | streaming_asr | EN_B00048_S02289_W000002 | 0.1036 | 9 | This kind of mussel is Mostly connected to my bones | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S02289_W000002 | 0.1143 | 14 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 640 | streaming_asr | EN_B00048_S02289_W000002 | 0.1071 | 15 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 1280 | streaming_asr | EN_B00048_S02289_W000002 | 0.1036 | 14 | This kind of mussel is Mostly collected to my bones | wer | 0.2000 |
| 160 | streaming_asr | EN_B00048_S02307_W000043 | 0.0641 | 18 | Let's check He likes to eep it's a I don't want to you they sally That isn't my shake | wer | 0.3333 |
| 320 | streaming_asr | EN_B00048_S02307_W000043 | 0.0549 | 12 | Let's check He likes to eep each I don't want to he they sal That isn't my shake to | wer | 0.3333 |
| 640 | streaming_asr | EN_B00048_S02307_W000043 | 0.0604 | 16 | Let's check He likes to eep each I don't want to you they sal That Isn't my shake to | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00048_S02307_W000043 | 0.0623 | 18 | Let's check He likes to eep heaths I don't want to he they sally That Isn't my shake to | wer | 0.3333 |
| 160 | streaming_asr | EN_B00048_S03599_W000339 | 0.2156 | 19 | And Such round I my shriek froze the blood of Every one close by | wer | 0.4615 |
| 320 | streaming_asr | EN_B00048_S03599_W000339 | 0.2062 | 21 | I turn to round I my shriek froze the blood of Everyone close by | wer | 0.3077 |
| 640 | streaming_asr | EN_B00048_S03599_W000339 | 0.2000 | 22 | I turned round I my shriek froze the blood of Everyone close by | wer | 0.1538 |
| 1280 | streaming_asr | EN_B00048_S03599_W000339 | 0.1437 | 19 | And turned around I my screek froze the blood of Everyone close by | wer | 0.2308 |
| 160 | streaming_asr | EN_B00048_S05933_W000060 | 0.2833 | 12 | That American has really interesting culture that with fed | wer | 0.5000 |
| 320 | streaming_asr | EN_B00048_S05933_W000060 | 0.2611 | 13 | So American has really interesting culture that with feder | wer | 0.5000 |
| 640 | streaming_asr | EN_B00048_S05933_W000060 | 0.1778 | 10 | So America has really interesting culture that with faith | wer | 0.4000 |
| 1280 | streaming_asr | EN_B00048_S05933_W000060 | 0.1667 | 12 | South merck has really interesting culture that that faith | wer | 0.4000 |
| 160 | streaming_asr | EN_B00048_S05961_W000019 | 0.1844 | 29 | I'll just I'll just use this there to explained orge of the vast variety of caryotic organisms | wer | 0.5625 |
| 320 | streaming_asr | EN_B00048_S05961_W000019 | 0.1671 | 25 | I just I use this there to explain orge of the vast variety of carriotic organisms | wer | 0.4375 |
| 640 | streaming_asr | EN_B00048_S05961_W000019 | 0.1556 | 29 | biologists I use this there to explained orge of the vast variety of carriotic organisms | wer | 0.3750 |
| 1280 | streaming_asr | EN_B00048_S05961_W000019 | 0.1326 | 25 | I'll just I'll just use this there to explained orge of the vast variety of carriotic organisms | wer | 0.5625 |
| 160 | streaming_asr | EN_B00048_S07042_W000076 | 0.2515 | 13 | And no I in the have time but things way before before a here | wer | 0.6429 |
| 320 | streaming_asr | EN_B00048_S07042_W000076 | 0.2270 | 13 | And no I in the have time but things way before you a here | wer | 0.5714 |
| 640 | streaming_asr | EN_B00048_S07042_W000076 | 0.1534 | 10 | And no I at the time have turned but things way before you a here | wer | 0.6429 |
| 1280 | streaming_asr | EN_B00048_S07042_W000076 | 0.1534 | 8 | I no I in the of turn but things way before you a here | wer | 0.5714 |
| 160 | streaming_asr | EN_B00048_S07862_W000265 | 0.1835 | 27 | The me mean of crossing large Areas of water was in no sailing ship driven I the wind | wer | 0.2222 |
| 320 | streaming_asr | EN_B00048_S07862_W000265 | 0.1543 | 21 | The only means of crossing large Areas of water was in no sailing ship driven I the wind | wer | 0.1111 |
| 640 | streaming_asr | EN_B00048_S07862_W000265 | 0.1330 | 19 | The only main of crossing large Areas of water was in no selling ship driven I the wind | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00048_S07862_W000265 | 0.1410 | 20 | The only mean of crossing large Areas of water was it no sailing ship driven I the wind | wer | 0.2222 |
| 160 | streaming_asr | EN_B00048_S08821_W000040 | 0.2193 | 24 | Or you're the participating going to funnaries the sweetened a of like bore the van if Possible | wer | 0.6667 |
| 320 | streaming_asr | EN_B00048_S08821_W000040 | 0.2259 | 27 | I are you they participating you funnaries the sweeping a with like borrow the van if Possible | wer | 0.6667 |
| 640 | streaming_asr | EN_B00048_S08821_W000040 | 0.1894 | 23 | I are you the participates you funnaries the sweeping a with like brought the then if Possible | wer | 0.7778 |
| 1280 | streaming_asr | EN_B00048_S08821_W000040 | 0.1761 | 23 | I are you the participates in funnaries the screeked a with like brought the then if Possible | wer | 0.7778 |
| 160 | streaming_asr | EN_B00048_S09662_W000003 | 0.1157 | 25 | Um here again So what don't you listen to the .log board the first time that listen how presented hasn't heard has goodbye and and could back and the but words | wer | 0.5625 |
| 320 | streaming_asr | EN_B00048_S09662_W000003 | 0.1550 | 30 | I'll pay here good so was don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm the that's listen how President hasn't known has goodbye. and and could back and the but words | wer | 0.6250 |
| 640 | streaming_asr | EN_B00048_S09662_W000003 | 0.1550 | 29 | I'll pay here great so what don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm that's listen how President hasn't had has goodbye. and and could back and the could words | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00048_S09662_W000003 | 0.1426 | 32 | I'll pay I'll pay good so what don't you listen to the <\|write_generate\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|eng\|><\|start_content\|><\|bicodec_global_3394\|><\|bicodec_semantic_6298\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|><\|bicodec_semantic_1061\|> board the first I'm that's listen how President hasn't had has goodbye. and and could back and the could words | wer | 0.6250 |
| 160 | streaming_asr | EN_B00052_S08802_W000006 | 0.1356 | 11 | We are Beautiful Pretty漂亮 | wer | 1.0000 |
| 320 | streaming_asr | EN_B00052_S08802_W000006 | 0.1388 | 11 | Were Beautiful Pretty漂亮 | wer | 0.6667 |
| 640 | streaming_asr | EN_B00052_S08802_W000006 | 0.1293 | 10 | Were Beautiful Pretty漂亮 | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00052_S08802_W000006 | 0.1104 | 12 | were Beautiful Pretty漂亮 | wer | 0.6667 |
| 160 | streaming_asr | EN_B00058_S01128_W000118 | 0.2270 | 9 | As tall for or China I stick out like saw the | wer | 0.4615 |
| 320 | streaming_asr | EN_B00058_S01128_W000118 | 0.2393 | 10 | As tall for or chinese I stick out like saw the | wer | 0.5385 |
| 640 | streaming_asr | EN_B00058_S01128_W000118 | 0.2393 | 9 | As tall for or China I stick out like saw some | wer | 0.4615 |
| 1280 | streaming_asr | EN_B00058_S01128_W000118 | 0.1779 | 10 | As tell for or China a stick out like saw the | wer | 0.6154 |
| 160 | streaming_asr | EN_B00058_S03125_W000010 | 0.1943 | 43 | We from come some count with base each qing is it's of from yeah to red indicating to supposition solution is a base at was I termor extinct turns red was a of and contacted with any kind of base | wer | 0.6190 |
| 320 | streaming_asr | EN_B00058_S03125_W000010 | 0.1833 | 43 | We tremor come some contemplate with base each change is it's of from yeah to red indicating to so be solution is a base at was I termor extinct turns red was a of some contacted with in can of base | wer | 0.6667 |
| 640 | streaming_asr | EN_B00058_S03125_W000010 | 0.1833 | 44 | I trumbery come some content with base each change is it's of from yeah to red indicating to supposition solution is bas. That was I termor extinct turns red was and of some contacted with in can of base | wer | 0.6905 |
| 1280 | streaming_asr | EN_B00058_S03125_W000010 | 0.1643 | 42 | I term come some content with base each change is it's of from yeah to red indicating to supposition solution is bas. That was I termor extinct turns red when and of and contacted with in can of base | wer | 0.6667 |
| 160 | streaming_asr | EN_B00058_S03144_W000037 | 0.2143 | 19 | Long compets Sentences with multiples paragraphs a email | wer | 0.4444 |
| 320 | streaming_asr | EN_B00058_S03144_W000037 | 0.1830 | 16 | long Complex Sentences with Multiple paragraphs a email | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S03144_W000037 | 0.1473 | 16 | long Complex Sentences with modifiable paragraphs a email | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00058_S03144_W000037 | 0.1562 | 16 | long Complex Sentences with modifiable paragraphs a email | wer | 0.3333 |
| 160 | streaming_asr | EN_B00058_S03808_W000018 | 0.1573 | 49 | We can also say I had gone having to functions in there to to a pass perfect as well was the perfect hands with the for so and she can see what inflation does is it changers it actually change the word | wer | 0.3023 |
| 320 | streaming_asr | EN_B00058_S03808_W000018 | 0.1418 | 48 | We can also say I had gone having to functions in there to indication a pass perfect as well was the perfect hands with the for so a she can see what inflation does is it changers it actually change the word | wer | 0.3023 |
| 640 | streaming_asr | EN_B00058_S03808_W000018 | 0.1251 | 47 | We you also say I had gone having to functions in there to in the cave a pass perfect as well was the perfect lengths with the for so a she can see what inflation does is it changers it actually change the word | wer | 0.3721 |
| 1280 | streaming_asr | EN_B00058_S03808_W000018 | 0.1263 | 46 | We you also say I had gone having to inflictions in there to in the cave a pass perfect as well was the perfect lengths with the for so and she can see what in fluctuation does is it changers it actually change the word | wer | 0.3953 |
| 160 | streaming_asr | EN_B00058_S03815_W000004 | 0.1818 | 24 | my child no how more matches strong in the the make one figure that post | wer | 0.6667 |
| 320 | streaming_asr | EN_B00058_S03815_W000004 | 0.1437 | 18 | My child no her but match it. strong in the the make one figure that post | wer | 0.6667 |
| 640 | streaming_asr | EN_B00058_S03815_W000004 | 0.1290 | 18 | My child no her ball manchic strong in the the make one figure that post | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00058_S03815_W000004 | 0.1232 | 17 | My child no her both manchic strong in the of make one figure that post | wer | 0.6667 |
| 160 | streaming_asr | EN_B00058_S04429_W000016 | 0.1757 | 21 | The you see these they green leaves And this sping flower where I of second | wer | 0.4375 |
| 320 | streaming_asr | EN_B00058_S04429_W000016 | 0.1634 | 25 | The you see these they told green leagues And this being flower where I of second | wer | 0.5625 |
| 640 | streaming_asr | EN_B00058_S04429_W000016 | 0.1287 | 21 | The will see these they tell green leagues And this being flower where I of second | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00058_S04429_W000016 | 0.1262 | 20 | The you see these they tell green leagues And this being flower where I of second | wer | 0.5625 |
| 160 | streaming_asr | EN_B00058_S04431_W000023 | 0.2446 | 12 | It's blood head cap all part of his body war | wer | 0.5000 |
| 320 | streaming_asr | EN_B00058_S04431_W000023 | 0.1793 | 12 | He's love head keep all part of his body warren | wer | 0.6000 |
| 640 | streaming_asr | EN_B00058_S04431_W000023 | 0.1685 | 13 | Here blood head keep all heart of his body warmer | wer | 0.5000 |
| 1280 | streaming_asr | EN_B00058_S04431_W000023 | 0.1685 | 13 | Here blood head keep all part of his body warmer | wer | 0.5000 |
| 160 | streaming_asr | EN_B00058_S06165_W000019 | 0.1800 | 56 | We are you you the short for where you add seem like a colon after the constructor in the in you use This Western text in now the first argument to pass to to west constructor will be store in the question text properity | wer | 0.4634 |
| 320 | streaming_asr | EN_B00058_S06165_W000019 | 0.1825 | 56 | Where you use the short form where you add seem like a colon after the constructor in in you use this question text in now the first argament to pass to to west constructor will be store in the question text properity | wer | 0.3659 |
| 640 | streaming_asr | EN_B00058_S06165_W000019 | 0.1886 | 55 | Or you use the short form where you at seemingly cold after the constructor in the in you use this question text in now the first argament to pass to to west constructor will be store in the quest text properity | wer | 0.3902 |
| 1280 | streaming_asr | EN_B00058_S06165_W000019 | 0.1837 | 58 | Or you use the short form where you at seemingly cold and after the constructor in the in you use This question text in now the first argment to pass to to west constructor will be store in the quest text properity | wer | 0.4146 |
| 160 | streaming_asr | EN_B00058_S06429_W000060 | 0.1414 | 21 | And A lot of believed at part lessly lose her be very give a to to with all love the date that was how there | wer | 0.7143 |
| 320 | streaming_asr | EN_B00058_S06429_W000060 | 0.1382 | 21 | the Along the believed at part loser be very give a to to with of love the date that was how there | wer | 0.6190 |
| 640 | streaming_asr | EN_B00058_S06429_W000060 | 0.1875 | 24 | the Along the believed that part loser be very give a to to with of love deed that was how there | wer | 0.5714 |
| 1280 | streaming_asr | EN_B00058_S06429_W000060 | 0.1612 | 23 | The Alliance believed a part loser be very give a to to with of love the date that was how there | wer | 0.5238 |
| 160 | streaming_asr | EN_B00058_S07483_W000027 | 0.2866 | 18 | And is this fix or high irons rogs | wer | 0.5556 |
| 320 | streaming_asr | EN_B00058_S07483_W000027 | 0.3439 | 15 | And use this Fake our high airwards rodden | wer | 0.5556 |
| 640 | streaming_asr | EN_B00058_S07483_W000027 | 0.2548 | 13 | And use this Fake a high airwards robin | wer | 0.4444 |
| 1280 | streaming_asr | EN_B00058_S07483_W000027 | 0.2357 | 12 | And use this face a high airwards robin | wer | 0.4444 |
| 160 | streaming_asr | EN_B00058_S07511_W000000 | 0.2513 | 20 | Getting everywhere yeah the house the orrying can you really of special the or to school | wer | 0.5789 |
| 320 | streaming_asr | EN_B00058_S07511_W000000 | 0.2256 | 17 | Getting everywhere yeah the house the ringing can you really of special the or to school | wer | 0.5789 |
| 640 | streaming_asr | EN_B00058_S07511_W000000 | 0.2308 | 19 | Again everywhere yeah the house the arning can you really of special the or to school | wer | 0.6316 |
| 1280 | streaming_asr | EN_B00058_S07511_W000000 | 0.2103 | 20 | Getting everywhere yeah the house the turning can you really of special the or to school | wer | 0.5789 |
| 160 | streaming_asr | EN_B00064_S08593_W000000 | 0.1382 | 25 | He Teachers the pass does not exist a fact which longed to as sphere of knowledge And which before before one and the world and a | wer | 0.3704 |
| 320 | streaming_asr | EN_B00064_S08593_W000000 | 0.1451 | 28 | He teach of the pass does not exist a fact which belongs to to a sphere of knowledge And which before before one and the world and a | wer | 0.3704 |
| 640 | streaming_asr | EN_B00064_S08593_W000000 | 0.1297 | 27 | He teacher the pass does not exist a fact which belongs to with the sphere from lodged And which before before one and the world and a | wer | 0.4074 |
| 1280 | streaming_asr | EN_B00064_S08593_W000000 | 0.1195 | 23 | He teach the pass does not exist a fact which belongs to with the sphere of knowledge And which before the one and the world and a | wer | 0.3333 |
| 160 | streaming_asr | EN_B00083_S00689_W000013 | 0.1719 | 32 | You you this a using information intended the to then calculate the now in his is it body to for the apartment This the number what need keep low | wer | 0.6000 |
| 320 | streaming_asr | EN_B00083_S00689_W000013 | 0.1652 | 29 | You you this a using information intention the to then calculated the now in the using body to for the apartment This the number what need keep low | wer | 0.5667 |
| 640 | streaming_asr | EN_B00083_S00689_W000013 | 0.1585 | 28 | You you this a using information intitude the to then calculated the now in the using body to the for the apartment This the number and need keep low | wer | 0.6000 |
| 1280 | streaming_asr | EN_B00083_S00689_W000013 | 0.1362 | 29 | You you this the using the information intention the to then calculate the now in is using body to the for the apartment This the number the need keep low | wer | 0.5667 |
| 160 | streaming_asr | EN_B00083_S02942_W000001 | 0.1897 | 33 | It is going to be of restorative that then be a many course for any writer so want to learn how to break and a list are own serious Stories | wer | 0.4516 |
| 320 | streaming_asr | EN_B00083_S02942_W000001 | 0.1897 | 33 | It is going to be be re sourced that thing be a many course for any writer so want to learn how to right and a list are own serious Stories | wer | 0.4516 |
| 640 | streaming_asr | EN_B00083_S02942_W000001 | 0.1874 | 34 | It is going to be be restors that singing be a many course for any writer so want to learn how to right and a list are own Series Stories | wer | 0.4516 |
| 1280 | streaming_asr | EN_B00083_S02942_W000001 | 0.1663 | 33 | It is going to be be restores that singing being many course for any writer his want to learn had to right and huddled here owning Series Stories | wer | 0.5484 |
| 160 | streaming_asr | EN_B00089_S01559_W000004 | 0.2066 | 12 | He never <\|glm_semantic_10236\|>ishes soaks or a long journey or homework | wer | 0.7500 |
| 320 | streaming_asr | EN_B00089_S01559_W000004 | 0.1983 | 16 | And never understood Socks or Arranged or home market | wer | 0.5000 |
| 640 | streaming_asr | EN_B00089_S01559_W000004 | 0.1777 | 13 | And never understood socks or Arranged or homework | wer | 0.2500 |
| 1280 | streaming_asr | EN_B00089_S01559_W000004 | 0.1405 | 10 | And never understood Socks or Arranged or homework | wer | 0.2500 |
| 160 | streaming_asr | EN_B00089_S03348_W000001 | 0.2551 | 29 | one the things that like do when each my introduction stronomy class this to be the cost the yes strawling picture rever the to | wer | 0.5517 |
| 320 | streaming_asr | EN_B00089_S03348_W000001 | 0.2346 | 27 | one the things that like do when each my introduce stronomy class his to be the class the yes strawling picture reve the a | wer | 0.5172 |
| 640 | streaming_asr | EN_B00089_S03348_W000001 | 0.2375 | 27 | one the things that like do when each my introduce stronomy class his to be the class the you strawling picture rever the they | wer | 0.5172 |
| 1280 | streaming_asr | EN_B00089_S03348_W000001 | 0.1760 | 26 | one the things that like do when each my introduction stronomy classes his to be then class the the strawling picture of the to | wer | 0.4483 |
| 160 | streaming_asr | EN_B00091_S07092_W000002 | 0.2258 | 19 | And This come to of what like to current on old of they and your in the and of sorry, turn. sigger | wer | 1.0526 |
| 320 | streaming_asr | EN_B00091_S07092_W000002 | 0.2500 | 21 | But the concern of more like to care on a of down there and your in the and of sorry, turn around. figure | wer | 0.9474 |
| 640 | streaming_asr | EN_B00091_S07092_W000002 | 0.2016 | 20 | But to comes of more like to care on a of down there and your in the and of thorothy returned. figure | wer | 0.8421 |
| 1280 | streaming_asr | EN_B00091_S07092_W000002 | 0.1976 | 20 | But to comes of more like a care on a of down there and the in the and of thoritary figure | wer | 0.7368 |
| 160 | streaming_asr | EN_B00091_S08343_W000001 | 0.1361 | 16 | Not and mommy came to serveys my at shiveness the gravations seremony | wer | 0.7500 |
| 320 | streaming_asr | EN_B00091_S08343_W000001 | 0.1492 | 20 | Both and mommy came to serveys my at shiveness the gravations severnity | wer | 0.7500 |
| 640 | streaming_asr | EN_B00091_S08343_W000001 | 0.1649 | 18 | But and moment came to serveys my at sh improvements The gravations severnese | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00091_S08343_W000001 | 0.1545 | 17 | But and Moments came to severage my at sh improvements The gravations thermony | wer | 0.7500 |
| 160 | streaming_asr | EN_B00097_S02875_W000006 | 0.2553 | 47 | But That for she you cross you like because we dong want who receives from the lord a of of body We one or seieve Anything also do in was one seieve the a but but to all about | wer | 0.7179 |
| 320 | streaming_asr | EN_B00097_S02875_W000006 | 0.2979 | 53 | But That for this you cross you like because we don't want who recieve from the lord a of of body We one or seieve Anything all to do in was one severe and a but but of or about | wer | 0.6923 |
| 640 | streaming_asr | EN_B00097_S02875_W000006 | 0.2819 | 49 | But the for this you grows you like because we don't want who recieve from the lord but of of body We one or seieve Anything all the do in with one seieve and a but but of all about | wer | 0.7179 |
| 1280 | streaming_asr | EN_B00097_S02875_W000006 | 0.2535 | 52 | But the for she you cross you like because we don't want who recieve from the lord but of of body We one or seieve Anything all the do in was one rescue and a up but of or about | wer | 0.6923 |
| 160 | streaming_asr | EN_B00097_S03849_W000000 | 0.1925 | 32 | You you and defe don't and lined Then though the American boisage democracy And Capitalistic Civilization of was enamis live and progressed | wer | 0.6296 |
| 320 | streaming_asr | EN_B00097_S03849_W000000 | 0.1863 | 34 | He for now defe dom and line Then though the American boisage democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6667 |
| 640 | streaming_asr | EN_B00097_S03849_W000000 | 0.1718 | 31 | Here for now defe don't and line Then no the American boys. democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00097_S03849_W000000 | 0.1429 | 31 | It for not defe dom and line then no the American boys. democracy And Capitalistic civilization of was enormous life and progressed | wer | 0.6296 |
| 160 | streaming_asr | HQ-Conversations_0000026067 | 0.2685 | 13 | 的还有很多零辜负的我这样的就是在那种打像的用 | cer | 0.5600 |
| 320 | streaming_asr | HQ-Conversations_0000026067 | 0.2824 | 16 | 的还很多灯笼鼓鼓的我这样就是在那种打现在的用 | cer | 0.6000 |
| 640 | streaming_asr | HQ-Conversations_0000026067 | 0.2963 | 15 | 的还很多电动鼓舞的我这样就是在那种大下来了的用 | cer | 0.6000 |
| 1280 | streaming_asr | HQ-Conversations_0000026067 | 0.3102 | 16 | 的还很多landlord辜负的我这样就是在那种大下来的用 | cer | 0.8000 |
| 160 | streaming_asr | HQ-Conversations_0000028308 | 0.1613 | 6 | 我兄弟咋了？ | cer | 0.6000 |
| 320 | streaming_asr | HQ-Conversations_0000028308 | 0.2097 | 7 | 我行了咋了 | cer | 0.8000 |
| 640 | streaming_asr | HQ-Conversations_0000028308 | 0.2258 | 5 | 我行了咋了 | cer | 0.8000 |
| 1280 | streaming_asr | HQ-Conversations_0000028308 | 0.1935 | 6 | 我行。咋了？ | cer | 1.0000 |
| 160 | streaming_asr | HQ-Conversations_0000041144 | 0.2094 | 20 | 对和谢谢人处这个极端什么其实我也有点这种焦虑 | cer | 0.4348 |
| 320 | streaming_asr | HQ-Conversations_0000041144 | 0.1986 | 20 | 对后下身处这个阶段嘛其实我也有点这种焦虑 | cer | 0.2174 |
| 640 | streaming_asr | HQ-Conversations_0000041144 | 0.1769 | 20 | 对后下人处这个阶段嘛其实我也有点这种焦虑 | cer | 0.2609 |
| 1280 | streaming_asr | HQ-Conversations_0000041144 | 0.1913 | 18 | 对和下身处这个阶段嘛其实我这有点这种焦虑 | cer | 0.3043 |
| 160 | streaming_asr | LibriSpeech_0000033920 | 0.1601 | 41 | And and representative who of put for everything they grass of line and over one to panicked over myth a national pressed. the space sever turn over to the craft industry | wer | 0.5833 |
| 320 | streaming_asr | LibriSpeech_0000033920 | 0.1469 | 40 | And in the representative who of put for everything they grass of find and over one to panicked over myth a national pressed the space sever with turn over to the craft industry | wer | 0.5556 |
| 640 | streaming_asr | LibriSpeech_0000033920 | 0.1535 | 41 | And in the representative who a put for everything they grass of I and over one to panicked over myth a national pressed. the space sever with turn over to the craft industry | wer | 0.5556 |
| 1280 | streaming_asr | LibriSpeech_0000033920 | 0.1469 | 43 | And and representative who a put for everything they grass of I and government one to penic over myth a national pressed the spaced sever with turn over to the craft industry | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000035237 | 0.1218 | 35 | In at country very very the great dear of mass it covers the ground just just grass does here But most interesting f interesting about is slamings is the wave in migrated | wer | 0.4688 |
| 320 | streaming_asr | LibriSpeech_0000035237 | 0.1269 | 34 | In at country <\|glm_semantic_2425\|> res the great dear of must it covers the ground just just rest does here But most interesting f interesting about one slamings is the wave in migrated | wer | 0.5000 |
| 640 | streaming_asr | LibriSpeech_0000035237 | 0.1168 | 34 | In at country <\|glm_semantic_2425\|> re the great dear of mass it covers the ground just just rest does here But most interesting f interesting about it slamings is the wave in my great | wer | 0.5312 |
| 1280 | streaming_asr | LibriSpeech_0000035237 | 0.1168 | 36 | In at country <\|glm_semantic_2425\|> re the great dear of mass it covers the ground just just rest does here But most interesting they about it slamming is the wave a my great | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000068252 | 0.1509 | 50 | The b Bureau of health has transformed the see have menela from me feet reinfested huck bed of contagious Zees to one of most helpful City some the love six thousand weber's have been collected | wer | 0.5000 |
| 320 | streaming_asr | LibriSpeech_0000068252 | 0.1550 | 44 | The beau of health has transformed the see have menela from me feet infested huck bed of contagious Zeses to one of most helpful City some the love six That was weber's have been collected | wer | 0.5294 |
| 640 | streaming_asr | LibriSpeech_0000068252 | 0.1469 | 45 | The Biro of health has transformed the see have menela for a feet infested huck bed of contagious Zesas to one of most helpful City some the love six That was weber's have been collected | wer | 0.5294 |
| 1280 | streaming_asr | LibriSpeech_0000068252 | 0.1442 | 44 | The Biro of health has transformed the see have menela from a feet reinfested huck bed of contagious Zeses to one of the most helpful City that the love six That was weber's have been collected | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000068284 | 0.1303 | 32 | Is not a were to facts but only of the meaning of facts It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 320 | streaming_asr | LibriSpeech_0000068284 | 0.1285 | 30 | Is not a world the facts but only of the meaning of fact It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 640 | streaming_asr | LibriSpeech_0000068284 | 0.1250 | 30 | Is not a world the facts but only of the meaning a facts It see point to view for judging facts It appertains to it Different ology | wer | 0.2143 |
| 1280 | streaming_asr | LibriSpeech_0000068284 | 0.1373 | 34 | Is not a world the facts but only of the meaning of facts It see point to view for judging facts It appertains to it Different ology | wer | 0.1786 |
| 160 | streaming_asr | LibriSpeech_0000090398 | 0.1457 | 16 | How eleven how giant will was and of what find of and | wer | 0.5000 |
| 320 | streaming_asr | LibriSpeech_0000090398 | 0.2211 | 16 | How again how giant will was and of but find to and | wer | 0.5833 |
| 640 | streaming_asr | LibriSpeech_0000090398 | 0.2161 | 18 | How elephant how giant she was and of but find of and | wer | 0.5000 |
| 1280 | streaming_asr | LibriSpeech_0000090398 | 0.1859 | 18 | How elephant how gent will was and of but find could and | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000090820 | 0.1173 | 35 | which hempt me in he house Nearly to wigs The pace kitchens see heavenly say from war the was days like to little boat no winter see the and were out and the fields all day hasking cork and and can in new | wer | 0.5625 |
| 320 | streaming_asr | LibriSpeech_0000090820 | 0.1280 | 39 | which hempt me in he of Nearly to weeks The pace kitchens see Heaven say from war the was days like to little boat no winter see the and where out and the fields all day hasking cork and and can in new | wer | 0.6042 |
| 640 | streaming_asr | LibriSpeech_0000090820 | 0.1307 | 41 | Which hipped in he of Nearly to wigs The pace kitchens see heavenly say from war the was days like to little boat in winter see the and were out and the heals all day hasking coral. and and can in noon | wer | 0.5833 |
| 1280 | streaming_asr | LibriSpeech_0000090820 | 0.1253 | 40 | Which hipped in he house Nearly to weeks The pace kitchens see heavenly say from war the was days like to little boat in winter see the and were how and the heals all day asking cork and and can in new | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000100601 | 0.1437 | 40 | In one insphyxiated jumble Well Tom more off on his three dray my at tension was attracted by man who stirred to little apart looking as if his thus was far away | wer | 0.3750 |
| 320 | streaming_asr | LibriSpeech_0000100601 | 0.1527 | 38 | In one in physiastic jumble Well Tom more off on his three rain my attention was attracted by a man who stirred to little a part looking as if he thought was father way | wer | 0.4688 |
| 640 | streaming_asr | LibriSpeech_0000100601 | 0.1347 | 34 | In one amphoziac. jumble. Well Tom more off on his three dray my attention was attracted by a man who still to the apart looking as if he thus was father away | wer | 0.4062 |
| 1280 | streaming_asr | LibriSpeech_0000100601 | 0.1302 | 36 | In one amphoziac jumble Well Tom more is on his three dray my attention was attracted by a man who still to the apart looking as if he thus was for away | wer | 0.4062 |
| 160 | streaming_asr | LibriSpeech_0000104521 | 0.1659 | 58 | And the drew up beside polis steps And age wink dressed the unform of Silver cloth came forward to sister a to light said the girl could to he personage showed was said one to master emperor | wer | 0.5500 |
| 320 | streaming_asr | LibriSpeech_0000104521 | 0.1730 | 57 | And the drew out up beside polis steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed was said one to master emperor | wer | 0.5000 |
| 640 | streaming_asr | LibriSpeech_0000104521 | 0.1671 | 62 | And the drew out up beside police steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed was said one to master emperor | wer | 0.5000 |
| 1280 | streaming_asr | LibriSpeech_0000104521 | 0.1635 | 61 | And the drew out of beside police steps And aged wink dressed the uniform of Silver cloth came forward to sister a to light said the girl to he personage showed where said one to master emperor | wer | 0.5250 |
| 160 | streaming_asr | LibriSpeech_0000124435 | 0.1714 | 52 | So Now all the of the children saw upon the replace apples sauce and scotch and to me that and sweep Potato and sour Potato not when the and could it muffled because not one was safed the meat | wer | 0.5135 |
| 320 | streaming_asr | LibriSpeech_0000124435 | 0.1714 | 53 | So Now all the of the children so upon the played apples sauce and squash and to me. and sweep Potatoes and sour Potatoes not when the and could the muffled because not one was safed with meat | wer | 0.5405 |
| 640 | streaming_asr | LibriSpeech_0000124435 | 0.1565 | 55 | So Now all the of the children so upon the played apple sauce and scotch and to me to and sweep Potatoes and sour Potato not when the and to to muffled because not one was safed with meat | wer | 0.5676 |
| 1280 | streaming_asr | LibriSpeech_0000124435 | 0.1565 | 58 | So Now all the of the children so upon the replace apple saw and squash and to me to and swiss Potatoes and sour Potato not not the and to to muffled because not one was seventh. with meat | wer | 0.5676 |
| 160 | streaming_asr | LibriSpeech_0000124551 | 0.1567 | 46 | You platter fellow as a and the salad were but to begin breakfast who and was discovered that was of it's member's was missing hendry was the absent one at first the was but the not take in of circumstance | wer | 0.5128 |
| 320 | streaming_asr | LibriSpeech_0000124551 | 0.1681 | 52 | You platter families some and sail were both to begin breakfast who and was discovered the was of it member's was missing hendry was the absent one at first the was but let not take in of circumstance | wer | 0.5385 |
| 640 | streaming_asr | LibriSpeech_0000124551 | 0.1695 | 51 | You planchers found as simple and the salad were both to begin breakfast who and was discovered the was a but member was missing hendry was the absent one I first the was but let not take in of circumstance | wer | 0.5897 |
| 1280 | streaming_asr | LibriSpeech_0000124551 | 0.1638 | 53 | The Plants found assembled and the salad were both to begin Breakfast who and was discovered that was a but member was missing hendry was the absent one I first the was but let not take in of circumstance | wer | 0.4872 |
| 160 | streaming_asr | LibriSpeech_0000158589 | 0.1288 | 38 | I I did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the please. boys of this is blap it for after receiving the affectionate grating of near the hold company | wer | 0.5556 |
| 320 | streaming_asr | LibriSpeech_0000158589 | 0.1493 | 46 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the place boys of this is blab it for after receiving the affectionate grating of near the hold company | wer | 0.5278 |
| 640 | streaming_asr | LibriSpeech_0000158589 | 0.1726 | 46 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the place voice of this is blaffe for after receiving the affectionate grating of near the hold company | wer | 0.4722 |
| 1280 | streaming_asr | LibriSpeech_0000158589 | 0.1479 | 44 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the please. voice of this is blaffe for after receiving the affectionate grating of near the hold company | wer | 0.4722 |
| 160 | streaming_asr | LibriSpeech_0000158773 | 0.1383 | 44 | I men do it blessings blessings berry a live that I redealize my capable my I a a conquer in I and when on you on security relaying my thrift the judgment and my not of world I chose is being some prefer of all there's | wer | 0.6327 |
| 320 | streaming_asr | LibriSpeech_0000158773 | 0.1383 | 48 | I men do it blessings blessings berry a believe that I redealizing my capable my I make a conquer in I and when on you on security relined my thrift the judgment and my not of world I chose is business some prefers of of the | wer | 0.6122 |
| 640 | streaming_asr | LibriSpeech_0000158773 | 0.1597 | 52 | I men joyed blessings blessings berry a live that I do you do my capable my my a little can in a I and when money on security relined my thrift the judgment and my knowledge of world I chose is business some prefers of a there's | wer | 0.6327 |
| 1280 | streaming_asr | LibriSpeech_0000158773 | 0.1289 | 44 | I men joyed blessings of every a believe that I you do like my capable my I make little conquer in I again when money on security relined my thrift the judgment and my knowledge of world I chose is business some prefers of of the | wer | 0.5918 |
| 160 | streaming_asr | LibriSpeech_0000168081 | 0.2588 | 74 | Business getting gawkins disference by baw saw directors considerations of corporator Policy all of which infl the Political market in economist mat some world a use the 结果 careful They formal connosis | wer | 0.5588 |
| 320 | streaming_asr | LibriSpeech_0000168081 | 0.2576 | 72 | Business getting g arguments dissession by baw saw rettest considerations of corporal Policy all of which infl the Political market in economist mat of world a you the results careful They formal connosis | wer | 0.5294 |
| 640 | streaming_asr | LibriSpeech_0000168081 | 0.2336 | 72 | Business getting g arguments dissections by board saw Directors considerations of corporal Policy all of it influenced the Political market in economist met of well a used the results careful They formal connosis | wer | 0.5588 |
| 1280 | streaming_asr | LibriSpeech_0000168081 | 0.2361 | 73 | Business getting g arguments dissections by board saw directors considerations of corporal Policy all of it influence the Political market in economist met some world a used the results careful They formal connosis | wer | 0.5294 |
| 160 | streaming_asr | LibriSpeech_0000215121 | 0.1652 | 45 | And to but you to me which your normsman of or I immediate counted him over of for to ban nuts one christ not the the his head the to no fest some But that is not all continued don't | wer | 0.6111 |
| 320 | streaming_asr | LibriSpeech_0000215121 | 0.1681 | 44 | And turn but you to me which your dormant of or I immed it counted can't over of forty about nuts on christ not the the his head the to no fest some But that is that all continued blind. | wer | 0.6389 |
| 640 | streaming_asr | LibriSpeech_0000215121 | 0.1768 | 46 | And turn but you to me which your dormancy of or I immediate counted can over of forty ban not on christ not the the his head the to no some But that is not all continue blind. | wer | 0.5833 |
| 1280 | streaming_asr | LibriSpeech_0000215121 | 0.1739 | 47 | And turn but you to me which your dormancy off or I me to recounted can over of forty ban not on christ not the the his head the to no some But that is not all continue blind. | wer | 0.6667 |
| 160 | streaming_asr | LibriSpeech_0000238719 | 0.1829 | 28 | Shall tray is the well to don't one if this kind of if thing be <\|glm_semantic_15509\|>to be permitted I may be going to to no old to opper to night | wer | 0.4643 |
| 320 | streaming_asr | LibriSpeech_0000238719 | 0.2043 | 30 | Just tray it the well to don't on if this kind of of thing and permitted I may be going to in a old to opper to night | wer | 0.4643 |
| 640 | streaming_asr | LibriSpeech_0000238719 | 0.1829 | 31 | How prey is the well to don't on if they kind of of thing and permitted I my be gilling to to in a old to the opper to night | wer | 0.4643 |
| 1280 | streaming_asr | LibriSpeech_0000238719 | 0.1686 | 32 | How prey is the well to don't one if the kind of of thing be <\|glm_semantic_13382\|>itted I me be gilling to to in a old to the opper to night | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000265196 | 0.1239 | 34 | I she be real great it you say nothing about this there or some in the had house and neighbourhood who son not there's is you state here and the you you out for kind go to bed read here or ex | wer | 0.4667 |
| 320 | streaming_asr | LibriSpeech_0000265196 | 0.1531 | 39 | I she by real great a you say nothing about this They or some in the had how and neighbourhood who son not fancy dies you state here and the you you out for kind go to bed read here are ex | wer | 0.5333 |
| 640 | streaming_asr | LibriSpeech_0000265196 | 0.1676 | 45 | I show by real great if you say nothing. about this They or some in the had how and neighbourhood who sincerely not fancy dies you state here and the you you a for kind of go to bed read here are ex | wer | 0.5333 |
| 1280 | streaming_asr | LibriSpeech_0000265196 | 0.1458 | 48 | I she by real great a you say nothing about this they or some in the had how and neighbourhood who sincerely in a fancy dies you state here and the you you a for kind of go to bed read here are books | wer | 0.5111 |
| 160 | streaming_asr | LibriSpeech_0000271006 | 0.1502 | 31 | And a made in the temp to at on you for for could sure the that when never get more the in enough to can my travelling expense I thanked and for his survives to | wer | 0.6562 |
| 320 | streaming_asr | LibriSpeech_0000271006 | 0.1502 | 35 | And a make can tempt to it money for for could sure the that where no get more the in enough to can my travelling expense I thanked im for he survives to | wer | 0.5938 |
| 640 | streaming_asr | LibriSpeech_0000271006 | 0.1319 | 33 | And not make can tempt to it money for you could sure the like where no get more the in enough to can my travelling expense I thanked im for he survives to | wer | 0.5625 |
| 1280 | streaming_asr | LibriSpeech_0000271006 | 0.1099 | 26 | And not make can you tempt to at money for for could sure the that where no get more the in enough to can my travelling expense I thanked and for he advise to | wer | 0.5938 |
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2511 | 18 | It's two laid now let just yeah this over way is | wer | 0.6000 |
| 320 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2555 | 17 | is to laid now let just a this over way is | wer | 0.7000 |
| 640 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2159 | 20 | is two late now the just a this over way is | wer | 0.6000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.1982 | 19 | is two laid and now that just a this over way is | wer | 0.8000 |
| 160 | streaming_asr | VCTK_0000006134 | 0.1294 | 11 | This influence the the dinned in that can air | wer | 0.6667 |
| 320 | streaming_asr | VCTK_0000006134 | 0.1471 | 12 | The influence the a ins in that can air | wer | 0.8889 |
| 640 | streaming_asr | VCTK_0000006134 | 0.1529 | 12 | They influence to the to dinned and that can arrow | wer | 0.6667 |
| 1280 | streaming_asr | VCTK_0000006134 | 0.1588 | 11 | This influence to the to int and that can arrow | wer | 0.5556 |
| 160 | streaming_asr | VCTK_0000029186 | 0.0861 | 6 | It so and being new challenged | wer | 0.8571 |
| 320 | streaming_asr | VCTK_0000029186 | 0.1126 | 10 | It so to being new challenged | wer | 0.7143 |
| 640 | streaming_asr | VCTK_0000029186 | 0.1126 | 11 | It could to being new challenged | wer | 0.7143 |
| 1280 | streaming_asr | VCTK_0000029186 | 0.1126 | 10 | It could to be new challenged | wer | 0.5714 |
| 160 | streaming_asr | VCTK_0000029362 | 0.1000 | 8 | I about is exhausted | wer | 0.5000 |
| 320 | streaming_asr | VCTK_0000029362 | 0.1313 | 8 | I about is exhausted | wer | 0.5000 |
| 640 | streaming_asr | VCTK_0000029362 | 0.1125 | 8 | I body is sustained | wer | 0.5000 |
| 1280 | streaming_asr | VCTK_0000029362 | 0.1125 | 8 | I body is exhausted | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0003918097 | 0.1226 | 28 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中丧来说的一个热点的就是就是甚至你们应该 | cer | 0.2553 |
| 320 | streaming_asr | emilia_zh_0003918097 | 0.1459 | 30 | 啊是然后前两天那这是就是他们这个话题成为一个大家中中三来说的一个热点的啊就是就是就是你们应该 | cer | 0.2979 |
| 640 | streaming_asr | emilia_zh_0003918097 | 0.1480 | 30 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中餐来说的一个热点啊就是就是其实你们应该 | cer | 0.2128 |
| 1280 | streaming_asr | emilia_zh_0003918097 | 0.1374 | 26 | 啊是然后前两天那就是就是他们这个话题成为一个大家中中餐来说的一个热点啊就是就是其实你们应该 | cer | 0.2128 |
| 160 | streaming_asr | emilia_zh_0003918326 | 0.2115 | 21 | 啊我我什么还有一个点听意思这是这种现在你人在<\|write_generate\|><\|cmn\|><\|start_content\|>这些人他们点互相关联 | cer | 1.5278 |
| 320 | streaming_asr | emilia_zh_0003918326 | 0.1987 | 20 | 啊我我什么呢还有一个点听意思这是这个现在了人在。这些人他们眼互相关联 | cer | 0.4444 |
| 640 | streaming_asr | emilia_zh_0003918326 | 0.2212 | 21 | 啊我我什么呢还有一个变听意思这是这个是了人在就这些人他们眼互相关联 | cer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0003918326 | 0.2276 | 23 | 啊我我什么呢还有一个变听意思这是这段是人人起来<\|write_generate\|><\|cmn\|><\|start_content\|>这些人他们眼互相关联 | cer | 1.5556 |
| 160 | streaming_asr | emilia_zh_0004002573 | 0.2196 | 32 | 啊哎呦哎呀因为他这这人人喜欢他你的内容然后家的的词然后你换这种那种拍他可能不喜欢这个内容他的会取消 | cer | 0.4333 |
| 320 | streaming_asr | emilia_zh_0004002573 | 0.2434 | 34 | 啊哎呦我要因为他有现在人能喜欢看你的内容然后家的的词然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4500 |
| 640 | streaming_asr | emilia_zh_0004002573 | 0.2649 | 35 | 啊哎呦哎呀因为他有现在人能喜欢看你这个内容然后家的的死然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4500 |
| 1280 | streaming_asr | emilia_zh_0004002573 | 0.2697 | 35 | 啊哎呦哎呀因为他有现在人能喜欢他你这个内容然后家的的词然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4667 |
| 160 | streaming_asr | emilia_zh_0004003103 | 0.2185 | 54 | 所以我就的这些可能父母个有的影响对他会影响我但是吵吵没有没有一套观念说我是必须得吃反抗什么都去去争取我要行的我窗外是觉得说我可以<\|write_generate\|><\|cmn\|><\|start_content\|>顺其自然自然而然是得到想要懂<\|write_generate\|><\|cmn\|><\|start_content\|>你就 | cer | 1.2184 |
| 320 | streaming_asr | emilia_zh_0004003103 | 0.2328 | 57 | 所以我就的之前可能父母个有的影响对他会影响但是吵吵没有没有一套观念说我是必须得是反抗什么都去去争取我要性的我从而是觉得说我可以可以顺其自然自然是得到想要等<\|write_generate\|><\|cmn\|><\|start_content\|>你就 | cer | 0.8046 |
| 640 | streaming_asr | emilia_zh_0004003103 | 0.2328 | 60 | 所以我就都这些可能父母我我的影响对他会影响但是吵了没有没有一套观念说我是必须得是反抗什么都去争取我要等行的我从是觉得说可以可以顺其自然自然是得到想要懂<\|write_generate\|><\|cmn\|><\|start_content\|>嗯 | cer | 0.7586 |
| 1280 | streaming_asr | emilia_zh_0004003103 | 0.2107 | 60 | 所以我的这些可能父母我我的影响对他会影响但是我没有没有一套观念说我是必须得是反抗什么都去争取我要的行的我从是觉得说我可以可以顺其自然自然是得到想要的<\|write_generate\|><\|cmn\|><\|start_content\|>嗯 | cer | 0.7126 |
| 160 | streaming_asr | emilia_zh_0004036114 | 0.1505 | 12 | 没有上过大学家里情况可以说是一言难尽。 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0004036114 | 0.1720 | 11 | 没有上过大学家里情况一个说是一言难尽。 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004036114 | 0.1613 | 11 | 没有上过大学家里情况一个说是一言难尽。 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004036114 | 0.1667 | 11 | 没有上国大学家里情况一个说是一言难尽。 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004064583 | 0.2181 | 24 | 我相信没因巧合都是一道自信一线索告诉男 | cer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0004064583 | 0.2150 | 22 | 我相信没一些巧合都是一道资讯一线索告诉男人 | cer | 0.2273 |
| 640 | streaming_asr | emilia_zh_0004064583 | 0.2150 | 24 | 我相信没一些巧合都是一道自信疑线索告诉难 | cer | 0.3636 |
| 1280 | streaming_asr | emilia_zh_0004064583 | 0.1963 | 23 | 我相信没一次巧合都是一道自信一现告诉难 | cer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0004064952 | 0.1744 | 10 | 也在国了下因为在我的脑子里善材料的光 | cer | 0.4737 |
| 320 | streaming_asr | emilia_zh_0004064952 | 0.1590 | 9 | 也在国了想依然在我的脑子里善贼亮的光 | cer | 0.3158 |
| 640 | streaming_asr | emilia_zh_0004064952 | 0.1795 | 10 | 也国乐想因为在我的脑子里善贼亮的光 | cer | 0.4737 |
| 1280 | streaming_asr | emilia_zh_0004064952 | 0.1692 | 10 | 也国了想因为在我的脑子里善贼亮的光 | cer | 0.4737 |
| 160 | streaming_asr | emilia_zh_0004111317 | 0.2418 | 54 | 我我主意我们为什么部挖坑潮水啊哦是的的我相信如果我们啊的足够神我们可以到水的东西这我们权责一个这一点呢他开始我办法 | cer | 0.3929 |
| 320 | streaming_asr | emilia_zh_0004111317 | 0.2467 | 58 | 我我主意我们为什么部挖坑找水啊哦是的的我相信如果我们啊的足够神我们可以到水的对这我们选择一个这一点看开始我爸爸 | cer | 0.3036 |
| 640 | streaming_asr | emilia_zh_0004111317 | 0.2503 | 63 | 我我主意我们为什么部挖坑潮水啊哦湿润的的我相信如果我们啊的足够神我们可以到水的堆热我们选择一个这一点那个开始我爸爸 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0004111317 | 0.2588 | 61 | 我有个主意我们为什么部挖坑找水呢哦湿润的的我相信如果我们啊的足够神我们可以到水的堆热我们选择一个这一点那个开始的爸爸 | cer | 0.3036 |
| 160 | streaming_asr | emilia_zh_0004111570 | 0.1850 | 28 | 我的意思是没有啊地方或者维度是我们被永远因果其中的要那样的地方来干什么呢 | cer | 0.1081 |
| 320 | streaming_asr | emilia_zh_0004111570 | 0.1803 | 23 | 我的意思是没有啊地方或者维度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1351 |
| 640 | streaming_asr | emilia_zh_0004111570 | 0.1780 | 22 | 我的意思是没有啊地方或者维度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1351 |
| 1280 | streaming_asr | emilia_zh_0004111570 | 0.1780 | 24 | 我的意思是没有啊的地方或者度是我们被永远巩固其中的要那样的地方来干什么那 | cer | 0.1622 |
| 160 | streaming_asr | emilia_zh_0004129851 | 0.1852 | 8 | 之间我两个人做到地方喝酒能 | cer | 0.5714 |
| 320 | streaming_asr | emilia_zh_0004129851 | 0.2037 | 12 | 之间都两个人做到的地方喝酒能 | cer | 0.5714 |
| 640 | streaming_asr | emilia_zh_0004129851 | 0.2222 | 11 | 直接都两个人做的地方喝酒能 | cer | 0.5714 |
| 1280 | streaming_asr | emilia_zh_0004129851 | 0.1852 | 12 | 之间都两个人做的地方喝酒能 | cer | 0.5714 |
| 160 | streaming_asr | emilia_zh_0004130152 | 0.2347 | 17 | 现在说英语的骗的安乐瞬间将变成痛苦例如 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004130152 | 0.2563 | 14 | 现在我拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0500 |
| 640 | streaming_asr | emilia_zh_0004130152 | 0.2238 | 14 | 现在我拥有的片刻的安乐瞬间在变成痛苦例如 | cer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004130152 | 0.2130 | 14 | 现在我拥有的片刻的安乐瞬间家变成痛苦例如 | cer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0004176058 | 0.1087 | 8 | 她传发明改变的美国 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0004176058 | 0.1565 | 12 | 新一代传发明改变的美国 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004176058 | 0.1609 | 12 | 新一代传发明改变的美国 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004176058 | 0.1565 | 11 | 新一代传发明改变的美国 | cer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0004176769 | 0.2881 | 19 | 最终我们和从尝尝尝起来找事情做 | cer | 0.3125 |
| 320 | streaming_asr | emilia_zh_0004176769 | 0.2373 | 19 | 最终我们和可能尝尝尝起来找事情做 | cer | 0.4375 |
| 640 | streaming_asr | emilia_zh_0004176769 | 0.2316 | 17 | 最终我们和能床上站起来找事情做 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004176769 | 0.2203 | 18 | 最终我们会能床上站起来找事情做 | cer | 0.1875 |
| 160 | streaming_asr | emilia_zh_0004212211 | 0.1376 | 18 | 这里暴露的显而易见的挫败感感觉源于孩儿的个人经历而不是政治信念本身 | cer | 0.1613 |
| 320 | streaming_asr | emilia_zh_0004212211 | 0.1590 | 20 | 这里表路的写一件的挫败感感觉源于卡尔的个人经历而不是政治新闻本身 | cer | 0.3226 |
| 640 | streaming_asr | emilia_zh_0004212211 | 0.1621 | 21 | 这里表率的显而易见的挫败来源于卡尔的个人经历而不是政治性面本身 | cer | 0.1613 |
| 1280 | streaming_asr | emilia_zh_0004212211 | 0.1529 | 18 | 这里表率的显而易见的挫败来源于卡尔的个人经历而不是政治性面本身 | cer | 0.1613 |
| 160 | streaming_asr | emilia_zh_0004270141 | 0.1509 | 16 | 他的小山上已经中了二十天了欲知再见注视着海面等他回来 | cer | 0.2308 |
| 320 | streaming_asr | emilia_zh_0004270141 | 0.1509 | 15 | 他的小山上已经中了二十天了欲知再见注视海面等他回了 | cer | 0.3077 |
| 640 | streaming_asr | emilia_zh_0004270141 | 0.1572 | 17 | 他的小山上已经中了二十天了欲指再见注视海面等他毁了 | cer | 0.3462 |
| 1280 | streaming_asr | emilia_zh_0004270141 | 0.1478 | 15 | 他的小山上已经。了二十天了欲知再见注视海面等他回了 | cer | 0.3077 |
| 160 | streaming_asr | emilia_zh_0004270182 | 0.2292 | 23 | 找到从前埋藏的已经很久很久的古的弹走崩掉螃蟹的老鼠的 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0004270182 | 0.2411 | 26 | 找到从前埋藏的已经很久很久的古德弹走工具螃蟹老鼠们 | cer | 0.2593 |
| 640 | streaming_asr | emilia_zh_0004270182 | 0.2649 | 27 | 找到从前埋藏的已经很久很久的古道挡走工具螃蟹老鼠们 | cer | 0.2593 |
| 1280 | streaming_asr | emilia_zh_0004270182 | 0.2500 | 27 | 找到从前埋藏的已经很久很久的古德挡走工具螃蟹老鼠们 | cer | 0.2593 |
| 160 | streaming_asr | emilia_zh_0004344705 | 0.2339 | 12 | 啊和马克正被一根粗粗树枝缠绕着缠绕着 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0004344705 | 0.2615 | 10 | 啊和mark.正被一根粗粗树枝缠绕着缠绕着 | cer | 0.6471 |
| 640 | streaming_asr | emilia_zh_0004344705 | 0.2294 | 11 | 安娜和mark正被一根粗粗树枝缠绕着缠绕着 | cer | 0.4706 |
| 1280 | streaming_asr | emilia_zh_0004344705 | 0.2477 | 14 | 安娜和mark.正被一根粗粗树枝缠绕着缠绕着 | cer | 0.5294 |
| 160 | streaming_asr | emilia_zh_0004358307 | 0.1673 | 33 | You was up to him him prove himself the here six to make the and ground of and and to music without the fate idea a how prayer were already | wer | 0.4333 |
| 320 | streaming_asr | emilia_zh_0004358307 | 0.1653 | 29 | You was up to him he prove himself the here six to make the and ground of and and to music without the face idea a how praud of were already | wer | 0.4333 |
| 640 | streaming_asr | emilia_zh_0004358307 | 0.1514 | 30 | You was up to him he prove himself of the here six to make the and road of him and he music without the face idea a how praud of were already | wer | 0.4333 |
| 1280 | streaming_asr | emilia_zh_0004358307 | 0.1315 | 33 | You was up to him he prove himself of the air six to make the and ride of him and he music without the face idea a how praud of were already | wer | 0.4333 |
| 160 | streaming_asr | emilia_zh_0004358957 | 0.1720 | 22 | 我听的家粗看几秒钟之后这才以为一张开口我跟鱼的这个 | cer | 0.5517 |
| 320 | streaming_asr | emilia_zh_0004358957 | 0.1892 | 22 | 我听的家租看几秒钟之后这才尾巴一张开口滚到鱼的这个 | cer | 0.5172 |
| 640 | streaming_asr | emilia_zh_0004358957 | 0.1646 | 23 | 我听这家租看几秒钟之后这才以为一张开口问道鱼的这个 | cer | 0.4828 |
| 1280 | streaming_asr | emilia_zh_0004358957 | 0.1646 | 21 | 我听的家租看几秒钟之后这才以为一张开口问道鱼的这个 | cer | 0.4828 |
| 160 | streaming_asr | emilia_zh_0004422761 | 0.2606 | 15 | This let many tire people to set to and hounds and Cities | wer | 0.5455 |
| 320 | streaming_asr | emilia_zh_0004422761 | 0.2500 | 14 | This let many tire people to set the and hounds and Cities | wer | 0.5455 |
| 640 | streaming_asr | emilia_zh_0004422761 | 0.1755 | 14 | This led many tire people to set to and hounds and Cities | wer | 0.4545 |
| 1280 | streaming_asr | emilia_zh_0004422761 | 0.1702 | 14 | This let many tired people to set to and hounds and Cities | wer | 0.5455 |
| 160 | streaming_asr | emilia_zh_0004472880 | 0.3022 | 14 | 那可不见得就是为了这个说打山两 | cer | 0.4737 |
| 320 | streaming_asr | emilia_zh_0004472880 | 0.2198 | 13 | 那可不见得就是为了这个的时候打山两 | cer | 0.3684 |
| 640 | streaming_asr | emilia_zh_0004472880 | 0.1978 | 11 | 那可不见得就是为了这个的时候打山两 | cer | 0.3684 |
| 1280 | streaming_asr | emilia_zh_0004472880 | 0.1758 | 9 | 那和不见得就是为了这个的时候打。山两 | cer | 0.4211 |
| 160 | streaming_asr | emilia_zh_0004519522 | 0.2671 | 10 | 这这个兄王的客人那呃是弱点来 | cer | 0.4667 |
| 320 | streaming_asr | emilia_zh_0004519522 | 0.2857 | 12 | 就这个兄王的客人那呃是昨天来 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004519522 | 0.2609 | 15 | 这个兄王的客人那呃是昨天来 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004519522 | 0.2298 | 12 | 这个姓王的客人那呃是昨天来 | cer | 0.2667 |
| 160 | streaming_asr | emilia_zh_0004519646 | 0.2212 | 14 | 深处毛茸茸的小转子摸了摸盒子里的东西 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004519646 | 0.2442 | 15 | 深处毛茸茸的小转折摸了摸盒子里的东西 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004519646 | 0.2535 | 16 | 深处毛茸茸的小转折摸了摸盒子里的东西 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004519646 | 0.2535 | 15 | 深处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004621436 | 0.2066 | 18 | But more red and asked you they good possibly come is paying Guess | wer | 0.5714 |
| 320 | streaming_asr | emilia_zh_0004621436 | 0.2254 | 20 | The mother road and asked he they could possibly come is pay Guess | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004621436 | 0.1972 | 21 | The mother road and asked he they could possibly come is pay g guests | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0004621436 | 0.1502 | 19 | The mother road and asked you they could positively come is paying g guests | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004633795 | 0.3054 | 24 | We he the I've game a position on the honor on in the flatt last time was the little 棵树 the basis | wer | 0.7500 |
| 320 | streaming_asr | emilia_zh_0004633795 | 0.3188 | 27 | We he the they game my position on the orner on the flatt last time was the little 棵树 the basis | wer | 0.7000 |
| 640 | streaming_asr | emilia_zh_0004633795 | 0.2617 | 24 | We here the they game my position on of corner on the flatt last time was the little 棵树 the basis | wer | 0.6500 |
| 1280 | streaming_asr | emilia_zh_0004633795 | 0.2450 | 25 | We here the I've game my pursuit on of honor a the slat last top with the little 棵树 the basis | wer | 0.7000 |
| 160 | streaming_asr | emilia_zh_0004659190 | 0.1583 | 24 | See the the when was close granting a he repressed when you on love you in Anything creation | wer | 0.5238 |
| 320 | streaming_asr | emilia_zh_0004659190 | 0.1847 | 24 | Seeing the the when was close granting a he repressed When you and love you in Anything creation | wer | 0.4762 |
| 640 | streaming_asr | emilia_zh_0004659190 | 0.1873 | 24 | Seeing the the when was close granting a he Microsoft When you one love you in Anything creation | wer | 0.4762 |
| 1280 | streaming_asr | emilia_zh_0004659190 | 0.1530 | 24 | Seeing the the when was close granting he requests when you one love you and Anything creation | wer | 0.4762 |
| 160 | streaming_asr | emilia_zh_0004692799 | 0.1610 | 46 | So of take in he string <\|glm_semantic_3935\|>ments Instead using the using these warms of a filled to communicate were a the train fam of was the turning the in now to procussion from that's very because of a | wer | 0.7027 |
| 320 | streaming_asr | emilia_zh_0004692799 | 0.1625 | 45 | So of you've taken you stream <\|glm_semantic_3935\|>smith Instead using the using the I these warms of a filled to communicate were a the train fam in is the turn them in now to caution some that's very because of a | wer | 0.7297 |
| 640 | streaming_asr | emilia_zh_0004692799 | 0.1641 | 48 | So of you've taken you stream extrements Instead using using I these warms of a filled to communicate were a the train fam in is the turn them in now to procussion some that's very because of a | wer | 0.7027 |
| 1280 | streaming_asr | emilia_zh_0004692799 | 0.1703 | 49 | So of you've taken you stream existence Instead using using a these warms of rotto filled to communicate were a the tried fam in is the turn them in now to caution from that's very because of a | wer | 0.7027 |
| 160 | streaming_asr | emilia_zh_0004706082 | 0.2174 | 8 | The quality have looking clever the and was | wer | 0.4444 |
| 320 | streaming_asr | emilia_zh_0004706082 | 0.2236 | 10 | But quality have looking clever or the and was | wer | 0.5556 |
| 640 | streaming_asr | emilia_zh_0004706082 | 0.1429 | 9 | The quality have looking clever the and was | wer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0004706082 | 0.1429 | 10 | The quoted have looking clever the and was | wer | 0.5556 |
| 160 | streaming_asr | emilia_zh_0004724705 | 0.2515 | 32 | And While had just ren't what who his like had member they were they cognitive but what see they pointing physical the first | wer | 0.7917 |
| 320 | streaming_asr | emilia_zh_0004724705 | 0.2602 | 34 | And Well had to ran there the we his heart like had with they were they cocking but what see they point physical the first | wer | 0.7500 |
| 640 | streaming_asr | emilia_zh_0004724705 | 0.2018 | 29 | And Well had who ren't the we his heart like had with that were they cockney but what see like point physical the for | wer | 0.7083 |
| 1280 | streaming_asr | emilia_zh_0004724705 | 0.2076 | 27 | And Well had who and the we his heart like had with that were did talking but what see like pointing physical the for | wer | 0.6667 |
| 160 | streaming_asr | emilia_zh_0004724727 | 0.1714 | 27 | is it ren't you going to you all but a the so for Today the new things is a a with a a All all a new | wer | 0.4783 |
| 320 | streaming_asr | emilia_zh_0004724727 | 0.1844 | 25 | Is it redeemant yeah all but a the some for Today the new things is a a with a All all new | wer | 0.4348 |
| 640 | streaming_asr | emilia_zh_0004724727 | 0.1974 | 26 | is it redeemable yeah how but a the some for Today the new things is a a with a All all a new | wer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0004724727 | 0.1866 | 26 | Is it raditional yeah how but a the some for Today the new things is a a with a All the a new | wer | 0.3913 |
| 160 | streaming_asr | emilia_zh_0004732647 | 0.1644 | 19 | You what did find to what made noises people people were for a And the was nothing the caves to out a | wer | 0.5417 |
| 320 | streaming_asr | emilia_zh_0004732647 | 0.1544 | 19 | You what did find to what made noises people people were fray a And the was nothing the caves to out a my | wer | 0.5833 |
| 640 | streaming_asr | emilia_zh_0004732647 | 0.1544 | 20 | You what did find to what made noises people people were for a And the was nothing the caves to had a my | wer | 0.5833 |
| 1280 | streaming_asr | emilia_zh_0004732647 | 0.1409 | 22 | He want to find to what made noises people people were for a And the was nothing the caves to to a and | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004754634 | 0.2222 | 13 | Perhaps here of take a joke suggested jack do | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0004754634 | 0.2157 | 14 | Perhaps here of can jerk suggested jack all | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0004754634 | 0.2157 | 14 | Perhaps every of take a jerk suggested jack do | wer | 0.6000 |
| 1280 | streaming_asr | emilia_zh_0004754634 | 0.1961 | 14 | Perhaps here of take a jerk suggested jack do | wer | 0.6000 |
| 160 | streaming_asr | emilia_zh_0004754844 | 0.1987 | 8 | But stop is class see and and king the is | wer | 0.8182 |
| 320 | streaming_asr | emilia_zh_0004754844 | 0.2185 | 8 | But stop is class see and and king the Earth | wer | 0.7273 |
| 640 | streaming_asr | emilia_zh_0004754844 | 0.1987 | 11 | But stop is class see and and king the Earth | wer | 0.7273 |
| 1280 | streaming_asr | emilia_zh_0004754844 | 0.1921 | 11 | But stop is class see and and king the Earth | wer | 0.7273 |
| 160 | streaming_asr | emilia_zh_0004776929 | 0.1789 | 22 | You evening They thousand down for a being families in a This was it | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0004776929 | 0.1474 | 16 | in evening They stessel down for a big families in a This was it | wer | 0.3571 |
| 640 | streaming_asr | emilia_zh_0004776929 | 0.1298 | 15 | in evening They sessed down for a big family in a This was it | wer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0004776929 | 0.1263 | 19 | In even They thessel down for of big family in a This was it | wer | 0.4286 |
| 160 | streaming_asr | emilia_zh_0004777075 | 0.1420 | 18 | You appear of dink shoulder the sunday Newspaper So what next day Tommy Tuesday | wer | 0.4375 |
| 320 | streaming_asr | emilia_zh_0004777075 | 0.1571 | 21 | You appeared of dink shoulder the sunday Newspaper So what next day comedy Tuesday | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004777075 | 0.1601 | 20 | You peer of dink shoulder the some day Newspaper So what next day Tommy Tuesday | wer | 0.5625 |
| 1280 | streaming_asr | emilia_zh_0004777075 | 0.1601 | 20 | You appeared of dink shoulder the some day Newspaper So what next day comedy Tuesday | wer | 0.6250 |
| 160 | streaming_asr | emilia_zh_0004797638 | 0.2105 | 10 | I Thank Qiao five take could not pickled chop. | wer | 0.8889 |
| 320 | streaming_asr | emilia_zh_0004797638 | 0.2158 | 10 | Hua Thank Q find take could not picly twist | wer | 0.8889 |
| 640 | streaming_asr | emilia_zh_0004797638 | 0.2368 | 10 | Fang Thank Qiao find take could not quickly chuckled | wer | 0.7778 |
| 1280 | streaming_asr | emilia_zh_0004797638 | 0.2632 | 12 | Fang Thank kills find take could not picly chop. | wer | 0.8889 |
| 160 | streaming_asr | emilia_zh_0004797649 | 0.2199 | 15 | You no told me has got would catch | wer | 0.6250 |
| 320 | streaming_asr | emilia_zh_0004797649 | 0.2147 | 15 | You no told me had got word catch | wer | 0.6250 |
| 640 | streaming_asr | emilia_zh_0004797649 | 0.1047 | 9 | You no told me has got word catch | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0004797649 | 0.1047 | 10 | You no told me has got word catch | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004804632 | 0.2184 | 14 | It this of the a are side when Several live these of some more gether | wer | 0.6429 |
| 320 | streaming_asr | emilia_zh_0004804632 | 0.2069 | 14 | It is after a are side when Several of these of the more gether | wer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0004804632 | 0.1724 | 13 | It is after a red side when Several of these of so more gether | wer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0004804632 | 0.1552 | 14 | It is upt a are side when seventh of these of so more gether | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004804709 | 0.1656 | 9 | And your unsafe for the the enter side town | wer | 0.7000 |
| 320 | streaming_asr | emilia_zh_0004804709 | 0.1722 | 7 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 640 | streaming_asr | emilia_zh_0004804709 | 0.1656 | 7 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 1280 | streaming_asr | emilia_zh_0004804709 | 0.1589 | 8 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 160 | streaming_asr | emilia_zh_0004841011 | 0.2326 | 17 | And all the she's way challenging and resurrecting He the friend count | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0004841011 | 0.3023 | 18 | And all the she's with challenging and resurrecting He the from count | wer | 0.6923 |
| 640 | streaming_asr | emilia_zh_0004841011 | 0.2279 | 16 | And all the use the with churning and resurrecting had the from count | wer | 0.6154 |
| 1280 | streaming_asr | emilia_zh_0004841011 | 0.1767 | 14 | And all the use the with churning and resurrecting had the from count | wer | 0.6154 |
| 160 | streaming_asr | emilia_zh_0004843191 | 0.1501 | 51 | And four not only European gerographer But European skolars garms all love else Knowledge began to draw maps with space left away a and They be and with the there was not perfect and the the were important things a the not no | wer | 0.5532 |
| 320 | streaming_asr | emilia_zh_0004843191 | 0.1475 | 50 | Hence four not don't European jographers But European skolars garms all love else Knowledge begun to draw maps with space left away a little and They be to with the the was not perfect in the the were important things the the not no | wer | 0.5957 |
| 640 | streaming_asr | emilia_zh_0004843191 | 0.1358 | 46 | And four not on European jographers But European skolars galmers all love else Knowledge begun to draw maps with space left away a little in They be to with the three was not perfect in the the were important things the the not no | wer | 0.5957 |
| 1280 | streaming_asr | emilia_zh_0004843191 | 0.1332 | 49 | And for not only European jographers But European skolars galmers all love else Knowledge begun to draw maps with space left away a and They be to with the three was not perfect in the the were important things the the not no | wer | 0.5745 |
| 160 | streaming_asr | emilia_zh_0004843272 | 0.1444 | 35 | You principled tended the that economic grow is supreme the or right please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even appiness of the pen don't money grew of | wer | 0.6129 |
| 320 | streaming_asr | emilia_zh_0004843272 | 0.1648 | 38 | It principled tennet the that economic growth is supreme de过的 or right please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even appiness of the pen don't con dominant grew fried | wer | 0.6129 |
| 640 | streaming_asr | emilia_zh_0004843272 | 0.1481 | 39 | It's principled tenanted the that con dominant growth is this supreme the or or please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even ap<\|glm_semantic_5487\|>per upper the pen don't con dominant grew fried | wer | 0.6452 |
| 1280 | streaming_asr | emilia_zh_0004843272 | 0.1537 | 39 | It's principled tenanted the that con dominant growth is supreme good or or please pro<\|glm_semantic_11271\|>s for the supreme good because Justice Freedom and even ap<\|glm_semantic_5487\|>per upper the pen don't money grew fried | wer | 0.5806 |
| 160 | streaming_asr | emilia_zh_0004873848 | 0.1979 | 15 | Margaret say let a with than on this open smile | wer | 0.6667 |
| 320 | streaming_asr | emilia_zh_0004873848 | 0.2240 | 13 | Margaret say let a with than on his open smile | wer | 0.6667 |
| 640 | streaming_asr | emilia_zh_0004873848 | 0.2396 | 14 | Margaret Face let a with then on a open smile | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0004873848 | 0.2135 | 14 | Margaret say let a with then on a open smile | wer | 0.6667 |
| 160 | streaming_asr | emilia_zh_0004874290 | 0.2271 | 17 | The book from miss Song ten arrive that's evening with the kind not find | wer | 0.6923 |
| 320 | streaming_asr | emilia_zh_0004874290 | 0.1992 | 18 | The book from miss song and arrives that's evening was the kind not find | wer | 0.6154 |
| 640 | streaming_asr | emilia_zh_0004874290 | 0.1873 | 18 | The book from miss <\|glm_semantic_14290\|>one arrives that's evening was the kind not find | wer | 0.5385 |
| 1280 | streaming_asr | emilia_zh_0004874290 | 0.1713 | 18 | The book from miss <\|glm_semantic_14290\|>onement arrives that's evening was the kind not find | wer | 0.5385 |
| 160 | streaming_asr | emilia_zh_0004879738 | 0.1635 | 8 | The Resignations furious Elizabeth and sunny | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004879738 | 0.1572 | 9 | The Resignations furious Elizabeth and sunny | wer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004879738 | 0.1635 | 8 | The resignations furious Elizabeth and sunny | wer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004879738 | 0.1447 | 8 | The recognitions inferiored Elizabeth and sunny | wer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0004880227 | 0.2326 | 21 | I is always member the hours I spend who the master the the house some busher | wer | 0.4375 |
| 320 | streaming_asr | emilia_zh_0004880227 | 0.2558 | 20 | I is always member the our I spent who some master the the house some busher | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004880227 | 0.2442 | 22 | I is always remember the hours I spent who the mass the the house some Thus | wer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0004880227 | 0.2132 | 18 | I it always remember the hours I spent who the master the the how s Thus | wer | 0.3750 |
| 160 | streaming_asr | emilia_zh_0004927443 | 0.1538 | 12 | 但是美国<\|write_generate\|><\|cmn\|><\|start_content\|>之所以懂悠闲还有一个更用的原因 | cer | 2.0455 |
| 320 | streaming_asr | emilia_zh_0004927443 | 0.1719 | 12 | 但是美国和之所以懂悠闲还有一个更用的原因 | cer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0004927443 | 0.1810 | 14 | 但是美国嗯之所以都悠闲还有一个嗯用的原因 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0004927443 | 0.1719 | 14 | 但是美国嗯之所以都悠闲还有一个嗯用的原因 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0004927721 | 0.1568 | 12 | 一个人只有在城府于一个能量时才能了解他啊 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0004927721 | 0.1653 | 14 | 一个人只有在臣服于于一个能量时才能了解他啊 | cer | 0.2105 |
| 640 | streaming_asr | emilia_zh_0004927721 | 0.1822 | 14 | 一个人持有在臣服于就这个能量时才能了解他啊 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0004927721 | 0.1737 | 13 | 一个人持有在臣服就这个能量时才能了解他啊 | cer | 0.2105 |
| 160 | streaming_asr | emilia_zh_0004943305 | 0.2019 | 16 | 我们国家层面涟漪历来都如此满满湿包装水啊 | cer | 0.4762 |
| 320 | streaming_asr | emilia_zh_0004943305 | 0.2081 | 15 | 我们国家层面年龄历来都如此麻木施巴增税啊 | cer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0004943305 | 0.1863 | 16 | 我们国家层面年龄历来都如此麻木施巴增税啊 | cer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0004943305 | 0.1832 | 15 | 我们国家层面年龄历来都如此满脸湿吧，脏睡。啊 | cer | 0.5238 |
| 160 | streaming_asr | emilia_zh_0004943713 | 0.1718 | 40 | 等因我是的他清出了你这这上面三的信仰可以成尝试问从业者黑洞租户他们你的情感的鱼场 | cer | 0.5116 |
| 320 | streaming_asr | emilia_zh_0004943713 | 0.2177 | 43 | 但是因我做到他清楚了你这这上面三的信仰可成尝试问从业者奋斗活跃。他们那个情感的愚蠢 | cer | 0.4186 |
| 640 | streaming_asr | emilia_zh_0004943713 | 0.2109 | 44 | 但是因我知道他清楚了你在在上面三的信仰可成尝试问从业者奋斗维护啊你的情感的愚蠢 | cer | 0.3023 |
| 1280 | streaming_asr | emilia_zh_0004943713 | 0.1973 | 42 | 但是是英我知道他清楚了你就这上面三对的信仰可以成尝试温从业者奋斗维护他你的情感的愚蠢 | cer | 0.3023 |
| 160 | streaming_asr | emilia_zh_0004999877 | 0.2551 | 34 | 我们就应该尽量一招所所讲的方法去实施这样才会有进步和收下啊 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0004999877 | 0.2247 | 29 | 我们就应该尽量一招佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004999877 | 0.2273 | 26 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0004999877 | 0.2172 | 24 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.0714 |
| 160 | streaming_asr | emilia_zh_0005000497 | 0.2357 | 23 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005000497 | 0.2500 | 23 | 街区企业所有的债务之后所得的的就是谢价值 | cer | 0.2857 |
| 640 | streaming_asr | emilia_zh_0005000497 | 0.2357 | 22 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005000497 | 0.2393 | 22 | 减去企业所有的债务之后所得的的就是谢价值 | cer | 0.1905 |
| 160 | streaming_asr | emilia_zh_0005059483 | 0.2941 | 25 | 故事的目的如果仅仅说了获取更大的他不是也就不成为不是在这对普通语言跟 | cer | 0.4571 |
| 320 | streaming_asr | emilia_zh_0005059483 | 0.2941 | 26 | 不是的目的如果仅仅说了获取更大的他不也就不成为不是在这对普通语言跟 | cer | 0.4571 |
| 640 | streaming_asr | emilia_zh_0005059483 | 0.2674 | 29 | 不是的目的如仅仅说了获取更大的他不也就不成为不是在这对普通语言和 | cer | 0.4857 |
| 1280 | streaming_asr | emilia_zh_0005059483 | 0.2513 | 28 | 不是的目的如仅仅说了获取更大看来不是也就不成为不是在这对普通语言和 | cer | 0.5143 |
| 160 | streaming_asr | emilia_zh_0005094293 | 0.2322 | 16 | 对于托德过程来说已经是一很难做到的事情 | cer | 0.2857 |
| 320 | streaming_asr | emilia_zh_0005094293 | 0.2607 | 17 | 对于错工程学来说已经是以及很难做到的事情 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005094293 | 0.2749 | 16 | 都脱工程学来说已经是以及很难做到的事情的 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0005094293 | 0.2701 | 17 | 都吐工程学来说已经是以及很难做到的事情的 | cer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0005094550 | 0.2611 | 36 | 他需要运用充满矛盾含糊不清概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1707 |
| 320 | streaming_asr | emilia_zh_0005094550 | 0.2611 | 37 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1463 |
| 640 | streaming_asr | emilia_zh_0005094550 | 0.2522 | 35 | 他需要应用购买矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.2195 |
| 1280 | streaming_asr | emilia_zh_0005094550 | 0.2456 | 36 | 他需要运用购买矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1951 |
| 160 | streaming_asr | emilia_zh_0005181378 | 0.1932 | 11 | 不是客气你阶段我的基本上其实你带回去 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0005181378 | 0.2443 | 14 | 不是客气你记得我的基本上其实你带回去 | cer | 0.3889 |
| 640 | streaming_asr | emilia_zh_0005181378 | 0.2273 | 14 | 不是客气你记得我的基本上说其实你带回去 | cer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0005181378 | 0.2386 | 14 | 不是客气你姐姐我的基本上其实你带回去 | cer | 0.3889 |
| 160 | streaming_asr | emilia_zh_0005244297 | 0.2857 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 320 | streaming_asr | emilia_zh_0005244297 | 0.2762 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 640 | streaming_asr | emilia_zh_0005244297 | 0.3048 | 19 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 1280 | streaming_asr | emilia_zh_0005244297 | 0.3000 | 20 | 总得做点不一样的事情把打电话报警还报出了车牌号 | cer | 0.0435 |
| 160 | streaming_asr | emilia_zh_0005313494 | 0.1962 | 10 | 对不能都乐来吧应该后面没有一个应该平 | cer | 0.5500 |
| 320 | streaming_asr | emilia_zh_0005313494 | 0.2025 | 11 | 不能都了来吧一个后面没有一个应该拼 | cer | 0.4500 |
| 640 | streaming_asr | emilia_zh_0005313494 | 0.2152 | 11 | 他就不能读了来吧一个后面没有一个应该平 | cer | 0.3500 |
| 1280 | streaming_asr | emilia_zh_0005313494 | 0.2215 | 11 | 他就不能读了来吧一个后面没有一个应该拼 | cer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0005347772 | 0.2700 | 17 | 另外民主党一部走了此处据民主党来说他们面临就困难的选择 | cer | 0.3000 |
| 320 | streaming_asr | emilia_zh_0005347772 | 0.2700 | 20 | 另外民主党一波走了此处据民主党来说他们面临这困难的选择 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0005347772 | 0.2852 | 20 | 令民主党一拨走了死胡同据民主党来说他们面临这困难的选择 | cer | 0.2667 |
| 1280 | streaming_asr | emilia_zh_0005347772 | 0.2510 | 17 | 零民主党一拨走了死胡同据民主党来说他们面临这困难的选择 | cer | 0.2667 |
| 160 | streaming_asr | emilia_zh_0005370483 | 0.1876 | 23 | 录有的就这就就是是是是一个什么感觉就是愚弄和他但错他不的啊 | cer | 0.6098 |
| 320 | streaming_asr | emilia_zh_0005370483 | 0.2174 | 27 | 读说的就就就就是是是是一个什么感觉就是鱼龙混的他但错好的不的啊 | cer | 0.4878 |
| 640 | streaming_asr | emilia_zh_0005370483 | 0.1991 | 28 | 读有的就就就就是是是是一个什么感觉就是鱼龙混的他天错他的不的。 | cer | 0.5122 |
| 1280 | streaming_asr | emilia_zh_0005370483 | 0.2174 | 28 | 读有的就就就就是是是是一个什么感觉就是鱼龙混的他但错他的不的好 | cer | 0.4878 |
| 160 | streaming_asr | emilia_zh_0005370632 | 0.1574 | 11 | 你是看不到错你这形态展发展发展与 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005370632 | 0.1617 | 10 | 你是看不到做你这形态展发展发展与 | cer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0005370632 | 0.1660 | 12 | 你是看不到做你这形态展发展发展与 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0005370632 | 0.1574 | 11 | 你是看不到说你这形态展发展发展与 | cer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0005420605 | 0.1633 | 10 | 就刚查是考一这这大一遍过太行的 | cer | 0.6471 |
| 320 | streaming_asr | emilia_zh_0005420605 | 0.1429 | 10 | 有的港口是靠一这这的一遍过太行的 | cer | 0.4706 |
| 640 | streaming_asr | emilia_zh_0005420605 | 0.1531 | 9 | 有的光卡是靠一这这的一遍过太行的 | cer | 0.4118 |
| 1280 | streaming_asr | emilia_zh_0005420605 | 0.1684 | 12 | 有的光卡纸靠一这这到一遍过太行的 | cer | 0.4118 |
| 160 | streaming_asr | emilia_zh_0005421693 | 0.4026 | 16 | 这不见东西是他昨晚制作的交通工具 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0005421693 | 0.3701 | 17 | 这不见东西是他昨晚制作交通工具 | cer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0005421693 | 0.3506 | 19 | 这不见东西是他昨晚制作交通工具 | cer | 0.1875 |
| 1280 | streaming_asr | emilia_zh_0005421693 | 0.3571 | 20 | 这不见东西是他我我制作交通工具 | cer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0005507035 | 0.2086 | 13 | 那么政府还会收回一个决定了的都跟心 | cer | 0.3810 |
| 320 | streaming_asr | emilia_zh_0005507035 | 0.2246 | 12 | 那么本政府还会收回一个决定了大都跟相信 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005507035 | 0.2353 | 14 | 那么本政府还会收回一个决定了大的的安心 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005507035 | 0.2299 | 15 | 那么日本政府还会收回一个决定了大都。安心 | cer | 0.2381 |
| 160 | streaming_asr | emilia_zh_0005507553 | 0.2139 | 13 | William you walk it me the water garden he that softly | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005507553 | 0.2086 | 14 | Will you walk it me the water garden he that softly | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0005507553 | 0.2032 | 13 | Will you walk it me the water garden he that softly | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005507553 | 0.2086 | 13 | Will you walk it me the what garden he that softly | wer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0005578304 | 0.1507 | 12 | 对素也回来就好奇的问 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0005578304 | 0.1507 | 12 | 对数年回来就好奇的问 | cer | 0.4167 |
| 640 | streaming_asr | emilia_zh_0005578304 | 0.1945 | 15 | 对所以回来就孩子的问 | cer | 0.5833 |
| 1280 | streaming_asr | emilia_zh_0005578304 | 0.1726 | 18 | 道数年回来就孩子的问 | cer | 0.5833 |
| 160 | streaming_asr | emilia_zh_0005578734 | 0.1135 | 8 | 给为一些你你能给陪伴给也发能给爱给爱 | cer | 0.2105 |
| 320 | streaming_asr | emilia_zh_0005578734 | 0.1081 | 10 | 就给为一些你你能给陪伴给办法能给爱给爱 | cer | 0.2105 |
| 640 | streaming_asr | emilia_zh_0005578734 | 0.1027 | 10 | 就给为一些你你能给陪伴给办法能给爱给爱 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0005578734 | 0.1081 | 11 | 就给有一些你你能给陪伴给吧能给啊给爱 | cer | 0.2632 |
| 160 | streaming_asr | emilia_zh_0005601476 | 0.3030 | 16 | 但是情绪中不如把心情找我 | cer | 0.4167 |
| 320 | streaming_asr | emilia_zh_0005601476 | 0.3273 | 16 | 外情绪中不如把心情找我 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005601476 | 0.3333 | 18 | 外情绪中不如把心情叫我 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005601476 | 0.3273 | 17 | 外情绪中不如把心情叫我 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0005669352 | 0.1815 | 30 | 是听啥的意思两宝红朗读都抖动的假帮助着责任路了一地哥哥趴下的东遮的例子一但是骗货 | cer | 0.5366 |
| 320 | streaming_asr | emilia_zh_0005669352 | 0.1835 | 37 | 是听沙的意思两宝红朗斗都抖动的一下帮助责任路了一定哥哥爬的东哥例子一暗示骗货 | cer | 0.5122 |
| 640 | streaming_asr | emilia_zh_0005669352 | 0.1915 | 38 | 是听沙的意思两宝红朗斗都抖动的一家帮助词儿路了一定哥哥爬的东哥例子一暗示骗货 | cer | 0.5122 |
| 1280 | streaming_asr | emilia_zh_0005669352 | 0.2056 | 36 | 是听沙的意思两朵红朗斗都抖动的一家帮助词儿路了一定哥哥爬的东哥的利益一暗示骗货 | cer | 0.4878 |
| 160 | streaming_asr | emilia_zh_0005670671 | 0.1906 | 25 | 时至今日切在人们社会的各主中所办的角度意见无代替其的生 | cer | 0.3939 |
| 320 | streaming_asr | emilia_zh_0005670671 | 0.1931 | 24 | 时至今日切在人们社会的各主中所败的角度已经无代替企业的生 | cer | 0.3636 |
| 640 | streaming_asr | emilia_zh_0005670671 | 0.2104 | 23 | 时至今日切在人们社会的各组主中所败的角度已经无代替企业的生 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005670671 | 0.2203 | 25 | 时至今日企业在人们社会的各组织中所败的角度一节无代替企业的生 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0005748506 | 0.2302 | 40 | 是说后来我觉得我理解这一个他其实告诉你你要想那么多就是你我一个呃想法的时候你去做就好我你先去眼睛 | cer | 0.2157 |
| 320 | streaming_asr | emilia_zh_0005748506 | 0.2199 | 43 | 这受后来我觉得我理解谁一个他其实告诉你你要想那么多就是你我一个呃想法的时候去做就好我你先去这样 | cer | 0.2745 |
| 640 | streaming_asr | emilia_zh_0005748506 | 0.2062 | 42 | 这说后来我觉得我理解这一个他其实告诉你你要想那么多就是你我一个呃想法的时候去做就好我你先去这样 | cer | 0.2549 |
| 1280 | streaming_asr | emilia_zh_0005748506 | 0.2045 | 39 | 这说后来我觉得我理解这一个他其实告诉你你要想什么多就是你就一个呃想法的时候去做就好我你先去这样 | cer | 0.2745 |
| 160 | streaming_asr | emilia_zh_0005749601 | 0.2584 | 32 | 是说刚才说的那个五点啊就五点配套的在稍微把他总结一下滑第一点就是认知蒙蒙不足 | cer | 0.2955 |
| 320 | streaming_asr | emilia_zh_0005749601 | 0.2713 | 31 | 就什么刚才说的那个无底啊就无点配套的在稍微把他东西一下滑低点优势认知懵不足 | cer | 0.5227 |
| 640 | streaming_asr | emilia_zh_0005749601 | 0.2765 | 32 | 就什么刚才说的那个无底啊就五点配套的在稍微把他东西一下花低点优势认知懵不足 | cer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0005749601 | 0.2894 | 32 | 就什么刚才是的那个无底啊就五点配套的在稍微把他东西一下花第一点就是认知功能不足 | cer | 0.3864 |
| 160 | streaming_asr | emilia_zh_0005781428 | 0.1728 | 12 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005781428 | 0.1885 | 13 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005781428 | 0.1780 | 10 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005781428 | 0.1675 | 10 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005818033 | 0.1287 | 10 | 我一个那个造成了啊这个他们这边 | cer | 0.5333 |
| 320 | streaming_asr | emilia_zh_0005818033 | 0.2047 | 12 | 报告一个那个造成啊这个他们这边 | cer | 0.4667 |
| 640 | streaming_asr | emilia_zh_0005818033 | 0.1520 | 11 | 报告一个那个过程啊这个咱们这边 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005818033 | 0.1287 | 9 | 报告一个那个过程啊这个咱们这边 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0005818215 | 0.1757 | 24 | 就打去去情绪切还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.3548 |
| 320 | streaming_asr | emilia_zh_0005818215 | 0.1980 | 23 | 就大请确实情绪去啊还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 640 | streaming_asr | emilia_zh_0005818215 | 0.2104 | 29 | 就大请确实情绪期间还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0005818215 | 0.1955 | 26 | 就大请确实情绪去啊还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0005853036 | 0.2254 | 26 | 做这个就如果我想长的话我排斥啊家里你介绍的也事儿呼吸没啥 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005853036 | 0.2159 | 28 | 做这个就如果我然后长的话我排斥者嗯家里就介绍的也事儿父亲没是 | cer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0005853036 | 0.2190 | 32 | 做这个就如果不是然后长的话我排斥嗯家里就介绍的也事儿会没是 | cer | 0.5312 |
| 1280 | streaming_asr | emilia_zh_0005853036 | 0.2000 | 28 | 的这个就如果不是然后找的话我不排斥嗯家里新介绍的也哎呼吸没是 | cer | 0.5312 |
| 160 | streaming_asr | emilia_zh_0005903796 | 0.2687 | 26 | 而且一个我咱们路节目钱为朋友给我信息说他中。打车然后打车司机大一点一个那个说 | cer | 0.5122 |
| 320 | streaming_asr | emilia_zh_0005903796 | 0.2844 | 25 | 而且刚刚我咱们路节目钱为朋友给我信息说他中打车然后卡车司机大一点一个能说 | cer | 0.4878 |
| 640 | streaming_asr | emilia_zh_0005903796 | 0.2625 | 27 | 而且刚刚我咱们路节目钱为朋友给我发微信说他中的打车然后卡车司机大一点一个那说 | cer | 0.4146 |
| 1280 | streaming_asr | emilia_zh_0005903796 | 0.2344 | 26 | 而且刚刚我咱们路节目钱为朋友给我发微信说他中的打车然后下车司机大一个那说 | cer | 0.3659 |
| 160 | streaming_asr | emilia_zh_0005905391 | 0.1991 | 11 | 的这影响那感觉是什么了就是麦当劳 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005905391 | 0.1948 | 12 | 的是影响那感觉是是什么了就是麦当劳 | cer | 0.4375 |
| 640 | streaming_asr | emilia_zh_0005905391 | 0.1775 | 12 | 的之影响的感觉是是什么了就是麦当劳 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0005905391 | 0.1515 | 15 | 之影响的感觉是是什么了就是麦当劳 | cer | 0.3750 |
| 160 | streaming_asr | emilia_zh_0005928718 | 0.2727 | 22 | 很好还是一个望着嗯船长现在是重要为一个牌 | cer | 0.3182 |
| 320 | streaming_asr | emilia_zh_0005928718 | 0.2400 | 22 | 传统还是一个人完整的嗯船长现在是重要为一个牌 | cer | 0.4091 |
| 640 | streaming_asr | emilia_zh_0005928718 | 0.2145 | 18 | 传统还是一直惯着的嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0005928718 | 0.2036 | 16 | 传统还是一直惯着的嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0005999475 | 0.2225 | 31 | 嗯这生活上你没有感觉特别的过因为我听说今年全球的费神之下然后现在现在变异常的过也 | cer | 0.4118 |
| 320 | streaming_asr | emilia_zh_0005999475 | 0.2454 | 36 | 然后社会生活你没有感觉特别的过因为我听说今年全球的费身之下然后现在现在变异常的国也 | cer | 0.4706 |
| 640 | streaming_asr | emilia_zh_0005999475 | 0.2500 | 35 | 嗯在生活中你没有感觉特别的过因为我听说今年全球的飞升之下然后现在现在变异常的过也 | cer | 0.4118 |
| 1280 | streaming_asr | emilia_zh_0005999475 | 0.2294 | 37 | 嗯在生活中你没有感觉特别的过进入我听说今年全球对飞升之下然后现在现在变异常的过也 | cer | 0.4706 |
| 160 | streaming_asr | emilia_zh_0006000255 | 0.2265 | 22 | 因为上次我在旅行天也得在介绍的是一个德国的路线然后我的两位小姐妹一个我说 | cer | 0.1892 |
| 320 | streaming_asr | emilia_zh_0006000255 | 0.2427 | 24 | 因为上次我在旅行天得家这的是一个德国的路线马然后我的两位小姐妹一个我说 | cer | 0.2162 |
| 640 | streaming_asr | emilia_zh_0006000255 | 0.2524 | 28 | 因为上次我在旅行天得家接着的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1892 |
| 1280 | streaming_asr | emilia_zh_0006000255 | 0.2492 | 25 | 因为上次我在旅行天给家接着的是一个德国的路线嘛然后我的两位小姐妹一个我说 | cer | 0.1622 |
| 160 | streaming_asr | emilia_zh_0006041629 | 0.2203 | 27 | 做他来谁学习后觉得现在去过去里其实创伤来但是整体来说我觉得这演员他他眼里在躲避有趣 | cer | 0.5417 |
| 320 | streaming_asr | emilia_zh_0006041629 | 0.1683 | 23 | 做看来谁信息后觉得下去过去里些创伤啊但是整体来说我觉得这演员他他眼里在特别有趣 | cer | 0.4167 |
| 640 | streaming_asr | emilia_zh_0006041629 | 0.1757 | 22 | 就是看像谁信息以后觉得现在就够里些创伤来但是整体来说我觉得这演员他在眼里在别有趣 | cer | 0.3542 |
| 1280 | streaming_asr | emilia_zh_0006041629 | 0.1782 | 21 | 就是看来谁信息后觉得这样就够里些创伤来但是整体来说我觉得这演员他他眼里在别有趣 | cer | 0.3958 |
| 160 | streaming_asr | emilia_zh_0006056256 | 0.1689 | 18 | 在在<\|write_generate\|><\|cmn\|><\|start_content\|>这个情况特别容易爱情因为觉得完全 | cer | 2.3182 |
| 320 | streaming_asr | emilia_zh_0006056256 | 0.1872 | 18 | 在啊的这种情况特别容易行用觉得完全 | cer | 0.4091 |
| 640 | streaming_asr | emilia_zh_0006056256 | 0.2374 | 21 | 啊啊的这种情况特别容易行因为觉得完全 | cer | 0.4091 |
| 1280 | streaming_asr | emilia_zh_0006056256 | 0.2329 | 18 | 啊啊的这种情况就特别容易爱情因为觉得完全 | cer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0006099379 | 0.1398 | 14 | 啊都了第一四居然个这样然后然后分开使用 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006099379 | 0.1613 | 17 | 啊都了你四转到个这样然后然后分开使用 | cer | 0.5789 |
| 640 | streaming_asr | emilia_zh_0006099379 | 0.1470 | 15 | 哦都了你四居然个量然后他分开使用 | cer | 0.3684 |
| 1280 | streaming_asr | emilia_zh_0006099379 | 0.1398 | 15 | 哦都了你四居然个量然后他分开使用 | cer | 0.3684 |
| 160 | streaming_asr | emilia_zh_0006119067 | 0.0861 | 7 | 而且这个还有一个很让人就是写我明白的呀这个是 | cer | 0.2174 |
| 320 | streaming_asr | emilia_zh_0006119067 | 0.1005 | 10 | 而且这还有一个很让人就是是我明白的点这个是 | cer | 0.1739 |
| 640 | streaming_asr | emilia_zh_0006119067 | 0.1148 | 13 | 而且这还有一个很让人就是是我明白的点这个是 | cer | 0.1739 |
| 1280 | streaming_asr | emilia_zh_0006119067 | 0.1148 | 11 | 而且这里还有一个很让人就是是我明白的呀这个是 | cer | 0.1739 |
| 160 | streaming_asr | emilia_zh_0006119250 | 0.2361 | 20 | 一就是呢我本来啊天蓬说朋友啊旁嘛旁然后还有点儿事儿 | cer | 0.4828 |
| 320 | streaming_asr | emilia_zh_0006119250 | 0.2262 | 24 | 一就是我呢我本来啊天蓬说吧朋友捧马捧然后还说点事儿 | cer | 0.4828 |
| 640 | streaming_asr | emilia_zh_0006119250 | 0.2426 | 23 | 一就是如果呢我本来啊肩膀说朋友啊捧捧捧啊，然后还有点儿事儿 | cer | 0.6207 |
| 1280 | streaming_asr | emilia_zh_0006119250 | 0.2393 | 22 | 一就是说那我本来啊天蓬说朋友啊捧捧捧啊，然后还说点事儿 | cer | 0.4828 |
| 160 | streaming_asr | emilia_zh_0006174150 | 0.2105 | 19 | 就是因为我的一些当时不好的情绪其实有传染呢 | cer | 0.0909 |
| 320 | streaming_asr | emilia_zh_0006174150 | 0.2068 | 18 | 就是因为我的一些当时不好的情绪其实有传染了 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0006174150 | 0.1880 | 18 | 就是因为我的一些当时不好了情绪其实有传染了 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0006174150 | 0.1955 | 20 | 就是因为我的一些当时不好了情绪其实对传染了 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006212201 | 0.2609 | 17 | 没有办法一直坚持别说的动所以买通那个忽然 | cer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0006212201 | 0.2609 | 14 | 他没有办法一直坚持别说的动所以买通你个胡人 | cer | 0.3043 |
| 640 | streaming_asr | emilia_zh_0006212201 | 0.2464 | 17 | 没有办法一直坚持结束的移动作为买通的个胡人 | cer | 0.4783 |
| 1280 | streaming_asr | emilia_zh_0006212201 | 0.2464 | 15 | 没有办法一直坚持别说的移动所以买通的个胡人 | cer | 0.3478 |
| 160 | streaming_asr | emilia_zh_0006212326 | 0.1359 | 28 | 嗯一员这个的正确是这小老虎就是他一直大家冒着这就是了各地上的王子隐藏的自己的真实身份。 | cer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006212326 | 0.1392 | 28 | 嗯一个人这个的正确是这小老虎就是他一直大家冒着这出了各地上的王子隐藏自的真实身份。 | cer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006212326 | 0.1748 | 33 | 如远远这个真正是这小老虎只是他一直大家冒着这出了个体上的王子隐藏自的真实身份是 | cer | 0.4500 |
| 1280 | streaming_asr | emilia_zh_0006212326 | 0.1845 | 37 | 无一个到的真正是这小老虎就是他一直大家冒着这出了的上的王子隐藏字你真实身份是 | cer | 0.5250 |
| 160 | streaming_asr | emilia_zh_0006270122 | 0.1567 | 18 | 你这样都是带的他他一个完整的这个原因啊他就是一 | cer | 0.3600 |
| 320 | streaming_asr | emilia_zh_0006270122 | 0.1604 | 23 | 你这样都是对的他他一个完整的这个原因啊他就是一 | cer | 0.3200 |
| 640 | streaming_asr | emilia_zh_0006270122 | 0.1679 | 18 | 你这样主是对的他是一个完整的这个原因啊他就是一 | cer | 0.2800 |
| 1280 | streaming_asr | emilia_zh_0006270122 | 0.1716 | 20 | 你这样读是对的他他一个完整的这个原因要他就是一 | cer | 0.3200 |
| 160 | streaming_asr | emilia_zh_0006303057 | 0.1626 | 49 | 一人吧就是以前一个无限挑战的我他们是一起的然后内有限偶然有一期干什么呢航空速擦这万全的玻璃<\|write_generate\|><\|cmn\|><\|start_content\|>这妈妈外圈的玻璃做生降级在外面其实在那个窗户外面 | cer | 0.8919 |
| 320 | streaming_asr | emilia_zh_0006303057 | 0.1563 | 46 | 一人吧就是以前一个无限挑战的他们是一起你的然后内有限责任有一期干什么呢旁名素擦这万全玻璃你这妈妈外圈的玻璃做生计在外面就是在那个窗户外面 | cer | 0.3378 |
| 640 | streaming_asr | emilia_zh_0006303057 | 0.1616 | 49 | 艺人吧就是以前一个无限挑战的我他们是一起你然后内路线扰乱有一期干什么呢旁名素擦这外圈玻璃你这妈妈外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.2973 |
| 1280 | streaming_asr | emilia_zh_0006303057 | 0.1626 | 47 | 一人我就是以前一个无限挑战的的他们是一起你然后那个无限偶然有一起干什么呢旁名素擦这个外圈玻璃<\|write_generate\|><\|cmn\|><\|start_content\|>这妈妈外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.8378 |
| 160 | streaming_asr | emilia_zh_0006304027 | 0.1643 | 19 | 如果按照我们刚刚所讲的这个规律来看的话这我们对的龙啊人是哪更呢 | cer | 0.2286 |
| 320 | streaming_asr | emilia_zh_0006304027 | 0.1902 | 28 | 如果按照什么刚刚所讲的这个规律来看的话这我们对应的龙啊赢是哪更呢 | cer | 0.2571 |
| 640 | streaming_asr | emilia_zh_0006304027 | 0.1729 | 23 | 如果按照什么刚刚所讲的这个规律来看的话这里我们对应的龙啊因是哪个呢 | cer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0006304027 | 0.1614 | 22 | 如果按照我们刚刚所讲的最大的规律来看的话最我们对应到龙啊因是哪个呢 | cer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0006330534 | 0.1993 | 20 | And had no once please several days at last I ceive to showed not | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006330534 | 0.2509 | 20 | And had no once for several days at last I receives to showed not | wer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0006330534 | 0.2612 | 20 | And had no once but several days at last I receives to show then | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0006330534 | 0.1615 | 24 | And had no once but Several days At last I receives to showed not | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0006350396 | 0.1614 | 25 | I'm on found be position a the 'clock with spent a you okay And we found logins nabas they | wer | 0.8235 |
| 320 | streaming_asr | emilia_zh_0006350396 | 0.1582 | 20 | My on found be position and the cluck with spent a in okay And we found logins they are they are | wer | 0.8824 |
| 640 | streaming_asr | emilia_zh_0006350396 | 0.1424 | 21 | My I am found be position and the cluck with spend the and okay And we found logins they are they are | wer | 0.8824 |
| 1280 | streaming_asr | emilia_zh_0006350396 | 0.1234 | 19 | I'm I on found be position a a 'clock with spend the and okay And we found logins they are they are | wer | 0.8824 |
| 160 | streaming_asr | emilia_zh_0006350510 | 0.1439 | 19 | Just she they had live very is related i've india And you terribly shy of men | wer | 0.6111 |
| 320 | streaming_asr | emilia_zh_0006350510 | 0.1547 | 20 | Just shit they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 640 | streaming_asr | emilia_zh_0006350510 | 0.1619 | 18 | Just shed they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0006350510 | 0.1331 | 15 | Just set they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 160 | streaming_asr | emilia_zh_0006366492 | 0.1111 | 30 | He Just no want say some gaw his great cherish He want to return and they with men These Choose will give may great power | wer | 0.4231 |
| 320 | streaming_asr | emilia_zh_0006366492 | 0.1412 | 29 | He Just no want say some gaw his great cherish He want to return and let with men These chews will give may great power | wer | 0.4231 |
| 640 | streaming_asr | emilia_zh_0006366492 | 0.1281 | 28 | He Just no want say some gaw his great <\|bicodec_global_2703\|><\|eng\|><\|start_content\|>she He want to return and let with men These retudes will give may great power | wer | 0.4231 |
| 1280 | streaming_asr | emilia_zh_0006366492 | 0.1036 | 24 | He the not want to say some gaw his great treasures He want to return and let with men These retches will give may great power | wer | 0.3462 |
| 160 | streaming_asr | emilia_zh_0006366864 | 0.1786 | 15 | I want to continue to make be house a happy hong for him | wer | 0.2308 |
| 320 | streaming_asr | emilia_zh_0006366864 | 0.1696 | 13 | I want it to continue to make believe house a happy hong for him | wer | 0.3077 |
| 640 | streaming_asr | emilia_zh_0006366864 | 0.1429 | 13 | I Wanted to continue to make believe house a happy home home him | wer | 0.1538 |
| 1280 | streaming_asr | emilia_zh_0006366864 | 0.1473 | 14 | And want to continue to make believe house a happy home home him | wer | 0.3077 |
| 160 | streaming_asr | emilia_zh_0006379722 | 0.1551 | 18 | But This couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0006379722 | 0.1320 | 18 | But This couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0006379722 | 0.1287 | 16 | But the couldn't lead the the to He must take then was him all the way | wer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0006379722 | 0.1287 | 16 | But the couldn't lead the there two He must take then will him all the way | wer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0006379861 | 0.2138 | 25 | We the Professor when into the cell he had one find all real in to ten all are bills | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006379861 | 0.1931 | 23 | I in Professor when into the cell he had one five all real in to ten all are bills | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006379861 | 0.1759 | 20 | I in the Professor when int the cell he had one five a a bill in to ten all re bills | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0006379861 | 0.1724 | 21 | I the Professor when int the cell he had one five out real in to ten all re bills | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0006430098 | 0.2280 | 23 | Just was offered Several good job But he want it wait and the coron | wer | 0.4286 |
| 320 | streaming_asr | emilia_zh_0006430098 | 0.2443 | 24 | Zhang Was offer Several good job But he want it wait and but coron. | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006430098 | 0.1889 | 19 | Zhang Was offered several good job But he want it wait and the corrot and | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0006430098 | 0.1498 | 18 | John Was offered Several good job But he want it wait and the corral and | wer | 0.4286 |
| 160 | streaming_asr | emilia_zh_0006430973 | 0.2112 | 24 | The Just just was was something the Greg could with out if had to But there writing does had to state | wer | 0.5455 |
| 320 | streaming_asr | emilia_zh_0006430973 | 0.1957 | 20 | The chess the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 640 | streaming_asr | emilia_zh_0006430973 | 0.1553 | 26 | The Just the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 1280 | streaming_asr | emilia_zh_0006430973 | 0.1460 | 23 | The chess the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 160 | streaming_asr | emilia_zh_0006435274 | 0.1283 | 14 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0006435274 | 0.1460 | 17 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0006435274 | 0.1018 | 13 | I bank charges interest are brother doesn't charge interested | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0006435274 | 0.1195 | 15 | I bank charges interest of brother doesn't charge interest | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006435497 | 0.1925 | 55 | But millo and balm three Different algorithms on the top you see selection sort on the bod my will see bubble sword and in Middle you all see and here in protection of but in law again is AKA merder sor Today | wer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0006435497 | 0.1761 | 55 | The Middle and barram three Different algorithms on the top you see selection sort on the bod my will s bubble sword and in Middle you all see and here and perception of let in law again is AKA merder sor Today | wer | 0.3556 |
| 640 | streaming_asr | emilia_zh_0006435497 | 0.1384 | 46 | The Middle and bodam three Different algorithms on the hop you see selection sort on the bod my will see bubble sword and in Middle you all see and here and protection of let in law again is AK merder sor Today | wer | 0.3778 |
| 1280 | streaming_asr | emilia_zh_0006435497 | 0.1321 | 48 | The Middle in the bodam three different algorithms on the top you see select sort on the bod my will see bubble sword and in Middle you all see and here and perasure of let in law again is AKA merder sor Today | wer | 0.3111 |
| 160 | streaming_asr | emilia_zh_0006446583 | 0.1923 | 27 | Now will can use am alright imaging see would is Actually hopping insight join what some one cracks nuckles | wer | 0.6500 |
| 320 | streaming_asr | emilia_zh_0006446583 | 0.1813 | 27 | Now will can use am alright imaging see would is actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0006446583 | 0.1758 | 26 | Now will can use am sorry. imaging see would is Actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 1280 | streaming_asr | emilia_zh_0006446583 | 0.1703 | 24 | Now will can use am sorry. imaging see would is Actually hopping insight join what someone cracks nuckles | wer | 0.6000 |
| 160 | streaming_asr | emilia_zh_0006447049 | 0.1837 | 49 | You part the in thing that what well it and contribute the society my take good not to you friend love me and I rode the of are could go to say this this was I to to realize after turn thirty not to on a | wer | 0.6000 |
| 320 | streaming_asr | emilia_zh_0006447049 | 0.1756 | 45 | You part of in thing like what well it and contribute but society my it good not the you friend love me and I rode to of are could go to say this this was day to to realize after turn thirty not to on a | wer | 0.6000 |
| 640 | streaming_asr | emilia_zh_0006447049 | 0.1837 | 47 | You part of and thing that what well it and contribute but society my it good no the you friend love me and I rode the of are could go to say this this was die to to realize after turn thirty not to on a | wer | 0.6200 |
| 1280 | streaming_asr | emilia_zh_0006447049 | 0.1707 | 48 | You part of in think they what well could and contribute but society my it good not the you friend love me and I rode the of are could go to say this this was die to to realize after turn thirty not to on a | wer | 0.6000 |
| 160 | streaming_asr | emilia_zh_0006464698 | 0.2249 | 29 | I one of ways could for example He one me to tell and my big count pass were and he can take photographed so for me from different dangles | wer | 0.5556 |
| 320 | streaming_asr | emilia_zh_0006464698 | 0.2227 | 32 | I one of ways could for ex stumble he one me to tell in my big count pass were and he can take photographed so for me from different dangles | wer | 0.6296 |
| 640 | streaming_asr | emilia_zh_0006464698 | 0.2271 | 30 | I one for has could for ex<\|glm_semantic_3040\|>bal he one me to tell in my being count pass were and he can take photographed so for me from different dangles | wer | 0.6667 |
| 1280 | streaming_asr | emilia_zh_0006464698 | 0.2249 | 32 | I one of ways clear for ex gamble he one me to tell in my being count pass were and he can take photographed so me from different dangles | wer | 0.5926 |
| 160 | streaming_asr | emilia_zh_0006464935 | 0.1719 | 30 | That Praise point where quantity the consumers want but I equal the quantity the cellar want produce is cold the each liberty breed | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0006464935 | 0.1929 | 33 | That price the point point liquidity the consumers want but I equal the quantity the cellar want produce is cold the each liberty breed | wer | 0.5769 |
| 640 | streaming_asr | emilia_zh_0006464935 | 0.1824 | 32 | That price the point point quality the consumers want but I equal the quantity the cellar want produce is cold the each liberty prey | wer | 0.5769 |
| 1280 | streaming_asr | emilia_zh_0006464935 | 0.1656 | 30 | That price the point point ac quantity the consumers want but I equal the quantity the cellar want produce is cold the equal liberty prey | wer | 0.5385 |
| 160 | streaming_asr | emilia_zh_0006503627 | 0.2255 | 35 | 昔日繁华都会情怀圣经经历如此肆无忌惮的烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0006503627 | 0.2276 | 31 | 昔日繁华都会情怀圣经经历如此肆无忌惮的烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0006503627 | 0.2276 | 32 | 昔日繁华都会情怀圣经经历如此肆无忌惮烧杀抢夺后被摧毁打击成为一片废墟 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0006503627 | 0.2296 | 35 | 昔日繁华多会情怀圣经经历如此肆无忌惮烧杀抢后被摧毁打击成为一片废墟 | cer | 0.1944 |
| 160 | streaming_asr | emilia_zh_0006544664 | 0.1993 | 34 | 后院货运商贸走了钱这种小车在广州的门的的是还是四当时也这一夜。注意到车的颜色是一种会来了 | cer | 0.4681 |
| 320 | streaming_asr | emilia_zh_0006544664 | 0.1926 | 30 | 货运货运商贸走我钱这种小车在广州的门的的是而且四当时也这物业这注意到车的颜色是一种会而 | cer | 0.4043 |
| 640 | streaming_asr | emilia_zh_0006544664 | 0.1742 | 33 | 货运货运商贸走了钱这种小车在广州的们的的是还是四当时也这物业这注意到车的颜色深色一种会而 | cer | 0.4894 |
| 1280 | streaming_asr | emilia_zh_0006544664 | 0.1692 | 30 | 客运货运商贸走了钱这种小车在广州们的的是而四当时人也物业这注意到车的颜色是一种会而 | cer | 0.4255 |
| 160 | streaming_asr | emilia_zh_0006544707 | 0.1421 | 22 | 这两人那也是在你说这个去见有登记以后呢现象而且这两两人其中有一个 | cer | 0.2647 |
| 320 | streaming_asr | emilia_zh_0006544707 | 0.1395 | 22 | 这了人那也是在你说这个区间有登记以后呢现象而且这两两人其中有一个 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0006544707 | 0.1342 | 22 | 这两个人那也是在你说的这个区间有经济结婚现象而且这两两人其中有一个 | cer | 0.1471 |
| 1280 | streaming_asr | emilia_zh_0006544707 | 0.1368 | 21 | 这两人那也是在你说的这个区间有登记结婚现象而这两两人其中有一个 | cer | 0.1471 |
| 160 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 25 | 就是在共有大学学习本年后第一份给你嗯的工作直接就是到博物馆冷猫 | cer | 0.2121 |
| 320 | streaming_asr | emilia_zh_0006598681 | 0.2500 | 28 | 就在共有大学学习半年多年第一份给你嗯的工作直接就是到博物馆冷猫 | cer | 0.2424 |
| 640 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 31 | 就是在共有学习学习半点和第一份给你安排的工作直接就是到博物馆冷猫 | cer | 0.2424 |
| 1280 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 31 | 就是在共有学习学习半年多年第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006610357 | 0.2480 | 27 | 今天觉得是非常确实关于非常只想但是似乎默默的这种做法他其实也并没有什么错觉得 | cer | 0.2791 |
| 320 | streaming_asr | emilia_zh_0006610357 | 0.2693 | 31 | 今天觉得是非常确实关键性非常之前但是似乎模模的这种做法他其实也并没有什么错我觉得 | cer | 0.2093 |
| 640 | streaming_asr | emilia_zh_0006610357 | 0.2693 | 31 | 今天觉得是非常确实关键性非常之前但是似乎模模的这种做法他其实也并没有错我觉得 | cer | 0.2558 |
| 1280 | streaming_asr | emilia_zh_0006610357 | 0.2507 | 31 | 这觉得是非常确实关键信息非常之前但是似乎做梦的这种做法他其实也并没有错我觉得 | cer | 0.2558 |
| 160 | streaming_asr | emilia_zh_0006658157 | 0.2424 | 34 | 所以我觉得是这样说谁任何品牌在中国办活动他肯定也是因为这一个活动深或者就一个产品本身他在 | cer | 0.2200 |
| 320 | streaming_asr | emilia_zh_0006658157 | 0.2254 | 40 | 这我觉得是这样说所以任何品牌在中国办活动。肯定也是因为在一个活动深或者就也产品本身他在 | cer | 0.2800 |
| 640 | streaming_asr | emilia_zh_0006658157 | 0.2121 | 40 | 最我觉得是这样说所以任何品牌在中国办活动。肯定也是因为在一个活动深或者就也产品本身他在 | cer | 0.2800 |
| 1280 | streaming_asr | emilia_zh_0006658157 | 0.2140 | 37 | 最我觉得是这样说所以任何并在中国办活动<\|write_generate\|><\|cmn\|><\|start_content\|>肯定也是因为在一个活动深或者就也产品本身他在 | cer | 1.1400 |
| 160 | streaming_asr | emilia_zh_0006713619 | 0.2237 | 17 | 不管那些心慢慢没要护寡赶紧护寡起来在给你三分钟时间互关玩发布了啊 | cer | 0.5135 |
| 320 | streaming_asr | emilia_zh_0006713619 | 0.2303 | 19 | 不管啊那些心男人你要护光赶紧护光起来在给你你三分钟时间互关玩啊，不了啊 | cer | 0.4595 |
| 640 | streaming_asr | emilia_zh_0006713619 | 0.2368 | 19 | 不管啊那些心们你要护光赶紧护光起来在给你三分钟时间互关玩啊，不了啊 | cer | 0.4324 |
| 1280 | streaming_asr | emilia_zh_0006713619 | 0.2368 | 19 | 不管啊那些心们你要护冠赶紧护冠起来在给你散恨事情互关玩下播了啊 | cer | 0.5135 |
| 160 | streaming_asr | emilia_zh_0006714517 | 0.1607 | 13 | 好接下来我我讲最重要的今天那间事情呢啊 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0006714517 | 0.1786 | 12 | 好接下来我我讲最重要今天那间事情呢啊 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0006714517 | 0.1667 | 11 | 好接下来我啊讲最重要今天那件事情呢啊 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0006714517 | 0.1607 | 11 | 好接下来我讲最重要今天那件事情了啊 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0006725267 | 0.3540 | 13 | 了出去留待也是帮助部分讨论院门 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006725267 | 0.3416 | 12 | 布兰登就是留着也是帮助部分讨论院门 | cer | 0.6316 |
| 640 | streaming_asr | emilia_zh_0006725267 | 0.3292 | 13 | 布莱顿就是留着也是帮助部分讨论院们 | cer | 0.6842 |
| 1280 | streaming_asr | emilia_zh_0006725267 | 0.3478 | 15 | 柏林出榴弹也是帮部分讨论院们 | cer | 0.5789 |
| 160 | streaming_asr | emilia_zh_0006731464 | 0.2278 | 13 | 他选择这种生活方式一定他自己的道理 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0006731464 | 0.1833 | 11 | 他选择这种生活方式一定的自己的道理 | cer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0006731464 | 0.1722 | 12 | 他选择这种生活方式一定的自己的道理 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0006731464 | 0.1667 | 11 | 他选择这种生活方式一定的自己道理 | cer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0006819602 | 0.2480 | 30 | 有半时辰过去城墙的高度还声响不到一张朝军完全停下了前途动作 | cer | 0.2581 |
| 320 | streaming_asr | emilia_zh_0006819602 | 0.2453 | 32 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 640 | streaming_asr | emilia_zh_0006819602 | 0.2183 | 26 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0006819602 | 0.2210 | 31 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0006874664 | 0.2236 | 34 | 那其中他有说到他非常成的就是希望如我说这个饮水没如果是查沙怎么样在对这段要说里面有觉得不要真实 | cer | 0.3077 |
| 320 | streaming_asr | emilia_zh_0006874664 | 0.2257 | 37 | 那其中他有说到他非常成的就是希望如果说这个饮水没有如果是查沙怎么样在对这段要说里面有觉得比较真实 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0006874664 | 0.2278 | 37 | 那其中他有说到他非常成的就是希望如果说这个谁谁没如果是查沙怎么样在对这的要说里面有觉得比较真实 | cer | 0.2885 |
| 1280 | streaming_asr | emilia_zh_0006874664 | 0.2173 | 36 | 那其中他有说到他非常成的就是希望如果说这个谁谁没如果是查沙怎么样在对这的要说里面有觉得比较真实 | cer | 0.2885 |
| 160 | streaming_asr | emilia_zh_0006883085 | 0.2453 | 16 | 在于终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 320 | streaming_asr | emilia_zh_0006883085 | 0.2075 | 17 | 在与终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0006883085 | 0.2217 | 17 | 在与终于停下几我都已经有点昏昏欲睡来了 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0006883085 | 0.2028 | 16 | 在与终于停下几我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 160 | streaming_asr | emilia_zh_0006940399 | 0.1642 | 16 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0006940399 | 0.1343 | 12 | 虽然他们他人都想进去浅显从跌倒的地方爬起来 | cer | 0.2381 |
| 640 | streaming_asr | emilia_zh_0006940399 | 0.1343 | 12 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0006940399 | 0.1194 | 13 | 虽然他们他人都想进去前行从跌倒的地方爬起来 | cer | 0.1905 |
| 160 | streaming_asr | emilia_zh_0006940509 | 0.3218 | 17 | 宏伟都是的街道很快变不了红雨 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0006940509 | 0.3276 | 16 | 宏伟都是的街道很快被不了红雨 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0006940509 | 0.2931 | 15 | 红都是的街道很快被不了红雨 | cer | 0.4667 |
| 1280 | streaming_asr | emilia_zh_0006940509 | 0.2529 | 14 | 红都是的街道很快被不了红雨 | cer | 0.4667 |
| 160 | streaming_asr | emilia_zh_0006990963 | 0.1652 | 14 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 320 | streaming_asr | emilia_zh_0006990963 | 0.1652 | 14 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 640 | streaming_asr | emilia_zh_0006990963 | 0.1741 | 15 | 现在正在是春天我们一会儿就去愿离 | cer | 0.2353 |
| 1280 | streaming_asr | emilia_zh_0006990963 | 0.1741 | 16 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 25 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 320 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 28 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 640 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 28 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 1280 | streaming_asr | emilia_zh_0007017041 | 0.3109 | 29 | 几乎是百分之九十以上的地方政府包括州县都无法完成预算 | cer | 0.0370 |
| 160 | streaming_asr | emilia_zh_0007017174 | 0.2617 | 24 | 从某种程度上到让不是出特点演可能说法啊是所得税恐怕就是工薪税啊 | cer | 0.2424 |
| 320 | streaming_asr | emilia_zh_0007017174 | 0.2727 | 27 | 从某种程度上啊到人不是出特别眼睛可能说法啊说吧啊所得税恐怕就是工薪税啊 | cer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0007017174 | 0.2948 | 28 | 从某种程度上啊当人不是出特别盐可能说法啊说吧啊所得税恐怕就是工薪税啊 | cer | 0.2121 |
| 1280 | streaming_asr | emilia_zh_0007017174 | 0.2645 | 31 | 从某种程度上啊当让不是出策略盐可能说法啊说所得税恐怕就是工薪税啊 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0007060532 | 0.3606 | 17 | 如果你对保格人偶间舞在就比如嘴说 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0007060532 | 0.3029 | 17 | 如果你对报告人偶件舞这样就比如嘴说 | cer | 0.5263 |
| 640 | streaming_asr | emilia_zh_0007060532 | 0.3029 | 15 | 如果你对表格人偶件是舞真爱就比如嘴说 | cer | 0.4211 |
| 1280 | streaming_asr | emilia_zh_0007060532 | 0.3125 | 17 | 如果你对表格人物件物真爱就别用嘴说 | cer | 0.2632 |
| 160 | streaming_asr | emilia_zh_0007120510 | 0.1531 | 10 | 他将极大的的激励整个团队伙伴的习性 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0007120510 | 0.1429 | 10 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0007120510 | 0.1378 | 10 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0007120510 | 0.1378 | 11 | 他家极大的激励整个团队伙伴的细心 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0007121845 | 0.1728 | 25 | 那么我们会也许只是依赖于一个技术比个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 320 | streaming_asr | emilia_zh_0007121845 | 0.1640 | 24 | 那么我们会也许只是依赖于一个技术bigger方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.1463 |
| 640 | streaming_asr | emilia_zh_0007121845 | 0.1693 | 26 | 那么我们会也许只是一烂鱼一个技术bigger方法一个优势或者是资源一些本地我们就去对抗市场 | cer | 0.2683 |
| 1280 | streaming_asr | emilia_zh_0007121845 | 0.1675 | 26 | 那么我们会也许只是一烂鱼一个技术比个方法一个优势或者是资源一些本地我们就去对抗市场 | cer | 0.1463 |
| 160 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 21 | 印度所说的空隙风险在一起用最好印度尼西亚也曾经年龄高风险 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007124543 | 0.2016 | 25 | 印度所说的空隙风险在一起用最高印度尼西亚也曾经面临高风险 | cer | 0.2143 |
| 640 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 23 | 印度所说的空风险在一起用最高印度尼西亚也曾经年龄高风险 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007124543 | 0.2094 | 27 | 印度所说的空风险在一起用最高印度尼西亚也曾经面临高风险 | cer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0007124790 | 0.1328 | 12 | 你好请问就是去老自的路吧男人看了解决也 | cer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0007124790 | 0.1441 | 11 | 你好请问这是去老自的路吧男人看了表情也 | cer | 0.3182 |
| 640 | streaming_asr | emilia_zh_0007124790 | 0.1130 | 12 | 你好请问这个是去了自的路吧男人看了调军也 | cer | 0.4091 |
| 1280 | streaming_asr | emilia_zh_0007124790 | 0.1130 | 13 | 你好请问这个是去老自的路吧男人看了决定也 | cer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0007169925 | 0.1077 | 29 | 啊犹太人似乎在这方面的要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何时间思考 | cer | 0.1190 |
| 320 | streaming_asr | emilia_zh_0007169925 | 0.1105 | 29 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩但他们也会利用任何实际思考 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007169925 | 0.1161 | 29 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何实际思考 | cer | 0.1190 |
| 1280 | streaming_asr | emilia_zh_0007169925 | 0.1035 | 31 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何实际思考 | cer | 0.1190 |
| 160 | streaming_asr | emilia_zh_0007170191 | 0.2204 | 13 | 这对犹太人来说肯定是不能接受等 | cer | 0.0667 |
| 320 | streaming_asr | emilia_zh_0007170191 | 0.2043 | 16 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007170191 | 0.2204 | 15 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007170191 | 0.2151 | 15 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007312342 | 0.1181 | 22 | 开他人生中的第一个支配账户他叫他怎么写支票一天上班是丹尼斯提到 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0007312342 | 0.1181 | 21 | 开他人生中的第一个支配账户他教他怎么写支票一天上班是丹尼丝提到 | cer | 0.1250 |
| 640 | streaming_asr | emilia_zh_0007312342 | 0.1224 | 23 | 开他人生中的第一个支配账户他叫他怎么也支票一天上班时丹尼斯提到 | cer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0007312342 | 0.1055 | 21 | 开他人生中的第一个支配账户他叫他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 160 | streaming_asr | emilia_zh_0007312674 | 0.1400 | 21 | 事实上轰隆也不需要你亲爱的啊他根本知道自己现在河地也认不出身边是谁和他大家在一起 | cer | 0.2051 |
| 320 | streaming_asr | emilia_zh_0007312674 | 0.1460 | 22 | 事实上轰隆也不需要你亲爱的啊他根本知道自己现在河地也认不出身边是谁谁他大家在一起 | cer | 0.2308 |
| 640 | streaming_asr | emilia_zh_0007312674 | 0.1643 | 25 | 事实上婚礼也不需要你亲爱的的他根本知道自己现在和地也认不出身边是谁和他大家在一起 | cer | 0.2051 |
| 1280 | streaming_asr | emilia_zh_0007312674 | 0.1542 | 26 | 事实上轰隆也不需要你亲爱的的他根本知道自己生在和地也认不出身边是谁和他带在一起 | cer | 0.1795 |
| 160 | streaming_asr | emilia_zh_0007353483 | 0.2070 | 16 | 不同的选择会创造出不同的未来关键而虚利益 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007353483 | 0.2507 | 16 | 不同的选择会创造出不同的未来关键而虚利益 | cer | 0.2273 |
| 640 | streaming_asr | emilia_zh_0007353483 | 0.2362 | 18 | 不同的选择会创造处不同未来关键而虚利益 | cer | 0.3182 |
| 1280 | streaming_asr | emilia_zh_0007353483 | 0.2391 | 21 | 不同选择会创造处不同未来观点而虚利益 | cer | 0.4545 |
| 160 | streaming_asr | emilia_zh_0007399682 | 0.1637 | 22 | 有些上甚至一更加严厉的态度精神是我们咬认清生命的脆弱告诉我没一个人 | cer | 0.2353 |
| 320 | streaming_asr | emilia_zh_0007399682 | 0.1615 | 24 | 有些上甚至一更加而言的态度精神我们要认清生命的脆弱告诉我没一个人 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0007399682 | 0.1438 | 23 | 有些上甚至一更加而言的态度精神是我们要认清生命脆弱告诉我没一个人 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0007399682 | 0.1460 | 22 | 有些上甚至一更加眼里的态度精神是我们要认清生命脆弱告诉我没一个人 | cer | 0.2941 |
| 160 | streaming_asr | emilia_zh_0007399687 | 0.1871 | 12 | 他在一千多年钱许下的落叶知识不去 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0007399687 | 0.1930 | 12 | 他在一千多年钱许下的落叶知识不去 | cer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0007399687 | 0.1871 | 13 | 他他一千多年钱许下的落叶知识不去 | cer | 0.4375 |
| 1280 | streaming_asr | emilia_zh_0007399687 | 0.1813 | 13 | 他他一千多年钱许下的落叶知识不去去 | cer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0007461662 | 0.1224 | 23 | 吧那么这个是导师新的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0007461662 | 0.1204 | 23 | 吧那么这个是导师信息的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0007461662 | 0.1367 | 25 | 把了吗这个是导师信息的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.1944 |
| 1280 | streaming_asr | emilia_zh_0007461662 | 0.1204 | 26 | 把那么这个是导师感兴趣的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0556 |
| 160 | streaming_asr | emilia_zh_0007462823 | 0.1336 | 12 | 因为他说到野餐四的图片a.是挑个尼克的 | cer | 0.6429 |
| 320 | streaming_asr | emilia_zh_0007462823 | 0.1382 | 12 | 因为他说到野餐所以的图片a.是挑个尼克的的 | cer | 0.5714 |
| 640 | streaming_asr | emilia_zh_0007462823 | 0.1382 | 11 | 因为他说到野餐所以的图片a.是pick a nick的 | cer | 0.4643 |
| 1280 | streaming_asr | emilia_zh_0007462823 | 0.1244 | 11 | 因为他说到野餐所以的图片a.是pick a nick的 | cer | 0.4643 |
| 160 | streaming_asr | emilia_zh_0007524507 | 0.2009 | 16 | 好我在来看又在找的这就用去杠杆他的颜色 | cer | 0.5652 |
| 320 | streaming_asr | emilia_zh_0007524507 | 0.2366 | 18 | 好我在来又在找的这个就是。去杠杆他的的颜色 | cer | 0.6087 |
| 640 | streaming_asr | emilia_zh_0007524507 | 0.2143 | 19 | 好我在来看就散着的这个就是用去更改的的颜色 | cer | 0.4348 |
| 1280 | streaming_asr | emilia_zh_0007524507 | 0.2054 | 18 | 好我在来看就散着的这个就是。去更改他的颜色 | cer | 0.4348 |
| 160 | streaming_asr | emilia_zh_0007526333 | 0.2165 | 28 | 到我自己的的这重点一个就是每年考试都会考的内容就是希腊戏剧的他作品比 | cer | 0.3611 |
| 320 | streaming_asr | emilia_zh_0007526333 | 0.2294 | 28 | 到了这个的的最重点一个就是每年考试都会考这个内容就是希腊戏剧的他作品比 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0007526333 | 0.2088 | 23 | 到了这个这个的最重点个就是每年考试都备考这个内容就是其实戏剧的他够比 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0007526333 | 0.2191 | 26 | 到了这这个的最重点个就是每年考试都高考这个内容就是其实戏剧的他过。比 | cer | 0.3611 |
| 160 | streaming_asr | emilia_zh_0007551053 | 0.2913 | 25 | 七还更多工作是我律师是我所在又工作经验来前台前辈们进行 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007551053 | 0.2880 | 25 | 七更多工作是我律师所所在又工作经验的前台前辈们进行 | cer | 0.2857 |
| 640 | streaming_asr | emilia_zh_0007551053 | 0.3204 | 26 | 其啊更多的工作是有律师所所在有工作经验的前台前辈们进行 | cer | 0.1786 |
| 1280 | streaming_asr | emilia_zh_0007551053 | 0.3010 | 27 | 七啊更多的工作是有律师思索的有工作经验的前台前辈们进行 | cer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0007555536 | 0.2076 | 21 | 就你将起的三个房间那里书籍全部把过来我要细细的查阅想 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007555536 | 0.1903 | 23 | 其实你家起的三个房间那里书籍全部把过来我也细细的查阅想 | cer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0007555536 | 0.2007 | 22 | 且你家其的三个房间那里收集全部把过来我要细细查阅想 | cer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0007555536 | 0.2111 | 21 | 且你家其的三个房间那里书籍全部把过来我要细细查阅想 | cer | 0.3571 |
| 160 | streaming_asr | emilia_zh_0007635379 | 0.2727 | 15 | 所以来比是对于果然来说是一个很重要的是 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0007635379 | 0.2929 | 13 | 所以来比是对于德国人来说是一个很重要的是 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0007635379 | 0.2727 | 14 | 所以来比是对于中国人来说是一个很重要是 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007635379 | 0.2828 | 17 | 所以来比是对于中国人来说是一个很重要是 | cer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0007635686 | 0.2044 | 21 | 我们的工作仍然没有实质性进展呢抓来几个小错误 | cer | 0.1364 |
| 320 | streaming_asr | emilia_zh_0007635686 | 0.1887 | 19 | 我们的工作仍然没有实质性进展呢抓了几个小错误 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0007635686 | 0.2075 | 15 | 我的工作仍然没有实质性进展呢抓来几个小错误 | cer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0007635686 | 0.1950 | 16 | 我的工作仍然没有实质性进展呢抓了几个小错误 | cer | 0.1364 |
| 160 | streaming_asr | emilia_zh_0007690753 | 0.1797 | 36 | 可能市场在行的的是仅仅了一年搞了二零零四年最漫长的下跌再发生上升宗旨再出线越线五连云的悲惨场面 | cer | 0.2549 |
| 320 | streaming_asr | emilia_zh_0007690753 | 0.1881 | 40 | 可能市场这。的的是仅仅了一年搞老二零零四年最漫长下跌再法式上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0007690753 | 0.1898 | 43 | 可能市场这。的的是仅仅了一年搞老二零零四年最漫长下跌再发生上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3137 |
| 1280 | streaming_asr | emilia_zh_0007690753 | 0.1898 | 41 | 可能市场这行的的是仅仅了一年搞老二零零四年最漫长下跌再发生上升宗旨再接触越线五连云的悲惨场面 | cer | 0.3137 |
| 160 | streaming_asr | emilia_zh_0007691054 | 0.1822 | 27 | 但从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标原你仅仅啊 | cer | 0.3514 |
| 320 | streaming_asr | emilia_zh_0007691054 | 0.1893 | 29 | 吧从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标远远仅仅啊 | cer | 0.3784 |
| 640 | streaming_asr | emilia_zh_0007691054 | 0.1939 | 31 | 吧从这个角度来讲我们没而已是一个比定律的增速更好更加重要指标远远仅啊 | cer | 0.3784 |
| 1280 | streaming_asr | emilia_zh_0007691054 | 0.1986 | 33 | 了从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标远远仅仅啊 | cer | 0.3784 |
| 160 | streaming_asr | emilia_zh_0007721270 | 0.2981 | 13 | 那我不管家公司时我通常就两件的 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0007721270 | 0.3043 | 15 | 当我不管家公司时我通常就两件是 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0007721270 | 0.2981 | 17 | 当我关于家公司时我通常就两件是 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0007721270 | 0.2795 | 14 | 当我有关家公司时我通常就两件的 | cer | 0.3889 |
| 160 | streaming_asr | emilia_zh_0007721307 | 0.2000 | 22 | 其实买的很多演讲你呢也我出现某些主题反复讲的现象马云说那重复是为了强调 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0007721307 | 0.2175 | 22 | 其实买的很多演讲你呢你和出现某些主题反复想的现象马云说呢重复是为了强调 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0007721307 | 0.2225 | 22 | 其实买的很多演讲你呢你我出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.1389 |
| 1280 | streaming_asr | emilia_zh_0007721307 | 0.2000 | 23 | 其实买的很多演讲你那你我出现某些主题反复假的现象马云说呢重复是为了强调 | cer | 0.1944 |
| 160 | streaming_asr | emilia_zh_0007761299 | 0.2364 | 18 | 我觉得这可能不社会我我觉得我跟这的公共的图书出的开心 | cer | 0.3929 |
| 320 | streaming_asr | emilia_zh_0007761299 | 0.2509 | 18 | 我觉得这可能不适合我我觉得我跟这的公司的同事出的开心 | cer | 0.2143 |
| 640 | streaming_asr | emilia_zh_0007761299 | 0.2327 | 18 | 我觉得这行业。社会我我觉得我跟这的公的同事出的开心 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0007761299 | 0.2364 | 20 | 我有这个行业。适合我我觉得我跟这的公是的同事出的开心 | cer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0007788542 | 0.1923 | 17 | 不说人开始接触期货觉得比较难理解什么什么期货呢 | cer | 0.2083 |
| 320 | streaming_asr | emilia_zh_0007788542 | 0.1958 | 17 | 不说人一开始接触期货觉得比难理解什么什么期货的 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0007788542 | 0.1923 | 19 | 不是人开始接触期货觉得比较难理解省什么期货的 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0007788542 | 0.1888 | 17 | 不是人开始接触期货觉得比难理解省省期货的 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0007790200 | 0.1376 | 34 | 特别就是了现在是小心私营企业卖厂的最佳时期大型所有实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007790200 | 0.1570 | 33 | 特别就是了现在是小心私营企业卖的最佳时机大型。实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2045 |
| 640 | streaming_asr | emilia_zh_0007790200 | 0.1499 | 32 | 特别就是了现在是小心私营企业卖的最佳时机大型。实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.2045 |
| 1280 | streaming_asr | emilia_zh_0007790200 | 0.1570 | 34 | 特别指了现在是小心私营企业卖的最佳时机大型实力的企业可以考虑像海外现场像海外搬迁 | cer | 0.1818 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
