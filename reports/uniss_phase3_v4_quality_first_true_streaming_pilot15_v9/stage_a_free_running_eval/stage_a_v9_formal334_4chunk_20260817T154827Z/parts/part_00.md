# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8858**
- Weighted CTC blank ratio: **0.1897**
- Weighted streaming WER/CER: **0.4694**
- Weighted causal-full WER/CER: **0.1629**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2511 | 18 | It's two laid now let just yeah this over way is | wer | 0.6000 |
| 320 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2555 | 17 | is to laid now let just a this over way is | wer | 0.7000 |
| 640 | streaming_asr | NCSSD_R_EN_0000000261 | 0.2159 | 20 | is two late now the just a this over way is | wer | 0.6000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.1982 | 19 | is two laid and now that just a this over way is | wer | 0.8000 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.1963 | 8 | That's true. What's about the mentality and that's | wer | 0.7143 |
| 320 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2699 | 10 | Just who's about the natty and that's | wer | 0.8571 |
| 640 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2577 | 11 | That's true. What about the mentality in bats | wer | 0.5714 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.2577 | 13 | That's true. What about the mentality in bats | wer | 0.5714 |
| 160 | streaming_asr | CommonVoice_EN_0000189191 | 0.2043 | 10 | And The boy brought teeth orse boss | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000189191 | 0.2258 | 9 | And The boy brought teeth wharshed whose | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000189191 | 0.2258 | 9 | And The boy brought teeth wharshed wholeser | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000189191 | 0.3011 | 12 | And The boy brought teeth wharshed wholeser The | wer | 0.8333 |
| 160 | streaming_asr | CommonVoice_EN_0000332324 | 0.1478 | 15 | And building post for can what where entirety d destroyed | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000332324 | 0.1652 | 15 | Many building post post can why where entirety d destroyed | wer | 0.8889 |
| 640 | streaming_asr | CommonVoice_EN_0000332324 | 0.1739 | 18 | Many building pose before can what where entirety d destroyed | wer | 0.8889 |
| 1280 | streaming_asr | CommonVoice_EN_0000332324 | 0.1348 | 14 | Many building post before can what were entirety d destroyed | wer | 0.7778 |
| 160 | causal_full_asr | CommonVoice_EN_0000430515 | 0.0982 | 11 | Later years he occasionally put determined lower order batsmen | wer | 0.3636 |
| 320 | causal_full_asr | CommonVoice_EN_0000430515 | 0.1368 | 13 | In later years he occasionally put a determined lower order batsman | wer | 0.0909 |
| 640 | causal_full_asr | CommonVoice_EN_0000430515 | 0.1193 | 11 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000430515 | 0.0982 | 12 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000501889 | 0.1192 | 15 | We is part to motorcycle neet to the building | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000501889 | 0.1917 | 17 | We Usually part to motorcycle nein to the building | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000501889 | 0.1969 | 16 | We is part to motorcycle nein to the building | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000501889 | 0.1813 | 17 | We is part to motorcycle nay to the building | wer | 0.6667 |
| 160 | streaming_asr | DailyTalk_0000001997 | 0.2418 | 10 | The scotch please | wer | 0.5000 |
| 320 | streaming_asr | DailyTalk_0000001997 | 0.2527 | 9 | The scotch please | wer | 0.5000 |
| 640 | streaming_asr | DailyTalk_0000001997 | 0.2747 | 11 | Ding Scott please | wer | 0.7500 |
| 1280 | streaming_asr | DailyTalk_0000001997 | 0.2527 | 10 | Give scotch please | wer | 0.2500 |
| 160 | streaming_asr | LibriSpeech_0000100601 | 0.1437 | 40 | In one insphyxiated jumble Well Tom more off on his three dray my at tension was attracted by man who stirred to little apart looking as if his thus was far away | wer | 0.3750 |
| 320 | streaming_asr | LibriSpeech_0000100601 | 0.1527 | 38 | In one in physiastic jumble Well Tom more off on his three rain my attention was attracted by a man who stirred to little a part looking as if he thought was father way | wer | 0.4688 |
| 640 | streaming_asr | LibriSpeech_0000100601 | 0.1347 | 34 | In one amphoziac. jumble. Well Tom more off on his three dray my attention was attracted by a man who still to the apart looking as if he thus was father away | wer | 0.4062 |
| 1280 | streaming_asr | LibriSpeech_0000100601 | 0.1302 | 36 | In one amphoziac jumble Well Tom more is on his three dray my attention was attracted by a man who still to the apart looking as if he thus was for away | wer | 0.4062 |
| 160 | causal_full_asr | LibriSpeech_0000192309 | 0.1924 | 23 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 320 | causal_full_asr | LibriSpeech_0000192309 | 0.1703 | 22 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 640 | causal_full_asr | LibriSpeech_0000192309 | 0.1451 | 20 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 1280 | causal_full_asr | LibriSpeech_0000192309 | 0.1104 | 17 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 160 | streaming_asr | LibriSpeech_0000238719 | 0.1829 | 28 | Shall tray is the well to don't one if this kind of if thing be <\|glm_semantic_15509\|>to be permitted I may be going to to no old to opper to night | wer | 0.4643 |
| 320 | streaming_asr | LibriSpeech_0000238719 | 0.2043 | 30 | Just tray it the well to don't on if this kind of of thing and permitted I may be going to in a old to opper to night | wer | 0.4643 |
| 640 | streaming_asr | LibriSpeech_0000238719 | 0.1829 | 31 | How prey is the well to don't on if they kind of of thing and permitted I my be gilling to to in a old to the opper to night | wer | 0.4643 |
| 1280 | streaming_asr | LibriSpeech_0000238719 | 0.1686 | 32 | How prey is the well to don't one if the kind of of thing be <\|glm_semantic_13382\|>itted I me be gilling to to in a old to the opper to night | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004002573 | 0.2196 | 32 | 啊哎呦哎呀因为他这这人人喜欢他你的内容然后家的的词然后你换这种那种拍他可能不喜欢这个内容他的会取消 | cer | 0.4333 |
| 320 | streaming_asr | emilia_zh_0004002573 | 0.2434 | 34 | 啊哎呦我要因为他有现在人能喜欢看你的内容然后家的的词然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4500 |
| 640 | streaming_asr | emilia_zh_0004002573 | 0.2649 | 35 | 啊哎呦哎呀因为他有现在人能喜欢看你这个内容然后家的的死然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4500 |
| 1280 | streaming_asr | emilia_zh_0004002573 | 0.2697 | 35 | 啊哎呦哎呀因为他有现在人能喜欢他你这个内容然后家的的词然后你换这种那种他他可能就喜欢你内容这个的会取消 | cer | 0.4667 |
| 160 | causal_full_asr | emilia_zh_0004036374 | 0.2377 | 33 | 所以有关共产主义运动的具体策略和结论的迅速来充当马克思主义哲学发展的内在逻辑迅速 | cer | 0.2250 |
| 320 | causal_full_asr | emilia_zh_0004036374 | 0.2523 | 31 | 所以有关共产主义运动的具体策略和结论的叙述来充当那马克思主义哲学发散的内在逻辑叙述 | cer | 0.1500 |
| 640 | causal_full_asr | emilia_zh_0004036374 | 0.2541 | 36 | 则有关共产主义运动的具体策略和结论的叙述来充当那马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0004036374 | 0.2559 | 37 | 则有关共产主义运动的具体策略和节目的叙述来充当那马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 160 | streaming_asr | emilia_zh_0004130152 | 0.2347 | 17 | 现在说英语的骗的安乐瞬间将变成痛苦例如 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004130152 | 0.2563 | 14 | 现在我拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0500 |
| 640 | streaming_asr | emilia_zh_0004130152 | 0.2238 | 14 | 现在我拥有的片刻的安乐瞬间在变成痛苦例如 | cer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004130152 | 0.2130 | 14 | 现在我拥有的片刻的安乐瞬间家变成痛苦例如 | cer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0004358957 | 0.1720 | 22 | 我听的家粗看几秒钟之后这才以为一张开口我跟鱼的这个 | cer | 0.5517 |
| 320 | streaming_asr | emilia_zh_0004358957 | 0.1892 | 22 | 我听的家租看几秒钟之后这才尾巴一张开口滚到鱼的这个 | cer | 0.5172 |
| 640 | streaming_asr | emilia_zh_0004358957 | 0.1646 | 23 | 我听这家租看几秒钟之后这才以为一张开口问道鱼的这个 | cer | 0.4828 |
| 1280 | streaming_asr | emilia_zh_0004358957 | 0.1646 | 21 | 我听的家租看几秒钟之后这才以为一张开口问道鱼的这个 | cer | 0.4828 |
| 160 | causal_full_asr | emilia_zh_0004665404 | 0.2057 | 24 | If the wine be sweet I would drink it with him and if it be bitter I would drink it with him also was my answer | wer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004665404 | 0.2143 | 27 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004665404 | 0.1886 | 26 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004665404 | 0.1543 | 24 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004692799 | 0.1610 | 46 | So of take in he string <\|glm_semantic_3935\|>ments Instead using the using these warms of a filled to communicate were a the train fam of was the turning the in now to procussion from that's very because of a | wer | 0.7027 |
| 320 | streaming_asr | emilia_zh_0004692799 | 0.1625 | 45 | So of you've taken you stream <\|glm_semantic_3935\|>smith Instead using the using the I these warms of a filled to communicate were a the train fam in is the turn them in now to caution some that's very because of a | wer | 0.7297 |
| 640 | streaming_asr | emilia_zh_0004692799 | 0.1641 | 48 | So of you've taken you stream extrements Instead using using I these warms of a filled to communicate were a the train fam in is the turn them in now to procussion some that's very because of a | wer | 0.7027 |
| 1280 | streaming_asr | emilia_zh_0004692799 | 0.1703 | 49 | So of you've taken you stream existence Instead using using a these warms of rotto filled to communicate were a the tried fam in is the turn them in now to caution from that's very because of a | wer | 0.7027 |
| 160 | streaming_asr | emilia_zh_0004777075 | 0.1420 | 18 | You appear of dink shoulder the sunday Newspaper So what next day Tommy Tuesday | wer | 0.4375 |
| 320 | streaming_asr | emilia_zh_0004777075 | 0.1571 | 21 | You appeared of dink shoulder the sunday Newspaper So what next day comedy Tuesday | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004777075 | 0.1601 | 20 | You peer of dink shoulder the some day Newspaper So what next day Tommy Tuesday | wer | 0.5625 |
| 1280 | streaming_asr | emilia_zh_0004777075 | 0.1601 | 20 | You appeared of dink shoulder the some day Newspaper So what next day comedy Tuesday | wer | 0.6250 |
| 160 | streaming_asr | emilia_zh_0004873848 | 0.1979 | 15 | Margaret say let a with than on this open smile | wer | 0.6667 |
| 320 | streaming_asr | emilia_zh_0004873848 | 0.2240 | 13 | Margaret say let a with than on his open smile | wer | 0.6667 |
| 640 | streaming_asr | emilia_zh_0004873848 | 0.2396 | 14 | Margaret Face let a with then on a open smile | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0004873848 | 0.2135 | 14 | Margaret say let a with then on a open smile | wer | 0.6667 |
| 160 | streaming_asr | emilia_zh_0004999877 | 0.2551 | 34 | 我们就应该尽量一招所所讲的方法去实施这样才会有进步和收下啊 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0004999877 | 0.2247 | 29 | 我们就应该尽量一招佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004999877 | 0.2273 | 26 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0004999877 | 0.2172 | 24 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收信啊 | cer | 0.0714 |
| 160 | causal_full_asr | emilia_zh_0005070340 | 0.2054 | 15 | 曾是了一项名副其实的哲学工作，因为他继续着前人的努力 | cer | 0.1200 |
| 320 | causal_full_asr | emilia_zh_0005070340 | 0.2145 | 18 | 从事了一项名酷其实的哲学工作，因为他继续着前人的努力 | cer | 0.0800 |
| 640 | causal_full_asr | emilia_zh_0005070340 | 0.2326 | 17 | 从事了一下名酷其实的哲学工作因为他继续着前人的努力 | cer | 0.0800 |
| 1280 | causal_full_asr | emilia_zh_0005070340 | 0.2326 | 18 | 从事了一下名酷其实的哲学工作因为他继续着前人的努力 | cer | 0.0800 |
| 160 | streaming_asr | emilia_zh_0005347772 | 0.2700 | 17 | 另外民主党一部走了此处据民主党来说他们面临就困难的选择 | cer | 0.3000 |
| 320 | streaming_asr | emilia_zh_0005347772 | 0.2700 | 20 | 另外民主党一波走了此处据民主党来说他们面临这困难的选择 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0005347772 | 0.2852 | 20 | 令民主党一拨走了死胡同据民主党来说他们面临这困难的选择 | cer | 0.2667 |
| 1280 | streaming_asr | emilia_zh_0005347772 | 0.2510 | 17 | 零民主党一拨走了死胡同据民主党来说他们面临这困难的选择 | cer | 0.2667 |
| 160 | streaming_asr | emilia_zh_0005578734 | 0.1135 | 8 | 给为一些你你能给陪伴给也发能给爱给爱 | cer | 0.2105 |
| 320 | streaming_asr | emilia_zh_0005578734 | 0.1081 | 10 | 就给为一些你你能给陪伴给办法能给爱给爱 | cer | 0.2105 |
| 640 | streaming_asr | emilia_zh_0005578734 | 0.1027 | 10 | 就给为一些你你能给陪伴给办法能给爱给爱 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0005578734 | 0.1081 | 11 | 就给有一些你你能给陪伴给吧能给啊给爱 | cer | 0.2632 |
| 160 | causal_full_asr | emilia_zh_0005780397 | 0.2148 | 47 | 工作对一定是最低的然后工作一定是消耗的所以再前两天呢就是我也做一丈嘛然后就跟一些小黄让做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.1538 |
| 320 | causal_full_asr | emilia_zh_0005780397 | 0.2148 | 49 | 工作对一定是对地的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友啊做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 640 | causal_full_asr | emilia_zh_0005780397 | 0.2199 | 50 | 工作对一定是对地的然后工作一定是消耗的所以在前两天呢就是我也做一件嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 1280 | causal_full_asr | emilia_zh_0005780397 | 0.2096 | 47 | 工作这一定是最低的然后工作一定是消耗的所以在前两天呢就是我也做HR然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0615 |
| 160 | streaming_asr | emilia_zh_0005818215 | 0.1757 | 24 | 就打去去情绪切还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.3548 |
| 320 | streaming_asr | emilia_zh_0005818215 | 0.1980 | 23 | 就大请确实情绪去啊还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 640 | streaming_asr | emilia_zh_0005818215 | 0.2104 | 29 | 就大请确实情绪期间还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0005818215 | 0.1955 | 26 | 就大请确实情绪去啊还比较低然后在加上我们啊最近都挺多事儿 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0006056256 | 0.1689 | 18 | 在在<\|write_generate\|><\|cmn\|><\|start_content\|>这个情况特别容易爱情因为觉得完全 | cer | 2.3182 |
| 320 | streaming_asr | emilia_zh_0006056256 | 0.1872 | 18 | 在啊的这种情况特别容易行用觉得完全 | cer | 0.4091 |
| 640 | streaming_asr | emilia_zh_0006056256 | 0.2374 | 21 | 啊啊的这种情况特别容易行因为觉得完全 | cer | 0.4091 |
| 1280 | streaming_asr | emilia_zh_0006056256 | 0.2329 | 18 | 啊啊的这种情况就特别容易爱情因为觉得完全 | cer | 0.3636 |
| 160 | causal_full_asr | emilia_zh_0006174175 | 0.2227 | 34 | 我正好来就记得老师在画画头头的讨论什么那大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.3478 |
| 320 | causal_full_asr | emilia_zh_0006174175 | 0.2133 | 34 | 我正好来就挤到老师在一块偷偷的讨论什么了大家就是老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.3043 |
| 640 | causal_full_asr | emilia_zh_0006174175 | 0.2251 | 34 | 我之后来就记得老师在一会儿偷偷的讨论什么了大家就是老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.2826 |
| 1280 | causal_full_asr | emilia_zh_0006174175 | 0.2133 | 34 | 我然后来就记得老师在一块偷偷的讨论什么呢大家其实老师都没有人说特别明确知道这个事儿算是吗 | cer | 0.2391 |
| 160 | streaming_asr | emilia_zh_0006303057 | 0.1626 | 49 | 一人吧就是以前一个无限挑战的我他们是一起的然后内有限偶然有一期干什么呢航空速擦这万全的玻璃<\|write_generate\|><\|cmn\|><\|start_content\|>这妈妈外圈的玻璃做生降级在外面其实在那个窗户外面 | cer | 0.8919 |
| 320 | streaming_asr | emilia_zh_0006303057 | 0.1563 | 46 | 一人吧就是以前一个无限挑战的他们是一起你的然后内有限责任有一期干什么呢旁名素擦这万全玻璃你这妈妈外圈的玻璃做生计在外面就是在那个窗户外面 | cer | 0.3378 |
| 640 | streaming_asr | emilia_zh_0006303057 | 0.1616 | 49 | 艺人吧就是以前一个无限挑战的我他们是一起你然后内路线扰乱有一期干什么呢旁名素擦这外圈玻璃你这妈妈外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.2973 |
| 1280 | streaming_asr | emilia_zh_0006303057 | 0.1626 | 47 | 一人我就是以前一个无限挑战的的他们是一起你然后那个无限偶然有一起干什么呢旁名素擦这个外圈玻璃<\|write_generate\|><\|cmn\|><\|start_content\|>这妈妈外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.8378 |
| 160 | streaming_asr | emilia_zh_0006379861 | 0.2138 | 25 | We the Professor when into the cell he had one find all real in to ten all are bills | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006379861 | 0.1931 | 23 | I in Professor when into the cell he had one five all real in to ten all are bills | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0006379861 | 0.1759 | 20 | I in the Professor when int the cell he had one five a a bill in to ten all re bills | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0006379861 | 0.1724 | 21 | I the Professor when int the cell he had one five out real in to ten all re bills | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0006464935 | 0.1719 | 30 | That Praise point where quantity the consumers want but I equal the quantity the cellar want produce is cold the each liberty breed | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0006464935 | 0.1929 | 33 | That price the point point liquidity the consumers want but I equal the quantity the cellar want produce is cold the each liberty breed | wer | 0.5769 |
| 640 | streaming_asr | emilia_zh_0006464935 | 0.1824 | 32 | That price the point point quality the consumers want but I equal the quantity the cellar want produce is cold the each liberty prey | wer | 0.5769 |
| 1280 | streaming_asr | emilia_zh_0006464935 | 0.1656 | 30 | That price the point point ac quantity the consumers want but I equal the quantity the cellar want produce is cold the equal liberty prey | wer | 0.5385 |
| 160 | causal_full_asr | emilia_zh_0006659550 | 0.2885 | 16 | 中期的数据看，已经有例这个不时啊他们能做出两次错的判断 | cer | 0.4138 |
| 320 | causal_full_asr | emilia_zh_0006659550 | 0.2727 | 16 | 中期的数据看起来有利益这个不时他能做出两次错的判断 | cer | 0.3448 |
| 640 | causal_full_asr | emilia_zh_0006659550 | 0.2806 | 18 | 中期的数据看就有利于这个布什让他能做出两次错误的判断 | cer | 0.2414 |
| 1280 | causal_full_asr | emilia_zh_0006659550 | 0.2846 | 17 | 中期的数据看及有利这个不时难道还能做出两次错误的判断 | cer | 0.3448 |
| 160 | streaming_asr | emilia_zh_0006714517 | 0.1607 | 13 | 好接下来我我讲最重要的今天那间事情呢啊 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0006714517 | 0.1786 | 12 | 好接下来我我讲最重要今天那间事情呢啊 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0006714517 | 0.1667 | 11 | 好接下来我啊讲最重要今天那件事情呢啊 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0006714517 | 0.1607 | 11 | 好接下来我讲最重要今天那件事情了啊 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0006990963 | 0.1652 | 14 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 320 | streaming_asr | emilia_zh_0006990963 | 0.1652 | 14 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 640 | streaming_asr | emilia_zh_0006990963 | 0.1741 | 15 | 现在正在是春天我们一会儿就去愿离 | cer | 0.2353 |
| 1280 | streaming_asr | emilia_zh_0006990963 | 0.1741 | 16 | 现在正在是春天我们一会儿就去院子离 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007169925 | 0.1077 | 29 | 啊犹太人似乎在这方面的要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何时间思考 | cer | 0.1190 |
| 320 | streaming_asr | emilia_zh_0007169925 | 0.1105 | 29 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩但他们也会利用任何实际思考 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007169925 | 0.1161 | 29 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何实际思考 | cer | 0.1190 |
| 1280 | streaming_asr | emilia_zh_0007169925 | 0.1035 | 31 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是抗辩的他们也会利用任何实际思考 | cer | 0.1190 |
| 160 | streaming_asr | emilia_zh_0007462823 | 0.1336 | 12 | 因为他说到野餐四的图片a.是挑个尼克的 | cer | 0.6429 |
| 320 | streaming_asr | emilia_zh_0007462823 | 0.1382 | 12 | 因为他说到野餐所以的图片a.是挑个尼克的的 | cer | 0.5714 |
| 640 | streaming_asr | emilia_zh_0007462823 | 0.1382 | 11 | 因为他说到野餐所以的图片a.是pick a nick的 | cer | 0.4643 |
| 1280 | streaming_asr | emilia_zh_0007462823 | 0.1244 | 11 | 因为他说到野餐所以的图片a.是pick a nick的 | cer | 0.4643 |
| 160 | streaming_asr | emilia_zh_0007691054 | 0.1822 | 27 | 但从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标原你仅仅啊 | cer | 0.3514 |
| 320 | streaming_asr | emilia_zh_0007691054 | 0.1893 | 29 | 吧从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标远远仅仅啊 | cer | 0.3784 |
| 640 | streaming_asr | emilia_zh_0007691054 | 0.1939 | 31 | 吧从这个角度来讲我们没而已是一个比定律的增速更好更加重要指标远远仅啊 | cer | 0.3784 |
| 1280 | streaming_asr | emilia_zh_0007691054 | 0.1986 | 33 | 了从这个角度来讲我们没而已是一个比经历的增速更好更加重要指标远远仅仅啊 | cer | 0.3784 |
| 160 | streaming_asr | EN_B00097_S02875_W000006 | 0.2553 | 47 | But That for she you cross you like because we dong want who receives from the lord a of of body We one or seieve Anything also do in was one seieve the a but but to all about | wer | 0.7179 |
| 320 | streaming_asr | EN_B00097_S02875_W000006 | 0.2979 | 53 | But That for this you cross you like because we don't want who recieve from the lord a of of body We one or seieve Anything all to do in was one severe and a but but of or about | wer | 0.6923 |
| 640 | streaming_asr | EN_B00097_S02875_W000006 | 0.2819 | 49 | But the for this you grows you like because we don't want who recieve from the lord but of of body We one or seieve Anything all the do in with one seieve and a but but of all about | wer | 0.7179 |
| 1280 | streaming_asr | EN_B00097_S02875_W000006 | 0.2535 | 52 | But the for she you cross you like because we don't want who recieve from the lord but of of body We one or seieve Anything all the do in was one rescue and a up but of or about | wer | 0.6923 |
| 160 | causal_full_asr | EN_B00052_S08813_W000015 | 0.2038 | 42 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans a cooperate and compromise | wer | 0.0385 |
| 320 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1942 | 40 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1808 | 42 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00052_S08813_W000015 | 0.1635 | 40 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S06429_W000060 | 0.1414 | 21 | And A lot of believed at part lessly lose her be very give a to to with all love the date that was how there | wer | 0.7143 |
| 320 | streaming_asr | EN_B00058_S06429_W000060 | 0.1382 | 21 | the Along the believed at part loser be very give a to to with of love the date that was how there | wer | 0.6190 |
| 640 | streaming_asr | EN_B00058_S06429_W000060 | 0.1875 | 24 | the Along the believed that part loser be very give a to to with of love deed that was how there | wer | 0.5714 |
| 1280 | streaming_asr | EN_B00058_S06429_W000060 | 0.1612 | 23 | The Alliance believed a part loser be very give a to to with of love the date that was how there | wer | 0.5238 |
| 160 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1593 | 12 | 温达你从饼干罐里偷了饼干 | wer | 1.0000 |
| 320 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1504 | 9 | 温达，你从饼干罐里偷了饼干 | wer | 1.0000 |
| 640 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1726 | 11 | Linda, you stole the cookies from the cookie jar. | wer | 0.2222 |
| 1280 | causal_full_asr | EN_B00048_S01182_W000001 | 0.1283 | 7 | Linda, you stole the cookies from the cookie jar. | wer | 0.2222 |
| 160 | streaming_asr | EN_B00048_S05933_W000060 | 0.2833 | 12 | That American has really interesting culture that with fed | wer | 0.5000 |
| 320 | streaming_asr | EN_B00048_S05933_W000060 | 0.2611 | 13 | So American has really interesting culture that with feder | wer | 0.5000 |
| 640 | streaming_asr | EN_B00048_S05933_W000060 | 0.1778 | 10 | So America has really interesting culture that with faith | wer | 0.4000 |
| 1280 | streaming_asr | EN_B00048_S05933_W000060 | 0.1667 | 12 | South merck has really interesting culture that that faith | wer | 0.4000 |
| 160 | streaming_asr | EN_B00058_S03144_W000037 | 0.2143 | 19 | Long compets Sentences with multiples paragraphs a email | wer | 0.4444 |
| 320 | streaming_asr | EN_B00058_S03144_W000037 | 0.1830 | 16 | long Complex Sentences with Multiple paragraphs a email | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S03144_W000037 | 0.1473 | 16 | long Complex Sentences with modifiable paragraphs a email | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00058_S03144_W000037 | 0.1562 | 16 | long Complex Sentences with modifiable paragraphs a email | wer | 0.3333 |
| 160 | streaming_asr | EN_B00091_S07092_W000002 | 0.2258 | 19 | And This come to of what like to current on old of they and your in the and of sorry, turn. sigger | wer | 1.0526 |
| 320 | streaming_asr | EN_B00091_S07092_W000002 | 0.2500 | 21 | But the concern of more like to care on a of down there and your in the and of sorry, turn around. figure | wer | 0.9474 |
| 640 | streaming_asr | EN_B00091_S07092_W000002 | 0.2016 | 20 | But to comes of more like to care on a of down there and your in the and of thorothy returned. figure | wer | 0.8421 |
| 1280 | streaming_asr | EN_B00091_S07092_W000002 | 0.1976 | 20 | But to comes of more like a care on a of down there and the in the and of thoritary figure | wer | 0.7368 |
| 160 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1784 | 17 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1878 | 15 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1315 | 14 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00036_S05339_W000016 | 0.1033 | 13 | To use the present perfect correctly you need to know things like | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
