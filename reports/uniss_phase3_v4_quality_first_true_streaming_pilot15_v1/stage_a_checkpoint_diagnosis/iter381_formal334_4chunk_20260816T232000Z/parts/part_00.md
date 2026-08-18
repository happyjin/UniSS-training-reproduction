# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9245**
- Weighted CTC blank ratio: **0.8769**
- Weighted streaming WER/CER: **0.2661**
- Weighted causal-full WER/CER: **0.1282**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9339 | 9 | It's too late now that staff gather this over way | wer | 0.4000 |
| 320 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9515 | 7 | As too late now that just get this over way | wer | 0.3000 |
| 640 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9251 | 9 | As too late now that just get this over way | wer | 0.3000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.9163 | 11 | As too late now that just get this over way this | wer | 0.4000 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9693 | 3 | That's true What's in八字 and that's | wer | 0.7143 |
| 320 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9632 | 2 | Just two what about the netting impacts | wer | 0.5714 |
| 640 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.9080 | 10 | That's true What about the magnetic impacts | wer | 0.2857 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.8957 | 11 | That's true What about the magnetic impacts | wer | 0.2857 |
| 160 | streaming_asr | CommonVoice_EN_0000189191 | 0.9462 | 6 | The The boy brought his orse boss | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000189191 | 0.9247 | 6 | The The boy brought tis whorse boss | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000189191 | 0.8978 | 10 | The The boy brought tis whorse trollor | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000189191 | 0.8978 | 9 | The The boy brought tis whorse tulfs are | wer | 0.8333 |
| 160 | streaming_asr | CommonVoice_EN_0000332324 | 0.7783 | 31 | Very building push for can word were intality destroyed | wer | 0.7778 |
| 320 | streaming_asr | CommonVoice_EN_0000332324 | 0.7522 | 31 | Many building spose for can wide were intality destroyed | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000332324 | 0.7478 | 30 | Many building was per can would were intality destroyed | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000332324 | 0.7348 | 31 | Many building both for can word word intirely destroyed | wer | 0.6667 |
| 160 | causal_full_asr | CommonVoice_EN_0000430515 | 0.8491 | 27 | Later years he occasionally put determined lower order batsmen | wer | 0.3636 |
| 320 | causal_full_asr | CommonVoice_EN_0000430515 | 0.8035 | 33 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000430515 | 0.8000 | 36 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7684 | 36 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000501889 | 0.8187 | 20 | We usually part to motorcycle near to the building | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000501889 | 0.8083 | 23 | We usually part to motorcycle near to the building | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000501889 | 0.8083 | 22 | We usually part to motorcycle near to the building | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000501889 | 0.8290 | 20 | We usually part to motorcycle near to the building | wer | 0.4444 |
| 160 | streaming_asr | DailyTalk_0000001997 | 0.9451 | 4 | In Scott Please | wer | 0.7500 |
| 320 | streaming_asr | DailyTalk_0000001997 | 0.9451 | 4 | The Scott Please | wer | 0.7500 |
| 640 | streaming_asr | DailyTalk_0000001997 | 0.9670 | 2 | Give Scott please | wer | 0.5000 |
| 1280 | streaming_asr | DailyTalk_0000001997 | 0.9451 | 4 | Give Scott please | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000100601 | 0.8234 | 54 | In one insphyxiously jumble While Tom was off on his fifth rate my attention was attracted by man who stood a little apart looking as if his thought were far way | wer | 0.1875 |
| 320 | streaming_asr | LibriSpeech_0000100601 | 0.7949 | 66 | In one insphyxiously jumble While Tom was off on his first rate my attention was attracted by man who stood a little Apart looking as if his thought were far way | wer | 0.1875 |
| 640 | streaming_asr | LibriSpeech_0000100601 | 0.7949 | 66 | In one influenced jumble While Tom was off on his third rate my attention was attracted by man who stood a little apart looking as if his thought were far way | wer | 0.1562 |
| 1280 | streaming_asr | LibriSpeech_0000100601 | 0.8024 | 60 | In one influenced jumble While Tom was off on his third rate my attention was attracted by a man who stood a little Apart looking as if his thought were far way | wer | 0.1250 |
| 160 | causal_full_asr | LibriSpeech_0000192309 | 0.8265 | 30 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 320 | causal_full_asr | LibriSpeech_0000192309 | 0.8265 | 29 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 640 | causal_full_asr | LibriSpeech_0000192309 | 0.8076 | 32 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 1280 | causal_full_asr | LibriSpeech_0000192309 | 0.8265 | 31 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 160 | streaming_asr | LibriSpeech_0000238719 | 0.8432 | 41 | Shall play is the world to don't one if this kind of if thing be permitted I may be going out to don't all to the operation to night | wer | 0.2857 |
| 320 | streaming_asr | LibriSpeech_0000238719 | 0.8385 | 41 | Charles play is the well to don't one if this kind if thing be permitted I may be going out to to know all to the opper to night | wer | 0.3571 |
| 640 | streaming_asr | LibriSpeech_0000238719 | 0.8052 | 48 | How pray is the well to don't on if this kind of of thing be permitted I may be going out to to all to the opper to night | wer | 0.2143 |
| 1280 | streaming_asr | LibriSpeech_0000238719 | 0.7886 | 53 | How pray is the well to don't one if this kind of if thing be permitted I may be going out to don't all to the opper to night | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004002573 | 0.8687 | 42 | 啊我认为要你会他有现在人可以能喜欢看你那个那种然后的家的分粉丝然后你换那种那种拍他可能就不喜欢这个那种他的会取消 | cer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0004002573 | 0.8687 | 46 | 啊哎呦喂哎呀因为他有现在人可以能喜欢看你那个那种然后的家的分粉丝然后你换那种那种拍他可能就不喜欢这个那种他那种会取消 | cer | 0.4167 |
| 640 | streaming_asr | emilia_zh_0004002573 | 0.8759 | 45 | 啊因为要因为他有现在人可以能喜欢看你那个那种然后的家的分粉丝然后你换那种那种拍他可能就不喜欢这个那种他那种会取消 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004002573 | 0.8759 | 46 | 啊因为要因为他有现在人可以能喜欢看你那个内容然后的家的分粉丝然后你换那种那种拍他可能就不喜欢这个内容他的会取消 | cer | 0.2667 |
| 160 | causal_full_asr | emilia_zh_0004036374 | 0.9837 | 6 | 德有关共产主义运动的具体策略和节目的叙述来充当马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 320 | causal_full_asr | emilia_zh_0004036374 | 0.9837 | 7 | 则有关共产主义运动的具体策略和节目的叙述来充当的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 640 | causal_full_asr | emilia_zh_0004036374 | 0.9837 | 7 | 则有关共产主义运动的具体策略和节目的叙述来充当的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 1280 | causal_full_asr | emilia_zh_0004036374 | 0.9837 | 7 | 则有关共产主义运动的具体策略和节目的叙述来充当的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 160 | streaming_asr | emilia_zh_0004130152 | 0.9856 | 4 | 现在所拥有的片刻的安乐瞬间讲变成痛苦例如 | cer | 0.0500 |
| 320 | streaming_asr | emilia_zh_0004130152 | 0.9819 | 4 | 现在做拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0500 |
| 640 | streaming_asr | emilia_zh_0004130152 | 0.9819 | 5 | 现在做拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0500 |
| 1280 | streaming_asr | emilia_zh_0004130152 | 0.9747 | 6 | 现在做拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0500 |
| 160 | streaming_asr | emilia_zh_0004358957 | 0.9754 | 9 | 我定的架子速度看几秒钟之后这才有一波一张开口我问道鱼什么这个 | cer | 0.4828 |
| 320 | streaming_asr | emilia_zh_0004358957 | 0.9754 | 8 | 我听这鸭子速度看了几秒钟之后这才有一波一张开口问道鱼我们这个 | cer | 0.4138 |
| 640 | streaming_asr | emilia_zh_0004358957 | 0.9779 | 8 | 我钉这鸭子足足看了几秒钟之后这才嘴巴一张开口问道鱼怎么这个 | cer | 0.1724 |
| 1280 | streaming_asr | emilia_zh_0004358957 | 0.9803 | 7 | 我盯着鸭子足足看了几秒钟之后这才有一波一张开口问道鱼怎么这个 | cer | 0.2069 |
| 160 | causal_full_asr | emilia_zh_0004665404 | 0.7400 | 49 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004665404 | 0.7514 | 50 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004665404 | 0.7257 | 48 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004665404 | 0.7114 | 51 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004692799 | 0.7663 | 84 | So we we've taken these string instruments instead of using the some as these warms of rotto filled communicators were in the drink fem we are of triumen in now to procution <\|glm_semantic_1457\|>imet that's a very precautive effect | wer | 0.4054 |
| 320 | streaming_asr | emilia_zh_0004692799 | 0.7554 | 88 | So we we've taken these string <\|glm_semantic_2294\|><\|glm_semantic_2294\|><\|glm_semantic_2294\|>instruments instead of using the some as these warms of rotto filled communicators were in the during fam were are of turn them in now to procution <\|glm_semantic_1457\|>imation that's a very precaution effect | wer | 0.4054 |
| 640 | streaming_asr | emilia_zh_0004692799 | 0.7167 | 101 | So all we've taken these string instruments instead using the the as these warms of rotto filled communicators were in the during family were start of trium them in now to procution instrumentation that's a very procussive effect | wer | 0.3784 |
| 1280 | streaming_asr | emilia_zh_0004692799 | 0.7307 | 97 | So all we've taken these string instruments instead of using the some as these warms of brotto filled communicators were in the during family were start of trium them in now to procution instrumentation that's a very procussive effect | wer | 0.3514 |
| 160 | streaming_asr | emilia_zh_0004777075 | 0.8127 | 32 | He appeared over dink shoulder at the sunday newspaper So what's next day comedy Tuesday | wer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004777075 | 0.8006 | 32 | He appeared over dink shoulder at the sunday newspaper So what's next day Kami tuesday | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0004777075 | 0.7674 | 40 | He performed over dink shoulder at the sunday newspaper So what's next day Tommy tuesday | wer | 0.1875 |
| 1280 | streaming_asr | emilia_zh_0004777075 | 0.7704 | 38 | He performed over dink shoulder at the sunday newspaper So what's next day Tommy tuesday | wer | 0.1875 |
| 160 | streaming_asr | emilia_zh_0004873848 | 0.8646 | 15 | Margaret face little out with the onest open smile | wer | 0.4444 |
| 320 | streaming_asr | emilia_zh_0004873848 | 0.8438 | 18 | Margaret face little a with the onest open smile | wer | 0.4444 |
| 640 | streaming_asr | emilia_zh_0004873848 | 0.8385 | 17 | Margaret face little art with the onest open smile | wer | 0.4444 |
| 1280 | streaming_asr | emilia_zh_0004873848 | 0.8385 | 16 | Margaret face little a with the onest open smile | wer | 0.4444 |
| 160 | streaming_asr | emilia_zh_0004999877 | 0.9924 | 3 | 我们就应该尽量一招摸索讲的方法去实施这样才会有进步和收效啊 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0004999877 | 0.9924 | 3 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效啊 | cer | 0.0357 |
| 640 | streaming_asr | emilia_zh_0004999877 | 0.9823 | 6 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效啊 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0004999877 | 0.9848 | 5 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效啊 | cer | 0.0357 |
| 160 | causal_full_asr | emilia_zh_0005070340 | 0.9849 | 4 | 曾设了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.1200 |
| 320 | causal_full_asr | emilia_zh_0005070340 | 0.9789 | 5 | 从事了一项名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005070340 | 0.9819 | 5 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 1280 | causal_full_asr | emilia_zh_0005070340 | 0.9789 | 5 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 160 | streaming_asr | emilia_zh_0005347772 | 0.9582 | 10 | 另外民主党被迫走了似乎对于民主党来说他们面临这个困难的选择 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0005347772 | 0.9506 | 12 | 另外民主党一波走进了死胡同对于民主党来说他们面临这个困难的选择 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0005347772 | 0.9582 | 10 | 另外民主党一波走进了似乎对于民主党来说他们面临这困难的选择 | cer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0005347772 | 0.9620 | 9 | 另外民主党一波走进了似乎对于民主党来说他们面临这困难的选择 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0005578734 | 0.9946 | 1 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 320 | streaming_asr | emilia_zh_0005578734 | 0.9730 | 5 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0005578734 | 0.9730 | 5 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0005578734 | 0.9730 | 4 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 160 | causal_full_asr | emilia_zh_0005780397 | 0.9605 | 19 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做一下嘛然后就跟一些小胖儿然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0005780397 | 0.9536 | 24 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做一下嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 640 | causal_full_asr | emilia_zh_0005780397 | 0.9450 | 27 | 工作对一定是最低的然后工作一定是消耗的所以在前两天呢就是我也做一下嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0769 |
| 1280 | causal_full_asr | emilia_zh_0005780397 | 0.9467 | 25 | 工作就一定是最低的然后工作一定是消耗的所以再前两天呢就是我也做HR嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 160 | streaming_asr | emilia_zh_0005818215 | 0.9975 | 1 | 觉得打你确实情绪气还比较低然后在加上我们要啊最近都挺多事儿 | cer | 0.2903 |
| 320 | streaming_asr | emilia_zh_0005818215 | 1.0000 | 0 | 觉得大家你确实情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.2581 |
| 640 | streaming_asr | emilia_zh_0005818215 | 0.9975 | 1 | 就大家请确实情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.1935 |
| 1280 | streaming_asr | emilia_zh_0005818215 | 0.9926 | 3 | 就打警确实情绪气呀还比较低然后在加上我们要然后最近都挺多事儿 | cer | 0.2903 |
| 160 | streaming_asr | emilia_zh_0006056256 | 0.9726 | 5 | 在这这这种情境的特别容易发行因为觉得完全 | cer | 0.4545 |
| 320 | streaming_asr | emilia_zh_0006056256 | 0.9589 | 7 | 了万就就这种情境的特别容易发行因为觉得完全 | cer | 0.4091 |
| 640 | streaming_asr | emilia_zh_0006056256 | 0.9680 | 5 | 了问啊就就这种情境的特别容易发行因为觉得完全 | cer | 0.4091 |
| 1280 | streaming_asr | emilia_zh_0006056256 | 0.9498 | 9 | 了问就去这种形象的特别容易发行因为觉得完全 | cer | 0.4545 |
| 160 | causal_full_asr | emilia_zh_0006174175 | 0.9645 | 13 | 我正好后来就几个老师在一块偷偷的讨论什么呢大家其实老师都没有人说特别明确知道这个事儿算什么 | cer | 0.1739 |
| 320 | causal_full_asr | emilia_zh_0006174175 | 0.9573 | 14 | 我一然后来就挤得老实实在一块偷偷的讨论什么大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.3043 |
| 640 | causal_full_asr | emilia_zh_0006174175 | 0.9550 | 14 | 你正好来就挤得老叔在一块偷偷的讨论什么大家就是老叔都没有人说特别明确知道这个事儿算什么 | cer | 0.2826 |
| 1280 | causal_full_asr | emilia_zh_0006174175 | 0.9597 | 13 | 我一然后来就挤得老叔在一块偷偷的讨论什么呢大家其实老叔都没有人说特别明确知道这个事儿算什么 | cer | 0.2609 |
| 160 | streaming_asr | emilia_zh_0006303057 | 0.9685 | 25 | 一人吧就是以前那个无限挑战的说他们是一起的然后那个现挑战有一期干什么呢防兵速擦这个万全的不理你着吗外圈的不离做生降级在外面就是在那个窗户外面 | cer | 0.2703 |
| 320 | streaming_asr | emilia_zh_0006303057 | 0.9633 | 27 | 一人吧就是以前那个无限挑战的时候他们是一起的然后那个现挑战有一期干什么呢旁目速擦这个万全的玻璃你这吗外圈的玻璃做生相机在外面就是在那个窗户外面 | cer | 0.1892 |
| 640 | streaming_asr | emilia_zh_0006303057 | 0.9496 | 33 | 一人吧就是以前的那个无限挑战的时候他们是一起的然后那个现讨论有一期干什么呢防兵速他这个外圈的玻璃这吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.2297 |
| 1280 | streaming_asr | emilia_zh_0006303057 | 0.9507 | 32 | 一人吧就是以前的那个无限挑战的时候他们是一起的然后那个现讨论有一期干什么呢防兵速擦这个万全的不理这吗外圈的玻璃坐着生相机在外面就是在那个窗户外面 | cer | 0.2432 |
| 160 | streaming_asr | emilia_zh_0006379861 | 0.7759 | 37 | We the Professor went into the cell he had one fight all bill and two ten all bills | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0006379861 | 0.7655 | 35 | When the Professor went into the cell he had one five all bill and two ten all bills | wer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0006379861 | 0.7414 | 42 | When the Professor went into the cell he had one five dull bill and two ten all bills | wer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0006379861 | 0.7655 | 34 | When the Professor went into the cell he had one five all bill and two ten all bills | wer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006464935 | 0.7463 | 64 | That price point where the quantity that consumers want about i equal the quantity that cellars want produce is called the equal liberty price | wer | 0.3462 |
| 320 | streaming_asr | emilia_zh_0006464935 | 0.7400 | 67 | That price the point where the quality that consumers want about by equal the quantity that cellars want produce is called the equal liberty price | wer | 0.3077 |
| 640 | streaming_asr | emilia_zh_0006464935 | 0.7212 | 67 | That price the point were the quantity that consumers want about by equal the quantity that cellars want produce is called the equal librarian pray | wer | 0.3846 |
| 1280 | streaming_asr | emilia_zh_0006464935 | 0.7212 | 67 | That price the point real quality the consumers want about i equal the quantity that cellars want produce is called the equal librarian pray | wer | 0.4615 |
| 160 | causal_full_asr | emilia_zh_0006659550 | 0.9565 | 8 | 中期的数据看其有利于这个故事他们能做出两次错的判断 | cer | 0.3103 |
| 320 | causal_full_asr | emilia_zh_0006659550 | 0.9486 | 11 | 中西的书籍看起来有利益这个不实那他们都说两次错的判断 | cer | 0.4138 |
| 640 | causal_full_asr | emilia_zh_0006659550 | 0.9289 | 15 | 中期的数据看其有利于这个不时哪怕能做出两次错的判断 | cer | 0.3448 |
| 1280 | causal_full_asr | emilia_zh_0006659550 | 0.9486 | 10 | 中期的数据显示有利于这个不时哪怕能做出两次错误的判断 | cer | 0.3448 |
| 160 | streaming_asr | emilia_zh_0006714517 | 0.9643 | 5 | 好接下来我玩讲最重要的今天那间事情呢啊 | cer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0006714517 | 0.9643 | 6 | 好接下来我我要讲最重要的今天那间事情呢啊 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0006714517 | 0.9643 | 6 | 好接下来我我要讲最重要的今天那件事情了啊 | cer | 0.1500 |
| 1280 | streaming_asr | emilia_zh_0006714517 | 0.9583 | 6 | 好接下来我要讲最重要的今天那件事情呢啊 | cer | 0.1500 |
| 160 | streaming_asr | emilia_zh_0006990963 | 0.9955 | 1 | 现在正好是春天我们一会儿就去院子离 | cer | 0.0588 |
| 320 | streaming_asr | emilia_zh_0006990963 | 0.9955 | 1 | 现在正好是春天我们一会儿就去院子离 | cer | 0.0588 |
| 640 | streaming_asr | emilia_zh_0006990963 | 0.9955 | 1 | 现在正好是春天我们一会儿就去院子离 | cer | 0.0588 |
| 1280 | streaming_asr | emilia_zh_0006990963 | 0.9911 | 2 | 现在正好是春天我们一会儿就去院子离 | cer | 0.0588 |
| 160 | streaming_asr | emilia_zh_0007169925 | 0.9986 | 1 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 320 | streaming_asr | emilia_zh_0007169925 | 0.9944 | 2 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 640 | streaming_asr | emilia_zh_0007169925 | 0.9930 | 3 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 1280 | streaming_asr | emilia_zh_0007169925 | 0.9944 | 2 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0007462823 | 0.9355 | 10 | 因为他说到picnic所以的图片a是pickin'的 | cer | 0.2143 |
| 320 | streaming_asr | emilia_zh_0007462823 | 0.9355 | 12 | 因为他说到picnic所以的图片a是pick in the的 | cer | 0.2857 |
| 640 | streaming_asr | emilia_zh_0007462823 | 0.9309 | 11 | 因为他说到picnic所以的图片a是pick a的 | cer | 0.2143 |
| 1280 | streaming_asr | emilia_zh_0007462823 | 0.9447 | 11 | 因为他说到picnic所以的图片a是pick a make的 | cer | 0.3214 |
| 160 | streaming_asr | emilia_zh_0007691054 | 0.9790 | 6 | 吧从这个角度来讲我们没而已是一个比近距离增速更好更加重要的指标人很简单的 | cer | 0.2973 |
| 320 | streaming_asr | emilia_zh_0007691054 | 0.9743 | 7 | 吧从这个角度来讲我们没而已水一个比定律论增速更好更加重要的指标原很简单的 | cer | 0.2973 |
| 640 | streaming_asr | emilia_zh_0007691054 | 0.9766 | 6 | 吧从这个角度来讲我们没而已使一个比定律论增速更好更加重要的指标原和简单呢 | cer | 0.3243 |
| 1280 | streaming_asr | emilia_zh_0007691054 | 0.9766 | 8 | 那从这个角度来讲我们没而已是一个比定律论增速更好更加重要指标原很仅仅呢 | cer | 0.3243 |
| 160 | streaming_asr | EN_B00097_S02875_W000006 | 0.7855 | 76 | But the for stiny is cross should <\|glm_semantic_14693\|> because we don't want to will receive from the Lord part of of body we want are seieve Anything positive means we want receives and the up part of or body | wer | 0.3846 |
| 320 | streaming_asr | EN_B00097_S02875_W000006 | 0.7713 | 77 | But the for stint is cross should <\|glm_semantic_14693\|> because we don't want to will receive from the Lord part of of body we want are seieve Anything positive means we want to sieve some the up a part of or body | wer | 0.3846 |
| 640 | streaming_asr | EN_B00097_S02875_W000006 | 0.7855 | 70 | But the for staying is cross should lex because we don't want to will receive from the Lord part of of body we want are seieve Anything positive means we want to sieve some the up part of or body | wer | 0.3590 |
| 1280 | streaming_asr | EN_B00097_S02875_W000006 | 0.7660 | 75 | But the first staying is cross you legs because we don't want to will receive from the Lord part of of body we want or seieve Anything positive means we want to sieve some a up part of or body | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00052_S08813_W000015 | 0.8058 | 56 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00052_S08813_W000015 | 0.7712 | 65 | We come from different parties but were Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0385 |
| 640 | causal_full_asr | EN_B00052_S08813_W000015 | 0.7596 | 67 | We come from different parties but were Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0385 |
| 1280 | causal_full_asr | EN_B00052_S08813_W000015 | 0.7769 | 64 | We come from different parties but were Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0385 |
| 160 | streaming_asr | EN_B00058_S06429_W000060 | 0.7368 | 45 | The Alliance believed that part dis disclosure be very give a to do with all of data that was help there | wer | 0.3333 |
| 320 | streaming_asr | EN_B00058_S06429_W000060 | 0.7237 | 50 | The alliance believed that part dis disclosure be very give a to do with all of the data that was help there | wer | 0.2857 |
| 640 | streaming_asr | EN_B00058_S06429_W000060 | 0.7105 | 52 | The alliance believed that part dis disclosure be very give a to do with all of the data that was help there | wer | 0.2857 |
| 1280 | streaming_asr | EN_B00058_S06429_W000060 | 0.6678 | 56 | The alliance believed that part dis disclosure be very give a to do with all of the data that was help there | wer | 0.2857 |
| 160 | causal_full_asr | EN_B00048_S01182_W000001 | 0.9381 | 8 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S01182_W000001 | 0.9248 | 10 | Winder you store the cookies from the cookie jar | wer | 0.2222 |
| 640 | causal_full_asr | EN_B00048_S01182_W000001 | 0.9115 | 12 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S01182_W000001 | 0.9071 | 12 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S05933_W000060 | 0.7167 | 26 | Now America has really interesting culture that would fit | wer | 0.2000 |
| 320 | streaming_asr | EN_B00048_S05933_W000060 | 0.7333 | 24 | South American has really interesting culture that would fit | wer | 0.2000 |
| 640 | streaming_asr | EN_B00048_S05933_W000060 | 0.6500 | 28 | South America has really interesting culture that would fit | wer | 0.1000 |
| 1280 | streaming_asr | EN_B00048_S05933_W000060 | 0.6611 | 28 | South American has a really interesting culture that would fit | wer | 0.1000 |
| 160 | streaming_asr | EN_B00058_S03144_W000037 | 0.8527 | 16 | Long Complex sentences with Multiple paragraphs in email | wer | 0.1111 |
| 320 | streaming_asr | EN_B00058_S03144_W000037 | 0.7857 | 27 | Long complexes sentences with Multiple paragraphs in email | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S03144_W000037 | 0.7902 | 27 | Long complexes sentences with modifiable paragraphs in email | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00058_S03144_W000037 | 0.7857 | 28 | Long complicate sentences with Multiple paragraphs in email | wer | 0.2222 |
| 160 | streaming_asr | EN_B00091_S07092_W000002 | 0.7298 | 35 | And the concern off what like it turn out little of family from had in the kind of of storytRNA sigger | wer | 0.8421 |
| 320 | streaming_asr | EN_B00091_S07092_W000002 | 0.6613 | 46 | But a comes off more like it care uncle little a family from had in the kind of of storytary figure | wer | 0.5263 |
| 640 | streaming_asr | EN_B00091_S07092_W000002 | 0.6250 | 52 | But a comes off more like a carrying uncle little a family were had in the kind of of thoracic material figure | wer | 0.5263 |
| 1280 | streaming_asr | EN_B00091_S07092_W000002 | 0.6089 | 54 | But a comes off more like a carrying uncle little a family were the in the kind of of Authority figure | wer | 0.4737 |
| 160 | causal_full_asr | EN_B00036_S05339_W000016 | 0.7981 | 21 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00036_S05339_W000016 | 0.7934 | 21 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00036_S05339_W000016 | 0.7512 | 23 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00036_S05339_W000016 | 0.7606 | 22 | To use the present perfect correctly you need to know things like | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
