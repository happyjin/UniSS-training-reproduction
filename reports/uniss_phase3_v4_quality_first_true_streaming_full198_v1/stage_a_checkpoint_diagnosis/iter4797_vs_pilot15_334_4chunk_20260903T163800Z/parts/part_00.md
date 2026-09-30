# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9611**
- Weighted CTC blank ratio: **0.7004**
- Weighted streaming WER/CER: **0.1439**
- Weighted causal-full WER/CER: **0.0871**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7709 | 34 | As still late now let's just get this over with | wer | 0.2000 |
| 320 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7621 | 33 | As too late now let's just get this over way | wer | 0.2000 |
| 640 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7577 | 34 | As too late now let's just get this over way | wer | 0.2000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7797 | 33 | As jewell laid now let just get this over way | wer | 0.5000 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.8528 | 19 | Just who's about the netting and that's | wer | 0.8571 |
| 320 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.8344 | 21 | Just two What's about the magnetic impact | wer | 0.5714 |
| 640 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7914 | 22 | That's true What about the mentality in that | wer | 0.4286 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7669 | 26 | That's true What about the genetic impact | wer | 0.1429 |
| 160 | streaming_asr | CommonVoice_EN_0000189191 | 0.8763 | 17 | The boy brought his horse closer | wer | 0.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000189191 | 0.8441 | 22 | The boy brought his whorse closser | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000189191 | 0.8118 | 23 | The boy brought his whorse closser | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000189191 | 0.7957 | 24 | The boy brought his wharst closer | wer | 0.1667 |
| 160 | streaming_asr | CommonVoice_EN_0000332324 | 0.7565 | 38 | Many building post for a wood where entirely destroyed | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000332324 | 0.7304 | 43 | Many building both for a wide war entirely destroyed | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000332324 | 0.7043 | 46 | Many building both for a wood war entirely destroyed | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000332324 | 0.6957 | 47 | Many building both for a wood war entirely destroyed | wer | 0.4444 |
| 160 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7579 | 50 | Later years he occasionally proved a determined lower order batsman | wer | 0.0909 |
| 320 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7228 | 52 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7123 | 55 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000430515 | 0.6912 | 58 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000501889 | 0.6684 | 41 | We usually part of motors cycles near to this building | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000501889 | 0.6218 | 43 | We usually part of motors cycles near to this building | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000501889 | 0.6218 | 43 | We usually part of motocycles near to this building | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000501889 | 0.6580 | 41 | We usually part of model cycles near to this building | wer | 0.4444 |
| 160 | streaming_asr | DailyTalk_0000001997 | 0.8352 | 11 | In Miss Scott please | wer | 0.7500 |
| 320 | streaming_asr | DailyTalk_0000001997 | 0.7802 | 15 | In Miss Scott Please | wer | 0.7500 |
| 640 | streaming_asr | DailyTalk_0000001997 | 0.7582 | 15 | Give me Scott Please | wer | 0.2500 |
| 1280 | streaming_asr | DailyTalk_0000001997 | 0.7692 | 14 | Give me Scott Please | wer | 0.2500 |
| 160 | streaming_asr | LibriSpeech_0000100601 | 0.6991 | 131 | In one enthusiastic jumble While Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 320 | streaming_asr | LibriSpeech_0000100601 | 0.7021 | 123 | In one enthusiastic jumble while Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 640 | streaming_asr | LibriSpeech_0000100601 | 0.6871 | 128 | In one enthusiastic jumble while Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 1280 | streaming_asr | LibriSpeech_0000100601 | 0.6841 | 133 | In one enthusiastic jumble while Tom was off on his third rade my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 160 | causal_full_asr | LibriSpeech_0000192309 | 0.7035 | 62 | In the hope of gaining a little more time he repeated his question no | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000192309 | 0.6972 | 63 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 640 | causal_full_asr | LibriSpeech_0000192309 | 0.6814 | 63 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 1280 | causal_full_asr | LibriSpeech_0000192309 | 0.6845 | 64 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 160 | streaming_asr | LibriSpeech_0000238719 | 0.6698 | 94 | So tray is the world to go on If this kind of thing be permitted I may be going out to dinner or to the opposite to night | wer | 0.1071 |
| 320 | streaming_asr | LibriSpeech_0000238719 | 0.6437 | 100 | Chao tray is the world to go on If this kind of thing be permitted I may be gilling out to dinner or to the oper to night | wer | 0.1429 |
| 640 | streaming_asr | LibriSpeech_0000238719 | 0.6271 | 100 | How tray is the world to go on if this kind of thing be permitted I may be going out to dinner or to the oper to night | wer | 0.0714 |
| 1280 | streaming_asr | LibriSpeech_0000238719 | 0.6247 | 99 | How tray is the world to go on if this kind of thing be permitted I may be going out to dinner or to the oper to night | wer | 0.0714 |
| 160 | streaming_asr | emilia_zh_0004002573 | 0.5871 | 160 | 啊因为没有因为他这样有些人可能喜欢看你的那个那种然后他家的那份粉丝然后你换那种那种去拍他可能就不喜欢你这个那种他可能就会去取消 | cer | 0.2167 |
| 320 | streaming_asr | emilia_zh_0004002573 | 0.5752 | 165 | 啊因为没有因为他叫有些人可能喜欢看你那个那种然后他家的那种粉丝然后你换那种那种去拍他可能就不喜欢你这个那种他可能就会去取消 | cer | 0.2333 |
| 640 | streaming_asr | emilia_zh_0004002573 | 0.5585 | 165 | 啊因为没有因为他叫有限人可能喜欢看你的一个内容然后他加的一份词然后你换那种那种去拍他可能就不喜欢你这个内容他可能就会取消 | cer | 0.2167 |
| 1280 | streaming_asr | emilia_zh_0004002573 | 0.5465 | 171 | 啊因为因为因为他叫有限人可能喜欢看你的一个内容然后他家的就粉丝然后你换那种那种去拍他可能就不喜欢你这个内容他可能就会取消 | cer | 0.2000 |
| 160 | causal_full_asr | emilia_zh_0004036374 | 0.8022 | 95 | 则有关共产主义运动的具体策略和节目的叙述来充当马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 320 | causal_full_asr | emilia_zh_0004036374 | 0.7985 | 100 | 则有关共产主义运动的具体策略和结论的叙述来充当了马克思主义哲学发展的内在逻辑叙述 | cer | 0.0750 |
| 640 | causal_full_asr | emilia_zh_0004036374 | 0.7931 | 102 | 则有关共产主义运动的具体策略和结论的叙述来重担的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0004036374 | 0.7913 | 103 | 则有关共产主义运动的具体策略和结论的叙述来重担的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0004130152 | 0.7834 | 57 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004130152 | 0.7762 | 56 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004130152 | 0.7870 | 55 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004130152 | 0.7798 | 56 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004358957 | 0.7912 | 82 | 我定的样子足够看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.2069 |
| 320 | streaming_asr | emilia_zh_0004358957 | 0.7690 | 85 | 我听这鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.1379 |
| 640 | streaming_asr | emilia_zh_0004358957 | 0.7715 | 85 | 我听这鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.1379 |
| 1280 | streaming_asr | emilia_zh_0004358957 | 0.7666 | 84 | 我盯着鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.0690 |
| 160 | causal_full_asr | emilia_zh_0004665404 | 0.6543 | 82 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004665404 | 0.6114 | 89 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004665404 | 0.5971 | 90 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004665404 | 0.5971 | 88 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004692799 | 0.5991 | 188 | So we taken these strings instrument instead of using the them as these warm of broad filled communicators we're in the string family We're sort of turing them in now two procussion instrument That's a very procussive effect | wer | 0.3243 |
| 320 | streaming_asr | emilia_zh_0004692799 | 0.5697 | 190 | So we taken these string instrument instead of using the them as these warm of broad filled communicators were in the string family where there's a lot turning them in now to protection instrument That's a very procussive effect | wer | 0.3243 |
| 640 | streaming_asr | emilia_zh_0004692799 | 0.5650 | 194 | So we taken these string instrument instead of using the them as these warm of brato filled communicators were in the string family where there's a lot treme them in now to protection instrument That's a very procussive effect | wer | 0.3514 |
| 1280 | streaming_asr | emilia_zh_0004692799 | 0.5573 | 196 | So we taken these string instrument instead of using the them as these warm of broad filled communicators were in the string family where there's a tongue them in now to protection instrument That's a very procussive effect | wer | 0.3243 |
| 160 | streaming_asr | emilia_zh_0004777075 | 0.6737 | 74 | He appeared over Dink's shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004777075 | 0.6556 | 78 | He appeared over dink shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0004777075 | 0.6465 | 79 | He perished over dink shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004777075 | 0.6677 | 76 | He perished over dink shoulder at the someday newspaper so what's the next day Kami Tuesday | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004873848 | 0.6979 | 40 | Margaret says little art with an honest open smile | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0004873848 | 0.7135 | 37 | Margaret says little up with an honest open smile | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004873848 | 0.6927 | 37 | Margaret faced little up with an honest open smile | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004873848 | 0.7083 | 37 | Margaret says little up with an honest open smile | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004999877 | 0.7904 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004999877 | 0.7854 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004999877 | 0.7854 | 79 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004999877 | 0.7904 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005070340 | 0.7825 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 320 | causal_full_asr | emilia_zh_0005070340 | 0.7855 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 640 | causal_full_asr | emilia_zh_0005070340 | 0.7734 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 1280 | causal_full_asr | emilia_zh_0005070340 | 0.7764 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 160 | streaming_asr | emilia_zh_0005347772 | 0.6654 | 85 | 另外民主党被剥夺了了死胡同据民主党来说他们面临这个困难的选择 | cer | 0.2333 |
| 320 | streaming_asr | emilia_zh_0005347772 | 0.6502 | 86 | 另外民主党被一波走进了死胡同对于民主党来说他们面临这个困难的选择 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0005347772 | 0.6578 | 82 | 另外民主党被一波走进了死胡同对于民主党来说他们面临这个困难的选择 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0005347772 | 0.6426 | 87 | 另外民主党被一波走进了死胡同对于民主党来说他们面临着困难的选择 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0005578734 | 0.7351 | 46 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 320 | streaming_asr | emilia_zh_0005578734 | 0.7297 | 46 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0005578734 | 0.7297 | 45 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0005578734 | 0.7189 | 46 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 160 | causal_full_asr | emilia_zh_0005780397 | 0.6890 | 162 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小黄然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 320 | causal_full_asr | emilia_zh_0005780397 | 0.6701 | 172 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做hr嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 640 | causal_full_asr | emilia_zh_0005780397 | 0.6873 | 168 | 工作都一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友们然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0308 |
| 1280 | causal_full_asr | emilia_zh_0005780397 | 0.6890 | 168 | 工作都一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0154 |
| 160 | streaming_asr | emilia_zh_0005818215 | 0.8342 | 61 | 却打起去情绪气啊还比较低然后在加上我们要让最近都挺多事儿 | cer | 0.3548 |
| 320 | streaming_asr | emilia_zh_0005818215 | 0.8144 | 69 | 觉得大家其实确实情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.1935 |
| 640 | streaming_asr | emilia_zh_0005818215 | 0.8094 | 67 | 就大家请去情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0005818215 | 0.8094 | 66 | 就大家请去情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0006056256 | 0.7580 | 51 | 我问啊这这种情况就特别容易发生因为我觉得完全 | cer | 0.2727 |
| 320 | streaming_asr | emilia_zh_0006056256 | 0.7534 | 50 | 那啊就这这种情况就特别容易发生因为我觉得完全 | cer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0006056256 | 0.7397 | 53 | 那啊啊这这种情况就特别容易发行因为我觉得完全 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0006056256 | 0.7671 | 48 | 我问啊这这种情况就特别容易发行因为我觉得完全 | cer | 0.3182 |
| 160 | causal_full_asr | emilia_zh_0006174175 | 0.7393 | 102 | 我一上来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.2174 |
| 320 | causal_full_asr | emilia_zh_0006174175 | 0.7299 | 107 | 一章后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.1957 |
| 640 | causal_full_asr | emilia_zh_0006174175 | 0.7180 | 105 | 一章后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确的知道这个事儿算什么 | cer | 0.1739 |
| 1280 | causal_full_asr | emilia_zh_0006174175 | 0.7085 | 109 | 一扔后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确的知道这个事儿算什么 | cer | 0.1739 |
| 160 | streaming_asr | emilia_zh_0006303057 | 0.7786 | 179 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那个无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你只要吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1216 |
| 320 | streaming_asr | emilia_zh_0006303057 | 0.7712 | 187 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1081 |
| 640 | streaming_asr | emilia_zh_0006303057 | 0.7650 | 193 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那个无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做升降机在外面就是在那个窗户外面 | cer | 0.0676 |
| 1280 | streaming_asr | emilia_zh_0006303057 | 0.7660 | 187 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1081 |
| 160 | streaming_asr | emilia_zh_0006379861 | 0.6379 | 77 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006379861 | 0.6310 | 74 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006379861 | 0.6034 | 80 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006379861 | 0.6034 | 79 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006464935 | 0.6520 | 119 | That price the point where the quality that consumers want to buy a equal the quality that sellers want to produce is called the equal librarian price | wer | 0.1538 |
| 320 | streaming_asr | emilia_zh_0006464935 | 0.6226 | 121 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 640 | streaming_asr | emilia_zh_0006464935 | 0.6143 | 122 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 1280 | streaming_asr | emilia_zh_0006464935 | 0.6164 | 119 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 160 | causal_full_asr | emilia_zh_0006659550 | 0.7352 | 58 | 中期的数据看其有利于这个布什让他能做出两次错误的判断 | cer | 0.2414 |
| 320 | causal_full_asr | emilia_zh_0006659550 | 0.7273 | 61 | 中期的数据看上去有利于这个布什但他能做出两次错误的判断 | cer | 0.1724 |
| 640 | causal_full_asr | emilia_zh_0006659550 | 0.7154 | 67 | 中期的数据看起来有利这个不实那他们做出的两次错误的判断 | cer | 0.2069 |
| 1280 | causal_full_asr | emilia_zh_0006659550 | 0.7115 | 67 | 中期的数据看上去有利于这个布什但他能做出两次错误的判断 | cer | 0.1724 |
| 160 | streaming_asr | emilia_zh_0006714517 | 0.6726 | 50 | 好接下来我讲最重要的今天的一件事情呢啊 | cer | 0.1000 |
| 320 | streaming_asr | emilia_zh_0006714517 | 0.6548 | 52 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 640 | streaming_asr | emilia_zh_0006714517 | 0.6488 | 52 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 1280 | streaming_asr | emilia_zh_0006714517 | 0.6488 | 54 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 160 | streaming_asr | emilia_zh_0006990963 | 0.7902 | 41 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006990963 | 0.7723 | 44 | 现在正好是春天我们一块儿就去院子里 | cer | 0.0588 |
| 640 | streaming_asr | emilia_zh_0006990963 | 0.7768 | 43 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006990963 | 0.7768 | 45 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007169925 | 0.8336 | 109 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包的他们也会利用任何时间思考 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007169925 | 0.8252 | 111 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 640 | streaming_asr | emilia_zh_0007169925 | 0.8210 | 114 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包等他们也会利用任何时间思考 | cer | 0.0238 |
| 1280 | streaming_asr | emilia_zh_0007169925 | 0.8182 | 117 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0007462823 | 0.7650 | 48 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007462823 | 0.7650 | 47 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 640 | streaming_asr | emilia_zh_0007462823 | 0.7373 | 54 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007462823 | 0.7281 | 55 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007691054 | 0.7547 | 95 | 把从这个角度来讲我们认为allie是一个比定律论增速更好更加重要的指标原因很简单 | cer | 0.2432 |
| 320 | streaming_asr | emilia_zh_0007691054 | 0.7547 | 97 | 那从这个角度来讲我们认为ROE是一个比定律论增速更好更加重要的指标远很简单 | cer | 0.1892 |
| 640 | streaming_asr | emilia_zh_0007691054 | 0.7500 | 98 | 那从这个角度来讲我们认为all是一个比定律论增速更好更加重要的指标源于很简单 | cer | 0.2162 |
| 1280 | streaming_asr | emilia_zh_0007691054 | 0.7570 | 96 | 那从这个角度来讲我们认为ROE是一个比定律论增速更好更加重要的指标远远很简单 | cer | 0.1892 |
| 160 | streaming_asr | EN_B00097_S02875_W000006 | 0.6259 | 151 | But the first thing is cross your legs because we don't want to want to receive from the lower part of our body We want to receive anything positive means we want to receive from the other upper part of our body | wer | 0.1026 |
| 320 | streaming_asr | EN_B00097_S02875_W000006 | 0.5957 | 160 | But uh for thing he's crossed your legs because we don't want to do receive from the law part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.1282 |
| 640 | streaming_asr | EN_B00097_S02875_W000006 | 0.6082 | 151 | But uh first thing is cross your legs because we don't want to do receive from the lower part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.0256 |
| 1280 | streaming_asr | EN_B00097_S02875_W000006 | 0.6082 | 151 | But uh first thing is cross your legs because we don't want to want receive from the lower part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.0256 |
| 160 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6327 | 134 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6135 | 135 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6135 | 139 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6038 | 142 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S06429_W000060 | 0.6020 | 92 | The aliance believed that parsh dis closure be very gigical to do with all of the data that was out there | wer | 0.2381 |
| 320 | streaming_asr | EN_B00058_S06429_W000060 | 0.5461 | 98 | The alliance believed that parsh dis closure be very gigical to do with all of the data that was help there | wer | 0.2381 |
| 640 | streaming_asr | EN_B00058_S06429_W000060 | 0.5559 | 96 | The alliance believed that parsh dislosure be very gigical to do with all of the data that was out there | wer | 0.1905 |
| 1280 | streaming_asr | EN_B00058_S06429_W000060 | 0.5592 | 96 | The alliance believed that parsh dislosure be very gigical to do with all of the data that was helped there | wer | 0.2381 |
| 160 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7788 | 34 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7389 | 41 | Winder you store the cookies from the cookie jar | wer | 0.2222 |
| 640 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7478 | 41 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7345 | 38 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S05933_W000060 | 0.5500 | 54 | South America has really interesting culture that would fit | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S05933_W000060 | 0.5333 | 55 | South America has a really interesting culture that would fit | wer | 0.0000 |
| 640 | streaming_asr | EN_B00048_S05933_W000060 | 0.5056 | 57 | South America has a really interesting culture that would fit | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S05933_W000060 | 0.4722 | 59 | South American has a really interesting culture that would fit | wer | 0.1000 |
| 160 | streaming_asr | EN_B00058_S03144_W000037 | 0.7143 | 44 | Long complex sentences with multiple paragraphs in email | wer | 0.1111 |
| 320 | streaming_asr | EN_B00058_S03144_W000037 | 0.6741 | 46 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S03144_W000037 | 0.6786 | 45 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00058_S03144_W000037 | 0.6786 | 46 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 160 | streaming_asr | EN_B00091_S07092_W000002 | 0.5565 | 81 | And it comes off more like I kind uncle or a family member than Annie kind of authoritarian figured | wer | 0.2632 |
| 320 | streaming_asr | EN_B00091_S07092_W000002 | 0.5000 | 93 | But it comes off more like a current uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 640 | streaming_asr | EN_B00091_S07092_W000002 | 0.4879 | 92 | But it comes off more like a current uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 1280 | streaming_asr | EN_B00091_S07092_W000002 | 0.4758 | 97 | But it comes off more like a carrying uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 160 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6948 | 49 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6808 | 47 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6620 | 47 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6620 | 49 | To use the present perfect correctly you need to know things like | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
