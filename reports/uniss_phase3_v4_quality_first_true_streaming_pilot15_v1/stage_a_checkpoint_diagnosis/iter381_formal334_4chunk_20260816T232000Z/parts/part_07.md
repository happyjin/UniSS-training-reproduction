# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9110**
- Weighted CTC blank ratio: **0.8572**
- Weighted streaming WER/CER: **0.3408**
- Weighted causal-full WER/CER: **0.1216**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000188343 | 0.9370 | 4 | They of of just thoroughly | wer | 0.6000 |
| 320 | streaming_asr | CommonVoice_EN_0000188343 | 0.8740 | 8 | They of so just thoroughly | wer | 0.6000 |
| 640 | streaming_asr | CommonVoice_EN_0000188343 | 0.8898 | 8 | They of so just thoroughly | wer | 0.6000 |
| 1280 | streaming_asr | CommonVoice_EN_0000188343 | 0.8661 | 7 | Day of so of thoroughly | wer | 0.8000 |
| 160 | streaming_asr | CommonVoice_EN_0000331841 | 0.8923 | 7 | Just did roll the me of the asked time | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000331841 | 0.8692 | 8 | You did rode th me the asked time | wer | 0.5000 |
| 640 | streaming_asr | CommonVoice_EN_0000331841 | 0.8385 | 9 | You did rode th me the asked time | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000331841 | 0.8846 | 9 | You did rode th me the asked time | wer | 0.5000 |
| 160 | causal_full_asr | CommonVoice_EN_0000352714 | 0.8542 | 27 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 320 | causal_full_asr | CommonVoice_EN_0000352714 | 0.8274 | 32 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 640 | causal_full_asr | CommonVoice_EN_0000352714 | 0.8155 | 32 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 1280 | causal_full_asr | CommonVoice_EN_0000352714 | 0.8065 | 32 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 160 | streaming_asr | CommonVoice_EN_0000471209 | 0.8151 | 18 | The boy began to did into to the journ | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000471209 | 0.7945 | 19 | The boy began to did into to the journ | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000471209 | 0.7123 | 22 | The boy began to did into to the June | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000471209 | 0.7466 | 22 | The boy began to deed in to the June | wer | 0.5000 |
| 160 | streaming_asr | HQ-Conversations_0000028308 | 1.0000 | 0 | 我英雄咋了 | cer | 0.8000 |
| 320 | streaming_asr | HQ-Conversations_0000028308 | 0.9677 | 2 | 对英雄咋办 | cer | 0.8000 |
| 640 | streaming_asr | HQ-Conversations_0000028308 | 0.9677 | 2 | 为兄弟咋了 | cer | 0.4000 |
| 1280 | streaming_asr | HQ-Conversations_0000028308 | 0.9839 | 1 | 为英雄咋了 | cer | 0.8000 |
| 160 | causal_full_asr | LibriSpeech_0000011649 | 0.7522 | 82 | Or on several hills as Roman as well as Bostonian history testify can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0294 |
| 320 | causal_full_asr | LibriSpeech_0000011649 | 0.7507 | 82 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000011649 | 0.7434 | 88 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000011649 | 0.7507 | 87 | Our unseveril hills as Roman as well as Bostonian history testify can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.1176 |
| 160 | streaming_asr | LibriSpeech_0000090820 | 0.7385 | 104 | which kept me in had house Nearly to weeks The basement kitchens seemed heavenly save and warmer in the of days like time little boat a winter see the men were out and the fields all day h asking corn and and came in noon | wer | 0.3750 |
| 320 | streaming_asr | LibriSpeech_0000090820 | 0.7237 | 105 | Which kept me in had house Nearly to weeks The basement kitchens seemed heavenly save and warmer in of days like time little boat a winter see the men were out and the fields all day h asking corn and and came in noon | wer | 0.3542 |
| 640 | streaming_asr | LibriSpeech_0000090820 | 0.7183 | 108 | Which kept me in had house Nearly to weeks The basement kitchens seemed heavenly save and warmer in of days like time little boat a winter see the men were out and the fields all day h asking corn and and came in a noon | wer | 0.3542 |
| 1280 | streaming_asr | LibriSpeech_0000090820 | 0.7264 | 107 | Which kept me in had house Nearly to weeks The basement kitchens seemed heavenly save and warmer in was days like time little boat a winter see the men were out in the fields all day h asking corn and and came in a noon | wer | 0.3333 |
| 160 | streaming_asr | LibriSpeech_0000215121 | 0.8014 | 74 | And turn by you to me with your ind endorsement of horse I imetic counted him over the forty bank notes Want criss another is his head in into a fascent But that's is not all continue dumbla | wer | 0.4167 |
| 320 | streaming_asr | LibriSpeech_0000215121 | 0.8000 | 78 | And turn by you to me would do your ind endorsement of horse I Immediately counted him over the forty bank notes Want crucy another is his head in into a fascent But that's is not all continued dumbla | wer | 0.4167 |
| 640 | streaming_asr | LibriSpeech_0000215121 | 0.7913 | 82 | And turn by you to me would do your ind endorsement of horse I Immediately counted him over the forty bank notes Want crucy another his his head in took of fascent But that's is not all continued dumbla | wer | 0.3889 |
| 1280 | streaming_asr | LibriSpeech_0000215121 | 0.7928 | 80 | And turn by you to me would do your ind endorsement of horse I Immediately counted him over the forty bank notes Want crucy another to his head in took a fascent But that's is not all continued dumbla | wer | 0.4167 |
| 160 | streaming_asr | emilia_zh_0003918326 | 0.9647 | 10 | 啊我我咱们还有一个点听一次这是这个事情里点起来就这些人他们眼互相关联 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0003918326 | 0.9679 | 9 | 啊我我什么呢还有一个点听有意思就是这些东西事情了点起来就这些人他们眼互相关联 | cer | 0.2778 |
| 640 | streaming_asr | emilia_zh_0003918326 | 0.9423 | 14 | 啊我我什么呢还有一个点听有意思就是这些东西事情人点起来就这些人他们眼互相关联 | cer | 0.2778 |
| 1280 | streaming_asr | emilia_zh_0003918326 | 0.9551 | 11 | 啊哦我什么呢还有一个点听有意思这是这些东西事情人点起来就这些人他们眼互相关联 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0003942539 | 0.9446 | 25 | 就我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0175 |
| 320 | causal_full_asr | emilia_zh_0003942539 | 0.9366 | 26 | 因为我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0003942539 | 0.9327 | 29 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 1280 | causal_full_asr | emilia_zh_0003942539 | 0.9287 | 30 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0004129851 | 1.0000 | 0 | 之间我两个人争做的地方喝酒呢 | cer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0004129851 | 0.9815 | 3 | 直接我两个人争做的地方喝酒呢 | cer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004129851 | 0.9815 | 3 | 直接我两个人争坐在地上喝酒呢 | cer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0004129851 | 0.9815 | 3 | 之间有两个人争做的地上喝酒呢 | cer | 0.3571 |
| 160 | streaming_asr | emilia_zh_0004358307 | 0.8685 | 40 | Also was up to him to prove himself of the there's six to make the im pr proud of him and his music without the fate is idea have how prroud were already | wer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0004358307 | 0.8466 | 48 | If was up to him to prove himself for the there six to make the im priled of him and his music without the fate idea of how prroud were already | wer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0004358307 | 0.8586 | 45 | If was up to him to prove himself for the there six to make the im priled of him and his music without the fate is idea of how prroud were already | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004358307 | 0.8745 | 39 | It was up to him to prove himself for the there six to make the im pr proud of him and his music without the fate is idea of how prroud were already | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004659190 | 0.8338 | 33 | See the the when was close to granting he repressed When you I love you can to anything creation | wer | 0.4286 |
| 320 | streaming_asr | emilia_zh_0004659190 | 0.8100 | 41 | Seeing the the win was close to granting he repressed when you I love you can to anything creation | wer | 0.3810 |
| 640 | streaming_asr | emilia_zh_0004659190 | 0.8127 | 43 | Seeing the the win was close to granting he requested when you I love you can to anything creation | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004659190 | 0.7784 | 50 | Seeing that the win was close to granting he requested when you I love you can to anything creation | wer | 0.2857 |
| 160 | causal_full_asr | emilia_zh_0004659501 | 0.9542 | 8 | And thought old you just right now or ten and then where do where to | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004659501 | 0.9570 | 8 | And thought old you just write nine or ten and then where do where to | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0004659501 | 0.9513 | 11 | And for old you just write nine or ten and then where do where to | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0004659501 | 0.9427 | 10 | And for old you just write nine or ten and then where do where to | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004776929 | 0.9053 | 17 | You evening They stethal down for had big families dinner This was eight | wer | 0.4286 |
| 320 | streaming_asr | emilia_zh_0004776929 | 0.8947 | 16 | In evening They stessed down for of big families dinner This was eight | wer | 0.3571 |
| 640 | streaming_asr | emilia_zh_0004776929 | 0.8526 | 23 | In evening They fessed down for of big families dinner This was eight | wer | 0.3571 |
| 1280 | streaming_asr | emilia_zh_0004776929 | 0.8456 | 23 | In evening They stessed down for of big families dinner This was eight | wer | 0.3571 |
| 160 | streaming_asr | emilia_zh_0004843272 | 0.7481 | 74 | It principled tennet is the that economic growth is supreme good or it least pro<\|glm_semantic_11271\|>s for with supreme good because justice freedom and even happiness all the pen don't economic grow | wer | 0.4194 |
| 320 | streaming_asr | emilia_zh_0004843272 | 0.7241 | 84 | It's principled tennet is that that economic growth is supreme good or it least pro<\|glm_semantic_11271\|>s for with supreme good because justice Freedom and even happiness all depended don't economic grow | wer | 0.3548 |
| 640 | streaming_asr | emilia_zh_0004843272 | 0.7444 | 74 | It's principled tennet is the that economic growth is supreme good or it least pro<\|glm_semantic_11271\|>s for with supreme good because justice freedom and even happiness all depended don't economic grow | wer | 0.3548 |
| 1280 | streaming_asr | emilia_zh_0004843272 | 0.7204 | 80 | It's principled tennet is that that economic growth is supreme good or it least pro<\|glm_semantic_11271\|>s for with supreme good because justice freedom and even happiness all depended don't economic grow | wer | 0.3548 |
| 160 | streaming_asr | emilia_zh_0004943713 | 0.9813 | 9 | 等因我知道太清楚了你在这上面三处的信仰和乘客尝试问从业者黑洞活跃他们那个情感得愚蠢 | cer | 0.3953 |
| 320 | streaming_asr | emilia_zh_0004943713 | 0.9728 | 13 | 但是因我知道他清楚了你在这上面三的信仰和乘客尝试问从业者黑洞活跃他们那个情感的愚蠢 | cer | 0.3488 |
| 640 | streaming_asr | emilia_zh_0004943713 | 0.9728 | 13 | 但是因我知道他清楚了你在这圣面三处的信仰和乘客尝试问从业者奋斗维护他们你的情感的愚蠢 | cer | 0.2558 |
| 1280 | streaming_asr | emilia_zh_0004943713 | 0.9762 | 10 | 但是是因我知道他清楚了你在这上面三的信仰和乘客尝试问从业者黑洞活跃他们你的情感的愚蠢 | cer | 0.3488 |
| 160 | causal_full_asr | emilia_zh_0005070101 | 0.9908 | 5 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 320 | causal_full_asr | emilia_zh_0005070101 | 0.9924 | 4 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 640 | causal_full_asr | emilia_zh_0005070101 | 0.9924 | 4 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 1280 | causal_full_asr | emilia_zh_0005070101 | 0.9939 | 4 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0005313494 | 0.9557 | 5 | 就不能都乐吧一个后面没有一个应该拼 | cer | 0.4500 |
| 320 | streaming_asr | emilia_zh_0005313494 | 0.9557 | 6 | 都不能都乐吧一个后面没有一个应该听 | cer | 0.5500 |
| 640 | streaming_asr | emilia_zh_0005313494 | 0.9367 | 8 | 他不能读乐吧那个后面没有一个应该平 | cer | 0.4500 |
| 1280 | streaming_asr | emilia_zh_0005313494 | 0.9367 | 8 | 他不能读乐吧一个后面没有一个应该拼 | cer | 0.4000 |
| 160 | streaming_asr | emilia_zh_0005578304 | 1.0000 | 0 | 对数归来就好奇的问 | cer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0005578304 | 1.0000 | 0 | 你对树叶归来就好奇的问 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005578304 | 1.0000 | 0 | 你许回来就好奇的问 | cer | 0.4167 |
| 1280 | streaming_asr | emilia_zh_0005578304 | 1.0000 | 0 | 你对树叶归来就好奇的问 | cer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0005714451 | 0.9825 | 4 | 是不是转转音接下来你可以看到其实前面这三五边儿 | cer | 0.5200 |
| 320 | causal_full_asr | emilia_zh_0005714451 | 0.9825 | 4 | 虽然状态应该接下来你可以看到其实前边的三部边儿 | cer | 0.4400 |
| 640 | causal_full_asr | emilia_zh_0005714451 | 0.9825 | 5 | 生产状态您接下来您可以看到其实前边的三目标 | cer | 0.2800 |
| 1280 | causal_full_asr | emilia_zh_0005714451 | 0.9755 | 6 | 顺便撞到您接下来你就可以看到其实前面的三部边儿 | cer | 0.4800 |
| 160 | streaming_asr | emilia_zh_0005818033 | 0.9532 | 5 | 我告诉你一个那个教程的啊给咱们这边 | cer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0005818033 | 0.9298 | 8 | 报告一个那个教程啊给咱们这边 | cer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0005818033 | 0.9357 | 6 | 投稿一个那个教程啊给咱们这边 | cer | 0.0667 |
| 1280 | streaming_asr | emilia_zh_0005818033 | 0.9415 | 8 | 投稿一个那个教程啊给咱们这边 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0006041629 | 0.9530 | 14 | 所以看了一些信息以后觉得现在就够里一些创伤把但是整体来说我觉得这演员他在我眼里变得特别有趣的 | cer | 0.2917 |
| 320 | streaming_asr | emilia_zh_0006041629 | 0.9480 | 16 | 做看来一些信息以后觉得现在觉得歌曲的一些创伤把但是整体来说我觉得就是演员他在我眼里变得特别有趣的 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0006041629 | 0.9505 | 16 | 就是看看一些信息以后觉得现在觉得够气了一些创伤打但是整体来说我觉得这演员他在我眼里变得特别有趣的 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0006041629 | 0.9604 | 14 | 就是看来这些信息以后觉得现在觉得够气离一些创伤把但是整体来说我觉得这演员他在我眼里变得特别有趣 | cer | 0.2292 |
| 160 | causal_full_asr | emilia_zh_0006099356 | 0.9295 | 15 | 就整个这个流向还不比较多但是你就非这种你确定是能拿到钱的什么 | cer | 0.2963 |
| 320 | causal_full_asr | emilia_zh_0006099356 | 0.9253 | 17 | 就整个这个流量还不比较多但是你非得这种你确定是能拿到钱的什么 | cer | 0.2963 |
| 640 | causal_full_asr | emilia_zh_0006099356 | 0.9336 | 14 | 就整个这个流向还不知道但是你确定这种你确定是能拿到钱的什么 | cer | 0.2963 |
| 1280 | causal_full_asr | emilia_zh_0006099356 | 0.9212 | 16 | 就整个这个留下还不知道但是你确定这种你确定是能拿到钱的什么 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0006270122 | 0.9590 | 10 | 你这样都是对的还是一个完整的这个原因啊才就是因为 | cer | 0.3200 |
| 320 | streaming_asr | emilia_zh_0006270122 | 0.9478 | 11 | 你这样都是对的他是一个完整的这个原因啊他就是因为 | cer | 0.3200 |
| 640 | streaming_asr | emilia_zh_0006270122 | 0.9478 | 12 | 你这样就是对的他是一个完整的这个原因啊他就是因为 | cer | 0.3200 |
| 1280 | streaming_asr | emilia_zh_0006270122 | 0.9403 | 14 | 你这样就是对的他是一个完整的这个原因啊他就是a | cer | 0.2800 |
| 160 | streaming_asr | emilia_zh_0006379722 | 0.9109 | 20 | But you couldn't leave the other other two He must take them with him all the way | wer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0006379722 | 0.8812 | 21 | But it couldn't leave the other other two He must take them was him all the way | wer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0006379722 | 0.8812 | 23 | But he couldn't leave the other other two He must take them was him all the way | wer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0006379722 | 0.8977 | 18 | But he couldn't leave the other other two He must take them was him all the way | wer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0006464698 | 0.7882 | 50 | Oh one for ways actually for example He wanted meat to tell him my big count passwork and he cat taking photographs of me from different angles | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0006464698 | 0.7904 | 48 | Oh one for ways actually for example He wanted meat to tell him my big count passwork and he cat taking photographs of me from different angles | wer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0006464698 | 0.7773 | 51 | Oh long for ways actually for example he wanted meat to tell him my binking count passwork and he cat taking photo of me from different angles | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0006464698 | 0.7358 | 57 | A long of whereas actually for example he wanted meat to tell him my binking count passwork and he cat taking photo of me from different angles | wer | 0.2963 |
| 160 | causal_full_asr | emilia_zh_0006610442 | 0.9760 | 3 | 我觉得这这这有什么有意思就是这种 | cer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0006610442 | 0.9641 | 5 | 我觉得这这这件事蛮有意思就是这种 | cer | 0.2222 |
| 640 | causal_full_asr | emilia_zh_0006610442 | 0.9461 | 8 | 嗯我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0006610442 | 0.9461 | 8 | 嗯我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006713619 | 0.9836 | 5 | 不管不那些心慢慢你要护馆赶紧护馆起来在给你你们三和时间护光晚上下播了啊 | cer | 0.5405 |
| 320 | streaming_asr | emilia_zh_0006713619 | 0.9605 | 10 | 不管<\|write_generate\|><\|cmn\|><\|start_content\|>那些新人们你们要互关赶紧互关起来在给你你们三和时间互关我要下播了啊 | cer | 1.3243 |
| 640 | streaming_asr | emilia_zh_0006713619 | 0.9572 | 12 | 不管啊啊那些新人们里面要护光赶紧护光材料在给你你们三分钟时间护光我要下播了啊 | cer | 0.4324 |
| 1280 | streaming_asr | emilia_zh_0006713619 | 0.9737 | 7 | 不管啊啊那些新人们里面要护光赶紧护光材料在给你你们三个人时间护光我要下播了啊 | cer | 0.4865 |
| 160 | streaming_asr | emilia_zh_0006940509 | 0.9770 | 3 | 宏伟都是的街道很快变不了红云 | cer | 0.2667 |
| 320 | streaming_asr | emilia_zh_0006940509 | 0.9770 | 4 | 宏伟都市的街道很快变不了红云 | cer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0006940509 | 0.9655 | 5 | 宏伟都市的的街道很快变不了红云 | cer | 0.2667 |
| 1280 | streaming_asr | emilia_zh_0006940509 | 0.9655 | 5 | 宏伟都市的的街道很快便不了红云 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0007124790 | 0.9718 | 8 | 你好请问这是去老人人家的路吧男人看了小军也 | cer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0007124790 | 0.9774 | 5 | 你好请问这是去老人人家的路吧男人看了小军也 | cer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0007124790 | 0.9859 | 4 | 你好请问这是去老人人家的路吧男人看了小军也 | cer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0007124790 | 0.9859 | 4 | 你好请问这个是去老人自家的路吧男人看了小军也 | cer | 0.2273 |
| 160 | streaming_asr | emilia_zh_0007461662 | 0.9673 | 12 | 把那么这个是导师感兴趣的领域啊第二个学自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0007461662 | 0.9673 | 13 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 640 | streaming_asr | emilia_zh_0007461662 | 0.9633 | 13 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 1280 | streaming_asr | emilia_zh_0007461662 | 0.9673 | 12 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 160 | streaming_asr | emilia_zh_0007690753 | 0.9763 | 12 | 可能市场每次行想了是仅仅了一年搞老二零零四年最漫长的下跌再度发生上周宗旨再继续出现跃现五连运的悲惨场面 | cer | 0.2941 |
| 320 | streaming_asr | emilia_zh_0007690753 | 0.9729 | 13 | 可能市场每次跟想的是仅仅了一年搞老二零零四年最漫长的下跌再度发生上升宗旨再再次出现跃现五连运的悲惨场面 | cer | 0.2549 |
| 640 | streaming_asr | emilia_zh_0007690753 | 0.9712 | 16 | 可能市场每次生想的是仅仅了一年搞老二零零四年最漫长的下跌再度发生上升终止再次出现跃现五连运的悲惨场面 | cer | 0.2549 |
| 1280 | streaming_asr | emilia_zh_0007690753 | 0.9661 | 18 | 可能市场每次生想的是仅仅了一年搞老二零零四年最漫长的下跌再度发生上升终止再次出现跃现五连运的悲惨场面 | cer | 0.2549 |
| 160 | streaming_asr | EN_B00083_S02942_W000001 | 0.7588 | 50 | It is going to be in re starts as frenching be a many course for any writers who want to learn how to great and polished their own serious stories | wer | 0.3226 |
| 320 | streaming_asr | EN_B00083_S02942_W000001 | 0.7354 | 57 | It is going to be be restores as should be a many course for any writers who want to learn how to great and polished their own serious stories | wer | 0.3226 |
| 640 | streaming_asr | EN_B00083_S02942_W000001 | 0.7400 | 55 | It is going to be a a restores as frenching be a many course for any writers who want to learn how to great and polished their own serious stories | wer | 0.2903 |
| 1280 | streaming_asr | EN_B00083_S02942_W000001 | 0.7354 | 57 | It is going to be a a restores as Chenando be a many course for any writers who want to learn how to great and polished their own serious stories | wer | 0.2903 |
| 160 | causal_full_asr | EN_B00083_S03017_W000001 | 0.8217 | 90 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop. So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success. I've tried to do that throughout my relationship with them. | wer | 0.1053 |
| 320 | causal_full_asr | EN_B00083_S03017_W000001 | 0.8248 | 89 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success if I try to do that throughout my relationship with them | wer | 0.1053 |
| 640 | causal_full_asr | EN_B00083_S03017_W000001 | 0.7920 | 109 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success if I try to do that throughout my relationship with them | wer | 0.1053 |
| 1280 | causal_full_asr | EN_B00083_S03017_W000001 | 0.7951 | 109 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would have built success if I try to do that throughout my relationship with them | wer | 0.1053 |
| 160 | streaming_asr | EN_B00013_S06799_W000009 | 0.8447 | 36 | I with that that can have better model is your safe fearness and help any users to have better d<\|glm_semantic_8776\|>ability appropriability tr<\|glm_semantic_7494\|> | wer | 0.5238 |
| 320 | streaming_asr | EN_B00013_S06799_W000009 | 0.8174 | 43 | And with that we can have better model Ensure safe fearness and help and user to have better userability appropriateness trust | wer | 0.2381 |
| 640 | streaming_asr | EN_B00013_S06799_W000009 | 0.8059 | 45 | And with that we can have better model assurance Safety fearness and help and user to have better userability appropriateness trust | wer | 0.2381 |
| 1280 | streaming_asr | EN_B00013_S06799_W000009 | 0.8105 | 43 | And with that we can have better model Ensure safe fearness and help and user to have better userability appropriability trust | wer | 0.2381 |
| 160 | causal_full_asr | EN_B00064_S09941_W000015 | 0.6684 | 65 | Interbedent Buddhism models are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.1304 |
| 320 | causal_full_asr | EN_B00064_S09941_W000015 | 0.6449 | 73 | In Tibetan Buddhism models are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 640 | causal_full_asr | EN_B00064_S09941_W000015 | 0.6266 | 73 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 1280 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5927 | 79 | In Tibetan Buddhism Mahas are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 160 | streaming_asr | EN_B00048_S03599_W000339 | 0.9125 | 17 | I turned around I my shriek froze the blood of everyone close by | wer | 0.0769 |
| 320 | streaming_asr | EN_B00048_S03599_W000339 | 0.8875 | 22 | At turned around I my shriek froze the blood of everyone close by | wer | 0.1538 |
| 640 | streaming_asr | EN_B00048_S03599_W000339 | 0.8625 | 24 | At turned around I my shriek froze the blood of everyone close by | wer | 0.1538 |
| 1280 | streaming_asr | EN_B00048_S03599_W000339 | 0.8688 | 22 | At turned around I my shriek froze the blood of everyone close by | wer | 0.1538 |
| 160 | streaming_asr | EN_B00058_S03125_W000010 | 0.7299 | 95 | When from comes in contacted with base it changes are it's called from yell to red indicating the to sobby solution is base at was why termor extinct turns red when a comes and contacted with any came of base | wer | 0.4762 |
| 320 | streaming_asr | EN_B00058_S03125_W000010 | 0.7156 | 97 | When promised comes in contacted with base it changes are it's call the from yell to red indicating the to sobri solution is base that was why termor extinct turns red when a comes and contacted with any came of base | wer | 0.4762 |
| 640 | streaming_asr | EN_B00058_S03125_W000010 | 0.6919 | 106 | When termor comes in contacted with base it changes are it's call the from yell to red indicating the to sobri solution is bas that was why termor extained turns red when a comes and contact with any kind of base | wer | 0.4524 |
| 1280 | streaming_asr | EN_B00058_S03125_W000010 | 0.6682 | 113 | When promised comes in contacted with base it changes are it's call the from yell to red indicating the to sobri solution is bas that was why termor extained turns red when a comes and contact with any kind of base | wer | 0.4524 |
| 160 | causal_full_asr | EN_B00058_S06163_W000045 | 0.7474 | 52 | So you original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that took place in September | wer | 0.0385 |
| 320 | causal_full_asr | EN_B00058_S06163_W000045 | 0.7577 | 51 | So your original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that's a place in September | wer | 0.0769 |
| 640 | causal_full_asr | EN_B00058_S06163_W000045 | 0.7423 | 56 | So you original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that's a place in September | wer | 0.1154 |
| 1280 | causal_full_asr | EN_B00058_S06163_W000045 | 0.7219 | 60 | So you original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that took place in September | wer | 0.0385 |
| 160 | streaming_asr | EN_B00058_S07511_W000000 | 0.6256 | 41 | Getting get everyone yeah of the house a worrying can be really top special the first to school | wer | 0.5263 |
| 320 | streaming_asr | EN_B00058_S07511_W000000 | 0.6000 | 46 | Getting everybody yeah of the house a worrying can be really top special the first to school | wer | 0.4211 |
| 640 | streaming_asr | EN_B00058_S07511_W000000 | 0.5846 | 47 | Getting everybody yet of the house the worrying can be really top special the first to school | wer | 0.3684 |
| 1280 | streaming_asr | EN_B00058_S07511_W000000 | 0.5590 | 48 | Getting get everyone yet of the house the worrying can be really top special of first to school | wer | 0.5263 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
