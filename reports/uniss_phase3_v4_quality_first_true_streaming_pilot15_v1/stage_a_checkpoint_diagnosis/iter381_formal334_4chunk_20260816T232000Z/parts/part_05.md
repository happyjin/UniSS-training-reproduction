# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9261**
- Weighted CTC blank ratio: **0.8753**
- Weighted streaming WER/CER: **0.2519**
- Weighted causal-full WER/CER: **0.1957**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000126367 | 0.8179 | 29 | He had always said grids to fishing for exicism and charity the the port | wer | 0.5385 |
| 320 | streaming_asr | CommonVoice_EN_0000126367 | 0.7857 | 31 | He had always a grids to <\|glm_semantic_13920\|>atrea for exicism and charity to to the poorer | wer | 0.4615 |
| 640 | streaming_asr | CommonVoice_EN_0000126367 | 0.7857 | 25 | He had always a grids of bitter support exicism and charity to to the poorer | wer | 0.5385 |
| 1280 | streaming_asr | CommonVoice_EN_0000126367 | 0.7643 | 27 | He had always that grew of bitter for exicism and charity to to the poorer | wer | 0.5385 |
| 160 | causal_full_asr | CommonVoice_EN_0000262462 | 0.8545 | 24 | It is across the Columbia River from Wint salmon Washington | wer | 0.1000 |
| 320 | causal_full_asr | CommonVoice_EN_0000262462 | 0.8436 | 22 | It is across the Columbia River from Wensham in Washington | wer | 0.2000 |
| 640 | causal_full_asr | CommonVoice_EN_0000262462 | 0.8618 | 19 | It is across the Columbia River from Wiesam in Washington | wer | 0.2000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000262462 | 0.8473 | 21 | It is across the Columbia River from Wistam in Washington | wer | 0.2000 |
| 160 | streaming_asr | CommonVoice_EN_0000286138 | 0.8466 | 28 | The bore reminded the old man that he had said something about hidden traged | wer | 0.1429 |
| 320 | streaming_asr | CommonVoice_EN_0000286138 | 0.8275 | 27 | The boy reminded the old man that he had said something about hidden t treasures | wer | 0.1429 |
| 640 | streaming_asr | CommonVoice_EN_0000286138 | 0.7955 | 33 | The boy remind of the old man that he had said something about hidden trasure | wer | 0.2143 |
| 1280 | streaming_asr | CommonVoice_EN_0000286138 | 0.7923 | 32 | The boy reminded the old man that he had said something about hidden trasure | wer | 0.0714 |
| 160 | streaming_asr | CommonVoice_EN_0000433992 | 0.8561 | 20 | Nobody want the discuss how you all end up trap and kind len't box | wer | 0.6429 |
| 320 | streaming_asr | CommonVoice_EN_0000433992 | 0.8450 | 23 | Nobody want to the discuss how you all end up trap in and kind lunch box | wer | 0.6429 |
| 640 | streaming_asr | CommonVoice_EN_0000433992 | 0.8413 | 27 | Nobody wanted the discuss how you all end got trap in a kind lunch pox | wer | 0.5714 |
| 1280 | streaming_asr | CommonVoice_EN_0000433992 | 0.8229 | 29 | Nobody wanted the discuss how your all in the up trap in a kind let's box | wer | 0.5714 |
| 160 | streaming_asr | CommonVoice_EN_0000593898 | 0.8250 | 28 | In reason years seen George's as spent be it medic costal routes | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000593898 | 0.8071 | 28 | In reason years Saint George's as spent be it medic costal roots | wer | 0.5833 |
| 640 | streaming_asr | CommonVoice_EN_0000593898 | 0.7786 | 30 | In reason years Saint George's as spent beyond it medic costal roots | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000593898 | 0.8000 | 31 | In reasoning years Saint George's as spanned beyond it's medic costal roots | wer | 0.5000 |
| 160 | causal_full_asr | DailyTalk_0000010084 | 0.7578 | 16 | Good morning sir What can I do for you | wer | 0.1111 |
| 320 | causal_full_asr | DailyTalk_0000010084 | 0.7500 | 16 | Good morning sir What can I do for you | wer | 0.1111 |
| 640 | causal_full_asr | DailyTalk_0000010084 | 0.7656 | 18 | The morning sir what can I do for you | wer | 0.2222 |
| 1280 | causal_full_asr | DailyTalk_0000010084 | 0.7656 | 15 | The morning sir what can I do for you | wer | 0.2222 |
| 160 | streaming_asr | LibriSpeech_0000068284 | 0.8838 | 37 | Is not a world to facts but only of the meaning of facts It say point to view for judging facts It appertains to I different ology | wer | 0.1786 |
| 320 | streaming_asr | LibriSpeech_0000068284 | 0.8644 | 43 | Is not a world to facts but only of the meaning of facts It say point to view for judging facts It appertains to I different ology | wer | 0.1786 |
| 640 | streaming_asr | LibriSpeech_0000068284 | 0.8715 | 40 | Is not a world to facts but only of the meaning of facts It say point to view for judging facts It appertains to I different ology | wer | 0.1786 |
| 1280 | streaming_asr | LibriSpeech_0000068284 | 0.8609 | 42 | Is not a world do facts but only of the meaning of facts It say point to view for judging facts It appertains to I different ology | wer | 0.1786 |
| 160 | streaming_asr | LibriSpeech_0000158773 | 0.7732 | 92 | Oh men joyed placings of liberty of believed that by using my capable my my make little -income and I again when money on security relined on my thrift my judgment and my knowledge of world I chose is business in preference all all those | wer | 0.3878 |
| 320 | streaming_asr | LibriSpeech_0000158773 | 0.7369 | 110 | Oh men joyed blessings of liberty I've believed that by using my capable my my make a little -income and I be again when money on security relying on my thrift my judgment and my knowledge of the world I chose is business in preference to all all those | wer | 0.3061 |
| 640 | streaming_asr | LibriSpeech_0000158773 | 0.7289 | 107 | All men joyed blessings of liberty I have believed that by utilizing my capital my my make a little -income and I again when money on security reliang my thrift my judgment and my knowledge of the world I chose his business in preference to all all those | wer | 0.2857 |
| 1280 | streaming_asr | LibriSpeech_0000158773 | 0.6859 | 126 | All men and joy the blessings of liberty of believed that by using my capital that my make a little income and I again when money on security relined on my thrift my judgment and my knowledge of the world I chose his business in preference to all all those | wer | 0.2653 |
| 160 | causal_full_asr | VCTK_0000006143 | 0.8101 | 20 | Being captain of this club is fantastic | wer | 0.0000 |
| 320 | causal_full_asr | VCTK_0000006143 | 0.8101 | 20 | Being captain of this club is fantastic | wer | 0.0000 |
| 640 | causal_full_asr | VCTK_0000006143 | 0.7911 | 21 | Being captain of this club is fantastic | wer | 0.0000 |
| 1280 | causal_full_asr | VCTK_0000006143 | 0.8101 | 19 | Being captain of this club is fantastic | wer | 0.0000 |
| 160 | streaming_asr | VCTK_0000029362 | 0.9375 | 8 | The but these exhausted | wer | 0.7500 |
| 320 | streaming_asr | VCTK_0000029362 | 0.9187 | 8 | The my body these exhausted | wer | 0.5000 |
| 640 | streaming_asr | VCTK_0000029362 | 0.8875 | 10 | I'm body is exhausted | wer | 0.2500 |
| 1280 | streaming_asr | VCTK_0000029362 | 0.8812 | 11 | My body is exhausted | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004111317 | 0.9770 | 17 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够神我们可以找到水的堆让我们选择一个地点可开始我妈妈 | cer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0004111317 | 0.9758 | 16 | 我有个主意我们为什么不挖坑潮水呢哦是的我相信如果我们挖的足够神我们可以找到水的堆让我们选择一个地点可开始我妈妈 | cer | 0.1607 |
| 640 | streaming_asr | emilia_zh_0004111317 | 0.9710 | 16 | 我有个主意我们为什么不挖坑找水呢哦湿的的我相信如果我们挖的足够神我们可以找到水的堆让我们选择一个地点和开始我妈妈 | cer | 0.1786 |
| 1280 | streaming_asr | emilia_zh_0004111317 | 0.9746 | 16 | 我有个主意我们为什么不挖坑找水呢哦湿的的我相信如果我们挖的足够神我们可以找到水的堆热我们选择一个地点可开始我爸爸 | cer | 0.1964 |
| 160 | streaming_asr | emilia_zh_0004270182 | 0.9554 | 13 | 知道从前埋藏的已经很久很久的过的赶走工具螃蟹的老鼠们 | cer | 0.2593 |
| 320 | streaming_asr | emilia_zh_0004270182 | 0.9405 | 16 | 找到从前埋藏的郁金香很久很久的过的打扫工具同学的老鼠们 | cer | 0.3704 |
| 640 | streaming_asr | emilia_zh_0004270182 | 0.9375 | 15 | 找到从前埋藏的郁金香很久很久的过的打扫工具同学的老鼠们 | cer | 0.3704 |
| 1280 | streaming_asr | emilia_zh_0004270182 | 0.9405 | 14 | 找到从前埋藏的郁金香很久很久的过的赶走工具同学的老鼠们 | cer | 0.2963 |
| 160 | streaming_asr | emilia_zh_0004621436 | 0.8357 | 23 | But mother road and asked me they could possibly come is paying Guess | wer | 0.3571 |
| 320 | streaming_asr | emilia_zh_0004621436 | 0.8028 | 26 | They mother road and asked me they could possibly come is paying Guess | wer | 0.3571 |
| 640 | streaming_asr | emilia_zh_0004621436 | 0.7700 | 29 | The mother road and asked me they could possibly come is paying Guests | wer | 0.2857 |
| 1280 | streaming_asr | emilia_zh_0004621436 | 0.7887 | 26 | The mother road and asked me they could partly come is paying Guests | wer | 0.3571 |
| 160 | causal_full_asr | emilia_zh_0004621493 | 0.8857 | 14 | He's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0004621493 | 0.8762 | 16 | He's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 640 | causal_full_asr | emilia_zh_0004621493 | 0.8048 | 22 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 1280 | causal_full_asr | emilia_zh_0004621493 | 0.8048 | 23 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 160 | streaming_asr | emilia_zh_0004754634 | 0.8627 | 12 | Perhaps maybe of second joke suggested the jack door | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004754634 | 0.8105 | 18 | Perhaps maybe of second joke suggested the jack door | wer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0004754634 | 0.7647 | 21 | Perhaps every of second jerk suggested the jack door | wer | 0.3000 |
| 1280 | streaming_asr | emilia_zh_0004754634 | 0.7778 | 20 | Perhaps there a second jerk suggested the jack door | wer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0004841011 | 0.8279 | 22 | I all the users with channing and resuracting had had friend count | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0004841011 | 0.7907 | 24 | And all though uses with challenging and resurrection had had from camps | wer | 0.6923 |
| 640 | streaming_asr | emilia_zh_0004841011 | 0.7209 | 33 | And all those users with churning and resuracting had had from count | wer | 0.5385 |
| 1280 | streaming_asr | emilia_zh_0004841011 | 0.6791 | 35 | And all those users which turning and res<\|glm_semantic_12758\|>ecting had had from count | wer | 0.4615 |
| 160 | causal_full_asr | emilia_zh_0004841554 | 0.8016 | 28 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 320 | causal_full_asr | emilia_zh_0004841554 | 0.7588 | 33 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004841554 | 0.7549 | 35 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 1280 | causal_full_asr | emilia_zh_0004841554 | 0.7665 | 31 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0004927721 | 0.9873 | 3 | 一个人只有在臣服于于这个能量时才能了解他啊 | cer | 0.1579 |
| 320 | streaming_asr | emilia_zh_0004927721 | 0.9873 | 2 | 一个人只有在臣服于之国能量时才能了解他啊 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0004927721 | 0.9831 | 3 | 一个人只有在臣服于之国能量时才能了解他啊 | cer | 0.1579 |
| 1280 | streaming_asr | emilia_zh_0004927721 | 0.9746 | 4 | 一个人持有在臣服于之国能量时才能了解他啊 | cer | 0.2105 |
| 160 | streaming_asr | emilia_zh_0005181378 | 0.9545 | 6 | 不是客气你这个我的基本上其实你带回去 | cer | 0.3889 |
| 320 | streaming_asr | emilia_zh_0005181378 | 0.9432 | 8 | 不是客气你记得我的几本书其实你带回去 | cer | 0.2778 |
| 640 | streaming_asr | emilia_zh_0005181378 | 0.9489 | 7 | 不是客气你记得我的基本上其实你带回去 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0005181378 | 0.9318 | 9 | 不是客气你记得我的几本书其实你带回去 | cer | 0.2778 |
| 160 | streaming_asr | emilia_zh_0005507035 | 0.9679 | 5 | 那么政府还会收回这个决定了大的很担心 | cer | 0.2857 |
| 320 | streaming_asr | emilia_zh_0005507035 | 0.9733 | 5 | 那么本政府还会收回这个决定了大都很担心 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005507035 | 0.9626 | 7 | 那么本政府还会收回这个决定了大的很关心 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005507035 | 0.9412 | 9 | 那么日本政府还会收回这个决定了大都很关心 | cer | 0.0952 |
| 160 | causal_full_asr | emilia_zh_0005600573 | 0.9851 | 3 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 320 | causal_full_asr | emilia_zh_0005600573 | 0.9888 | 1 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005600573 | 0.9851 | 2 | 参与的话是一千多然后直播间里面是十呃 | cer | 0.0556 |
| 1280 | causal_full_asr | emilia_zh_0005600573 | 0.9851 | 3 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 160 | streaming_asr | emilia_zh_0005749601 | 0.9716 | 9 | 是什么刚才说的那个五点啊就五点配套的在稍微把它他总结下了花第一点就是认知功能不足 | cer | 0.2727 |
| 320 | streaming_asr | emilia_zh_0005749601 | 0.9561 | 13 | 这什么刚才说的那个五点往就五点配套的在稍微把它他同学下了花第一点就是认知功能不足 | cer | 0.3182 |
| 640 | streaming_asr | emilia_zh_0005749601 | 0.9535 | 14 | 这什么刚才说的那个五点往就五点配套的在稍微把他总结下了花第一点就是那只功能不足 | cer | 0.3409 |
| 1280 | streaming_asr | emilia_zh_0005749601 | 0.9483 | 15 | 这什么刚才说的那个五点往就五点配套的在稍微把他总结下了花第一点就是那支功能不足 | cer | 0.3409 |
| 160 | streaming_asr | emilia_zh_0005999475 | 0.9495 | 18 | 哦社会你没有感觉特别的贵因为我听说今年全球的费神之下然后现在现在变得异常的过 | cer | 0.3922 |
| 320 | streaming_asr | emilia_zh_0005999475 | 0.9518 | 17 | 然后社会你没有感觉特别的贵因为我听说今年全球的费神之下然后新加坡现在变得异常的过 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0005999475 | 0.9450 | 18 | 然后在社会上你没有感觉特别的贵因为我听说今年全球的费生之下然后新加坡现在变得异常的过 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005999475 | 0.9427 | 21 | 然后在社会上因为没有感觉特别的鬼因为我听说今年全球的卫生之下然后新加坡现在变得异常的过 | cer | 0.3725 |
| 160 | causal_full_asr | emilia_zh_0006041799 | 0.9163 | 14 | 在这个委托发生个十年前两个人就合伙开了这家面馆 | cer | 0.0833 |
| 320 | causal_full_asr | emilia_zh_0006041799 | 0.9163 | 16 | 在这个委托发生个十年前两个人就合伙开了这家面馆 | cer | 0.0833 |
| 640 | causal_full_asr | emilia_zh_0006041799 | 0.9075 | 16 | 在这个委托发生个十年前两个人就合伙开了这家面馆 | cer | 0.0833 |
| 1280 | causal_full_asr | emilia_zh_0006041799 | 0.9251 | 14 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 160 | streaming_asr | emilia_zh_0006212201 | 0.9420 | 10 | 啊还没有办法一直坚持别说的举动所以卖通你个仆人 | cer | 0.3478 |
| 320 | streaming_asr | emilia_zh_0006212201 | 0.9469 | 9 | 他还没有办法一直坚持别说的举动所以卖通了一一个仆人 | cer | 0.2609 |
| 640 | streaming_asr | emilia_zh_0006212201 | 0.9372 | 11 | 他还没有办法一直监视别墅的举动所以买通了一一个仆人 | cer | 0.0870 |
| 1280 | streaming_asr | emilia_zh_0006212201 | 0.9372 | 11 | 他还没有办法一直监视别墅的举动所以买通了一一个仆人 | cer | 0.0870 |
| 160 | streaming_asr | emilia_zh_0006366492 | 0.8663 | 39 | He did not want to sit to God his great trasure He wanted to return and let with men These riches will give me great power | wer | 0.1154 |
| 320 | streaming_asr | emilia_zh_0006366492 | 0.8625 | 43 | He did not want to sit to God his great trasure He wanted to return and little with men These riches will give me great power | wer | 0.1154 |
| 640 | streaming_asr | emilia_zh_0006366492 | 0.8588 | 41 | He did not want to sit to God his great trasure He wanted to return and little with men These riches will give me great power | wer | 0.1154 |
| 1280 | streaming_asr | emilia_zh_0006366492 | 0.8343 | 45 | He did not want to sit to God his great trasure He want to return and little with men These riches will give me great power | wer | 0.1538 |
| 160 | streaming_asr | emilia_zh_0006446583 | 0.8214 | 32 | Now will be can use MRI imaging see would is actually hopping inside joined when someone cracks someone nuckles | wer | 0.4500 |
| 320 | streaming_asr | emilia_zh_0006446583 | 0.8297 | 32 | Now we can use MRI imaging see would is actually happening inside joined when someone cracks someone nuckles | wer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0006446583 | 0.8379 | 30 | Now will be can use MRI imaging see would is actually happening inside the joined when someone cracks the nuckles | wer | 0.3500 |
| 1280 | streaming_asr | emilia_zh_0006446583 | 0.8022 | 38 | Now we can use MRI imaging see would is actually happening inside the join When someone cracks the nuckles | wer | 0.2500 |
| 160 | causal_full_asr | emilia_zh_0006502797 | 0.9354 | 26 | 我觉得这个委托本可我觉得那时候古代的时候和他在那古人都已经玩了比如说这个明代开始就玩任何的古董发明石头 | cer | 0.4098 |
| 320 | causal_full_asr | emilia_zh_0006502797 | 0.9220 | 28 | 我觉得这个委托不认可我觉得那时候古代的时候和他的时候那古人都已经玩了比如说这个明代开始就玩这个核桃嗯因为你们发明石头 | cer | 0.3607 |
| 640 | causal_full_asr | emilia_zh_0006502797 | 0.9020 | 33 | 我觉得这个胃特别不冷我觉得那时候古代的时候核核的时候那古人都已经玩了比如说这个明代开始就玩这个核核嗯那个发明石头 | cer | 0.3279 |
| 1280 | causal_full_asr | emilia_zh_0006502797 | 0.9042 | 33 | 我觉得这个委托人可我我觉得那时候古代的时候核核的时候他那古人都已经玩了比如说这个明代开始就玩这个核核嗯你发明石头 | cer | 0.3607 |
| 160 | streaming_asr | emilia_zh_0006610357 | 0.9627 | 11 | 今天我觉得是非常确实关键性非常之前但是似乎恶魔的这种做法他其实也并没有什么错我觉得 | cer | 0.1628 |
| 320 | streaming_asr | emilia_zh_0006610357 | 0.9680 | 9 | 今天我觉得是非常确实关键性非常之前但是似乎恶魔的这种做法他其实也并没有什么错我觉得 | cer | 0.1628 |
| 640 | streaming_asr | emilia_zh_0006610357 | 0.9680 | 9 | 这个觉得是非常确实关键性非常之前但是似乎恶魔的这种做法他其实也并没有什么错我觉得 | cer | 0.1628 |
| 1280 | streaming_asr | emilia_zh_0006610357 | 0.9653 | 9 | 这个觉得是非常确实关键性非常之前但是似乎恶魔的这种做法他其实也并没有什么错我觉得 | cer | 0.1628 |
| 160 | streaming_asr | emilia_zh_0006883085 | 0.9953 | 1 | 所以与终于停下自己或都已经有点昏昏欲睡了 | cer | 0.3158 |
| 320 | streaming_asr | emilia_zh_0006883085 | 0.9906 | 2 | 在与终于停下自己或都已经有点昏昏欲睡了 | cer | 0.2105 |
| 640 | streaming_asr | emilia_zh_0006883085 | 0.9906 | 2 | 在与终于停下自己或都已经有点昏昏欲睡了 | cer | 0.2105 |
| 1280 | streaming_asr | emilia_zh_0006883085 | 0.9906 | 2 | 在与终于停下自己会都已经有点昏昏欲睡了 | cer | 0.2105 |
| 160 | streaming_asr | emilia_zh_0007121845 | 0.9843 | 7 | 那么我们会也许只是依赖于一个技术一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007121845 | 0.9843 | 7 | 那么我们会也许只是依赖于一个技术一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007121845 | 0.9878 | 5 | 那么我们会也许只是依赖于一个技术一个方法一个优势或者是资源一些人力我们就去对抗市场 | cer | 0.0244 |
| 1280 | streaming_asr | emilia_zh_0007121845 | 0.9895 | 4 | 那么我们会也许只是一烂于一个技术一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0488 |
| 160 | streaming_asr | emilia_zh_0007399682 | 0.9801 | 7 | 有些上市甚至一更加严厉的态度仅我们要认清生命的脆弱告诉我我们没一个人 | cer | 0.1765 |
| 320 | streaming_asr | emilia_zh_0007399682 | 0.9779 | 8 | 有些上市甚至一更加严厉的态度谨慎我们要认清生命的脆弱告诉我我们们一个人 | cer | 0.1765 |
| 640 | streaming_asr | emilia_zh_0007399682 | 0.9757 | 9 | 有些上市甚至以更加严厉的态度谨慎我们要认清生命的脆弱告诉我我们们一个人 | cer | 0.1471 |
| 1280 | streaming_asr | emilia_zh_0007399682 | 0.9757 | 9 | 有些上市甚至以更加严厉的态度精神我们要认清生命的脆弱告诉我们们一个人 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007635379 | 0.9545 | 7 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0007635379 | 0.9495 | 8 | 所以来比是对于德国人来说是一个很重要的事实 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0007635379 | 0.9646 | 6 | 所以来比是对于德国人来说是一个很重要的事实 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0007635379 | 0.9596 | 6 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 160 | causal_full_asr | emilia_zh_0007761003 | 0.9703 | 6 | 如果你向你的团队成员分析走的更远那你就需要 | cer | 0.2273 |
| 320 | causal_full_asr | emilia_zh_0007761003 | 0.9554 | 8 | 如果你向你的团队成员分你走得更远那你就需要 | cer | 0.1364 |
| 640 | causal_full_asr | emilia_zh_0007761003 | 0.9517 | 9 | 如果你向你的团队成员分你走得更远那你就需要 | cer | 0.1364 |
| 1280 | causal_full_asr | emilia_zh_0007761003 | 0.9554 | 10 | 如果你向你的团队成员分你走得更远那你就需要 | cer | 0.1364 |
| 160 | streaming_asr | emilia_zh_0007790200 | 0.9877 | 6 | 特别指出了现在是小心私营企业卖场的最佳时机大型有实力的企业可以考虑向海外现场向海外搬迁 | cer | 0.0682 |
| 320 | streaming_asr | emilia_zh_0007790200 | 0.9859 | 7 | 特别指出了现在是小心私营企业卖场的最佳时机大型有实力的企业可以考虑向海外现场向海外搬迁 | cer | 0.0682 |
| 640 | streaming_asr | emilia_zh_0007790200 | 0.9877 | 6 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的企业可以考虑向海外现场向海外搬迁 | cer | 0.0455 |
| 1280 | streaming_asr | emilia_zh_0007790200 | 0.9894 | 3 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的企业可以考虑向海外现场向海外搬迁 | cer | 0.0455 |
| 160 | streaming_asr | EN_B00013_S05834_W000745 | 0.7155 | 51 | The first if the night could still stand through was straw asilians the broken bones in a just chest would make and lose any of build a to fight | wer | 0.4800 |
| 320 | streaming_asr | EN_B00013_S05834_W000745 | 0.7011 | 55 | The first if the night could still stand through was strong asilians the broken bones in a just chest would make him lose any of build a to fight to | wer | 0.4400 |
| 640 | streaming_asr | EN_B00013_S05834_W000745 | 0.6609 | 62 | Even if the night could still stand through was strong asilians the broken bones in a just chest would make him lose any of billibility to fight | wer | 0.2800 |
| 1280 | streaming_asr | EN_B00013_S05834_W000745 | 0.6351 | 69 | Even if the night could still stand through was strong asilians the broken bones in a is chest would make him lose any of billibility to fight | wer | 0.2800 |
| 160 | causal_full_asr | EN_B00013_S06748_W000028 | 0.8532 | 16 | Right just skateboard said his dad it's too slippery so that I'm | wer | 0.4545 |
| 320 | causal_full_asr | EN_B00013_S06748_W000028 | 0.8257 | 21 | Ride just skate forward said his dad it's two slippery so that I'm | wer | 0.6364 |
| 640 | causal_full_asr | EN_B00013_S06748_W000028 | 0.8119 | 22 | Ride your skateboard set as dad it's too slippery so that I'm | wer | 0.4545 |
| 1280 | causal_full_asr | EN_B00013_S06748_W000028 | 0.8165 | 22 | Ride your skateboard set as dad it's too slippery so that I'm | wer | 0.4545 |
| 160 | streaming_asr | EN_B00048_S02289_W000002 | 0.9107 | 12 | This kind of mussel is mostly collected to my bones | wer | 0.2000 |
| 320 | streaming_asr | EN_B00048_S02289_W000002 | 0.9179 | 14 | This kind of mussel is mostly collected to my bones | wer | 0.2000 |
| 640 | streaming_asr | EN_B00048_S02289_W000002 | 0.9107 | 13 | This kind of mussel is mostly collected to my bones | wer | 0.2000 |
| 1280 | streaming_asr | EN_B00048_S02289_W000002 | 0.9036 | 14 | This kind of muscles is mostly collected to my bones | wer | 0.2000 |
| 160 | causal_full_asr | EN_B00048_S09601_W000039 | 0.8028 | 63 | Every two months my coworkers and I would come together to discuss the new semester schedule meetings where usually held in the staff room at our institute | wer | 0.0714 |
| 320 | causal_full_asr | EN_B00048_S09601_W000039 | 0.8131 | 64 | Every two months my coworkers and I would come together to discuss the new semester schedule our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S09601_W000039 | 0.8097 | 63 | Every two months my coworkers and I would come together to discuss the new semester schedule our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S09601_W000039 | 0.8010 | 64 | Every two months my coworkers and I would come together to discuss the new semester schedule our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S09662_W000003 | 0.7459 | 73 | I'm afraid very again So what don't you listen two the Dialogue for the first time that's listen how President has a has goodbye and and will back and look get words | wer | 0.4688 |
| 320 | streaming_asr | EN_B00048_S09662_W000003 | 0.7624 | 63 | I'll pay very good So what don't you listen two the Dialogue for the first time that's listen how President house has goodbye and then will back and look get words | wer | 0.3750 |
| 640 | streaming_asr | EN_B00048_S09662_W000003 | 0.7335 | 68 | I'll pay very good So what don't you listen two the Dialogue for the first time that's listen how President has a has goodbye and don't will back and look get words | wer | 0.4375 |
| 1280 | streaming_asr | EN_B00048_S09662_W000003 | 0.7231 | 74 | On the very good So what don't you listen two the Dialogue for the first time that's listen how President has a has goodbye and and will back and look at words | wer | 0.4062 |
| 160 | streaming_asr | EN_B00058_S06165_W000019 | 0.8333 | 73 | We you you the short form why you add asymmetry after the constructor in the then you use this question text and now the first argument just pass through a west construction will be stored in the question text property | wer | 0.3415 |
| 320 | streaming_asr | EN_B00058_S06165_W000019 | 0.8090 | 84 | Weren't you used the short form why you add asymmetrical and after the constructor and then you use this question text and now the first argument which pass through a west conductor will be stored in the question text property | wer | 0.2683 |
| 640 | streaming_asr | EN_B00058_S06165_W000019 | 0.7786 | 94 | Or you used the short form where you add asymmetrical and after the constructor and then you use this question text and now the first argument which pass through a west conductor will be stored in the question text property | wer | 0.2195 |
| 1280 | streaming_asr | EN_B00058_S06165_W000019 | 0.7749 | 96 | Or you used the short form where you add asymmetrical and after the constructor in the then you use this question text and now the first argument which pass through a west conductor will be stored in the question text property | wer | 0.2683 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
