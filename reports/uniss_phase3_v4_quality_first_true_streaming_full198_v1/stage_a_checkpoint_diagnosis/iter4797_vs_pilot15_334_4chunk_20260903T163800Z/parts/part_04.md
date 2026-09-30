# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9583**
- Weighted CTC blank ratio: **0.7029**
- Weighted streaming WER/CER: **0.1210**
- Weighted causal-full WER/CER: **0.1467**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000116742 | 0.7951 | 35 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000116742 | 0.7541 | 38 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000116742 | 0.7705 | 34 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000116742 | 0.7746 | 36 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 160 | causal_full_asr | CommonVoice_EN_0000160093 | 0.8000 | 27 | The press is on my t-negence sir said Palt | wer | 0.4444 |
| 320 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7951 | 29 | The press is a mighty engine Sir said Palt | wer | 0.1111 |
| 640 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7854 | 31 | The press is a mighty engine Sir said Palt | wer | 0.1111 |
| 1280 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7756 | 33 | The press is a mighty engine Sir said Pott | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000285524 | 0.7635 | 60 | The townships proximity to the sea favored marine trade and fish factors | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000285524 | 0.7407 | 64 | The townships pro proximity to the sea favored marine trade and fish factors | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000285524 | 0.7151 | 67 | The townships pro proximity to the sea favored marine trade and fish f factories | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000285524 | 0.7179 | 67 | The townships pro proximity to the sea favored moring tried in fish factories | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000430128 | 0.7520 | 45 | Lord talk in your retains his captancy with designed white rose bad | wer | 0.6429 |
| 320 | streaming_asr | CommonVoice_EN_0000430128 | 0.7400 | 49 | Lu Tuo can you hear his captancy designed their white rows bad | wer | 0.7857 |
| 640 | streaming_asr | CommonVoice_EN_0000430128 | 0.7280 | 54 | Lord Hawke can ear daily's his captancy designed their white rows bad | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000430128 | 0.7200 | 54 | Lord Hawke can ear daily's his captancy designed their white rows bad | wer | 0.6429 |
| 160 | streaming_asr | CommonVoice_EN_0000555853 | 0.7869 | 40 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000555853 | 0.7582 | 40 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000555853 | 0.7336 | 41 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000555853 | 0.7418 | 42 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 160 | causal_full_asr | DailyTalk_0000009768 | 0.6667 | 43 | Bring us a bottle of rumi more teeny and red wine | wer | 0.3000 |
| 320 | causal_full_asr | DailyTalk_0000009768 | 0.6398 | 45 | Bring us the bottle of rumney more teeny and red wine | wer | 0.4000 |
| 640 | causal_full_asr | DailyTalk_0000009768 | 0.6505 | 44 | Bring us the bottle of rumi more teeny and red wine | wer | 0.4000 |
| 1280 | causal_full_asr | DailyTalk_0000009768 | 0.6720 | 43 | Bring us the bottle of Remy Martinez and Redwine | wer | 0.4000 |
| 160 | streaming_asr | LibriSpeech_0000068252 | 0.6833 | 165 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 320 | streaming_asr | LibriSpeech_0000068252 | 0.6725 | 170 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 640 | streaming_asr | LibriSpeech_0000068252 | 0.6523 | 175 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 1280 | streaming_asr | LibriSpeech_0000068252 | 0.6442 | 173 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousand lepers have been collected | wer | 0.0882 |
| 160 | streaming_asr | LibriSpeech_0000158589 | 0.6466 | 178 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Blackett for after receiving the affectionate greetings of near the whole company | wer | 0.1944 |
| 320 | streaming_asr | LibriSpeech_0000158589 | 0.6260 | 183 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plathet for after receiving the affectionate greetings of nearly the whole company | wer | 0.1944 |
| 640 | streaming_asr | LibriSpeech_0000158589 | 0.6192 | 179 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plattet for after receiving the affectionate greetings of near the whole company | wer | 0.2222 |
| 1280 | streaming_asr | LibriSpeech_0000158589 | 0.6315 | 177 | I always did think Mary Harris resembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plattet for after receiving the affectionate greetings of nearly the whole company | wer | 0.1667 |
| 160 | causal_full_asr | LibriSpeech_0000271146 | 0.6321 | 101 | But it is full of angel leaving the mounds enormous bones which the Indians attribute some gigantic race which lived in a past age | wer | 0.3043 |
| 320 | causal_full_asr | LibriSpeech_0000271146 | 0.6244 | 106 | But it is full of Angela living the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.2609 |
| 640 | causal_full_asr | LibriSpeech_0000271146 | 0.6347 | 101 | For it is full of anteliving mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.1304 |
| 1280 | causal_full_asr | LibriSpeech_0000271146 | 0.6244 | 105 | For it is full of anteliving the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.1739 |
| 160 | streaming_asr | VCTK_0000029186 | 0.8079 | 24 | It's going to be new challenge | wer | 0.1429 |
| 320 | streaming_asr | VCTK_0000029186 | 0.8146 | 21 | It's going to be new challenge | wer | 0.1429 |
| 640 | streaming_asr | VCTK_0000029186 | 0.8079 | 23 | It's going to be new challenge | wer | 0.1429 |
| 1280 | streaming_asr | VCTK_0000029186 | 0.8212 | 21 | It's going to be new challenge | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004064952 | 0.7282 | 50 | 也在国老下依然在我的脑子里甚至嘴上的关 | cer | 0.4211 |
| 320 | streaming_asr | emilia_zh_0004064952 | 0.7179 | 48 | 要在国若下依然在我的脑子里闪着贼亮的光 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0004064952 | 0.7179 | 52 | 压在国虏下依然在我的脑子里闪着贼亮的光 | cer | 0.1053 |
| 1280 | streaming_asr | emilia_zh_0004064952 | 0.7077 | 53 | 压在鼓篓下依然在我的脑子里闪着嘴亮的光 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0004270141 | 0.7736 | 67 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 320 | streaming_asr | emilia_zh_0004270141 | 0.7673 | 68 | 他的笑山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0769 |
| 640 | streaming_asr | emilia_zh_0004270141 | 0.7547 | 71 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 1280 | streaming_asr | emilia_zh_0004270141 | 0.7484 | 73 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 160 | causal_full_asr | emilia_zh_0004472296 | 0.8292 | 49 | 里面的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 320 | causal_full_asr | emilia_zh_0004472296 | 0.8385 | 47 | 明艳的不对啊这是赵白书带我给你的面纱您 | cer | 0.4211 |
| 640 | causal_full_asr | emilia_zh_0004472296 | 0.8323 | 50 | 明艳的不对啊这是赵白书带我跟你念上音念 | cer | 0.3684 |
| 1280 | causal_full_asr | emilia_zh_0004472296 | 0.8447 | 47 | 宁愿的不对啊这是赵白书待我给你念上音 | cer | 0.2632 |
| 160 | streaming_asr | emilia_zh_0004519646 | 0.7742 | 47 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004519646 | 0.7604 | 47 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004519646 | 0.7558 | 46 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004519646 | 0.7604 | 47 | 身处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004732647 | 0.5436 | 99 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 320 | streaming_asr | emilia_zh_0004732647 | 0.5369 | 101 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 640 | streaming_asr | emilia_zh_0004732647 | 0.5336 | 96 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 1280 | streaming_asr | emilia_zh_0004732647 | 0.5302 | 98 | He wanted to find out what made the noiseers that he people were afraid of and there were was nothing in the caves to tell him | wer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0004732654 | 0.5809 | 72 | But quite suddenly the fire ahead gave a pale flicker and went down and the clocking ceased | wer | 0.0588 |
| 320 | causal_full_asr | emilia_zh_0004732654 | 0.5643 | 74 | But quite suddenly the far ahead gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004732654 | 0.5560 | 74 | But quite suddenly the firehead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 1280 | causal_full_asr | emilia_zh_0004732654 | 0.5436 | 74 | But quite suddenly the firehead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 160 | streaming_asr | emilia_zh_0004804709 | 0.6490 | 41 | render the unsafe for the them to venture outside the town | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004804709 | 0.6358 | 41 | render the unsafe for them to venture outside the town | wer | 0.1000 |
| 640 | streaming_asr | emilia_zh_0004804709 | 0.5894 | 47 | render the unsafe for them to venture outside the town | wer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004804709 | 0.6026 | 46 | render it unsaved for them to venture outside the town | wer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0004927443 | 0.6878 | 59 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004927443 | 0.6787 | 62 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004927443 | 0.6968 | 57 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004927443 | 0.6878 | 58 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005094550 | 0.7323 | 107 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005094550 | 0.7279 | 113 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005094550 | 0.7212 | 116 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005094550 | 0.7146 | 116 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005347375 | 0.7212 | 57 | 但我们却不知道他是什么只能确定他存在一定效应 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0005347375 | 0.7389 | 56 | 但我们却不知道他是什么只能确定他存在一粒小叶 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0005347375 | 0.7301 | 59 | 但我们却不知道他是什么只能确定他存在以利效应 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0005347375 | 0.7301 | 60 | 但我们却不知道他是什么只能确定他存在引力效应 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0005421693 | 0.7273 | 39 | 这物件东西是它昨晚制作的交通工具 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0005421693 | 0.7208 | 41 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005421693 | 0.7143 | 43 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005421693 | 0.7078 | 43 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005748506 | 0.7388 | 135 | 这是我后来我觉得我理解这个意思他其实告诉你你你不要想那么多就是你有这一个呃想法的时候你你去做就好了你先去体验 | cer | 0.1176 |
| 320 | streaming_asr | emilia_zh_0005748506 | 0.7388 | 133 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 640 | streaming_asr | emilia_zh_0005748506 | 0.7285 | 137 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 1280 | streaming_asr | emilia_zh_0005748506 | 0.7251 | 138 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 160 | streaming_asr | emilia_zh_0005928718 | 0.7964 | 53 | 很好还是一直关着呢船长现在是他重要的一个牌 | cer | 0.1364 |
| 320 | streaming_asr | emilia_zh_0005928718 | 0.7891 | 54 | 船行还是一直关着呢船长现在是他重要的一个牌 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0005928718 | 0.7782 | 55 | 船行还是一只关着呢船长现在是他重要的一个牌 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0005928718 | 0.7818 | 55 | 船行还是一个关着嗯船长现在是他重要的一个牌 | cer | 0.1364 |
| 160 | causal_full_asr | emilia_zh_0005960324 | 0.7287 | 93 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005960324 | 0.7313 | 93 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005960324 | 0.7235 | 96 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005960324 | 0.7364 | 91 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006174150 | 0.7669 | 56 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006174150 | 0.7632 | 59 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006174150 | 0.7519 | 62 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006174150 | 0.7444 | 62 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006350510 | 0.6187 | 73 | John said they have lived a very isolated life in India and he was terribly shy of women | wer | 0.1111 |
| 320 | streaming_asr | emilia_zh_0006350510 | 0.6115 | 77 | Draw said the had lived a very isolated life in India and he was terribly shy of women | wer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0006350510 | 0.6043 | 79 | John said they had lived a very isolated life in India and he was terribly shy of women | wer | 0.0556 |
| 1280 | streaming_asr | emilia_zh_0006350510 | 0.6115 | 79 | John said the had lifted very isolated life in India and he was terribly shy of women | wer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0006405437 | 0.7517 | 77 | The two little boys can't have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 320 | causal_full_asr | emilia_zh_0006405437 | 0.7244 | 79 | The two little boys can have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 640 | causal_full_asr | emilia_zh_0006405437 | 0.7130 | 81 | The two little boys can have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 1280 | causal_full_asr | emilia_zh_0006405437 | 0.6925 | 81 | The two little boys can't have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 160 | streaming_asr | emilia_zh_0006435497 | 0.6579 | 190 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in middle you will see see and here in a preciation of what end log again is AKA merch sort today | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006435497 | 0.6327 | 200 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in the middle you will see see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1556 |
| 640 | streaming_asr | emilia_zh_0006435497 | 0.6289 | 194 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in the middle you will see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0006435497 | 0.6264 | 192 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubble sort and in the middle you will see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006598681 | 0.7033 | 82 | 就是在公牛大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0303 |
| 320 | streaming_asr | emilia_zh_0006598681 | 0.6800 | 87 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006598681 | 0.6700 | 89 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006598681 | 0.6867 | 86 | 就是在公有大学学习半点之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0303 |
| 160 | streaming_asr | emilia_zh_0006819602 | 0.7655 | 81 | 有半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0645 |
| 320 | streaming_asr | emilia_zh_0006819602 | 0.7628 | 83 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 640 | streaming_asr | emilia_zh_0006819602 | 0.7628 | 84 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 1280 | streaming_asr | emilia_zh_0006819602 | 0.7574 | 85 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 160 | streaming_asr | emilia_zh_0007120510 | 0.7653 | 41 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007120510 | 0.7551 | 43 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007120510 | 0.7449 | 45 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007120510 | 0.7500 | 45 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007353483 | 0.8192 | 53 | 普通的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007353483 | 0.8222 | 57 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 640 | streaming_asr | emilia_zh_0007353483 | 0.8251 | 57 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0007353483 | 0.8222 | 58 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 160 | causal_full_asr | emilia_zh_0007353988 | 0.7675 | 111 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0638 |
| 320 | causal_full_asr | emilia_zh_0007353988 | 0.7509 | 123 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0638 |
| 640 | causal_full_asr | emilia_zh_0007353988 | 0.7417 | 129 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0007353988 | 0.7454 | 127 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007555536 | 0.7370 | 67 | 其你将其余的三个房间里的收集全部版过来我要细细的查阅一下 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0007555536 | 0.7163 | 75 | 请你将其余的三个房间里的收集全部版过来我要细细的查阅一下 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007555536 | 0.7093 | 75 | 请你将其余的三个房间里的收集全部把过来我要细细的查阅一下 | cer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0007555536 | 0.6955 | 77 | 请你将其余的三个房间里的收集全部把过来我要细细的查阅一下 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007788542 | 0.7517 | 66 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 320 | streaming_asr | emilia_zh_0007788542 | 0.7483 | 64 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 640 | streaming_asr | emilia_zh_0007788542 | 0.7343 | 66 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 1280 | streaming_asr | emilia_zh_0007788542 | 0.7343 | 67 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 160 | streaming_asr | EN_B00043_S02661_W000018 | 0.7283 | 48 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 320 | streaming_asr | EN_B00043_S02661_W000018 | 0.7208 | 52 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 640 | streaming_asr | EN_B00043_S02661_W000018 | 0.6943 | 55 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00043_S02661_W000018 | 0.6755 | 57 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7288 | 92 | That's the image Let's go through the main door Turn F to walk down to the end of the car to the end of the car and it's the last car on the right | wer | 0.5385 |
| 320 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7076 | 97 | That's good Miss Just go through the main door Turn F work down to the end of the car to the end and it's the last car on the right | wer | 0.3462 |
| 640 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7076 | 99 | Certainly miss Just go through the main door Turn FET walk down to the end of the car to the end and it's the last car on the right | wer | 0.2308 |
| 1280 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7119 | 100 | Certainly miss Just go through the main door Turn left walk down to the end of the car to the end and it's the last car on the right | wer | 0.1923 |
| 160 | streaming_asr | EN_B00048_S01234_W000037 | 0.5818 | 46 | And in our case we're not sure this animal is going to be a dog | wer | 0.0625 |
| 320 | streaming_asr | EN_B00048_S01234_W000037 | 0.5212 | 54 | And in our case we're not sure if the animal is going to be a dog | wer | 0.0625 |
| 640 | streaming_asr | EN_B00048_S01234_W000037 | 0.4970 | 57 | And in our case we're not sure if this animal is going to be a dog | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S01234_W000037 | 0.5030 | 59 | And in our case we're not sure if this animal is going to be a dog | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S08821_W000040 | 0.6080 | 80 | Are you in participating in a fun raise of this weekend a little like to borrow the van If possible | wer | 0.4444 |
| 320 | streaming_asr | EN_B00048_S08821_W000040 | 0.5914 | 84 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 640 | streaming_asr | EN_B00048_S08821_W000040 | 0.5847 | 81 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00048_S08821_W000040 | 0.5914 | 82 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7161 | 45 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7203 | 44 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7203 | 44 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S08821_W000047 | 0.6907 | 46 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S04431_W000023 | 0.6957 | 36 | Use blood head kept all parts of his body warm | wer | 0.2000 |
| 320 | streaming_asr | EN_B00058_S04431_W000023 | 0.6359 | 43 | His blood had kept all parts of his body warm | wer | 0.0000 |
| 640 | streaming_asr | EN_B00058_S04431_W000023 | 0.6630 | 36 | His blood had kept all parts of his body warm | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00058_S04431_W000023 | 0.6467 | 38 | His blood had kept all parts of his body warm | wer | 0.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
