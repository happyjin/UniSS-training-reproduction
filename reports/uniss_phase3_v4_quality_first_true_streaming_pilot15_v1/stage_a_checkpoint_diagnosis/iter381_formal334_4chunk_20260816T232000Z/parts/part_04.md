# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9249**
- Weighted CTC blank ratio: **0.8735**
- Weighted streaming WER/CER: **0.2224**
- Weighted causal-full WER/CER: **0.1880**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000116742 | 0.8566 | 17 | It is one after better now mountains in sweetened | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000116742 | 0.8689 | 18 | It is one after better now mountains in sweetened | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000116742 | 0.8934 | 15 | He is one after better now mountains in sweetened | wer | 0.5556 |
| 1280 | streaming_asr | CommonVoice_EN_0000116742 | 0.8975 | 13 | It is one after better now mountains in sweetened | wer | 0.4444 |
| 160 | causal_full_asr | CommonVoice_EN_0000160093 | 0.9024 | 13 | The press is on mighty engines Sir said Palt | wer | 0.3333 |
| 320 | causal_full_asr | CommonVoice_EN_0000160093 | 0.8732 | 15 | The presence of mighty engines Sir said Paul | wer | 0.5556 |
| 640 | causal_full_asr | CommonVoice_EN_0000160093 | 0.8829 | 11 | The process of mighty engine Sir said Paul | wer | 0.4444 |
| 1280 | causal_full_asr | CommonVoice_EN_0000160093 | 0.9024 | 9 | The press is a mighty engine sir said Palt | wer | 0.1111 |
| 160 | streaming_asr | CommonVoice_EN_0000285524 | 0.8889 | 20 | The townships proximity to the see favor murray tried and fish factory | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000285524 | 0.8832 | 24 | The townships proximity to of the sea favored murmuring tried and fish factory | wer | 0.5000 |
| 640 | streaming_asr | CommonVoice_EN_0000285524 | 0.8604 | 26 | The townships proximity to of the sea favor murray tried and fish factory | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000285524 | 0.8575 | 25 | The townships proximity to of the sea favor murray tried and fish factory | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000430128 | 0.8320 | 25 | Lord talk can your tastes his captancy with design white rose bad | wer | 0.7143 |
| 320 | streaming_asr | CommonVoice_EN_0000430128 | 0.8080 | 27 | Lord talk can your tastes his captancy with design white rose bad | wer | 0.7143 |
| 640 | streaming_asr | CommonVoice_EN_0000430128 | 0.8200 | 26 | Luo talk can your daily his captancy with design white brows bad | wer | 0.8571 |
| 1280 | streaming_asr | CommonVoice_EN_0000430128 | 0.8040 | 29 | Lord to can your days his captancy with designed white rose bad | wer | 0.6429 |
| 160 | streaming_asr | CommonVoice_EN_0000555853 | 0.9057 | 14 | Shaving water Sam he said from than the cotton | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000555853 | 0.8811 | 17 | Shaving water Sam said from than the cottons | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000555853 | 0.8770 | 19 | Shaving water Sam he said from than the cotton | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000555853 | 0.8852 | 18 | Shaving water Sam he said from than the cottons | wer | 0.5000 |
| 160 | causal_full_asr | DailyTalk_0000009768 | 0.7688 | 20 | Bring us a bottle of rumney more tea and red wine | wer | 0.3000 |
| 320 | causal_full_asr | DailyTalk_0000009768 | 0.7419 | 23 | Bring us the bottle of Remi Moretini and Red wine | wer | 0.3000 |
| 640 | causal_full_asr | DailyTalk_0000009768 | 0.7204 | 26 | Bring us the bottle of Remi Mortini and Red wine | wer | 0.3000 |
| 1280 | causal_full_asr | DailyTalk_0000009768 | 0.7473 | 24 | Bring us the bottle of Remi Mortini and Red wine | wer | 0.3000 |
| 160 | streaming_asr | LibriSpeech_0000068252 | 0.8288 | 74 | The bereal of health has transformed the city of Minilla from it feet infested h<\|glm_semantic_11815\|><\|write_generate\|><\|eng\|><\|start_content\|>hug bed of contagious 疾病的 to one of most helpful city on the globe six thousand weppers have been collected | wer | 0.3529 |
| 320 | streaming_asr | LibriSpeech_0000068252 | 0.8194 | 78 | The bereal of health has transformed the city of Manila from a feet infested hock bed of contagious z diseases to one of the most helpful city on the globe six thousand weppers have been collected | wer | 0.2647 |
| 640 | streaming_asr | LibriSpeech_0000068252 | 0.8288 | 75 | The bereal of health has transformed the city of Manila from a feet infested hock bed of contagious z diseases to one of the most helpful city on the globe six thousand weppers have been collected | wer | 0.2647 |
| 1280 | streaming_asr | LibriSpeech_0000068252 | 0.8221 | 76 | The beer of health has transformed the city of Manila from a feet infested hock bed of contagious z diseases to one of the most helpful city on the globe six thousand weppers have been collected | wer | 0.2647 |
| 160 | streaming_asr | LibriSpeech_0000158589 | 0.7849 | 83 | I always did think Mary harris res resemblance the Tiny Mary hairs was pretty is child I remember so the pleasant force of mrs blackfoot for after receiving the affectionate Greetings of nearer the hold company | wer | 0.4444 |
| 320 | streaming_asr | LibriSpeech_0000158589 | 0.7877 | 87 | I always did think Mary harris resembold the Tiny Mary harris was pretty is child I remember said the pleasant voice of mrs blative for after receiving the affectionate Greetings of near the the hold company | wer | 0.3611 |
| 640 | streaming_asr | LibriSpeech_0000158589 | 0.7863 | 88 | I always did think marry harris resembold the Tiny Mary harris was pretty is child I remember said the pleasant voice of mrs blaffe for after receiving the affectionate Greetings of nearer the hold company | wer | 0.3333 |
| 1280 | streaming_asr | LibriSpeech_0000158589 | 0.7795 | 89 | I always did think Mary harris resembold the Tiny Mary hairs was pretty is child I remember said the pleasant voice of mrs blaffe for after receiving the affectionate Greetings of near the the hold company | wer | 0.3889 |
| 160 | causal_full_asr | LibriSpeech_0000271146 | 0.7979 | 48 | But it is fully unto leaving the mounds in enormous bones which the Indians attribute some gigantic race which lived in a past age | wer | 0.3913 |
| 320 | causal_full_asr | LibriSpeech_0000271146 | 0.7850 | 51 | It is fully on to leaving the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.3478 |
| 640 | causal_full_asr | LibriSpeech_0000271146 | 0.7513 | 56 | Fitters volunteered leaving the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.3478 |
| 1280 | causal_full_asr | LibriSpeech_0000271146 | 0.7202 | 60 | For it is full of anteliving the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.1739 |
| 160 | streaming_asr | VCTK_0000029186 | 0.9073 | 8 | It could to be new challeng | wer | 0.5714 |
| 320 | streaming_asr | VCTK_0000029186 | 0.9139 | 8 | It can it being new challeng | wer | 0.8571 |
| 640 | streaming_asr | VCTK_0000029186 | 0.9205 | 8 | It's can to be new challeng | wer | 0.4286 |
| 1280 | streaming_asr | VCTK_0000029186 | 0.9073 | 8 | It's can it be new challeng | wer | 0.5714 |
| 160 | streaming_asr | emilia_zh_0004064952 | 0.9538 | 6 | 也在国了下依然在我的脑子里闪嘴亮关 | cer | 0.3684 |
| 320 | streaming_asr | emilia_zh_0004064952 | 0.9436 | 8 | 也在国乐下依然在我的脑子里闪着贼俩的关 | cer | 0.2632 |
| 640 | streaming_asr | emilia_zh_0004064952 | 0.9436 | 9 | 要在国路下依然在我的脑子里闪着贼亮关 | cer | 0.2632 |
| 1280 | streaming_asr | emilia_zh_0004064952 | 0.9385 | 9 | 亚在古鲁想依然在我的脑子里闪着贼亮关 | cer | 0.3158 |
| 160 | streaming_asr | emilia_zh_0004270141 | 0.9591 | 10 | 他的小山上已经做了二十天了预知在注视着海面等着他回来 | cer | 0.1154 |
| 320 | streaming_asr | emilia_zh_0004270141 | 0.9528 | 13 | 他的像山上已经做了二十天了预支就在注视着海面等着他回来 | cer | 0.1923 |
| 640 | streaming_asr | emilia_zh_0004270141 | 0.9497 | 14 | 他的像山上已经做了二十天了预支就在注视着海面等着他回来 | cer | 0.1923 |
| 1280 | streaming_asr | emilia_zh_0004270141 | 0.9465 | 12 | 他的小山上已经做了二十天了预支就在注视着海面等着他回来 | cer | 0.1538 |
| 160 | causal_full_asr | emilia_zh_0004472296 | 0.9907 | 2 | 明眼的不对啊这是赵白叔带我给你们念上念 | cer | 0.2632 |
| 320 | causal_full_asr | emilia_zh_0004472296 | 0.9876 | 4 | 明艳的不对啊这是赵白叔带我给你们念上念 | cer | 0.2632 |
| 640 | causal_full_asr | emilia_zh_0004472296 | 0.9907 | 3 | 宁渊的不对啊这是赵白叔带我跟你们念上您 | cer | 0.3684 |
| 1280 | causal_full_asr | emilia_zh_0004472296 | 0.9907 | 2 | 明眼的不对啊这是赵白叔带我跟你们念上音 | cer | 0.3684 |
| 160 | streaming_asr | emilia_zh_0004519646 | 0.9770 | 4 | 身处毛茸茸的小爪子摸摸摸盒子里的东西 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004519646 | 0.9585 | 7 | 身处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0004519646 | 0.9539 | 8 | 深处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0004519646 | 0.9631 | 6 | 身处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004732647 | 0.6711 | 53 | He what did find out what made noises that people people were fraid of and the was nothing in the caves to tell lem | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0004732647 | 0.6510 | 59 | He what did find out what made noises that people people were a phrase of and the was nothing in the caves to tell him | wer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004732647 | 0.6510 | 59 | He what did find out what made noises that people people were a phrase our and the was nothing in the caves to tell him | wer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0004732647 | 0.6779 | 58 | He want to find out what made noises that people people were a phrase our and the was nothing in the caves to tell him | wer | 0.3333 |
| 160 | causal_full_asr | emilia_zh_0004732654 | 0.7759 | 31 | The quite suddenly the far ahead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 320 | causal_full_asr | emilia_zh_0004732654 | 0.7427 | 35 | But quite suddenly the fire had gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004732654 | 0.6888 | 40 | But quite suddenly the fire had gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 1280 | causal_full_asr | emilia_zh_0004732654 | 0.7054 | 42 | But quite suddenly the firehead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 160 | streaming_asr | emilia_zh_0004804709 | 0.7550 | 20 | Ran your unseen for the the of mature aside town | wer | 0.8000 |
| 320 | streaming_asr | emilia_zh_0004804709 | 0.7483 | 20 | Rend red unseen for the them of enter left side town | wer | 0.8000 |
| 640 | streaming_asr | emilia_zh_0004804709 | 0.7152 | 21 | Rend your un safeties for them to enter outside town | wer | 0.6000 |
| 1280 | streaming_asr | emilia_zh_0004804709 | 0.7152 | 22 | Rend drew unseen for them to mature outside town | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004927443 | 0.9864 | 3 | 但是美国人之所以懂悠闲还有一个更重要的原因 | cer | 0.0455 |
| 320 | streaming_asr | emilia_zh_0004927443 | 0.9729 | 4 | 但是美国人之所以懂悠闲还有一个更重要的原因 | cer | 0.0455 |
| 640 | streaming_asr | emilia_zh_0004927443 | 0.9683 | 6 | 但是美国人之所以懂悠闲还有一个更重要的原因 | cer | 0.0455 |
| 1280 | streaming_asr | emilia_zh_0004927443 | 0.9638 | 7 | 但是美国人人之所以懂悠闲还有一个更重要的原因 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0005094550 | 0.9779 | 9 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费要长的天赋就不能够希望解释清楚 | cer | 0.1220 |
| 320 | streaming_asr | emilia_zh_0005094550 | 0.9735 | 10 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0488 |
| 640 | streaming_asr | emilia_zh_0005094550 | 0.9735 | 11 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费要长的篇幅就不能够希望解释清楚 | cer | 0.0732 |
| 1280 | streaming_asr | emilia_zh_0005094550 | 0.9646 | 12 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费要长的天赋就不能够希望解释清楚 | cer | 0.1220 |
| 160 | causal_full_asr | emilia_zh_0005347375 | 0.9735 | 6 | 但我们却不知道他是什么只能确定他存在以灭小野 | cer | 0.2727 |
| 320 | causal_full_asr | emilia_zh_0005347375 | 0.9602 | 7 | 但我们却不知道他是什么只能确定他存在以立小业 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0005347375 | 0.9558 | 7 | 但我们却不知道他是什么只能确定他存在以立小业 | cer | 0.2727 |
| 1280 | causal_full_asr | emilia_zh_0005347375 | 0.9602 | 6 | 但我们却不知道他是什么只能确定他存在以立效应 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0005421693 | 0.9221 | 9 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005421693 | 0.9221 | 10 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005421693 | 0.9610 | 4 | 这五件东西是他昨晚制作交通工具 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005421693 | 0.9286 | 8 | 这五件东西是他昨晚制作交通工具 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005748506 | 0.9622 | 19 | 是是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这这个呃想法的时候你去做就好啊你先去去 | cer | 0.0980 |
| 320 | streaming_asr | emilia_zh_0005748506 | 0.9553 | 21 | 这受后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这这个呃想法的时候你去做就好的话你先去这样 | cer | 0.1569 |
| 640 | streaming_asr | emilia_zh_0005748506 | 0.9553 | 21 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这这个呃想法的时候你去做就好的话你先去去 | cer | 0.1176 |
| 1280 | streaming_asr | emilia_zh_0005748506 | 0.9570 | 21 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这这个呃想法的时候你去做就好的话你先去去 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0005928718 | 0.9709 | 6 | 团还是一个完整的嗯船长现在是重要为一个牌 | cer | 0.3636 |
| 320 | streaming_asr | emilia_zh_0005928718 | 0.9709 | 7 | 船行还是一直完整嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0005928718 | 0.9745 | 6 | 船行还是一只观着嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0005928718 | 0.9745 | 7 | 船行还是一只观着嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 160 | causal_full_asr | emilia_zh_0005960324 | 0.9793 | 7 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005960324 | 0.9664 | 9 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005960324 | 0.9638 | 10 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005960324 | 0.9612 | 11 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006174150 | 0.9737 | 5 | 就是因为我的一些当时不好的情绪其实会有传染呢 | cer | 0.0455 |
| 320 | streaming_asr | emilia_zh_0006174150 | 0.9699 | 6 | 就是因为我的一些当时不好的情绪其实会有传染呢 | cer | 0.0455 |
| 640 | streaming_asr | emilia_zh_0006174150 | 0.9737 | 5 | 就是因为我的一些当时不好的情绪其实会有传染呢 | cer | 0.0455 |
| 1280 | streaming_asr | emilia_zh_0006174150 | 0.9699 | 6 | 就是因为我的一些当时不好的情绪其实会传染呢 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0006350510 | 0.8094 | 30 | Just said they have live to very isolate life in India And he was terribly shy of women | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0006350510 | 0.7806 | 35 | Zhang said they have live to very isolate life in India And he was terribly shy of women | wer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0006350510 | 0.7626 | 37 | John said they have live to very isolated life in India And he was terribly shy of women | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0006350510 | 0.7590 | 39 | John said the have live to very isolated life in India And he was terribly shy of women | wer | 0.2222 |
| 160 | causal_full_asr | emilia_zh_0006405437 | 0.8565 | 31 | The two little boys kind of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.0500 |
| 320 | causal_full_asr | emilia_zh_0006405437 | 0.8292 | 37 | The two little boys kind of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.0500 |
| 640 | causal_full_asr | emilia_zh_0006405437 | 0.8246 | 34 | The two little boys kind of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.0500 |
| 1280 | causal_full_asr | emilia_zh_0006405437 | 0.8428 | 31 | The two little boys can have club tax rules as clear as mud tax rules as clear as mud | wer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0006435497 | 0.7912 | 83 | The middle the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in middle you will see and here in a appreciation of let and log again is AKA merch sort Today | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006435497 | 0.7597 | 90 | The middle the bottom three different algorithms on that top you see selection sort on the bottom you will see bubbles sort and in middle you will see and here in a appreciation of let in log again is AKA merch sort today | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0006435497 | 0.7711 | 93 | The middle the bottom three different algorithms on that top you see selection sort on the bottom you will see bubbles sort and in middle you will see and here and appreciation of let in log again is AKA merge sort today | wer | 0.1556 |
| 1280 | streaming_asr | emilia_zh_0006435497 | 0.7849 | 84 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in middle you will see and here and appreciation of let in log again is AKA merge sort today | wer | 0.0889 |
| 160 | streaming_asr | emilia_zh_0006598681 | 0.9400 | 11 | 就是在工作大学学习本年后第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.1515 |
| 320 | streaming_asr | emilia_zh_0006598681 | 0.9633 | 7 | 就是在公牛大学学习半年后第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0006598681 | 0.9667 | 8 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.0303 |
| 1280 | streaming_asr | emilia_zh_0006598681 | 0.9667 | 7 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.0303 |
| 160 | streaming_asr | emilia_zh_0006819602 | 0.9650 | 12 | 有半个时辰过去城墙的高度还圣像不到一张曹俊完全停下了天图的动作 | cer | 0.1935 |
| 320 | streaming_asr | emilia_zh_0006819602 | 0.9650 | 12 | 又半个时辰过去城墙的高度还剩下不到一张曹静完全停下了天图的动作 | cer | 0.1613 |
| 640 | streaming_asr | emilia_zh_0006819602 | 0.9623 | 13 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了天图的动作 | cer | 0.1290 |
| 1280 | streaming_asr | emilia_zh_0006819602 | 0.9569 | 13 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了天图的动作 | cer | 0.1290 |
| 160 | streaming_asr | emilia_zh_0007120510 | 0.9592 | 6 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0007120510 | 0.9541 | 7 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0007120510 | 0.9439 | 8 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0007120510 | 0.9490 | 7 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0007353483 | 0.9825 | 5 | 不同的选择会创造出不同的未来关键词虚拟利益 | cer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0007353483 | 0.9767 | 7 | 不同的选择会创造出不同的未来关键词二需体利益 | cer | 0.1364 |
| 640 | streaming_asr | emilia_zh_0007353483 | 0.9796 | 6 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0007353483 | 0.9796 | 5 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 160 | causal_full_asr | emilia_zh_0007353988 | 0.9705 | 12 | 两颗核弹都是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0851 |
| 320 | causal_full_asr | emilia_zh_0007353988 | 0.9760 | 10 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0638 |
| 640 | causal_full_asr | emilia_zh_0007353988 | 0.9760 | 11 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是一味的侦察车 | cer | 0.0638 |
| 1280 | causal_full_asr | emilia_zh_0007353988 | 0.9742 | 11 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能让防护区域偏转第二颗打中了一艘也许是一<\|glm_semantic_15842\|>二的侦察车 | cer | 0.5532 |
| 160 | streaming_asr | emilia_zh_0007555536 | 0.9585 | 10 | 其你将其的三个房间里书籍全部版过来我要细细的查阅一下 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0007555536 | 0.9619 | 8 | 请你将其余的三个房间里书籍全部把过来我要细细的查阅一下 | cer | 0.1071 |
| 640 | streaming_asr | emilia_zh_0007555536 | 0.9585 | 9 | 请你将其余的三个房间里书籍全部版过来我要细细的查阅一下 | cer | 0.1071 |
| 1280 | streaming_asr | emilia_zh_0007555536 | 0.9550 | 10 | 请你将其余的三个房间里书籍全部版过来我们要细细的查阅一下 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007788542 | 0.9965 | 1 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 320 | streaming_asr | emilia_zh_0007788542 | 0.9930 | 2 | 不少人开始接触期货觉得比较难以理解什么不是期货呢 | cer | 0.0833 |
| 640 | streaming_asr | emilia_zh_0007788542 | 0.9965 | 1 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 1280 | streaming_asr | emilia_zh_0007788542 | 1.0000 | 0 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 160 | streaming_asr | EN_B00043_S02661_W000018 | 0.7925 | 24 | Because opper frequently are medal dramatic not to say unrealistic | wer | 0.3333 |
| 320 | streaming_asr | EN_B00043_S02661_W000018 | 0.7849 | 25 | Because opper frequently are medal dramatic not to say unrealistic | wer | 0.3333 |
| 640 | streaming_asr | EN_B00043_S02661_W000018 | 0.7698 | 26 | Because opper frequently are medal dramatic not to say unrealistic | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00043_S02661_W000018 | 0.7774 | 27 | Because opper frequently are medal dramatic not to say unrealistic | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00043_S02699_W000000 | 0.8517 | 39 | That's the image That's go through the main door Turned F worked down to the end of the corridor and it's the last door on the right | wer | 0.2692 |
| 320 | causal_full_asr | EN_B00043_S02699_W000000 | 0.8432 | 43 | This is the miss Just go through the main door turn left work down to the end of the cutter and it's the last door on the right | wer | 0.1923 |
| 640 | causal_full_asr | EN_B00043_S02699_W000000 | 0.8242 | 48 | Set in the mid just go through the main door turn left walk down to the end of the car to the end of the car and it's the last door on the right | wer | 0.4231 |
| 1280 | causal_full_asr | EN_B00043_S02699_W000000 | 0.8051 | 54 | Suddenly met That's go through the main door Turned F worked down to the end of the corridor and it's the last door on the right | wer | 0.2308 |
| 160 | streaming_asr | EN_B00048_S01234_W000037 | 0.6848 | 29 | And in cases were not sure this animals going to be it dog | wer | 0.4375 |
| 320 | streaming_asr | EN_B00048_S01234_W000037 | 0.6303 | 32 | And in cases were not sure this animals going to be it dog | wer | 0.4375 |
| 640 | streaming_asr | EN_B00048_S01234_W000037 | 0.6121 | 35 | And in cases were not sure this animals going to be it dog | wer | 0.4375 |
| 1280 | streaming_asr | EN_B00048_S01234_W000037 | 0.6000 | 35 | And a cases were not sure of this Animals going to be it dog | wer | 0.5000 |
| 160 | streaming_asr | EN_B00048_S08821_W000040 | 0.7209 | 48 | And united is participating in a funnaries this weeken a little like bar the van if possible | wer | 0.4444 |
| 320 | streaming_asr | EN_B00048_S08821_W000040 | 0.7209 | 49 | Are you is participating in a funnaries this weeken a would like bar the van if possible | wer | 0.3889 |
| 640 | streaming_asr | EN_B00048_S08821_W000040 | 0.6811 | 50 | Are united is participating in a funnaries this weeken a would like bar the van if possible | wer | 0.3889 |
| 1280 | streaming_asr | EN_B00048_S08821_W000040 | 0.6811 | 47 | Are united is participating in a funnery this weeken a would like bar the van if possible | wer | 0.3889 |
| 160 | causal_full_asr | EN_B00048_S08821_W000047 | 0.8729 | 18 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S08821_W000047 | 0.8602 | 16 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S08821_W000047 | 0.8517 | 18 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S08821_W000047 | 0.8644 | 16 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S04431_W000023 | 0.8152 | 19 | Use blood had kept all parts of his body warmed | wer | 0.2000 |
| 320 | streaming_asr | EN_B00058_S04431_W000023 | 0.8315 | 18 | His blood had kept all parts of his body warmed | wer | 0.1000 |
| 640 | streaming_asr | EN_B00058_S04431_W000023 | 0.8043 | 21 | His blood had kept all parts of his body warmed | wer | 0.1000 |
| 1280 | streaming_asr | EN_B00058_S04431_W000023 | 0.8152 | 16 | His blood had kept all parts of his body warmed | wer | 0.1000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
