# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8854**
- Weighted CTC blank ratio: **0.1849**
- Weighted streaming WER/CER: **0.3907**
- Weighted causal-full WER/CER: **0.1901**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000116742 | 0.1311 | 11 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000116742 | 0.1516 | 11 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000116742 | 0.1557 | 12 | You this one after better now mountains in s Sweden | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000116742 | 0.1230 | 11 | It This one after better now mountains in s Sweden | wer | 0.5556 |
| 160 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1561 | 12 | 强大的引擎或支援 | wer | 1.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000160093 | 0.2098 | 14 | A presence of my tangerine sir said Paul | wer | 0.7778 |
| 640 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1902 | 13 | A presence of my tangents are said Palt | wer | 0.8889 |
| 1280 | causal_full_asr | CommonVoice_EN_0000160093 | 0.1610 | 15 | A presence of my tangents are said part | wer | 0.8889 |
| 160 | streaming_asr | CommonVoice_EN_0000285524 | 0.1168 | 21 | The Townships pro proximity to the see favor moorie tried and fish f factories | wer | 0.5833 |
| 320 | streaming_asr | CommonVoice_EN_0000285524 | 0.1368 | 22 | The Townships pro proximity to of see favor moorie tried and fish factories | wer | 0.5833 |
| 640 | streaming_asr | CommonVoice_EN_0000285524 | 0.1396 | 20 | The Townships pro proximity to the see favor moorie tried and fish factories | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000285524 | 0.1567 | 20 | The Townships pro proximity to of see favor murray tried and fish Factories | wer | 0.5833 |
| 160 | streaming_asr | CommonVoice_EN_0000430128 | 0.1800 | 14 | Luo to can you test was capacity designed why rose bad | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000430128 | 0.2040 | 18 | Luo to can you taste was cap can see designed why rose bad | wer | 0.8571 |
| 640 | streaming_asr | CommonVoice_EN_0000430128 | 0.2280 | 18 | Luo who can you they was cap can see designed white rose bad | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000430128 | 0.2040 | 16 | Law who can you they was capancy designed white rose bad | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000555853 | 0.1148 | 12 | Sam some he said from the the a | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000555853 | 0.1721 | 16 | Shaving water Sam some he said from then the a | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000555853 | 0.1680 | 13 | Shaving water Sam some he said from the the a | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000555853 | 0.1762 | 15 | Shaving water Sam He he said from the the a | wer | 0.6250 |
| 160 | causal_full_asr | DailyTalk_0000009768 | 0.1667 | 11 | Bring us a bottle of Remi Mortini and red wine | wer | 0.2000 |
| 320 | causal_full_asr | DailyTalk_0000009768 | 0.1882 | 12 | Bring us a bottle of Remi Mortini and red wine | wer | 0.2000 |
| 640 | causal_full_asr | DailyTalk_0000009768 | 0.1828 | 13 | Bring us the bottle of Remy Martinez and Red wine | wer | 0.2000 |
| 1280 | causal_full_asr | DailyTalk_0000009768 | 0.1828 | 12 | Bring us to bottle of Remy Martinez and Red wine | wer | 0.2000 |
| 160 | streaming_asr | LibriSpeech_0000068252 | 0.1509 | 50 | The b Bureau of health has transformed the see have menela from me feet reinfested huck bed of contagious Zees to one of most helpful City some the love six thousand weber's have been collected | wer | 0.5000 |
| 320 | streaming_asr | LibriSpeech_0000068252 | 0.1550 | 44 | The beau of health has transformed the see have menela from me feet infested huck bed of contagious Zeses to one of most helpful City some the love six That was weber's have been collected | wer | 0.5294 |
| 640 | streaming_asr | LibriSpeech_0000068252 | 0.1469 | 45 | The Biro of health has transformed the see have menela for a feet infested huck bed of contagious Zesas to one of most helpful City some the love six That was weber's have been collected | wer | 0.5294 |
| 1280 | streaming_asr | LibriSpeech_0000068252 | 0.1442 | 44 | The Biro of health has transformed the see have menela from a feet reinfested huck bed of contagious Zeses to one of the most helpful City that the love six That was weber's have been collected | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000158589 | 0.1288 | 38 | I I did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the please. boys of this is blap it for after receiving the affectionate grating of near the hold company | wer | 0.5556 |
| 320 | streaming_asr | LibriSpeech_0000158589 | 0.1493 | 46 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the place boys of this is blab it for after receiving the affectionate grating of near the hold company | wer | 0.5278 |
| 640 | streaming_asr | LibriSpeech_0000158589 | 0.1726 | 46 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the place voice of this is blaffe for after receiving the affectionate grating of near the hold company | wer | 0.4722 |
| 1280 | streaming_asr | LibriSpeech_0000158589 | 0.1479 | 44 | I always did think Mary harris resemble the Chinese Mary harris was pretty is child a child remember some the please. voice of this is blaffe for after receiving the affectionate grating of near the hold company | wer | 0.4722 |
| 160 | causal_full_asr | LibriSpeech_0000271146 | 0.2021 | 36 | But it's full of ant living mounds and enormous springs with the Indian such a beet some gigantic race for chilft in a past age | wer | 0.6522 |
| 320 | causal_full_asr | LibriSpeech_0000271146 | 0.2280 | 38 | But it's full of antiquity in the mounds and enormous bones with the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.6087 |
| 640 | causal_full_asr | LibriSpeech_0000271146 | 0.2228 | 39 | But it's full of untilving the mounds in enormous bones with the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.5652 |
| 1280 | causal_full_asr | LibriSpeech_0000271146 | 0.2202 | 35 | For it is full of until leaving the mounds enormous bones for the Indian sacrifice some gigantic race which lived in a past age. | wer | 0.4348 |
| 160 | streaming_asr | VCTK_0000029186 | 0.0861 | 6 | It so and being new challenged | wer | 0.8571 |
| 320 | streaming_asr | VCTK_0000029186 | 0.1126 | 10 | It so to being new challenged | wer | 0.7143 |
| 640 | streaming_asr | VCTK_0000029186 | 0.1126 | 11 | It could to being new challenged | wer | 0.7143 |
| 1280 | streaming_asr | VCTK_0000029186 | 0.1126 | 10 | It could to be new challenged | wer | 0.5714 |
| 160 | streaming_asr | emilia_zh_0004064952 | 0.1744 | 10 | 也在国了下因为在我的脑子里善材料的光 | cer | 0.4737 |
| 320 | streaming_asr | emilia_zh_0004064952 | 0.1590 | 9 | 也在国了想依然在我的脑子里善贼亮的光 | cer | 0.3158 |
| 640 | streaming_asr | emilia_zh_0004064952 | 0.1795 | 10 | 也国乐想因为在我的脑子里善贼亮的光 | cer | 0.4737 |
| 1280 | streaming_asr | emilia_zh_0004064952 | 0.1692 | 10 | 也国了想因为在我的脑子里善贼亮的光 | cer | 0.4737 |
| 160 | streaming_asr | emilia_zh_0004270141 | 0.1509 | 16 | 他的小山上已经中了二十天了欲知再见注视着海面等他回来 | cer | 0.2308 |
| 320 | streaming_asr | emilia_zh_0004270141 | 0.1509 | 15 | 他的小山上已经中了二十天了欲知再见注视海面等他回了 | cer | 0.3077 |
| 640 | streaming_asr | emilia_zh_0004270141 | 0.1572 | 17 | 他的小山上已经中了二十天了欲指再见注视海面等他毁了 | cer | 0.3462 |
| 1280 | streaming_asr | emilia_zh_0004270141 | 0.1478 | 15 | 他的小山上已经。了二十天了欲知再见注视海面等他回了 | cer | 0.3077 |
| 160 | causal_full_asr | emilia_zh_0004472296 | 0.1646 | 15 | 里面的不对啊这是造白书带有给你们念上念 | cer | 0.2632 |
| 320 | causal_full_asr | emilia_zh_0004472296 | 0.1677 | 15 | 明年的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 640 | causal_full_asr | emilia_zh_0004472296 | 0.1801 | 17 | 宁愿的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 1280 | causal_full_asr | emilia_zh_0004472296 | 0.1770 | 14 | 宁愿的不对啊这是照白书带我给你们念上念 | cer | 0.2105 |
| 160 | streaming_asr | emilia_zh_0004519646 | 0.2212 | 14 | 深处毛茸茸的小转子摸了摸盒子里的东西 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004519646 | 0.2442 | 15 | 深处毛茸茸的小转折摸了摸盒子里的东西 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004519646 | 0.2535 | 16 | 深处毛茸茸的小转折摸了摸盒子里的东西 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004519646 | 0.2535 | 15 | 深处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004732647 | 0.1644 | 19 | You what did find to what made noises people people were for a And the was nothing the caves to out a | wer | 0.5417 |
| 320 | streaming_asr | emilia_zh_0004732647 | 0.1544 | 19 | You what did find to what made noises people people were fray a And the was nothing the caves to out a my | wer | 0.5833 |
| 640 | streaming_asr | emilia_zh_0004732647 | 0.1544 | 20 | You what did find to what made noises people people were for a And the was nothing the caves to had a my | wer | 0.5833 |
| 1280 | streaming_asr | emilia_zh_0004732647 | 0.1409 | 22 | He want to find to what made noises people people were for a And the was nothing the caves to to a and | wer | 0.5000 |
| 160 | causal_full_asr | emilia_zh_0004732654 | 0.2531 | 19 | The quiet suddenly the fire had given a pale flicker and went down and the clocking ceased | wer | 0.2941 |
| 320 | causal_full_asr | emilia_zh_0004732654 | 0.2448 | 19 | But quite suddenly the fire hit gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004732654 | 0.2199 | 20 | But quite suddenly the fire had given a pale flicker and went down and the clocking ceased. | wer | 0.2353 |
| 1280 | causal_full_asr | emilia_zh_0004732654 | 0.1618 | 18 | But quite suddenly the fire ahead gave a pale flicker and went down and the clocking ceased. | wer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0004804709 | 0.1656 | 9 | And your unsafe for the the enter side town | wer | 0.7000 |
| 320 | streaming_asr | emilia_zh_0004804709 | 0.1722 | 7 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 640 | streaming_asr | emilia_zh_0004804709 | 0.1656 | 7 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 1280 | streaming_asr | emilia_zh_0004804709 | 0.1589 | 8 | Ran your unsafe for the the enter side town | wer | 0.7000 |
| 160 | streaming_asr | emilia_zh_0004927443 | 0.1538 | 12 | 但是美国<\|write_generate\|><\|cmn\|><\|start_content\|>之所以懂悠闲还有一个更用的原因 | cer | 2.0455 |
| 320 | streaming_asr | emilia_zh_0004927443 | 0.1719 | 12 | 但是美国和之所以懂悠闲还有一个更用的原因 | cer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0004927443 | 0.1810 | 14 | 但是美国嗯之所以都悠闲还有一个嗯用的原因 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0004927443 | 0.1719 | 14 | 但是美国嗯之所以都悠闲还有一个嗯用的原因 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0005094550 | 0.2611 | 36 | 他需要运用充满矛盾含糊不清概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1707 |
| 320 | streaming_asr | emilia_zh_0005094550 | 0.2611 | 37 | 他需要运用充满矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1463 |
| 640 | streaming_asr | emilia_zh_0005094550 | 0.2522 | 35 | 他需要应用购买矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.2195 |
| 1280 | streaming_asr | emilia_zh_0005094550 | 0.2456 | 36 | 他需要运用购买矛盾含糊不清的概念因此这个问题不花费报酬的天赋就不能够希望解释清楚 | cer | 0.1951 |
| 160 | causal_full_asr | emilia_zh_0005347375 | 0.2080 | 17 | 但我们却不知道他是什么只能确定他存在一定小野 | cer | 0.2727 |
| 320 | causal_full_asr | emilia_zh_0005347375 | 0.1726 | 16 | 但我们却不知道他是什么只能确定他存在一定小野 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0005347375 | 0.1858 | 17 | 但我们却不知道他是什么只能确定他存在以立小业 | cer | 0.2727 |
| 1280 | causal_full_asr | emilia_zh_0005347375 | 0.1814 | 15 | 但我们却不知道他是什么只能确定他存在以立小意 | cer | 0.2727 |
| 160 | streaming_asr | emilia_zh_0005421693 | 0.4026 | 16 | 这不见东西是他昨晚制作的交通工具 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0005421693 | 0.3701 | 17 | 这不见东西是他昨晚制作交通工具 | cer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0005421693 | 0.3506 | 19 | 这不见东西是他昨晚制作交通工具 | cer | 0.1875 |
| 1280 | streaming_asr | emilia_zh_0005421693 | 0.3571 | 20 | 这不见东西是他我我制作交通工具 | cer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0005748506 | 0.2302 | 40 | 是说后来我觉得我理解这一个他其实告诉你你要想那么多就是你我一个呃想法的时候你去做就好我你先去眼睛 | cer | 0.2157 |
| 320 | streaming_asr | emilia_zh_0005748506 | 0.2199 | 43 | 这受后来我觉得我理解谁一个他其实告诉你你要想那么多就是你我一个呃想法的时候去做就好我你先去这样 | cer | 0.2745 |
| 640 | streaming_asr | emilia_zh_0005748506 | 0.2062 | 42 | 这说后来我觉得我理解这一个他其实告诉你你要想那么多就是你我一个呃想法的时候去做就好我你先去这样 | cer | 0.2549 |
| 1280 | streaming_asr | emilia_zh_0005748506 | 0.2045 | 39 | 这说后来我觉得我理解这一个他其实告诉你你要想什么多就是你就一个呃想法的时候去做就好我你先去这样 | cer | 0.2745 |
| 160 | streaming_asr | emilia_zh_0005928718 | 0.2727 | 22 | 很好还是一个望着嗯船长现在是重要为一个牌 | cer | 0.3182 |
| 320 | streaming_asr | emilia_zh_0005928718 | 0.2400 | 22 | 传统还是一个人完整的嗯船长现在是重要为一个牌 | cer | 0.4091 |
| 640 | streaming_asr | emilia_zh_0005928718 | 0.2145 | 18 | 传统还是一直惯着的嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0005928718 | 0.2036 | 16 | 传统还是一直惯着的嗯船长现在是重要为一个牌 | cer | 0.2727 |
| 160 | causal_full_asr | emilia_zh_0005960324 | 0.1576 | 24 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005960324 | 0.1654 | 25 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005960324 | 0.1550 | 26 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005960324 | 0.1499 | 25 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006174150 | 0.2105 | 19 | 就是因为我的一些当时不好的情绪其实有传染呢 | cer | 0.0909 |
| 320 | streaming_asr | emilia_zh_0006174150 | 0.2068 | 18 | 就是因为我的一些当时不好的情绪其实有传染了 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0006174150 | 0.1880 | 18 | 就是因为我的一些当时不好了情绪其实有传染了 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0006174150 | 0.1955 | 20 | 就是因为我的一些当时不好了情绪其实对传染了 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006350510 | 0.1439 | 19 | Just she they had live very is related i've india And you terribly shy of men | wer | 0.6111 |
| 320 | streaming_asr | emilia_zh_0006350510 | 0.1547 | 20 | Just shit they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 640 | streaming_asr | emilia_zh_0006350510 | 0.1619 | 18 | Just shed they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 1280 | streaming_asr | emilia_zh_0006350510 | 0.1331 | 15 | Just set they had live very is related i've india And he terribly shy of men | wer | 0.5556 |
| 160 | causal_full_asr | emilia_zh_0006405437 | 0.1503 | 21 | The two little boys count of club tax rules are as clear as mud tax rules ours clear as mud | wer | 0.1000 |
| 320 | causal_full_asr | emilia_zh_0006405437 | 0.1640 | 25 | The two little boys kind of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.0500 |
| 640 | causal_full_asr | emilia_zh_0006405437 | 0.1435 | 25 | The two little boys can of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0006405437 | 0.1253 | 24 | The two little boys can of club tax rules are as clear as mud tax rules are as clear as mud | wer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0006435497 | 0.1925 | 55 | But millo and balm three Different algorithms on the top you see selection sort on the bod my will see bubble sword and in Middle you all see and here in protection of but in law again is AKA merder sor Today | wer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0006435497 | 0.1761 | 55 | The Middle and barram three Different algorithms on the top you see selection sort on the bod my will s bubble sword and in Middle you all see and here and perception of let in law again is AKA merder sor Today | wer | 0.3556 |
| 640 | streaming_asr | emilia_zh_0006435497 | 0.1384 | 46 | The Middle and bodam three Different algorithms on the hop you see selection sort on the bod my will see bubble sword and in Middle you all see and here and protection of let in law again is AK merder sor Today | wer | 0.3778 |
| 1280 | streaming_asr | emilia_zh_0006435497 | 0.1321 | 48 | The Middle in the bodam three different algorithms on the top you see select sort on the bod my will see bubble sword and in Middle you all see and here and perasure of let in law again is AKA merder sor Today | wer | 0.3111 |
| 160 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 25 | 就是在共有大学学习本年后第一份给你嗯的工作直接就是到博物馆冷猫 | cer | 0.2121 |
| 320 | streaming_asr | emilia_zh_0006598681 | 0.2500 | 28 | 就在共有大学学习半年多年第一份给你嗯的工作直接就是到博物馆冷猫 | cer | 0.2424 |
| 640 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 31 | 就是在共有学习学习半点和第一份给你安排的工作直接就是到博物馆冷猫 | cer | 0.2424 |
| 1280 | streaming_asr | emilia_zh_0006598681 | 0.2567 | 31 | 就是在共有学习学习半年多年第一份给你安排的工作直接就是到博物馆了猫 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006819602 | 0.2480 | 30 | 有半时辰过去城墙的高度还声响不到一张朝军完全停下了前途动作 | cer | 0.2581 |
| 320 | streaming_asr | emilia_zh_0006819602 | 0.2453 | 32 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 640 | streaming_asr | emilia_zh_0006819602 | 0.2183 | 26 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0006819602 | 0.2210 | 31 | 又半时辰过去城墙的高度还剩下不到一张超级完全停下了前途动作 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0007120510 | 0.1531 | 10 | 他将极大的的激励整个团队伙伴的习性 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0007120510 | 0.1429 | 10 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0007120510 | 0.1378 | 10 | 他将极大的激励整个团队伙伴的细心 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0007120510 | 0.1378 | 11 | 他家极大的激励整个团队伙伴的细心 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0007353483 | 0.2070 | 16 | 不同的选择会创造出不同的未来关键而虚利益 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007353483 | 0.2507 | 16 | 不同的选择会创造出不同的未来关键而虚利益 | cer | 0.2273 |
| 640 | streaming_asr | emilia_zh_0007353483 | 0.2362 | 18 | 不同的选择会创造处不同未来关键而虚利益 | cer | 0.3182 |
| 1280 | streaming_asr | emilia_zh_0007353483 | 0.2391 | 21 | 不同选择会创造处不同未来观点而虚利益 | cer | 0.4545 |
| 160 | causal_full_asr | emilia_zh_0007353988 | 0.2085 | 36 | 两颗核弹都是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0426 |
| 320 | causal_full_asr | emilia_zh_0007353988 | 0.1956 | 36 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 640 | causal_full_asr | emilia_zh_0007353988 | 0.1900 | 36 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 1280 | causal_full_asr | emilia_zh_0007353988 | 0.1993 | 38 | 两颗核弹倒是攻击了驱逐者的活动区域，但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0213 |
| 160 | streaming_asr | emilia_zh_0007555536 | 0.2076 | 21 | 就你将起的三个房间那里书籍全部把过来我要细细的查阅想 | cer | 0.3214 |
| 320 | streaming_asr | emilia_zh_0007555536 | 0.1903 | 23 | 其实你家起的三个房间那里书籍全部把过来我也细细的查阅想 | cer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0007555536 | 0.2007 | 22 | 且你家其的三个房间那里收集全部把过来我要细细查阅想 | cer | 0.4286 |
| 1280 | streaming_asr | emilia_zh_0007555536 | 0.2111 | 21 | 且你家其的三个房间那里书籍全部把过来我要细细查阅想 | cer | 0.3571 |
| 160 | streaming_asr | emilia_zh_0007788542 | 0.1923 | 17 | 不说人开始接触期货觉得比较难理解什么什么期货呢 | cer | 0.2083 |
| 320 | streaming_asr | emilia_zh_0007788542 | 0.1958 | 17 | 不说人一开始接触期货觉得比难理解什么什么期货的 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0007788542 | 0.1923 | 19 | 不是人开始接触期货觉得比较难理解省什么期货的 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0007788542 | 0.1888 | 17 | 不是人开始接触期货觉得比难理解省省期货的 | cer | 0.3333 |
| 160 | streaming_asr | EN_B00043_S02661_W000018 | 0.1623 | 18 | Because opper frequent your mellowed dramatic not to say unrealistic | wer | 0.5556 |
| 320 | streaming_asr | EN_B00043_S02661_W000018 | 0.1887 | 19 | Because opper frequent your mellowed and not to say unrealistic | wer | 0.5556 |
| 640 | streaming_asr | EN_B00043_S02661_W000018 | 0.1472 | 16 | Big opper -frequency your mellowed and not to say unrealistic | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00043_S02661_W000018 | 0.1396 | 16 | Big opper -frequency are mellowed and not to say unrealistic | wer | 0.5556 |
| 160 | causal_full_asr | EN_B00043_S02699_W000000 | 0.1589 | 30 | Set the mesh. Just goes through the main door turn left walked down to the end of the corridor and it's the last door on the right | wer | 0.1923 |
| 320 | causal_full_asr | EN_B00043_S02699_W000000 | 0.1716 | 27 | This is a mesh just goes through the main door turn F to walk down to the end of the corridor and it's the last door on the right | wer | 0.2692 |
| 640 | causal_full_asr | EN_B00043_S02699_W000000 | 0.2309 | 26 | That may show just goes through the main door turn F to walk down to the end of the corridor and it's the last door on the right | wer | 0.2308 |
| 1280 | causal_full_asr | EN_B00043_S02699_W000000 | 0.2097 | 26 | Suddenly missed. That's good through the main door turned F to walk down to the end of the corridor and it's the last door on the right. | wer | 0.3077 |
| 160 | streaming_asr | EN_B00048_S01234_W000037 | 0.2485 | 14 | And no cases when that sure the and more going be it dog | wer | 0.6875 |
| 320 | streaming_asr | EN_B00048_S01234_W000037 | 0.2000 | 10 | And no cases were that sure the and more going to be it dog | wer | 0.6250 |
| 640 | streaming_asr | EN_B00048_S01234_W000037 | 0.1818 | 11 | And the cases run that sure the and going to be a dog | wer | 0.5625 |
| 1280 | streaming_asr | EN_B00048_S01234_W000037 | 0.1758 | 10 | And the cases run that sure the and going to be a dog | wer | 0.5625 |
| 160 | streaming_asr | EN_B00048_S08821_W000040 | 0.2193 | 24 | Or you're the participating going to funnaries the sweetened a of like bore the van if Possible | wer | 0.6667 |
| 320 | streaming_asr | EN_B00048_S08821_W000040 | 0.2259 | 27 | I are you they participating you funnaries the sweeping a with like borrow the van if Possible | wer | 0.6667 |
| 640 | streaming_asr | EN_B00048_S08821_W000040 | 0.1894 | 23 | I are you the participates you funnaries the sweeping a with like brought the then if Possible | wer | 0.7778 |
| 1280 | streaming_asr | EN_B00048_S08821_W000040 | 0.1761 | 23 | I are you the participates in funnaries the screeked a with like brought the then if Possible | wer | 0.7778 |
| 160 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1441 | 13 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1525 | 14 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1525 | 13 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S08821_W000047 | 0.1144 | 10 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S04431_W000023 | 0.2446 | 12 | It's blood head cap all part of his body war | wer | 0.5000 |
| 320 | streaming_asr | EN_B00058_S04431_W000023 | 0.1793 | 12 | He's love head keep all part of his body warren | wer | 0.6000 |
| 640 | streaming_asr | EN_B00058_S04431_W000023 | 0.1685 | 13 | Here blood head keep all heart of his body warmer | wer | 0.5000 |
| 1280 | streaming_asr | EN_B00058_S04431_W000023 | 0.1685 | 13 | Here blood head keep all part of his body warmer | wer | 0.5000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
