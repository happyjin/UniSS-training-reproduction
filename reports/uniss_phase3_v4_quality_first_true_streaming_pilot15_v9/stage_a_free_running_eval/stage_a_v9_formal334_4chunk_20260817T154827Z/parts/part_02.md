# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.8832**
- Weighted CTC blank ratio: **0.1780**
- Weighted streaming WER/CER: **0.4403**
- Weighted causal-full WER/CER: **0.3035**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042530 | 0.1232 | 21 | It International exchanged under under is Multiple functions building of caffeeteria combination and as for | wer | 0.8462 |
| 320 | streaming_asr | CommonVoice_EN_0000042530 | 0.1469 | 18 | He International exchanged under under in Multiple functions building of capitaria combination and as for | wer | 0.9231 |
| 640 | streaming_asr | CommonVoice_EN_0000042530 | 0.1801 | 24 | He International exchanged under it in Multiple functions building of caffeeteria accommodation and asks from | wer | 0.8462 |
| 1280 | streaming_asr | CommonVoice_EN_0000042530 | 0.2062 | 28 | He International exchanged and and in Multiple functions building of capitory accommodation and asks from | wer | 0.8462 |
| 160 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1015 | 14 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1015 | 13 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000116798 | 0.0985 | 15 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000116798 | 0.1108 | 14 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000238037 | 0.1071 | 10 | You one of fire the of and but but it it | wer | 0.8571 |
| 320 | streaming_asr | CommonVoice_EN_0000238037 | 0.1688 | 13 | You five of fire can all you of the it could the | wer | 0.7857 |
| 640 | streaming_asr | CommonVoice_EN_0000238037 | 0.1786 | 15 | You five of fire can all but you of the it could the | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000238037 | 0.2468 | 18 | You Find of fire can all the you for the the could the | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000381265 | 0.2105 | 13 | You vexisted int plane began | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000381265 | 0.3158 | 13 | You vexisted ins't plane began | wer | 0.8000 |
| 640 | streaming_asr | CommonVoice_EN_0000381265 | 0.3487 | 13 | You vexisted ins't plane megan | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000381265 | 0.1908 | 11 | You vexisted ins't plane megan | wer | 1.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1102 | 6 | Nothing personal in it | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1575 | 8 | Nothing personal on it | wer | 0.2500 |
| 640 | causal_full_asr | CommonVoice_EN_0000502585 | 0.2205 | 8 | Nothing personal in it | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000502585 | 0.1654 | 10 | Nothing personal on it | wer | 0.2500 |
| 160 | streaming_asr | CommonVoice_EN_0000520146 | 0.1793 | 12 | Many chross supporter I you spring expanse | wer | 0.7500 |
| 320 | streaming_asr | CommonVoice_EN_0000520146 | 0.2120 | 12 | Many chross supporter I leave spring suspension | wer | 0.7500 |
| 640 | streaming_asr | CommonVoice_EN_0000520146 | 0.1848 | 13 | Many rock's supporter I leave spere suspensions | wer | 0.7500 |
| 1280 | streaming_asr | CommonVoice_EN_0000520146 | 0.1739 | 13 | Many rocks supporter I leave spere suspensions | wer | 0.7500 |
| 160 | streaming_asr | LibriSpeech_0000033920 | 0.1601 | 41 | And and representative who of put for everything they grass of line and over one to panicked over myth a national pressed. the space sever turn over to the craft industry | wer | 0.5833 |
| 320 | streaming_asr | LibriSpeech_0000033920 | 0.1469 | 40 | And in the representative who of put for everything they grass of find and over one to panicked over myth a national pressed the space sever with turn over to the craft industry | wer | 0.5556 |
| 640 | streaming_asr | LibriSpeech_0000033920 | 0.1535 | 41 | And in the representative who a put for everything they grass of I and over one to panicked over myth a national pressed. the space sever with turn over to the craft industry | wer | 0.5556 |
| 1280 | streaming_asr | LibriSpeech_0000033920 | 0.1469 | 43 | And and representative who a put for everything they grass of I and government one to penic over myth a national pressed the spaced sever with turn over to the craft industry | wer | 0.5833 |
| 160 | streaming_asr | LibriSpeech_0000124435 | 0.1714 | 52 | So Now all the of the children saw upon the replace apples sauce and scotch and to me that and sweep Potato and sour Potato not when the and could it muffled because not one was safed the meat | wer | 0.5135 |
| 320 | streaming_asr | LibriSpeech_0000124435 | 0.1714 | 53 | So Now all the of the children so upon the played apples sauce and squash and to me. and sweep Potatoes and sour Potatoes not when the and could the muffled because not one was safed with meat | wer | 0.5405 |
| 640 | streaming_asr | LibriSpeech_0000124435 | 0.1565 | 55 | So Now all the of the children so upon the played apple sauce and scotch and to me to and sweep Potatoes and sour Potato not when the and to to muffled because not one was safed with meat | wer | 0.5676 |
| 1280 | streaming_asr | LibriSpeech_0000124435 | 0.1565 | 58 | So Now all the of the children so upon the replace apple saw and squash and to me to and swiss Potatoes and sour Potato not not the and to to muffled because not one was seventh. with meat | wer | 0.5676 |
| 160 | causal_full_asr | LibriSpeech_0000238204 | 0.1461 | 38 | What an entrepreneur those have a test why men who have a task who why it can only be speedy adventures a sort of person one reads of in books and it in needs a meal out | wer | 0.6875 |
| 320 | causal_full_asr | LibriSpeech_0000238204 | 0.1332 | 37 | Would answer one of those ever-taunts why men who ever-taunts who whyds can only be speedy adventurers, the sort person one reads of in books and knitting means a meal. | wer | 0.3750 |
| 640 | causal_full_asr | LibriSpeech_0000238204 | 0.1605 | 45 | What answer one of those evertized why men who evertized who was can only be speedy adventures, the sort person one reads of in books and it in needs and meal on | wer | 0.4375 |
| 1280 | causal_full_asr | LibriSpeech_0000238204 | 0.1676 | 43 | Would answer one of those I've a tussent why men who have a tussent who why it can only be speedy adventures the sort person one reads of in books and that it means a million | wer | 0.5312 |
| 160 | streaming_asr | LibriSpeech_0000271006 | 0.1502 | 31 | And a made in the temp to at on you for for could sure the that when never get more the in enough to can my travelling expense I thanked and for his survives to | wer | 0.6562 |
| 320 | streaming_asr | LibriSpeech_0000271006 | 0.1502 | 35 | And a make can tempt to it money for for could sure the that where no get more the in enough to can my travelling expense I thanked im for he survives to | wer | 0.5938 |
| 640 | streaming_asr | LibriSpeech_0000271006 | 0.1319 | 33 | And not make can tempt to it money for you could sure the like where no get more the in enough to can my travelling expense I thanked im for he survives to | wer | 0.5625 |
| 1280 | streaming_asr | LibriSpeech_0000271006 | 0.1099 | 26 | And not make can you tempt to at money for for could sure the that where no get more the in enough to can my travelling expense I thanked and for he advise to | wer | 0.5938 |
| 160 | streaming_asr | emilia_zh_0004036114 | 0.1505 | 12 | 没有上过大学家里情况可以说是一言难尽。 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0004036114 | 0.1720 | 11 | 没有上过大学家里情况一个说是一言难尽。 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004036114 | 0.1613 | 11 | 没有上过大学家里情况一个说是一言难尽。 | cer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004036114 | 0.1667 | 11 | 没有上国大学家里情况一个说是一言难尽。 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004176769 | 0.2881 | 19 | 最终我们和从尝尝尝起来找事情做 | cer | 0.3125 |
| 320 | streaming_asr | emilia_zh_0004176769 | 0.2373 | 19 | 最终我们和可能尝尝尝起来找事情做 | cer | 0.4375 |
| 640 | streaming_asr | emilia_zh_0004176769 | 0.2316 | 17 | 最终我们和能床上站起来找事情做 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004176769 | 0.2203 | 18 | 最终我们会能床上站起来找事情做 | cer | 0.1875 |
| 160 | causal_full_asr | emilia_zh_0004343392 | 0.1476 | 13 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 320 | causal_full_asr | emilia_zh_0004343392 | 0.1550 | 14 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 640 | causal_full_asr | emilia_zh_0004343392 | 0.1587 | 16 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 1280 | causal_full_asr | emilia_zh_0004343392 | 0.1882 | 17 | 他还不走天黑了怎么办呢？小红猴子说 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0004472880 | 0.3022 | 14 | 那可不见得就是为了这个说打山两 | cer | 0.4737 |
| 320 | streaming_asr | emilia_zh_0004472880 | 0.2198 | 13 | 那可不见得就是为了这个的时候打山两 | cer | 0.3684 |
| 640 | streaming_asr | emilia_zh_0004472880 | 0.1978 | 11 | 那可不见得就是为了这个的时候打山两 | cer | 0.3684 |
| 1280 | streaming_asr | emilia_zh_0004472880 | 0.1758 | 9 | 那和不见得就是为了这个的时候打。山两 | cer | 0.4211 |
| 160 | causal_full_asr | emilia_zh_0004692595 | 0.1528 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004692595 | 0.1661 | 19 | Different equation and then later the differential equation interpreted in black diagram terms. | wer | 0.0769 |
| 640 | causal_full_asr | emilia_zh_0004692595 | 0.1362 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004692595 | 0.1362 | 17 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004724705 | 0.2515 | 32 | And While had just ren't what who his like had member they were they cognitive but what see they pointing physical the first | wer | 0.7917 |
| 320 | streaming_asr | emilia_zh_0004724705 | 0.2602 | 34 | And Well had to ran there the we his heart like had with they were they cocking but what see they point physical the first | wer | 0.7500 |
| 640 | streaming_asr | emilia_zh_0004724705 | 0.2018 | 29 | And Well had who ren't the we his heart like had with that were they cockney but what see like point physical the for | wer | 0.7083 |
| 1280 | streaming_asr | emilia_zh_0004724705 | 0.2076 | 27 | And Well had who and the we his heart like had with that were did talking but what see like pointing physical the for | wer | 0.6667 |
| 160 | streaming_asr | emilia_zh_0004797649 | 0.2199 | 15 | You no told me has got would catch | wer | 0.6250 |
| 320 | streaming_asr | emilia_zh_0004797649 | 0.2147 | 15 | You no told me had got word catch | wer | 0.6250 |
| 640 | streaming_asr | emilia_zh_0004797649 | 0.1047 | 9 | You no told me has got word catch | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0004797649 | 0.1047 | 10 | You no told me has got word catch | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004879738 | 0.1635 | 8 | The Resignations furious Elizabeth and sunny | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004879738 | 0.1572 | 9 | The Resignations furious Elizabeth and sunny | wer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004879738 | 0.1635 | 8 | The resignations furious Elizabeth and sunny | wer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004879738 | 0.1447 | 8 | The recognitions inferiored Elizabeth and sunny | wer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0005059483 | 0.2941 | 25 | 故事的目的如果仅仅说了获取更大的他不是也就不成为不是在这对普通语言跟 | cer | 0.4571 |
| 320 | streaming_asr | emilia_zh_0005059483 | 0.2941 | 26 | 不是的目的如果仅仅说了获取更大的他不也就不成为不是在这对普通语言跟 | cer | 0.4571 |
| 640 | streaming_asr | emilia_zh_0005059483 | 0.2674 | 29 | 不是的目的如仅仅说了获取更大的他不也就不成为不是在这对普通语言和 | cer | 0.4857 |
| 1280 | streaming_asr | emilia_zh_0005059483 | 0.2513 | 28 | 不是的目的如仅仅说了获取更大看来不是也就不成为不是在这对普通语言和 | cer | 0.5143 |
| 160 | causal_full_asr | emilia_zh_0005245611 | 0.2576 | 20 | 他提出预祝的商品只有再次革新才能更好的满足欧洲人的需求 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005245611 | 0.2508 | 23 | 他提出预注的商品只有再次革新才能更好的满足住人的需求 | cer | 0.1111 |
| 640 | causal_full_asr | emilia_zh_0005245611 | 0.2644 | 24 | 他提出预注的商品只有再次革新才能更好的满足住人的需求 | cer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0005245611 | 0.2576 | 27 | 他提出预注的商品只有再次革新才能更好的满足欧洲人的需求 | cer | 0.0370 |
| 160 | streaming_asr | emilia_zh_0005370632 | 0.1574 | 11 | 你是看不到错你这形态展发展发展与 | cer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0005370632 | 0.1617 | 10 | 你是看不到做你这形态展发展发展与 | cer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0005370632 | 0.1660 | 12 | 你是看不到做你这形态展发展发展与 | cer | 0.3750 |
| 1280 | streaming_asr | emilia_zh_0005370632 | 0.1574 | 11 | 你是看不到说你这形态展发展发展与 | cer | 0.3125 |
| 160 | streaming_asr | emilia_zh_0005669352 | 0.1815 | 30 | 是听啥的意思两宝红朗读都抖动的假帮助着责任路了一地哥哥趴下的东遮的例子一但是骗货 | cer | 0.5366 |
| 320 | streaming_asr | emilia_zh_0005669352 | 0.1835 | 37 | 是听沙的意思两宝红朗斗都抖动的一下帮助责任路了一定哥哥爬的东哥例子一暗示骗货 | cer | 0.5122 |
| 640 | streaming_asr | emilia_zh_0005669352 | 0.1915 | 38 | 是听沙的意思两宝红朗斗都抖动的一家帮助词儿路了一定哥哥爬的东哥例子一暗示骗货 | cer | 0.5122 |
| 1280 | streaming_asr | emilia_zh_0005669352 | 0.2056 | 36 | 是听沙的意思两朵红朗斗都抖动的一家帮助词儿路了一定哥哥爬的东哥的利益一暗示骗货 | cer | 0.4878 |
| 160 | streaming_asr | emilia_zh_0005903796 | 0.2687 | 26 | 而且一个我咱们路节目钱为朋友给我信息说他中。打车然后打车司机大一点一个那个说 | cer | 0.5122 |
| 320 | streaming_asr | emilia_zh_0005903796 | 0.2844 | 25 | 而且刚刚我咱们路节目钱为朋友给我信息说他中打车然后卡车司机大一点一个能说 | cer | 0.4878 |
| 640 | streaming_asr | emilia_zh_0005903796 | 0.2625 | 27 | 而且刚刚我咱们路节目钱为朋友给我发微信说他中的打车然后卡车司机大一点一个那说 | cer | 0.4146 |
| 1280 | streaming_asr | emilia_zh_0005903796 | 0.2344 | 26 | 而且刚刚我咱们路节目钱为朋友给我发微信说他中的打车然后下车司机大一个那说 | cer | 0.3659 |
| 160 | causal_full_asr | emilia_zh_0005926417 | 0.1906 | 24 | 没有办法就我自己没有办法去得出这样的判断或者被问出来横着呢是参考一些呃 | cer | 0.2500 |
| 320 | causal_full_asr | emilia_zh_0005926417 | 0.2072 | 23 | 没有办法就我自己没有办法去得出这样的判断或者得问出来横着呢是参考一些呃 | cer | 0.2500 |
| 640 | causal_full_asr | emilia_zh_0005926417 | 0.1961 | 20 | 没有办法就我自己没有办法去得出这样的判断或者得问出来横着呢是参考一些呃 | cer | 0.2500 |
| 1280 | causal_full_asr | emilia_zh_0005926417 | 0.2127 | 21 | 没有办法看就我自己没有办法去得出这样的判断或者得问出来横着能是参考一些呃 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006119067 | 0.0861 | 7 | 而且这个还有一个很让人就是写我明白的呀这个是 | cer | 0.2174 |
| 320 | streaming_asr | emilia_zh_0006119067 | 0.1005 | 10 | 而且这还有一个很让人就是是我明白的点这个是 | cer | 0.1739 |
| 640 | streaming_asr | emilia_zh_0006119067 | 0.1148 | 13 | 而且这还有一个很让人就是是我明白的点这个是 | cer | 0.1739 |
| 1280 | streaming_asr | emilia_zh_0006119067 | 0.1148 | 11 | 而且这里还有一个很让人就是是我明白的呀这个是 | cer | 0.1739 |
| 160 | streaming_asr | emilia_zh_0006330534 | 0.1993 | 20 | And had no once please several days at last I ceive to showed not | wer | 0.5000 |
| 320 | streaming_asr | emilia_zh_0006330534 | 0.2509 | 20 | And had no once for several days at last I receives to showed not | wer | 0.4286 |
| 640 | streaming_asr | emilia_zh_0006330534 | 0.2612 | 20 | And had no once but several days at last I receives to show then | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0006330534 | 0.1615 | 24 | And had no once but Several days At last I receives to showed not | wer | 0.5000 |
| 160 | causal_full_asr | emilia_zh_0006330755 | 0.2212 | 16 | And now later they were standing in the graveyard of the old stone church. | wer | 0.3571 |
| 320 | causal_full_asr | emilia_zh_0006330755 | 0.1935 | 16 | An hour later they were standing in the graveyard of the old stone church. | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0006330755 | 0.1982 | 17 | An hour later they were standing in the graveyard of the old stone church. | wer | 0.2143 |
| 1280 | causal_full_asr | emilia_zh_0006330755 | 0.1797 | 16 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0006430973 | 0.2112 | 24 | The Just just was was something the Greg could with out if had to But there writing does had to state | wer | 0.5455 |
| 320 | streaming_asr | emilia_zh_0006430973 | 0.1957 | 20 | The chess the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 640 | streaming_asr | emilia_zh_0006430973 | 0.1553 | 26 | The Just the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 1280 | streaming_asr | emilia_zh_0006430973 | 0.1460 | 23 | The chess the was was something the Greg could without if had to But the writing does had to state | wer | 0.4545 |
| 160 | streaming_asr | emilia_zh_0006544664 | 0.1993 | 34 | 后院货运商贸走了钱这种小车在广州的门的的是还是四当时也这一夜。注意到车的颜色是一种会来了 | cer | 0.4681 |
| 320 | streaming_asr | emilia_zh_0006544664 | 0.1926 | 30 | 货运货运商贸走我钱这种小车在广州的门的的是而且四当时也这物业这注意到车的颜色是一种会而 | cer | 0.4043 |
| 640 | streaming_asr | emilia_zh_0006544664 | 0.1742 | 33 | 货运货运商贸走了钱这种小车在广州的们的的是还是四当时也这物业这注意到车的颜色深色一种会而 | cer | 0.4894 |
| 1280 | streaming_asr | emilia_zh_0006544664 | 0.1692 | 30 | 客运货运商贸走了钱这种小车在广州们的的是而四当时人也物业这注意到车的颜色是一种会而 | cer | 0.4255 |
| 160 | streaming_asr | emilia_zh_0006731464 | 0.2278 | 13 | 他选择这种生活方式一定他自己的道理 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0006731464 | 0.1833 | 11 | 他选择这种生活方式一定的自己的道理 | cer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0006731464 | 0.1722 | 12 | 他选择这种生活方式一定的自己的道理 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0006731464 | 0.1667 | 11 | 他选择这种生活方式一定的自己道理 | cer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0006992873 | 0.1272 | 16 | 绿色的城堡大曹人首先警觉起来 | cer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0006992873 | 0.1623 | 18 | 绿色的城堡道草人首先警觉起来 | cer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0006992873 | 0.2105 | 18 | The green castle was first alerted by Cao Ren. | cer | 2.7143 |
| 1280 | causal_full_asr | emilia_zh_0006992873 | 0.2018 | 18 | The green castle was first alerted by Cao Ren. | cer | 2.7143 |
| 160 | streaming_asr | emilia_zh_0007017174 | 0.2617 | 24 | 从某种程度上到让不是出特点演可能说法啊是所得税恐怕就是工薪税啊 | cer | 0.2424 |
| 320 | streaming_asr | emilia_zh_0007017174 | 0.2727 | 27 | 从某种程度上啊到人不是出特别眼睛可能说法啊说吧啊所得税恐怕就是工薪税啊 | cer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0007017174 | 0.2948 | 28 | 从某种程度上啊当人不是出特别盐可能说法啊说吧啊所得税恐怕就是工薪税啊 | cer | 0.2121 |
| 1280 | streaming_asr | emilia_zh_0007017174 | 0.2645 | 31 | 从某种程度上啊当让不是出策略盐可能说法啊说所得税恐怕就是工薪税啊 | cer | 0.2424 |
| 160 | streaming_asr | emilia_zh_0007312342 | 0.1181 | 22 | 开他人生中的第一个支配账户他叫他怎么写支票一天上班是丹尼斯提到 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0007312342 | 0.1181 | 21 | 开他人生中的第一个支配账户他教他怎么写支票一天上班是丹尼丝提到 | cer | 0.1250 |
| 640 | streaming_asr | emilia_zh_0007312342 | 0.1224 | 23 | 开他人生中的第一个支配账户他叫他怎么也支票一天上班时丹尼斯提到 | cer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0007312342 | 0.1055 | 21 | 开他人生中的第一个支配账户他叫他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 160 | streaming_asr | emilia_zh_0007526333 | 0.2165 | 28 | 到我自己的的这重点一个就是每年考试都会考的内容就是希腊戏剧的他作品比 | cer | 0.3611 |
| 320 | streaming_asr | emilia_zh_0007526333 | 0.2294 | 28 | 到了这个的的最重点一个就是每年考试都会考这个内容就是希腊戏剧的他作品比 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0007526333 | 0.2088 | 23 | 到了这个这个的最重点个就是每年考试都备考这个内容就是其实戏剧的他够比 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0007526333 | 0.2191 | 26 | 到了这这个的最重点个就是每年考试都高考这个内容就是其实戏剧的他过。比 | cer | 0.3611 |
| 160 | streaming_asr | emilia_zh_0007721307 | 0.2000 | 22 | 其实买的很多演讲你呢也我出现某些主题反复讲的现象马云说那重复是为了强调 | cer | 0.1389 |
| 320 | streaming_asr | emilia_zh_0007721307 | 0.2175 | 22 | 其实买的很多演讲你呢你和出现某些主题反复想的现象马云说呢重复是为了强调 | cer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0007721307 | 0.2225 | 22 | 其实买的很多演讲你呢你我出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.1389 |
| 1280 | streaming_asr | emilia_zh_0007721307 | 0.2000 | 23 | 其实买的很多演讲你那你我出现某些主题反复假的现象马云说呢重复是为了强调 | cer | 0.1944 |
| 160 | streaming_asr | EN_B00052_S08802_W000006 | 0.1356 | 11 | We are Beautiful Pretty漂亮 | wer | 1.0000 |
| 320 | streaming_asr | EN_B00052_S08802_W000006 | 0.1388 | 11 | Were Beautiful Pretty漂亮 | wer | 0.6667 |
| 640 | streaming_asr | EN_B00052_S08802_W000006 | 0.1293 | 10 | Were Beautiful Pretty漂亮 | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00052_S08802_W000006 | 0.1104 | 12 | were Beautiful Pretty漂亮 | wer | 0.6667 |
| 160 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1408 | 26 | I found all that hard to believe too and must have been terrible and there was nothing anyone could do about it. | wer | 0.0909 |
| 320 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1408 | 24 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it. | wer | 0.0455 |
| 640 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1529 | 28 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it. | wer | 0.0455 |
| 1280 | causal_full_asr | EN_B00043_S01623_W000011 | 0.1456 | 26 | I found all that hard to believe to It must have been terrible and there was nothing anyone could do about it. | wer | 0.0909 |
| 160 | streaming_asr | EN_B00089_S01559_W000004 | 0.2066 | 12 | He never <\|glm_semantic_10236\|>ishes soaks or a long journey or homework | wer | 0.7500 |
| 320 | streaming_asr | EN_B00089_S01559_W000004 | 0.1983 | 16 | And never understood Socks or Arranged or home market | wer | 0.5000 |
| 640 | streaming_asr | EN_B00089_S01559_W000004 | 0.1777 | 13 | And never understood socks or Arranged or homework | wer | 0.2500 |
| 1280 | streaming_asr | EN_B00089_S01559_W000004 | 0.1405 | 10 | And never understood Socks or Arranged or homework | wer | 0.2500 |
| 160 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1235 | 8 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1412 | 10 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07041_W000493 | 0.1059 | 9 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07041_W000493 | 0.0765 | 8 | 也许他们就是不喜欢我，所以他们不想和我讲话 | wer | 1.0000 |
| 160 | streaming_asr | EN_B00048_S07042_W000076 | 0.2515 | 13 | And no I in the have time but things way before before a here | wer | 0.6429 |
| 320 | streaming_asr | EN_B00048_S07042_W000076 | 0.2270 | 13 | And no I in the have time but things way before you a here | wer | 0.5714 |
| 640 | streaming_asr | EN_B00048_S07042_W000076 | 0.1534 | 10 | And no I at the time have turned but things way before you a here | wer | 0.6429 |
| 1280 | streaming_asr | EN_B00048_S07042_W000076 | 0.1534 | 8 | I no I in the of turn but things way before you a here | wer | 0.5714 |
| 160 | streaming_asr | EN_B00058_S03815_W000004 | 0.1818 | 24 | my child no how more matches strong in the the make one figure that post | wer | 0.6667 |
| 320 | streaming_asr | EN_B00058_S03815_W000004 | 0.1437 | 18 | My child no her but match it. strong in the the make one figure that post | wer | 0.6667 |
| 640 | streaming_asr | EN_B00058_S03815_W000004 | 0.1290 | 18 | My child no her ball manchic strong in the the make one figure that post | wer | 0.6667 |
| 1280 | streaming_asr | EN_B00058_S03815_W000004 | 0.1232 | 17 | My child no her both manchic strong in the of make one figure that post | wer | 0.6667 |
| 160 | streaming_asr | EN_B00036_S05316_W000048 | 0.1617 | 20 | This Sinth floor see star has three for why arm spend at taste first see action | wer | 0.7059 |
| 320 | streaming_asr | EN_B00036_S05316_W000048 | 0.1848 | 20 | This seven flowers see star has three for wide arm spend at taste for see fortune | wer | 0.5882 |
| 640 | streaming_asr | EN_B00036_S05316_W000048 | 0.1452 | 16 | This SMFLO see star has three for why arm spend at taste first see fortune | wer | 0.6471 |
| 1280 | streaming_asr | EN_B00036_S05316_W000048 | 0.1617 | 20 | This Sunflower see start has three for why arm spend at taste first see reaching | wer | 0.6471 |
| 160 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1275 | 10 | You're unmute to a gun at cheery. Sorry good morning everybody | wer | 0.6667 |
| 320 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1235 | 11 | You're unmute we're gonna hear you. Sorry good morning everybody | wer | 0.5833 |
| 640 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1155 | 10 | You're unmute we're gonna hear you. Sorry good morning everybody | wer | 0.5833 |
| 1280 | causal_full_asr | EN_B00083_S08530_W000016 | 0.1195 | 12 | You're unmute we're gonna carry you. Sorry good morning everybody. | wer | 0.7500 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
