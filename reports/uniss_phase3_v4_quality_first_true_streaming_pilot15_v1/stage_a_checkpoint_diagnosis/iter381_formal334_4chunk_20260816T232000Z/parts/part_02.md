# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9176**
- Weighted CTC blank ratio: **0.8872**
- Weighted streaming WER/CER: **0.2929**
- Weighted causal-full WER/CER: **0.1419**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042530 | 0.8294 | 36 | He International exchanged owner is in multifunctional building with caffeeteria accommodation and classrooms | wer | 0.3846 |
| 320 | streaming_asr | CommonVoice_EN_0000042530 | 0.7891 | 44 | He International exchanged owner is in multinational building with caffeeteria accommodation and classrooms | wer | 0.4615 |
| 640 | streaming_asr | CommonVoice_EN_0000042530 | 0.7915 | 46 | The International exchanged center is in multifunctional building with caffeeteria Accommodation and classrooms | wer | 0.2308 |
| 1280 | streaming_asr | CommonVoice_EN_0000042530 | 0.7796 | 47 | The International exchanged center is in multifunctional building with caffeeteria Accommodation and classrooms | wer | 0.2308 |
| 160 | causal_full_asr | CommonVoice_EN_0000116798 | 0.8862 | 19 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000116798 | 0.8677 | 25 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000116798 | 0.8708 | 23 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000116798 | 0.8585 | 26 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000238037 | 0.9805 | 3 | We find a fiery the out the you for I it it | wer | 1.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000238037 | 0.9675 | 5 | New find of fiery can out the mim for the the could him | wer | 0.7857 |
| 640 | streaming_asr | CommonVoice_EN_0000238037 | 0.9481 | 6 | New find of prairie can out of you for the the could him | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000238037 | 0.9643 | 4 | New find of fire can all of name for the the pick him | wer | 0.7857 |
| 160 | streaming_asr | CommonVoice_EN_0000381265 | 0.9474 | 6 | You existed didn't playing began | wer | 0.6000 |
| 320 | streaming_asr | CommonVoice_EN_0000381265 | 0.9079 | 9 | You vaccinated since plane began | wer | 0.6000 |
| 640 | streaming_asr | CommonVoice_EN_0000381265 | 0.9079 | 9 | You vexisted int plane megan | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000381265 | 0.9145 | 9 | You vexisted int plane began | wer | 0.8000 |
| 160 | causal_full_asr | CommonVoice_EN_0000502585 | 0.9055 | 5 | Nothing personal in it | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8976 | 8 | Nothing personal in it | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8740 | 9 | Nothing personal in it | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8740 | 9 | Nothing personal in it | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000520146 | 0.8424 | 19 | Many trucks are supposed by leave spring Suspensions | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000520146 | 0.8043 | 24 | Many trucks are sup supporter by leave spring Suspensions | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000520146 | 0.8261 | 21 | Manage trucks are supported by by leave spring Suspensions | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000520146 | 0.7717 | 26 | Manage trucks are supported by by leave spring suspensions | wer | 0.3750 |
| 160 | streaming_asr | LibriSpeech_0000033920 | 0.7129 | 97 | And the representatives who been picked for everything but their grass of sign and government one in to penic over <\|glm_semantic_10416\|>ief of national pro prestige the space ever with turn over to the craft industry | wer | 0.4167 |
| 320 | streaming_asr | LibriSpeech_0000033920 | 0.6881 | 104 | And in the representatives who been pitt for everything their grasp of sign and government one and to penic over <\|glm_semantic_10416\|>ief of national pro prestige the space ever would turn over to the craft industry | wer | 0.4444 |
| 640 | streaming_asr | LibriSpeech_0000033920 | 0.6733 | 107 | And in the representatives who been picked for everything their grass of sign and government one and to penic over <\|glm_semantic_3636\|>ief of national pro prestige the space ever would turn over to the craft industry | wer | 0.4444 |
| 1280 | streaming_asr | LibriSpeech_0000033920 | 0.6799 | 107 | And in the representatives who been pitt for everything their grass of sign and government one in to penic over <\|glm_semantic_10416\|>ief of national pro prestige the space ever would turn over to the craft industry | wer | 0.4722 |
| 160 | streaming_asr | LibriSpeech_0000124435 | 0.8232 | 73 | So now all the the children saw up on the plates apples sauce and squash and to meditate and sweep Potato and sour potato not when the him could they muffled because not one was sif'd with meat | wer | 0.4595 |
| 320 | streaming_asr | LibriSpeech_0000124435 | 0.8313 | 72 | So now all the the children saw upon the plates apples sauce and squash and to meditate and sweet Potato and sour Potato Not when the him could they muffled because not one was safed by with meat | wer | 0.4054 |
| 640 | streaming_asr | LibriSpeech_0000124435 | 0.8097 | 80 | So now all the the children saw up on the plates apples sauce and scotch and tomatoes and sweet potatoes and salar Potato Not when the him could they muffled because not one was safed with meat | wer | 0.4865 |
| 1280 | streaming_asr | LibriSpeech_0000124435 | 0.8232 | 75 | So now all the the children saw up on the plates apples saw and squash and to meditate and sweet potato and sour potato not when the him could they muffled because not one was safed with meat | wer | 0.4595 |
| 160 | causal_full_asr | LibriSpeech_0000238204 | 0.8897 | 43 | What answer one of those avatars Why men who have a tars who wives can only be speedy adventurers The sort person one reads of in books and it in needs a meal lot | wer | 0.4375 |
| 320 | causal_full_asr | LibriSpeech_0000238204 | 0.8811 | 47 | Would answer one of those advertisements Why men who advertise who wives can only be speedy adventurers The sort person one reads of in books and it means a meal lot | wer | 0.2500 |
| 640 | causal_full_asr | LibriSpeech_0000238204 | 0.8782 | 50 | Would answer one of those advertisements Why men who advertise who wives can only be speedy adventurers The sort person one reads of in books and netizens may laugh | wer | 0.2500 |
| 1280 | causal_full_asr | LibriSpeech_0000238204 | 0.8854 | 45 | Would answer one of those advertisements Why men who advertise who wives can only be speedy adventurers The sort person one reads of in books and knitting makes a meal like | wer | 0.2500 |
| 160 | streaming_asr | LibriSpeech_0000271006 | 0.8168 | 50 | And not make any attempt to that money for for quite sure the I when never forget more that in enough to k my travelling expenses I thank you him for his advise | wer | 0.3750 |
| 320 | streaming_asr | LibriSpeech_0000271006 | 0.7875 | 56 | And not make any attempt to at money for were quite sure the I were never get more that in enough to k my travelling expenses I thank you him for his advise | wer | 0.3438 |
| 640 | streaming_asr | LibriSpeech_0000271006 | 0.7949 | 56 | And not make any attempt to at money for your quite sure the I were never get more that in enough to k my travelling expenses I thank you him for his advise | wer | 0.3438 |
| 1280 | streaming_asr | LibriSpeech_0000271006 | 0.7875 | 58 | And not make any attempt to at money for he were quite sure that I were never get more that in enough to pay my travelling expenses I thank you him for his advise to | wer | 0.2812 |
| 160 | streaming_asr | emilia_zh_0004036114 | 0.9946 | 1 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004036114 | 1.0000 | 0 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004036114 | 1.0000 | 0 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004036114 | 0.9892 | 2 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004176769 | 0.9944 | 1 | 最终我们和从床上爬起来找点事情做 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0004176769 | 1.0000 | 0 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004176769 | 1.0000 | 0 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004176769 | 1.0000 | 0 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004343392 | 0.9852 | 4 | 涛海波走天黑了怎么办呢小红猴子说 | cer | 0.1875 |
| 320 | causal_full_asr | emilia_zh_0004343392 | 0.9926 | 2 | 涛海波走天黑了怎么办呢小红猴子说 | cer | 0.1875 |
| 640 | causal_full_asr | emilia_zh_0004343392 | 0.9963 | 1 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004343392 | 0.9926 | 2 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004472880 | 0.9341 | 10 | 那可不简单就是为了热的时候打着山两份 | cer | 0.3684 |
| 320 | streaming_asr | emilia_zh_0004472880 | 0.9396 | 9 | 那可不见得就是为了热的时候打着山凉快 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0004472880 | 0.9341 | 9 | 那可不见得就是为了热的时候打着扇子凉快 | cer | 0.1053 |
| 1280 | streaming_asr | emilia_zh_0004472880 | 0.9396 | 8 | 那可不见得就是为了热的时候打着扇子凉快 | cer | 0.1053 |
| 160 | causal_full_asr | emilia_zh_0004692595 | 0.7608 | 40 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004692595 | 0.7475 | 44 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004692595 | 0.7342 | 43 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004692595 | 0.7641 | 39 | Different equation and then later the differential equation interpreted in black diagram terms | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004724705 | 0.8246 | 32 | And while had to and when we is already I had commit that were really cognitive but which seen like wanting physical of first | wer | 0.5833 |
| 320 | streaming_asr | emilia_zh_0004724705 | 0.7895 | 42 | And while had to and when we deserted I had commit that were really cocking of but which seemed like worrying physical of first | wer | 0.5833 |
| 640 | streaming_asr | emilia_zh_0004724705 | 0.7865 | 41 | And while had to and when we is already I had commit that were really cocking but which seemed like worrying physical of first | wer | 0.5833 |
| 1280 | streaming_asr | emilia_zh_0004724705 | 0.7749 | 42 | And while had to and when we deserted I had commit that were really cocking but which seemed like worrying physical of first | wer | 0.5417 |
| 160 | streaming_asr | emilia_zh_0004797649 | 0.9791 | 3 | In no Tony has got word catch | wer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0004797649 | 0.9267 | 7 | In no told me has got the word catch | wer | 0.5000 |
| 640 | streaming_asr | emilia_zh_0004797649 | 0.9215 | 7 | In no told me has got the word catch | wer | 0.5000 |
| 1280 | streaming_asr | emilia_zh_0004797649 | 0.9267 | 8 | In no told me has got the word catch | wer | 0.5000 |
| 160 | streaming_asr | emilia_zh_0004879738 | 0.7233 | 22 | The resignations infuriated Elizabeth and sunny | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004879738 | 0.7170 | 24 | The resignations infuriated Elizabeth and sunny | wer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004879738 | 0.7170 | 21 | The resignations infuriated Elizabeth and sunny | wer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004879738 | 0.7358 | 21 | The resignations infuriated Elizabeth and sunny | wer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0005059483 | 0.9813 | 6 | 不是的目的如果仅仅如果了获取更大的他不是也就不成为不是在这对普通语言 | cer | 0.4286 |
| 320 | streaming_asr | emilia_zh_0005059483 | 0.9733 | 8 | 不是的目的如果仅仅说为了获取更大的他不是也就不成为不是在这对普通语言 | cer | 0.4000 |
| 640 | streaming_asr | emilia_zh_0005059483 | 0.9759 | 8 | 不是的目的如果仅仅说为了获取更大的他不是也就不成为不是在这对普通语言 | cer | 0.4000 |
| 1280 | streaming_asr | emilia_zh_0005059483 | 0.9786 | 6 | 不是的目的如果仅仅说为了获取更大的他不是也就不成为不是在这对普通语言 | cer | 0.4000 |
| 160 | causal_full_asr | emilia_zh_0005245611 | 0.9661 | 7 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 320 | causal_full_asr | emilia_zh_0005245611 | 0.9593 | 9 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 640 | causal_full_asr | emilia_zh_0005245611 | 0.9525 | 11 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 1280 | causal_full_asr | emilia_zh_0005245611 | 0.9458 | 11 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 160 | streaming_asr | emilia_zh_0005370632 | 0.9830 | 4 | 你是看不到说你这个形态展发展上去 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0005370632 | 0.9830 | 4 | 你是看不到说你这个形态展发展上去 | cer | 0.1250 |
| 640 | streaming_asr | emilia_zh_0005370632 | 0.9702 | 7 | 你是看不到说你这个形态展发展起来上去 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005370632 | 0.9702 | 4 | 你是看不到说你这个形态展发展上去 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0005669352 | 0.9798 | 8 | 是听沙仪式两把红狼头都抖动的一下帮助着而落了一定哥哥趴下上的东边的例子一看是骗子货 | cer | 0.3902 |
| 320 | streaming_asr | emilia_zh_0005669352 | 0.9617 | 14 | 是听沙拉的一两朵红狼都都抖动了一下帮助着而落了一定哥哥爬上的东边的例子一看是骗子货物 | cer | 0.3171 |
| 640 | streaming_asr | emilia_zh_0005669352 | 0.9657 | 14 | 是听沙子仪式两朵红廊道都抖动了一下帮助着而落了一定哥哥爬上的东根的例子一当然是骗子货物 | cer | 0.4146 |
| 1280 | streaming_asr | emilia_zh_0005669352 | 0.9657 | 14 | 是听沙的两朵红狼头都抖动的一下帮助着而落了一定哥哥爬上的东哥的例子一个看是骗子货 | cer | 0.3415 |
| 160 | streaming_asr | emilia_zh_0005903796 | 0.9563 | 10 | 而且给他我咱们录节目节目我朋友过微信说他正在打车然后打车的司机大姐这个的说 | cer | 0.3171 |
| 320 | streaming_asr | emilia_zh_0005903796 | 0.9437 | 13 | 而且刚刚我咱们路节目节目我朋友过微信说他正在打车然后打车的司机大姐一个能说 | cer | 0.3171 |
| 640 | streaming_asr | emilia_zh_0005903796 | 0.9500 | 14 | 而且刚刚我咱们路节目节目我朋友过微信说他正在打车然后打车的司机大姐一个的说 | cer | 0.3171 |
| 1280 | streaming_asr | emilia_zh_0005903796 | 0.9437 | 14 | 而且刚刚我咱们路节目节目我朋友过微信说他正在打车然后打车的司机大姐一个的说 | cer | 0.3171 |
| 160 | causal_full_asr | emilia_zh_0005926417 | 0.9724 | 9 | 没有办法判断就我自己没有办法去得出这样的判断和结论出来同时呢是参考一些呃 | cer | 0.1389 |
| 320 | causal_full_asr | emilia_zh_0005926417 | 0.9779 | 7 | 没有办法判就我自己没有办法去得出这样的判断和结论出来很正常是参考一些呃 | cer | 0.1111 |
| 640 | causal_full_asr | emilia_zh_0005926417 | 0.9751 | 7 | 没有办法判就我刺激没有办法去得出这样的判断或者得问出来控制的是参考一些呃 | cer | 0.2778 |
| 1280 | causal_full_asr | emilia_zh_0005926417 | 0.9779 | 6 | 没有办法判就我自己没有办法去得出这样的判断或者得问出来同志呢是参考一些呃 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006119067 | 0.9617 | 6 | 而且这点还有一个很让人就是小我们明白的点这这个是 | cer | 0.2174 |
| 320 | streaming_asr | emilia_zh_0006119067 | 0.9330 | 10 | 而且这点还有一个很让人就是想我们明白的点啊这这个是 | cer | 0.1739 |
| 640 | streaming_asr | emilia_zh_0006119067 | 0.9330 | 12 | 而且这里还有一个很让人就是想我们明白的点啊这一个是 | cer | 0.1304 |
| 1280 | streaming_asr | emilia_zh_0006119067 | 0.9282 | 13 | 而且这里还有一个很让人就是想我们明白的点啊这一个是 | cer | 0.1304 |
| 160 | streaming_asr | emilia_zh_0006330534 | 0.8797 | 17 | I had no once put Several days At last I received to short note | wer | 0.2143 |
| 320 | streaming_asr | emilia_zh_0006330534 | 0.8660 | 18 | I had no once for Several days At last I received to short note | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0006330534 | 0.8419 | 23 | I had no answer for Several days At last I received to short note | wer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0006330534 | 0.8522 | 25 | I had no answer for Several days At last I received to short note | wer | 0.0714 |
| 160 | causal_full_asr | emilia_zh_0006330755 | 0.7465 | 29 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0006330755 | 0.7189 | 33 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006330755 | 0.7051 | 33 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0006330755 | 0.7281 | 34 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0006430973 | 0.8137 | 37 | The just to was was something thing Greg could do without if he had to But their writing does had to state | wer | 0.4545 |
| 320 | streaming_asr | emilia_zh_0006430973 | 0.8106 | 37 | The chest of was was something that Greg could without if he had to But the writing does had to state | wer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0006430973 | 0.8230 | 33 | The just of was was something that Greg could without if he had to But the writing does had to state | wer | 0.3182 |
| 1280 | streaming_asr | emilia_zh_0006430973 | 0.8106 | 36 | The chests of was was something that Greg could without if he had to But their writing does had to state | wer | 0.3636 |
| 160 | streaming_asr | emilia_zh_0006544664 | 0.9682 | 17 | 后院后院商贸走了很勤这种小吃在广州的们的的是而且司机当时也没注意只注意到车的颜色是这种灰色打了 | cer | 0.2979 |
| 320 | streaming_asr | emilia_zh_0006544664 | 0.9665 | 17 | 好运货运商贸走和琴这种小车在广州的满载的是而且司机当时也没注意只注意到车的颜色深色这种灰色了 | cer | 0.2553 |
| 640 | streaming_asr | emilia_zh_0006544664 | 0.9631 | 18 | 好运货运商贸走了很勤这种小车在广州的满载的是而且司机当时也没注意只注意到车的颜色是这种灰色哪儿了 | cer | 0.2128 |
| 1280 | streaming_asr | emilia_zh_0006544664 | 0.9648 | 18 | 合约货运商贸走了很勤这种小车在广州的满载的是而且司机当时也没注意只注意到车的颜色是这种灰色哪儿了 | cer | 0.2340 |
| 160 | streaming_asr | emilia_zh_0006731464 | 0.9778 | 3 | 他选择这种生活方式一定有自己的道理 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0006731464 | 0.9722 | 4 | 他选择这种生活方式一定有自己的道理 | cer | 0.0556 |
| 640 | streaming_asr | emilia_zh_0006731464 | 0.9778 | 3 | 他选择这种生活方式一定有自己的道理 | cer | 0.0556 |
| 1280 | streaming_asr | emilia_zh_0006731464 | 0.9722 | 4 | 他选择这种生活方式一定有自己的道理 | cer | 0.0556 |
| 160 | causal_full_asr | emilia_zh_0006992873 | 0.9868 | 2 | 绿色的城堡大曹人首先警觉起来 | cer | 0.2857 |
| 320 | causal_full_asr | emilia_zh_0006992873 | 0.9825 | 3 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006992873 | 0.9868 | 1 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0006992873 | 0.9868 | 2 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007017174 | 0.9642 | 10 | 从某种程度上到人不是出特别严的说法啊就是所得税恐怕就是工薪税 | cer | 0.1515 |
| 320 | streaming_asr | emilia_zh_0007017174 | 0.9669 | 9 | 从某种程度上到人不是出特别严的说法啊就就是所得税恐怕就是工薪税啊 | cer | 0.1515 |
| 640 | streaming_asr | emilia_zh_0007017174 | 0.9697 | 10 | 从某种程度上当人不是出特别严的说法啊就就是所得税恐怕就是工薪税啊 | cer | 0.1212 |
| 1280 | streaming_asr | emilia_zh_0007017174 | 0.9587 | 11 | 从某种程度上啊当人不是出特别严的说法啊就就是所得税恐怕就是工薪税啊 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0007312342 | 0.9958 | 2 | 凯莱他人生中的第一个支票账户他将他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 320 | streaming_asr | emilia_zh_0007312342 | 0.9958 | 2 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 640 | streaming_asr | emilia_zh_0007312342 | 0.9937 | 3 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 1280 | streaming_asr | emilia_zh_0007312342 | 0.9937 | 3 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 160 | streaming_asr | emilia_zh_0007526333 | 0.9304 | 21 | 那了最近过的的最重点一个就是每年考试都会考这个内容就是其实戏剧的他作品你 | cer | 0.3611 |
| 320 | streaming_asr | emilia_zh_0007526333 | 0.9356 | 19 | 到了这个的的最重点一个就是每年考试都会考这个内容就是其实继续做要要够品你 | cer | 0.3611 |
| 640 | streaming_asr | emilia_zh_0007526333 | 0.9330 | 20 | 到了最近各的最重点一个就是每年考试都会考这个内容就是不知道继续做要他过品你 | cer | 0.3889 |
| 1280 | streaming_asr | emilia_zh_0007526333 | 0.9304 | 22 | 到了这些各的最重点一个就是每年考试都会考这个内容就是其实戏剧的他过评你 | cer | 0.3056 |
| 160 | streaming_asr | emilia_zh_0007721307 | 0.9825 | 6 | 其实马云的很多演讲你呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0007721307 | 0.9700 | 10 | 其实马云的很多演讲你呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0556 |
| 640 | streaming_asr | emilia_zh_0007721307 | 0.9750 | 8 | 其实马云的很多演讲你呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0556 |
| 1280 | streaming_asr | emilia_zh_0007721307 | 0.9750 | 8 | 其实马云的很多演讲你呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0556 |
| 160 | streaming_asr | EN_B00052_S08802_W000006 | 0.9685 | 7 | Word Beautiful Pretty | wer | 0.3333 |
| 320 | streaming_asr | EN_B00052_S08802_W000006 | 0.9558 | 7 | Word Beautiful Pretty | wer | 0.3333 |
| 640 | streaming_asr | EN_B00052_S08802_W000006 | 0.9653 | 5 | Word Beautiful Pretty | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00052_S08802_W000006 | 0.9653 | 5 | Word Beautiful Pretty | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00043_S01623_W000011 | 0.8811 | 31 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00043_S01623_W000011 | 0.8762 | 32 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00043_S01623_W000011 | 0.8519 | 33 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00043_S01623_W000011 | 0.8519 | 32 | I found all that hard to believe too it must have been terrible and there was nothing anyone could do about it | wer | 0.0000 |
| 160 | streaming_asr | EN_B00089_S01559_W000004 | 0.8636 | 15 | He never understood soaks or Arrange or homework | wer | 0.3750 |
| 320 | streaming_asr | EN_B00089_S01559_W000004 | 0.8512 | 17 | And never understood socks or <\|write_generate\|><\|eng\|><\|start_content\|>Along<\|bicodec_semantic_6746\|><\|bicodec_semantic_4285\|><\|bicodec_semantic_5482\|><\|bicodec_semantic_1623\|><\|bicodec_semantic_284\|><\|bicodec_semantic_533\|><\|bicodec_semantic_1965\|><\|bicodec_semantic_7094\|><\|bicodec_semantic_1089\|><\|bicodec_semantic_7652\|><\|bicodec_semantic_3546\|><\|bicodec_semantic_139\|><\|bicodec_semantic_7354\|><\|bicodec_semantic_7354\|><\|bicodec_semantic_7306\|><\|bicodec_semantic_5589\|><\|bicodec_semantic_3047\|> or homework | wer | 0.2500 |
| 640 | streaming_asr | EN_B00089_S01559_W000004 | 0.8512 | 18 | And ever understood socks or Alonging are homework | wer | 0.5000 |
| 1280 | streaming_asr | EN_B00089_S01559_W000004 | 0.8099 | 22 | And ever understood socks or Alonging are homework | wer | 0.5000 |
| 160 | causal_full_asr | EN_B00048_S07041_W000493 | 0.8176 | 17 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07041_W000493 | 0.8059 | 18 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07041_W000493 | 0.7882 | 20 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07041_W000493 | 0.7647 | 21 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S07042_W000076 | 0.7975 | 25 | I no I I didn't have time to put things with before you got here | wer | 0.2143 |
| 320 | streaming_asr | EN_B00048_S07042_W000076 | 0.6810 | 31 | I no I I didn't have time to put things way before you got here | wer | 0.2143 |
| 640 | streaming_asr | EN_B00048_S07042_W000076 | 0.6319 | 32 | I no I hadn't have time to put things way before you got here | wer | 0.2143 |
| 1280 | streaming_asr | EN_B00048_S07042_W000076 | 0.6626 | 30 | I no I at the have time to put things way before you got here | wer | 0.2857 |
| 160 | streaming_asr | EN_B00058_S03815_W000004 | 0.9472 | 11 | My child No her both m Magic strong in love to make one figure that past | wer | 0.5333 |
| 320 | streaming_asr | EN_B00058_S03815_W000004 | 0.9384 | 14 | My child No her ball m Magic strong in a to make one figure that past | wer | 0.5333 |
| 640 | streaming_asr | EN_B00058_S03815_W000004 | 0.9267 | 15 | My child No her will m Magic is strong in a to make one figure that past | wer | 0.4667 |
| 1280 | streaming_asr | EN_B00058_S03815_W000004 | 0.9238 | 14 | My child No her ball m Magic is strong in a to make one figure that past | wer | 0.4667 |
| 160 | streaming_asr | EN_B00036_S05316_W000048 | 0.8317 | 26 | This Sunflower see star has a three foot wide arms span at the tastes for see fortune | wer | 0.4118 |
| 320 | streaming_asr | EN_B00036_S05316_W000048 | 0.7954 | 28 | This sunflower see star has a three foot wide arms span and tastes for see fortune | wer | 0.3529 |
| 640 | streaming_asr | EN_B00036_S05316_W000048 | 0.7756 | 33 | This sunflower see star has a three foot wide arms span at the tastes for see fortune | wer | 0.4118 |
| 1280 | streaming_asr | EN_B00036_S05316_W000048 | 0.7789 | 32 | This sunflower see star has a three foot wide arms span at the tastes for see <\|glm_semantic_4848\|>chers | wer | 0.4118 |
| 160 | causal_full_asr | EN_B00083_S08530_W000016 | 0.8486 | 18 | You're unmute we're gonna hear you Sorry Good morning everybody | wer | 0.5000 |
| 320 | causal_full_asr | EN_B00083_S08530_W000016 | 0.8486 | 18 | You're unmute we're gonna hear you Sorry Good morning everybody | wer | 0.5000 |
| 640 | causal_full_asr | EN_B00083_S08530_W000016 | 0.8287 | 21 | You're unmute we're gonna hear you Sorry Good morning everybody | wer | 0.5000 |
| 1280 | causal_full_asr | EN_B00083_S08530_W000016 | 0.8127 | 21 | Your unmute we're gonna cure you Sorry Good morning everybody | wer | 0.5833 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
