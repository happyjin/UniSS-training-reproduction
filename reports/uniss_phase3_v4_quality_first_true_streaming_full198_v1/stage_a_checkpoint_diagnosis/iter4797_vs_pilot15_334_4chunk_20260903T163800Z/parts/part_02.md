# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 172
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9511**
- Weighted CTC blank ratio: **0.7134**
- Weighted streaming WER/CER: **0.1718**
- Weighted causal-full WER/CER: **0.1140**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000042530 | 0.7204 | 84 | The international exchange center is a multiplefunctional building with capturia accommodation and classrooms | wer | 0.1538 |
| 320 | streaming_asr | CommonVoice_EN_0000042530 | 0.6872 | 94 | The international exchange center is in multiplefunctional building with capitalia accommodation and classrooms | wer | 0.2308 |
| 640 | streaming_asr | CommonVoice_EN_0000042530 | 0.6517 | 98 | The international exchange center is in multiplefunctional building with capitalia accommodation and classrooms | wer | 0.2308 |
| 1280 | streaming_asr | CommonVoice_EN_0000042530 | 0.6540 | 97 | The international exchange center is in multiplefunctional building with capt interior accommodation and classrooms | wer | 0.3077 |
| 160 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7662 | 52 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7292 | 60 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7354 | 57 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7354 | 57 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000238037 | 0.9318 | 17 | Who find a fire in that book You are the right Which book | wer | 0.9286 |
| 320 | streaming_asr | CommonVoice_EN_0000238037 | 0.9221 | 19 | Who find a fire in now the moon for the white which you | wer | 0.8571 |
| 640 | streaming_asr | CommonVoice_EN_0000238037 | 0.8929 | 25 | New find of the fire in Now we need to preside the book on | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000238037 | 0.9156 | 23 | You find the fire in now before you start spire the pick on | wer | 0.8571 |
| 160 | streaming_asr | CommonVoice_EN_0000381265 | 0.7829 | 26 | You existed in playing megan | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000381265 | 0.7632 | 27 | You existed in playing megan | wer | 0.8000 |
| 640 | streaming_asr | CommonVoice_EN_0000381265 | 0.7763 | 27 | You vexisted in playing megan | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000381265 | 0.7566 | 29 | You've existed since playing megan | wer | 0.6000 |
| 160 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8346 | 15 | Nothing personal on it | wer | 0.2500 |
| 320 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8425 | 15 | Nothing personal in it | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8346 | 15 | Nothing personal in it | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8189 | 17 | Nothing personal in it | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000520146 | 0.6793 | 42 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000520146 | 0.6576 | 44 | Many trucks are supported by a leave spring suspensions | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000520146 | 0.6522 | 44 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000520146 | 0.6359 | 44 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 160 | streaming_asr | LibriSpeech_0000033920 | 0.5842 | 178 | And then the representatives who had been picked for everything but their grass of science and government when to penic over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 320 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 640 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 1280 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a miff of national prestige The space effort was turned over to the ear craft in industry | wer | 0.2500 |
| 160 | streaming_asr | LibriSpeech_0000124435 | 0.7206 | 152 | So now all the the children saw upon their plates apples sauce and squash and tomeadow and sweet potato and sour potato Not one of them could eat mouthful because not one was safed with meat | wer | 0.1892 |
| 320 | streaming_asr | LibriSpeech_0000124435 | 0.7018 | 154 | So now all the the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato Not one of their and cody they mouthful because not one was safed with meat | wer | 0.2162 |
| 640 | streaming_asr | LibriSpeech_0000124435 | 0.6734 | 160 | So now although the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato not one of their and cody they malful because not one was safed with the meat | wer | 0.1622 |
| 1280 | streaming_asr | LibriSpeech_0000124435 | 0.6721 | 163 | So now although the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato not one of their and cody they malful because not one was satisfied with the meat | wer | 0.1351 |
| 160 | causal_full_asr | LibriSpeech_0000238204 | 0.7607 | 122 | What answer one of those ever tells Why men who advertise for wives can only be speedy adventurers The sort of person one reads of in books and that in needs and meal lines | wer | 0.3125 |
| 320 | causal_full_asr | LibriSpeech_0000238204 | 0.7292 | 132 | Would answer one of those advertisements Why men who advertise for whites can only be speedy adventurers the sort of person one reads of in books and knitting meets a meal on | wer | 0.1875 |
| 640 | causal_full_asr | LibriSpeech_0000238204 | 0.7235 | 132 | Would answer one of those advertisements Why men who advertise for wives can only be speedy adventurers the sort of person one reads of in books and knitting needs a meal | wer | 0.1875 |
| 1280 | causal_full_asr | LibriSpeech_0000238204 | 0.7149 | 134 | Would answer one of those evertastant Why men who advertise for wives can only be speedy adventurers the sort of person one reads of in books and they don't need some meal life | wer | 0.2188 |
| 160 | streaming_asr | LibriSpeech_0000271006 | 0.6465 | 133 | And not make an attempt to get money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 320 | streaming_asr | LibriSpeech_0000271006 | 0.6282 | 134 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 640 | streaming_asr | LibriSpeech_0000271006 | 0.6245 | 136 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 1280 | streaming_asr | LibriSpeech_0000271006 | 0.6447 | 131 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 160 | streaming_asr | emilia_zh_0004036114 | 0.7151 | 51 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004036114 | 0.7097 | 49 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004036114 | 0.6989 | 54 | 没有上我的大学家里情况可以说是一言难尽 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0004036114 | 0.7043 | 51 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004176769 | 0.7401 | 40 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004176769 | 0.7232 | 44 | 最终我们会从床上爬起来找到点事情做 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0004176769 | 0.7401 | 41 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004176769 | 0.7288 | 44 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004343392 | 0.8339 | 37 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004343392 | 0.8192 | 43 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004343392 | 0.8044 | 45 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004343392 | 0.8044 | 45 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004472880 | 0.7088 | 50 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0004472880 | 0.7198 | 44 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 640 | streaming_asr | emilia_zh_0004472880 | 0.7088 | 48 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 1280 | streaming_asr | emilia_zh_0004472880 | 0.7198 | 46 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 160 | causal_full_asr | emilia_zh_0004692595 | 0.6711 | 72 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004692595 | 0.6312 | 77 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 640 | causal_full_asr | emilia_zh_0004692595 | 0.6146 | 80 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 1280 | causal_full_asr | emilia_zh_0004692595 | 0.6113 | 81 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0004724705 | 0.5731 | 103 | And while I had to run it when we started I had d submitted that we really talking about what seems like wanting physical to first | wer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0004724705 | 0.5234 | 113 | And while I had to open one we simulated I had some that were really talking about what seems like wanting physical to first | wer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0004724705 | 0.5175 | 112 | And while I had to manage when we surrendered I had to submit that we're really talking about what seems like wanting physical to first | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004724705 | 0.5175 | 111 | And while I had to open the end research I had submit that we're really talking about what seems like wanting physical to first | wer | 0.3750 |
| 160 | streaming_asr | emilia_zh_0004797649 | 0.8377 | 23 | In no Tony has got the word catch | wer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004797649 | 0.8115 | 26 | In now Tony has got the word catch | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0004797649 | 0.8115 | 25 | In no tony has got the word catch | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004797649 | 0.8063 | 26 | In a tony has got the word catch | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 44 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004879738 | 0.5849 | 44 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 41 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 41 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0005059483 | 0.7246 | 97 | 不是的目的如果仅仅说为了获取更大的答案不是也就不成为故事在这对于普通语言 | cer | 0.3714 |
| 320 | streaming_asr | emilia_zh_0005059483 | 0.7246 | 99 | 不是的目的如果仅仅是为了获取更大的答案不是也就不成为布什再者对普通员 | cer | 0.2571 |
| 640 | streaming_asr | emilia_zh_0005059483 | 0.7193 | 97 | 故事的目的如果仅仅是为了获取更大的答案不是也就不成为故事再者对于普通人 | cer | 0.2286 |
| 1280 | streaming_asr | emilia_zh_0005059483 | 0.7193 | 96 | 不是的目的如果仅仅是为了获取更大的答案不是也就不成为故事再者对于普通人 | cer | 0.2286 |
| 160 | causal_full_asr | emilia_zh_0005245611 | 0.7458 | 69 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 320 | causal_full_asr | emilia_zh_0005245611 | 0.7322 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 640 | causal_full_asr | emilia_zh_0005245611 | 0.7288 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 1280 | causal_full_asr | emilia_zh_0005245611 | 0.7288 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 160 | streaming_asr | emilia_zh_0005370632 | 0.8085 | 37 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005370632 | 0.7872 | 44 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005370632 | 0.7787 | 44 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005370632 | 0.7787 | 42 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005669352 | 0.7762 | 103 | 这天杀的一杀两口红狼头都抖动了一下帮助儿子落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.2683 |
| 320 | streaming_asr | emilia_zh_0005669352 | 0.7581 | 111 | 是天杀的一生两朵红狼头都抖动了一下绑着子儿落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.1951 |
| 640 | streaming_asr | emilia_zh_0005669352 | 0.7621 | 109 | 是听沙子一声两朵红狼头都抖动了一下绑着子儿落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.1463 |
| 1280 | streaming_asr | emilia_zh_0005669352 | 0.7601 | 106 | 是听沙子一声两朵红狼头都抖动了一下棒着子儿落了一地哥哥爬上了东边的栗子一看是骗子货 | cer | 0.0976 |
| 160 | streaming_asr | emilia_zh_0005903796 | 0.6844 | 94 | 而且刚才我咱们录节目之前我朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0976 |
| 320 | streaming_asr | emilia_zh_0005903796 | 0.6531 | 102 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 640 | streaming_asr | emilia_zh_0005903796 | 0.6406 | 106 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 1280 | streaming_asr | emilia_zh_0005903796 | 0.6406 | 102 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 160 | causal_full_asr | emilia_zh_0005926417 | 0.7403 | 85 | 没有办法判就我自己没有办法去得出这样的判断和结论出来可能只能是参考一些呃 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005926417 | 0.7403 | 89 | 没有办法判就我自己没有办法去得出这样的判断和结论出来从只能是参考一些呃 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005926417 | 0.7320 | 90 | 没有办法判断就我自己没有办法去得出这样的判断或者结论出来横指可能是参考一些呃 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005926417 | 0.7320 | 92 | 没有办法判就我自己没有办法去得出这样的判断或者结论出来横指可能是参考一些呃 | cer | 0.1389 |
| 160 | streaming_asr | emilia_zh_0006119067 | 0.7177 | 50 | 而且一点还有一个很让人人就是想的明白的点这个是 | cer | 0.2174 |
| 320 | streaming_asr | emilia_zh_0006119067 | 0.6938 | 54 | 而且也得还有一个很让人人就是想的明白的点啊这个是是 | cer | 0.2174 |
| 640 | streaming_asr | emilia_zh_0006119067 | 0.6699 | 58 | 而且也还有一个很让人人就是想我们明白的点啊这个是是 | cer | 0.2609 |
| 1280 | streaming_asr | emilia_zh_0006119067 | 0.6507 | 61 | 而且也还有一个很让人就是想我们明白的点啊这个是是 | cer | 0.2174 |
| 160 | streaming_asr | emilia_zh_0006330534 | 0.7320 | 56 | I had no answer for several days At last I received a short note | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006330534 | 0.7285 | 56 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006330534 | 0.7113 | 58 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006330534 | 0.7285 | 55 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006330755 | 0.5991 | 66 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0006330755 | 0.5714 | 63 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006330755 | 0.5714 | 64 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0006330755 | 0.5622 | 65 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0006430973 | 0.6273 | 92 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0006430973 | 0.6025 | 92 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0006430973 | 0.5994 | 93 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0006430973 | 0.5807 | 96 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006544664 | 0.7789 | 118 | 后院货运商贸走了很勤这种小车在广州那么晚的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1702 |
| 320 | streaming_asr | emilia_zh_0006544664 | 0.7705 | 125 | 货运货运商贸走了很勤这种小车在广州那么短的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1489 |
| 640 | streaming_asr | emilia_zh_0006544664 | 0.7739 | 128 | 贺运霍玉商贸走了很勤这种小车在广州那满大街都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1277 |
| 1280 | streaming_asr | emilia_zh_0006544664 | 0.7638 | 131 | 客运货运商贸走了很勤这种小车在广州那么满的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1277 |
| 160 | streaming_asr | emilia_zh_0006731464 | 0.7056 | 48 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006731464 | 0.7056 | 48 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006731464 | 0.6889 | 49 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006731464 | 0.6889 | 49 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006992873 | 0.8158 | 36 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0006992873 | 0.8158 | 34 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006992873 | 0.8026 | 39 | 绿色的城堡稻草人首先警叫起来 | cer | 0.0714 |
| 1280 | causal_full_asr | emilia_zh_0006992873 | 0.8114 | 36 | 绿色的城堡稻草人首先警叫起来 | cer | 0.0714 |
| 160 | streaming_asr | emilia_zh_0007017174 | 0.7410 | 83 | 从某种程度上啊当然不是出特地演个的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0909 |
| 320 | streaming_asr | emilia_zh_0007017174 | 0.7300 | 88 | 从某种程度上啊当然不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007017174 | 0.7245 | 91 | 从某种程度上啊当人不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0303 |
| 1280 | streaming_asr | emilia_zh_0007017174 | 0.7245 | 89 | 从某种程度上啊当然不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007312342 | 0.8101 | 82 | 凯勒他人生中的第一个支票账户他将他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 320 | streaming_asr | emilia_zh_0007312342 | 0.8059 | 86 | 凯勒他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 640 | streaming_asr | emilia_zh_0007312342 | 0.8101 | 86 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 1280 | streaming_asr | emilia_zh_0007312342 | 0.8038 | 82 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 160 | streaming_asr | emilia_zh_0007526333 | 0.7268 | 99 | 哦了这些各的最重点一个就是每年考试都会考这个内容就是普希拉戏剧作家和作品 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0007526333 | 0.7191 | 102 | 到了这些各的最重点一个就是每年考试都会考这个内容就是普希拉戏剧作家和作品 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0007526333 | 0.7242 | 101 | 到了这些各的最重点一个就是每年考试都会考这个内容就是古希腊戏剧作家和作品 | cer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0007526333 | 0.7294 | 98 | 到了这些各的最重点一个就是每年考试都会考这个内容就是古希腊戏剧作家和作品 | cer | 0.0833 |
| 160 | streaming_asr | emilia_zh_0007721307 | 0.7425 | 91 | 其实马云的很多演讲里呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0278 |
| 320 | streaming_asr | emilia_zh_0007721307 | 0.7400 | 93 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007721307 | 0.7425 | 92 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007721307 | 0.7250 | 97 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 160 | streaming_asr | EN_B00052_S08802_W000006 | 0.9274 | 15 | Word beautiful Pretty | wer | 0.3333 |
| 320 | streaming_asr | EN_B00052_S08802_W000006 | 0.9085 | 16 | Word beautiful pretty | wer | 0.3333 |
| 640 | streaming_asr | EN_B00052_S08802_W000006 | 0.9117 | 19 | Word beautiful pretty | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00052_S08802_W000006 | 0.9148 | 18 | Word beautiful pretty | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00043_S01623_W000011 | 0.7160 | 80 | I found all that I had to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0909 |
| 320 | causal_full_asr | EN_B00043_S01623_W000011 | 0.6942 | 89 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00043_S01623_W000011 | 0.7039 | 89 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00043_S01623_W000011 | 0.6869 | 90 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 160 | streaming_asr | EN_B00089_S01559_W000004 | 0.7934 | 37 | And every understood soaks or Lange are homework | wer | 0.6250 |
| 320 | streaming_asr | EN_B00089_S01559_W000004 | 0.7603 | 40 | And ever understood sox or Laungie or homework | wer | 0.5000 |
| 640 | streaming_asr | EN_B00089_S01559_W000004 | 0.7438 | 42 | And ever understood soaks or Lange are homework | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00089_S01559_W000004 | 0.7562 | 42 | And ever understood soaks or Lange are homework | wer | 0.6250 |
| 160 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6471 | 44 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6235 | 46 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6235 | 47 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6118 | 46 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S07042_W000076 | 0.6012 | 49 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 320 | streaming_asr | EN_B00048_S07042_W000076 | 0.5337 | 54 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 640 | streaming_asr | EN_B00048_S07042_W000076 | 0.5337 | 52 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S07042_W000076 | 0.5460 | 53 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S03815_W000004 | 0.7830 | 54 | My child no her but magic strong enough to make one forget their past | wer | 0.2000 |
| 320 | streaming_asr | EN_B00058_S03815_W000004 | 0.7390 | 60 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 640 | streaming_asr | EN_B00058_S03815_W000004 | 0.7155 | 61 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 1280 | streaming_asr | EN_B00058_S03815_W000004 | 0.7185 | 60 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 160 | streaming_asr | EN_B00036_S05316_W000048 | 0.7162 | 57 | This sunderflow sees star has a three foot wide arms span as a taste for sear chance | wer | 0.3529 |
| 320 | streaming_asr | EN_B00036_S05316_W000048 | 0.6766 | 67 | This sunflower sees star has a three foot wide arms span and taste for sear chance | wer | 0.2941 |
| 640 | streaming_asr | EN_B00036_S05316_W000048 | 0.6502 | 65 | This sunflower sees star has a three foot wide arms span at the taste for sear chance | wer | 0.3529 |
| 1280 | streaming_asr | EN_B00036_S05316_W000048 | 0.6172 | 69 | This sunflower sees star has a three foot wide arms span as the taste for sear chance | wer | 0.3529 |
| 160 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7729 | 40 | You're unmute we're gonna cheer you Sorry Good morning everybody | wer | 0.5833 |
| 320 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7131 | 49 | You're unmute We're gonna hear you Sorry Good morning everybody | wer | 0.5000 |
| 640 | causal_full_asr | EN_B00083_S08530_W000016 | 0.6932 | 50 | You're unmute We're not here You Sorry Good morning everybody | wer | 0.5833 |
| 1280 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7012 | 48 | You're unmute We're not here You Sorry Good morning everybody | wer | 0.5833 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
