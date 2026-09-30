# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9509**
- Weighted CTC blank ratio: **0.6812**
- Weighted streaming WER/CER: **0.2483**
- Weighted causal-full WER/CER: **0.0903**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000188343 | 0.8504 | 16 | They have such as thoroughly | wer | 0.4000 |
| 320 | streaming_asr | CommonVoice_EN_0000188343 | 0.8268 | 19 | They have such as thoroughly | wer | 0.4000 |
| 640 | streaming_asr | CommonVoice_EN_0000188343 | 0.8425 | 17 | They have such as thoroughly the | wer | 0.6000 |
| 1280 | streaming_asr | CommonVoice_EN_0000188343 | 0.8031 | 18 | Day of such as thoroughly the | wer | 1.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000331841 | 0.7692 | 23 | You did roll the some of last time | wer | 0.6250 |
| 320 | streaming_asr | CommonVoice_EN_0000331841 | 0.7538 | 22 | You did wrote the some of last time | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000331841 | 0.7462 | 24 | You did wrote the some of last time | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000331841 | 0.7385 | 26 | You did wrote the some of last time | wer | 0.6250 |
| 160 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7262 | 62 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 320 | causal_full_asr | CommonVoice_EN_0000352714 | 0.6935 | 69 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 640 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7024 | 69 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 1280 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7083 | 68 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 160 | streaming_asr | CommonVoice_EN_0000471209 | 0.7603 | 29 | The boy began to do dear into the journ | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000471209 | 0.6849 | 34 | The boy began to deag into the journ | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000471209 | 0.6644 | 36 | The boy began to do dear and to the june | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000471209 | 0.6781 | 36 | The boy began to do dear and to the june | wer | 0.6250 |
| 160 | streaming_asr | HQ-Conversations_0000028308 | 0.7742 | 13 | 对兄弟咋了 | cer | 0.4000 |
| 320 | streaming_asr | HQ-Conversations_0000028308 | 0.7903 | 12 | 对兄弟咋了 | cer | 0.4000 |
| 640 | streaming_asr | HQ-Conversations_0000028308 | 0.7742 | 12 | 我想也咋办 | cer | 0.8000 |
| 1280 | streaming_asr | HQ-Conversations_0000028308 | 0.7258 | 16 | 我曾经也砸了 | cer | 1.2000 |
| 160 | causal_full_asr | LibriSpeech_0000011649 | 0.6062 | 172 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000011649 | 0.5900 | 172 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000011649 | 0.5914 | 173 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000011649 | 0.5796 | 176 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000090820 | 0.6442 | 186 | Which kept me in the how house fearnily too weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out in the fields all day hasking corn and when they came in noon | wer | 0.1458 |
| 320 | streaming_asr | LibriSpeech_0000090820 | 0.6132 | 195 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavenly safe and warm in those days like a tight little boat in a winter sea The men were out and the fields all day hasking corn and when they came in a noon | wer | 0.0833 |
| 640 | streaming_asr | LibriSpeech_0000090820 | 0.6199 | 194 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out and the fields all day husking corn and when they came in a noon | wer | 0.0833 |
| 1280 | streaming_asr | LibriSpeech_0000090820 | 0.6199 | 192 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out in the fields all day husking corn and when they came in a noon | wer | 0.0625 |
| 160 | streaming_asr | LibriSpeech_0000215121 | 0.6580 | 166 | And return by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Want to crystall another to his head in talk of offensive But that's his not all continued Dumbla | wer | 0.3611 |
| 320 | streaming_asr | LibriSpeech_0000215121 | 0.6594 | 167 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wanted Christ another to his head in talking of offensive But that's his not all continued Dumbla | wer | 0.3056 |
| 640 | streaming_asr | LibriSpeech_0000215121 | 0.6449 | 172 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wanted Christ another to his head in Ockham and of offensive But that's his not all continued Dumbla | wer | 0.3333 |
| 1280 | streaming_asr | LibriSpeech_0000215121 | 0.6435 | 172 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wants acrys to another to his head in talk of offensive But that's his not all continued Dumblaugh | wer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0003918326 | 0.7212 | 78 | 啊我我咱们还有一个点听一次这是这个事情连连起来就这些人他们演互相关联 | cer | 0.3611 |
| 320 | streaming_asr | emilia_zh_0003918326 | 0.6827 | 84 | 啊哦我想到还有一个点听一次就是这事情能连起来就这些人他们演互相关联 | cer | 0.2778 |
| 640 | streaming_asr | emilia_zh_0003918326 | 0.6699 | 85 | 啊我我想到还有一个点挺有意思的就是这些东西是人脸起来就这些人他们也互相关联 | cer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0003918326 | 0.6474 | 90 | 啊我我想到还有一个点你听有意思就是这些东西是人脸起来就这些人他们也互相关联 | cer | 0.1667 |
| 160 | causal_full_asr | emilia_zh_0003942539 | 0.6812 | 145 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 320 | causal_full_asr | emilia_zh_0003942539 | 0.6614 | 152 | 其实我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0003942539 | 0.6554 | 156 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 1280 | causal_full_asr | emilia_zh_0003942539 | 0.6535 | 159 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 35 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 36 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 36 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004129851 | 0.7654 | 37 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004358307 | 0.6534 | 124 | You was up to him to prove himself for there's six two make them proud of him and his music without the fate idea of how proud they were already | wer | 0.1333 |
| 320 | streaming_asr | emilia_zh_0004358307 | 0.6375 | 129 | It was a to him to prove himself for there's six two make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0004358307 | 0.6315 | 127 | It was up to him to prove himself for their sakes to make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.0667 |
| 1280 | streaming_asr | emilia_zh_0004358307 | 0.6275 | 128 | It was up to him to prove himself for their sakes to make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0004659190 | 0.6438 | 93 | Seeing the the wind was close to granting what he replaced When you are loved you can you anything creation | wer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0004659190 | 0.6306 | 95 | Seeing the the wind was close to granting what he replaced When you are loved you can do anything creation | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004659190 | 0.6095 | 99 | Seeing the the wind was close to granting what he replaced When you are loved you can do anything in creation | wer | 0.0952 |
| 1280 | streaming_asr | emilia_zh_0004659190 | 0.6121 | 102 | Seeing that the wind was close to granting what he replaced When you are loved you can you anything in creation | wer | 0.0952 |
| 160 | causal_full_asr | emilia_zh_0004659501 | 0.8023 | 48 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0004659501 | 0.7794 | 48 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0004659501 | 0.8023 | 47 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0004659501 | 0.7937 | 48 | And for old you just write nine or ten And then where do where to | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004776929 | 0.7439 | 59 | In the evening they stefled down for a big family dinner This was eat | wer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0004776929 | 0.6877 | 65 | In evening they stessled down for a big family dinner This was it | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004776929 | 0.7158 | 56 | In the evening they settled down for a big family dinner This was it | wer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0004776929 | 0.7123 | 58 | In the evening they st settled down for a big family dinner This was it | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004843272 | 0.5907 | 151 | It's principle tenderness that I con dominant growth is this supreme good or it least approxy for the supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.3226 |
| 320 | streaming_asr | emilia_zh_0004843272 | 0.5722 | 156 | It's principle tennett is that I economic growth is the supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.1935 |
| 640 | streaming_asr | emilia_zh_0004843272 | 0.5630 | 158 | It's principle tendency that I economic growth is this supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0004843272 | 0.5704 | 154 | It's principle tennant is that I economic growth is this supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.2258 |
| 160 | streaming_asr | emilia_zh_0004943713 | 0.7874 | 113 | 等因我知道他清楚了你在这上面删除了夕阳和乘客尝试问从业者黑洞狐狐他们那个情感的愚蠢 | cer | 0.4651 |
| 320 | streaming_asr | emilia_zh_0004943713 | 0.7721 | 118 | 当时因我知道他清楚了你在这上面三处的信仰和乘客的尝试问从业者奋斗无与乎他那个情感的愚蠢 | cer | 0.3488 |
| 640 | streaming_asr | emilia_zh_0004943713 | 0.7755 | 116 | 但是因我知道他清楚了你在这上面删除的信仰和乘客的尝试问从业者奋斗维护他们你的情感的愚蠢 | cer | 0.2093 |
| 1280 | streaming_asr | emilia_zh_0004943713 | 0.7806 | 114 | 但是因我知道他清楚了你在这上面删除的信仰和乘客尝试问从业者黑洞维护他们你的情感的愚蠢 | cer | 0.2791 |
| 160 | causal_full_asr | emilia_zh_0005070101 | 0.8058 | 109 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 320 | causal_full_asr | emilia_zh_0005070101 | 0.8043 | 118 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 640 | causal_full_asr | emilia_zh_0005070101 | 0.7997 | 115 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 1280 | causal_full_asr | emilia_zh_0005070101 | 0.8012 | 117 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0005313494 | 0.6582 | 46 | 那不能读了吧因为后面没有一个应该的拼 | cer | 0.3000 |
| 320 | streaming_asr | emilia_zh_0005313494 | 0.6772 | 47 | 那不能读了吧因为后面没有一个应该的拼 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0005313494 | 0.6076 | 54 | 他不能读了吧因为后面没有一个应该的拼 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005313494 | 0.5823 | 56 | 他就不能读了吧因为后面没有一个应该的拼 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0005578304 | 0.9205 | 26 | 对树叶归来就好奇的问 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005578304 | 0.9178 | 26 | 对树叶归来就好奇的问 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005578304 | 0.9233 | 25 | 对树叶回来就好奇的问 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005578304 | 0.9233 | 23 | 你对树叶回来就好奇的问 | cer | 0.2500 |
| 160 | causal_full_asr | emilia_zh_0005714451 | 0.8077 | 51 | 顺便撞到您接下来你可以看到其实前面的三目标 | cer | 0.4400 |
| 320 | causal_full_asr | emilia_zh_0005714451 | 0.8112 | 52 | 生产状态您接下来你可以看到其实前面的三目标 | cer | 0.2800 |
| 640 | causal_full_asr | emilia_zh_0005714451 | 0.7762 | 56 | 生产状态您接下来你可以看到其实前面的三目标 | cer | 0.2800 |
| 1280 | causal_full_asr | emilia_zh_0005714451 | 0.7832 | 56 | 生产状态您接下来你可以看到其实前边的三目标 | cer | 0.2400 |
| 160 | streaming_asr | emilia_zh_0005818033 | 0.7135 | 43 | 投票一个那个教程的啊给咱们这边 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0005818033 | 0.7193 | 43 | 投稿一个那个教程的啊给咱们这边 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0005818033 | 0.7018 | 43 | 投稿一个那个教程的啊给咱们这边 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0005818033 | 0.7193 | 43 | 投稿一个那个教程啊给咱们这边 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0006041629 | 0.6931 | 105 | 就看了一些信息以后觉得虽然是够的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1458 |
| 320 | streaming_asr | emilia_zh_0006041629 | 0.6955 | 111 | 所以看了一下一些信息以后觉得虽然觉得歌词的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0006041629 | 0.6832 | 115 | 就是看完这些信息以后觉得虽然觉得够的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1042 |
| 1280 | streaming_asr | emilia_zh_0006041629 | 0.6807 | 116 | 就是看完这些信息以后就既然这歌词了一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1458 |
| 160 | causal_full_asr | emilia_zh_0006099356 | 0.6515 | 75 | 就整个这个流向还比较多但是你非得走你确定是能拿到钱的，是吗 | cer | 0.1852 |
| 320 | causal_full_asr | emilia_zh_0006099356 | 0.6432 | 79 | 就整个这个流量还不比较多但是你非得知道你确定是能拿到钱的什么 | cer | 0.2963 |
| 640 | causal_full_asr | emilia_zh_0006099356 | 0.6473 | 77 | 就整个这个流量还不叫多但是你确定是能拿到钱的，是吗 | cer | 0.2593 |
| 1280 | causal_full_asr | emilia_zh_0006099356 | 0.6473 | 78 | 就整个这个流量还不叫多但是你确定这种你确定是能拿到钱的，是吗 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0006270122 | 0.7537 | 63 | 你这样不是不对的啊还是一个完整的这个原因啊它就是对应 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006270122 | 0.7388 | 65 | 你这样都是不对的啊是一个完整的这个原因啊它就是恶意 | cer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0006270122 | 0.7090 | 67 | 你这样不是不对的啊他是一个完整的这个原因啊他就是恶意 | cer | 0.2400 |
| 1280 | streaming_asr | emilia_zh_0006270122 | 0.7052 | 69 | 你这样就是不对的啊他是一个完整的这个原因啊他就是恶意 | cer | 0.2400 |
| 160 | streaming_asr | emilia_zh_0006379722 | 0.7294 | 56 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0006379722 | 0.7261 | 57 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0006379722 | 0.6931 | 60 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0006379722 | 0.7228 | 57 | But he couldn't leave the other of two He must take them with him all the way | wer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0006464698 | 0.6463 | 113 | I'll law to for his actually for example he wanted me to tell him my big account pastword and he kept taking photos of me from different angles | wer | 0.2963 |
| 320 | streaming_asr | emilia_zh_0006464698 | 0.6092 | 121 | A lot of ways I actually for example he wanted me to tell him my big account pastword and he kept taking photos of me from different angles | wer | 0.1481 |
| 640 | streaming_asr | emilia_zh_0006464698 | 0.6026 | 120 | A lot of for his actually for example he wanted me to tell him my bank account pastword and he kept taking photos of me from different angles | wer | 0.1481 |
| 1280 | streaming_asr | emilia_zh_0006464698 | 0.6004 | 121 | A longer of ways actually for example he wanted me to tell him my bank account pastword and he kept taking photos of me from different angles | wer | 0.1111 |
| 160 | causal_full_asr | emilia_zh_0006610442 | 0.7126 | 39 | 我觉得这这这有什么有意思就是这种 | cer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0006610442 | 0.7365 | 39 | 我觉得这这这什么有意思就是这种 | cer | 0.2778 |
| 640 | causal_full_asr | emilia_zh_0006610442 | 0.6886 | 45 | 我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0006610442 | 0.7006 | 43 | 我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006713619 | 0.6743 | 88 | 不管啊那些新人们你们要互关赶紧互关起来再给你们三分钟时间互关玩下播了啊 | cer | 0.1351 |
| 320 | streaming_asr | emilia_zh_0006713619 | 0.6612 | 94 | 不管啊那些新人们你们要互关赶紧互关起来再给你们三分钟时间互关玩下播了啊 | cer | 0.1351 |
| 640 | streaming_asr | emilia_zh_0006713619 | 0.6513 | 94 | 不管啊啊那些新人们你们要护光赶紧护光起来再给你们三分钟时间护光我要下播了啊 | cer | 0.2703 |
| 1280 | streaming_asr | emilia_zh_0006713619 | 0.6447 | 94 | 不管啊那些新人们你们要护光赶紧护光起来再给你们三分钟时间护光我要下播了啊 | cer | 0.2432 |
| 160 | streaming_asr | emilia_zh_0006940509 | 0.7586 | 38 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 320 | streaming_asr | emilia_zh_0006940509 | 0.7241 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 640 | streaming_asr | emilia_zh_0006940509 | 0.7299 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 1280 | streaming_asr | emilia_zh_0006940509 | 0.7241 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 160 | streaming_asr | emilia_zh_0007124790 | 0.8220 | 57 | 你好请问这是去老人之家的路吧男人看了小军一眼 | cer | 0.0455 |
| 320 | streaming_asr | emilia_zh_0007124790 | 0.8164 | 55 | 你好请问这是去老人之家的路吧男人看了小军一眼 | cer | 0.0455 |
| 640 | streaming_asr | emilia_zh_0007124790 | 0.8192 | 55 | 你好请问这是去老人之家的路吧男人看了小俊一眼 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007124790 | 0.8192 | 52 | 你好请问这是去老人之家的路吧男人看了小俊一眼 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007461662 | 0.8102 | 85 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 320 | streaming_asr | emilia_zh_0007461662 | 0.8000 | 89 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 640 | streaming_asr | emilia_zh_0007461662 | 0.8041 | 88 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 1280 | streaming_asr | emilia_zh_0007461662 | 0.8020 | 86 | 吧那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 160 | streaming_asr | emilia_zh_0007690753 | 0.7627 | 129 | 可能市场每次新想到的是仅仅过来一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1765 |
| 320 | streaming_asr | emilia_zh_0007690753 | 0.7593 | 132 | 可能是场每次生响到的时仅仅过来一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0007690753 | 0.7559 | 133 | 可能是场没成想到的是仅仅过了一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1373 |
| 1280 | streaming_asr | emilia_zh_0007690753 | 0.7559 | 135 | 可能是场没曾想到的是仅仅过了一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1176 |
| 160 | streaming_asr | EN_B00083_S02942_W000001 | 0.6089 | 121 | It is going to be an resources I actually be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.2258 |
| 320 | streaming_asr | EN_B00083_S02942_W000001 | 0.5948 | 119 | It is going to be a resources as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 640 | streaming_asr | EN_B00083_S02942_W000001 | 0.6019 | 121 | It is going to be a resources as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 1280 | streaming_asr | EN_B00083_S02942_W000001 | 0.6089 | 120 | It is going to be a right resource as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 160 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6363 | 243 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that would build success I've tried to do that throughout my relationship with them | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6107 | 257 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 640 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6096 | 259 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 1280 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6055 | 259 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 160 | streaming_asr | EN_B00013_S06799_W000009 | 0.6484 | 106 | Now with that we can have better model is safety fairness and help any users to have better resibility appropriate trust | wer | 0.2381 |
| 320 | streaming_asr | EN_B00013_S06799_W000009 | 0.6256 | 108 | Now with that we can have better model and sure safety fairness and help any user to have better resibility appropriate trust | wer | 0.2381 |
| 640 | streaming_asr | EN_B00013_S06799_W000009 | 0.6256 | 106 | And with that we can have better model and sure 50 fairness and help and user to have better visibility appropriate trust | wer | 0.2381 |
| 1280 | streaming_asr | EN_B00013_S06799_W000009 | 0.6324 | 105 | And when that we can have better model and sure 50 furnace and help and user to have better visibility appropriate trust | wer | 0.3333 |
| 160 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5587 | 124 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 320 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5561 | 122 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 640 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5614 | 118 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 1280 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5666 | 119 | In Tibetan Buddhism Mahlers are mainly used to count montras These montras can be recited for different purposes linked to working with mind | wer | 0.1304 |
| 160 | streaming_asr | EN_B00048_S03599_W000339 | 0.7656 | 49 | That's turned around I've my speak froze the blood of everyone close by | wer | 0.2308 |
| 320 | streaming_asr | EN_B00048_S03599_W000339 | 0.7500 | 56 | I turned around I might sreak froze the blood of everyone close by | wer | 0.2308 |
| 640 | streaming_asr | EN_B00048_S03599_W000339 | 0.7438 | 56 | I turned around I and my sreak froze the blood of everyone close by | wer | 0.1538 |
| 1280 | streaming_asr | EN_B00048_S03599_W000339 | 0.7562 | 52 | I turned around and my s shriek froze the blood of everyone close by | wer | 0.0769 |
| 160 | streaming_asr | EN_B00058_S03125_W000010 | 0.6509 | 171 | When tremor comes in contact with a base it changes a good color from yellow to red indicating that suppose solution is a base That is why attered stained turns red when a comes in contact with any kind of base | wer | 0.2143 |
| 320 | streaming_asr | EN_B00058_S03125_W000010 | 0.6193 | 178 | When tremor comes in contact with a base it changes a good color from yellow to red indicating that sopy solution is a base That is what why attered stained turns red when it comes in contact with any kind of base | wer | 0.2143 |
| 640 | streaming_asr | EN_B00058_S03125_W000010 | 0.6066 | 182 | When tremor comes in contact with a base it changes a good color from yellow to red indicating the to soapy solutions is a base That is what why attered stained turns red when it comes in contact with any kind of base | wer | 0.2381 |
| 1280 | streaming_asr | EN_B00058_S03125_W000010 | 0.5893 | 186 | When tremor comes in contact with a base it changes in its color from yellow to red indicating that to soapy solution is a base That is what why attered stained turns red when a come in contact with any kind of base | wer | 0.2143 |
| 160 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5816 | 117 | So you original meeting that we did a bunch about the shows about of seeing people in the inner earth You said that took place in September | wer | 0.1538 |
| 320 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5765 | 129 | So your original meeting that we did a bunch about the shows about of seeing people in the inner earth You said that took place in September | wer | 0.1154 |
| 640 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5383 | 132 | So your original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that's a place in September | wer | 0.0769 |
| 1280 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5357 | 129 | So your original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that took place in September | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S07511_W000000 | 0.4769 | 74 | Getting everybody out of the house and morning can be we really taught especially the first day school | wer | 0.2632 |
| 320 | streaming_asr | EN_B00058_S07511_W000000 | 0.4667 | 77 | Getting everybody out of the house and morning can be where really talk especially the first of school | wer | 0.2632 |
| 640 | streaming_asr | EN_B00058_S07511_W000000 | 0.4308 | 83 | Getting everybody out of the house and morning can be you really taught especially the first of school | wer | 0.2632 |
| 1280 | streaming_asr | EN_B00058_S07511_W000000 | 0.3897 | 91 | Getting everybody out of the house and morning can be very really taught especially the first time school | wer | 0.3158 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
