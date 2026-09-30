# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9579**
- Weighted CTC blank ratio: **0.7030**
- Weighted streaming WER/CER: **0.1524**
- Weighted causal-full WER/CER: **0.1228**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000068601 | 0.7665 | 71 | The archival of Liberalism collects document on the history of allevated Liberalism | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000068601 | 0.7500 | 75 | The archipelago of Liberalism collects document on the history of allevized Oberilism | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000068601 | 0.7429 | 76 | The archival of Liberalism collects document on the history of allevized Oberilese | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000068601 | 0.7217 | 77 | The archival the freemeworthy collects document on the history of allevized Olympias | wer | 0.5000 |
| 160 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6727 | 50 | Then he's an old man He's going to spend a month in Africa | wer | 0.0769 |
| 320 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6455 | 54 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6045 | 56 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 1280 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6227 | 53 | When he's an old man he's going to spend a month in Africa | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000263325 | 0.7386 | 41 | He has one younger brother Trevor read Nelson | wer | 0.1250 |
| 320 | streaming_asr | CommonVoice_EN_0000263325 | 0.7303 | 45 | He has one younger brother travour read Nelson | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000263325 | 0.7303 | 42 | He has one younger brother traveller read Nelson | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000263325 | 0.7095 | 44 | He has one younger brother traveller read Nelson | wer | 0.2500 |
| 160 | streaming_asr | CommonVoice_EN_0000381462 | 0.7802 | 35 | Chirrod spent the rest of his little life with Mrusus | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000381462 | 0.7629 | 38 | Chirrod spent a rest of his slow life with Mrusus | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000381462 | 0.7457 | 42 | Chirrived spent the rest of his slow life would Mrusus | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000381462 | 0.7457 | 43 | Girard spent the rest of his slept life with Mrusus | wer | 0.2222 |
| 160 | streaming_asr | CommonVoice_EN_0000555807 | 0.8491 | 27 | The color intensifies as a stop brightens | wer | 0.2857 |
| 320 | streaming_asr | CommonVoice_EN_0000555807 | 0.8208 | 30 | The color intensifies as a star brightens | wer | 0.1429 |
| 640 | streaming_asr | CommonVoice_EN_0000555807 | 0.8019 | 33 | The color intensifies as a star brightens | wer | 0.1429 |
| 1280 | streaming_asr | CommonVoice_EN_0000555807 | 0.7972 | 34 | The color intensifies as a star brightens | wer | 0.1429 |
| 160 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8643 | 28 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8107 | 40 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8214 | 38 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8107 | 38 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000035237 | 0.6345 | 139 | In that country there is a great deal of mass It covers the ground just as grass does here But the most interesting thing about these lemmings is the way they migrate | wer | 0.0312 |
| 320 | streaming_asr | LibriSpeech_0000035237 | 0.6176 | 143 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 640 | streaming_asr | LibriSpeech_0000035237 | 0.6091 | 141 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 1280 | streaming_asr | LibriSpeech_0000035237 | 0.6176 | 141 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 160 | streaming_asr | LibriSpeech_0000124551 | 0.5954 | 200 | The platter family assembled in the cellar were a boat to begin breakfast where it was discovered that what of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.2051 |
| 320 | streaming_asr | LibriSpeech_0000124551 | 0.5855 | 203 | The platter family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.1026 |
| 640 | streaming_asr | LibriSpeech_0000124551 | 0.5926 | 200 | The plants are family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.1282 |
| 1280 | streaming_asr | LibriSpeech_0000124551 | 0.5783 | 202 | The planters family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one And first there was but little notice taken of the circumstance | wer | 0.0769 |
| 160 | causal_full_asr | LibriSpeech_0000263750 | 0.7320 | 112 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1000 |
| 320 | causal_full_asr | LibriSpeech_0000263750 | 0.7136 | 114 | In a magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1500 |
| 640 | causal_full_asr | LibriSpeech_0000263750 | 0.7119 | 115 | In a magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1500 |
| 1280 | causal_full_asr | LibriSpeech_0000263750 | 0.7018 | 116 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sidley he announced in walk Reggie | wer | 0.0500 |
| 160 | streaming_asr | VCTK_0000006134 | 0.6647 | 37 | There's influence to the ident is like an arrow | wer | 0.2222 |
| 320 | streaming_asr | VCTK_0000006134 | 0.6706 | 38 | This influence to the audience is like an arrow | wer | 0.0000 |
| 640 | streaming_asr | VCTK_0000006134 | 0.6941 | 36 | This influence to the audience is like an arrow | wer | 0.0000 |
| 1280 | streaming_asr | VCTK_0000006134 | 0.6647 | 38 | This influence to the audience is like an arrow | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004064583 | 0.8006 | 54 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004064583 | 0.7850 | 56 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004064583 | 0.7757 | 61 | 我相信每一次巧合都是一道自信一个线索告诉我们 | cer | 0.0909 |
| 1280 | streaming_asr | emilia_zh_0004064583 | 0.7850 | 58 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004212211 | 0.7339 | 75 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 320 | streaming_asr | emilia_zh_0004212211 | 0.7125 | 81 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 640 | streaming_asr | emilia_zh_0004212211 | 0.7217 | 79 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 1280 | streaming_asr | emilia_zh_0004212211 | 0.7156 | 78 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 160 | causal_full_asr | emilia_zh_0004422836 | 0.7768 | 128 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能坦白交代罪行杰西在斯打中失守之命按当时编区的法令 | cer | 0.1961 |
| 320 | causal_full_asr | emilia_zh_0004422836 | 0.7768 | 130 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在斯打中失守致命按当时编区的法令 | cer | 0.2157 |
| 640 | causal_full_asr | emilia_zh_0004422836 | 0.7752 | 135 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在斯打中失守致命按当时编区的法令 | cer | 0.2157 |
| 1280 | causal_full_asr | emilia_zh_0004422836 | 0.7691 | 136 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在四大中失守致命按当时编区的法令 | cer | 0.2157 |
| 160 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 33 | 这这个姓王的客人呢呃是昨天来 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 35 | 对这位姓王的客人呢呃是昨天来 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 33 | 对这位姓王的客人呢呃是昨天来 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0004519522 | 0.7764 | 31 | 这这位姓王的客人呢呃是昨天来的 | cer | 0.0667 |
| 160 | causal_full_asr | emilia_zh_0004705832 | 0.6254 | 181 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live like kids | wer | 0.0833 |
| 320 | causal_full_asr | emilia_zh_0004705832 | 0.6193 | 183 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live like kids | wer | 0.0833 |
| 640 | causal_full_asr | emilia_zh_0004705832 | 0.6101 | 190 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live on kids | wer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0004705832 | 0.6070 | 188 | These words are automatic we no longer visualize it which is why it takes kids are longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live on kids | wer | 0.0833 |
| 160 | streaming_asr | emilia_zh_0004724727 | 0.7354 | 91 | Is it traditional in out But as you should for today the new things is uh speak with a a other for new | wer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0004724727 | 0.7375 | 86 | Is it traditional in out But a thing so for today the new things is a speaking with a a other for new | wer | 0.3478 |
| 640 | streaming_asr | emilia_zh_0004724727 | 0.7289 | 85 | Is it traditional in our but uh I still for today the new things is uh speak with a a other for new | wer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0004724727 | 0.7462 | 81 | Is it traditional in our but uh the military for today the new things is uh speak with a a other type new | wer | 0.3913 |
| 160 | streaming_asr | emilia_zh_0004804632 | 0.5920 | 60 | It is often a pretty sight when several of these boats are more together | wer | 0.0714 |
| 320 | streaming_asr | emilia_zh_0004804632 | 0.5287 | 61 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004804632 | 0.5115 | 62 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0004804632 | 0.4770 | 65 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004880227 | 0.6783 | 64 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0004880227 | 0.6744 | 65 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0004880227 | 0.6357 | 64 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0004880227 | 0.6240 | 66 | I shall always remember the hours I spent with the master of the house of Usher | wer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005094293 | 0.7204 | 55 | 对于托过程来说已经是一件很难做到的事情了 | cer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0005094293 | 0.7109 | 57 | 但是错工程学来说已经是一件很难做到事情了 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005094293 | 0.7014 | 58 | 但是错误工程学来说已经是一件很难做到的事情了 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005094293 | 0.6825 | 58 | 但是从工程学来说已经是一件很难做到的事情了 | cer | 0.1429 |
| 160 | causal_full_asr | emilia_zh_0005313767 | 0.7912 | 47 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005313767 | 0.7766 | 55 | 您有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0005313767 | 0.7875 | 52 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005313767 | 0.7912 | 52 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005420605 | 0.7602 | 41 | 有的高考是考一这只大一变如果太行的 | cer | 0.5294 |
| 320 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 42 | 有的光卡是靠一这只大一变是太行的 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 43 | 有的光卡是靠一这只大一变是不太行的 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 43 | 有的高考是靠一这只大一变是不太行的 | cer | 0.3529 |
| 160 | streaming_asr | emilia_zh_0005670671 | 0.7921 | 75 | 时至今日且在人类社会的各类主流中所败的角色已经无可代替企业的生 | cer | 0.2424 |
| 320 | streaming_asr | emilia_zh_0005670671 | 0.7847 | 78 | 时至今日企业在人类社会的各类组织中所办的角色已经无可代替企业的生 | cer | 0.1212 |
| 640 | streaming_asr | emilia_zh_0005670671 | 0.7847 | 77 | 时至今日企业在人类社会的各类组织中所办的角色已经无可代替企业的生 | cer | 0.1212 |
| 1280 | streaming_asr | emilia_zh_0005670671 | 0.7871 | 75 | 时至今日企业在人类社会的各类组织中所办的角色意见无可代替企业的生 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0005905391 | 0.8095 | 39 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0005905391 | 0.7965 | 42 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0005905391 | 0.7965 | 41 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 1280 | streaming_asr | emilia_zh_0005905391 | 0.8009 | 40 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 160 | causal_full_asr | emilia_zh_0005960185 | 0.7004 | 62 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 320 | causal_full_asr | emilia_zh_0005960185 | 0.7048 | 62 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.1667 |
| 640 | causal_full_asr | emilia_zh_0005960185 | 0.6916 | 63 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005960185 | 0.6872 | 63 | 啊我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.2083 |
| 160 | streaming_asr | emilia_zh_0006119250 | 0.8328 | 49 | 以这种呢我本来啊天蓬说蓬啊蓬马蓬马然后还着点事儿 | cer | 0.5172 |
| 320 | streaming_asr | emilia_zh_0006119250 | 0.7934 | 60 | 有人说呢我本来啊天蓬说我蓬啊蓬马蓬马然后还着点事儿 | cer | 0.4828 |
| 640 | streaming_asr | emilia_zh_0006119250 | 0.7738 | 63 | 一个就是呢我本来啊天平说我朋友我捧马捧马然后还着点事儿 | cer | 0.5172 |
| 1280 | streaming_asr | emilia_zh_0006119250 | 0.7672 | 64 | 意思就是说呢我本来啊天蓬说我蓬吧捧马捧马然后还着点事儿 | cer | 0.3793 |
| 160 | streaming_asr | emilia_zh_0006350396 | 0.6835 | 75 | My aunt found the opposition as a clock with spenlo and drew gongs and we found logins nearby | wer | 0.2941 |
| 320 | streaming_asr | emilia_zh_0006350396 | 0.6519 | 80 | My aunt found me a position as a clock with spenlo and your jogins and we found logins nearby | wer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0006350396 | 0.6329 | 79 | My aunt found the opposition as a clock with spenlo and drew juggins and we found logins nearby | wer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0006350396 | 0.6234 | 78 | My aunt found the opposition as a clock with spenlo and drew juggins and we found logins nearby | wer | 0.2941 |
| 160 | causal_full_asr | emilia_zh_0006404958 | 0.6429 | 74 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 320 | causal_full_asr | emilia_zh_0006404958 | 0.6607 | 68 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 640 | causal_full_asr | emilia_zh_0006404958 | 0.6929 | 63 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 1280 | causal_full_asr | emilia_zh_0006404958 | 0.6750 | 67 | The characters are like monkeys in winter shaking up river wines while drinking water | wer | 0.1333 |
| 160 | streaming_asr | emilia_zh_0006435274 | 0.7080 | 44 | I bank charges interest a brother doesn't charge interest | wer | 0.1111 |
| 320 | streaming_asr | emilia_zh_0006435274 | 0.6726 | 46 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006435274 | 0.6991 | 44 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006435274 | 0.6858 | 47 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006544707 | 0.7447 | 90 | 这两个人呢也是在你说的这个区间有冬季结婚的现象而且这两个人其中有一个 | cer | 0.0588 |
| 320 | streaming_asr | emilia_zh_0006544707 | 0.7342 | 90 | 这两个人呢也是在你说的这个区间有冬季结婚的现象而且这两个人其中有一个 | cer | 0.0588 |
| 640 | streaming_asr | emilia_zh_0006544707 | 0.7263 | 92 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两个呢人其中有一个 | cer | 0.1176 |
| 1280 | streaming_asr | emilia_zh_0006544707 | 0.7158 | 94 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两个呢人其中有一个 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0006874664 | 0.7152 | 121 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查查怎么样但对于这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 320 | streaming_asr | emilia_zh_0006874664 | 0.6814 | 138 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查查怎么样那对于这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 640 | streaming_asr | emilia_zh_0006874664 | 0.6962 | 134 | 那其中它有说到它非常诚恳的就是希望如果说是金水怎么样如果是查杀怎么样那对这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 1280 | streaming_asr | emilia_zh_0006874664 | 0.6920 | 135 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查杀怎么样那对这一段描述里面有觉得比较真实 | cer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0007060532 | 0.7500 | 46 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0007060532 | 0.7308 | 49 | 如果你对某个人偶见是五十真爱就不用嘴说 | cer | 0.3158 |
| 640 | streaming_asr | emilia_zh_0007060532 | 0.7212 | 49 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 1280 | streaming_asr | emilia_zh_0007060532 | 0.7356 | 50 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 160 | causal_full_asr | emilia_zh_0007060544 | 0.7250 | 40 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0007060544 | 0.7063 | 43 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0007060544 | 0.7063 | 42 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0007060544 | 0.7125 | 41 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007312674 | 0.7566 | 109 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他待在一起 | cer | 0.0256 |
| 320 | streaming_asr | emilia_zh_0007312674 | 0.7586 | 105 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007312674 | 0.7566 | 109 | 事实上通力也不需要你亲爱的的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0769 |
| 1280 | streaming_asr | emilia_zh_0007312674 | 0.7627 | 106 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007551053 | 0.7346 | 73 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007551053 | 0.7443 | 74 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 640 | streaming_asr | emilia_zh_0007551053 | 0.7411 | 75 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007551053 | 0.7411 | 74 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007761299 | 0.6764 | 78 | 我觉得这个行业不适合我我觉得我跟这个咱公司的同事处的不开心 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007761299 | 0.6727 | 78 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007761299 | 0.6582 | 80 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007761299 | 0.6545 | 80 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 160 | streaming_asr | EN_B00043_S01954_W000033 | 0.6746 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life is | wer | 0.2500 |
| 320 | streaming_asr | EN_B00043_S01954_W000033 | 0.6509 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life is | wer | 0.2500 |
| 640 | streaming_asr | EN_B00043_S01954_W000033 | 0.6358 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renovable resource And not just in our private life is | wer | 0.2917 |
| 1280 | streaming_asr | EN_B00043_S01954_W000033 | 0.6315 | 114 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life | wer | 0.2083 |
| 160 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6606 | 90 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6483 | 88 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 640 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6177 | 92 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 1280 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6330 | 90 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 160 | streaming_asr | EN_B00089_S03348_W000001 | 0.5220 | 119 | One of the things that like to do in each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 320 | streaming_asr | EN_B00089_S03348_W000001 | 0.5161 | 118 | One of the things that out like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.1034 |
| 640 | streaming_asr | EN_B00089_S03348_W000001 | 0.5073 | 121 | One of the things that I like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 1280 | streaming_asr | EN_B00089_S03348_W000001 | 0.5044 | 119 | One of the things that I like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 160 | streaming_asr | EN_B00048_S07862_W000265 | 0.6995 | 77 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 320 | streaming_asr | EN_B00048_S07862_W000265 | 0.6755 | 79 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 640 | streaming_asr | EN_B00048_S07862_W000265 | 0.6809 | 78 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 1280 | streaming_asr | EN_B00048_S07862_W000265 | 0.6862 | 72 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 160 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7222 | 36 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07870_W000001 | 0.6852 | 34 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7037 | 36 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7160 | 33 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S04429_W000016 | 0.8119 | 49 | The do see these little green leaves And this big flower will I've a second louvre | wer | 0.3750 |
| 320 | streaming_asr | EN_B00058_S04429_W000016 | 0.7921 | 53 | Do you see these little green luses And this big flower where I've a second louvre | wer | 0.3125 |
| 640 | streaming_asr | EN_B00058_S04429_W000016 | 0.7698 | 58 | Do you see these little green leaves And this being flower where I've a second | wer | 0.2500 |
| 1280 | streaming_asr | EN_B00058_S04429_W000016 | 0.7624 | 61 | Do you see these little green leaves And this being flower where I've a second | wer | 0.2500 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
