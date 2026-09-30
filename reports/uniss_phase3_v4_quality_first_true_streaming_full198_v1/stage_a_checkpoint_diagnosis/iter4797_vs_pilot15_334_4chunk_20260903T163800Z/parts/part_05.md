# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 164
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9568**
- Weighted CTC blank ratio: **0.6943**
- Weighted streaming WER/CER: **0.2112**
- Weighted causal-full WER/CER: **0.1787**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | streaming_asr | CommonVoice_EN_0000126367 | 0.7179 | 63 | He had always said grids of retired for exicism and charity to the poor | wer | 0.3846 |
| 320 | streaming_asr | CommonVoice_EN_0000126367 | 0.6750 | 65 | He had always a grudge of preterious exicism and charity to the poor | wer | 0.3077 |
| 640 | streaming_asr | CommonVoice_EN_0000126367 | 0.6714 | 67 | He had always a grudge of preterious exicism and charity to the poor | wer | 0.3077 |
| 1280 | streaming_asr | CommonVoice_EN_0000126367 | 0.6714 | 68 | He had always to grew up before exicism and charity to the poor | wer | 0.3846 |
| 160 | causal_full_asr | CommonVoice_EN_0000262462 | 0.6945 | 54 | It is across the Columbia River from Wensham in Washington | wer | 0.2000 |
| 320 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7055 | 52 | It is across the Columbia River from Wensham in Washington | wer | 0.2000 |
| 640 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7127 | 54 | It is across the Columbia River from Watsam in Washington | wer | 0.2000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7091 | 57 | It is across the Columbia River from Watsam in Washington | wer | 0.2000 |
| 160 | streaming_asr | CommonVoice_EN_0000286138 | 0.7316 | 57 | The boy reminded the old man that he had said something about hidden dresses | wer | 0.0714 |
| 320 | streaming_asr | CommonVoice_EN_0000286138 | 0.6901 | 65 | The boy reminded the old man that he had said something about hidden dresses | wer | 0.0714 |
| 640 | streaming_asr | CommonVoice_EN_0000286138 | 0.6837 | 66 | The boy reminded the old man that he had said something about hidden treasure | wer | 0.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000286138 | 0.6581 | 70 | The boy reminded the old man that he had said something about hidden treasure | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000433992 | 0.7159 | 56 | Nobody wants to the discuss how you'll all end up trap the at kind lunch box | wer | 0.7143 |
| 320 | streaming_asr | CommonVoice_EN_0000433992 | 0.6937 | 56 | Nobody wants to the discuss how you'll all end up trap and I can't learn box | wer | 0.7143 |
| 640 | streaming_asr | CommonVoice_EN_0000433992 | 0.6605 | 63 | Nobody wants to the discuss how you all end up trap and a kind lunch box | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000433992 | 0.6605 | 60 | Nobody wanted to discuss how you all ended up trap and a charm lunch box | wer | 0.4286 |
| 160 | streaming_asr | CommonVoice_EN_0000593898 | 0.6964 | 63 | In recent years Saint George's has spent the hundredth Medical School roots | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000593898 | 0.6893 | 59 | In recent years Saint George's has expended by its medical school roots | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000593898 | 0.6714 | 61 | In recent years St George's has a span beyond its medical school roots | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000593898 | 0.6750 | 63 | In recently years St George's has spent beyond its medical school roots | wer | 0.3333 |
| 160 | causal_full_asr | DailyTalk_0000010084 | 0.7188 | 27 | Good morning sir What can I do for you | wer | 0.1111 |
| 320 | causal_full_asr | DailyTalk_0000010084 | 0.7109 | 28 | Good morning sir What can I do for you | wer | 0.1111 |
| 640 | causal_full_asr | DailyTalk_0000010084 | 0.7109 | 28 | Good morning sir What can I do for you | wer | 0.1111 |
| 1280 | causal_full_asr | DailyTalk_0000010084 | 0.6953 | 28 | The morning sir what can I do for you | wer | 0.2222 |
| 160 | streaming_asr | LibriSpeech_0000068284 | 0.7271 | 102 | Is not a world of facts but only a the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0714 |
| 320 | streaming_asr | LibriSpeech_0000068284 | 0.6919 | 107 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 640 | streaming_asr | LibriSpeech_0000068284 | 0.7007 | 108 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 1280 | streaming_asr | LibriSpeech_0000068284 | 0.7007 | 108 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 160 | streaming_asr | LibriSpeech_0000158773 | 0.5879 | 218 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my my meek little income and I began to win money on security relying on my threat my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2449 |
| 320 | streaming_asr | LibriSpeech_0000158773 | 0.5597 | 221 | All men in joy the blessings of the liberty have lived that by utilizing my capital at my might make a little income in IBM began to win money on security relying on my threat my judgment and my knowledge of the world I choose this business in preferance to all others | wer | 0.2653 |
| 640 | streaming_asr | LibriSpeech_0000158773 | 0.5490 | 228 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my my make little income and I'd be going to win money on security relying on my thrift my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2857 |
| 1280 | streaming_asr | LibriSpeech_0000158773 | 0.5477 | 228 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my might make a little income and I'd be going to win money on security relying on my thrift my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2449 |
| 160 | causal_full_asr | VCTK_0000006143 | 0.7089 | 34 | Being captain of this club is fantastic | wer | 0.0000 |
| 320 | causal_full_asr | VCTK_0000006143 | 0.7025 | 34 | Being captain of this club is fantastic | wer | 0.0000 |
| 640 | causal_full_asr | VCTK_0000006143 | 0.6835 | 35 | Being captain of this club is fantastic | wer | 0.0000 |
| 1280 | causal_full_asr | VCTK_0000006143 | 0.6772 | 35 | Being captain of this club is fantastic | wer | 0.0000 |
| 160 | streaming_asr | VCTK_0000029362 | 0.8500 | 18 | The body is exhausted | wer | 0.2500 |
| 320 | streaming_asr | VCTK_0000029362 | 0.8562 | 17 | The body is exhausted | wer | 0.2500 |
| 640 | streaming_asr | VCTK_0000029362 | 0.8500 | 19 | My body is exhausted | wer | 0.0000 |
| 1280 | streaming_asr | VCTK_0000029362 | 0.8625 | 20 | My body is exhausted | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004111317 | 0.8029 | 143 | 我有个主意我们为什么不挖坑找水呢啊哦是的我相信如果我们挖的足够深我们可以找到水的东西那我们选择一个地点然后开始挖挖 | cer | 0.1071 |
| 320 | streaming_asr | emilia_zh_0004111317 | 0.7993 | 150 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆那我们选择一个地点然后开始挖挖吧 | cer | 0.0714 |
| 640 | streaming_asr | emilia_zh_0004111317 | 0.7896 | 151 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆认为我们选择一个地点然后开始挖挖吧 | cer | 0.0893 |
| 1280 | streaming_asr | emilia_zh_0004111317 | 0.7908 | 157 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆认为我们选择一个地点然后开始挖挖吧 | cer | 0.0893 |
| 160 | streaming_asr | emilia_zh_0004270182 | 0.7827 | 67 | 知道的从前埋藏的水晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2963 |
| 320 | streaming_asr | emilia_zh_0004270182 | 0.7649 | 73 | 找到的从前埋藏的水晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004270182 | 0.7619 | 72 | 找到的从前埋藏的雨晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004270182 | 0.7589 | 72 | 找到的从前埋藏的雨晶很久很久的骨头赶走工具螃蟹的老鼠们 | cer | 0.1852 |
| 160 | streaming_asr | emilia_zh_0004621436 | 0.6901 | 45 | But mother wrote and asked me if they could possibly come as paying guess | wer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0004621436 | 0.6244 | 51 | They mother wrote and asked me if they could possibly come as paying guess | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004621436 | 0.6291 | 50 | They mother wrote and asked me they could possibly come as paying guests | wer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0004621436 | 0.6103 | 56 | There mother wrote and asked me if they could possibly come as paying guests | wer | 0.0714 |
| 160 | causal_full_asr | emilia_zh_0004621493 | 0.6905 | 44 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 320 | causal_full_asr | emilia_zh_0004621493 | 0.7190 | 41 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0004621493 | 0.6381 | 49 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 1280 | causal_full_asr | emilia_zh_0004621493 | 0.6286 | 49 | Here's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 160 | streaming_asr | emilia_zh_0004754634 | 0.6667 | 40 | Perhaps there is second joke suggested the jack door | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004754634 | 0.5882 | 44 | Perhaps may have a second joke suggested the jack or | wer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0004754634 | 0.6144 | 43 | Perhaps there are a second joke suggested the jack door | wer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004754634 | 0.5948 | 46 | Perhaps there is second joke suggested the jack or | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004841011 | 0.6512 | 55 | I all those users who churning and resuracting have the lowfriend camps | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0004841011 | 0.5907 | 57 | And all those users who are churning and resuracting have the lowfriend counts | wer | 0.3846 |
| 640 | streaming_asr | emilia_zh_0004841011 | 0.5907 | 56 | And all those users who are churning and resuracting have been loafromed camps | wer | 0.4615 |
| 1280 | streaming_asr | emilia_zh_0004841011 | 0.5674 | 61 | And all those users who are churning and resuracting have been loafed from camps | wer | 0.5385 |
| 160 | causal_full_asr | emilia_zh_0004841554 | 0.6654 | 64 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 320 | causal_full_asr | emilia_zh_0004841554 | 0.6342 | 66 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 640 | causal_full_asr | emilia_zh_0004841554 | 0.6265 | 67 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 1280 | causal_full_asr | emilia_zh_0004841554 | 0.6265 | 66 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 160 | streaming_asr | emilia_zh_0004927721 | 0.7669 | 51 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 320 | streaming_asr | emilia_zh_0004927721 | 0.7585 | 52 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0004927721 | 0.7542 | 54 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0004927721 | 0.7585 | 52 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0005181378 | 0.6932 | 52 | 不是客气你记得我的几本书请你带回去 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0005181378 | 0.6705 | 53 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0005181378 | 0.6761 | 52 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0005181378 | 0.6591 | 54 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0005507035 | 0.7273 | 50 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005507035 | 0.7059 | 54 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005507035 | 0.6898 | 54 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005507035 | 0.6845 | 55 | 那么日本政府还会收回这个决定了大的很关心 | cer | 0.1429 |
| 160 | causal_full_asr | emilia_zh_0005600573 | 0.8401 | 39 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 320 | causal_full_asr | emilia_zh_0005600573 | 0.8216 | 44 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005600573 | 0.7918 | 47 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 1280 | causal_full_asr | emilia_zh_0005600573 | 0.8104 | 44 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 160 | streaming_asr | emilia_zh_0005749601 | 0.7390 | 94 | 是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了花第一点就是那功能不足 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0005749601 | 0.7209 | 97 | 就是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是认知功能不足 | cer | 0.1591 |
| 640 | streaming_asr | emilia_zh_0005749601 | 0.7158 | 97 | 这是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是那功能不足 | cer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0005749601 | 0.7132 | 101 | 这是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是那功能不足 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0005999475 | 0.7179 | 111 | 哦在生活上你没有感觉到特别的贵因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2353 |
| 320 | streaming_asr | emilia_zh_0005999475 | 0.7110 | 117 | 然后在生活上你没有感觉到特别的贵因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2745 |
| 640 | streaming_asr | emilia_zh_0005999475 | 0.6927 | 125 | 然后在生活上你没有感觉到特别的鬼因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0005999475 | 0.6904 | 127 | 然后在生活上因为没有感觉到特别的鬼因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.3137 |
| 160 | causal_full_asr | emilia_zh_0006041799 | 0.6960 | 64 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 320 | causal_full_asr | emilia_zh_0006041799 | 0.6740 | 67 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 640 | causal_full_asr | emilia_zh_0006041799 | 0.6564 | 66 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 1280 | causal_full_asr | emilia_zh_0006041799 | 0.6608 | 66 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 160 | streaming_asr | emilia_zh_0006212201 | 0.7005 | 56 | 那没有办法一直坚持别说的举动所以买通了一个仆人 | cer | 0.1739 |
| 320 | streaming_asr | emilia_zh_0006212201 | 0.6667 | 61 | 他没有办法抑制监视别墅的举动所以买通了一个仆人 | cer | 0.0870 |
| 640 | streaming_asr | emilia_zh_0006212201 | 0.6618 | 59 | 他没有办法一直坚持别说的举动所以买通了一个仆人 | cer | 0.1304 |
| 1280 | streaming_asr | emilia_zh_0006212201 | 0.6522 | 61 | 他没有办法抑制监视别墅的举动所以买通了一个仆人 | cer | 0.0870 |
| 160 | streaming_asr | emilia_zh_0006366492 | 0.7156 | 103 | He did not want to sit in God his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0385 |
| 320 | streaming_asr | emilia_zh_0006366492 | 0.7062 | 102 | He didn't not want to sit and God his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0385 |
| 640 | streaming_asr | emilia_zh_0006366492 | 0.6949 | 109 | He didn't not want to sit and guard his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0769 |
| 1280 | streaming_asr | emilia_zh_0006366492 | 0.7156 | 105 | He didn't not want to sit and guard his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0006446583 | 0.6676 | 92 | Now we can use emori emaging to see what is actually happening inside the joint When someone cracks nuckles | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006446583 | 0.6511 | 93 | Now we can use emori emaging to see what is actually happening inside the joint when someone cracks nuckles | wer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0006446583 | 0.6456 | 93 | Now we can use emori emaging to see what is actually happening inside the joint when someone cracks the nuckles | wer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0006446583 | 0.6236 | 96 | Now we can use MRI imaging to see what is actually happening inside the joint when someone cracks the nuckles | wer | 0.1000 |
| 160 | causal_full_asr | emilia_zh_0006502797 | 0.6904 | 129 | 我觉得这个委托人可不冷我觉得那你说古代的时候和他的手套那古人都已经挖到比如说这个明代开始就把这核桃把那个发明石头 | cer | 0.3443 |
| 320 | causal_full_asr | emilia_zh_0006502797 | 0.6637 | 135 | 我觉得这个胃腿不认可我觉得那时候古代的时候喝黑的舌头那古人都已经玩了比如说这个明代开始就把这黑的舌头就变成外面石头了 | cer | 0.4262 |
| 640 | causal_full_asr | emilia_zh_0006502797 | 0.6459 | 140 | 我觉得这个胃特别不认可我觉得那时候古代的时候喝他的舌头那古人都已经挖了比如说这个明代开始就把这核桃就变成外面石头了 | cer | 0.3607 |
| 1280 | causal_full_asr | emilia_zh_0006502797 | 0.6437 | 138 | 我觉得这个胃特别不认可我觉得那时候古代的时候喝喝的时候他那古人都已经玩了比如说这个明代开始就把这核桃就发明出套了 | cer | 0.2951 |
| 160 | streaming_asr | emilia_zh_0006610357 | 0.6667 | 114 | 今天我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 320 | streaming_asr | emilia_zh_0006610357 | 0.6800 | 112 | 之前我觉得是非常确实关键性是非常之前但是似乎恶魔的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 640 | streaming_asr | emilia_zh_0006610357 | 0.6747 | 110 | 最近我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 1280 | streaming_asr | emilia_zh_0006610357 | 0.6747 | 111 | 这件我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1395 |
| 160 | streaming_asr | emilia_zh_0006883085 | 0.7594 | 48 | 所以与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 320 | streaming_asr | emilia_zh_0006883085 | 0.7453 | 51 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0006883085 | 0.7358 | 51 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0006883085 | 0.7311 | 55 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0007121845 | 0.7923 | 108 | 那么我们会也许只是依赖于一个技术比较方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0488 |
| 320 | streaming_asr | emilia_zh_0007121845 | 0.7871 | 110 | 那么我们会也许只是依赖于一个技术比一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 640 | streaming_asr | emilia_zh_0007121845 | 0.7784 | 112 | 那么我们会也许只是依赖于一个技术一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007121845 | 0.7766 | 113 | 那么我们会也许只是依赖于一个技术比一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 160 | streaming_asr | emilia_zh_0007399682 | 0.7788 | 91 | 有些上市甚至一个更加严厉的态度仅我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1471 |
| 320 | streaming_asr | emilia_zh_0007399682 | 0.7677 | 90 | 有些丧失甚至一个更加严厉的态度仅是我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1765 |
| 640 | streaming_asr | emilia_zh_0007399682 | 0.7699 | 89 | 有些丧失甚至以更加严厉的态度警示我们要认清生命的脆弱告诉我们每一个人 | cer | 0.0588 |
| 1280 | streaming_asr | emilia_zh_0007399682 | 0.7699 | 91 | 有些丧失甚至以更加严厉的态度竟是我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007635379 | 0.7071 | 53 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0007635379 | 0.7222 | 46 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 640 | streaming_asr | emilia_zh_0007635379 | 0.6717 | 54 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 1280 | streaming_asr | emilia_zh_0007635379 | 0.6818 | 53 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 160 | causal_full_asr | emilia_zh_0007761003 | 0.7770 | 52 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 320 | causal_full_asr | emilia_zh_0007761003 | 0.7732 | 49 | 如果你向你的团队成员分析走得更远那你就需要 | cer | 0.1818 |
| 640 | causal_full_asr | emilia_zh_0007761003 | 0.7621 | 55 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 1280 | causal_full_asr | emilia_zh_0007761003 | 0.7621 | 55 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 160 | streaming_asr | emilia_zh_0007790200 | 0.7813 | 111 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 320 | streaming_asr | emilia_zh_0007790200 | 0.7690 | 121 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 640 | streaming_asr | emilia_zh_0007790200 | 0.7672 | 121 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 1280 | streaming_asr | emilia_zh_0007790200 | 0.7584 | 123 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 160 | streaming_asr | EN_B00013_S05834_W000745 | 0.5632 | 105 | Even if the night could still stand through a strong resilience the broken bones and is chested with making him lose any ability to fight | wer | 0.2800 |
| 320 | streaming_asr | EN_B00013_S05834_W000745 | 0.5431 | 110 | Even if the night could still stand through a strong resilience the broken bones and a chest would make him lose any ability to fight | wer | 0.1600 |
| 640 | streaming_asr | EN_B00013_S05834_W000745 | 0.5517 | 107 | Even if the night could still stand through a strong resilience the broken bones and a chest would make him lose any ability to fight | wer | 0.1600 |
| 1280 | streaming_asr | EN_B00013_S05834_W000745 | 0.5374 | 109 | Even if the night could still stand through a strong resilience the broken bones and his chest would make him lose any ability to fight | wer | 0.1200 |
| 160 | causal_full_asr | EN_B00013_S06748_W000028 | 0.7294 | 42 | I just skateboards said his dad it's two slippery so that I'm | wer | 0.6364 |
| 320 | causal_full_asr | EN_B00013_S06748_W000028 | 0.6835 | 45 | Ride your skateboard said his dad it's two slippers and Adam | wer | 0.2727 |
| 640 | causal_full_asr | EN_B00013_S06748_W000028 | 0.6789 | 46 | I just skateboard said his dad it's two slippers and Adam | wer | 0.4545 |
| 1280 | causal_full_asr | EN_B00013_S06748_W000028 | 0.7018 | 45 | Ride your skateboard said his dad it's two slippers and Adam | wer | 0.2727 |
| 160 | streaming_asr | EN_B00048_S02289_W000002 | 0.7821 | 38 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S02289_W000002 | 0.7786 | 46 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 640 | streaming_asr | EN_B00048_S02289_W000002 | 0.7750 | 44 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 1280 | streaming_asr | EN_B00048_S02289_W000002 | 0.7714 | 44 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 160 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6609 | 144 | Every two months my coworkers and I would come together to discuss the new semester schedule Uhm meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 320 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6401 | 141 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6263 | 146 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6280 | 145 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S09662_W000003 | 0.6054 | 140 | Okay very good So what don't we need listen to the dialogue for the first time that's listen how President Isaac Holmes says goodbye And then we'll come back and look at the words | wer | 0.1562 |
| 320 | streaming_asr | EN_B00048_S09662_W000003 | 0.5888 | 144 | Okay very good So what don't we need listen to the dialogue for the first time that's listen how President Isaac Holmes has goodbye And then we'll come back and look at the words | wer | 0.1875 |
| 640 | streaming_asr | EN_B00048_S09662_W000003 | 0.5971 | 143 | Okay very good So why don't we let listen to the dialogue for the first time let's listen how President Isaac Holmes has goodbye And then we'll come back and look at the words | wer | 0.1250 |
| 1280 | streaming_asr | EN_B00048_S09662_W000003 | 0.5971 | 144 | Okay very good So why don't we let listen to the dialogue for the first time let's listen how President Isaac Holmes says goodbye And then we'll come back and look at the words | wer | 0.0938 |
| 160 | streaming_asr | EN_B00058_S06165_W000019 | 0.6582 | 196 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 320 | streaming_asr | EN_B00058_S06165_W000019 | 0.6509 | 197 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 640 | streaming_asr | EN_B00058_S06165_W000019 | 0.6350 | 201 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 1280 | streaming_asr | EN_B00058_S06165_W000019 | 0.6241 | 205 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
