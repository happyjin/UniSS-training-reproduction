# rl_epoch2_runtime_v2：四条长音频试听与问题分析

## 结论口径

旧对照是每个 18–30 秒窗口重置全部状态的 bounded-window pseudo-streaming。新结果在完整文件内保留因果 WhisperVQ 前端状态、ASR/MT 已提交文本、TTS ACK 队列和播放时钟；仅 LLM acoustic prompt 使用 24 秒有界 ring 并重算，因此这里不宣称 LLM KV-cache 实时部署。质量门只记录，不阻断后续阶段。

## 总表

| 音频 | 方向 | 旧首音频 | 新首音频 | 旧译音覆盖 | 新译音覆盖 | 旧最大内部静音 | 新最大内部静音 | 新 WRITE | 未发音队列 | TTS失败 | RTF旧→新 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_en_helen_keller_full | eng→cmn | 21420 ms | 74880 ms | 0.145 | 0.288 | 39800 ms | 60000 ms | 38 | 0 | 0 | 5.26→4.43 |
| long_en_shimon_peres_full | eng→cmn | 17280 ms | 18560 ms | 0.098 | 0.169 | 51600 ms | 47400 ms | 32 | 0 | 0 | 3.78→3.47 |
| long_zh_singapore_vietnam_full | cmn→eng | 25720 ms | 10240 ms | 0.140 | 0.492 | 44900 ms | 59100 ms | 63 | 0 | 0 | 4.38→4.35 |
| long_zh_zhangheqiao_full | cmn→eng | 6400 ms | 26240 ms | 0.148 | 0.441 | 45200 ms | 49400 ms | 44 | 0 | 0 | 5.07→4.91 |

## 分音频试听与诊断

### long_en_helen_keller_full

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/student_v2_long_demo_v1/wav16k/english_helen_keller_part03.wav`
- 连续翻译音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_helen_keller_full/long_en_helen_keller_full/stereo_left_source_right_translation.wav`
- 完整 ASR：This is a liberalExpress myself, grewdaining tears in physical exhaustionim unlikely that anyone would come to suchn educated, but you'll serve him with aAunt Tom, who hadThe conductor, too, was kind, often when he went his rounds, a close了他的 coat-tails while heShapless thing, this improvised all with no nose, mouth, ears or eyes. Nothing that even the imagination of a child could convert into a fake. Curiously enough, the absence of eyes struck me more than all the other defects put together. I pointed this out to everybody with provoking persistency, but no one seemed equalI'd idea, however, shocking to my mind. And the problem was solved. I tumbled off the seat and searched under it until I found my aunt's cape, which was streamed with large beads. I pulled twoBut immediately I lost all interest in the doll. During the trip I did not have one fit of temper. There were so many things to keep my mind in,WeJin did Dr. Bell to some in the house. This is wonderful achievement in this direction. He helped me on his knee while I examined his watchLove Dr. Bill advised my father to write to Mr. Anagnes, director of the Perkins Institution in Boston, the same of Dr. House great labors for the blind and ask him if he had a teacher competent in my educationhad been foundEnd of chapter, read primary youth at England
- 完整增量翻译：这是一个自由派的表达，我因身体疲惫而流下眼泪，几乎没有人会来接受这样的教育，但你会为他请来一位有指挥的阿姨，她也是一位善良的阿姨，经常在他外出时，他会把外套的下摆拉得很紧，而他则无能为力，这让他完全失去了知觉、嘴巴、耳朵或眼睛。即使是一个孩子的想象也无法将我比作其他所有缺陷。我指出这一点，这让我更加困惑，但似乎没有人能像我这样让我感到震惊。然而，问题在于，我从座位上摔下来，搜查了里面，直到找到了我姑姑的斗篷，上面布满了大珠子。我立刻失去了对这个娃娃的兴趣。在旅行中，我并没有脾气暴躁。有太多的事情让我无法集中注意力，而贝尔博士则建议我父亲给我写信，询问他是否对我的教育有足够了解，我非常惊讶。在这一章结束时，我读了《波士顿的初等儿童》。我注意到他身上没有一个像我这样的脾气暴躁的人。我检查了他的手表，发现里面有很多珠子。我立刻感到非常惊讶。但后来我意识到，我自己的教育能力并不强，我无法想象他会对我的父亲产生如此大的影响。
- 首次发声 74880 ms；共 38 次发声；最大 WRITE 间隔 65920 ms。
- 24 秒 acoustic ring rollover 14 次，WhisperVQ encoder position reset 11 次；人工窗口误 final 次数 0。
- early-END 拒绝 2 次；semantic continuation 0 次；TTS 失败 0 次；最终未发音队列 0 条。
- 有发声事件 23 个决策点；连续音频健康=True，译音/源音时长比=0.288。

### long_en_shimon_peres_full

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/student_v2_long_demo_v1/wav16k/english_shimon_peres_interview.wav`
- 连续翻译音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_en_shimon_peres_full/long_en_shimon_peres_full/stereo_left_source_right_translation.wav`
- 完整 ASR：What do you hope to achieve with your presidency You would like in the long term to make the help of people to become the people that leave some roomAnd the our talent put themHe saidBericSmall peopleI quite convict the government. If you're already seen as both of those things, I'm verySupplied my life was quite comfortable. IYou've been with us better.Such a school. Why was iLike it is, they are so much so much opposite your head right away when I ask thatteer children about the things that youHelp them. I think it's always done very slowly. Why should they notThey don't ask, I don't bother them. Imagine they know because they're the right aboutUniqueBut, you know, those things can go, go, go, go, go, go, a bunch of them. One of theThat they're trying to change the narrative in the American media. You're very rarely ever hear about Israel outside of the Palestinian, Lebanese conflicts. So in their trying to say, listen, we have so much more here in Israel. We need to solve this, but we also have other things on offer. No, it's time. The first is to waitsoNo
- 完整增量翻译：你希望在总统任期内实现什么？你希望在长期内帮助人们成为那些留有空间的人，而我们的才能让他们成为他们。他说，贝里克，小人物，我相当谴责政府。如果你已经被视为这两者之一，我非常有资格。我的生活相当舒适。你和我们在一起更好。所以这所学校。我为什么喜欢它，是因为当我问那些孩子关于你帮助他们的事情时，他们非常反对。我认为这所学校总是进展得很慢。他们为什么不喜欢，他们不问，我不打扰他们。想象一下，他们知道你是对的，是的，是的，是的，是的，很多。他们试图改变美国媒体的叙事。你很少会听到关于以色列的更多内容。我们还需要解决这个问题，但也有其他事情可以提供。不，是时候了。第一件事是等待，所以不
- 首次发声 18560 ms；共 32 次发声；最大 WRITE 间隔 48640 ms。
- 24 秒 acoustic ring rollover 17 次，WhisperVQ encoder position reset 13 次；人工窗口误 final 次数 0。
- early-END 拒绝 1 次；semantic continuation 0 次；TTS 失败 0 次；最终未发音队列 0 条。
- 有发声事件 26 个决策点；连续音频健康=True，译音/源音时长比=0.169。

### long_zh_singapore_vietnam_full

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/student_v2_long_demo_v1/wav16k/chinese_singapore_vietnam_relations.wav`
- 连续翻译音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_singapore_vietnam_full/long_zh_singapore_vietnam_full/stereo_left_source_right_translation.wav`
- 完整 ASR：新加坡越南关系出自维基百科自由百科全书网址是extipicalandalmostlastsetaweekperioddotoff侧录音根据二零一五年十一月十四日版本压迫和越南，包括现在新压迫共和国与越南社会主义共和国之间达成双边关系。新加坡与十九世纪起便与越南有双貌往来越一同是东南亚国家联盟的成员。历史新加坡与越南的双网往来是十九世纪开始俯视派出军队和装备对付支持越南独立运动人士一九五新加坡从马来西亚独立时，越南共和国以东为首，普京、阿普尔、铁角的亚洲、瓦加尔支伊十月，越南总理犯文童访问西夏伯成为越南同意后，受魏访问西夏伯的越南政府首脑李小七八年十二月越南派兵占领也任且不在人民共和国，两国关系是一九九零年越南从柬埔寨撤军以及一九九五年合作框架联合声明二零一三年新加坡总理离险农法院两国间谍站点伙伴关系既满怀欢喜仍然打四十亿美元，新加坡主要像越南出口口货币主要以原油为主，一九九三年五月五日，越南停下波河，所有委员会成立二零零五年十二元六日相同资讯和集体讯科技教育及及时省海洋省和岸岸省设有越南西亚波工业园区。二零一三年九月十三日，新加坡贸易和工业部部长林勋强与越南计划及化关系这也就叫二年级有一万六千名运动员，官员承接受新加坡合作计划的培训，还敢一定要表现环境轻初中教育被在英语、企业或玩活动有良好表现的越南学生发放动文奖学金
- 完整增量翻译：S Singapore's relations with Vietnam come from the Wikipedia Free Encyclopedia, the website's URL is "extipical Dandal Most Last Set" by Weikyip.com, based on the 14th version of "压迫与越南" on November 14, 2015, including the new压迫共和国 and the current bilateral relationship between the new and the socialist republic of Vietnam. Since the 19th century, Singapore has had a dual relationship with Vietnam, becoming a member of the Southeast Asian League. Since the 19th century, the dual network of Singapore and Vietnam has been looking down on and deploying troops to deal with the independent movements of the Vietnamese independence movement in the United States. Since the independence of Singapore began in Malaysia, the Vietnamese Republic has been leading the country in the east, with Putin and Apple being the vassal states of the United States. Since the 19th century, the United States has had a dual relationship with Vietnam, becoming the leader of the Vietnamese League. Since the 19th century, the United States has been looking down on and deploying troops to deal with the independent movements of the Vietnamese independence movement in South Korea. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the 19th century, the United States has been leading the country eastward, and the Soviet Union has been the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States. Since the independence of the United States began in Malaysia, the United States has been leading the country eastward, with Putin and Apple being the vassal states of the United States.
- 首次发声 10240 ms；共 63 次发声；最大 WRITE 间隔 72320 ms。
- 24 秒 acoustic ring rollover 16 次，WhisperVQ encoder position reset 13 次；人工窗口误 final 次数 0。
- early-END 拒绝 0 次；semantic continuation 0 次；TTS 失败 0 次；最终未发音队列 0 条。
- 有发声事件 23 个决策点；连续音频健康=True，译音/源音时长比=0.492。

### long_zh_zhangheqiao_full

- 源音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/student_v2_long_demo_v1/wav16k/chinese_zhangheqiao_township.wav`
- 连续翻译音频：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_continuous.wav`
- 全局时间轴：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/translation_global_timeline.wav`
- 左源右译立体声：`/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1/rl_epoch2_runtime_v2/parts/long_zh_zhangheqiao_full/long_zh_zhangheqiao_full/stereo_left_source_right_translation.wav`
- 完整 ASR：张河桥乡来自维吉百科自由的百科全书张河桥二零零一年时，便入街山乡，即经街山镇先清淡时接山拱舍建立张和小祥驻地，张和小祥原无居民，一九七五年，接山拱舍机关由接山前往此处建立藏秋，徐坦细节大相相近的那矿产资源有白云岩、石灰岩等第三级行政区划，一九五八年底，张和乔向下四个管理区，共二十八个行政村和二十八个自然村，张和乔向下下的行政村为下泻五酸后，战死酸尚随桑丘二村桑丘三村新庄村、长庄一村、长庄二村、长庄三村、东村村、南村村第四街荆棘，正好桥桥农业总产值一千六百三十九万元，粮食总产量为两万一千二百六十吨，但十二百其实七公斤工业方面，一九九三年张和乔香工业产值四千一百七十五十万元。当年张和乔香财政总收入为一百一十六点零九万且人口与距二零零零年中国第五次人口普查家庭九千零六十九户，把家庭户总人口三万四千二百七十万人平均每户三点七八人十四占总人口的百分之九点九八，本地居住的人口中有三万四千零四人拥有本地户籍，第六节基础设施二十三度一九九三年时，正好相将近的有幼儿园二十八所小学是七本应推迟二零一九年七月十八日
- 完整增量翻译：Zhang Heqiao Township comes from the free encyclopedia of Wikipedia. Zhang Heqiao was born in 2001 and entered the town of Xie Mountain. When the town was first established, it was light-hearted to take the mountain arches to build a small and small town. Zhang He and Xiaoxiang were originally homeless, and in 1975, they took the mountain arches to establish a hidden autumn camp, with the details of the Xituan and the Baiyun Rock being very similar to the third-level administrative area, which was established at the end of 1958. At the end of 1958, Zhang He and Xiaoxiang managed to manage four administrative villages and twenty-eight natural villages, including the third-level administrative area, which was then established. After the fall of the Sichuan and Xiaoxiang dynasties, Zhang He and Xiaoxiang moved to the fourth-level administrative village, which was the Xituan and the Baiyun Rock. The capital of the village was located in the southern part of the city, Nanchang, which was the capital of the Sichuan and Xiaoxiang dynasties. After the fall of 1958, Zhang He and Xiaoxiang moved to four administrative villages, including the second-level administrative area, which was the Xituan and the Baiyun Rock. The capital of the village was located in the southern part of the city, Nanchang, which was the capital of the Sichuan and Xiaoxiang dynasties. After the fall of 1958, Zhang He and Xiaoxiang moved to the fourth-level administrative village, which was the capital of Nanchang and Xiaoxiang dynasties. The second-level administrative village was Nanchang, which was the capital of the Sichuan and Xiaoxiang dynasties. The capital of Nanchang and Xiaoxiang was located in the southern part of the city, Nanchang, which is the capital of the Sichuan and Xiaoxiang dynasties. The capital of Nanchang and Xiaoxiang was located in the southern part of the city, Nanchang, which is the capital of Nanchang and Xiaoxiang dynasties. The second-level administrative village, Nanchang and Xiaoxiang
- 首次发声 26240 ms；共 44 次发声；最大 WRITE 间隔 69120 ms。
- 24 秒 acoustic ring rollover 14 次，WhisperVQ encoder position reset 11 次；人工窗口误 final 次数 0。
- early-END 拒绝 0 次；semantic continuation 0 次；TTS 失败 0 次；最终未发音队列 0 条。
- 有发声事件 20 个决策点；连续音频健康=True，译音/源音时长比=0.441。

## 本阶段能回答与不能回答的问题

可以直接判断：窗口状态是否重置、文本是否在 TTS 失败后丢失、320 semantic token 截断是否继续、是否在真实文件结束前发声、WRITE 间隔和全局时间轴空白是否改善。

不能仅凭这四条无人工参考的外部长音频报告 BLEU/WER，也不能把固定 speaker token 等同于客观音色一致性分数。后续 A/B/C/D 归因会加入离线 teacher/reference 路由；最终 RL 对照仍使用同一 runtime v2，避免把 runtime 修复误算成模型训练收益。
