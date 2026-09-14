NaturalFlow 试听 v2 —— 同样的 8 条,同一推理配置与同一部署栈,只换检查点。

  1_before/               训练前
  2_paper_fewest_pauses/  严格按论文训练   <- 停顿最少
  3_guard_best_quality/   论文 + 起点门槛  <- 翻译质量最好
  0_raw_before/  0_raw_paper/              未经渲染栈,用来听切口本身

部署栈 = 切点吸附 200ms -> 边缘淡化 40ms -> 舒适噪声 -> 1000ms 播放缓冲。

RealSI 777 条配对,相对训练前:

                     静音        首次发声   片段     ASR-BLEU en->zh
  2_paper    -0.0401 +-0.0103    -97 ms   +0.73    +0.10 (不显著)
  3_guard    -0.0236 +-0.0103    -63 ms   +0.62    +1.25 [+0.40,+2.06] 显著

切口质量(边界数 / 切在浊音中 / 截断 / 尾部比):
  before   2300  0.3970  0.4122  0.7086
  paper    2865  0.3616  0.3460  0.5379   <- 切口最多,但三项都最好
  guard    2783  0.3791  0.3501  0.5457

一个此前没意识到的点:训练前的模型一直在“少说”——输出中位 45 字,
参考译文中位 58 字。paper 版把它拉到 62 字,所以听感上不只是停顿变少,
还有原本被截掉的内容回来了。

01_en2zh_en2zh-02-health-s006   source 21.0s
   fragments  before 12  ->  paper 18  guard 21
   before: 所以我认为未来医疗保健系统的一个方向是努力保持人们处于健康的状态而不是我们如何在人们真正遇到严重
   paper : 所以我认为其中一个方向是我们未来医疗保健系统的方向是尝试保持人们处于健康的状态而不是我们如何应对医疗问题当人们出现严重症状时
   guard : 所以我认为其中一个方向是我们未来医疗保健系统的方向是努力保护人们处于健康的状态而不是我们如何应对医疗问题当人们真的出现严重
   ref   : 所以我认为未来医疗保健系统的一大努力方向是，让人们保持健康状态，而不是等人们真正出现严重症状了，再去想怎么处理这些医疗问题。

02_en2zh_en2zh-05-law-s023   source 20.5s
   fragments  before 15  ->  paper 15  guard 17
   before: 所以我的意思是当我代表一个对他们有严重事实的人时我认为这不仅仅是为了这个客户这是为了让下一个真正无辜的人有自由宪法的保护以及他们
   paper : 所以我的意思是当我代表一个有严重事实反对他们的人时我认为这不仅仅是为了这个客户这是为了让下一个真正无辜的人拥有自由并拥有宪法的保
   guard : 所以我的意思是当我代表一个有严重事实反对他们的人时我认为这不仅仅是为了这个客户这是为了让下一个真正无辜的人拥有自由并拥有宪法的保
   ref   : 所以我的意思是，当我为一个被控有严重罪行的人辩护时，我会想，这不仅仅是为了这个客户。这是为了之后真正无辜的人能有宪法的保障和自由

03_en2zh_en2zh-10-art-s014   source 17.4s
   fragments  before 14  ->  paper 13  guard 13
   before: 他告诉玛丽一把剑会刺穿她的灵魂孩子看着毛茸茸的陌生人就像任何婴儿一样试图逃跑他伸手去摸他的母亲就像她冲动地向他伸出手一样
   paper : 他告诉玛丽一把剑会刺穿她的灵魂孩子看着毛茸茸的陌生人就像任何婴儿一样试图逃跑他伸手去摸他的母亲只是一只羊她冲动地向他伸出手
   guard : 他告诉玛丽一把剑会刺穿她的灵魂孩子看着毛茸茸的陌生人就像任何婴儿一样试图逃跑他伸手去摸他的母亲就像她冲动地想要他一样。
   ref   : 他告诉玛丽一把剑会穿透她的灵魂。这个孩子看着这个须发茂密的陌生人，然后像所有其他小孩一样试图躲开他。小孩想要找妈妈，而他妈妈也激

04_en2zh_en2zh-02-health-s003   source 17.2s
   fragments  before 9  ->  paper 10  guard 10
   before: 所以我们现在试图使用不同类型的设备进行数字化同时使用半导体计算机配置文件来配置所谓
   paper : 所以我们试图现在做的事情是尝试使用不同类型的设备进行数字化以及半圆形多焦点配置文件来配置人类所谓的健康状况
   guard : 所以我们试图现在做的是使用不同类型的设备进行数字化以及半双分子谱来分析所谓的人类健康状况
   ref   : 我们目前正在尝试往数字化方向发展，利用不同类型的设备以及分子图谱之类的手段描绘所谓的人体健康状况，

05_zh2en_zh2en-04-fin-s019   source 18.3s
   fragments  before 11  ->  paper 10  guard 7
   before: For example in this company you have released this product but e
   paper : For example in this company you have this um you have experience
   guard : For example in this company you have this product but even if yo
   ref   : For example, a company launches a product, while even if the com

06_zh2en_zh2en-04-fin-s002   source 18.2s
   fragments  before 7  ->  paper 14  guard 15
   before: Well it's just that what does c mean It means that everyone is u
   paper : Well it's just that the matter is irrational meaning it's like b
   guard : Well it's just that the matter hasn't been resolved it's just th
   ref   : The systemic risk mentioned here refers to undiversifiable risk.

07_zh2en_zh2en-04-fin-s008   source 17.5s
   fragments  before 8  ->  paper 12  guard 12
   before: Then for example in 2008 when Bell radar was around all of them 
   paper : Then um um for example in 2008 when Bell radar was around all of
   guard : Then for example in 2008 when Bell radar was around all of it ac
   ref   : For example, during the Bear raid in 2008, everything resulted f

08_zh2en_zh2en-04-fin-s027   source 16.9s
   fragments  before 7  ->  paper 8  guard 11
   before: But actually this series is just one kind of every stage it is e
   paper : But in fact this series is just one kind of every stage it is ea
   guard : But in fact this series is just a kind of every stage it is each
   ref   : But in fact, for the whole process at each stage, its decision m
