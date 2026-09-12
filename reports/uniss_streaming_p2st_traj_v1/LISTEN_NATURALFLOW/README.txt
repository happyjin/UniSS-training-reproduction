NaturalFlow 试听:同样的 8 条,同一推理配置,只换检查点。

  1_base_deployed/        训练前 + 完整部署栈
  2_naturalflow_deployed/ DPO 400 步 + 完整部署栈   <- 两两对比听这两个
  0_raw_base/             训练前,未经渲染栈
  0_raw_naturalflow/      DPO 400 步,未经渲染栈

部署栈 = 切点吸附 200ms -> 边缘淡化 40ms -> 舒适噪声 -> 1000ms 播放缓冲。
两组都上了同一套栈,所以差别只来自模型。

RealSI 777 条配对上的整体差异:
  静音占比  -0.0123 (95% CI [0.0033, 0.0213])
  首次发声  -1 ms
  片段数    +0.23 (95% CI +-0.09)
  字数      +0.8
  最长输出  1859 -> 256 字

01_en2zh_en2zh-02-health-s006
   source 21.0s  fragments 12 -> 12
   base : 所以我认为未来医疗保健系统的一个方向是努力保持人们处于健康的状态而不是我们如何在人们真正遇到严重
   nflow: 所以我认为未来健康生态系统的一个方向是努力保持人们处于健康的状态而不是我们如何在人们真正遇到严重

02_en2zh_en2zh-05-law-s023
   source 20.5s  fragments 15 -> 16
   base : 所以我的意思是当我代表一个对他们有严重事实的人时我认为这不仅仅是为了这个客户这是为了让下一个真正无辜的人有自由宪法的保护以及他们的权利能够被
   nflow: 所以我的意思是当我代表一个对他们有严重事实的人时我认为这不仅仅是为了这个客户这是为了让真正无辜的下一个人有自由和宪法的保护以及他们的权利能够

03_en2zh_en2zh-10-art-s014
   source 17.4s  fragments 14 -> 14
   base : 他告诉玛丽一把剑会刺穿她的灵魂孩子看着毛茸茸的陌生人就像任何婴儿一样试图逃跑他伸手去摸他的母亲就像她冲动地向他伸出手一样
   nflow: 他告诉玛丽一把剑会刺穿她的灵魂孩子看着毛茸茸的陌生人就像任何婴儿一样试图逃跑他伸手去拿他的母亲就像她冲动地向他伸出手一样

04_en2zh_en2zh-02-health-s003
   source 17.2s  fragments 9 -> 8
   base : 所以我们现在试图使用不同类型的设备进行数字化同时使用半导体计算机配置文件来配置所谓
   nflow: 所以我们现在试图使用不同类型的设备进行数字化同时使用半导体来配置所谓

05_zh2en_zh2en-04-fin-s019
   source 18.3s  fragments 11 -> 8
   base : For example in this company you have released this product but even if
   nflow: For example if this company releases this product but even if your com

06_zh2en_zh2en-04-fin-s002
   source 18.2s  fragments 7 -> 10
   base : Well it's just that what does c mean It means that everyone is unable 
   nflow: Well it's just that things are irrational meaning that people are unab

07_zh2en_zh2en-04-fin-s008
   source 17.5s  fragments 8 -> 10
   base : Then for example in 2008 when Bell radar was around all of them actual
   nflow: Then for example in 2008 when Bell radar was around all of it actually

08_zh2en_zh2en-04-fin-s027
   source 16.9s  fragments 7 -> 8
   base : But actually this series is just one kind of every stage it is each st
   nflow: But actually this series is just one kind of every stage it is every s
