# v1 vs v2：格式修正的听感对照

同一批 128 条，同一冻结 Thinker。译文逐字相同的 128/128 条，
所以听到的差别全部来自 Talker 读到的隐状态格式。

    v1 iter 8,000   回复带转写前缀（训练格式），推理给的是单行   mean 12.06
    v2 iter 2,000   训练与推理都是单行译文                      mean 18.53

每组：`_源` 输入，`_参考` 目标语参考，`_v1` 旧格式，`_v2` 修正后。

## 01_zh2en_v2大胜

- 参考译文：baotuo temple may refer to
- 模型译文：Baotuo Temple may refer to:
- **v1** ASR：Books who temper a poem might refer to   — 句级 BLEU 11.0
- **v2** ASR：Bottle sample may refer to   — 句级 BLEU 39.8

## 02_zh2en_v2小胜

- 参考译文：shannon hills is a city in saline county arkansas united states
- 模型译文：Xianglong Mountain is a city in Saline County, Arkansas, United States.
- **v1** ASR：Xianglong Shanzhen is a city of the Bailing County, as grown in the States.   — 句级 BLEU 9.8
- **v2** ASR：Xiong'an Lanbangton is a city in Khao San Lai County, Afghanistan, United States.   — 句级 BLEU 15.3

## 03_zh2en_持平

- 参考译文：fullerton is a town located in kensas state u s under municipal of hempstead county
- 模型译文：Fulton is a town in Hempstead County, Arkansas, United States.
- **v1** ASR：Afan Transichan in Fastet County, S.S.R., U.S.   — 句级 BLEU 2.3
- **v2** ASR：Furtin is a Venn and Hapstead County R.S. Gas, United States.   — 句级 BLEU 4.9

## 04_zh2en_v1更好

- 参考译文：he joined the people's liberation army in the same year
- 模型译文：Joined the People's Liberation Army in the same year.
- **v1** ASR：Join the Timings Liberation Army in the same year.   — 句级 BLEU 27.8
- **v2** ASR：Reading The Cheapest Liberation Army in the Same Year   — 句级 BLEU 8.5

## 05_en2zh_v2大胜

- 参考译文：英国首相迈克尔·凯罗夜里醒来得知备受爱戴的皇室成员苏珊娜公主已被绑架。
- 模型译文：英国首相迈克尔·科尔在夜里醒来,得知深受爱戴的皇室成员苏珊娜公主被绑架了。
- **v1** ASR：俄科手指量迈克尔克在胃地醒来读书同然厌的皇室平原苏申东指被绑架了   — 句级 BLEU 10.0
- **v2** ASR：英国首冷迈克尔卡在夜务醒来得知重生爱戴的皇室成员苏珊途战总指被绑架了   — 句级 BLEU 40.2

## 06_en2zh_v2小胜

- 参考译文：宝陀寺，可以指
- 模型译文：Butuo Temple可能指:
- **v1** ASR：dumber a twelve   — 句级 BLEU 0.0
- **v2** ASR：but two timit可能只   — 句级 BLEU 6.9

## 07_en2zh_持平

- 参考译文：露露夫人终究与三姐弟达成了协议。
- 模型译文：毕竟,露露太太和她的三个兄弟姐妹达成了协议。
- **v1** ASR：一般说露露太太和他的三个兄弟姐妹达成了协议   — 句级 BLEU 21.4
- **v2** ASR：现着露露太太和他的三个兄弟姐妹达成了协议   — 句级 BLEU 22.5

## 08_en2zh_v1更好

- 参考译文：富尔顿是一个位于美国阿肯色州亨普斯特德县的市镇。
- 模型译文：富勒顿是美国堪萨斯州亨普斯特德县下属的一座城市。
- **v1** ASR：富尔顿是美国堪萨斯州汉佩斯特德县相房的一座城镇   — 句级 BLEU 23.6
- **v2** ASR：富拉顿是美国堪萨斯州亨皮斯特德线段相馆的一周吃针   — 句级 BLEU 10.4

## 全量 128 条

    v2 更好 89 条，v1 更好 20 条，持平 19 条
