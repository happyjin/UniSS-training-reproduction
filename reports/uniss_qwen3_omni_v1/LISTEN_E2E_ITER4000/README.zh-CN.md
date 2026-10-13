# iter 4,000 端到端试听：两次调用 vs 一次调用

同一个 Talker（checkpoint 相同）、同一句译文（两臂 Text-BLEU 完全相同，
38.34 / 39.24）。听到的任何差别，只可能来自隐状态那 3.5% 的数值扰动。

每组四个文件：`_源` 输入音频，`_参考` 目标语参考音频，
`_两次` 现行两次调用的输出，`_一次` 单次调用的输出。

## 全量 128 条的比对

    译文逐字相同     128/128   ← 开关只动了隐状态，证毕
    波形逐字节相同     1/128    ← 那 3.5% 几乎总是改变输出
    码长差（不同的127条）中位 20，最大 263

Talker 是自回归的：某一步 argmax 翻了面，后面整条轨迹就岔开了，
长度也跟着变（20 个码 @ 50 Hz ≈ 0.4 秒）。所以「几乎每条都不同」
是预期之内的。

值得注意的是聚合分数只差 1.62 BLEU（13.10 vs 11.48）——
轨迹几乎全不一样，质量却相当。对 Phase F 是个好消息。

## 聚合分数

    两次调用   zh→en  7.89   en→zh 18.30   mean 13.10
    一次调用   zh→en  5.96   en→zh 17.00   mean 11.48
    Text-BLEU（两臂相同）   38.34 / 39.24

## 01_zh2en_好  （zh→en，好）

- 参考译文：passengers can also use their mobile phone scan q r codes to pay for the ride
- 模型译文（两臂相同）：Passengers can also use their mobile phones to scan the code to pay for the ride.
- 两次调用 ASR：Dressants can also use their fursomes to sew the glove to pay for the ride.   — 句级 BLEU 38.4
- 一次调用 ASR：Dressins can also use mobile songs to sign the craft pool, pay for the dry time,   — 句级 BLEU 14.4

## 02_zh2en_中位  （zh→en，中位）

- 参考译文：plague in the temple wrote the vertical couplet said in the dates and seasonal festivals below ceremonies will be held in the temple
- 模型译文（两臂相同）：There are plaques and couplets in the temple. On the following days and seasons, the temple will hold ceremonies to celebrate.
- 两次调用 ASR：there are powarts and paminos in the power on the fillin days on sighs the papal will house psoriasis to colorate   — 句级 BLEU 4.3
- 一次调用 ASR：There are packs and pomperits in the parks. On the following days of Scythe, the Bucky will help Saratyrs to come play.   — 句级 BLEU 4.0

## 03_zh2en_差  （zh→en，差）

- 参考译文：gmelin later published several textbooks on chemistry pharmaceutical science mineralogy and botany
- 模型译文（两臂相同）：Gmelin later published several textbooks on chemistry, pharmaceutical science, mineralogy and botany.
- 两次调用 ASR：THEIR GRAKE PROCEDS SEVERAL DEGREE MARK, EHONDA, PHYLACTURE, MENOLOGY, AND VOGENY.   — 句级 BLEU 0.0
- 一次调用 ASR：Theory of Valmhosa, Manitobulets on Karmirisai-sai   — 句级 BLEU 3.2

## 04_en2zh_好  （en→zh，好）

- 参考译文：好奇怪的石头
- 模型译文（两臂相同）：一块很奇怪的石头
- 两次调用 ASR：一件很奇怪的石头   — 句级 BLEU 51.7
- 一次调用 ASR：一界很奇怪的石头   — 句级 BLEU 51.7

## 05_en2zh_中位  （en→zh，中位）

- 参考译文：同年加入中国人民解放军。
- 模型译文（两臂相同）：同年,他加入了人民解放军。
- 两次调用 ASR：谁能他加入了人民技术官队   — 句级 BLEU 9.6
- 一次调用 ASR：门上他加入了人民界庄严   — 句级 BLEU 9.7

## 06_en2zh_差  （en→zh，差）

- 参考译文：刘约子。
- 模型译文（两臂相同）：刘月季
- 两次调用 ASR：中野鸡   — 句级 BLEU 0.0
- 一次调用 ASR：中冶西   — 句级 BLEU 0.0
