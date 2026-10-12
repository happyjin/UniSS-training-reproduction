# 试听：Qwen3-Omni-30B cascade vs Stage 1

**随机抽取**（固定种子 20261012），不是挑出来的好例子。

每条四个文件：

| 后缀 | 内容 |
|---|---|
| `_1源音频` | 输入的真实人声 |
| `_2译文_Qwen3` | **Qwen3-Omni-30B 翻译 + 我们的 Talker 发声** |
| `_3译文_Stage1` | 同一条，Qwen2.5-Omni-3B 翻译(旧) |
| `_4参考译音` | CVSS-T 目标端参考 |

听两件事：译文对不对；`_2` 和 `_3` 的**音色是否都跟着 `_1` 走**（音色由 UniSS 的 32 个 global token 直通声码器决定，两者用的是同一条通路）。

## cmn_eng_00（中→英）

- 源文：这段时期也是佛教艺术和建筑的繁荣期。
- **Qwen3 译文**：This period was also a time of prosperity for Buddhist art and architecture.
- Stage 1 译文：This period was also a period of prosperity for Buddhist art and architecture.
- 参考译文：this was also a flourishing period of buddhism art and architecture

## cmn_eng_01（中→英）

- 源文：沃特曼斯科纳是位于美国加利福尼亚州因皮里尔县的一个非建制地区。
- **Qwen3 译文**：Watsonville is an unincorporated community in Imperial County, California, United States.
- Stage 1 译文：Watman-Skona is an unincorporated community in Imperial County, California, United States.
- 参考译文：waterman corner is an unincorporated community in imperial county california united states

## cmn_eng_02（中→英）

- 源文：包含东环线正线、机场支线和黄茅坪支线，大部分位于重庆市主城区范围。
- **Qwen3 译文**：It includes the main line of the East Ring Line, the airport branch line and the Huangmaoping branch line, most of which are located in the main urban area of Chongqing.
- Stage 1 译文：It includes the East Loop Line, the Zhengxian Line, the Airport Line and the Huangmaoping Line, most of which are located in the main urban area of Chongqing.
- 参考译文：includes eastern ring main line airport line and huangmao ping line mostly within chongqing main city vicinity

## cmn_eng_03（中→英）

- 源文：好奇怪的石头
- **Qwen3 译文**：What a strange stone.
- Stage 1 译文：What a strange stone!
- 参考译文：quite a weird stone

## cmn_eng_04（中→英）

- 源文：毕业于力行中学补校，曾任阳明山信用合作社理事、阳信银行董事。
- **Qwen3 译文**：Graduated from Lihsing Junior High School, served as a director of Yangmingshan Credit Union and a director of Yangxin Bank.
- Stage 1 译文：Graduated from Lixing Middle School, served as the director of Yangmingshan Credit Cooperative and the director of Yangxing Bank.
- 参考译文：graduated from lixing middle school once served as director of yangmingshan credit cooperative and director of yangxin bank

## cmn_eng_05（中→英）

- 源文：最终夺冠。
- **Qwen3 译文**：Finally won the championship.
- Stage 1 译文：Win the championship.
- 参考译文：won the championship at last

## cmn_eng_06（中→英）

- 源文：瓦根舍因以推广开放式学习技术知名。
- **Qwen3 译文**：Wagener is known for promoting open learning technology.
- Stage 1 译文：Wagenstein is known for promoting open learning technologies.
- 参考译文：wagenshe is famous for promoting open learning technology

## cmn_eng_07（中→英）

- 源文：万历八年，登进士第二甲第三十一名。
- **Qwen3 译文**：In the eighth year of Wanli, he passed the Jinshi examination and ranked 31st in the second class.
- Stage 1 译文：In the eighth year of Wanli, he was ranked 31st among the second grade of Jinshi.
- 参考译文：in the eighth year of wanli win the thirty first place at jinshi second round

## eng_cmn_00（英→中）

- 源文：this was also a flourishing period of buddhism art and architecture
- **Qwen3 译文**：这也是佛教艺术和建筑繁荣的时期。
- Stage 1 译文：这也是佛教艺术和建筑繁荣的时期。
- 参考译文：这段时期也是佛教艺术和建筑的繁荣期。

## eng_cmn_01（英→中）

- 源文：waterman corner is an unincorporated community in imperial county california united states
- **Qwen3 译文**：沃特曼角是美国加利福尼亚州帝国县的一个非建制社区。
- Stage 1 译文：沃特曼角位于美国加利福尼亚州伊普里尔县的一个未建制社区。
- 参考译文：沃特曼斯科纳是位于美国加利福尼亚州因皮里尔县的一个非建制地区。

## eng_cmn_02（英→中）

- 源文：includes eastern ring main line airport line and huangmao ping line mostly within chongqing main city vicinity
- **Qwen3 译文**：包括环线、国铁、机场线和璧铜线,主要位于重庆主城附近。
- Stage 1 译文：包括重庆主城的东环线、机场线和环线。
- 参考译文：包含东环线正线、机场支线和黄茅坪支线，大部分位于重庆市主城区范围。

## eng_cmn_03（英→中）

- 源文：quite a weird stone
- **Qwen3 译文**：一块很奇怪的石头
- Stage 1 译文：“真奇怪的石头。”
- 参考译文：好奇怪的石头

## eng_cmn_04（英→中）

- 源文：graduated from lixing middle school once served as director of yangmingshan credit cooperative and director of yangxin bank
- **Qwen3 译文**：毕业于立新中学，曾任阳明山信用合作社主任、永信银行主任。
- Stage 1 译文：毕业于立信中学，曾任阳明山信用合作社和阳信银行的董事。
- 参考译文：毕业于力行中学补校，曾任阳明山信用合作社理事、阳信银行董事。

## eng_cmn_05（英→中）

- 源文：won the championship at last
- **Qwen3 译文**：终于赢得了冠军。
- Stage 1 译文：最终赢得了冠军。
- 参考译文：最终夺冠。

## eng_cmn_06（英→中）

- 源文：wagenshe is famous for promoting open learning technology
- **Qwen3 译文**：Wag enchay 以推广开放式学习技术而闻名。
- Stage 1 译文：Wageningen University is famous for promoting open learning technology.
- 参考译文：瓦根舍因以推广开放式学习技术知名。

## eng_cmn_07（英→中）

- 源文：in the eighth year of wanli win the thirty first place at jinshi second round
- **Qwen3 译文**：万历八年，中进士第二甲第三十一名。
- Stage 1 译文：在王磊的第八年，他赢得了第三十一届锦池第二轮比赛的冠军。
- 参考译文：万历八年，登进士第二甲第三十一名。

