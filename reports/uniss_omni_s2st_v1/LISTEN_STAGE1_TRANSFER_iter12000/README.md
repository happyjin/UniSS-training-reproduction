# Stage 1 试听 — UniSS 迁移初始化, iter 12000

每条两份:

- `NN_方向_生成.wav` — 模型输出
- `NN_方向_金标.wav` — 同一句的**金标 BiCodec 码**,经同一个声码器、
  同一批 32 个说话人 token 解出 —— 这条链路的上限

| # | 方向 | 参考文本 | 生成码 | 金标码 | 生成时长 |
|---|---|---|---|---|---|
| 01 | eng→cmn | 姜涛曾就读轩尼诗道官立下午小学、邓肇坚维多利亚官立中学和青年学院。 | 600 | 444 | 12.00s |
| 02 | cmn→eng | keung to attended hennessy road government p | 600 | 424 | 12.00s |
| 03 | eng→cmn | 在工作了九年后，伯爵不幸去世。 | 600 | 198 | 12.00s |
| 04 | cmn→eng | after nine years of work the earl passed awa | 600 | 144 | 12.00s |
| 05 | eng→cmn | 整个系统称为键接合。 | 600 | 174 | 12.00s |
| 06 | cmn→eng | the complete system is called conjugation | 600 | 136 | 12.00s |
| 07 | eng→cmn | 一些听众跟随至办事处，与审讯人员进行说理斗争控诉。 | 600 | 322 | 12.00s |
| 08 | cmn→eng | some of the audience followed to the office  | 600 | 198 | 12.00s |
| 09 | eng→cmn | 一些附近的居民爬上较高的树木而逃过了海啸。 | 600 | 345 | 12.00s |
| 10 | cmn→eng | some nearby residents climbed on the high tr | 600 | 201 | 12.00s |
| 11 | eng→cmn | 里舍·巴塔尼是一位印度演员，常出演宝莱坞电影。 | 600 | 351 | 12.00s |
| 12 | cmn→eng | rishi bhutani is an indian actor who often a | 600 | 242 | 12.00s |
| 13 | eng→cmn | 后被选中为庶吉士，散馆授编修。 | 600 | 417 | 12.00s |
| 14 | cmn→eng | later he was selected as a shujishi and was  | 600 | 268 | 12.00s |
| 15 | eng→cmn | 许多法律制度对大多数或所有的案件，或者是某些特定类型的案件会使用法官审判。 | 600 | 504 | 12.00s |
| 16 | cmn→eng | many legal systems adopt judges for most or  | 600 | 265 | 12.00s |

## 先说结论

**16 条全部撞满 600 码的上限(12.00 秒),模型从不发 eos。** 金标是 136–444 码
(2.7–8.9 秒),所以生成长度是金标的 2.25 倍(中位)。

金标那一列请一并听:它证明声码器、BiCodec 码本映射和说话人通路都是好的,
差距完全在 Talker 自己。

训练侧的对应事实:CVSS-T dev 从 iter 4000 起就走平在 6.86–6.90,
九千步没有实质进展,而 dev 6.86 对应困惑度约 950 —— 离可懂语音仍差两个数量级。
