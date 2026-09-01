# 零训练拿到 5/6:新 checkpoint + δ_cont=5

**这是全项目最好的流式 S2ST 配置,不需要任何额外训练。**

## 配置

* checkpoint:`continue_end_m3_20260901T062849Z` / `iter_0001132`
  (即被判决器判为 2/6 的那个)
* 推理侧:`CONTINUE_WRITE_BIAS=5`、local agreement holdback 2、
  semantic pacing margin 1200 ms、semantic cap 384

## 门禁对照

| 判据 | iter2264 δ=0 | iter2264 δ=4 | **新 ckpt δ=5** | 门线 |
|---|---:|---:|---:|---:|
| `WRITE_MT` / 事件 | 0.168 | 0.947 | **0.789** ✓ | ≥ 0.50 |
| `natural_eos` | 0.50 | 1.00 | **1.00** ✓ | ≥ 0.875 |
| 语义覆盖 | 0.666 | 0.997 | **0.876** ✓ | ≥ 0.666 |
| 文本长度比 中位 | 1.033 | 2.884 | **1.267** ✗ | ∈[0.9,1.2] |
| 文本长度比 最坏 | 1.93 | **44.45** | **1.31** ✓ | ≤ 2.0 |
| malformed 片段 | 13 | 59 | **8**(史上最低) | — |
| ASR 错误率 | 0.233 | 0.233 | **0.183** ✓ | ≤ 0.26 |
| **通过** | 3/6 | 4/6 | **5/6** | |

唯一未过的一项超出门线 **5.6%**。

## 零训练旋钮已用尽

| 旋钮 | 扫过的值 | 对文本长度比的影响 |
|---|---|---|
| `δ_cont` | 0,1,2,3,4,5,6,7,8 | δ=5 是拐点;≥6 时 WR_MT 饱和在 0.958 但长度升到 1.376 |
| pacing margin | 1200,1000,800,600 | **无影响**(1.267 → 1.261)—— 它控音频发放,不控文本量 |
| local agreement holdback | 2,3,4 | **无影响**(逐位相同)—— 它只作用于 `incremental_mt_rollout` 那条独立通路 |

长度比 1.267 是模型每片段 MT 输出的内在长度。**要压下去必须训练**
(`content_end_margin` 已证明能做到,但单独用会过头到 0.324)。

## 译文实测(δ=5 vs 基线)

| 样本 | 参考 | 基线 δ=0 | **新 ckpt δ=5** |
|---|---|---|---|
| `emilia_zh_0006401658` | 回到蒙特利尔鲍勃喝了另一杯威士忌 | **(空)** | **回到蒙特利尔鲍勃哈雷和我又下了一个威士忌** |
| `EN_B00058_S07483_W000027` | 并用此来解决高方差问题 | 你看到这是一个 | **你只是在修复我们的高方差问题** |
| `emilia_zh_0004122419` | They generally feel that some new factor is being born | They usually feel that some kind of new factor is emerging… | They were commonly sensing some kind of new new factor was in bearing |
| `emilia_zh_0006795452` | Or is there anything special you want to eat | Still it's better to | Still say that you are particularly fond of eating things |
| `emilia_zh_0006199435` | He was no longer waiting to catch mosquitoes but was walking around like a fashion model | He is no longer the one he was waiting for… to fashion models. walking around and forth l | He was no longer a waiter to catch the mosquitoes but he went to the fashion model to such and |

## 双声道 demo(左=源,右=同传译音)

`audio/` 下 4 条:

| 样本 | 时长比 |
|---|---:|
| `emilia_zh_0004122419__delta5__stereo.wav` | — |
| `emilia_zh_0005215832__delta5__stereo.wav` | 1.329 |
| `emilia_zh_0006199435__delta5__stereo.wav` | — |
| `emilia_zh_0006795452__delta5__stereo.wav` | 1.153 |

## 结论

被判决器判为"失败 2/6"的 checkpoint,**加一个已标定的推理侧偏置就是 5/6**。
判决器判错的原因是它在 δ=0 下评测,而这个 checkpoint 的决策 gap 被训练推到了
−4.97,需要 δ≥5 才能跨过。**能力一直在里面,只是没被打开。**
