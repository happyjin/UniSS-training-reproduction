# Stage 2 中途试听（iter 3500 / 6000）

训练**尚未结束**。这是训练进行到 58% 时导出的 checkpoint，用来确认声音正常、
不是等到最后才发现问题。正式数字以训练结束后的完整 CVSS-T 为准。

## 怎么听

`wav/` 下每条样本三个文件：

| 后缀 | 是什么 |
|---|---|
| `_源音频` | 输入的真实人声 |
| `_译文.wav` | **模型产出的翻译语音** |
| `_参考译音` | CVSS-T 的目标端参考（合成音） |

听两件事：译文说的对不对，以及 `_译文` 的音色是否跟着 `_源音频` 走——
音色保持靠的是 UniSS 的 32 个 bicodec_global token 直通声码器，不经过语言模型。

## 样本

| 样本 | 转写（模型产出） | 译文（模型产出） | 参考译文 | 时长 vs 参考 |
|---|---|---|---|---|
| cmn2eng_00 | 正巧母亲往外探头。 | Just as my mother leaned out to look. | it happened that mother was looking out | 3.10s / 2.04s |
| cmn2eng_01 | 至今为止元气火拳总共发行了两张专辑。 | So far, Yuan Qi Huo Quan has released two albums. | genki rockets has thus far released two albums | 2.86s / 2.74s |
| cmn2eng_02 | 失业率降到十七年以来的新低点。 | The unemployment rate has fallen to a new low in seventeen years. | unemployment rate has dropped to a new low in seventeen years | 4.70s / 3.81s |
| eng2cmn_03 | It happened that mother was looking out. | 母亲恰好在向外看。 | 正巧母亲往外探头 | 2.14s / 3.24s |
| eng2cmn_04 | Jenky Rockets has thus far released two albums. | Jenky Rockets 至今已发行了两张专辑。 | 至今为止，元气火箭总共发行了两张专辑。 | 3.30s / 5.06s |
| eng2cmn_05 | Unemployment rate has dropped to a new low in seventeen years. | 失业率已（**截断**） | 失业率降到十七年来的新低点 | 1.14s / 4.46s |

## 已知问题，以及它不是什么

`eng2cmn_05` 被截断，原因是**这个试听脚本**把 `max_new_tokens` 设成了 96 好在
CPU 上跑快些；中文在该 tokenizer 下更费 token，于是在译到一半时撞上预算。正式
评测脚本默认 256，不受影响。

它顺带暴露了一个会静默出错的真问题：`split_reply` 原本取最后一行，而跑飞的回复
会多出一行重试片段，于是打分打到了片段上。训练目标严格是 `转写\n译文` 两行，
现已改为取第二行（commit d43e915a）。

## 复现

```bash
# 1. 导出 checkpoint
PYTHONPATH=. python -m experiments.uniss_omni_s2st_v1.training.export_stage2_from_megatron \
  --checkpoint-dir checkpoints/uniss_omni_s2st_v1/stage2_formal_20261011T043541Z/iter_0003500 \
  --output .../exported/stage2_iter0003500.pt

# 2. 生成（训练占满 GPU 时，模型跑 CPU，只有声码器用 GPU）
#    训练结束后直接用正式脚本即可：
CKPT=.../stage2_iter0006000.pt OUT=.../table1_stage2 \
  bash experiments/uniss_omni_s2st_v1/scripts/run_stage2_table1.sh
```

注：生成这批样本时我的 CPU 任务把 192 核吃满，训练步时从 3.0s 涨到 5.0s；
把任务降到 nice 19 后恢复。以后在训练期间做 CPU 旁路要先降优先级。
