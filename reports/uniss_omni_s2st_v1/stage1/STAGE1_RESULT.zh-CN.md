# Stage 1 Talker warmup:**验收不通过**

**日期** 2026-10-09 · Megatron,8×H200 · 816,355 条训练语料 · 1.92M 样本

## 结论先放前面

Talker 生成的语音**基本不可懂**。而同一条解码链路、同一批 32 个说话人 token,
喂金标码就能解出高质量语音 —— 所以问题完全在 Talker,不在声码器、码本映射或数据通路。

| | cmn→eng ASR-BLEU | UTMOS | eng→cmn ASR-BLEU | UTMOS |
|---|---|---|---|---|
| **金标码(上限)** | **81.92** | **3.749** | **72.57** | **3.129** |
| Stage 1 Talker | **0.016** | 1.487 | **0.006** | 1.480 |

时长比均值 2.20(cmn→eng)/ 2.94(eng→cmn):生成的码序列是金标的两倍以上,
模型还没学会何时停。

训练本身是健康的 —— 损失单调下降、超过基线、GPU 吃满 —— 但 dev CE 7.29 对应
**困惑度约 1,470**(在 8,193 个码上)。一个能出可懂语音的码语言模型,困惑度要在
几十的量级。曲线好看和能用之间差着两个数量级,这一点损失曲线本身看不出来。

## 训练确实比之前好,只是远远不够

等数据量对比(都是 192 万样本):

| 样本数 | DDP(scale=1, LR 2e-4) | Megatron(scale=8, LR 3.5e-4) | 差 |
|---|---|---|---|
| 960,000 | 7.7658 | 7.4201 | −0.346 |
| 1,536,000 | 7.6415 | 7.3053 | −0.336 |
| 1,920,000 | 7.6429 | **7.3556** | −0.287 |

完整 50 点曲线在 `curve_megatron_formal.jsonl`:8.8634(iter 100)→ 最低
**7.2936**(iter 4200)→ 7.3556(iter 5000)。第 3 个 epoch 起有小幅回升,
按 dev 选点应取 iter 4000(7.3053),声学验收用的就是它。

注意这两项同时变了(embed_scale 与 LR),这个对比**不能归因到其中任何一项**。

## 为什么失败:诊断已经指出过

`CONDITIONING_SCALE.zh-CN.md` 里实测:Thinker 隐状态行范数 **173.2**,
码嵌入 **4.78**,而 Talker 把两者**相加之后**才投影。文本到达期间,
"刚发出的是哪个码"以 0.028 的信噪比存活。受控消融显示,offset 5–20 上
带文本反而更差。

`embed_scale=8` 把码行放大到 38,仍比 173 低 4.5 倍 —— 方向对了,幅度不够。

## 下一步的三条路(按证据强弱)

1. **把尺度真正拉平**:`embed_scale` 取到 32–36 量级。必须用**同一条 LR 曲线**
   的两臂对照 —— 上次那个 scale=8 的实验就是栽在这里(见
   `MEGATRON_SATURATION.zh-CN.md` 末节)。
2. **改为归一化条件化项**,而不是放大码嵌入。偏离 Omni 的构造,但直接消除失衡。
3. **大幅延长 Stage 1**。192 万样本对一个从零重画的 8,192 路码本可能本就太少;
   Qwen 自己的 Talker 用的数据多得多。

在 1 和 2 做出干净结论之前,不应推进到 Stage 2 —— Stage 2 要联合训练 Thinker
与 Talker,把一个还不能发声的 Talker 带进去,只会让归因更困难。

## 验收是怎么跑的

```bash
# 1. 从 Megatron 的 torch_dist 检查点导出 Talker
python -m experiments.uniss_omni_s2st_v1.training.export_talker_from_megatron \
  --checkpoint-dir checkpoints/uniss_omni_s2st_v1/stage1_megatron_formal/iter_0004000 \
  --output .../talker_iter0004000.pt      # 293 张量 / 384.6M,键集合与原 checkpoint 完全一致

# 2. 生成 + 解码(8 卡),金标码一并解码作为上限
CKPT=... OUT=... LIMIT=400 \
  bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_acceptance.sh

# 3. 客观指标，按方向分开
EXPECTED_PAIRS=200 MODES=tts \
  bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh <dir> "cmn->eng"
```

指标脚本是 `run_objective_metrics.sh` 的兄弟副本,只改两处:`--modes` 由
`quality performance` 改为 `tts`(Stage 1 是 TTS 预热,不是双模式 S2ST),
以及 Text-BLEU 在没有 vLLM 产物时跳过(TTS 没有"生成的译文"可打分)。
ASR-BLEU、UTMOS、SLC、分片与合并全部逐字节相同,数字因此与已发表的可比。

原脚本未作改动 —— 它被已完成的正式评测引用。

## 未完成项

**AutoPCP 没跑出来**:它要从 HuggingFace 拉模型,本机无外网,缓存里也没有。
本地只有 `evaluation_models/AutoPCP-multilingual-v2` 的比较器本体。
计划里 Stage 1 的门写的就是 A.PCP,所以这一项仍缺 —— 但以 ASR-BLEU 0.016 /
UTMOS 1.49 对 81.92 / 3.75 的差距,补上 AutoPCP 不会改变结论。
