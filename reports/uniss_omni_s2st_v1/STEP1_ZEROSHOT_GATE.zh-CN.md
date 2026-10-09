# 步骤 1 决策门:Qwen2.5-Omni-3B 零样本 S2TT

**日期** 2026-10-09 · **结论:通过,继续替换计划**

## 为什么先跑这一步

替换基座的理由来自别人论文里的一个数字:Qwen-Omni2.5 在 CVSS-T 上
Text-BLEU 34.85 / 24.39,而我们的 0.5B 是 24.15 / 15.33。那是**他们的协议、
他们的解码、他们的参考**。在投入数据流水线和三个训练阶段之前,同一个问题要在
**我们的音频、我们的参考、我们的打分器**上重新问一遍。

门的设定(写在 `OMNI_REPLACEMENT_PLAN.zh-CN.md`):显著高于 24.15 / 15.33 则继续;
若差距远小于论文的 +10,说明差距来自协议或数据而非基座,**停下重新评估**。

## 结果

CVSS-T test 全量 4,897 对 × 2 方向,`evaluation.text_metrics` 仓库协议
(去标点、OpenCC 简体化,再 sacrebleu),贪心解码,Talker 关闭。

| 方向 | 我们 Phase3 (0.5B) | **Omni-3B 零样本** | 差值 | 论文 7B 零样本 | 论文的差值 |
|---|---|---|---|---|---|
| en→zh | 24.15 | **30.56** | **+6.41** | 34.85 | +10.70 |
| zh→en | 15.33 | **26.65** | **+11.32** | 24.39 | +9.06 |

两个方向平均 +8.9,与论文 7B 的 +9.9 同一量级 —— 而这是 **3B、零样本、一步训练都没做**。

值得单独指出:zh→en 的 **26.65 高于论文 7B 的 24.39**,也已接近
SimulS2ST-Omni **微调后**的 3B(26.41)。我们原先最差的方向,在换基座后
变成差距最大的受益方向。

**门:通过。** 差距不在协议也不在数据,就在基座。

## 解码健康度

| 方向 | 长度比中位 | p95 | 退化率 | BP | 空输出 |
|---|---|---|---|---|---|
| en→zh | 1.00 | 1.83 | 0.16% (8/4897) | 1.00 | 0 |
| zh→en | 1.00 | 1.60 | 0.25% (12/4897) | 1.00 | 0 |

(退化 = 某个 3-gram 重复 ≥5 次。)

## 两个会悄悄给出错误数字的坑

记下来,因为两个都**不会报错**,只会让 BLEU 变成别的东西。

**一、`max_new_tokens` 被静默忽略。** Omni 的 `generate()` 显式声明
`thinker_max_new_tokens=1024`,并在文档里写明带前缀的关键字**优先于**不带前缀的
`max_new_tokens`。所以我传的 128 完全没生效。贪心解码在生僻输入(物种分类名之类)
上会进入重复循环,一路跑满 1024 —— 有一条假设是 **1023 词,对应的参考只有 11 词**。

代价不是边角:第一次 100 条抽样跑出 zh→en **13.91**,`sys_len 2382` 对
`ref_len 1188`,正好 2 倍。改用 `thinker_max_new_tokens=256` 后,同样的模型、
同样的协议,全量是 **26.65**。中位长度比当时就已经是 1.0 —— 整个长度膨胀来自
极少数跑飞的输出,而它们足以把语料级 BLEU 拉掉一半。

**二、`apply_chat_template` 对单个对话也返回列表。** 逐个对话调用会得到
嵌套列表,`replace_multimodal_special_tokens` 随后对一个 list 做 `re.finditer`。
这个坑会直接抛异常,所以反而是安全的;整批一次调用即可。

另外两处是预防性的,不是事后发现的:解码器批量生成必须**左填充**,否则短样本
从自己的 padding 中间开始解码;分片只给自己那一片打分,**语料 BLEU 不是各分片
BLEU 的平均**,所以 `merge_shards.py` 合并后整体重打一次。

## 环境:没有安装任何包

`Qwen2_5OmniProcessor` 在这台机器上**无法实例化** —— 它声明了一个需要
torchvision 的视频处理器,而这里没有 torchvision,也不能装:conda 环境是
一个固定的 Megatron/CUDA 栈,有在跑的任务依赖它。

S2TT 两条视觉分支都不做任何事,所以 processor 由 checkpoint 自己的 tokenizer、
Whisper 特征提取器和图像处理器重建
(`experiments/uniss_omni_s2st_v1/runtime/omni_processor.py`)。图像处理器必须保留 ——
`replace_multimodal_special_tokens` 在函数顶部无条件读两个 `merge_size`,
在真正分派到各自分支之前。视频那个用一个只带该字段的替身代替,取值来自
checkpoint 自己的配置;这个值永远不会被消费,因为只有 `video_token` 分支会用它,
而没有视频时该分支不可能触发。

## 复现

```bash
export PYTHONPATH=. HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
R=/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test
O=reports/uniss_omni_s2st_v1/step1_zeroshot

# 8 卡,每个方向 4 片
g=0
for D in zh2en en2zh; do for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY -m experiments.uniss_omni_s2st_v1.evaluation.omni_zeroshot_s2tt \
    --model pretrained_models/Qwen2.5-Omni-3B \
    --manifest $R/manifests/cvss_t_zh_en_test_pairs.jsonl \
    --audio-root $R --direction $D --batch-size 8 \
    --shard $i --num-shards 4 --output $O/shards/full_${D}_s${i}.json &
  g=$((g+1))
done; done; wait

for D in zh2en en2zh; do
  $PY -m experiments.uniss_omni_s2st_v1.evaluation.merge_shards \
    --shards $O/shards/full_${D}_s*.json --direction $D --output $O/full_${D}.json
done
```

全量两个方向 8 卡约 10 分钟。逐条假设保存在 `full_{zh2en,en2zh}.json` 的 `rows` 里。

## 这一步没有碰任何东西

不训练、不改动既有流水线、不装包。`evaluation/cvss_t/canonicalize.py` 的两个
新增参数默认值等于原有常量,所有既有调用方行为逐位不变。

## 下一步

Stage 0 外科手术与 no-op 验证。另外,训练用的 dev 曲线数据已就位,见
`cvss-t: a dev split to curve against` 一次提交:CVSS-T dev 4,843 对,
与 test 在 id 和原文两个层面零重叠,配 1,000 对按时长分层的固定子集。
