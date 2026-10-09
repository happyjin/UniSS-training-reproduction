# Stage 0:手术与空操作验证

**日期** 2026-10-09 · **结论:通过** · 不训练,不写回 checkpoint

## 一个本来要自己造、结果 Omni 自带的东西

计划在"要新建的组件"里列了**交叉条件通路**:"Omni Thinker 与 Talker 维度不同,
需投影层"。检查 checkpoint 时发现它已经存在并且是**预训练好的**:

```
talker.thinker_to_talker_proj.weight   [896, 2048]
```

而且 Talker 的 `forward` 揭示了条件化的真实形式 —— 不是交叉注意力:

```python
inputs_embeds   = embed_tokens(codec_ids) + thinker_reply_part[:, :1, :]   # 2048 维相加
talker_lm_input = thinker_to_talker_proj(inputs_embeds)                    # → 896
```

两个推论直接决定了训练时怎么喂数据:条件化是**在 Thinker 的 2048 维空间里相加**;
以及**每生成一个码,消费一个文本隐状态**。

## 码本尺寸正好对上,一个张量都不用改形状

| | 槽位 | 用途 |
|---|---|---|
| Omni Talker 词表 | 8,448 | — |
| 其中码字 | **0 – 8,191** | 对应我们 BiCodec 语义码本的 **8,192** |
| 保留块 | 8,192 – 8,447 | 特殊符 pad 8292 / bos 8293 / eos 8294 / mask 8296 |

**没有 resize,没有 append,四个特殊符 id 原样保留。** 变的是槽位的**含义** ——
Omni 的第 k 行编码它自己 codec 的第 k 个码,和 BiCodec 的第 k 个码毫无关系,
所以这些行要重画。只动两个张量:

```
model.embed_tokens.weight  [8448, 2048]   码嵌入(Thinker 维度)
codec_head.weight          [8448,  896]   输出头
```

全部 24 层 transformer、以及 `thinker_to_talker_proj`,**一概不动**。

## 决定性的检查:因果与错位

这是唯一一个接线错误过不去的测试,所以它排在所有数字前面。位置 t 吃的是
`[bos, codes[0..t-1]]`、预测 `codes[t]`,因此 `codes[t]` 从输入下标 t+1 进入。
扰动它,**位置 ≤ t 的每一个 logit 必须逐位不变**,而 t+1 必须变。

| 探测位置 | 扰动前及该位的最大 logit 变化 | 被扰动位上的变化 |
|---|---|---|
| 开头 (输入下标 1) | **0.000** | 0.812 |
| 中间 (下标 34) | **0.000** | 2.516 |
| 结尾 (下标 65) | **0.000** | 1.531 |

**精确的零。** 标签错位一格、文本流误对齐到它要预测的码、因果掩码损坏 —— 三类
错误都会在这里现形。另外所有 32 条样本的最大码值是 **8,190 < 8,192**,码本没有被
静默截断。

## 初始化:扫描,不是论证

新输出头该按什么尺度初始化,不是显然的,所以做成参数量出来。先纠正一个我一开始
弄错的事实:**码行与保留行的尺度差一个数量级**,用全张量统计量两边都不对。

| 张量 | 码行逐元素 std | 保留行逐元素 std |
|---|---|---|
| `embed_tokens` | 0.1096 | 0.0102 |
| `codec_head` | 0.0204 | 0.0109 |

按码行自身尺度重画,再用 `head_scale` 缩放头的那一笔(32 条样本,均匀 CE = ln(8192) = **9.011**):

| head_scale | CE | 熵 | eos 质量 |
|---|---|---|---|
| 1.0 | 11.365 | 6.680 | 0.059 |
| 0.5 | 9.914 | 7.584 | 0.122 |
| 0.25 | 9.576 | 7.774 | 0.142 |
| **0.1** | **9.489** | **7.820** | 0.148 |
| 0.0 | 9.477 | 7.828 | 0.150 |

按预训练尺度(1.0)重画的头,**和训练好的头一样自信、却是随机的** —— CE 比均匀
分布还高 2.35,熵低 2.33。那等于让训练开场就先和自己的噪声打架。

**取 head_scale = 0.1。** 与 0.0 相差 0.012 nats,实质等同,但保留一点行间差异,
不至于让所有码的 logit 精确相等。

CE 9.489 与 9.011 之间那 0.478 不是异常,是 eos 占掉的:各位置上
`ln(1/(1-eos_t))` 的均值,因 eos 质量在位置间方差很大(个别位置高达 0.8)而
由 Jensen 不等式高于按均值算的 0.16。

## 一个我写错又改回来的判据

我最初把"概率质量落在 256 个保留行上"写成了码区间错误的证据。**这是错的。**
拆开量之后:那 19.4%(32 样本下 14.8%)**几乎全在 eos 上**,其余 255 行只占
**0.36%**。而 eos 是合法输出,必须留着。

eos 发放这么强本身也讲得通:4 个汉字的译文要配 53–62 个码,文本流后段全是 pad
嵌入 —— 在 Omni 自己的训练里,那正是"该停了"的信号。模型说的没错,它只是还不
知道我们的码在文本耗尽后仍要继续。

### 顺带发现的一个真缺口

目标序列里**没有 eos**,照这样训出来的 Talker 永远学不会何时停。这在这里比看上去
严重:文本远早于码耗尽,能告诉模型话到哪儿结束的,只有 eos 标签本身。
`build_codec_sequence` 已补上可选的 `eos_id`。

## 输出屏蔽:保留,但按它真实的价值说

8,448 个槽位里只有 8,193 个在我们的设置下可达(8,192 个码 + eos)。其余 255 行是
Omni 自己 codec 的簿记,我们的目标里永远不含,声码器也没有对应条目。

屏蔽它们让 CE 动了 **0.004 nats** —— 所以**这不是质量改进,不该当成质量改进来说**。
它买到的是保证:无论训练中输出头怎么漂移,解码都不可能吐出一个没有声学含义的 token。

## 数据通路

| | |
|---|---|
| 源音频 | manifest 的 `wds://` 字节范围 → 本地 tar 直接 pread → 完整 FLAC |
| 目标码 | 本地 parquet 的 `target_bicodec`(8,192 码本) |
| 说话人 | `bicodec_global` 32 个 token,**绕过语言模型** |

manifest 自带的 `target_codec_codes` 是 Qwen3-TTS-Tokenizer 12 Hz,**不是本项目的
codec**,没有采用。两边按 `provenance.source_row_index` 位置对齐 —— 位置对齐意味着
上游一次重排就会把音频配到别人的码上并照常训练,所以 id 与 `target_bicodec` 长度
两道校验是硬失败。解码时长与 manifest 的 `source_duration_ms` 实测完全一致。

## 复现

```bash
export PYTHONPATH=. HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python

CUDA_VISIBLE_DEVICES=0 $PY -m experiments.uniss_omni_s2st_v1.evaluation.stage0_forward_check \
  --manifest data/unist_omni_v1/manifests/manifest-train-00001.jsonl \
  --samples 32 \
  --output reports/uniss_omni_s2st_v1/stage0/forward_check.json

# 看屏蔽本身值多少
#   加 --no-output-mask
```

单卡数分钟。单元测试 `experiments/uniss_omni_s2st_v1/tests/`(39 项)。

## 下一步

Stage 1 Talker warmup:冻结音频编码器与 Thinker,只训 Talker 与交叉通路,
TTS 任务(目标文本 → BiCodec 码)。dev 曲线用
`CVSS/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl`,
与上报的 test 集零重叠。
