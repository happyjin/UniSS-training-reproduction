# Phase C:装配与验证

**日期** 2026-10-12 · **结论:四项门全部通过,未训练一步。**

---

## 架构

```
源音频
  → Qwen3-Omni-30B-A3B Thinker（AuT 音频塔 + 48 层 MoE，hidden 2048）  ← 替换
      ↓ 隐状态，每个 codec step 一个
  global_embed(32 个 bicodec_global) ⊕ talker.embed(码)                 ← 保留
      ↓ thinker_to_talker_proj [896, 2048]                              ← 保留
  Talker（24 层 h896，Qwen2.5-Omni 的壳 + UniSS phase-3 权重 + Stage 1） ← 保留
      ↓ 8,192 单流语义码 @ 50 Hz
  BiCodec: detokenize(global[:32], semantic[32:])                       ← 保留
      ↓ 波形
```

删掉:Qwen3 自带的 Talker / code2wav,Qwen2.5 的 Thinker。

与 Phase B 的区别:**只有一个 Thinker,中间不再有文本字符串**。这是流式的前提。

## 四项门

| # | 检查 | 结果 |
|---|---|---|
| 1 | 维度端到端对齐 | thinker **2048** → proj **[896, 2048]** → talker **896**;codec 词表 8448,可发射 **8193**(8192 码 + eos);global_embed **[4096, 2048]** ✅ |
| 2 | **因果性** | 最大泄漏 **0.000e+00(精确零)**,扰动处移动 7.3–8.3 ✅ |
| 3 | 音频预处理 | 10.6s → (1,128,1058),5.9s → (1,128,590),**变长,≈100 帧/秒**;抽取 **4–33 ms** ✅ |
| 4 | 显存 | 权重 59.8 GiB,推理峰值 **60.0 GiB** / 139.8 ✅ |

第 3 项原本是风险项:Qwen2.5 的特征抽取不管音频多长都填到 300 秒,曾让
Stage 2 慢 17 倍(抽取 477 ms)。**Qwen3 没有这个问题。**

## 发现并修掉的一个静默 bug

`build_text_stream` 用 `thinker.get_input_embeddings()(talker.text_eos_token)`
取文本流的收尾向量。这两个 id(151859 pad / 151861 eos)**超出 tokenizer 词表**
(151,665),是 Omni 在 embedding 表里为 Talker 保留的槽位。

实测两边的值:

| | 151859(pad)与 151861(eos) |
|---|---|
| Qwen2.5-Omni | **完全相同的向量**(`torch.equal` = True,余弦 1.000000) |
| Qwen3-Omni | 两个不相干的向量(余弦 **−0.038**),**从未为此用途训练过** |

两个推论:

1. **Stage 1/2 里那个「先一个 eos 再铺 pad」的收尾,实际一直就是「pad, pad, ...」**
   —— 代码区分了二者,权重没有。不是 bug,是个从未起作用的设计。
2. 换到 Qwen3 后,借用这两个槽位等于给 Talker 喂两个任意方向当作「文本结束」。

**修法:把这两个标记改成可学习参数**(`StreamMarkers`),用 Qwen3 对应行初始化,
在 Phase D 里随 Talker 一起拟合。彻底去掉对任何词表的依赖。

## 仍待办(Phase D 之前必须做)

**重建 TTS cache 的 token id 部分。** `build_tts_cache.py:112` 存的是用
Qwen2.5 tokenizer 切的 `prompt_ids` / `reply_ids`(5.6 GB)。两边词表不同
(151,665 vs 151,676),拿旧 id 喂 Qwen3 的 embedding 表取到的是别的词,**而且不报错**。

码和 global token 那 9.7 GB **不受影响**,它们从音频提取,与基座无关。

## 复现

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python \
  -m experiments.uniss_qwen3_omni_v1.evaluation.phase_c_check \
  --talker-checkpoint checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt \
  --rows 3 --output eval_outputs/uniss_qwen3_omni_v1/phase_c_check.json
```
