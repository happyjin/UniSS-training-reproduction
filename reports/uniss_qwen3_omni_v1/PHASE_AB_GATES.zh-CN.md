# Phase A / B 决策门:Qwen3-Omni-30B-A3B

**日期** 2026-10-12 · 全量 CVSS-T test,两方向各 4,897 对
**结论:两个门都通过,五项指标无一劣于 Stage 1。零训练。**

---

## 结果

| 方向 | 指标 | Stage 1(3B) | **Qwen3 cascade** | 变化 |
|---|---|---:|---:|---:|
| Zh→En | **ASR-BLEU** | 24.82 | **27.68** | **+2.86** |
| Zh→En | Text-BLEU | 26.82 | **30.83** | **+4.01** |
| Zh→En | A.PCP | 2.925 | 2.930 | +0.005 |
| Zh→En | SIM-O | 0.825 | 0.826 | +0.001 |
| Zh→En | UTMOS | 3.454 | 3.462 | +0.008 |
| En→Zh | **ASR-BLEU** | 28.70 | **34.14** | **+5.44** |
| En→Zh | Text-BLEU | 30.45 | **38.40** | **+7.95** |
| En→Zh | A.PCP | 2.774 | 2.779 | +0.005 |
| En→Zh | SIM-O | 0.880 | 0.880 | −0.000 |
| En→Zh | UTMOS | 3.862 | 3.866 | +0.004 |

放回全表(en→zh / zh→en):

| 系统 | Text-BLEU | ASR-BLEU | A.PCP |
|---|---|---|---|
| 0.5B Phase3 | 24.15 / 15.33 | 23.58 / 12.05 | 2.79 / 2.91 |
| Ours Dec-only 3B | 31.59 / 24.90 | 27.12 / 23.41 | 2.96 / 2.75 |
| Omni Stage 1 | 30.45 / 26.82 | 28.70 / 24.82 | 2.774 / 2.925 |
| Omni Stage 2 | 29.91 / 23.77 | 23.09 / 18.73 | 2.782 / 2.850 |
| **Qwen3 cascade** | **38.40 / 30.83** | **34.14 / 27.68** | 2.779 / **2.930** |

## 唯一的变量是译文是谁写的

与 Stage 1 的 cascade 相比,只有一处不同:

| | Stage 1 cascade | Qwen3 cascade |
|---|---|---|
| **翻译器** | Qwen2.5-Omni-3B Thinker(冻结) | **Qwen3-Omni-30B-A3B Thinker(冻结)** |
| Talker | `talker_iter0040000.pt` | **同一个** |
| 码本手术 / `global_embed` | — | **同一套** |
| 声码器 | UniSS BiCodec | **同一个** |
| 解码 | temp 0.9 / top-k 40 / top-p 0.8 / rp 1.3 / 700 步 | **同一组** |
| 测试集 / 打分 | CVSS-T 4,897 / `text_metrics` + harness | **同一份同一套** |

所以增益可以干净地归因到基座。

## 音色通路未受影响,如预期

A.PCP 两方向 +0.005,SIM-O ±0.001。换的是 Thinker,而 UniSS 的 32 个
global token 是从源音频提取后**直通声码器**的(`bicodec_tokenizer.py:147`:
`global_tokens = tokens[:32]`),整条路径不经任何语言模型。实测证实了这一点。

## Talker 损耗变大了,这是 Phase D 的目标

| 方向 | Stage 1 | Qwen3 cascade |
|---|---:|---:|
| Zh→En | 2.00 | **3.15** |
| En→Zh | 1.75 | **4.26** |

Talker 从未见过 Qwen3 产出的文本分布(更长、用词不同),所以念得更吃力。
净收益仍为正,但这 1.15 / 2.51 是**留在桌上的钱** —— Phase D 让 Talker 适应
新 Thinker 后,理论上可以收回,ASR-BLEU 还能再往上。

## 配置

**Phase A(新环境 `qwen3omni`,transformers 5.19.0)**
- `Qwen3-Omni-30B-A3B-Instruct`,bf16,sdpa;**仅用 Thinker**,
  `talker`/`code2wav`/`token2wav` 已 `delattr`
- 贪心解码,`thinker_max_new_tokens=256`,batch 4,8 分片
- 峰值显存 65.7 GiB;提示词与打分协议**由 3B 的 Step 1 门 import 复用**,不是复制

**Phase B(老环境 `uniss-eval`,transformers 4.53.1)**
- Qwen2.5-Omni Talker + `talker_iter0040000.pt` + `retarget(head_scale=0.1)`
- temp 0.9 / top-k 40 / top-p 0.8 / rp 1.3 / max_steps 700,32 分片
- BiCodec 16 kHz

两个进程、两个环境,因为 Qwen3 要 transformers ≥4.57 而我们的 Talker 在 4.53 上。
这正是 cascade 的天然形态,也让整件事不需要一行训练代码。

## 这不是交付形态

cascade 必须先拿到完整译文才能发声,**流式下做不到**。它是决策用的测量手段。
真正的端到端是 Phase C/D/E:把 Qwen3 Thinker 和我们的 Talker 装进一个模型,
Talker 直读隐状态,然后重训。

**门已通过,继续。**

## 复现

```bash
# Phase A（新环境）
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
R=/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test
$PY -m experiments.uniss_qwen3_omni_v1.evaluation.qwen3_zeroshot_s2tt \
  --manifest $R/manifests/cvss_t_zh_en_test_pairs.jsonl --audio-root $R \
  --direction zh2en --shard 0 --num-shards 8 --output .../shard_0.json

# Phase B（老环境）
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
$PY -m experiments.uniss_qwen3_omni_v1.evaluation.speak_translations \
  --checkpoint checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt \
  --translations .../phaseA_zeroshot/zh2en/merged.json \
  --dev-parquet data/cvss_t/tokenized/cvss_t_zh_en_v1/cvss_t_zh_en_test.parquet \
  --shard 0 --num-shards 32 --output .../cmn_eng
```

原始结果:`eval_outputs/uniss_qwen3_omni_v1/{phaseA_zeroshot,phaseB_cascade}/`
