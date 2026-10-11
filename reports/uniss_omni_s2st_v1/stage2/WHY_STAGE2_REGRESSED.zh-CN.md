# Stage 2 为什么不如 Stage 1

**日期** 2026-10-11 · 模型 `stage2_formal_20261011T043541Z` iter 6000
**结论** Talker 权重是好的。退步来自**联合条件化路径欠训 28 倍**,不是架构问题,
不需要更换 Talker 或 codec。

---

## 一、事实:完整 CVSS-T 对比

两个方向各 4,897 对,与 Stage 1 同一套打分代码。

| 方向 | 指标 | Stage 1 | Stage 2 | 变化 |
|---|---|---:|---:|---:|
| Zh→En | **ASR-BLEU** | 24.82 | **18.73** | **−6.09** |
| Zh→En | Text-BLEU | 26.82 | 23.77 | −3.05 |
| Zh→En | A.PCP | 2.925 | 2.850 | −0.075 |
| Zh→En | SIM-O | 0.825 | 0.816 | −0.009 |
| Zh→En | UTMOS | 3.454 | 3.481 | +0.027 |
| En→Zh | **ASR-BLEU** | 28.70 | **23.09** | **−5.61** |
| En→Zh | Text-BLEU | 30.45 | 29.91 | −0.54 |
| En→Zh | A.PCP | 2.774 | 2.782 | +0.008 |
| En→Zh | SIM-O | 0.880 | 0.882 | +0.002 |
| En→Zh | UTMOS | 3.862 | 3.847 | −0.015 |

**音色通路完好。** SIM-O 与 A.PCP 两个方向都持平或微升 —— UniSS 的 32 个
global token 绕过语言模型直通声码器,在联合训练后没有被破坏。退步**只在可懂度**。

换个参照系:对项目真正要替换的 0.5B Phase3(Text 24.15/15.33,ASR 23.58/12.05,
A.PCP 2.79/2.91,顺序 en→zh / zh→en),Stage 2 的翻译 zh→en **+8.44**,
语音 zh→en **+6.68**,en→zh 语音基本打平。**Stage 2 没有输给替换目标,
它输给的是 Stage 1 的 cascade。**

## 二、瓶颈定位

`Text-BLEU − ASR-BLEU` 这个差值与翻译质量无关,只衡量 Talker 把文本变成语音
时丢掉多少。

| 方向 | Stage 1 | Stage 2 |
|---|---:|---:|
| Zh→En | 2.00 | **5.04** |
| En→Zh | 1.75 | **6.82** |

En→Zh 的译文几乎没动(−0.54),语音却掉了 5.61 —— **这个方向的退步几乎全部
是 Talker**。两方向损耗增量相近,指向一个与语言无关的共因。

## 三、三个假说,三次裁决

### 假说 1:句首被源语转写污染 —— **已否**

`build_text_stream` 从 code step 0 起逐一铺开 reply 的 token。Stage 2 的 reply 是
`转写\n译文`,目标码却是译文的音频,于是每条语句的前约 16 步(50 Hz 下 0.32 秒)
在读源语言转写。

加 `--talker-text translation` 让 Talker 改从译文起读,随机 1,000 条实测:

| 设置 | Zh→En ASR | En→Zh ASR |
|---|---:|---:|
| 从转写起读(训练口径) | 19.64 | 23.49 |
| 从译文起读 | **16.35** | **20.87** |

**反而更差**(−3.29 / −2.62)。Talker 已学会利用转写那一段,拿掉它是把输入
推出训练分布。假说否定。

局限:这是不重训改推理,证明不了"按译文重新对齐训练"不会更好;但它否定了
"句首被污染所以 Talker 变差"这个机制。

### 假说 2:译文退步来自提示词还是 LoRA —— **测不出来,设计失败**

2×2(适配器开/关 × S2TT/S2ST 提示),同一批随机 1,000 条:

| | S2TT 提示 | S2ST 提示 |
|---|---:|---:|
| Zh→En 适配器关(=Stage 1) | **28.17** | 13.30 |
| Zh→En 适配器开(=Stage 2) | 5.32 | **24.76** |
| En→Zh 适配器关(=Stage 1) | **30.71** | 5.57 |
| En→Zh 适配器开(=Stage 2) | 4.26 | **30.24** |

对角线精确复现(24.76 / 30.24 与独立跑出的 Stage 2 数字一致),说明测量可信。
但**两个交叉格无效**:LoRA 是连同 S2ST 提示一起训的,给它 S2TT 提示它就跑飞
(平均输出 336 字符 vs 72);冻结模型没学过两行格式,`split_reply` 取错行。
那四个数测的是格式错配,不是"提示词的代价"或"LoRA 的代价"。

要真拆开需另训一个 S2TT 提示的 LoRA。为解释 −3.41 的译文退步再跑一轮训练,
不划算 —— 主要矛盾不在这里。

### 假说 3:Talker 权重坏了,还是条件化路径变了 —— **成立:是路径**

**同一份 Stage 2 Talker 权重**,只换喂给它的东西。随机 1,000 条:

| 配置 | 方向 | Text-BLEU | ASR-BLEU | **Talker 损耗** | A.PCP |
|---|---|---:|---:|---:|---:|
| 联合路径(= Stage 2) | Zh→En | 24.76 | 19.64 | **5.12** | 2.857 |
| **cascade 路径** | Zh→En | 28.17 | **24.41** | **3.75** | **2.917** |
| 联合路径(= Stage 2) | En→Zh | 30.24 | 23.49 | **6.75** | 2.787 |
| **cascade 路径** | En→Zh | 30.71 | **27.89** | **2.82** | 2.769 |
| *Stage 1 基线* | *Zh→En / En→Zh* | *26.82 / 30.45* | *24.82 / 28.70* | *2.00 / 1.75* | *2.925 / 2.774* |

Talker 损耗 **6.75 → 2.82**(En→Zh),权重一个字节没变。ASR-BLEU 基本回到
Stage 1 水平(24.41 vs 24.82,27.89 vs 28.70),A.PCP 也回到 2.917(vs 2.925)。

**Talker 权重没坏。退步在条件化路径。**

残留的次要项:同走 cascade 路径,Stage 2 Talker 的损耗 3.75 / 2.82 仍高于
Stage 1 的 2.00 / 1.75(+1.75 / +1.07)。Stage 2 的全量 SFT 确实让 Talker 在
原路径上轻微退步,但只占总退步的小部分。

## 四、根因:联合条件化欠训 28 倍

Talker 在两条路径上读到的东西完全不同:

- **cascade**:一句干净 TTS 提示的隐状态,里面只有要念的那句话
- **联合**:`转写\n译文` 在**源音频上下文中**产生的隐状态,纠缠着源音频、
  转写、任务指令

后者信息密度高得多,也难学得多。而训练量是:

| | 样本数 | 条件化 |
|---|---:|---|
| Stage 1 | 40,000 步 × 全局批 384 = **15,360,000** | 干净 TTS(简单) |
| Stage 2 的 S2ST 份额 | 6,000 × 256 × 35.7% = **549,000** | 联合(困难) |

**1/28 的训练量,更难的任务。** Talker 在新条件化上严重欠训,这就是那
5–6 BLEU 的来源。

## 五、为什么不能简单改用 cascade

Stage 3 的流式要求 Talker **增量**读取 Thinker 的隐状态 —— 那正是联合路径。
cascade 必须先拿到完整译文才能重新编码,流式下做不到。

所以联合路径必须训好,这不是可绕开的工程选择。

## 六、归因汇总

| 现象 | 原因 | 量级 |
|---|---|---|
| ASR-BLEU −5.6 ~ −6.1 | **联合条件化欠训** | 主因,En→Zh 约 3.9 分 |
| 同路径下 Talker 略差 | Stage 2 全量 SFT 的副作用 | 次要,1.1 ~ 1.8 分 |
| Text-BLEU −0.5 ~ −3.4 | LoRA + S2ST 提示(未能拆开) | Zh→En 明显,En→Zh 可忽略 |

**不需要更换 Talker 或 codec(CosyVoice 等)。** 同一套权重在合适条件化下
已能跑出 24.41 / 27.89,架构没有问题。

## 七、方法上的两个教训

**1. 有偏子集。** checkpoint 选择用了 `rows[:200]` —— 按 id 排序后的前 200 条,
不是随机抽样。同一份全量结果里这 200 条给 Text-BLEU 27.06,全量 23.77,偏高 3.29。
我据此把 5500 的下降判为噪声、把 6000 判为最佳,并以"指示性"口径上报。
现已改用固定种子随机抽样:随机 1,000 条给 19.64 / 24.76,对全量 18.73 / 23.77,
偏差 0.3–1.0。

更糟的是,我**用这份有偏数据否定了假说 1**(当时子集显示两方向不对称 3.61 vs 8.46,
全量其实对称)。假说 1 最终仍被否,但那次否定的理由是错的。

**2. 打分口径。** 消融首版直接调 sacrebleu,没做项目 `evaluation.text_metrics`
的归一化(参考是全小写无标点),得到的数只有正确值的一半,与项目所有其他数字
不可比。任何新打分都必须走同一个入口。

## 八、复现

```bash
# 全量 Table 1
CKPT=.../exported/stage2_iter0006000.pt OUT=.../table1_stage2_iter6000 LIMIT=0 \
  GPUS="0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7" \
  bash experiments/uniss_omni_s2st_v1/scripts/run_stage2_table1.sh

# 假说 1:句首对齐
TALKER_TEXT=translation SUBSET=data/cvss_t/manifests/cvss_t_zh_en_v1/random1000_both.jsonl \
  CKPT=... OUT=.../onset_translation bash .../run_stage2_table1.sh

# 假说 2:2×2 消融
python -m experiments.uniss_omni_s2st_v1.evaluation.prompt_lora_ablation \
  --checkpoint ... --dev-parquet ... --subset ... --output ...

# 假说 3:cascade 路径（决定性）
python -m experiments.uniss_omni_s2st_v1.evaluation.cascade_s2st \
  --checkpoint .../stage2_iter0006000.pt --subset ... --output .../cascade_stage2talker
```

原始结果:`eval_outputs/uniss_omni_s2st_v1/{table1_stage2_iter6000,onset_full,
onset_translation,cascade_stage2talker,ablation}/`
