# Qwen3 级联 S2ST 部署文档（中英双向）

当前**效果最好、可直接部署**的中英↔英中语音到语音翻译方案。

读完这份文档你应该能从零跑通:输入一段中文或英文语音,输出另一种语言的语音,并复现下面的指标。

---

## 1. 这个系统能做什么

输入一段语音,输出另一种语言的语音,**保留源说话人的音色**。

在 CVSS-T test 全量(每方向 4,897 条)上的实测:

| 方向 | Text-BLEU | **ASR-BLEU** |
|---|---|---|
| 中→英 | 30.83 | **27.68** |
| 英→中 | 38.40 | **34.14** |

作为参照,同口径下 SimulS2ST-Omni (arXiv 2607.19810) 在 CVSS-T En→Zh 报的是 31.12。

**它不能做的事:** 这是离线(整句)方案,不是流式同传。必须等整句说完才开始翻译。流式版本仍在开发。

---

## 2. 架构

```
源音频 (16 kHz wav/mp3)
   │
   ├─【第一段】Qwen3-Omni-30B-A3B Thinker ── 零样本 S2TT，无需训练
   │                                         输出：译文文本
   │                        ↓
   │                  merged.json  ← 文本落盘，两段在此解耦
   │                        ↓
   └─【第二段】Qwen2.5-Omni-3B Thinker ──── 把译文套进 TTS 提示并编码
                           │                输出：隐状态
                           ▼
                  UniSS Talker (393M)  ── 读隐状态 → BiCodec 语义码
                           │             ↑ 同时读 32 个说话人 token
                           ▼
                  BiCodec 解码器 ──────── 码 → 波形
```

**为什么分两段跑:** Qwen3-Omni 需要 transformers 5.x,UniSS Talker 跑在 4.53。两个环境不能共存于一个进程,所以第一段写文本、第二段读文本。

**音色从哪来:** BiCodec 把音频编码成 `32 个说话人 token + 语义码`。说话人那 32 个 token **完全不经过任何语言模型**,直接从参考音频取、直接送进解码器。所以换骨干模型不影响音色能力。

---

## 3. 环境与资源

### 3.1 两个 conda 环境(已就绪,不要改)

| 环境 | 路径 | 关键版本 | 用途 |
|---|---|---|---|
| `qwen3omni` | `/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni` | python 3.12.13 / torch 2.6.0+cu124 / **transformers 5.19.0** | 第一段 |
| `uniss-eval` | `/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval` | python 3.12.13 / torch 2.6.0+cu124 / **transformers 4.53.1** | 第二段 + 评测 |

两个环境共用同一个 torch,`qwen3omni` 只是叠加了新版 transformers,额外占 161 MB。

### 3.2 权重

| 路径 | 大小 | 说明 |
|---|---|---|
| `pretrained_models/Qwen3-Omni-30B-A3B-Instruct` | 66 G | 第一段,**零样本,未微调** |
| `pretrained_models/Qwen2.5-Omni-3B` | 12 G | 第二段的 Thinker 外壳 |
| `pretrained_models/UniSS` | 5.4 G | BiCodec 编解码器 |
| `checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt` | 750 M | **Talker 权重,这是唯一训练过的部分** |

### 3.3 硬件

| 阶段 | 单卡显存 | 说明 |
|---|---|---|
| 第一段 | **约 66 GB** | 30B MoE，bf16。80 GB 卡可单卡跑 |
| 第二段 | 约 20 GB | 3B Thinker + 393M Talker + 声码器 |

全量 4,897 条 × 2 方向,8 卡约 6 小时。

---

## 4. 跑一遍完整流程

下面以 **中→英** 为例。跑另一方向只改三处:

- 第一段 `--direction zh2en` → `en2zh`(**manifest 不变** —— 同一个 `_pairs.jsonl` 服务两个方向,`--direction` 决定哪边当源)
- 第二段 `--dev-parquet` 换成 `cvss_t_en_zh_test.parquet`
- 输出目录 `cmn_eng` → `eng_cmn`

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
OUT=eval_outputs/my_deploy          # 换成你自己的目录
```

### 第一段:翻译(新环境)

```bash
PY3=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
R=/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test

mkdir -p $OUT/phaseA/zh2en
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY3 -m experiments.uniss_qwen3_omni_v1.evaluation.qwen3_zeroshot_s2tt \
    --manifest $R/manifests/cvss_t_zh_en_test_pairs.jsonl \
    --audio-root $R --direction zh2en \
    --shard $i --num-shards 8 \
    --output $OUT/phaseA/zh2en/shard_$i.json &
  sleep 60          # 错峰：八个分片同时加载 30B 会把 CPU 打满
done
wait
```

合并分片:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python - <<'PY'
import json, glob, os
d = os.environ["OUT"] + "/phaseA/zh2en"
rows = []
for f in sorted(glob.glob(d + "/shard_*.json")):
    rows += json.load(open(f))
json.dump(rows, open(d + "/merged.json", "w"), ensure_ascii=False)
print(f"{len(rows)} 条 -> {d}/merged.json")
PY
```

`merged.json` 是一个 list,每条形如:

```json
{"id": "common_voice_zh-CN_18524189.mp3",
 "audio": "/.../source_zh/common_voice_zh-CN_18524189.mp3.wav",
 "hypothesis": "Passengers can also use their mobile phones to scan the code to pay for the ride."}
```

### 第二段:发音(老环境)

```bash
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
T=data/cvss_t/tokenized/cvss_t_zh_en_v1

mkdir -p $OUT/phaseB/cmn_eng
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PYE -m experiments.uniss_qwen3_omni_v1.evaluation.speak_translations \
    --checkpoint checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt \
    --translations $OUT/phaseA/zh2en/merged.json \
    --dev-parquet $T/cvss_t_zh_en_test.parquet \
    --shard $i --num-shards 8 \
    --output $OUT/phaseB/cmn_eng &
done
wait
cat $OUT/phaseB/cmn_eng/spoken_*.jsonl > $OUT/phaseB/cmn_eng/results.jsonl
```

波形写在 `$OUT/phaseB/cmn_eng/wav/`,每条一个 wav。

> **parquet 是什么:** 它提供每条的参考译文(算 BLEU 用)和 `bicodec_global`(32 个说话人 token,决定输出音色)。如果你要跑**自己的音频**而不是 CVSS-T,见第 6 节。

### 评分

```bash
CUDA_VISIBLE_DEVICES=0 $PYE -m evaluation.asr_transcribe \
  --input $OUT/phaseB/cmn_eng/results.jsonl \
  --output $OUT/phaseB/cmn_eng/asr.jsonl \
  --whisper-model openai/whisper-large-v3 \
  --paraformer-model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --device cuda:0 --batch-size 16

$PYE -m evaluation.text_metrics --input $OUT/phaseB/cmn_eng/asr.jsonl \
  --output $OUT/phaseB/cmn_eng/asr_bleu.json \
  --hypothesis-field asr_text --reference-field reference_text
```

中文输出用 Paraformer 识别,英文用 Whisper,脚本按 `tgt_lang` 自动选。

应该得到 **27.68**(中→英)。对不上先看第 7 节。

---

## 5. 关键参数

`speak_translations.py` 的默认值就是产出上面指标的配置,**不要随意改**:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-steps` | 700 | 码的生成上限。实测中位 276,撞上限的不足 1% |
| `--temperature` | 见脚本默认 | 改了指标会变 |
| `--repetition-penalty` | 见脚本默认 | 同上 |
| `--speech-tokenizer` | `pretrained_models/UniSS` | BiCodec |

`qwen3_zeroshot_s2tt.py`:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-new-tokens` | 256 | 译文长度上限 |
| `--batch-size` | 4 | 30B 下再大容易 OOM |

---

## 6. 跑自己的音频

`speak_translations.py` 依赖 parquet 提供说话人 token 和参考文本。跑自有数据需要造一个同结构的 parquet。它读这 13 列,**少一列就会报错**:

```
id, translation, transcription, src_lang, tgt_lang,
bicodec_global, target_bicodec,
source_audio_path, reference_audio_path,
synthetic_source, synthetic_reference,
source_audio_duration_seconds, reference_audio_duration_seconds
```

真正影响输出的只有三列,其余只要类型对、可以填占位值:

| 列 | 内容 | 填不对会怎样 |
|---|---|---|
| `id` | 与 `merged.json` 的 `id` 对齐 | 对不上就匹配不到译文 |
| `tgt_lang` | `eng` 或 `cmn` | 决定评分时用 Whisper 还是 Paraformer |
| `bicodec_global` | 32 个 int,参考音频的说话人 token | **决定输出音色** |
| `translation` | 参考译文 | 只评测用;不评测可填空串 |

(输出 jsonl 里的 `reference_text` 字段是由 `translation` 派生的,parquet 里没有这个列。)

`bicodec_global` 的取法 —— 用 UniSS 编码一段参考音频,**取前 32 个 token**:

```python
from uniss import UniSSTokenizer
tk = UniSSTokenizer.from_pretrained("pretrained_models/UniSS", device="cuda:0")
tokens = tk.encode("reference_speaker.wav")
bicodec_global = tokens[:32]        # 说话人
# tokens[32:] 是语义码，这里用不到
```

想让输出用**源说话人的音色**,就拿源音频算这 32 个 token。

---

## 7. 常见问题

**第一段八个分片一起起,七个卡住不动**
不是磁盘,是 CPU。一个分片加载 30B 要占 24–32 核两到四分钟,八个一起上会把 192 核打满。命令里的 `sleep 60` 错峰就是为这个,**不要删**。

**`ptxas` PermissionError**
inductor 想编译。设 `TORCHDYNAMO_DISABLE=1`。

**第二段 OOM**
确认用的是 `uniss-eval` 而不是 `qwen3omni`,后者不会加载 3B Thinker 但会把 transformers 版本搞错。单卡 20 GB 够用。

**BLEU 只有一半**
几乎一定是评分协议不对 —— 直接调 sacrebleu 会绕过 `evaluation.text_metrics` 的归一化。**一律用 `evaluation.text_metrics`**,字段用 `--hypothesis-field asr_text --reference-field reference_text`。

**译文看着对但语音是乱码**
检查 Talker 权重路径。`talker_iter0040000.pt` 是 750 MB;如果加载的文件明显更大,很可能误拿了含优化器状态的 Megatron checkpoint。

**只想快速验证,不想跑 4,897 条**
两段都支持 `--limit`。注意 `--limit N` 取的是 **id 排序后的前 N 条**,不是随机抽样 —— 这批样本比全量**偏高约 3.3 BLEU**,只能用于自检,不能当作最终指标上报。

---

## 8. 可以改和不要改

**可以换的:**

- 第一段换任何能做 S2TT 的模型 —— 它只输出文本,Talker 不在乎谁写的。这是这个架构最大的好处。
- 说话人参考音频 —— 换 `bicodec_global` 即可,无需重训。

**不要动的:**

- Talker 权重。它是整条链路里唯一训练过的部分(Stage 1,40,000 步 × GBS 384 = 15.36M 样本),并且**专门为「读 TTS 提示的隐状态」训练**。换一种输入会显著掉分 —— 实测同一份权重读 S2ST 回复状态是 19.64,读 TTS 提示是 24.41。
- 第二段的 Qwen2.5-Omni-3B。Talker 的输入投影是对着它的隐状态拟合的。

---

## 9. 参考

- 指标出处与完整实验记录:`reports/uniss_qwen3_omni_v1/PHASE_AB_GATES.zh-CN.md`
- 原始输出(含全部 wav):`eval_outputs/uniss_qwen3_omni_v1/{phaseA_zeroshot,phaseB_cascade}/`
- 子集偏差说明:`reports/uniss_qwen3_omni_v1/PROBE_BASELINE_64.zh-CN.md`
- SimulS2ST-Omni 对比:https://arxiv.org/abs/2607.19810
