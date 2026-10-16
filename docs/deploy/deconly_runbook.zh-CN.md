# Decoder-only 路线:训练与评测操作手册

这条路线把 Talker 整个拿掉,**让 Qwen3 的 Thinker 自己吐 BiCodec 码**。

读完这份文档你应该能:起训、看懂日志、在指定步数做端到端探针、把分数放进和其他路线同一张表里比较。

> **状态:代码已写完并通过语法与批构造器检查,但从未跑过一步。**前向、显存、吞吐都还没有实测数据。第 6 节列出首次运行必须盯的东西。

---

## 1. 为什么有这条路线

当前主线是 Thinker–Talker:Thinker 出隐状态,独立的 Talker 读状态产码。它的问题是**两个模块之间的那道边界**:

```
同一个 Talker、同一套参数
  读【无音频前向】的状态（级联）     31.08
  读【有音频前向】的状态（端到端）     0.69
两族状态余弦 0.594，范数 191.8 vs 143.6
线性最小二乘把余弦只从 0.594 拉到 0.605（留出集）→ 没有稳定映射
```

四条训练路线试图跨过这道边界:

| 路线 | 配置 | 最好成绩 |
|---|---|---|
| v2 | s2st 75% + tts 25%,LR 6e-5 | 18.53 @2k |
| v3 | s2st 100%,LR 6e-5 | 20.00 @累计 5k |
| Phase F | 五任务 + Thinker LoRA,LR 1e-4 | 7.66,收敛 |
| a2s | 音频条件自监督 2:1,LR 1e-4 | 6.41 @1k |
| **phase_h** | **s2st 100%,LR 1e-4,跑长** | **24.96 @14k,仍在爬** |

**Decoder-only 从定义上消除这道边界** —— 听到音频的那个模型就是吐码的那个模型,没有第二个模块带着自己学过的输入分布。

代价是天花板更低。SimulS2ST-Omni(arXiv 2607.19810)实测:

```
offline CVSS-T En→Zh    Dec-only 27.12   vs   Thinker–Talker 31.12
RealSI 流式 m4          Dec-only 15.61   vs   Thinker–Talker 25.54
m1（最低延迟）           Dec-only 12.70 → 4.59，崩
```

**所以这是备选,不是主线。**phase_h 已经到 24.96 且仍在爬,目标 31.08;只有当它停在 25 以下够不到天花板时,这条路才值得投入。

---

## 2. 架构

```
input_ids    [<|audio_pad|>×N][指令][译文 token][说话人槽×32][码槽×T]
                                    ↑ lm_head             ↑ audio_head

inputs_embeds 由脚本自己造：
  音频位置    模型用 input_ids 定位占位符后自己 masked_scatter 进来
  文本位置    thinker 原生词表
  说话人位置  global_embed(speaker_ids)
  码位置      audio_embed(code_ids)
```

**没有 Talker,没有 projector。**

### 可训练参数

```
Qwen3 Thinker  31,723.7 M  ❄️ 冻结
  └ LoRA 第 32–47 层 q/k/v/o，r32/α64      4.5 M  ✓
audio_embed    Embedding(8193, 2048)      16.8 M  ✓
global_embed   Embedding(4096, 2048)       8.4 M  ✓
audio_head     Linear(2048, 8193)         16.8 M  ✓
                                        ─────────
                                          46.5 M  = 全部参数的 0.15%
```

### 损失

```
L = w_text · text_ce + w_code · code_ce          默认 1.0 / 1.0
```

两项都做 causal-LM 移位。监督单流的任务只贡献一项。

### 两个容易写错、这里已经处理的点

**① 为什么是两张表,不是 UniSS 那种共享词表**

Qwen3 的 152,064 行承载着整个项目依赖的零样本翻译(Text-BLEU 38.34 / 39.24)。扩表会把 8,193 个随机初始化的码行塞进文本竞争的那个 softmax,文本位置可能吐码、码位置可能吐字。两个头让**模态由位置决定**,而且码的 softmax 从 160,257 降到 8,193。

**② 码位 `start` 由 `start-1` 预测**

`start-1` 是最后一个说话人槽。从 `start` 读会让每一步都在训练「上一个码」—— 这是这类实现最常见的 off-by-one。

---

## 3. 文件清单

全部是新文件,**不改动任何既有脚本或实验**:

```
training/deconly_batch.py        单流批构造
training/deconly_model.py        模型 + 损失
training/pretrain_deconly.py     Megatron 入口
training/export_deconly.py       导出 audio_embed / audio_head / global_embed / LoRA
evaluation/deconly_e2e.py        端到端推理（两段 generate，共享 KV cache）
scripts/run_deconly.sh           训练启动
scripts/run_deconly_probe.sh     探针（8 卡分片 + ASR + BLEU）
```

---

## 4. 起训

### 4.1 环境

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
# 训练用 qwen3omni 环境（transformers 5.19.0），脚本内部已指定，无需手动 activate
```

### 4.2 第一段:热身(防 modality underfitting)

论文给 Dec-only 专门设了热身阶段 —— **不先热身,模型会只学一个模态、忽略另一个**。

```bash
RUN=deconly_warmup_$(date -u +%Y%m%dT%H%M%SZ)

UNISS_DECONLY_MIXTURE=warmup \
UNISS_S2ST_REPLY=translation \
SAVE_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN \
TB_DIR=$PWD/runs/uniss_qwen3_omni_v1/$RUN \
MBS=8 GBS=256 LR=1e-4 MIN_LR=2e-5 \
TRAIN_ITERS=3000 WARMUP_ITERS=200 \
SAVE_INTERVAL=1000 EVAL_INTERVAL=500 LOG_INTERVAL=20 \
MASTER_PORT=29751 \
bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly.sh \
  2>&1 | tee /tmp/deconly_warmup.log
```

`UNISS_DECONLY_MIXTURE=warmup` 给出 `asr 0.2 : s2tt 1.0 : s2st 1.5` —— asr 和 s2tt 撑住文本侧,s2st 教码侧。

### 4.3 第二段:正式训练

```bash
RUN2=deconly_main_$(date -u +%Y%m%dT%H%M%SZ)

UNISS_S2ST_REPLY=translation \
SAVE_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN2 \
TB_DIR=$PWD/runs/uniss_qwen3_omni_v1/$RUN2 \
MBS=8 GBS=256 LR=1e-4 MIN_LR=2e-5 \
TRAIN_ITERS=20000 WARMUP_ITERS=200 \
SAVE_INTERVAL=2000 EVAL_INTERVAL=500 LOG_INTERVAL=20 \
MASTER_PORT=29753 \
bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly.sh \
  2>&1 | tee /tmp/deconly_main.log
```

不设 `UNISS_DECONLY_MIXTURE` 时默认 `s2st 100%`。

> **热身到正式之间没有自动接力。**Megatron 会在 `SAVE_DIR` 存在 checkpoint 时自动 `--load`,所以如果想接着热身的权重训,把 `SAVE_DIR` 设成同一个目录并调大 `TRAIN_ITERS`,不要换目录。

### 4.4 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `UNISS_DECONLY_MIXTURE` | 不设 = `s2st 100%` | 设 `warmup` = `asr/s2tt/s2st = 0.2:1:1.5` |
| `UNISS_DECONLY_LORA_FROM` | `32` | LoRA 起始层。`all` = 全部 48 层(显存会大很多) |
| `MBS` | **8** | 序列比双流长约 5 倍,**不要照搬双流的 32** |
| `GBS` | 256 | |
| `LR` / `MIN_LR` | 1e-4 / 2e-5 | 论文 Stage 2 的值 |
| `TEXT_WEIGHT` / `CODE_WEIGHT` | 1.0 / 1.0 | 损失权重 |
| `SAVE_INTERVAL` | 2000 | **必须能整除所有探针点**,否则导出会失败 |

---

## 5. 探针

### 5.1 导出

```bash
R=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN2
IT=0002000                                     # 换成要测的步数，7 位补零

PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES="" \
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
  -m experiments.uniss_qwen3_omni_v1.training.export_deconly \
  --checkpoint-dir $R/iter_$IT \
  --output $R/exported/deconly_iter$IT.pt
```

导出会在两种情况下**直接报错而不是产出一个看着正常的文件**:

- 缺少 `audio_embed` / `global_embed` / `audio_head` 任何一个
- **LoRA 的 B 矩阵全为零** —— B 是零初始化的,全零意味着这个 checkpoint 根本没训练过

### 5.2 跑探针

```bash
CKPT=$R/exported/deconly_iter$IT.pt \
OUT=$PWD/eval_outputs/uniss_qwen3_omni_v1/$RUN2/deconly_iter$IT \
LIMIT=64 \
bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly_probe.sh
```

**需要 8 张卡,训练必须先停。**约 30–40 分钟(分片按 90 秒错峰启动,不要删那个 `sleep`)。

输出:

```
  ASR-BLEU tts:cmn->eng: XX.XX (n=64)
  ASR-BLEU tts:eng->cmn: XX.XX (n=64)
  Text-BLEU tts:cmn->eng: XX.XX (n=64)
  Text-BLEU tts:eng->cmn: XX.XX (n=64)
DECONLY_PROBE_DONE
```

### 5.3 建议的探针点

```
iter   500    最早的健康检查：Text-BLEU 有没有塌
iter 2,000    与所有其他路线的公平对照点
iter 4,000
iter 8,000
iter 20,000
```

---

## 6. 怎么读结果

### 6.1 Text-BLEU 是先决条件,不是成绩

```
基线 38.34 / 39.24     ← Thinker 冻结时的水平
```

Dec-only 给 Thinker 上了 LoRA,所以**翻译能力可能被破坏**。判据:

```
Text-BLEU ≈ 38 / 39    → 正常，继续看 ASR-BLEU
明显下滑                → LoRA 在毁翻译，降 LORA_FROM 的层数或降 LR
塌到个位数              → 模型在复读。检查译文里是不是同一句重复到 token 上限
```

> 这个失败模式真实发生过:Phase F iter 1,000 时 Text-BLEU 从 38.34 掉到 5.25,输出是「Liu Yuzi. Liu Yuzi. Liu Yuzi.」。原因是训练目标里没有结束符 —— `deconly_batch.py` 已经把 `<|im_end|>` 加进监督,但这是新代码,**第一次跑务必确认**。

### 6.2 ASR-BLEU 的对照表

同一批 64 条/方向,同一个声码器和 ASR:

| 配置 | zh→en | en→zh | mean |
|---|---|---|---|
| gold 码 → 声码器 → ASR | 80.69 | 72.80 | 76.75 |
| 现有级联(双模型,两次前向) | 32.22 | 36.21 | **34.21** |
| 级联(单 Qwen3,两次前向) | 28.54 | 33.63 | **31.08** |
| Thinker–Talker 端到端 @14k | 23.57 | 26.34 | **24.96** |
| Thinker–Talker 起点 | 0.32 | 1.06 | 0.69 |
| **Dec-only** | ? | ? | **论文参考 27.12** |

> **注意:**这 64 条是 `rows[:64]`(id 排序取前 64),**比全量 4,897 条偏高 +3.30**。跨实验比较一律用上表的值;对外报数必须跑全量。

### 6.3 判据

```
Dec-only 跑到 8,000 步  > 25   有竞争力，继续
                        20–25  和 Thinker–Talker 同级，但天花板更低 → 不值得
                        < 20   这条路也不行 → 回到主线
```

---

## 7. 首次运行必须盯的东西

**代码从未跑过一步。**以下都没有实测数据:

| 项 | 风险 | 怎么查 |
|---|---|---|
| **显存** | 序列约 413 位(双流约 81),MBS 8 可能仍 OOM | 看前 20 步 `nvidia-smi`;OOM 就降到 4 |
| **吞吐** | 估计 10–15 样本/秒(双流是 155) | 日志里的 `elapsed time per iteration` |
| **前向** | `inputs_embeds` + `input_features` 同传的音频合并路径**只做过源码核对,没跑过** | 第一步不报错即通过 |
| **码损失量级** | `code_ce` 起步应在 8–9(ln 8193 ≈ 9.0) | 低于 5 说明标签对齐有问题 |
| **modality underfitting** | 一个头学、另一个被忽略 | `text_ce` 和 `code_ce` 要**同时**下降 |

**建议第一次只跑 50 步就停**,确认上面五项,再放长线。

```bash
TRAIN_ITERS=50 SAVE_INTERVAL=50 ... bash .../run_deconly.sh
```

---

## 8. 不要做的事

- **不要在脚本正在被执行时修改它。**bash 按字节偏移边读边执行,中途插入几行会让它读错位置。这个项目已经因此废掉过两次探针(`syntax error near unexpected token 'done'`)。
- **不要 `rm` 任何输出目录。**探针脚本遇到非空目录会报路径并退出,清理由人来做。
- **不要让 `PROBE_AT` 和 `SAVE_INTERVAL` 对不上。**探针会去导出一个从没写过的 `iter_XXXXXXX`,然后失败。
- **不要照搬双流的 MBS 32。**序列长 5 倍。

---

## 9. 参考

- 架构与实测对照:`reports/uniss_qwen3_omni_v1/`
- 子集偏差说明:`reports/uniss_qwen3_omni_v1/PROBE_BASELINE_64.zh-CN.md`
- 当前可部署的方案:`docs/deploy/qwen3_cascade_s2st_deploy.zh-CN.md`
- SimulS2ST-Omni:https://arxiv.org/abs/2607.19810 · 仓库 https://github.com/hasaki321/SimulS2ST-Omni
