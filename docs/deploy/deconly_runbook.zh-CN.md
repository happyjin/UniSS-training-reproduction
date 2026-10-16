# Decoder-only 路线:训练与评测操作手册

这条路线把 Talker 整个拿掉,**让 Qwen3 的 Thinker 自己吐 BiCodec 码**。

读完这份文档你应该能:起训、看懂日志、在指定步数做端到端探针、把分数放进和其他路线同一张表里比较。

> **状态:前向和反向已在 CPU 上验证通过;显存、吞吐、Megatron 集成和推理脚本仍未验证。**
> 第 7 节列出首次 GPU 运行必须盯的东西,以及为什么仍然要先跑 50 步冒烟。

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

默认路线(`UNISS_DECONLY_LORA_FROM=none`)**完全不动 Thinker**:

```
Qwen3 Thinker  31,723.7 M  ❄️ 冻结，与发布版逐比特一致
音频塔                     ❄️ 冻结
lm_head                    ❄️ 冻结
global_embed   Embedding(4096, 2048)       8.4 M  ✓  说话人槽的输入
audio_embed    Embedding(8193, 2048)      16.8 M  ✓  码的输入
audio_head     Linear(2048, 8193)         16.8 M  ✓  码的输出
                                        ─────────
                                          42.0 M  = 全部参数的 0.13%
```

加 `UNISS_DECONLY_LORA_FROM=32` 会额外训 LoRA 第 32–47 层 q/k/v/o(r32/α64,4.5 M,合计 46.5 M),代价见 4.2。

### 损失

```
L = w_text · text_ce + w_code · code_ce
```

**默认路线下 `w_text` 会被自动置零**,所以实际只有 `code_ce` 一项。原因:`lm_head` 和它下面每一层都冻结,`text_ce` 背后一个可训练参数都没有,梯度恒为零。该项仍然照算照记 —— 一切冻结它就不会变,是免费的健康指示灯。

带 LoRA 时两项都有效,`w_text` 的作用是拦住 LoRA 为了码把翻译改坏。两项都做 causal-LM 移位。

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
scripts/orchestrate_deconly.sh   编排器：停训 → 导出 → 探针 → 续训，全自动
```

---

## 4. 起训

### 4.1 环境

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
# 训练用 qwen3omni 环境（transformers 5.19.0），脚本内部已指定，无需手动 activate

# 默认路线：Thinker 完全不动，只训序列两端的 42.0M
export UNISS_DECONLY_LORA_FROM=none
export UNISS_DECONLY_GLOBAL_INIT=$PWD/checkpoints/uniss_qwen3_omni_v1/phase_d_20261012T083329Z/exported/phase_d_iter0014000.pt
export UNISS_S2ST_REPLY=translation
```

**这三行一次 export，后面每条命令都靠它。**换终端要重新 export。编排器那条命令会把它们显式写进 `env`，因为 `setsid` 之后的环境要自己带。

### 4.2 两种模式:默认不动 Thinker

| | **B. 冻结 Thinker**(本文档默认) | **A. 带 LoRA**(备选) |
|---|---|---|
| 开关 | `UNISS_DECONLY_LORA_FROM=none` | 不设,或设成 `32` |
| 可训练 | **42.0M,全在序列两端** | 46.5M(含 LoRA 4.5M) |
| Thinker | **与发布版逐比特一致** | 第 32–47 层 q/k/v/o 被改 |
| `text_ce` | **自动置零** —— 背后没有可训练参数 | 有效,梯度流进 LoRA |
| 翻译 | **可证明不变**(38.34 / 39.24) | 可能被改坏,靠 `text_ce` 拦 |
| 显存 | **保留 48 层** —— 最早的可训练参数 `audio_embed` 喂第 0 层 | 保留 16 层激活 |

> **B 的显存可能更高,不是更低。**省掉 `lm_head` 约 570 MB 和 5% FLOPs,多出 32 层激活边界。谁大没测过。

B 回答的是一个更窄、也更干净的问题:**冻结的骨干,光靠一张新的输入表和一个新的输出头,能不能吐码。**

`UNISS_DECONLY_GLOBAL_INIT` 让 `global_embed` 从双流训好的那张表起步,而不是从噪声。phase_d 练了 14,000 步,形状正好 `(4096, 2048)`。形状对不上会直接报错,不会静默广播。

**导出时要加 `--no-lora`**(编排器会自动加)。没有它的话,空 adapter 会被当成「这个 checkpoint 根本没训练」—— 两者长得一样。

> **默认路线不能配 `UNISS_DECONLY_MIXTURE=warmup`。**asr 和 s2tt 只监督文本,而 `text_ce` 已被置零,这类 batch 里没有任何可训练参数进入计算图,backward 会直接报 `does not require grad` —— 2.7 份里有 1.2 份,44% 的 batch 崩。模型启动时会拦住这个组合并说明原因。热身只在带 LoRA 时有意义(见 4.3)。

#### 完整流程,三条命令

4.1 的三行 export 要先执行。

**第 1 步 —— 50 步冒烟。必须先跑这个,不要跳。**代码从未在 GPU 上跑过一步,显存和吞吐都没有实测数据。

```bash
RUN=deconly_smoke_$(date -u +%Y%m%dT%H%M%SZ)
SAVE_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN \
TB_DIR=$PWD/runs/uniss_qwen3_omni_v1/$RUN \
MBS=8 GBS=64 LR=1e-4 MIN_LR=1e-4 \
TRAIN_ITERS=50 WARMUP_ITERS=10 \
SAVE_INTERVAL=50 EVAL_INTERVAL=10000 LOG_INTERVAL=5 \
MASTER_PORT=29751 \
bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly.sh 2>&1 | tee /tmp/deconly_smoke.log
```

前 20 步盯 `nvidia-smi`。OOM 就把 `MBS` 降到 4,再不行降到 2。

**第 2 步 —— 过拟合 200 条。**这一步回答「管线能不能学」,见 4.2b。

**第 3 步 —— 正式训练 + 自动探针。**

```bash
RUN=deconly_main_$(date -u +%Y%m%dT%H%M%SZ)
setsid nohup env \
  UNISS_DECONLY_LORA_FROM=none \
  UNISS_DECONLY_GLOBAL_INIT=$UNISS_DECONLY_GLOBAL_INIT \
  UNISS_S2ST_REPLY=translation \
  RUN_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN \
  MBS=8 GBS=256 LR=1e-4 MIN_LR=2e-5 \
  TRAIN_ITERS=20000 PROBE_AT="2000 4000 8000 14000" \
  MAX_ITERS=30000 EXTEND_BY=5000 GAIN_STOP=1.0 PROBE_LIMIT=64 \
  SAVE_INTERVAL=2000 MASTER_PORT=29753 \
  bash experiments/uniss_qwen3_omni_v1/scripts/orchestrate_deconly.sh \
  > /tmp/deconly_orch.log 2>&1 < /dev/null & disown
```

探针点刻意对齐主线双流的 2000 / 4000 / 14000,方便直接比。分数在 `$RUN_DIR/orchestrator/probes.tsv`。

> 正式训练前先用 `DRY_RUN=1` 校验一次配置(把 `setsid nohup env` 换成 `env`、加 `DRY_RUN=1`、去掉末尾的重定向和 `& disown`)。

### 4.2b 最快的验证:过拟合 200 条

比 50 步冒烟有信息量得多。冒烟只说明不崩,这个说明**能不能学**。

```bash
RUN=deconly_overfit_$(date -u +%Y%m%dT%H%M%SZ)

AUDIO_LIMIT=200 \
UNISS_S2ST_REPLY=translation \
SAVE_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN \
TB_DIR=$PWD/runs/uniss_qwen3_omni_v1/$RUN \
MBS=8 GBS=64 LR=1e-4 MIN_LR=1e-4 \
TRAIN_ITERS=500 WARMUP_ITERS=20 \
SAVE_INTERVAL=500 EVAL_INTERVAL=10000 LOG_INTERVAL=10 \
MASTER_PORT=29757 \
bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly.sh 2>&1 | tee /tmp/deconly_overfit.log
```

200 条 / GBS 64 ≈ 3 个 batch 一轮,500 步 ≈ 160 轮,足够背下来。只看 `code_ce`:

```
ln(8193) = 9.01     起点
掉到 1 以下          → 管线接对了，模型有能力学码
卡在 6–7 不动        → 结构有问题。去查三处：
                       code_start 的 start-1 错位
                       audio_embed 的初始化尺度
                       code_labels 的对齐
```

然后**在同样这 200 条上跑探针**(`LIMIT=64`,`--dev-parquet` 指向同一份)。训练集上 ASR-BLEU 高 → 训练、导出、推理、声码器整条链都正确,剩下的只是泛化。

> **先用默认路线(冻结 Thinker)跑这个测试** —— 测的就是你实际要训的东西。
>
> 如果 `code_ce` 卡住不降,**再用 `UNISS_DECONLY_LORA_FROM=24` 重跑一次**来区分两种原因:
>
> ```
> 加了 LoRA 就能过拟合   → 管线是对的，是冻结骨干容量不够 → 这条路线走不通，回到带 LoRA
> 加了 LoRA 还是卡住     → 接线错了 → 查 start-1 错位 / audio_embed 尺度 / code_labels 对齐
> ```

### 4.3 热身(**只在带 LoRA 时适用**)

> **默认路线(冻结 Thinker)跳过这一节。**热身是为了防止两个头一个学一个被忽略,而默认路线只有一个头在训(`text_ce` 已置零),没有可偏向的对象。配上去会被启动时的守卫拦住。本节只在你改用带 LoRA 时才相关。

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

### 4.4 第二段:正式训练

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

### 4.5 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `UNISS_DECONLY_MIXTURE` | 不设 = `s2st 100%` | 设 `warmup` = `asr/s2tt/s2st = 0.2:1:1.5` |
| `UNISS_DECONLY_LORA_FROM` | 代码默认 `32`,**本文档 4.1 export 成 `none`** | `none` = 完全不动 Thinker(见 4.2);数字 = LoRA 起始层;`all` = 全部 48 层 |
| `UNISS_DECONLY_GLOBAL_INIT` | 不设 = 随机 | `global_embed` 从双流导出暖启动,需 `(4096, 2048)` |
| `AUDIO_LIMIT` | `0` = 全量 | 限制样本数,过拟合测试用 |
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
  --output $R/exported/deconly_iter$IT.pt --no-lora
```

**默认路线必须带 `--no-lora`。**没有它,空 adapter 会被当成「这个 checkpoint 根本没训练」而拒绝 —— 两者长得一样。带 LoRA 训的则**不能**加这个开关,加了会报错。编排器按 `UNISS_DECONLY_LORA_FROM` 自动决定。

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

### 5.3 全自动:用编排器

不想手动停训/导出/探针的话,直接用编排器,它会在每个探针点停训 → 导出 → 跑探针 → 续训,**每个测量点零步损失**:

```bash
RUN2=deconly_main_$(date -u +%Y%m%dT%H%M%SZ)

setsid nohup env \
  UNISS_DECONLY_LORA_FROM=none \
  UNISS_DECONLY_GLOBAL_INIT=$UNISS_DECONLY_GLOBAL_INIT \
  UNISS_S2ST_REPLY=translation \
  RUN_DIR=$PWD/checkpoints/uniss_qwen3_omni_v1/$RUN2 \
  MBS=8 GBS=256 LR=1e-4 MIN_LR=2e-5 \
  TRAIN_ITERS=20000 PROBE_AT="500 2000 4000 8000" \
  MAX_ITERS=30000 EXTEND_BY=5000 GAIN_STOP=1.0 PROBE_LIMIT=64 \
  SAVE_INTERVAL=500 MASTER_PORT=29753 \
  bash experiments/uniss_qwen3_omni_v1/scripts/orchestrate_deconly.sh \
  > /tmp/deconly_orch.log 2>&1 < /dev/null & disown
```

分数会追加到 `$RUN_DIR/orchestrator/probes.tsv`,日志在 `orchestrator/orchestrator.log`。

**先用 `DRY_RUN=1` 校验配置。**把上面命令里的 `setsid nohup env` 换成 `env`、加上 `DRY_RUN=1`、去掉末尾的 `> ... &  disown`,它会检查完配置就退出,不启动任何训练:

```bash
DRY_RUN=1 RUN_DIR=... PROBE_AT="500 2000 4000 8000" SAVE_INTERVAL=500 \
  bash experiments/uniss_qwen3_omni_v1/scripts/orchestrate_deconly.sh
# -> DRY_RUN=1，不启动。配置检查通过。
```

编排器在启动前会挡住两类配置错误,两类都踩过、且都不会自己说清楚:

- **`PROBE_AT` 的值不能被 `SAVE_INTERVAL` 整除** —— 导出会去找一个从没写过的 `iter_XXXXXXX`
- **首个探针点 ≤ `WARMUP_ITERS`** —— Megatron 的 `assert lr_warmup_steps < lr_decay_steps` 会在重启循环里反复触发,编排器会一直重试

> **不要为了「看看它会干什么」去跑编排器。**它内部是 `setsid nohup ... & disown`,`timeout` 或 Ctrl-C 杀掉的只是编排器本身,已经起来的 8 卡训练会继续活着。要看配置就用 `DRY_RUN=1`。

### 5.4 建议的探针点

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

## 7. 已验证 / 未验证

### 已在 CPU 上验证通过

```
前向          total 14.48   text_ce 4.06   code_ce 10.41（随机参考 ln 8193 = 9.01）
反向          跑通，40 秒
梯度路由      audio_embed / global_embed / audio_head 全部非零
              LoRA .b 64/64 非零，.a 0/64 —— 零初始化 B 的预期行为，全部 64 层都在图里
冻结参数      0 个拿到梯度
两种模式      带 LoRA 和冻结 Thinker 都验过。冻结模式下 thinker 可训练参数 0 个，
              可训练合计 42.0M/31,761M，text_weight 自动置零，total == code_ce
global_embed  从 phase_d 暖启动，形状 (4096, 2048) 对齐
音频合并      inputs_embeds + input_features 同传，模型仍按 input_ids 定位并 scatter 音频
索引算术      code_start 正确，无越界
推理全链路    load_pieces → 文本 generate → 复用 KV cache → 说话人槽 → 码循环 → 声码器
              用随机权重的合成 checkpoint 跑通 1 条样本，4 个码解码出 0.080 秒音频
              （4 × 20 ms，BiCodec 的 50 Hz 对得上），译文正确、无报错
```

> 合成 checkpoint 的码是噪声,这一项验证的是路径不是质量。

### 仍未验证 —— 第一次 GPU 运行要盯这些

| 项 | 风险 | 怎么查 |
|---|---|---|
| **显存** | 序列约 400 位(双流约 81),MBS 8 可能 OOM | 前 20 步看 `nvidia-smi`;OOM 降到 4 |
| **吞吐** | 估计 10–15 样本/秒(双流 155) | 日志 `elapsed time per iteration` |
| **Megatron 集成** | `model_provider` / `forward_step` / 数据集 provider 整条链没跑过。CPU 测试用了假 config 直接构造模型,绕开了这一层 | 第一步不报错即通过 |
| **8 卡并发探针** | 单条在 CPU 上通过，8 个分片同时抢卡没试过 | 第一次探针；分片日志在 `$OUT/<方向>/logs/` |
| **modality underfitting** | 一个头学、另一个被忽略 | `text_ce` 和 `code_ce` 要**同时**下降 |

**所以第一次仍然只跑 50 步:**

```bash
TRAIN_ITERS=50 SAVE_INTERVAL=50 ... bash .../run_deconly.sh
```

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
