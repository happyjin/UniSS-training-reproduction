# 当前最佳模型:与 SimulS2ST-Omni Dec-only 的对比、checkpoint 路径、推理方法

## 一、与论文 Dec-only 基线的详细对比

论文的 Dec-only 是**与我们同类的架构** —— 统一解码器直接预测语义码,
论文原文:*"appending 16,384 code tokens to the vocabulary, single autoregressive head"*,
而且*"leverages its entire **3B** backbone for code generation"*。
**我们是 0.52B,参数量约为它的 1/6。**

数据来源:论文 Table 5(RealSI sentence-level,ASR-BLEU,greedy 解码),
`data/external/simuls2st_omni_demo/paper.txt`。

### ASR-BLEU(RealSI)

| 系统 | 参数 | 读步 | En→Zh | Zh→En |
|---|---|---|---:|---:|
| **我们(步骤3,k4=640 ms)** | **0.52B** | 0.64 s | **18.75** | **13.71** |
| **我们(步骤3,k25=4 s)**¹ | 0.52B | 4 s | — | — |
| 论文 Dec-only Full | 3B | m1 = 1 s | 12.70 | 10.88 |
| 论文 Dec-only Full | 3B | m2 = 2 s | 14.98 | 12.94 |
| 论文 Dec-only Full | 3B | m3 = 3 s | 15.61 | 14.01 |
| 论文 Dec-only Full | 3B | m4 = 4 s | 17.44 | 15.45 |
| 论文 Dec-only 10% 数据 | 3B | m1 / m4 | 13.02 / 16.52 | 10.29 / 15.12 |
| 论文 Talker(双流,他们的主结果) | 3B | m1 / m4 | 17.30 / 24.04 | 12.49 / 18.78 |

¹ 步骤 3 的 k25 未跑;步骤 2 的 k25 是 18.92 / 14.65,可作下界参考。

### 逐档对齐后的读法

| 对比 | En→Zh | Zh→En |
|---|---|---|
| 我们 0.64 s **对** 他们 m1(1 s) | **18.75 对 12.70(+6.05)** | **13.71 对 10.88(+2.83)** |
| 我们 0.64 s **对** 他们 m2(2 s) | **18.75 对 14.98(+3.77)** | 13.71 对 12.94(+0.77) |
| 我们 0.64 s **对** 他们 m3(3 s) | **18.75 对 15.61(+3.14)** | 13.71 对 14.01(**−0.30**) |
| 我们 0.64 s **对** 他们 m4(4 s) | **18.75 对 17.44(+1.31)** | 13.71 对 15.45(**−1.74**) |

**En→Zh:我们用 0.64 秒的读步,超过他们 4 秒读步的 Dec-only 1.31 分,参数量只有 1/6。**
**Zh→En:我们在 1–2 秒档领先,3 秒以上被反超。**

对他们的主结果(Talker 双流)仍有差距:m4 上 24.04 / 18.78,**领先我们 5.29 / 5.07**。
论文把这归因为架构(§4.2 模态干扰、§B.2 理解与生成的表征冲突),不是数据或规模。

## 一bis、同传延迟指标全族(RealSI 777 条,最佳配置)

由 SimulEval 1.1.0 自己的 scorer 打分,实现按 sha256 锁定
(`2fbb749de96a0d36…`,见 `evaluation/simuleval_latency.py`),
所以这些是领域通用的定义而非本项目的重新实现。

| 指标 | En→Zh | Zh→En | 含义 |
|---|---:|---:|---|
| **LAAL** | **4712** | **4189** | 长度自适应平均滞后,ms。同传的主延迟指标 |
| **AL** | 4712 | 4189 | 平均滞后。与 LAAL 同值(译文长度未短于参考时两者一致) |
| **AP** | 0.09 | 0.17 | 平均比例,0–1,越小越早 |
| **DAL** | 3062 | 3157 | 可微平均滞后,ms |
| **StartOffset** | **2865** | **2974** | 首个语音块的发出时刻,ms |
| **EndOffset** | 3811 | 3293 | 源结束后仍在说的时长,ms |
| **NumChunks** | 4.27 | 3.67 | 每条输出的语音块数 |
| **RTF** | 1.61 | 1.57 | 实时率(见下) |
| 样本数 | 346 | 431 | 0 条被跳过 |

### 计算感知(Computation-Aware)版本

把模型自身的计算耗时算进听者延迟 —— 这是部署时听者真正经历的延迟:

| 指标 | En→Zh | Zh→En | 相对非 CA |
|---|---:|---:|---|
| **LAAL_CA** | **5464** | **4849** | +752 / +660 ms |
| AL_CA | 5464 | 4849 | 同上 |
| AP_CA | 0.12 | 0.20 | |
| DAL_CA | 3995 | 3790 | +933 / +633 ms |
| StartOffset_CA | 3139 | 3172 | +274 / +198 ms |

**RTF 1.61 / 1.57 > 1 意味着当前实现还不能实时** —— 计算比音频慢。
论文对自己的 Dec-only 也报告了同一问题(§F.2:*"the Dec-only structure's RTF is
substantially higher than the Talker's at every comparable tier, confirming that the
unified decoder is too heavy for efficient continuous audio generation"*),
并说明从 m4 起 chunk 变大后才 RTF < 1。我们在 0.64 s 读步下 RTF 1.6,
**若要实时需要更大读步或更快的解码实现**,这是已知的、与架构同源的限制。

### 时间线构成

| | En→Zh | Zh→En |
|---|---:|---:|
| 源时长 | 7.4 s | 6.7 s |
| 放置后时间线 | 11.3 s | 10.1 s |
| 纯语音(拼接) | 6.6 s | 5.5 s |
| 自然终止率 | 0.976 | 0.985 |
| 撞分帧预算比例 | 0.024 | 0.015 |

自然终止率 0.98、撞预算 0.02 —— **几乎所有片段都是模型自己选择结束的**,
不是被预算截断的,所以先验 4.0 没有把生成推到硬上限之外。

### 其他指标(仅我们有测量,论文未在 Table 5 报告)

| 指标 | 我们最佳 | 参照 |
|---|---:|---|
| LAAL(En→Zh / Zh→En) | 4712 / 4189 ms | — |
| 语音密度 codes/char 对 gold | 0.93 / 1.07 | gold = 1.00 |
| speech/source | 0.925 / 0.873 | gold ceiling 1.01 |
| 截断率 | 0.441 | **可达下界 0.368**(完美分段 gold) |
| Silence Ratio | 0.287 | gold ceiling 0.123 |
| 片段/条 | 4.3 / 3.7 | — |

### 必须说明的不可比之处

* **读步不同**:我们 0.64 s,他们最小 1 s。粗读步天然更准,所以"我们 0.64 s 超过他们 4 s"
  这个说法对我们**有利且成立**;反过来若比同一时延档,我们没有 1 s 的 arm。
* **测试集同为 RealSI sentence-level**,但我们用 777 条冻结选集,论文未公布具体条数。
* **ASR-BLEU 协议**:我们用 `asr_bleu_repo`(带归一化,与本项目历史数字一致);
  论文未说明其归一化细节。差异可达 2 分左右(我们的 `raw` 协议给 16.03 / 11.74)。
* 论文数字为 greedy 解码,我们也是 greedy(beam 实测无效)。

## 二、最佳 checkpoint

```
Megatron 原生:
  checkpoints/uniss_streaming_p2st_traj_v1/rollin_20260906T094846Z/iter_0001592

HF 导出(推理用这个):
  checkpoints/exported_hf/uniss_streaming_p2st_traj_v1_rollin_20260906T094846Z_iter_0001592_hf
```

**世系**(每一步都是前一步的继续训练):

| 阶段 | checkpoint | 内容 |
|---|---|---|
| 离线基座 | `qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075` | — |
| Stage-A 交接 | `stage_a_formal8_20260816T224100Z/iter_0000381` | — |
| C(纯 CE) | `uniss_streaming_p2st_pure_ce_v1/p2st_epoch1_replay_20260902T170132Z/iter_0004236` | 事件级三家族 |
| 步骤 1 | `uniss_streaming_p2st_traj_v1/nir_stratified_20260904T225149Z/iter_0003876` | NIR 单调性分层 |
| 步骤 2 | `uniss_streaming_p2st_traj_v1/chunk640_asridle_20260905T170518Z/iter_0003180` | 640 ms 固定网格 + ASR IDLE |
| **步骤 3(最佳)** | **`rollin_20260906T094846Z/iter_0001592`** | 同池再训 1592 步(roll-in 实际未激活,见 STEP1_TO_3_RESULT §三.2) |

## 三、推理方法

### 环境

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
source experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/experiment.env
export PYTHONPATH=. PYTHONIOENCODING=utf-8 LC_ALL=C.UTF-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NVLIB=$(find "$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages/nvidia" \
  -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:\
/usr/local/cuda-12.8/targets/x86_64-linux/lib:$(dirname "$PYTHON_BIN")/../lib:\
${LD_LIBRARY_PATH:-}:$NVLIB"
```

`experiment.env` 提供 `PYTHON_BIN`、`V1_CHECKPOINT`、`WHISPERVQ_MODEL`。
**不要新建环境或安装包** —— 所需依赖都在 `conda_envs/uniss-train` 里。

### 最佳配置

```bash
$PYTHON_BIN -m experiments.uniss_streaming_p2st_pure_ce_v1.evaluation.realsi_rollout \
  --selection    data/processed/realsi_p2st_v1/REALSI_SENT_SELECTION.json \
  --candidate-hf checkpoints/exported_hf/uniss_streaming_p2st_traj_v1_rollin_20260906T094846Z_iter_0001592_hf \
  --v1-checkpoint "$V1_CHECKPOINT" \
  --whispervq-model "$WHISPERVQ_MODEL" \
  --bicodec-model pretrained_models/UniSS/bicodec \
  --output-root reports/<你的目录> --arm <arm 名> \
  --read-stride 4 \
  --source-holdback 1 --target-holdback 0 \
  --length-prior-scale 4.0 \
  --min-fragment-tokens 16 \
  --min-final-chunk-ms 320 \
  --shard-index 0 --num-shards 24 --device cuda:0
```

24 个分片可并发(8 卡 × 3),跑完把 `MANIFEST_g*.json` 合并成 `MANIFEST.json`。

### 每个参数的含义与实测依据

| 参数 | 值 | 为什么 |
|---|---|---|
| `--read-stride 4` | 640 ms | 与训练的固定网格对齐(`BLOCK_MS 160` × 4)。实测越大越糟:160→1920 ms 截断率 0.507→0.762 |
| `--source-holdback 1 --target-holdback 0` | s1t0 | RealSI 上首音 −971/−830 ms,代价仅 −0.65/−0.32 ASR-BLEU |
| `--length-prior-scale 4.0` | 4.0 | **777 条上单调最优**:0→13.19/10.05、1.0→17.42/12.91、2.0→17.72/13.51、**4.0→18.37/13.89**。注意 8.0 会崩(zh→en 14.78) |
| `--min-fragment-tokens 16` | 320 ms | 短片段清零(13.7%→0%),首音不变,代价 +60 ms 等待。修饰性 |
| `--min-final-chunk-ms 320` | 320 ms | 尾块不足 320 ms 时并入上一读步,省一轮不可能有新词的推理。8 条里 5 条触发 |

**默认关闭、实测无效,不要开**:

| 参数 | 结论 |
|---|---|
| `--text-num-beams` | 无效。lp=1.0 重复爆炸(译文/参考 1.85),lp=0.6 退化成贪心(逐字相同) |
| `--semantic-temperature/top-k/top-p` | 无效。与贪心不可区分,且有单条塌陷 |
| `--text-temperature/top-k/top-p` | 无效。候选重排的 oracle 只比贪心低 0.006 |
| `--silence-fill` | 无效。BiCodec 静音 9e-6 RMS,比 VoiceBox 低 60 倍 |
| `--speed` | 无效。0.7/0.8/0.9 均使最长空档停在 1660–1700 ms 且拉长尾部 |

### 可选后处理:舒适噪声

空档帧的 70.7% 是精确数字零,听起来像断线。以下把它换成模型自己停顿的电平
(不改任何有声样本,因此 ASR-BLEU 逐字不变):

```bash
$PYTHON_BIN -m experiments.uniss_streaming_p2st_traj_v1.evaluation.comfort_noise \
  --manifest <arm>/MANIFEST.json --output-arm <arm>_comfort
```

### 评测

```bash
# ASR-BLEU:英文走 Whisper(uniss-train),中文走 Paraformer(uniss-eval)
ARM=<arm 名> bash <scratchpad>/realsi_asr.sh
# 汇总对比(两种 BLEU 协议、按方向分开、LAAL、密度)
$PYTHON_BIN -m experiments.uniss_streaming_p2st_traj_v1.evaluation.realsi_compare \
  --rollout-root <rollout 根> --arm <arm 名> --markdown out.md
# 割裂与静音
$PYTHON_BIN -m experiments.uniss_streaming_p2st_traj_v1.evaluation.cut_placement   --manifest <arm>/MANIFEST.json --label <arm>
$PYTHON_BIN -m experiments.uniss_streaming_p2st_traj_v1.evaluation.silence_budget  --manifest <arm>/MANIFEST.json --label <arm>
```

**比较时务必同时跑 `--length-prior-scale 1.0` 的 arm**,否则训练侧的进步会被推理参数掩盖。

### 可听样例

```
reports/uniss_streaming_p2st_traj_v1/longform_curve/
  demo_step3_final_prior4/stereo/          最佳配置,8 条
  demo_step3_final_prior4_comfort/stereo/  + 舒适噪声
```
左声道 = 源,右声道 = 译音,严格发出时间线。
