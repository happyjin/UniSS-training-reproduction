# 多说话人同传 S2ST 基线（Table 1）与说话人归属的 ASR-BLEU：预登记 v2

- 写于 2026-11-16 晚（布拉格），**早于任何新输出**。
- 授权：用户要求把不同方法的多说话人 speech-to-speech 同传基线放进 Table 1，把 ASR-BLEU 换成 cp/tcp 版本；中途需要拍板的事按最佳选择决定。
- 规则：
  - 测试集只做推理和打分，不调任何参数；每个系统、每个集合只解码一次；
  - 我们自己的系统不因这些结果做任何改动；
  - 外部系统只用官方权重和代码，不训练、不微调。

## 1. 系统（Table 1 的行）

| 行 | 系统 | 状态 | 说话人归属 |
|---|---|---|---|
| S1 | SeamlessStreaming（零样本，官方默认） | 已有（E7，`e7_seamless_multispk_prereg`） | 因果 diarizer 重标（v8 规则） |
| S2 | **X-Translator**（开源说话人感知的级联 S2ST：Qwen3-ASR 1.7B + LMT-60-8B + X-Voice 0.4B，官方默认 `xt-default`） | **待跑**（E8） | 因果 diarizer 重标（同 S1）；它自己的说话人提示跟踪只用于克隆音色 |
| S3 | **级联：不区分说话人的流式模型（同训练数据）+ 因果重标 + 基础 Talker** | **待跑**（E9，可行性见 §4） | 因果 diarizer 重标 |
| B | 截止基线（我们的 Thinker + 基础 Talker） | 已有 | Thinker 的说话人 token |
| MIST | B 的 Thinker + 播放层 | 已有 | Thinker 的说话人 token |

- **E8 的输入单位**：与 E7 完全相同，即三个集合的在线测试窗口（AMI Zh→En 3973 窗、AliMeeting 4065 窗、AMI En→Zh 5126 窗）。
  - 每个窗口是一个独立会话（新的 ASR / 说话人状态），复用单说话人驱动 `experiments/baselines_xtranslator_v1/xt_realsi.py` 的逐帧逻辑。
  - 时钟、块的写出时刻、按队列播放、TTS 失败的段（记作 `unspoken`）都与 RealSI 登记 `paper/baselines_run_plan_xtranslator_v1.md` §2 相同。
  - 段的源区间、在线协议的钳位、因果重标、输出格式都与 E7 §3–§4 相同。
  - 方向：AMI Zh→En 和 Ali 为 zh→en，AMI En→Zh 为 en→zh。
- **E9 的定义**：
  - v8 的 speaker-blind Simul 模型（每个方向一个，训练与 MIST-Simul 相同，说话人 token 置零），在同一批在线测试窗口上按 chunk 协议解码；
  - 每段按 v8 因果规则重标说话人，再用基础 Talker（与 B 相同的设置）合成两个 TTS 种子。

## 2. 指标（全部用我们的代码；两个 TTS 种子取均值，外部系统只有一次运行）

- **tcpASR-BLEU**：即论文现在的 ASR-BLEU（`score_run --asr`：语音转写的 tcpBLEU，时间受约束、按说话人归属）。不变。
- **cpASR-BLEU**（新增，会议级，同 CHiME-6 cpWER 的做法）：
  1. 每场会议、每个输出说话人标签，把该说话人所有转写段按会议时间顺序拼接；
  2. 每个参考说话人，把与 tcpBLEU 相同口径的计分单位（非犹豫、有参考）按时间拼接；
  3. 用匈牙利算法、以 chrF 最大化做一对一说话人匹配；`unk` 不参与匹配；
  4. 对所有（输出串, 参考串）对，加上未匹配的（输出, 空）和（空, 参考），计算语料 BLEU；
  5. 规范化和分词与 tcpBLEU 相同（英文 13a，中文 zh）。
- **同口径的文本 cpBLEU / tcpBLEU**：放附录。
- **延迟（chunk 协议，不计算耗时）**：语音 LAAL_block（主）、EndOffset、StartOffset，使用与 Table 1 相同的代码。
- **显著性**：MIST 对每个基线做会议级配对的符号翻转检验和 bootstrap；cp 与 tcp 两族，每族在 3 个集合 × 4 个基线上做 Holm 校正。
  - 外部系统只有一个 TTS 运行，与 MIST 两个种子的均值相比，这一点照实注明。

## 3. 消融表（R0–R4）

- 指标换成 tcpASR-BLEU（数值与现在的 ASR-BLEU 相同）。cpASR-BLEU 用 CPU 补算后放附录。
- 播放层不改变说话人标签（标签来自 Thinker），所以 B 与 MIST 的 DER 相同，正文写明这一点。

## 4. 可行性规则（事先固定）

- **E9**：若 v8 speaker-blind Simul 的检查点或现有解码工具在 1 小时准备时间内跑不通，或 8 卡在 E8 之外放不下，就不做 E9，并照实说明。不会为此改动 Table 1 的其他行。
- **GPU**：
  - 只在显存余量足够时叠加任务（已用 + 估计 + 6 GiB ≤ 135 GiB）；
  - vLLM 引擎错开 60 s 启动；
  - 占位训练按 PID 停（先核对命令行），不停 guard；
  - 训练卡不碰。

## 5. 文本对比（附录，只用 CPU）：DiariST-AliMeeting

- AliMeeting 测试集（= DiariST-AliMeeting test，IHM-MIX 条件，人工英文参考）。
- 用 DiariST 官方 `eval.py` 和官方 195 个 3–6 分钟小会话，对 B/MIST 的 Thinker 文本和语音转写计算 SAgBLEU / SAtBLEU。
- 与 DiariST 论文 Table 1/2 的 IHM-MIX 数字（Whisper-small 级联、DiariST t-SOT+t-vec）并列引用。
- 需写明：我们是整场会议解码，他们是逐个小会话解码；DiariST 只输出文本、没有报告延迟。

## 修订 1（同日晚，在任何 X-Translator / 级联输出之前）

**问题**：第一次用 CPU 计算 cpASR-BLEU 时（只算了 B / MIST / Seamless），En→Zh 和 AliMeeting 的 cp 低于 tcp，例如 B 的 En→Zh：cp 21.05，tcp 31.65。

**原因**：tcpBLEU（`score_ext`）会忽略落在"无参考语音"上的假设段。这类语音包括犹豫，以及参考窗口在审核中删掉的部分（En→Zh 与 AliMeeting 都有）。而 §2 的 cp 只在参考一侧用了同一个范围，假设一侧把这些语音的译文也拼了进去，导致精确率被稀释。

**修正**：假设一侧用与 `score_ext` 相同的规则：带 2 s 容差、重叠最多的标注话语如果不计分，该段丢弃。这样两侧覆盖同一个计分范围。

修正后 cp ≥ tcp，结果为：

| 集合 | B（cp / tcp） | MIST（cp） |
|---|---|---|
| AMI En→Zh | 35.83 / 31.65 | 36.48 |
| AliMeeting | 22.52 / 16.73 | — |
| AMI Zh→En | 24.84 / 18.07 | — |

这一修正是在看过上述三个已知系统的初步 cp 数字之后做的，照实记录在此。
