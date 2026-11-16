# E8 结果：X-Translator（开源级联 S2ST，只做测试推理）多说话人外部基线

## 概况

- **完成时间**：原始运行 2026-11-16 22:24 布拉格时间（`E8_QUEUE_DONE`）；电平校准变体 2026-11-17 00:01 布拉格时间（`E8L_QUEUE_DONE`）。所有集合都没有失败。
- **登记**：`multispk_s2st_baselines_prereg_v2.zh-CN.md` 的 §1 S2、§2、§4，以及修订 2（c18ee367a，写于看到任何 X-Translator 分数之前）。
- **代码**：`experiments/s2st_baselines_v1/xtranslator_multispk/`。
- **汇总**：`eval_outputs/mist_w3po_v1/e8_xtranslator/e8_summary_v1.json`（`summarize_e8.py` 生成），tracked 副本在 `reports/mist_w3po_v1/e8_xtranslator_results/`。
- **系统**：官方代码和权重（Qwen3-ASR-1.7B + LMT-60-8B + X-Voice 0.4B Stage2，配置 `xt-default`），不训练，不调参。
- **输入**：每个在线测试窗口是一个独立会话，与 E7 用的窗口相同。
- **评测**：tcpBLEU / cpBLEU、ASR-BLEU、chunk 协议延迟都用我们自己的代码。

## Table 1 用哪一行（修订 2 第 4 条，事先固定）

| 集合 | Table 1 行 |
|---|---|
| AMI Zh→En | 电平校准运行 `jtest_ami_xtranslator_level` |
| AMI En→Zh | 电平校准运行 `jtest_en_xtranslator_level` |
| AliMeeting | 原始运行 `jtest_ali_xtranslator_default`（g = 1，没有电平变体） |

原始运行和能量门统计放附录，作为探索性结果。

## 1. 主要结果

列说明：
- tcp：时间受约束、按说话人归属的 BLEU；cp：会议级最小置换 BLEU（`cp_asr_bleu.py`）。
- ASR-BLEU 指语音转写后的分数，英文用 Whisper-large-v3，中文用 Paraformer。
- 延迟单位为秒，按 chunk 协议计算，不计计算耗时。
- **LAAL_block 是主口径**（块起点口径，`table1_lat.sh` 的 blockstart）。LAAL_spread 即 E7 表里的"语音 LAAL"（词在块内均匀分布）。

| 运行 | 集合 | 文本 tcpBLEU all / ov / 1spk | 文本 cpBLEU | **tcpASR-BLEU** all / ov / 1spk | **cpASR-BLEU** | 文本 LAAL | **LAAL_block** | LAAL_spread | StartOffset | EndOffset |
|---|---|---|---|---|---|---|---|---|---|---|
| **level（Table 1）** | AMI Zh→En | 8.88 / 3.64 / 11.19 | 11.99 | **8.64** / 3.96 / 10.73 | **12.05** | 2.46 | **2.69** | 3.65 | 2.45 | 3.44 |
| **level（Table 1）** | AMI En→Zh | 25.05 / 15.64 / 29.18 | 28.99 | **19.96** / 11.26 / 23.89 | **23.31** | 2.48 | **2.74** | 3.57 | 2.74 | 3.32 |
| **default（Table 1）** | AliMeeting | 16.21 / 9.64 / 18.12 | 22.22 | **13.77** / 7.84 / 16.42 | **19.46** | 2.35 | **2.77** | 3.62 | 2.59 | 3.80 |
| default（附录） | AMI Zh→En | 4.71 / 1.00 / 6.72 | 6.50 | 4.53 / 1.15 / 6.33 | 6.48 | 2.55 | 2.75 | 3.74 | 2.57 | 3.45 |
| default（附录） | AMI En→Zh | 15.05 / 8.10 / 18.40 | 17.20 | 10.86 / 5.00 / 13.86 | 12.54 | 2.89 | 3.13 | 3.98 | 3.09 | 2.89 |

**单说话人下界**：同一份输出全部标成一个说话人后的文本 tcpBLEU all。

| 集合 | level | default |
|---|---|---|
| AMI Zh→En | 4.95 | 3.14 |
| AMI En→Zh | 12.94 | 8.43 |
| AliMeeting | — | 9.97 |

**参照 E7（SeamlessStreaming，同一窗口）**：

| 集合 | tcpASR-BLEU | LAAL_block | LAAL_spread | EndOffset |
|---|---|---|---|---|
| AMI Zh→En | 8.48 | 2.73 | 3.29 | 2.53 |
| AMI En→Zh | 11.31 | 2.50 | 3.19 | 3.04 |
| AliMeeting | 9.42 | 3.24 | 3.96 | 2.72 |

**读法**：
- **ASR-BLEU**：
  - Table 1 的三行里，X-Translator 在 En→Zh（19.96 vs 11.31）和 Ali（13.77 vs 9.42）上明显高于 Seamless，在 AMI Zh→En 上持平（8.64 vs 8.48）。
  - 与我们的系统比：E7 报告里 MIST 在 AMI Zh→En 是 20.45、En→Zh 是 32.33，X-Translator 仍明显更低。
- **延迟**：块起点口径 LAAL 在 2.7–2.8 s。原因是它要等 ASR 的标点提交、终结或说话人切换之后，才整段翻译和合成。
- **显著性**：与 MIST 的配对检验和 Holm 校正由 `table1_analysis.py` 统一计算，本报告不做。

## 2. 电平校准的作用（修订 2）

增益只用各集合的 dev 在线窗口确定，规则是 g = clip(0.1 / m_dev, 1, 31.6)。测试窗口不参与。

| 集合 | dev 窗口数 | m_dev | g | dev 上能量门始终关闭的比例（原始 → 施加 g 后） | 测试输入上的比例（原始 → 施加 g 后） | 测试输入削顶采样比例 / 有削顶的窗口 |
|---|---|---|---|---|---|---|
| AMI Zh→En | 3618 | 0.0553 | 1.808（+5.2 dB） | 18.1% → 2.4% | 32.2% → 8.4% | 7×10⁻⁷ / 30 窗（0.8%） |
| AMI En→Zh | 4996 | 0.0464 | 2.157（+6.7 dB） | 20.6% → 7.9% | 27.7% → 9.7% | 1.37×10⁻³ / 493 窗（9.6%） |
| AliMeeting | — | — | 1（不做） | — | 1.4%（只占有声时长的 0.2%） | — |

- **能量门**：X-Translator 客户端把 0.16 s 分成一块，块 RMS 低于 0.01 就不送 ASR（`Qwen3StreamingRecognizer.silence_threshold`）。官方网页前端开着 `autoGainControl: true`，我们读文件时绕过了这一步。
- **输入电平对比**（每窗最响块 RMS 的中位数）：RealSI 0.137，AMI 测试 0.016 / 0.018，Ali 0.066。
- **只看输入的统计**：AMI 两个集合在原始输入下，能量门始终关闭的窗口占 32% / 28%，对应 diarizer 有声时长的 18% / 11%（`input_gate_stats_v1.json`）。
- **实际运行里的削顶**：

  | 集合 | 削顶采样数 | 占比 |
  |---|---|---|
  | AMI Zh→En | 244 | 7×10⁻⁷ |
  | En→Zh | 575,166 | 0.137% |

  按每行 `counters.clipped_samples` 汇总，与只看输入的预估一致。
- **空窗口**：AMI Zh→En 1328 → 423，En→Zh 1573 → 657。
- **分数**：tcpASR-BLEU 4.53 → 8.64、10.86 → 19.96；cpASR-BLEU 6.48 → 12.05、12.54 → 23.31。延迟反而略降，因为更多段在窗口结束前就提交了。

## 3. 计数

| 运行 | 集合 | 窗口 | 段 | unspoken（TTS 失败） | MT 失败 | 空窗口 | 出错窗口 | unk 段 | 说话人切换 flush | 语音路数 | 语音时长 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| level | AMI Zh→En | 3973 | 11501 | 1 | 0 | 423 | 5 | 0 | 589 | 4617 | 22188 |
| level | AMI En→Zh | 5126 | 15389 | 20 | 0 | 657 | 4 | 1 | 1469 | 5761 | 27501 |
| default | AliMeeting | 4065 | 27816 | 0 | 0 | 95 | 4 | 38 | 3546 | 6556 | 45336 |
| default | AMI Zh→En | 3973 | 7719 | 1 | 0 | 1328 | 1 | 1 | 308 | 3346 | 15475 |
| default | AMI En→Zh | 5126 | 11531 | 15 | 0 | 1573 | 3 | 2 | 841 | 4550 | 20484 |

- **出错窗口**：两次运行共 17 个，全部是 Qwen3-ASR 服务端返回 "Failed to apply Qwen3ASRProcessor"，服务日志里的条数与之逐一对应。这是模型侧失败，按 E7 规则记为空输出。
- **TTS 失败**：共 37 段（原始运行 16，电平变体 21），都是模型侧：
  - X-Voice 返回 500（"received an empty list of sequences"）；
  - 少数返回 400（没有可用的说话人提示，或文本为空）；
  - 没有超时，也没有因显存不足而失败。
  - 这些段作为只有文字的 unspoken 段计入文本打分。
- **segment_block_mismatch**：En→Zh 有 31 / 33，AMI 有 1，Ali 为 0。原因是 TTS 失败的段可能出现在窗口中间，而延迟工具按顺序把段和块对上，同一路后面的段就对不上了。受影响的段约占 0.2–0.3%，结果照实报告。
- **窗口完整**：每次运行的 build 都会检查所有窗口齐全，否则拒绝。全部通过，没有丢失窗口。

## 4. 执行记录与偏离（都不改变模型设置）

1. **服务器连接故障的处理**：连不上我们自己的服务（requests ConnectionError）时，驱动不写这个窗口、以退出码 3 结束，之后用 `--resume` 续跑。
   - 原始运行发生 2 次：LMT 和 ASR 服务各 1 次 RemoteDisconnected，服务本身没有停。电平变体运行中 0 次。
   - 这比原系统直接丢掉该段更有利于 X-Translator，属于基础设施处理。
2. **队列 v2**（2026-11-16 约 20:01 布拉格时间）：
   - 起因：v1 在同一张卡上相隔 11 s 启动了两个 Whisper 任务，第一个的显存还没显示出来，3 号卡最高到约 140 GiB。v1 的 ASR 日志重定向也只作用在末尾的 `touch` 上，日志丢了。其间有 1 个 ASR 分片失败一次后重试成功；原因不明，日志已丢，发生在显存最高的时候。
   - 处理：按 PID 停掉 v1（之前确认它没有正在执行的后处理步骤），启动 `e8_queue_v2.py`。它接管了正在运行的 22 个解码驱动和 4 个 ASR 任务，没有重复启动任何任务。
   - v2 的改动：ASR 显存估计 24 GiB；服务运行期间每卡最多 1 个 ASR 任务，同一张卡上两次启动间隔 ≥ 120 s；日志重定向覆盖整条命令。
   - 其后每次启动都满足"已用 + 估计 + 6 ≤ 135 GiB"。
3. **ali__dec06 的 "rc -1"**：v2 对接管的进程拿不到退出码，一律记 −1，并不是失败。该分片打印了 `E8_SHARD_DONE`，part 文件里 82/82 个窗口按顺序都在，没有重复，也没有续跑的必要。
4. **CPU 线程**（协调者 2026-11-16 约 20:30 布拉格时间要求，只是基础设施设置）：
   - 原始运行的服务没有限线程：单个 X-Voice 进程平均 713% CPU、326 个线程，整机 load 约 777。
   - 电平变体的每个服务都通过新文件 `thread_limit_run.py 4` 启动：OMP / MKL / OPENBLAS / TORCH_NUM_THREADS = 4，并调用 `torch.set_num_threads(4)`。原服务文件不改。驱动用同样的线程环境，ASR 和打分任务另加 MKL / OPENBLAS = 8。
   - 实测 X-Voice 约 25% CPU、22 个线程，运行期间整机 load 约 50。
   - 线程设置记录在服务记录（`servers/*.json` 的 `threads`）和每行的 `config.cpu_threads` 里。
5. **段规则**（按登记）：
   - 译文为空的段按 RealSI §2 丢弃并计数；两次运行都是 0 条。
   - 段的源区间按 E7 §3 取 [上一段写出时刻, 本段写出时刻]。X-Translator 自己估计的源区间和 CAM++ 说话人只记在 raw 里，说话人只用于克隆音色。
   - 中文输出按 E7 §3 去掉与汉字相邻的空格。
   - 各集合只有一次 TTS 运行，X-Voice 的采样不设种子。
6. **冒烟测试只用 dev**：原始版用了 3 + 2 个窗口；`--gain` 版和限线程服务各单独测过。电平变体的冒烟借用了原始运行一个 slot 的服务，请求只是排队，输出分开存放。
7. **E8 原始运行的服务启动顺序**：按 slot 交错；vLLM 引擎之间全局错开 60 s；每张卡第一次启动前按 PID 停占位（先核对命令行）；guard（2726951）和 gpu_watch（52045）没有动过。解码结束后按 PID 停掉全部服务（含 EngineCore 子进程），由 guard 补回占位。

## 5. 文件

| 内容 | 路径 |
|---|---|
| 运行目录 | `eval_outputs/mist_w3po_v1/decode/jtest_{ami,en,ali}_xtranslator_default/`（及 `_1spk`）、`jtest_{ami,en}_xtranslator_level/`（及 `_1spk`） |
| 语音 | `eval_outputs/mist_w3po_v1/speech/<run>/talker/` |
| raw | `eval_outputs/mist_w3po_v1/e8_xtranslator/raw{,_level}/<set>/` |
| cp 分数 | `eval_outputs/mist_w3po_v1/e8_xtranslator/cp/` |
| E7 口径延迟 | `eval_outputs/mist_w3po_v1/crf/t_eval/<run>/` |
| Table 1 延迟单元 | `eval_outputs/mist_w3po_v1/joint/table1_v14/lat/{perword,blockstart}/<run>.json(.units.jsonl)` |
| 日志 | `logs/mist_w3po_v1/e8/`、`logs/mist_w3po_v1/e8_level/` |
| 服务记录 | `eval_outputs/mist_w3po_v1/e8_xtranslator/servers/` |
| 增益与能量门统计 | `level_gain_v1.json`、`input_gate_stats_v1.json` |
