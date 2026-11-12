# 最终测试集评测预登记 v1（final_test_prereg_v1）

- 写入时间：2026-11-12 布拉格约 20:15（19:15 UTC）。用户要求：把所有有效方法在测试集上跑一遍，按 GPU 并行。
- 写入时，**本文件里的五个系统都还没有在任何测试集上解码过**，测试集上也没有算过任何与它们有关的数字。为建测试窗口做的检查只用了窗口、音频和分离器后验（第 2 节），没有用参考译文。
- 每个测试集**只跑一次**，配置在下面写死，**不在测试集上做任何选择**（不调参、不换种子、不挑系统、不按测试结果改判定）。测试结果只是对 dev 结论的描述性确认。
- 本文件提交之后的任何修改，一律以"修正 N"的形式追加在文末，写明时间和原因，不改动已有内容。

## 1. 系统

五个系统分两组。第一组的三个系统用**同一份 A′ Thinker 解码**（每个测试集一份）。

| 代号 | 名称 | Thinker | talker 输入 | talker | 角色 |
|---|---|---|---|---|---|
| **(a)** | A′（c1+b1+s2） | v2′ `crf_v2p_20261110T075131Z` step_03000 | 逐段 | T1c + c1+b1+s2 | 第一组，参照 |
| **(b)** | A′ + PL-EM | 同上（同一份解码） | 逐段 | PL-EM（`talker_plem_20261111T001759Z`，c1b1s2 设置） | 第一组 |
| **(c)** | A′rl + PL-EM | 同上（同一份解码） | 读级块（`merge_read_segments`） | PL-EM | 第一组，**主系统** |
| **(d)** | VE1rl + PL-EM | VE1 `ve1_20261111T214454Z` step_03000（种子 20261109） | 读级块 | PL-EM | 第二组，**延迟优先的取舍点** |
| **(e)** | D1vrl + PL-EM | D1v `d1v_20261112T120413Z` step_03000 | 读级块 | PL-EM | 第二组，**延迟优先的取舍点** |

- (d)、(e) 在报告里明确写成取舍点，不写成"有效方法"。是否进论文由用户决定。
- **解码**（三个 Thinker 相同，与 dev 逐项相同）：v8 merged Thinker 加 CRF 检查点，`decode_crf_ckpt_finalpost --final-read-post close_detect`，`--policy wait --lag-ms 2000 --context 0 --dir zh2en --batch 16 --max-gib 120`，每 1.0 s 读一次，loop guard（k 6、maxp 12、budget-tps 25、slack 50），按会议分 8 片。
- **语音链**（与 dev 逐项相同）：talker prep（`gstar_talker_v6 prep --causal --blocks --preset lowlat`）→ 每个种子 8 个 talker 分片（**分片数不改**，vLLM 带种子的采样受批组成影响）→ Whisper ASR（每个 talker 分片 2 个子分片）→ ASR-BLEU。两个 TTS 种子：种子 1（默认）和种子 2（`--seed 20261109`）。
  - c1+b1+s2 和 PL-EM 的 talker 参数、CTC 打分服务器都与 dev 相同。
- **编排**：`scripts/final_test_orch.py`。它就是 `thinker_t_eval_orch.py`，只换与测试集有关的几处：窗口、参考窗口和 `--set`、talker prep 的数据目录、延迟工具的参考集。每条命令启动前都检查，不能残留任何 dev 路径。逐条对比过 dev 和测试的 talker 命令，换掉测试集路径之后完全相同。
  - 每个测试集一个驱动脚本 `scripts/final_test_chain.sh`，事后指标由 `scripts/final_test_post.sh` 计算。

## 2. 测试集、窗口与参考

| 测试集 | 会议 | 会话音频 / 分离器后验 | C2 gap 0.6 窗口 | 参考 |
|---|---|---|---|---|
| **AMI zh→en**（中文重配音） | 16（`meetings.json` 的 test） | `zh_mix_v8` / `nemotron_lowlat_zh8` | **3973 个，5.90 h** | `data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl`：AMI 人工英文，与 v8 / v9 相同；dev 用的是同一构建的 `windows_validation_ref.jsonl` |
| **AliMeeting** | 20 | `near_mix` / `nemotron_lowlat` | **4065 个，10.31 h** | **DiariST 公开的人工英文参考** `data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl`（prereg_v9 §6，`align_diarist_refs.py`，16358 / 16358 句对齐）；没有人工参考的句子当作不计分 |

- 两个测试集都只有 zh→en，模型也只做 zh→en。

**窗口的检测器配置与 dev 完全相同**（dev 的 C2 窗口 `windows_zh2en_c2g06_v1` 由 `build_windows_zh2en.py --splits validation --gap 0.6` 建出）：
- 新脚本 `data/build_test_windows_c2.py` 用同一个函数 `build_windows_zh2en.online_windows(post, 0.6)`：
  - 任何槽位 > 0.5 时开窗；静默 0.6 s 或满 20 s 时关窗；
  - 音频取 [开窗 − 0.2 s, 最后活跃帧 + 0.3 s]；
  - 开窗、关窗的检测时刻都按 1.04 s 前瞻计算。
- 会议、会话音频和后验用 v6 测试窗口的定义（`reports/gstar_si_v6/build_online_windows.py` 的 SETS）。所有窗口都保留，不过滤。
- 后验是硬盘上已有的整场 lowlat 后验（文件元数据 `preset: lowlat`），所以**不需要在 GPU 上重跑分离器**。
- 与原测试窗口的区别只有 gap：0.6 s，原来是 1.0 s。

**复现检查**（只看窗口，没有用参考）：
- 同一脚本用 `--gap 1.0` 重建，与原测试窗口 `data/online_v6_20261031/windows_test_online_{zh2en,ali}.jsonl` **逐条相同**：3365 / 2837 条，除音频和后验路径外所有字段差异为 0。
- gap 0.6 的窗口里，与原窗口区间相同的（AMI 2751 个、Ali 1137 个）抽查 111 / 46 个，音频采样和后验数组**逐位相同**。

**chunk 协议的窗口**：
- 新脚本 `data/chunk_windows_set.py`，就是 `specclose build --judge off`，只换成该测试集的后验和会话音频。
- 在 dev 上，它重建出的窗口与 `detector_v1/windows_chunk_v1/windows_dev_online.jsonl` **逐字节相同**。
- 测试集上（`data/windows_test_c2g06_chunk_v1/{zh2en,ali}/`），窗口与 C2 窗口逐条相同，只改两个时间戳，没有一个比现行协议晚。
  - 开窗平均早 316 ms；关窗平均早 399 ms（AMI）、402 ms（Ali）。dev 是 316 / 401 ms。

## 3. 指标（每个测试集分别报告）

1. **文本 tcpBLEU** all / ov / 1spk：`score_run`，在线协议，v8 重叠感知打分器。每个 Thinker 一份。
2. **语音 ASR-BLEU**：Whisper 转写 talker 语音后的 tcpBLEU。报告两个 TTS 种子各自的值和合并值。(a)、(b) 是逐段输入，(c)、(d)、(e) 是读级块输入。
3. **每条流语音 LAAL** all / ov / 1spk（`speech_latency_lit` 的 `speech_laal_online`）：
   - **现行协议**：C2 窗口；
   - **chunk 协议**：chunk 窗口。
   - 两种都报，两种子平均，**作为主延迟数字**。
4. **EndOffset**（单元级，online），两种协议。
5. **单通道 LAAL**（`online_1ch`），两种协议。
6. **文本 LAAL**（`--text-only`），两种协议，按 Thinker 报告。
7. **听感**：
   - UTMOS、SIM（`talker_quality`）；
   - talker WER（`speech_gap_decomp` 的 fidelity）；
   - 说话人内部停顿（`talker_block_pauses`）：句中停顿 ≥ 0.5 s 的次 / 分钟和秒 / 分钟，以及全部停顿；
   - 单词块占比、强制截断块；
   - UTMOS 配对差：同一条流配对，两种子合并，按会议 bootstrap。用 `evaluation/listen_paired_meeting.py`，即 `pmc_listen_compare.py` 的方法，只把会议键改成 `_on_` 之前的部分；否则 AliMeeting 会按房间分组。在 dev 上两者输出完全相同。
- 参考集和打分集合（universe）通过 `evaluation/on_refset.py` 传给基于 `metric_headroom` 的工具。在 dev 上用它重算 D1v 的文本 LAAL，与已存 json 逐字段相同。
- 不算 COMET、MetricX。

## 4. 比较（描述性确认 dev 结论；不改变任何已有判定）

用 `gate_2seed`，每个测试集分别做：
- 规则：点估计 ≥ 参考 − 0.2，并且按会议配对 bootstrap（10k）P(差 ≤ 0) < 0.95；
- 项目：文本 all、文本 ov、语音 all（两种子合并每会议统计量）；
- 测试集的会议数：AMI 16，Ali 20。

| 比较 | dev 上的结论（复现目标） |
|---|---|
| **(c) vs (a)** | 文本相同；语音 23.69 vs 23.26，**+0.43**（P 0.000），通过；LAAL 4.050 vs 4.152（chunk 3.857 vs 3.961） |
| **(c) vs (b)** | 文本相同；语音 +0.08（P 0.21），通过（不劣）；LAAL 4.050 vs 4.071 |
| **(d) vs (c)** | 文本 all −0.11、ov +0.02；语音 23.64 vs 23.69，−0.06（P 0.64），通过；LAAL 3.756 vs 4.050（chunk 3.586 vs 3.857） |
| **(e) vs (c)** | 文本 all +0.24、ov **−0.22**（不过）；语音 23.36 vs 23.69，**−0.34**（P 0.984，不过）；LAAL 3.548 vs 4.050（chunk 3.391 vs 3.857） |

- UTMOS 配对差也做同样四个比较。
- **"dev 结论在测试集上复现"的定义**（每个比较、每个测试集分别判）：
  - `gate_2seed` 三项的过 / 不过与 dev 相同；
  - 每条流语音 LAAL 差的符号与 dev 相同（两种协议分别看）。
  - 不一致就照实写"测试集上没有复现"，并写清是哪一项。
- 不论结果如何：(c) 仍是主系统；(d)、(e) 仍只作为取舍点报告；不在测试集上改任何配置，也不补跑别的系统。
- 两个测试集分别报告，不合并。
- 这些比较只作描述，不做多重比较校正。

## 5. 执行与已知偏差的处理

- **顺序**：
  1. 两个测试集的 A′ 解码；
  2. A′ 解码完成后，立即启动 (c)、(b)、(a) 的语音链，同时开始 VE1 和 D1v 的解码；
  3. VE1、D1v 解码完成后，启动 (d)、(e) 的语音链。
- **卡位**：
  - 8 张卡都用，`--cap135`：每卡最多 135 GiB；每卡最多 2 个解码，按 64 GiB 计划；talker 和 ASR 在显存允许时叠放；同一张卡上 vLLM 启动间隔 ≥ 60 s；
  - 我们的占卡程序由编排器按 PID 停掉；
  - 不碰其他代理（R3、R4）的进程，按 nvidia-smi 的实际用量留余量。
- **合并 pair_lost**：读级块合并（`merge_read_segments`）若停在 pairing lost，按以下规则处理，每个系统、每个测试集分别判：
  - 合并本身丢失 ≤ 2 个窗口（pair_lost ≤ 2），并且合并输出完整（行数 = 窗口数，`score_text.json` 在）：照用，丢失的窗口列为已知偏差；
  - 否则：该系统的语音链不启动，问协调者。
  - 测试集上没有 dev 的"pair_ok ≥ 3589"这条线，因为窗口数不同。
- **作业失败**：编排器对同一作业的自动重试（例如 OOM、vLLM 启动失败）是同一配置的重新执行，不算选择。永久失败的作业照实报告，不换配置、不换种子。
- **不做**：训练、调参、换种子、COMET / MetricX、QABC 方法；不在 RealSI / ACL 上做任何事。

## 6. 报告

- 结果写进 `reports/mist_w3po_v1/final_test_results_v1.zh-CN.md` 和 `LOG.zh-CN.md`，提交并 push。
- 输出位置：
  - 解码和语音：`eval_outputs/mist_w3po_v1/decode/ft_<集>_*`、`speech/ft_<集>_*`；
  - 延迟、门槛、听感：`crf/t_eval/ft_<集>_*`（现行协议的 lat_*）、`crf/t_eval/final_test_v1/<集>/`（chunk 协议、gate、听感）、`talker_crf/quality/ft_<集>_*`；
  - 编排日志：`logs/mist_w3po_v1/final_test/`。
