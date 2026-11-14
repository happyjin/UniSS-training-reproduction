# E7：SeamlessStreaming 直接喂多说话人测试窗口（外部 S2ST 基线）——登记

写于 2026-11-14 21:57（布拉格），在三个多说话人测试集上跑任何 SeamlessStreaming 输出之前。来源：`docs/mist_v10_plan.zh-CN.md`
§3.2 E7、`reports/mist_w3po_v1/multispeaker_baselines_plan_v1.zh-CN.md` §4.2 B-sim-2，用户 2026-11-14 约 21:45 的决定
（外部基线只要是开源模型、只做测试推理即可）。下面的设置、规则和报告项现在固定，结果出来以后不改。

## 1. 系统与设置（只推理，一个点）

- 系统、权重、代码、环境与单说话人登记 `paper/baselines_run_plan_seamless_v1.md` §1–§2 完全相同：HF
  `facebook/seamless-streaming`（本地 `/opt/dlami/nvme/jasonleeeli/models/seamless-streaming`），官方
  `SeamlessStreamingS2STAgent`（非 expressive），`seamless_communication` 9a081e93、`simuleval` 1.1.4、`fairseq2` 0.2.1，
  venv `/opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines`，fp32，argmax。不训练、不微调、不适配。
- **只用官方默认一个点**：`decision_threshold=0.5`、`min_starting_wait_w2vbert=192`、`min_unit_chunk_size=50`，其余与官方
  `streaming_evaluate --task s2st` 相同（`source_segment_size=320` ms、`no_early_stop`、`max_len_a=0`、`max_len_b=100`）。
- 语言：AMI zh→en、AliMeeting zh→en 用 `tgt_lang=eng`；AMI En→Zh 用 `tgt_lang=cmn`（单说话人驱动的代码）。
- 测试集只用来打分：每个集合只解码一次，不在 dev 或 test 上调任何东西。开跑前只在 2–3 个 dev 窗口上做冒烟测试，检查代码能跑通、
  格式对得上，不看分数。

## 2. 输入单位：我们检测器的在线窗口

- 三个集合的在线测试窗口，与我们的系统完全相同（`joint_lib.sh`）：AMI zh→en `WT_AMI`（3973 窗）、AliMeeting `WT_ALI`
  （4065 窗）、AMI En→Zh `WT_EN`（5126 窗）。每个窗口的音频（`audio`，16 kHz）单独喂给 agent，每个窗口前 `agent.reset()`。
- 驱动循环与单说话人驱动相同（SimulEval `SentenceLevelEvaluator` 的循环）：每步 320 ms，最后一块带 `finished`，读完后发
  `EmptySegment(finished=True)`，直到输出 `finished`。
- **说明**：这等于把我们的说话人活动检测 / 窗口切分交给了 Seamless。它自己没有切分长会议的机制；这样它拿到的是和我们一样的
  切分、一样的时钟。

## 3. 输出与时间

- 每个非空输出语音段 = 一个输出段（`segments` 一项，talker 时间线上一个 block）。段的文字 = 这段语音说的文本 token
  （单说话人驱动的记录方式：单元解码器每次写出时收到的 token；跨段的词算在开头所在的段，`<unk>` 不算）。中文输出去掉与汉字相邻的
  空格（中文不用空格，拉丁词之间的空格保留）。文本解码器已写出、但到窗口结束都没被说出的尾部 token，作为一个只有文字、没有语音的段
  （记 `unspoken`）。
- 时间都是窗口内的源读入时长（计算无关，与我们的系统相同）：
  - 段的文本时刻 `write_ms` = 这段最后一个 token 从文本解码器写出时已读入的源时长（对应我们 Thinker 的提交时刻）；
  - 语音块的 `write_ms` = 声码器发出这段语音时已读入的源时长；`start_ms = max(write_ms, 同一路上一块的结束)`（每一路排队，
    与我们的 talker 时间线相同），`dur_ms` = 语音时长。
- 段的源区间 `[start_s, end_s]`：Seamless 不给源对齐。固定为 [上一段的文本时刻, 本段的文本时刻]（窗口第一段从 0 开始），即两次
  文本写出之间新读入的源音频。
- 打分时用在线协议（`score_run` 原样）：提交时刻 ≥ `open_detect_ms`，源读完后写出的段 ≥ `close_detect_ms`。
- 极少数文本为空的语音段（单说话人 RealSI 里 1706 段中有 2 段）：音频并入同一窗口的上一块（没有上一块就并入下一块），保证
  时间线工具里“段 ↔ 块”一一对应。条数照实报告。

## 4. 说话人归属（因果，事先固定）

Seamless 不分说话人。每个输出段用 **v8 登记的因果重标规则**（`docs/mist_prereg_v8.zh-CN.md` §3，代码
`reports/gstar_si_v8/overlap_score.relabel_causal` 原样调用）：用窗口自带的分离器后验（窗口 jsonl 的 `post.lowlat`，0.08 s
一帧，与我们系统用的是同一份），提交时刻 t（在线协议钳位后）只能看 t − 1.04 s 之前的帧：
1. 段源区间 [a, b] 内平均后验最大的说话人位置；
2. 没有可用帧时，取 [a − 2 s, t − 1.04 s] 内最活跃的位置；
3. 仍然没有，标为未知（`unk`，评分时当作未匹配的输出说话人）。

不看整场会议的映射，不看未来的帧。
**下界变体（只算文本）**：同一份输出，所有段都标成同一个说话人（`_1spk` 运行目录），报 tcpBLEU all / ov / 1spk。

## 5. 指标（全部用我们自己的代码）

- **文本**：`experiments.mist_w3po_v1.evaluation.score_run`（v8 overlap 评分，`online=True`）→ `score_text.json`：
  tcpBLEU all / ov / 1spk，另记 CR、collapse、duplicate_rate、omitted / inserted。参考：AMI zh→en `RT_AMI`（`--set zh2en`），
  Ali `RT_ALI`（`--set ali`），En→Zh `RE_TEST`（`--set en2zh`）。
- **语音**：每个（窗口 × 归属的说话人）一路，talker 格式的 `results.jsonl`（`speech_path` = 这一路各块首尾相接的纯语音）。
  ASR 与我们的流水线相同：`evaluation.asr_transcribe`（英文 Whisper-large-v3，中文 Paraformer-zh）→ `asr_merge` →
  `asr_hyp` → `score_run --asr` → `score_asr.json`（ASR-BLEU all / ov / 1spk）。只有一个“种子”（Seamless 是确定性的）。
- **延迟（chunk 协议，11-12 起的统一口径）**：zh→en 两个集合用 `on_refset` + `speech_latency_lit`，En→Zh 用 `en2zh_lat`，
  `--win` = chunk 协议窗口（`WT_AMIC` / `WT_ALIC` / `WT_ENC`）：
  - 文本：`--text-only` → `unit.text_laal_online`；
  - 语音：`--talker <上面的 talker 目录>` → **`unit.speech_laal_online`（主口径，每路排队，与我们相同）**，
    `unit.speech_end_online`（EndOffset）、`unit.speech_start_online`（StartOffset）；另报 `speech_laal_online_1ch`
    （同一会议所有输出共用一个声道，更接近 Seamless 实际的单路输出），作补充。
- 另报：窗口数、输出段数、未说出的尾部文字段数、空文本语音段数、出错窗口数（出错的窗口记为空输出，照样计入）、unk 段数。

## 6. 报告

- 结果：`reports/mist_w3po_v1/e7_seamless_multispk_results.zh-CN.md` 和
  `eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json`。
- 运行目录：`eval_outputs/mist_w3po_v1/decode/jtest_{ami,ali,en}_seamless_default/`（以及 `_1spk`），语音
  `eval_outputs/mist_w3po_v1/speech/jtest_{ami,ali,en}_seamless_default/talker/`；目录已存在就拒绝运行。
- 代码：`experiments/mist_w3po_v1/e7_seamless/`（单说话人驱动的副本加多说话人部分；原文件不改）。
- 论文主表里作为“外部 S2ST 系统，输入混合会议音频（我们的窗口切分 + 因果说话人归属）”一行；注明它用了我们的检测器窗口和分离器后验。

## 7. 修订记录（测试解码之前，只改执行细节）

- 2026-11-14 22:04（布拉格）：代码放在 `experiments/s2st_baselines_v1/seamless_multispk/`（不是 §6 写的
  `experiments/mist_w3po_v1/e7_seamless/`），这样卡位工具（`joint_lib.sh` 的 `is_baseline_eval`）把这些进程认作外部基线评测。
  原始解码输出 `eval_outputs/mist_w3po_v1/e7_seamless/raw/<set>/`，日志 `logs/mist_w3po_v1/e7/`。设置、规则、指标都不变。
  冒烟测试：dev 上 2 + 3 个 zh→en 窗口、2 个 En→Zh 窗口，只检查格式和整条打分链（build → score_run → ASR → 延迟工具）能跑通。
