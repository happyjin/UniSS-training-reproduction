# 电竞解说数据上的多说话人同传 baseline 试听：数据分析与方案（2026-11-18）

## 1. 数据是什么

`s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/data_fortrain/domaindata_game/` 是 kenxsong 的王者荣耀（HOK）
电竞训练数据流水线产物，来源是 YouTube 赛事 / 解说视频的**真实原声**：

| 目录 | 内容 |
|---|---|
| `zh2en_production_20261106/production/runs/<video>/` | 中文原声 → 英文合成。167 个视频 |
| `en2zh_voxcpm2_20261106/continuation/production/runs/<video>/`（及 `resynth/`） | 英文原声 → 中文合成（VoxCPM2） |
| `zh2en_entity_correction_20261109/`、`auto_entity_multilingual_20261112/`、`source_entity_integration_20261116/` | 后续的姓名 / 术语重审版本（清单与重合成音频，原始音频不变） |
| `multilingual_pipeline_20261116/`、`nvme_cleanup_20261117/` | 多语种小批实验、清理记录 |

每个视频目录里：
- `source/<video>_partNN.wav`：**连续的原始音频**（16 kHz 单声道，每段约 15 分钟）。
- `audio/source/<segment>.wav`：约 30 s 的片段，但它是**把语音块拼接起来的**（`piecewise source concatenation`，去掉了静音和音乐），时间轴不连续。
- `records/<segment>.json`：三路 ASR 的源文本、分轮（turn）、说话人、`target_text`（GPT 翻译）、门控结果、`source_mapping`（拼接时间 ↔ 原始时间）。
- `diarization/<segment>.json`：pyannote + ERes2Net 的片段内说话人分离。

## 2. 一条音频里是否有多个说话人

有。以中文视频 `-6G-beNhHEs` 为例（928 个含语音的 30 s 片段）：

| 片段内说话人数（pyannote） | 1 | 2 | 3 | 4 | 5 | ≥6 |
|---|---|---|---|---|---|---|
| 片段数 | 54 | 156 | 295 | 273 | 101 | 49 |

- 94% 的片段有 ≥2 个说话人。分离器会把同一个人拆成两个标签，所以真实人数会少一些。但赛事解说通常是 2–3 位解说轮流说话，偶尔夹带选手语音或现场声。
- 说话人重叠约占语音时长的 4.6%，比 AMI 和 AliMeeting 少。这里主要是快速轮换，很少有长时间的抢话。
- 说话人编号只在片段内有效（`clip-local, not cross-clip identity proof`），不能跨片段对应同一个人。

所以这批数据**可以**作为多说话人同传的输入，两个方向都有：中文原声 → 英文，英文原声 → 中文。

## 3. 能否用多说话人同传 baseline 生成同传语音

可以。用的 baseline 是 **B**，也就是论文 Table 1 中的内部 baseline：联合双向 Thinker `joint_v8h6k merged_s06000` 加 identity CRF，配原始 T1c Talker。它与 AliMeeting Eval 的 B 行流程完全一致：

1. **整段音频**：在原始 `partNN.wav` 上截取连续片段。不用拼接后的 30 s 片段，因为同传必须在真实时间轴上进行。
2. **NeMo Nemotron 低延迟分离器后验**（preset `lowlat`，nemo_diar venv）。
3. **C2 窗口**：gap 0.6，规则同 `build_test_windows_c2`。
4. **Thinker 文本解码**：finalpost close_detect、wait、lag 2000、8 个分片；用 `alieval_orch.py` 的做法替换集合定义，编排器、占卡和让位都沿用原样。
5. **T1c Talker**：`--speed-policy fixed --history-ban 8`，两个 TTS 种子，用来试听。
6. **拼合**：按 `build_ami_stream_demo.py` 的方式，把每个说话人流的同传语音放到它所在窗口的时间轴上，生成试听音频。

**评分**：参考译文用 kenxsong 已审核轮次的源 ASR 文本和 GPT 译文，映射回原始时间轴，作为**银标参考**。这样编排器的评分步骤能照常运行，并给出一个近似的 tcpASR-BLEU。这个分数只用来参考，不能与论文中的数字比较，也不进论文。

## 4. 样本选择（不看系统输出，先定好）

- 每个方向 5 个视频，每个视频截取 1 段 **5 分钟**的连续音频，共 10 段、50 分钟。
- 选窗口的规则：分数 = 语音覆盖率 × 说话人切换次数，只用原始数据的元数据计算，不看任何系统输出。另外要求：
  - 音乐和无语音的比例低；
  - 有已审核的轮次（这样才有银标参考）；
  - 每个视频只取一段；
  - 优先选不同的赛事和解说。
- 这批数据从未用于 MIST 或 B 的训练和调参。这里只做推理和试听，不训练，也不调任何参数。

## 5. 试听网页

- 用 claude.ai 的私有 Artifact 网页。每个样本提供四个版本：
  - 同传混音（原声压低，译音在上）；
  - 左右声道（左原声、右译音）；
  - 只听译音；
  - 只听原声。
- 页面上有字幕（Thinker 的译文，按说话人着色），点击可跳转。
- 每个样本标出说话人数、窗口数、银标 ASR-BLEU 和延迟（LAAL）。
- 音频用 MP3。总大小控制在 Artifact 的限制之内：单个文件不超过 15 MB，单次发布不超过 64 MB。

## 6. GPU

8 张卡当前全是不存 checkpoint 的占位程序，没有其他代理的实验。编排器按惯例按 PID 停掉占位程序、接管卡（`--cap135`）。guard 不停，跑完后由它补回占位。

## 7. 产物位置

- 数据：`data/game_esports_si_v1/{zh2en,en2zh}/`（session_audio、diar、refs）。
- 窗口：`eval_outputs/mist_w3po_v1/data/windows_game_{zh2en,en2zh}_c2g06_v1/`。
- 解码和语音：`eval_outputs/mist_w3po_v1/{decode,speech}/game_b_{zh2en,en2zh}*`。
- 试听包：`demo_packages/game_multispk_b_demo_v1/`，并发布为 Artifact。
- 过程记录：`reports/mist_w3po_v1/LOG.zh-CN.md`。
