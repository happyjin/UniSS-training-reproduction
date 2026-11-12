# SeamlessStreaming（RealSI 句级 S2ST）作为同传基线：登记 v1

写于 2026-11-12 16:05（布拉格），在 RealSI 上跑任何 SeamlessStreaming 输出之前。执行者：R1。总则见
`paper/baselines_run_plan_v1.md`（tracked 副本 `reports/s2st_baselines_v1/baselines_run_plan_v1.md`）§1–2：协议就是论文
自己的代码。本文件的 tracked 副本：`reports/s2st_baselines_v1/baselines_run_plan_seamless_v1.md`。

规则：
- 只做推理：官方发布的权重 + 官方 agent，不做任何训练、微调或适配。网格只扫 SeamlessStreaming agent 自带的推理延迟参数。
- 下面的网格和所有设置现在固定。RealSI 只打分，不调参；网格外不加点，结果出来以后也不改网格。
- 网格范围只在 dev（`data/si_spont_v1`，非测试集）上校准过，见 §3。
- 我们自己的系统不因这些结果改任何东西。

## 1. 模型、代码与环境

| 项 | 取值 |
|---|---|
| 权重 | HF `facebook/seamless-streaming`，repo commit `82746da73c2927b962e53ad2f723babf1ae0e0b5`，本地 `/opt/dlami/nvme/jasonleeeli/models/seamless-streaming`（协调者下载，不改动）。sha256：`seamless_streaming_unity.pt` ce7dfdb0…89eb，`seamless_streaming_monotonic_decoder.pt` 7797deda…335c，`vocoder_v2.pt` 20c50c3e…ca13（与 HF 的 LFS sha256 一致）；`tokenizer.model` 026a7682…3f9d（与卡片里 seamless-m4t-large 的 tokenizer 是同一文件，xet hash 相同）；`spm_char_lang38_tc.model` 9e7f2075…b493（与 seamless-m4t-v2-large 的同名文件相同） |
| 推理代码 | GitHub `facebookresearch/seamless_communication` commit `9a081e935c29c6b1e0e05bcbf3fcb623ad306524`（本地 clone `/opt/dlami/nvme/jasonleeeli/src_baselines/seamless_communication`，未改动），`fairseq2==0.2.1`（fairseq2n 0.2.1），`simuleval==1.1.4`，`torch==2.2.2+cu121`，完整列表 `experiments/s2st_baselines_v1/seamless/pip_freeze.txt` |
| 环境 | 新建专用 venv `/opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines`，解释器 `conda_envs/multi_speaker/bin/python3.11`（协调者批准：fairseq2n 0.2.x 只有 cp38–cp311 的 wheel）。只往这个 venv 里装。安装脚本 `experiments/s2st_baselines_v1/seamless/install_env.sh` |
| 权重加载 | fairseq2 的 `@user` 卡片覆盖，把 checkpoint / tokenizer 指到上面的本地文件（不联网下载，不改卡片里的任何模型配置）。`SSL_CERT_FILE` 指向 venv 里的 certifi（conda 的 python3.11 缺 CA 证书） |

## 2. Agent 与固定设置

- **Agent**：官方 `seamless_communication.streaming.agents.seamless_streaming_s2st.SeamlessStreamingS2STAgent`，即官方
  `streaming_evaluate --task s2st` 用的 agent（非 expressive）：OnlineFeatureExtractor → OfflineWav2VecBertEncoder →
  UnitYMMATextDecoder（单调注意力文本解码器）→ NARUnitYUnitDecoder（非自回归 T2U）→ VocoderAgent（`vocoder_v2`，
  speaker id −1，16 kHz）。语音输出，不用目标说话人音色（这个 agent 不克隆音色）。
- **参数**：与官方 CLI（`cli/streaming/evaluate.py`，s2st 分支）完全相同：`source_segment_size=320` ms，
  `min_starting_wait_w2vbert=192`，`decision_threshold=0.5`，`no_early_stop=True`，`max_len_a=0`，`max_len_b=100`，
  `min_unit_chunk_size=50`；其余用 agent 默认值（`decision_method=min`，`p_choose_start_layer=0`，不开 `block_ngrams`，
  `d_factor=1.0`，`max_consecutive_write=50`，`shift_size=10` ms，`window_size=25` ms，不 `denormalize`）。
  只有 §3 网格里的三个推理延迟参数会改。
- **精度**：fp32。官方 README（`cli/streaming/README.md`）写明论文数字用 `--dtype fp32`；CLI 默认是 fp16。
- **解码确定**：文本、单元都是 argmax，每个点每个方向只跑一次。
- **驱动方式**：在进程里用 SimulEval 1.1.4 自己的构建函数（`simuleval.utils.agent.build_system_args`，与
  `streaming_evaluate` 相同）建 agent，再按 SimulEval `SentenceLevelEvaluator` 的循环原样喂数据：每句 `agent.reset()`；
  每步发 320 ms 源音频（`SpeechInputInstance.send_source` 的逻辑，最后一块不足 320 ms 时带 `finished`；读完后发
  `EmptySegment(finished=True)`）；`agent.pushpop`；直到输出 `finished`。每个非空输出语音段的延迟 = 发出时已读入的源时长
  （SimulEval `step_to_delay`，不计算耗时）。
  - 两个只做记录的子类（不改任何决策）：单元解码器每次写出单元时，记下自上次写出以来收到的文本 token；声码器每发出一段语音，
    就把这些 token 配给这一段。这就是每段语音说的文字（T2U 每次正是为这些 token 生成单元）。
  - 与官方 `streaming_evaluate` 的差别（都是为了用论文自己的协议）：
    1. 不用它的 dataloader。它默认用 silero-VAD 去掉句首句尾静音，而论文所有策略用的是同一个 RealSI 句子切片、同一个时钟。
       这里直接喂论文的切片（`si_infer.realsi_items()`：talk.wav[start:end]，16 kHz，float32，[-1, 1]，与官方 dataloader 的
       `AudioDecoder(float32)` 刻度相同）。
    2. 不用它的质量打分器（Whisper-large-v2 ASR-BLEU）和它的 StartOffset / EndOffset，改用论文的 ASR-BLEU 和 D1 / D0w（§4）。
- **数据**：RealSI 句级，`si_infer.realsi_items()` 原样（id = `<talk>_<k:03d>`，参考 = `trg_text`）。En→Zh 346 句，
  Zh→En 431 句。已核对：id、切片时长、参考与论文存的 QABC 运行逐条一致。

## 3. 固定网格（每个方向 8 个点，两个方向相同）

SeamlessStreaming agent 自带三个推理延迟参数：
- `decision_threshold` τ（文本解码器单调注意力的写出阈值，0–1，“越小延迟越低”）；
- `min_starting_wait_w2vbert` W（编码器开始前至少读入的 fbank 帧数，10 ms 一帧；官方 192 = 1.92 s）；
- `min_unit_chunk_size` C（单元解码器每次至少攒够的单元数，20 ms 一个；官方 50 = 1 s 语音）。

| 点 | τ | W（帧） | C（单元） | 说明 |
|---|---|---|---|---|
| `t0.1_w192_c50` | 0.1 | 192 | 50 | 官方 W/C，扫 τ |
| `t0.3_w192_c50` | 0.3 | 192 | 50 | 官方 W/C，扫 τ |
| `t0.5_w192_c50` | 0.5 | 192 | 50 | **官方默认**（`streaming_evaluate --task s2st`） |
| `t0.7_w192_c50` | 0.7 | 192 | 50 | 官方 W/C，扫 τ |
| `t0.9_w192_c50` | 0.9 | 192 | 50 | 官方 W/C，扫 τ |
| `t0.1_w32_c25` | 0.1 | 32 | 25 | 低延迟设置 |
| `t0.3_w32_c25` | 0.3 | 32 | 25 | 低延迟设置 |
| `t0.5_w32_c25` | 0.5 | 32 | 25 | 低延迟设置 |

机器可读：`experiments/s2st_baselines_v1/seamless/grid_v1.json`。其余参数一律按 §2 固定。

**dev 校准（只用 dev，非测试集）**：`data/si_spont_v1/dev_{en2zh,zh2en}_ref.jsonl` 各取前 60 句（自发语音，均长 7.2 / 6.4 s，
与 RealSI 的 7.4 / 6.7 s 相近），每个方向 15 个设置，只看文本 BLEU 和延迟（没跑 ASR）。输出
`eval_outputs/s2st_baselines_v1/seamless/dev_calib/`，表 `…/dev_calib/pilot_table.txt`。要点：
- τ 对延迟的影响很小：官方 W/C 下 τ 0.1→0.9，D1 En→Zh 只从 2.48 到 2.76 s，Zh→En 从 2.44 到 3.18 s。
- 减小 W、C 能再降一点：W32/C25 时 D1 最低 En→Zh 2.35 s、Zh→En 2.17 s（τ=0.1）。
- 再往下（W0/C10）D0w 更低，但 En→Zh 每句的段数翻倍（约 18 段），语音变长、排队变长，D1 反而升到 2.71 s；Zh→En 只比 W32/C25 低 0.01 s。所以不进网格。
- D1 里排队占了很大一块：例如 En→Zh τ0.2/W32/C25 的 StartOffset 中位数 1.28 s、D0w 1.64 s，而 D1 是 2.43 s。

**预期（写在看到 RealSI 结果之前）**：m1（1.18 s）和 m2（1.98 / 1.92 s）大概率没有点进得去，报“--”；m3（2.46 / 2.40 s）
可能只有低延迟设置进得去；m4（2.98 s）应能用官方设置。如果某档没有点进得去，就报“--”，不另加点，也不改网格。

## 4. 评测协议（论文自己的代码，与 `paper/baselines_run_plan_v1.md` §2 相同）

- **每句一行 `results.jsonl`**（si_infer 的格式）：`blocks` = 输出的语音段，按发出顺序，每段
  `write_ms`（发出时已读入的源时长，句内时钟，不计算耗时）、`start_ms`（开始播放 = max(write_ms, 上一段播完)，含排队）、
  `dur_ms`、`text`（这一段说的文字；一个被段边界切开的词 / 字母数字串算在它开头所在的那一段，`<unk>` 不算文字）。
  `speech_path` = 各段首尾相接、不加等待的纯语音（给 ASR）；`audio_path` = 带等待的时间线音频。
  `translated_text` = 文本解码器的全部输出（去掉 `<unk>`）。
- **ASR-BLEU（只用纯语音）**：`PYE reports/uniss_si_v4/speech_only.py <run>/realsi_<d>`，再
  `PYE -m evaluation.asr_transcribe --input results_speechonly.jsonl --output asr_speech.jsonl --device cuda:0 --batch-size 4`
  （英文 Whisper-large-v3，中文 Paraformer-large；`HF_HOME`、`MODELSCOPE_CACHE` 按 v1 §2）。
  `evaluation/text_metrics.corpus_bleu` 打分（英文 NFKC、小写、去标点，13a；中文 NFKC、t2s、去标点、按字，zh）。
  缺失的 ASR 按空串计，覆盖率要 ≥ 99%。全部句子都计入（没有语音的句子 = 空假设）。
- **Text BLEU**：`translated_text` 对 `translation_ref`，同一个函数。
- **D1**（主口径）：论文的段 LAAL，`collect_results.realsi_points` 原样：每个单位（中文：汉字 + 拉丁词；英文：词）取所在语音段的
  `start_ms`，`si_metrics.laal`，按句平均（没有输出单位的句子不计入延迟，与论文相同）。
- **D0w**（次口径）：`laal_defs.sentence_defs()` 的 `D0w`：同样的单位取所在段的 `write_ms`，不含排队（SimulEval 逐词 LAAL）。
- 另外记录（只作参考，不用于分档）：SimulEval 默认的语音 LAAL（D0，每段一个延迟）、D2、StartOffset / EndOffset 中位数。
- **分档**：同一方向、同一预算内，取 D1 不超过预算的点中 ASR-BLEU 最高的；没有点进得去就报“--”。预算 = SimulS2ST-Omni 的
  m1–m4：En→Zh 1.18 / 1.98 / 2.46 / 2.98 s，Zh→En 1.18 / 1.92 / 2.40 / 2.98 s。按 D0w 再分一次档，作为次要列。
- **代码**：`experiments/s2st_baselines_v1/seamless/`：`run_seamless.py`（生成）、`score_seamless.py`（打分，每个点每个方向一个
  json）、`realsi_queue.py`（排队与 GPU 守护）、`summarize_seamless.py`（汇总与分档）、`start_realsi.sh`（启动）。
- **输出**：生成 `eval_outputs/s2st_baselines_v1/seamless/realsi/<点>/realsi_<d>/`；结果
  `paper/baselines_results/seamless/<点>_<d>.json`（asr_bleu、text_bleu、laal_d1、laal_d0w、n_sentences、config 等）和
  `paper/baselines_results/seamless/summary.md`。

## 5. 计算与 GPU

- 每个进程一份 fp32 模型，峰值约 12 GiB；每个点每个方向切 4（En→Zh）/ 5（Zh→En）个分片，多进程并行。
- 4–7 号卡：上卡前看 nvidia-smi 实际占用，新作业 12 GiB + 已有占用 + 6 GiB 余量 ≤ 135 GiB；同一张卡上相邻两次启动隔 ≥ 60 s；
  别人的 GPU 进程（如 vLLM 引擎）在这张卡上出现不到 90 s 时不启动。某卡总占用超过 135 GiB 时，杀掉我在这张卡上最新的作业
  （按 PID、核对命令行），之后 `--resume` 续跑。
- 0–3 号卡（D1v 训练）：只在训练稳定（≥ 20 步）后叠放，我的作业每卡 ≤ 3 个（≤ 45 GiB）；训练日志出现 OOM / 报错，或最近 15 步的
  中位步长超过叠放前基线的 1.4 倍，就立即撤掉我在 0–3 上的全部作业并停用这些卡，同时报告协调者。
- 不碰别人的进程；只按 PID 杀自己的作业。

## 6. 文件

- 登记：本文件；tracked 副本 `reports/s2st_baselines_v1/baselines_run_plan_seamless_v1.md`。
- dev 校准表：`eval_outputs/s2st_baselines_v1/seamless/dev_calib/pilot_table.txt`（每行：dev 方向、设置、text BLEU、D1、D0w、
  D0、StartOffset 中位数、每句段数）。
