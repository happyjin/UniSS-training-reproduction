# X-Translator（RealSI 句级）作为同传 S2ST 基线：登记 v1

写于 2026-11-12 20:40（布拉格时间）。此时还没有在 RealSI 上跑过任何 X-Translator 输出。执行者：R4。

- 总则见 R1 的 `paper/baselines_run_plan_v1.md`（tracked 副本 `reports/s2st_baselines_v1/baselines_run_plan_v1.md`）§1–2：协议就是论文自己的代码。
- 本文件的 tracked 副本：`reports/s2st_baselines_v1/baselines_run_plan_xtranslator_v1.md`（`paper/` 不在 git 里）。

规则：
- 下面的网格和所有设置现在固定。RealSI 只打分，不调参。网格外不加点，结果出来以后也不改网格。
- 只做推理：只用官方发布的权重和官方代码，不训练、不微调。
- 我们自己的系统不因这些结果改任何东西。
- 不报 COMET 或 MetricX。

## 1. 系统与版本（X-Translator 自己的默认组合）

| 项 | 取值 |
|---|---|
| 代码 | GitHub `zhaoyx239/X-Translator` commit `dc6a59a`（MIT）。本地：`/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Translator`，不改动 |
| ASR | `Qwen/Qwen3-ASR-1.7B`（Apache-2.0，HF `7278e1e7`），用它自己的 adapter `server/qwen3asr/qwen3asr_server.py`，qwen-asr 0.0.6 的 vLLM 流式后端 |
| MT | `NiuTrans/LMT-60-8B`（Apache-2.0，HF `c77b9bf3`），按 `server/LMT/start_lmt_vllm.sh` 用 vLLM OpenAI 服务起，served-model-name `lmt`，bf16，max-model-len 1024 |
| TTS | X-Voice 0.4B Stage2（权重 CC-BY-NC-4.0，HF `XRXRX/X-Voice` `7f24fe77`；代码 `sunnyxrxrx/X-Voice` `817bcdd`，MIT），用它自己的 `server/xvoice/server.py`，参数取 `start_x_voice.sh`：`--stage 2 --no-auto-detect-lang`，ckpt `XVoice_Base_Stage2/model_70000`，语速预测 `SpeedPredictor/model_28000`。vocoder 是 `charactr/vocos-mel-24khz`（`0feb3fdd`） |
| 说话人提示 | `backend/speaker_prompt.py` 的 SpeakerPromptTracker，CAM++ `iic/speech_campplus_sv_zh-cn_16k-common`（ModelScope，Apache-2.0）。参数取 config.json：clone_window 6 s，窗 1.0 s，hop 0.5 s，change threshold 0.45 |
| 环境 | 新建专用 venv `/opt/dlami/nvme/jasonleeeli/venvs/xtranslator`（Python 3.12.13，由 uniss-eval 的解释器建）：`qwen-asr[vllm]==0.0.6`（vllm 0.14.0、transformers 4.57.6）；X-Voice 用 `--no-deps` 装，它的依赖逐个装，并用约束文件锁住 torch / transformers / numpy |
| espeak-ng | X-Voice 锁定的 commit `b0b605c8` 编译到 venv 自己的前缀下，通过 `PHONEMIZER_ESPEAK_LIBRARY` / `PHONEMIZER_ESPEAK_DATA_PATH` 指过去。不跑官方的 `prepare_ipa.sh`：它会装到 `/usr`，还会删系统库 |
| 权重目录 | `/opt/dlami/nvme/jasonleeeli/models/xtranslator/` |

所有资源参数（端口、`--gpu-memory-utilization`、实例数）只影响资源，不影响解码。
- adapter 里 Qwen3-ASR 的端口写死为 8001，没法开多个实例。所以用一个小启动器 import 它的模块，照抄 `main()`，只换端口。
- 评测代码没有发布（README 的 TODO 写着“Release evaluation code”，还没勾）。评测完全按我们的协议做。

## 2. 驱动方式（模拟源时钟，不计算耗时）

不经过网页，写一个同步驱动 `experiments/baselines_xtranslator_v1/xt_realsi.py`。它不改动地调用 X-Translator 的以下部件：
- `Qwen3StreamingRecognizer`：承诺层；
- `Qwen3ASR` / `Qwen3ASRService`：HTTP 客户端；
- `translation_segment_pairs_for_text` / `estimate_segment_source_audio_durations`：切段；
- `OpenAIChatTranslateService`：LMT 原版 prompt，temperature 0，top_k 20，top_p 0.6，max_tokens 256；
- `XVoiceTTSService`：输出 48 kHz 16-bit PCM；
- `SpeakerPromptTracker`。

被替换的只有 `backend/session.py` 里 asyncio / websocket 这层胶水。驱动逐条照抄它的 qwen3 路径：
- `_handle_audio`、`_next_asr_chunk`、`_process_asr_chunk`、`_emit_asr_output`（含去重规则）；
- `_request_asr_boundary_flush`（说话人切换）、`_flush_asr_finalize`、`_reset_asr_turn`；
- `_tts_loop`：翻译、取说话人提示、合成，失败就跳过该段。

时钟与提交：
- 每个 RealSI 句子是一个独立会话（新的 ASR / 说话人状态，上下文每句清空），与 si_infer 一样按 start/end 从整段 talk wav 切出 16 kHz 单声道。
- 音频按前端原生的 512 样本帧（32 ms）送入。时钟 = 已送入的源音频时长；计算时间不计入（computation-unaware，与论文相同）。
- 每帧先进 ASR 缓冲，再送说话人跟踪器；如果它报说话人切换，就按原逻辑做边界 flush。
- ASR 缓冲一满一个请求块（默认 2560 样本），就立即处理。实时系统落后时会丢掉积压音频，模拟中不会发生。
- 句末等同前端发出 `stop`：处理剩余缓冲，再 `finalize`。
- 每当 ASR 产生一个 final（标点提交、静音端点、边界 flush 或句末 finalize），切出的每个翻译段都成为一个块：
  - `write_ms` = 该时刻已送入的源音频（ms）；
  - 该段在 `write_ms` 立刻翻译，并用当时导出的说话人提示合成（因果）。
- 播放：块从 max(write_ms, 上一块结束) 开始，不加计算时间。即论文的队列模型。
- TTS 音频：取 X-Translator 交给前端的 48 kHz PCM，用 soxr 降到 16 kHz，拼成时间线 wav 和只含语音的 wav。
- 合成失败或译文为空的段（X-Voice 返回错误）：在原系统里就是没有声音。这样的段不成块，文字记进 `unspoken_text`，个数单独报。
- `translated_text` = 各段 MT 输出按顺序拼接：中文直接拼，英文用空格拼。英文块从第二块起开头加一个空格。
- 每个点每个方向只跑一次。LMT 是贪心；X-Voice 的采样由它自己的服务器决定，不另设种子（与发布代码相同）。

结果行的格式与 R1 的 SeamlessStreaming（`experiments/s2st_baselines_v1/seamless/run_seamless.py`）相同：
- 字段：`id`、`tgt_lang`、`translation_ref`、`translated_text`、`audio_path`（时间线）、`speech_path`（只含语音）、`source_duration_ms`；
- `blocks[{write_ms,start_ms,dur_ms,text}]`、`start_offset_ms`、`end_offset_ms`、`unspoken_text`、`config`；
- 另加 X-Translator 自己的源段文本、说话人和提示路径。

## 3. 固定网格（每个方向 4 个点）

按代码确认的事实：
- `config.json` 的 `asr.chunk_seconds=1.6`、`translation.emit_min_chars`、`emit_interval_seconds` 在 qwen3 路径里都没有被用到。
- 真正起作用的请求块是 `asr.paraformer_chunk_size`（默认 2560 样本 = 0.16 s）。它同时决定静音端点：连续 8 块 RMS < 0.01 才算端点，默认 1.28 s。
- ASR 服务端每 0.6 s（`chunk_size_sec`，adapter 里写死）解码一次，`unfixed_chunk_num` 为 2，`unfixed_token_num` 为 5。
- 提交规则：文本里至少有 2 个标点时，提交到倒数第 2 个标点为止。

| 点 | 改动（其余全部为发布默认） | 说明 |
|---|---|---|
| `xt-default` | 无 | **主点**，发布的 config.json 原样（只改服务地址、方向） |
| `xt-c1280` | `asr.paraformer_chunk_size` = 1280 | config 暴露的旋钮：请求块 0.08 s，静音端点 0.64 s |
| `xt-c5120` | `asr.paraformer_chunk_size` = 5120 | config 暴露的旋钮：请求块 0.32 s，静音端点 2.56 s |
| `xt-a1.6` | Qwen3-ASR 服务端 `chunk_size_sec` = 1.6 | 论文附录 C 写的 “1.6-s accumulated request chunk”。config 的 `chunk_seconds=1.6` 在代码里不起作用，这里把 1.6 s 用在真正的解码块上。次要点 |

预期（写在看到结果之前）：
- 提交需要两个标点，所以句级输入里大部分内容会在句末 finalize 时才提交，翻译也按段独立进行、不带上下文。
- 因此 D1 很可能超过 m3、甚至 m4 的预算，m1 几乎肯定没有点。
- 某档没有点就报“--”，不另加点。不管落在哪一档，`xt-default` 都单独报一行（ASR-BLEU 和它的 D1）。

## 4. 协议（与论文 QABC 点、R1 / R2 的基线相同）

- **数据**：`si_infer.realsi_items()`，En→Zh 346 句，Zh→En 431 句；参考 = `trg_text`。
- **ASR-BLEU**：
  1. `reports/uniss_si_v4/speech_only.py`；
  2. `evaluation.asr_transcribe --batch-size 4`，英文用 Whisper-large-v3，中文用 Paraformer；用 uniss-eval 环境，设好 `HF_HOME` / `MODELSCOPE_CACHE`；
  3. `evaluation/text_metrics.corpus_bleu`。缺的 ASR 行按空串计，ASR 覆盖率必须 ≥ 99%。
- **文本 BLEU**：`translated_text` 对 `translation_ref`，用同一个函数。
- **延迟**：
  - D1 = 论文的段 LAAL（`collect_results.realsi_points`：每个单位记它所在块的 `start_ms`，含排队）；
  - D0w = `laal_defs.sentence_defs()` 的 D0w（单位记块的 `write_ms`）；
  - 都是句均值，与 R1 的 `score_seamless.score()` 用同一份代码。打分直接 import 它，不另写公式。
- **分档**：每个方向、每个预算，取 D1 ≤ 预算的点里 ASR-BLEU 最高的；没有就报“--”。另附按 D0w 分档，只作参考。
  - En→Zh 预算：1.18 / 1.98 / 2.46 / 2.98 s；
  - Zh→En 预算：1.18 / 1.92 / 2.40 / 2.98 s。

## 5. 输出

- 代码：`experiments/baselines_xtranslator_v1/`。
- 运行目录：`eval_outputs/baselines_xtranslator_v1/<点>/realsi_<方向>/`。
- `paper/baselines_results/xtranslator/<点>_<方向>.json`：字段与 seamless / hysimulmt 一致，即 asr_bleu、text_bleu、laal_d1、laal_d0w、n、无输出句数、配置、run 路径，另加 X-Translator 的计数（TTS 失败段、边界 flush、块/句）。
- 同目录还有 `tiers.json` 和 `summary.md`。tracked 副本放 `reports/s2st_baselines_v1/xtranslator_results/`。
