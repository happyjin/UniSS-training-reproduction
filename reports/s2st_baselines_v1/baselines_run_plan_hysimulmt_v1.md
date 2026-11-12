# Hy-SimulMT-30B-A3B（RealSI 句级）作为同传 S2ST 基线：登记 v1

写于 2026-11-12 15:30（布拉格），在 RealSI 上跑任何 Hy-SimulMT 输出之前。执行者：R2。总则见 R1 的
`paper/baselines_run_plan_v1.md`（tracked 副本 `reports/s2st_baselines_v1/baselines_run_plan_v1.md`）§1–2：
协议就是论文自己的代码。本文件的 tracked 副本：`reports/s2st_baselines_v1/baselines_run_plan_hysimulmt_v1.md`
（`paper/` 不在 git 里）。

规则：
- 下面的网格和所有设置现在固定。RealSI 只打分，不调参；网格外不加点，结果出来以后也不改网格。
- 我们自己的系统不因这些结果改任何东西。
- 只替换 Thinker 一侧（策略 + 翻译）：Talker、ASR、BLEU、延迟公式、分档规则都与论文的 QABC 点相同。

## 1. 模型与服务

| 项 | 取值 |
|---|---|
| 权重 | `tencent/Hy-SimulMT-30B-A3B`（仅译文输出，bf16，HYV3SpeechForCausalLM，带 audio tower），HF tree `56c233df75bf09e386aa6282b01730b461afc83f`，本地 `/opt/dlami/nvme/jasonleeeli/models/Hy-SimulMT-30B-A3B`（不改动） |
| 推理代码 | GitHub `Tencent-Hunyuan/Hy-SimulMT` commit `7efcb18c521a7025b37d4bc2230a1df8ade83950`，SimulInfer 0.5.18 |
| 服务 | 新建专用 venv `/opt/dlami/nvme/jasonleeeli/venvs/hysimulmt`（Python 3.12.13）：`vllm[audio]==0.31.0`、`simulst==0.1.0`、SimulInfer 0.5.18（`--no-deps`，提供 n-gram / 目标语言 logits 处理器和 FireRed VAD）、`fireredvad[cpu]` |
| 模型目录 | 按 SimulInfer `deploy.py stage_model` 的做法另建一个目录：权重软链接、其余小文件复制，再跑 `simulinfer-prepare-model` 写入 SimulInfer 0.5.18 的 `openai.jinja` 和 `generation_config.json`。原下载目录不改 |
| vLLM 命令 | 与 `deploy.py model_command` 相同：`vllm serve <staged> --served-model-name Hy-SimulMT-30B-A3B --dtype bfloat16 --tensor-parallel-size 1 --enable-prefix-caching --enable-chunked-prefill --limit-mm-per-prompt '{"audio":1024}' --chat-template-content-format openai --logits-processors simulinfer.logits_plugins:NoChineseTargetProcessor`；只改资源参数：`--max-num-seqs 64`、`--gpu-memory-utilization 0.6`（单卡，与别的作业叠放） |
| 解码 | SimulInfer 自己的设置：贪心（temperature 0），每次最多 64 token（ST 256），no-repeat n-gram 3/3，ST repetition penalty 1.05，Zh→En 屏蔽目标汉字（`no_chinese_target=true`，部署配置 `configs/deploy/target_only.toml` 的值） |

**驱动方式**：不经过网页，直接在进程里调用 SimulInfer 自己的 `simulinfer.translation.VLLMSession`，与它的文件上传路径
（`app.translate`：`push(全部音频)` 再 `finish()`）完全一样。每个 RealSI 句子是一个独立会话（句级，上下文每句清空）。
推理设置取 SimulInfer `Settings` 的默认值，加上部署配置 `configs/deploy/target_only.toml` 的 `[inference]` 段
（first_chunk_ms 1600、ngram 3/3、model_logits_processors、no_chinese_target、simulst_crop_speech=false、
st_crop_speech=true）；`audio_mode=standard`，FireRed 流式 VAD 打开（默认），ST 前的 FireRed 整段 VAD 打开（默认）。
多个句子并发发给同一个 vLLM 服务；不计算耗时（computation-unaware，与论文相同）。

脚本：`experiments/baselines_hysimulmt_v1/`（驱动、blocks 转换、打分）。输出：`eval_outputs/baselines_hysimulmt_v1/`。

## 2. 固定网格（每个方向 6 个点）

Hy-SimulMT 的策略（`VLLMSession.step`）：每读完一块音频调用一次模型，模型输出非空就提交（EMIT，只追加，不修改已提交的文本），
空就等待（WAIT）；连续 `st_wait_chunks` 个有声 WAIT 块、或句尾/尾部静音时，对未翻译的音频做一次整段翻译（ST 补译）并提交。
SimulInfer 校验只允许首块 1600 ms、后续块 800 或 1600 ms。它暴露的延迟参数就是块长和 `st_wait_chunks`（≥0 的整数，默认 4）。

| 点 | 块长（首块 / 后续） | st_wait_chunks | 说明 |
|---|---|---|---|
| `hy-std-w4` | 1600 / 1600 ms | 4 | 原生 standard（默认、推荐） |
| `hy-sc-w4` | 1600 / 800 ms | 4 | 原生 short_chunk |
| `hy-std-w2` | 1600 / 1600 ms | 2 | 暴露参数，非默认 |
| `hy-sc-w2` | 1600 / 800 ms | 2 | 暴露参数，非默认 |
| `hy-std-w1` | 1600 / 1600 ms | 1 | 暴露参数，非默认（最激进） |
| `hy-sc-w1` | 1600 / 800 ms | 1 | 暴露参数，非默认（最激进） |

其他参数一律默认。Hy-SimulMT 是贪心解码，每个点每个方向只跑一次 Thinker；Talker 跑三个种子。

预期（写在看到结果之前）：首块固定 1.6 s，第一次提交不早于 1.6 s（短于 1.6 s 的句子在句尾提交），所以 m1（1.18 s）
大概率没有点能进；网格的目标是覆盖 m2–m4。如果所有点都高于某档预算，该档报“--”，不另加点。

## 3. 提交 → Talker 块的映射

- SimulInfer 的每个事件（`chunk` 或句尾的 `flush`）带 `end_ms` = 该事件发生时已读入的源音频时长（句内时钟，ms）。
  事件的 `text` 非空（去掉首尾空白后）就成为一个 Talker 块：`write_ms = end_ms`，`text` = 该事件提交的文本。
  句尾那一块的 `end_ms` 就是句长。
- 中文目标：块文本去掉首尾空白，原样。
- 英文目标：块文本去掉首尾空白；从第二块起，开头加一个空格。例外：如果这次提交在词中间接着上一次（上一次提交的原文以字母或数字结尾，
  这一次以字母或数字开头，且两者之间没有空白），就不加空格，原样交给 Talker。
- `final`：最后一个非空块记 `final=true`，`write_ms` 保持它的真实提交时刻。如果 Hy-SimulMT 的最后一次提交早于句尾（句尾那一块
  没有新文本），不把它挪到句尾：挪了等于人为加延迟。这种句子的数目在结果里报出。
- `translation` = Hy-SimulMT 的 `full_text`（各次提交原样拼接）。文本 BLEU 用它。
- 每个 id 都写进 blocks 文件。一句也没提交的句子 `blocks` 为空，si_infer 会跳过它：这些句子的 ASR 文本和译文都按空串计
  （BLEU 分母仍是全部 346 / 431 句），D1 / D0w 与论文代码一样跳过没有输出单位的句子；数目单独报。

## 4. 协议（与论文 QABC 点相同）

- **数据**：`si_infer.realsi_items()`，En→Zh 346 句、Zh→En 431 句；句子音频 = 整段 talk wav 按 start/end 切出（16 kHz 单声道，与 si_infer 相同）。
- **Talker**：streaming UniSS T1c@1500（`eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss`），
  `si_infer --text-policy file --min-words 2 --speed-policy block --history-ban 8 --gpu-mem 0.15 --num-shards 2 --limit 0`，
  在 vllm_omni_030 里跑（不安装任何东西），种子：默认 20261021、1、2；`--window` 只是标签（块长秒数）。块从
  max(写出时刻, 上一块结束) 开始播放，不加计算时间。
- **ASR-BLEU**：`reports/uniss_si_v4/speech_only.py`（只取语音段）→ `evaluation.asr_transcribe --batch-size 4`
  （英文 Whisper-large-v3，中文 Paraformer）→ `evaluation/text_metrics.corpus_bleu`（规范化后 sacreBLEU，中文 `zh`、英文 `13a`）。
  缺的 ASR 行按空串计，ASR 覆盖率 ≥ 99%（相对 Talker 结果行）。
- **文本 BLEU**：`translation`（committed text）对 `translation_ref`，同一函数。
- **延迟**：D1 = 论文的段 LAAL（`collect_results.realsi_points`：每个单位记它所在块的开始播放时刻 `start_ms`，含排队）；
  D0w = `laal_defs.sentence_defs()` 的 D0w（单位记所在块的写出时刻，不含排队）。都是句均值；一个点 = 三个 Talker 种子的均值。
  ASR-BLEU 也是三个种子的均值（与论文相同）。
- **分档**：每个方向、每个预算（En→Zh 1.18 / 1.98 / 2.46 / 2.98 s，Zh→En 1.18 / 1.92 / 2.40 / 2.98 s；m4 作补充），
  取 D1 ≤ 预算的点里 ASR-BLEU 最高的；没有就报“--”。另附按 D0w 分档的结果，只作参考。
- **不报** COMET / MetricX。

## 5. 输出

- `paper/baselines_results/hysimulmt/<点>_<方向>.json`：asr_bleu（三种子均值和各种子）、text_bleu、laal_d1、laal_d0w、n、
  无输出句数、最后提交早于句尾的句数、配置、各 run 的路径。
- `paper/baselines_results/hysimulmt/summary.md`：所有点和分档结果。

## 6. 可选（次要，单独标注）

时间允许时，再用 Qwen3-TTS 0.6B（S3-S2ST 的后端，`venvs/tts_schemes` 里的 qwen-tts）朗读同样的块，作为“Hy-SimulMT + Qwen3-TTS”
的次要结果，明确标为 secondary，不进主表。具体设置在跑之前另行登记。

## 7. 修订 1（2026-11-12 15:40 布拉格，仍在任何 RealSI 运行之前）

用户经协调者明确：基线只跑推理；Hy-SimulMT 只用官方发布的权重和官方推理代码的**两档原生设置**，不做任何训练、微调或适配；
语音部分接论文已训好的 talker，同样只做推理。因此：

- 网格缩为每个方向 **2 个点**：`hy-std-w4`（standard，1600 ms 块）和 `hy-sc-w4`（short_chunk，首块 1600 ms、之后 800 ms）。
  `st_wait_chunks` 保持官方默认 4。§2 表中 `hy-std-w2 / hy-sc-w2 / hy-std-w1 / hy-sc-w1` 四个点撤销，不跑。
- 其余（§1 服务与驱动、§3 映射、§4 协议与分档、§5 输出）不变。驱动只调用 SimulInfer 0.5.18 自己的 `VLLMSession`，不改它的代码；
  vLLM 只改资源参数（并发数、显存比例、上下文上限），不影响解码。
- 预期仍是：m1 大概率无点；两个点落在哪些档由结果决定，不加点。
