# Hy-SimulMT + Qwen3-TTS（RealSI 句级）参考结果：登记 v1

写于 2026-11-12 20:10（布拉格），在用 Qwen3-TTS 合成任何 RealSI 块之前。执行者：R3。本文件的 tracked 副本：
`reports/s2st_baselines_v1/baselines_run_plan_hysimulmt_qwen3tts_v1.md`（`paper/` 不在 git 里）。
这是 R2 登记（`paper/baselines_run_plan_hysimulmt_v1.md`）§6 预告的次要结果。

规则：
- 只做推理：官方权重，不训练、不微调、不改任何模型。RealSI 只打分，不调参。
- 只有一种设置（下面固定）；结果出来以后不改设置、不加点。
- 只替换语音生成器：Hy-SimulMT 的提交文本块、时间轴排队模型、ASR、BLEU、延迟公式、分档规则都与 R2 的
  “Hy-SimulMT + 论文 talker”完全相同。Hy-SimulMT 不重跑。
- 论文不改；结果先给用户看。

## 1. 模型与环境

| 项 | 取值 |
|---|---|
| 权重 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，HF revision `5d83992436eae1d760afd27aff78a71d676296fc`（2026-01-29），本地 `/opt/dlami/nvme/jasonleeeli/models/Qwen3-TTS-12Hz-0.6B-Base`（含 `speech_tokenizer/`，不改动） |
| 为什么是 Base | 0.6B 有 Base 和 CustomVoice 两个版本；只有 Base 能做声音克隆（CustomVoice 只有预置音色）。SimulInfer（Hy-SimulMT 官方推理代码，S3-S2ST 的语音后端）做“输入音色克隆”时也要求 Base（`config.clone_source_enabled` / `tts_launch.py`） |
| 推理代码 | `qwen-tts 0.1.1` 的 `Qwen3TTSModel.generate_voice_clone`，在已有专用 venv `/opt/dlami/nvme/jasonleeeli/venvs/tts_schemes`（Python 3.12，torch 2.6.0+cu124，transformers 4.57.3）里跑，不安装任何东西 |
| 精度 | bf16，`attn_implementation="sdpa"`（与 SimulInfer `tts_backends/qwen.py` 相同；该 venv 没有 flash-attn） |

## 2. 设置（固定）

- **文本块**：R2 的提交块，原文件 `eval_outputs/baselines_hysimulmt_v1/thinker/<点>/blocks_<方向>.jsonl`
  （tracked 副本 `reports/s2st_baselines_v1/hysimulmt_results/blocks_*.jsonl`），两个原生点 `hy-std-w4`、`hy-sc-w4`，两个方向。
  每个提交块一次 TTS 请求，文本 = 块文本去掉首尾空白（英文块开头的连接空格不送进 TTS；词中断开的块照原样单独合成）。
  去掉空白后没有任何字母/数字/汉字的块（只有标点）不合成、语音长度为 0，与 SimulInfer `tts.Speech.run` 相同
  （共 4 块：std Zh→En 1、sc En→Zh 2、sc Zh→En 1）。
- **音色**：克隆源说话人。参考音频 = 该句源音频从句首到第一次提交时刻（至少 1 s；Hy-SimulMT 第一次提交不早于 1.6 s），
  与论文 talker（`si_infer`）取的参考片段完全相同，因果；按 SimulInfer `SourceVoice` 的规则做峰值归一（峰值 < 0.5 或 > 0.95
  时缩放到 0.7）。用 **x-vector only** 模式（只用说话人向量，`x_vector_only_mode=True`，与 SimulInfer 用源音频克隆时的请求相同）：
  部分片段没有对应的准确转写，不用 ICL 模式。同一句的所有块用同一个说话人向量；不用前面块的合成语音做历史参考。
- **文本输入**：`language` = `Chinese`（En→Zh）/ `English`（Zh→En）；`non_streaming_mode=True`（SimulInfer 在只用说话人向量时的设置）。
- **解码**：checkpoint 的 `generation_config.json`，不改任何值：`do_sample=true, temperature=0.9, top_k=50, top_p=1.0,
  repetition_penalty=1.05`，sub-talker 同为 `0.9 / 50 / 1.0`，`max_new_tokens=8192`。
- **种子**：采样解码，按论文惯例跑 **三个种子**（0 / 1 / 2）：冒烟测试显示 batch 32 时 RTF 约 0.11，三个种子总共只需几十 GPU 分钟。
  块按文本长度排序后固定分成每批 32 块（批的组成与种子无关、可复现），每批 `torch.manual_seed(种子基数 + 批序号)`，
  种子基数 20261112 / 1000000 / 2000000。
- **时间轴**（与 `si_infer` 相同，不计合成耗时）：块 i 的开始播放时刻 = max(write_ms_i, 上一块播放结束)，播放时长 = 合成音频长度。
  Qwen3-TTS 输出 24 kHz，先重采样到 16 kHz（`librosa.resample` 默认 soxr_hq）再拼时间轴和 speech-only 音频
  （ASR 的 Whisper 要求 16 kHz；论文 talker 的输出也是 16 kHz）。没有语速控制（Qwen3-TTS 没有，按原速）。
- **与 SimulInfer 完整 TTS 流程的差别**（按本协议不用）：它把相隔 < 650 ms 的文本攒成一段（`tts_min_chars 40 / tts_batch_ms 650`）、
  有“成对历史”ICL 续写、Zh→En 播放加速 1.2 倍（atempo）、源音色快照取会话前 2.4 s 再每 10 s 更新。本协议固定为“每个提交块一次合成 +
  论文的排队模型”，与 R2 的论文 talker 运行一致，这样两者只差语音生成器。

## 3. 打分（与 R2 `experiments/baselines_hysimulmt_v1/score_hy.py` 相同）

- **数据**：`si_infer.realsi_items()`，En→Zh 346 句、Zh→En 431 句。
- **ASR-BLEU**：`reports/uniss_si_v4/speech_only.py` → `evaluation.asr_transcribe --batch-size 4`（英文 Whisper-large-v3，中文 Paraformer）
  → `evaluation/text_metrics.corpus_bleu`（规范化 + sacreBLEU）。分母为全部句子；缺的 ASR 行按空串计；ASR 覆盖率 ≥ 99%。
- **文本 BLEU**：提交文本，与 R2 相同（应与 R2 的数一致，作为检查）。
- **延迟**：D1 = `collect_results.realsi_points` 的段 LAAL（单位记所在块的开始播放时刻 `start_ms`，含排队）；
  D0w = `laal_defs.sentence_defs()` 的 D0w（单位记所在块的写出时刻）。一个点 = 三个种子的均值（ASR-BLEU 也是）。
- **分档**：取 D1 ≤ 预算的点里 ASR-BLEU 最高的（En→Zh 1.18 / 1.98 / 2.46 / 2.98 s；Zh→En 1.18 / 1.92 / 2.40 / 2.98 s）；另附按 D0w 分档，仅作参考。
- 不报 COMET / MetricX。

## 4. 输出

- 代码：`experiments/baselines_hysimulmt_qwen3tts_v1/`（任务、TTS worker、时间轴拼接、打分）。运行输出：`eval_outputs/baselines_hysimulmt_qwen3tts_v1/`。
- 结果：`paper/baselines_results/hysimulmt_qwen3tts/<点>_<方向>.json`（asr_bleu、text_bleu、laal_d1、laal_d0w、n、config、各种子）、
  `tiers.json`、`summary.md`（含与“Hy-SimulMT + 论文 talker”的并列表）；tracked 副本 `reports/s2st_baselines_v1/hysimulmt_qwen3tts_results/`。
