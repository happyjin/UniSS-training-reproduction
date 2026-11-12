# QABC 提交块 + Qwen3-Omni 原生 Talker（RealSI 流式）：登记 v1

写于 2026-11-12 20:25（布拉格），在 Qwen3-Omni 原生 talker 合成任何 RealSI 块之前。执行者：R5。
本文件的 tracked 副本：`reports/s2st_baselines_v1/baselines_run_plan_qabc_omnitalker_v1.md`（`paper/` 不在 git 里）。

目的：离线 CVSS-T 表里 Qwen3-Omni 原生 Talker 已经低于论文的 talker。这里在**流式** RealSI 设置下做同样的比较：
文本侧完全固定（QABC 在论文 tier 点上的提交块和写出时刻），只把语音生成器换成 Qwen3-Omni-30B-A3B-Instruct 自己的
Thinker → Talker → code2wav，两者用完全相同的协议打分。

规则：
- 只做推理：官方权重，不训练、不微调、不改模型。RealSI 只打分，不调任何东西。
- 只有一种设置（下面固定）；结果出来以后不改设置、不加点、不重跑挑结果。
- 只替换语音生成器：提交块（文本、write_ms）、排队模型、ASR、BLEU、延迟公式、分档规则都与论文相同。QABC 不重跑。
- 论文不改；结果交给协调者整合。

## 1. 点（论文 tier 表 `tab_realsi_main.tex` 的 QABC 列，m1–m4，两个方向）

提交块取自论文存下的 run：`eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500/system_<族>[-seed1|-seed2]_m<窗>/realsi_<方向>/results.jsonl`
（`collect_results.realsi_points` 读的同一批 run；`text_policy=file`，三个 talker 种子的提交块逐块相同，已检查）。

| 方向 | 档 | 预算 (s) | 族（b0） | 读窗 m | 论文 ASR-BLEU / D1 | 句数 | 块数 | 合成请求（去掉纯标点块） |
|---|---|---|---|---|---|---|---|---|
| En→Zh | m1 | 1.18 | swb-q-bm1（−1） | 0.5 | 25.33 / 1.13 | 346 | 2044 | 1969（75） |
| En→Zh | m2 | 1.98 | swb-q-b1（1） | 0.5 | 28.26 / 1.77 | 346 | 1720 | 1632（88） |
| En→Zh | m3 | 2.46 | swb-q-b1（1） | 1 | 30.33 / 2.25 | 346 | 1403 | 1347（56） |
| En→Zh | m4 | 2.98 | swb-q-b1（1） | 2 | 30.51 / 2.75 | 346 | 1073 | 1040（33） |
| Zh→En | m1 | 1.18 | swb-q-bm1-zh（−1） | 0.5 | 17.83 / 1.14 | 431 | 2145 | 2144（1） |
| Zh→En | m2 | 1.92 | swb-q-b0-zh（0） | 1 | 21.42 / 1.70 | 431 | 1759 | 1758（1） |
| Zh→En | m3 | 2.40 | swb-q-b1-zh（1） | 1 | 22.46 / 2.18 | 431 | 1603 | 1603（0） |
| Zh→En | m4 | 2.98 | swb-q-b1-zh（1） | 2 | 22.65 / 2.67 | 431 | 1206 | 1206（0） |

- 每个种子 12,699 个请求，三个种子共 38,097 个。
- **复现检查（已做，登记前）**：用本次的打分代码（`score.py paper`）重算论文 talker 的 8 个点，三种子均值的 ASR-BLEU
  和 D1 都在两位小数上等于 tier 表（例如 En→Zh m2 28.26 / 1.77，Zh→En m2 21.42 / 1.70）；论文 run 的每一块都满足
  start = max(write_ms, 上一块结束)（违反 0 次）。

## 2. 设置（固定）

| 项 | 取值 |
|---|---|
| 权重 | `pretrained_models/Qwen3-Omni-30B-A3B-Instruct`（官方，与离线 CVSS-T 原生 run 相同） |
| 引擎 | vLLM-Omni 0.30.0，venv `/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030`（bin 放 PATH），默认三段配置（`experiments/s2st_baselines_v1/qwen3omni_native_split.yaml` / `_onecard.yaml`，与 CVSS-T 原生 run 相同；只有显存比例、max_model_len、max_num_seqs 与官方 yaml 不同，不影响输出） |
| 音色 | 预置 `Ethan`（transformers 的默认 speaker，也是 CVSS-T 原生 run 用的）；Qwen3-Omni 不能克隆源说话人 |
| 解码 | 与 CVSS-T 原生 run 相同：Thinker 贪心（temperature 0，max 256 token）；Talker temperature 0.9、top_k 50、top_p 1.0、repetition_penalty 1.05（官方 deploy 默认），max 4096；code2wav 贪心、repetition_penalty 1.1 |
| 种子 | talker 采样，按论文惯例跑 **三个种子**；vLLM 每请求 seed = 20261020 + 种子号（种子 0 = CVSS-T 原生 run 的 seed），三段同一个 seed |

**提示（固定）**：每个提交块一次请求，与上下文无关（不带前面块的文本或语音）。
- system 轮：`You are a text-to-speech reader. Say the user's message aloud exactly as it is written, word for word, in its own language. Do not answer it, translate it, complete it, correct it, or add anything. Reply with the user's text only.`
- user 轮：块文本本身（去掉首尾空白；英文块开头的连接空格不送；词中断开的块照原样单独送）。
- 用 Qwen3-Omni processor 的 chat template 构造（`add_generation_prompt=True`），与 CVSS-T 原生 run 相同。
- 指令放在 system 轮的原因：vLLM-Omni 的 talker 跳过 system 轮（`qwen3_omni.py` “Talker should ignore thinker system prompt”），
  所以 talker 的条件只有块文本（user 轮）和 Thinker 的回答，最接近“朗读给定文本”。
- 去掉空白后没有任何字母、数字或汉字的块（纯标点）不发请求、语音长度为 0（与 SimulInfer `tts.Speech.run`、R3 相同）。
  论文 talker 给这些块平均约 0.2 s 的语音；它们不含输出单位，只影响排队。
- Thinker 说什么，talker 就说什么：不过滤、不重试、不截掉首尾静音、不调语速（原生 talker 没有语速控制）。

**冒烟检查（不是 RealSI）**：开跑 RealSI 之前，用同一套代码在 Thinker **训练数据**的写出片段上
（`data/si_thinker_oracle_prefix_v1/rows_train.jsonl`，En→Zh 和 Zh→En 各 60 段，`random.Random(0)`）跑一次，只检查流程和提示是否基本可用。
如果提示在冒烟里明显失效（例如大量回答问题或翻译），会在开跑 RealSI 之前改一次提示，并在本文件里写明改动和原因；RealSI 上不改。

**时间轴**（与论文 `si_infer` 相同，不计合成耗时）：块 k 开始播放 = max(write_ms_k, 上一块开始 + 上一块时长)，时长 = 合成音频长度。
Omni 输出 24 kHz，用 `torchaudio.functional.resample` 转 16 kHz（与 CVSS-T 原生 run 相同）；speech-only 音频 = 各块语音首尾相接。

## 3. 打分（与论文相同；论文 talker 的 8 个点用同一份代码重算）

- **ASR-BLEU**：`reports/uniss_si_v4/speech_only.py` → `evaluation.asr_transcribe --batch-size 4`（uniss-eval；英文 Whisper-large-v3，
  中文 Paraformer）→ `evaluation/text_metrics.corpus_bleu`（规范化 + sacreBLEU）。分母为全部句子，缺的 ASR 行按空串计；覆盖率 ≥ 99%。
- **D1**：`collect_results.realsi_points` 的段 LAAL，原样（每个输出单位记所在块的开始播放时刻 `start_ms`，含排队）。
- **D0w**：`eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py` 的 `sentence_defs()["D0w"]`（单位记所在块的写出时刻；
  与 talker 无关，两边应相同，作为检查）。
- **talker WER（两边都报）**：句级，speech-only ASR 对比 talker 拿到的提交文本（各块文本相连），两边都按 BLEU 的规范化
  （`normalize_for_bleu`：英文按词，中文按字）；语料级 = 编辑数 / 参考 token 数，另报替换 / 删除 / 插入。
  论文 talker 的 run 只留了结果和 ASR、没有留音频，所以无法逐块转写；两边的并列比较因此用句级。
- **只对 Omni 报的附加诊断**：
  - Thinker 忠实度：每块 Thinker 输出文本对比送进去的文本（规范化后）：完全一致的比例、token 级 WER（替换 / 删除 / 插入）、三个种子是否一致、最差例子。
  - 块级 talker WER（种子 0）：每块自己的 wav 单独转写（同一 ASR，batch 4），对比该块送进去的文本。
  - 如果 Thinker 完全一致的比例低于 95%，或插入率明显高于论文 talker，就在结果里标为“系统性偏离”，并给出例子。
- **点** = 三个种子的均值（ASR-BLEU、D1、D0w、talker WER）。主比较是同一个点上两种 talker 的差（同样的文本、同样的写出时刻）。
- **分档**：每个方向，在这 4 个点里取 D1 ≤ 预算、ASR-BLEU 最高的点（En→Zh 1.18 / 1.98 / 2.46 / 2.98 s；Zh→En 1.18 / 1.92 / 2.40 / 2.98 s），
  两种 talker 各自分档；另附按 D0w 分档，仅作参考。原生 talker 说得更慢时，D1 会因排队变大，某些点可能进不了原来的档。
- 不报 COMET / MetricX；不报 SIM（原生 talker 不克隆音色）。

## 4. 代码与输出

- 代码：`experiments/s2st_baselines_v1/qabc_omnitalker/`
  - `common.py`（点、提示、种子）
  - `make_jobs.py`（提交块 → 请求分片，每个种子 8 片）
  - `synth.py`（vLLM-Omni 引擎：加载一次，认领分片直到做完）
  - `start_engines.sh`（按显存检查放引擎，停占卡程序只按 PID，间隔 ≥ 60 s 启动）
  - `assemble.py`（时间轴、忠实度、ASR 流水线）
  - `score.py`（论文 talker 重算 + Omni，分档）
- 运行输出：`eval_outputs/s2st_baselines_v1/qabc_omnitalker_realsi/`。
- 结果：`paper/baselines_results/qabc_omnitalker/`
  - `<方向>_<档>.json`：两种 talker 的 asr_bleu、laal_d1、laal_d0w、talker_wer、n、各种子，以及 Omni 的忠实度
  - `summary.json`
  - `summary.md`：并列表（论文 talker 对 Omni 原生 talker）
- tracked 副本：`reports/s2st_baselines_v1/qabc_omnitalker_results/`。

## 5. 执行（GPU）

- 所有卡上都有别的代理的评测作业；最终测试集评测（executor v9）优先，卡紧时等待，不挤。
- 每次启动前看 nvidia-smi 的实际用量，每卡合计 ≤ 135 GiB（约 5 GiB 余量）。
  - 引擎实际占用：thinker 卡约 88 GB；talker + code2wav 卡约 34 GB；单卡放三段约 122 GB。
- 同一张卡上两次 vLLM 启动至少间隔 60 s（脚本默认 90 s），不同时启动两个引擎。
- 守护进程 `keep_cards_busy_v3.sh` 从不停；只在启动前按 PID（先核对 cmdline）停该卡的占卡 supervisor。只杀自己的进程，按 PID。
