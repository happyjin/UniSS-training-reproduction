# 流式同传（S2ST）部署文档：贝叶斯提交，m2 档

当前**效果最好、可直接运行**的中英双向流式语音到语音翻译方案。读完本文，你应该能：

1. 用一条命令把任意一段中文或英文录音翻译成另一种语言的语音，边听边译，保留说话人音色；
2. 看懂输出里的每个文件和延迟数字；
3. 知道什么时候需要换配置，以及已知的限制。

本文只覆盖 **m2 档**（读入窗口 0.5–1 s，平均延迟约 1.7–1.8 s），这是质量与延迟最均衡的配置。

---

## 1. 方案一览

| 组件 | 选用 | 是否训练 |
|---|---|---|
| Thinker（听 + 译） | Qwen3-Omni-30B-A3B-Instruct + **ABC 适配器**（LoRA，已合并） | 只训练了 LoRA，其余冻结 |
| 提交策略（何时说） | **贝叶斯最优提交（ABC）+ 语音队列风险（SDR）** | SDR 无参数 |
| Talker（说） | UniSS T1c（export_1500），按块流式生成 BiCodec 语音码 | 已微调，固定 |
| 音色 | **贝叶斯音色记忆，30 s 后冻结**（单人录音） | 无参数 |

工作方式（因果流式）：

```
源语音 ──每 Δ 秒读入一次──▶ Thinker 续写整句草稿 ──▶ 贝叶斯规则：只说出"大概率不会再改"的前缀
                                                        │（Talker 还在说话时阈值自动变严，等待是免费的）
                                                        ▼
                                  Talker 把提交的文本块说出来（音色来自贝叶斯音色记忆）
                                                        ▼
                                  按提交时刻放回原录音时间轴（前一块没说完，下一块顺延）
```

每次读入只用到"到当时为止已经听到的音频"，不偷看未来；长录音先按停顿切成 3–15 s 的句段，逐段流式处理，再拼回原时间轴。

## 2. 效果（m2 档，RealSI 测试集，语音输出）

ASR-BLEU 越高越好；延迟为 segment LAAL（秒）。三次 Talker 随机种子取均值。

| 方向 | 本方案 | 基线 LocalAgreement（同一 Thinker/Talker） | SimulS2ST-Omni（论文报告值） |
|---|---|---|---|
| 英→中 | **28.26**（1.77 s） | 27.42（1.84 s） | 24.21（m2 预算 1.98 s） |
| 中→英 | **21.42**（1.70 s） | 20.31（1.78 s） | 19.55（m2 预算 1.92 s） |

- 2–3 s 区间的预注册检验中，完整贝叶斯系统显著优于 LocalAgreement（英→中 +0.78，p=.039；中→英 +0.58，p=.020）。
- 长篇技术报告（ACL60/60，英→中，文本）上同样领先：约 2.2 s 延迟时 50.53 BLEU，比延迟相近的 LocalAgreement（49.09）高 1.4。
- 计算开销：Thinker 每次读入约 57–59 ms（单路、单卡 H200），比 LocalAgreement 还快（它在"等待标记"处停止生成）。

## 3. 环境

所有路径都在 `/opt/dlami/nvme/jasonleeeli/` 下，**不需要额外安装任何东西**。

| 用途 | 环境 | Python 路径 |
|---|---|---|
| **推理**（Thinker、Talker，基于 vLLM） | venv **`vllm_omni_030`**（不是 conda） | `/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python` |
| 音频切分、拼接、评测（ASR-BLEU） | conda **`uniss-eval`** | `/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python` |
| 训练 / 合并适配器（部署不需要） | conda `qwen3omni` | `/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python` |

脚本里已经写死了上面的解释器路径，**不需要 `conda activate`**。

GPU：2 张 H200（每张 143 GB）。Thinker 一张卡（vLLM 占约 85% 显存），Talker 另一张卡（约 35%）。两个引擎不要放在同一张卡上同时启动（vLLM 会因显存探测冲突失败）。

## 4. 模型文件

仓库根目录：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`（以下为相对路径）

| 模型 | 路径 | 大小 |
|---|---|---|
| Thinker（ABC 适配器已合并进 Qwen3-Omni） | `eval_outputs/uniss_qwen3_omni_v1/thinker_op_v3_20261023T213631Z/thinker_merged_i200` | 66 GB |
| Talker（UniSS T1c） | `eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss` | 3.5 GB |
| BiCodec / GLM-4 分词器（Talker 自动加载） | `pretrained_models/UniSS` | 5.4 GB |

## 5. 一条命令运行

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
AUDIO=/path/to/talk.wav DIRECTION=zh2en OUT=outputs/my_talk_zh2en \
GPU_THINKER=0 GPU_TALKER=1 \
  bash experiments/uniss_qwen3_omni_v1/scripts/run_stream_s2st.sh
```

- `DIRECTION`：`zh2en`（中→英）或 `en2zh`（英→中）。
- `AUDIO`：任意采样率、单/双声道的 wav、flac、ogg 或 mp3；脚本自动转成 16 kHz 单声道。
- `OUT`：**必须是一个不存在的新目录**；脚本拒绝覆盖已有输出。
- 想先看会执行哪些命令：加 `DRY_RUN=1`，只打印、不运行。

脚本会按方向自动选择 m2 参数：

| 方向 | 读入窗口 `WINDOW` | 等待偏置 `BIAS` | SDR 语速 r | Talker 提速 / 追赶 |
|---|---|---|---|---|
| en2zh | 0.5 s | 1 | 0.191 s/汉字 | 不提速 / 不追赶 |
| zh2en | 1 s | 0 | 0.339 s/英文词 | 提速 2 档 / 落后时最多 1.3 倍追赶 |

可选参数（一般不用改）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `VOICE` | `bayes` | 音色方法：`bayes`（贝叶斯记忆 + 30 s 冻结，单人推荐）、`ema`（多人对话推荐，见第 8 节）、`plain`（每句只用开头的音频） |
| `GPU_THINKER` / `GPU_TALKER` | 0 / 1 | 用哪两张卡 |
| `LIMIT_S` | 0 | 只处理前多少秒（0 = 全部），试跑时有用 |
| `WINDOW` / `BIAS` | 见上表 | 只有在要换档位时才改；`BIAS` 越大越保守、延迟越高 |

## 6. 每一步在做什么

脚本依次执行 6 步，日志都在 `OUT/` 下：

| 步骤 | 内容 | 环境 | 日志 |
|---|---|---|---|
| 0 | 转成 16 kHz 单声道（`OUT/<文件名>.wav`） | uniss-eval | 终端输出 |
| 1 | 按停顿切成 3–15 s 的句段（`OUT/pieces/`、`items.jsonl`，含每段在原录音中的偏移） | uniss-eval | `step1_segment.log` |
| 2 | 生成 `items_talk.jsonl`（加上整段录音路径，供音色记忆只取"新听到的语音"） | uniss-eval | — |
| 3 | Thinker 流式提交：每 Δ 秒读入一次，ABC+SDR 决定提交多少译文（`OUT/thinker/blocks_*.jsonl`） | vllm_omni_030 | `step3_thinker.log` |
| 4 | Talker 逐块发声（`OUT/talker/`） | vllm_omni_030 | `step4_talker.log` |
| 5 | 按提交时刻放回原时间轴（`OUT/result/`） | uniss-eval | `step5_stitch.log` |

需要单独跑某一步时，用 `DRY_RUN=1` 打印出的命令即可（第 2 步是脚本里的一小段 Python）。

## 7. 输出

`OUT/result/` 下：

| 文件 | 内容 |
|---|---|
| `translation.wav` | 只有译文语音，放在原录音的时间轴上（听音色、发音最方便） |
| `stereo.wav` | 左声道原声、右声道译文——戴耳机最接近现场同传的听感 |
| `mix.wav` | 原声（音量压到 0.35）+ 译文，混成单声道 |
| `*.ogg` | 上面三个文件的压缩版，方便发给别人听 |
| `transcript.json` | 每句：原声开始时刻 `start_s`、译文实际出声时刻 `speech_at_s`（都是原录音时间轴上的秒数）、因前句未说完的顺延 `queued_s`、提交记录 `blocks`（`[距该句段开头的秒数, 提交的文本]`）、完整译文 `translation` |

`step5_stitch.log` 的最后一行是汇总：录音时长 `seconds`、句数 `pieces`、平均顺延 `queued_mean_s`（语音块因前一块还没说完而推迟的平均秒数）。

**关于延迟的两个口径。** 第 2 节的 1.7–1.8 s 是 RealSI 上逐句计算的 LAAL（每句单独计时）。处理连续的长录音时，如果译文语速跟不上原声，语音会排队累积，所以 `speech_at_s − start_s` 会更大：例如第 11 节的 m2 测试中，译文中位比原声晚约 3.3–3.4 s，其中平均顺延约 1.8–2.0 s。中→英默认已开启提速和追赶来缓解；排队明显时可以考虑更大的提速档（`si_infer --speed-shift`）。

中间结果：`OUT/thinker/blocks_*.jsonl`（每次读入的草稿与提交块），`OUT/talker/results.jsonl`（每句的语音块、时长、音色 token 等）。

## 8. 音色：单人用 bayes，多人用 ema

贝叶斯音色记忆把一场演讲里已经听到的有效语音按时长加权平均（只用去掉静音的新语音，不重复拼接），累计 30 s 后冻结音色，避免句与句之间音色跳变。

- **单人录音（讲座、朗读、播报）**：用默认的 `VOICE=bayes`。在 RealSI 上，"像说话人"的相似度从 0.778 提高到 0.830（英→中），相邻句音色一致性从 0.880 提高到 0.925；翻译质量不变。
- **多人对话（访谈、会议）**：用 `VOICE=ema`。贝叶斯记忆目前**不能可靠地检测换说话人**，会把两个人的音色平均成一个，听起来反而不稳；EMA 只跟随最近 6 s 的说话人。

## 9. 已知限制

1. **这是基于录音文件的因果流式，不是在线服务。** 每次读入只看已听到的音频、提交时刻和出声时刻都按真实时间计算，但程序一次性处理整段录音（多句并行，速度快于实时）。接麦克风做在线服务需要另写读入循环；按实测计算量（Thinker 每次读入约 60 ms），单路实时是可行的。
2. **延迟数字不含计算时间。** 单卡单路实测，把计算时间算进去后延迟最多增加约 0.2 s。
3. **长录音按停顿切段。** 一段话超过 15 s 没有停顿会被强行切开；切点落在词中间时，那一句的开头可能不完整。
4. **口语场景下 ABC 的优势与训练批次有关。** 用不同随机种子重训的适配器，在 RealSI 上平均与 LocalAgreement 持平；SDR 对每个适配器都有提升，因此始终开启。长篇报告（ACL60/60）上三个适配器都显著领先。
5. **多人对话的音色**见第 8 节。

## 10. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `拒绝覆盖已有输出` | `OUT` 已存在；换一个新目录名 |
| 日志里 `Free memory on device ... less than desired GPU memory utilization` | 该卡上有别的进程（包括占卡程序）；换空闲的卡，或先停掉占卡程序 |
| Talker 报 `maximum context length` | 已默认 `--max-model-len 16384`；若仍出现，说明某句极长，可把 `LIMIT_S` 调小试跑定位 |
| `音频转换失败` | 输入格式 soundfile 读不了；先用 ffmpeg 转成 wav |
| 译文全空或很少 | 检查 `step1_segment.log` 的句段数；录音太安静或全是音乐时切分会失败 |

## 11. 实测运行时间（本文档写作时的端到端验证）

在 2 张 H200 上用本脚本的命令逐步实测（结果见第 12 节的路径）：

| 测试音频 | 方向 | 配置 | Thinker | Talker | 总计 |
|---|---|---|---|---|---|
| 张河桥乡，前 70 s，48 kHz 双声道输入 | zh2en | Δ=1 s，b₀=1 | 91 s | 180 s | 274 s |
| 张河桥乡，前 70 s，48 kHz 双声道输入 | zh2en | m2 默认（Δ=1 s，b₀=0） | 88 s | 66 s | 158 s |
| Helen Keller，前 60 s，16 kHz 输入 | en2zh | m2 默认（Δ=0.5 s，b₀=1） | 90 s | 67 s | 158 s |

时间主要花在两个模型的加载上（Thinker 约 1.5 分钟，Talker 约 1 分钟；第一次运行时 vLLM 还要编译，会更久）；加载之后处理速度远快于实时。两次 m2 测试的输出都完整（10 / 14 个句段，没有空译文），译文中位比原声晚约 3.3–3.4 s（含平均顺延 1.8–2.0 s）。

## 12. 参考

- 本方案的论文与全部实验：`paper/bayes_simul_s2st_v2/`（`main.pdf`，补充实验在附录 E）
- 试听页（四段长录音，m1–m6 各档，EMA / 贝叶斯音色可切换）：https://claude.ai/artifact/19SNwCTDgWmwaLcYztsv6T （私有，需要作者在 Share 菜单里分享给你）
- 离线（整句）方案的部署文档：`docs/deploy/qwen3_cascade_s2st_deploy.zh-CN.md`
- 一键脚本：`experiments/uniss_qwen3_omni_v1/scripts/run_stream_s2st.sh`
