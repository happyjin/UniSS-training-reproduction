# OpenBroadcast-SI v1：音色条件两条线共用的开源全链路测试集

方案依据：`reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md` §1.4、§7、§7.2d–§7.2f。

**这是测试集。** 两条线（Idea 1 ZNE、Idea 2 CA-Kalman）都不得在上面做任何选择、调参或训练。只用开源数据构建；电竞数据只贡献了一个统计量，即提示 BVR 的分位数。

---

## 0. 预登记：各评测条件"见过"哪些语料（在运行任何 baseline 之前写定）

### 0.1 三个评测条件

三个条件的语音流**完全相同**（同一批 CoVoST 2 test 说话人、同样的句子和时间轴），只有背景不同。

| 条件 | 会议 ID | 语音语料 | 背景语料 | 与两条线的关系 |
|---|---|---|---|---|
| **ID-2**（领域内） | `obz001…`（zh→en）、`obe001…`（en→zh） | CoVoST 2 test（Common Voice v4 test 划分） | MUSAN 音乐和噪声中留出的 15%，加 FSD50K eval 中留出的 15%（`HELDOUT_BACKGROUNDS*.txt`） | 背景语料与两条线相同，但文件不重叠；说话人不重叠 |
| **OOD-1**（背景语料领域外） | `ooz001…`、`ooe001…` | 同上，逐条相同 | FMA-small 的 Instrumental 类（音乐床）、DEMAND 的 5 个人多的公共环境（人群/环境段）、ESC-50 的 8 类音效 | 三个背景语料两条线都**完全不使用** |
| **clean**（无背景对照） | `ocz001…`、`oce001…` | 同上，逐条相同 | 无 | 上限和"无害"对照 |

ID-2 和 OOD-1 共用同一份背景事件时间表（每段的起止、增益、在文件中的相对位置）、同一个 BVR 目标和同样的压低深度，只把文件换成另一个语料里的。所以两者之差只来自背景语料。

### 0.1b 两个方向是两种结构，结果分开报告

- **zh→en = 长流：** 37 条流、2.0 h；每条 2–4 人，每人多句，流长 61–336 s（中位数 214 s）。说话人取自 CoVoST 2 zh-CN test 中至少有 8 条的说话人。
- **en→zh = 短流，也就是"首句阶段"：** 148 条流、2.0 h；每条 2–4 人，每人恰好 3 句，流长 17–82 s（中位数 47 s）。
  - 原因：CoVoST 2 en→zh test 每个说话人最多只有 3 条（9472 人中，1 条的 4959 人、2 条的 2967 人、3 条的 1546 人），拼不出每人多句的 5 分钟流。
  - 后果：音色提示大多只能从每人最初一两句中取得，所以这个方向衡量的是"刚开口、可用语音很少"时的音色条件。
- 协调者 2026-11-19 同意这一偏差。两个方向的结果分开报告，不合并。

### 0.2 两条线见过的语料（来自方案 §7.1–§7.2c 和各自的预登记）

| 线 | 语音 | 背景 | 其他 |
|---|---|---|---|
| Idea 1（ZNE，不训练） | LibriSpeech test/dev（英）、AISHELL-3 test 说话人（中），只用于诊断和外推族选择 | MUSAN、FSD50K eval（**排除留出文件**） | HDemucs（MUSDB18-HQ 训练）、冻结的 BiCodec 和 Talker |
| Idea 2（CA-Kalman，训练） | LibriSpeech train-clean-100/360、AISHELL-3 train、Common Voice zh（**排除 `HELDOUT_SPEECH.txt` 中的说话人和整个 CoVoST 2 test 划分**） | MUSAN、FSD50K eval（**排除留出文件**），按文件 8:1:1 划分 | WenetSpeech 等真实带噪数据（无监督项） |
| 本基准 | CoVoST 2 test，两条线都没用过 | ID-2：上述两个语料中的留出文件；OOD-1：FMA、DEMAND、ESC-50 | — |

另外说明：
- 被测系统（Thinker `joint_v8h6k s06000`、T1c Talker）的训练数据里没有 CoVoST 2 test。
- FMA 与 MUSAN 音乐中的 fma 子集都来自 Free Music Archive 网站，但不是同一个语料发布。构建时已排除与 MUSAN 音乐同艺术家的曲目，文件不重叠。
- ESC-50 与 FSD50K 都来自 Freesound；Freesound id 出现在 FSD50K（dev 或 eval）中的 ESC-50 片段已排除。

### 0.3 留出清单（已提交）

| 文件 | 内容 |
|---|---|
| `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt` | FSD50K eval 留出 1560 / 10231 条，并写明规则 |
| `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS_MUSAN.txt` | MUSAN 音乐和噪声，同一规则（下载完成后写入） |
| `data/open_voicecond_v1/bench/HELDOUT_SPEECH.txt` | 基准用到的 553 个 Common Voice 说话人（client_id）和 2609 条片段 |

留出规则：`int(sha1('OpenBroadcast-SI-v1|' + 路径).hexdigest(), 16) % 1000 < 150`，即固定的 15%。路径是 MUSAN 的 tar 成员名（如 `musan/music/fma/music-fma-0003.wav`），或 `FSD50K.eval_audio/<fname>.wav`。

FMA、DEMAND、ESC-50 整体保留给本基准。

---

## 1. 构造

代码：`experiments/mist_w3po_v1/voicecond/bench_build.py`。每一步都拒绝覆盖已有输出，只用 CPU（分离器除外）。

### 1.1 源语音与参考译文

- 来源：CoVoST 2 官方 test 划分。从团队 S3 的本地镜像 `s3_mirror/zeyangsong_data/s2tt.tar.zst` 中解出 `s2tt/covost/hf/{zh-CN_en,en_zh-CN}/test-*.parquet`。每行含 Common Voice 音频、`client_id`、原文和人工译文。
- 说话人 ID 直接用 CV 的 `client_id`。
- 每条片段的处理：
  - 转 16 kHz 单声道；
  - 去掉首尾静音（10 ms 帧，阈值取"最响帧 −40 dB"和 −55 dBFS 中较高者，两侧各留 0.15 s）；
  - 有效语音 RMS 归一到 −26 dBFS；
  - 去掉以下片段：时长不在 1.5–12 s 之间、削波超过 0.1%、或几乎无声。
- 结果：
  - zh→en：190 个至少有 8 条的说话人，2554 条中保留 2499 条（3.71 h）；
  - en→zh：1546 个恰好有 3 条的说话人，4638 条中保留 4579 条（6.78 h）。

### 1.2 多人流（两个方向的结构见 §0.1b）

统计量来自 AMI train 的 137 场会议，只用非犹豫、≥ 1 s 的段；同一说话人相邻且间隔 < 1.5 s 的段合并成一个轮次。具体抽样：

- 轮次目标时长：从 AMI 轮次时长中抽样（中位数 3.5 s，P90 13.8 s）。同一说话人连续放句子，直到达到目标时长。
- 轮内句间停顿：从 AMI 中抽样，截到 0.1–1.5 s。
- 换人间隙：从 AMI 中抽样（中位数 −0.23 s，56% 为负值即重叠），截到 [−3, 4] s。另有两条限制：
  - 新轮次不早于上一轮最后一句的中点；
  - 新轮次不早于更早所有轮次的结束，所以任何时刻至多两人同时说话。
- 下一个说话人：从其余还有句子的说话人中选，按剩余句数加权。
- 每条流有 2、3、4 人的各占约三分之一。zh→en 的流达到 300 s 后在当前轮次结束处收尾；en→zh 的流用完句子就结束。
  - en→zh 中，92% 的说话人恰好 3 句，6% 为 2 句，2% 为 1 句。原因是轮换规则：当只剩上一个说话人还有句子时，流就结束。
- 每个说话人在流内另加 ±2 dB 的固定增益。

### 1.3 广播式背景（ID-2 与 OOD-1 用同一份时间表）

背景分三层：

| 层 | 时间表（与语料无关） | ID-2 语料 | OOD-1 语料 |
|---|---|---|---|
| 音乐床（持续） | 每 28 s 换一段 30 s 的选段，2 s 交叉淡化；选段增益 ±2 dB | MUSAN 音乐，留出且无人声（ANNOTATIONS 标 N）：68 首，5.2 h | FMA-small Instrumental：979 首（同 MUSAN 艺术家的 21 首已排除），8.2 h |
| 人群/环境段 | 泊松过程，平均每 20 s 一次；每次 2–8 s，淡入淡出 0.3 s；相对音乐床 −3 至 +6 dB | FSD50K eval 留出的 Crowd、Cheering、Applause、Clapping、Chatter 片段（排除带讲话或歌唱标签的）：54 条 | DEMAND 的 PCAFETER、PRESTO、PSTATION、SPSQUARE、TMETRO（第 1 声道）：5 × 5 min |
| 音效 | 泊松过程，平均每 15 s 一次；每次 ≤ 3 s；相对音乐床 −6 至 +4 dB | FSD50K eval 留出的 Explosion、Gunshot、Whoosh、Boom、Fireworks（43 条），加 MUSAN 留出噪声（160 条） | ESC-50 的 fireworks、glass_breaking、door_wood_knock、can_opening、clock_alarm、thunderstorm、mouse_click、keyboard_typing（Freesound id 与 FSD50K 重叠的 114 条已排除）：206 条 |

- **事件如何取文件：** 每个事件抽两个均匀随机数 `u_file` 和 `u_off`，按"语料文件数 × u_file"取文件，按 u_off 定在文件中的偏移。所以 ID-2 和 OOD-1 只在语料上不同。
- **压低（侧链）：** 有语音的 10 ms 帧上，背景压低 6–12 dB（每条流固定一个值）；起效 50 ms，释放 400 ms。
- **BVR：** 定义为语音活动区间内"压低后的背景能量 / 语音能量"。每条流的目标值，取电竞提示 BVR（444 条 HDemucs 测量值）在分位数 0.05–0.95 上的分层值，顺序随机。
  - 各流实测 BVR 与目标完全一致：中位数 −14.3 dB，P10 约 −27 dB，P90 约 −8 dB，范围 −40 至 −6 dB。
  - 少数流（ID-2：zh→en 2 条、en→zh 4 条；OOD-1：3 条、5 条）混合后峰值超过 0.99。处理方法是把整条混合（语音和背景一起）等比缩小，比例记在 `meetings.json` 的 `limiter` 字段。
- **无背景版本：** 只有语音轨，其余完全相同。

### 1.4 流水线格式

与 `data/game_esports_si_v1/<dir>/` 相同。数据目录为 `data/open_voicecond_v1/bench/<cond>/<dir>/`，包含：

- `session_audio/`：zh→en 为 `mix/<m>.wav`，en→zh 为 `ihm_mix/<m>.Mix-Headset.wav`，16 kHz；
- `meetings.json`：流编号、条件、说话人、BVR、压低深度、`limiter`、三个条件之间的对应会议 ID（`twins`）；
- `refs_windows.jsonl`：每句一个窗口；`utts` 中有原文、人工译文 `ref`、真实说话人 `speaker_id`，时间相对于窗口；
- `utts/<m>.jsonl`：会议时钟上的每一句及其 CV 片段 ID；
- `bg_track/<m>.wav`：混入的背景（已压低、已缩放），只在 ID-2 和 OOD-1 中有；
- `diar/nemotron_lowlat/<m>.{post.pt,rttm}`：Nemotron-3-Diarization 的 lowlat 预设（`scripts/voicecond_bench_diar.sh`，在卡 3 上运行）。

窗口：

- C2 在线窗口（间隙 0.6 s）：`eval_outputs/mist_w3po_v1/data/windows_obsi_<cond>_<dir>_c2g06_v1/windows_test_online_<dir>.jsonl`；
- 分离器块协议窗口：`..._c2g06_chunk_v1/windows_dev_online.jsonl`（`specclose.cmd_build`，关闭判别器）。

其他文件：

- 共享源文件：`src/<dir>/{clips/,pool.jsonl,plan.json,bg_plan.json,spkref.json}`；
- 背景素材：`bgsrc/<pool>/` 和 `bgsrc/pools.json`，含每个文件的许可；
- 统计：`meta/stats.json`（AMI 统计和 BVR 值）、`summary.json`。

`reports/gstar_si_v6/meeting_score.py` 的 `universe` 加了一个回退分支：会议 ID 形如 `o[boc][ze]NNN` 时，读本基准的 `utts/<m>.jsonl`。做法与电竞回退相同，是纯增加，对其他集合没有影响。

## 2. 统计

| | zh→en（长流） | en→zh（短流 / 首句阶段） |
|---|---|---|
| 流数 | 37 | 148 |
| 总时长 / 语音时长 | 2.01 h / 1.96 h | 2.00 h / 1.92 h |
| 流长（最短 / 中位数 / 最长） | 61 / 214 / 336 s | 17 / 47 / 82 s |
| 每流人数（2 / 3 / 4 人的流数） | 13 / 12 / 12 | 50 / 49 / 49 |
| 说话人 / 句数 | 110 / 1326 | 443 / 1283 |
| 每人句数（最少 / 中位数 / 最多） | 2 / 10 / 27 | 1 / 3 / 3 |
| 重叠占语音时间 | 7.5% | 8.2% |
| BVR（P10 / 中位数 / P90，ID-2 与 OOD-1 相同） | −26.7 / −14.3 / −8.0 dB | −27.6 / −14.2 / −7.8 dB |
| C2 窗口（无背景版本） | 948 | 970 |

AMI 参照：重叠占语音时间 9.6%；换人处 56% 为重叠。

## 3. 运行方式

### 3.1 编排器包装（只替换集合定义）

`scripts/voicecond_bench_orch.py` 的做法与 `game_orch.py` 相同，在本进程里把集合定义换掉，然后调用 `joint_orch.py`：

- zh→en：内部用 `--set ali`，即 `final_test_orch` 的测试链；英文分词；`GSTAR_DATA` 指向本条件的数据目录；会话为 `mix`。
- en→zh：内部用 `--set test`，即 `en2zh_orch` 的测试链；会话为 `ihm_mix`。

参数：

- 必须给 `--cond id2|ood1|clean --dir zh2en|en2zh`；
- 必须给 `--exclude`：本代理和 Idea 1 用 `4 5 6 7`，Idea 2 用 `0 1 2 3`；
- 其余参数原样传给 `joint_orch`，包括 decode 参数、Talker、`--cap135`、账本、评分和 C2 延迟。

### 3.2 baseline B

`scripts/voicecond_bench_run_b.sh` 与 `game_run.sh` 中的 B 完全相同：

- Thinker：`joint_v8h6k merged s06000` + identity，window 1，`--cap135`；
- 语音：冻结的 T1c Talker，两个 TTS 种子（20261021、20261109）；
- 卡：`--exclude 4 5 6 7`；
- 每条链：文本编排器 → T1c 编排器 → `bench_eval`；
- 运行名：`obsi_b_<cond>_<dir>`（文本）、`obsi_b_<cond>_<dir>__t1c`（种子 1）、`…__t1c_tts2`（种子 2）；
- 编排器目录：`logs/mist_w3po_v1/voicecond_bench/<cond>_<dir>{,_t1c}`；完成标记：`VB_<cond>_<dir>_DONE`。

## 4. 给两条线的接口：只重跑 Talker，逐流替换音色 token

**做法：** 复用 baseline B 的文本解码，只重跑 T1c Talker 链（prep → 8 个 talker 分片 × 2 个种子 → ASR → 评分 → C2 延迟）。每个 talker 作业由下面的模块代替原来的 `si_infer`：

- `experiments.mist_w3po_v1.voicecond.bench_talker_override`
- 它原样运行 `si_infer`，只把 `BiCodecTokenizer.encode_wav_to_tokens` 换掉：遇到 sidecar 中列出的条目，就返回 sidecar 给的 32 个 global token。
- si_infer 用这 32 个 token 同时作为 LM 条件和解码器的 d-vector。

**sidecar 格式**（json）：`{"<talker 条目 id>": [32 个整数 0–4095] | "<另一段 16 kHz wav 的路径>"}`

- 条目 id 就是音色提示文件名的主干，也是 baseline 运行中 `speech/obsi_b_<cond>_<dir>__t1c/talker_in/items.jsonl` 的 `id`，例如 `obz001_on_0003__S2`。
- 原始提示文件在 `talker_in_all/voice_{causal,fallback}/<id>.wav`。
- 值为路径时，改为编码那段 wav（例如 HDemucs 人声）。
- 不在 sidecar 中的条目保持原编码。
- 退出时 talker 日志会打印 `[voice-override] … tokens N, wav N, original N`，用来核对覆盖了多少条。

**为什么条目和种子不会变：** prep 是确定性的，输入是同一份 hyp、同一套窗口、同一份分离器结果，所以条目和块与 baseline 逐条相同。分片方式、`--seed`（种子 1 用默认的 20261021，种子 2 用 20261109）以及其余参数都不变。因此新运行与 B 的唯一差别就是被替换的音色 token。

**命令示例**（Idea 1 用卡 0–3；Idea 2 改用 `--exclude 0 1 2 3`）：

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
C=id2; D=zh2en; N=obsi_zne_${C}_${D}__t1c          # 新运行名，不得与已有目录重名
setsid nohup $PY scripts/voicecond_bench_orch.py --cond $C --dir $D --voice-override <sidecar.json> \
  --base $B2 --run logs/mist_w3po_v1/<你的目录>/$N --name $N --ckpt - --speech-from obsi_b_${C}_${D} \
  --talker t1c --window 1 --cap135 --exclude 4 5 6 7 < /dev/null > logs/mist_w3po_v1/<你的目录>/$N.launch.log 2>&1 9>&- &
```

先加 `--dry` 检查：每个 `__talk<k>` 作业的命令里都应出现 `bench_talker_override --voice-override`。

## 5. 统一评测命令（任何运行都用它，结果可直接比较）

```bash
PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond <c> --dir <d> --name <运行名，种子 1>
```

- 自动同时评 `<name>` 和 `<name>_tts2` 两个种子；只用 CPU。
- 输出：`eval_outputs/mist_w3po_v1/voicecond_bench/<name>/metrics.json`（汇总、每个种子、每条流的均值，用于按流配对检验），以及 `items_<n>.jsonl`（逐条目）。
- 表格：`python -m experiments.mist_w3po_v1.voicecond.bench_table <name> ...`

各项指标：

| 指标 | 定义 |
|---|---|
| tcpASR-BLEU | `decode/<n>/score_asr.json` 的 `tcpBLEU_all`（overlap_score，**真实 CV 说话人**）；另附 ov / 1spk |
| 不分说话人的 ASR-BLEU | 每条流把所有输出段按时间（窗口起点 + 段起点）连成一篇，所有参考句按时间连成一篇，算语料级 BLEU（zh→en 用 13a 分词，en→zh 用 zh 分词）；文本也同样算一遍 |
| 文本 tcpBLEU | `score_text.json` |
| 语音 LAAL（主延迟） | 分离器块协议：在 `windows_obsi_<cond>_<dir>_c2g06_chunk_v1` 上跑 speech_latency_lit（zh→en 经 `on_refset --dr ali`，en→zh 经 `en2zh_lat`），取 `unit.speech_laal_online.all.mean`；EndOffset 为 `unit.speech_end_online.all.mean`，只作补充 |
| SQUIM-PESQ / SI-SDR / STOI | torchaudio SQUIM objective，无参考，取每条 talker 流的前 20 s（与 `game_muffle_analysis.squim` 相同） |
| 频谱对比度 / 平坦度 | `game_voice_texture.feats`：300–4000 Hz 的平坦度中位数，5 个倍频带对比度的均值 |
| 说话人相似度 | WavLM-base-plus-SV（x-vector，与 BiCodec 无关，本机 `evaluation_models/wavlm-base-plus-sv`，即项目 SIM-O 用的模型）。计算该 talker 流的语音，与其真实说话人在本流中**干净 CoVoST 片段**的平均嵌入之间的余弦。条目 (窗口, 槽位) 分配给"槽位在该窗口内的分离器活动（> 0.5）与其语句重叠最多"的真实说话人；分配不上的条目只计数，不计分 |

注：本机的 WavLM-Large 只有主干、没有说话人验证头；没有独立的 ECAPA 权重（BiCodec 内部的 ECAPA 不独立）；CAM++ 只有 X-Translator 带的中文版。所以选 wavlm-base-plus-sv，它与 BiCodec 相互独立。

（以下各节在运行完成后填写。）
