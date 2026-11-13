# En→Zh 方向：A′ + PL-EM 全流程计划与预登记 v1（en2zh_full_plan_prereg_v1）

- 写入时间：2026-11-12 布拉格约 22:50（21:50 UTC）。
- 写入时，En→Zh 还**没有任何 GPU 作业**：没有训练、没有解码、没有合成，也没有任何 En→Zh 的 dev 或 test 数字。已经做过的只有第 2 节列出的 CPU 数据构建和检查，这些检查都不涉及译文质量。
- 用户决定：
  - 加 En→Zh 方向：原始英文 AMI → 中文，配方用目前最好的那一套，**只跑一个种子**；
  - 训练步数在验证集上选，用**快速代理**，不跑 dev 全集的语音链。
- 本文件提交之后的任何修改，一律以"修正 N"的形式追加在文末，写明时间和原因，不改动已有内容。

## 0. 一句话

En→Zh 的配方与 zh→en 的主系统 A′rl + PL-EM **逐项相同，只换方向**：

| 部件 | zh→en（主系统） | En→Zh（本计划） |
|---|---|---|
| Thinker | v2′（SerialCRF，`crf_formal_and_next` §1.1），step_03000，finalpost | 同一个训练目标和训练设置。步数按第 3 节的规则从 500…3000 里选 |
| 窗口 | C2 gap 0.6 | 同一个检测器规则，用英文分离器后验 |
| talker 输入 | 读级块（`merge_read_segments`） | 同；中文块合并时不加空格（`--sep ''`） |
| talker | PL-EM，c1+b1+s2（`aprime_plem_crf_formal_and_latency` §1.2） | 同一个 PL-EM 配方，在中文候选上重新训练（第 4 节）。检查点按第 4.4 节的规则从 400 / 800 / 1200 里选 |
| 延迟 | 现行协议和 chunk 协议都报 | 同 |

- **D1v**（在 VE1 配方上做 DAgger）是延迟更低的取舍点，本计划**不包括**，只作为可选项。En→Zh 要先补做以下几步，约多出 10–12 小时：
  - T5 证据时刻标签：要在 GPU 上打分；
  - DART 清单；
  - VE1 训练：4 卡，约 4.5 小时；
  - D1v roll-in：约 2–5 卡·时；
  - D1v 训练：约 4 小时。
- 是否追加由用户决定。

## 1. 数据与通道

- **源端**：原始英文 AMI 会议。
  - 会话音频用 `ihm_mix`（头戴麦混音，取通道均值），**单通道输入**；
  - 分离器用 Nemotron 低延迟预设 `diar/nemotron_lowlat`。硬盘上已有整场后验，**不需要 GPU 重跑**。
- **目标端**：中文。
  - 参考是 MIST v7/v8 的 En→Zh 参考：AMI 人工英文转写，经 3-MT + QE 流程译成中文，属于 **MT 参考**，论文里照此写明。
  - dev 用 `data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl`（18 场会议，3203 个参考窗口）；
  - test 用 `data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl`（16 场，v7/v8 的主参考集，不用 B1 扩展参考）。
- **训练数据**：只用 AMI train。dev 只用于选检查点和描述，test 只跑一次。RealSI、ACL 60/60、AliMeeting 不碰；不用 COMET / MetricX；不用 QABC 方法。

## 2. 已经建好的数据（CPU；提交 dd2b8ec8、041f4a00、052e0428）

| 数据 | 路径（`eval_outputs/mist_w3po_v1/`） | 规模 | 检查 |
|---|---|---|---|
| dev C2 窗口 | `data/windows_en2zh_c2g06_v1` | 18 场，4996 个窗口，7.77 h | 同一脚本用 gap 1.0 重建 v6 的 En→Zh 测试窗口：3387 / 3387 条，除音频和后验路径外**逐字段相同** |
| dev5 窗口 | `data/windows_en2zh_c2g06_v1_dev5` | 5 场（ES2011a、IB4003、IB4011、IS1008c、TS3004b，与 zh→en 的 dev5 相同），1248 个窗口 | — |
| train16 窗口（PL-EM 数据） | `data/windows_en2zh_c2g06_train16_v1` | 按字母序前 16 个 train 会议，7600 个窗口，12.56 h | zh→en 的同一规则得到 15 场、5613 个窗口：En→Zh 不需要 A8 跳过，会议又更长 |
| test C2 窗口 | `data/windows_test_en2zh_c2g06_v1` | 16 场，5126 个窗口，7.30 h | 只用了窗口、音频和后验，没有用参考 |
| chunk 协议窗口 | `data/windows_en2zh_c2g06_chunk_v1`、`…_chunk_v1_dev5`、`data/windows_test_en2zh_c2g06_chunk_v1` | 4996 / 1248 / 5126 | 判别器关闭时的 `specclose build`：只改两个时间戳，没有一个比现行协议晚。开窗平均早 318 ms，关窗平均早 400 ms（dev）/ 402 ms（test）。zh→en 是 316 / 401 ms |
| Thinker 训练窗口 | `crf/sched_en2zh_v1` | 21,581 个窗口，143,168 次读，L_ref 0.862 s | 见下方说明 |
| dev sched 窗口（held-out 损失用） | `crf/sched_en2zh_dev_v1` | 3203 个窗口 | v8 验证行 14,994 / 14,994 匹配，stream_target 3203 / 3203 |

**Thinker 训练窗口的说明**
- 构建方式：`build_sched_windows` 新增的 en2zh 选项，取中文文本、扁平的 slot map、AMI 词时间、不 strip、按字计。
- 起点模型：En→Zh 的 v8 是 `thinker_gstar_s2dl_20261029b`。它是在 `e2e_gstar_stream_v1/rows_train.jsonl` 上从 O2 续训 3000 步得到的，与 zh→en 的 v8 对称。
- 截止标签：用 AMI 词时间戳，lag 1.0 s。这与 En→Zh v8 的行逐行相同。zh→en 那边是按字数比例估计。
- 检查结果：
  - v8 读行 103,828 / 103,828 全部匹配，0 不匹配；
  - 窗口自带的 stream_target 21,581 / 21,581 相同；
  - 越界 token 0 个；
  - serial_v2p 的 `--check-all`：标签路径的行与 C0 的行 21,581 / 21,581 相同；有真顺序选择的窗口 6,094 个；
  - 每步 pass 1 平均 84.1 行、pass 2 平均 116.5 行（zh→en 是 61.5 / 100.4），所以每步会更慢一些。

## 3. Thinker：训练与检查点选择

### 3.1 训练（`scripts/en2zh_train_thinker.sh` → `run_crf_v2p.sh`）

- 训练目标：`train_crf_sft --objective serial_v2p --tau 0.5 --final-read-full-post 1`，即 v2′ 的在线确定性退火 EM。
- 设置与 A′ 逐项相同：
  - 3000 步，每步约 64 个读行；
  - 学习率 1e-4，warm-up 50，余弦退到 1e-5；AdamW (0.9, 0.95)，wd 0，梯度裁剪 1.0；
  - LoRA r32 / α64 + 投影器 + 259 个新 token 行；
  - 扰动 p 0.2，层级权重开；
  - 种子 20261109，**只跑一个种子**；
  - 打包参数：TOK 32000、CAP 120、fast-moe、prefetch、逐行特征。
- 只改三项：`--sched crf/sched_en2zh_v1 --base thinker_gstar_s2dl_20261029b/thinker_merged --dir en2zh`。
- 每 500 步存一次检查点：500、1000、…、3000。
- 4 张独占卡，按 takeover 语义接手：
  - 等槽位上只剩我们的占卡程序；
  - noclobber 占槽；`sync_train_exclude` 随之让所有评测编排器避开这 4 张卡；
  - 按 PID 停掉占卡程序，同时启动训练；
  - 核对 4 个 rank 都在。

### 3.2 选择代理：dev5 纯文本解码（`scripts/en2zh_ckpt_proxy.sh`，与训练并行）

- 每个检查点的 DONE 一出现，就在**非训练的 4 张卡**上开一个编排器（`en2zh_orch.py --set dev5 --talker none`）：
  - 解码 1248 个 dev5 窗口：4 个窗口分片，1.0 s 读；其余解码参数与 A′ 相同（finalpost close_detect、policy wait、lag 2000、context 0、batch 16、loop guard）；
  - 文本 tcpBLEU all / ov / 1spk：overlap_score，中文分词，在线协议；
  - 文本 LAAL，现行协议，按中文单位计（第 5.3 节）。
- 附带一个**近乎零成本的辅助信号**（`crf/heldout_loss.py`，同卡，约 5 分钟，**只记录，不参与选择**）：dev5 会议的 sched 窗口上：
  - 标签路径的 C0 损失；
  - v2′ 格的对数似然 NLL；
  - τ = 0.5 时标签路径的后验质量。
- 全部检查点跑完后，再记录 chunk 协议下的文本 LAAL（CPU）。

### 3.3 固定规则（在任何结果之前写死；`scripts/en2zh_select.py thinker`，单元测试 `test_en2zh_select.py`）

- 候选：s ∈ {500, 1000, 1500, 2000, 2500, 3000}。每个候选取三个 dev5 数：
  - Q(s)：文本 tcpBLEU-all；
  - O(s)：文本 tcpBLEU-ov；
  - L(s)：文本 LAAL，现行协议。
- **可行集**：L(s) ≤ L(3000) + 0.1 s。
- **选法**：在可行集里取 Q 最高的。打平（Q 保留两位小数后相等）时依次比较：O 更高 → L 更低 → 步数更大。
- 例外：
  - 某个检查点的解码最终失败（编排器同配置重试后仍失败）：该点不参加，照实记录；
  - step_03000 没有完整输出：**不选择**，问协调者。
- 只记录、不参与选择：tcpBLEU-1spk、chunk 协议文本 LAAL、held-out 损失。

### 3.4 选定之后（`scripts/en2zh_after_select.sh` 第 1–3 步）

- 训练正常退出（rc 0）后释放槽位。
- 用选中的检查点做 **dev 全集解码**：4996 个窗口，8 个分片按会议分。
- 读级合并 `merge_read_segments --sep ''`：
  - pair_lost ≤ 2 且输出完整：照用，列为已知偏差；
  - 否则停下来问协调者。
- 从合并后的解码切出 dev5 部分，给第 4.4 节的 talker 代理用。

## 4. talker：PL-EM（中文）

### 4.1 中文 CTC 打分器（协调者已批准下载）

PL-EM 势函数里的 f = −min(CTC NLL/字, 10)，b1 选择也要用它。原来的打分器是英文 wav2vec2，没法给中文打分。

- **模型**：`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（Apache-2.0，Common Voice zh-CN，按字输出）。
  - revision `99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd`；
  - 位置 `/opt/dlami/nvme/jasonleeeli/models/wav2vec2-large-xlsr-53-chinese-zh-cn/`；
  - `pytorch_model.bin` 的 sha256 `de031fd4…63ee4`，完整值见目录里的 `REVISION.txt`；
  - 用现有的 qwen3omni 环境加载，没有安装任何包。
- **独立性**：En→Zh 的评测 ASR 是 Paraformer-zh（`asr_transcribe` 按 tgt_lang=cmn 自动选择）。打分器与它是不同的模型，不会出现"拿评测 ASR 挑候选"的循环。
- **接口**：`si_ctc_scorer --lang zh`，只加选项，默认 en 不变，有单元测试。
  - 目标文本 = 前缀 + 本块，只保留模型词表里的字；拉丁字母转大写；空格和标点去掉；
  - NLL 按目标字平均，与英文相同。
- **CPU 自检**（`evaluation/ctc_zh_selfcheck.py`，`eval_outputs/mist_w3po_v1/en2zh/ctc_zh_selfcheck_v1.json`）：
  - 音频：train 会议 EN2001a、EN2001b 里 UniSS talker 生成的中文语音（zh_revoice 片段，取自 zh_mix_v8 会话）；
  - 条件：单人、1.5–8 s，共 24 条。

  | 目标文本 | NLL/字 均值 | 正确读法更低的比例 |
  |---|---|---|
  | 正确 | 2.97（中位数 2.11） | — |
  | 换成别的句子 | 14.85 | 1.00 |
  | 漏后半句 | 11.71 | 0.96 |
  | 多出半句 | 8.25 | 1.00 |
  | 30% 的字替换 | 7.68 | 1.00 |

  - 最优路径对文本的 CER 是 0.317。原因是会话混音里有背景声，速度也改过，不影响相对排序。
  - **结论：数值合理，能区分明显读错的候选。**
- **已知差异**：中文"每字"NLL 的量级比英文大，正确读法的中位数约 2.1。β = 5 和上限 10 都照原配方，**不调**。

### 4.2 候选采样（`scripts/en2zh_talker_orch.py`，与 zh→en 的 traindec / sample 阶段相同）

1. 用选中的 Thinker 检查点解码 train16 窗口（与 A′ 相同的解码参数，8 个分片）。
2. talker prep：`--session ihm_mix --diar nemotron_lowlat --sep ''`，再丢掉超过 400 块的流。
3. T1c（`si_t1c_20261021T181440Z/autoext/export_1500`，MIST v7/v8 的 En→Zh 也用它）加 `--direction en2zh`，按 A′ 的 talker 设置 c1+b1+s2 采样：`--speed-policy queue --speed-shift 2 --select-k 2 --cand-speeds 0,2 --cand-k 4 --dump-cands --cand-cover`。
   - 每块 8 个候选：A′ 语速 4 个，再快 2 档 4 个；全部用中文 CTC 打分；
   - 续写路径沿 A′ 规则（b1，k = 2）前进；
   - 8 个 talker 分片，4 个中文 CTC 打分服务器。
- zh→en 的 sample 阶段还在 dev5 上采样，那是给它的预检用的。En→Zh **不做预检**：PL-EM 已经是确定的配方。

### 4.3 训练行与训练

- **训练行**：`talker_rows --mode plem --lang cmn --units char`。新增两个选项，默认 eng / word 不变，有单元测试。
  - 序列里的语言 token 用 cmn（180372），与 si_infer 采样时一致；
  - 块的 W_n 按中文单位计。
  - 其余与 zh→en 完全相同：
    - 积压格：Δ 80 ms，B 8 s；
    - τ 0.5、β 5、f = −min(NLL, 10)；
    - λ 用对偶二分求，使预算 = A′ 规则路径自己的语料延迟 − 0.3 s；预算按中文单位、在 En→Zh 自己的数据上算；
    - 训练权重取 A′ 历史状态上的条件后验；q < 0.01 的候选不入行；
    - 语速软标签。
- **训练**：`run_talker_crf.sh MODE=plem`，从 T1c 全参数训练。
  - 学习率 3e-5，warmup 50，余弦退到 3e-6；1200 步；
  - 4 卡，GBS 4，每组 ≤ 24000 个 token；
  - 对 T1c 的 KL 权重 0.5，收尾 token 的 CE 权重 3，语速损失权重 1.0；
  - 种子 20261110，只训一次；
  - 每 400 步存一次，导出 400 / 800 / 1200（`export_e2.py`）。
  - 用 Thinker 让出来的 4 张卡：占槽 + takeover，约 30 分钟。

### 4.4 talker 检查点选择（同一个思路；`en2zh_select.py talker`）

- 每个导出在 dev5 上跑语音代理：
  - 输入：选中 Thinker 的 dev 全集解码，读级合并后切出 dev5；
  - 推理用 PL-EM c1b1s2 的设置：`--backlog-token --select-k 2`、queue 语速、speed-shift 2，加中文 CTC；
  - **只用一个 TTS 种子**（默认种子）。
- 每个候选取三个 dev5 数：
  - Q：语音 ASR-BLEU-all（Paraformer-zh，中文分词）；
  - O：语音 ASR-BLEU-ov；
  - L：每条流的语音 LAAL，现行协议。
- **规则**：可行集 L ≤ L(1200) + 0.1 s，在其中取 Q 最高的。打平（两位小数）时依次比较：O 更高 → L 更低 → 迭代数更大。
- 例外：1200 没有完整输出就不选，问协调者。
- chunk 协议的 LAAL 只记录。

## 5. 评测与指标

### 5.1 dev 全集（两个 TTS 种子：默认种子和 `--seed 20261109`）

| 代号 | 系统 | 角色 |
|---|---|---|
| **(c)** | A′rl + PL-EM：选中的 Thinker + 读级块 + 选中的 PL-EM | **主系统** |
| **(a)** | A′ + T1c c1+b1+s2：同一份 Thinker 解码，逐段输入 | 参照，对应 zh→en 测试预登记里的 (a) |

- 门槛（`gate_2seed`，(c) 对 (a)）：
  - 规则：点估计 ≥ 参考 − 0.2，且按会议配对 bootstrap P(差 ≤ 0) < 0.95；
  - 项目：文本 all、文本 ov（两者文本按构造相同）、语音 all（两种子合并）。
- 这只是**描述**：配方由用户事先定下，En→Zh 不论 dev 结果如何都跑一次 test。
- 但如果 (c) 的语音 < (a) − 0.2：先把结果报给用户，再跑 test。ASR-BLEU 的硬约束在报告里写明。test 照样跑 (c) 和 (a) 两个系统，不在 test 上换配置。

### 5.2 test（只跑一次；`scripts/en2zh_test.sh`）

- 数据：AMI En→Zh 测试集，16 场，C2 窗口 5126 个，参考见第 1 节。
- 系统：(c)、(a)，配置与 dev 逐项相同。
- 在 dev 全集评测完成后才开始（等 `EN2ZH_DEV_DONE`）。不在 test 上做任何选择。
- 合并 pair_lost 的规则同第 3.4 节。
- **zh→en 的测试集**（AMI zh→en、AliMeeting）按 `final_test_prereg_v1` 今晚在跑，结果并列报告，不在本计划里改动。

### 5.3 指标

1. **文本 tcpBLEU** all / ov / 1spk：`score_run --set en2zh`，即 v8 overlap_score，在线协议，sacreBLEU 的 zh 分词。
2. **语音 ASR-BLEU** all / ov / 1spk：Paraformer-zh 转写 talker 语音后的 tcpBLEU。两个种子各报一次，再报合并值。
3. **每条流语音 LAAL** all / ov / 1spk：`speech_latency_lit` 的 `speech_laal_online`。
   - 现行协议用 C2 窗口，chunk 协议用 chunk 窗口，两种都报；
   - 两个种子取平均；**这是主延迟数字**。
4. **EndOffset**：单元级，online，两种协议。还报单通道 LAAL（online_1ch），只作补充。
5. **文本 LAAL**（`--text-only`），两种协议。
6. **中文延迟单位**（`evaluation/zh_units.py`、`en2zh_lat.py`，新文件，有测试）：
   - 一个汉字算一个单位，一个拉丁词或数字串算一个单位，标点不计；
   - 工具本身不改：在把文本"按单位加空格"后的副本上运行；
   - 参考集为 En→Zh，universe 为该会议全部有标注的 AMI 话语（同 overlap_score）。
   - 在 zh→en 上用"原样"模式运行，与已存的 A′rl + PL-EM 延迟 json **逐字段相同**，说明包装本身不改变数字。
   - **跨方向的 LAAL 不能直接比**：单位不同，一个是英文词，一个是汉字。
7. **听感**：
   - UTMOS、SIM（`talker_quality`）；
   - UTMOS 配对差：(c) 对 (a)，同流配对，两种子合并，按会议 bootstrap（`listen_paired_meeting`）；
   - talker CER（`evaluation/talker_cer_zh.py`，新文件）：talker 输入文本对 Paraformer 转写，按中文单位，含截断比例；
   - 说话人内部停顿和每块单位数（`talker_block_pauses`，在按单位加空格的副本上跑）。
- 不算 COMET、MetricX。

## 6. GPU 计划

- 0–3 号卡或 4–7 号卡由协调者指定。
- **Thinker 训练**：4 张独占卡，训练卡上不叠任何东西（由 sync_train_exclude 保证）。
- **另外 4 张卡**：每个检查点的 dev5 解码（4 个分片，每卡最多 2 个解码，每卡 ≤ 135 GiB，`--cap135`）和 held-out 损失（1 个作业）。每个检查点约 15–20 分钟；检查点约每 37 分钟出一个，所以评测跟得上。
- **训练结束后**：
  - dev 全集解码、train16 解码、候选采样用 8 张卡（编排器按真实显存放置，用共享账本，同卡 vLLM 启动间隔 ≥ 60 s）；
  - PL-EM 训练占用 Thinker 让出的 4 张卡，约 30 分钟；
  - 之后 dev5 语音代理、dev 全集语音链、test 用全部可用的卡。
- 占卡程序由编排器按 PID 停掉；守护脚本 v3 不碰；其他代理的进程不碰。

## 7. 墙钟估计（从拿到卡开始，T0；布拉格时间按 T0 = 02:00 计）

| 阶段 | 时长 | 约到 |
|---|---|---|
| Thinker 3000 步：每步约 4.3–4.8 s（每步行数是 zh→en 的约 1.23 倍；zh→en 同类 4 卡训练是 3.8–4.0 s/步） | 3.6–4.0 h | 06:00 |
| 最后一个 dev5 代理和选择 | 0.3 h | 06:20 |
| dev 全集解码（4996 个窗口，约是 zh→en 的 1.4 倍），同时做 train16 解码（7600 个窗口） | 1.2 h | 07:30 |
| 候选采样（约 26k 块，是 zh→en 的约 1.35 倍） | 1.3 h | 08:50 |
| rows、PL-EM 训练、导出 | 0.7 h | 09:30 |
| dev5 语音代理（3 个，1 个种子，并行）和选择 | 0.5 h | 10:00 |
| dev 全集 (c)、(a)，各 2 个种子；门槛、两种协议的延迟、听感 | 1.3 h | 11:20 |
| test 一次：解码、合并、(c) 与 (a) 各 2 个种子、事后指标 | 2.0 h | 13:20 |

- **dev 结论约在 T0 + 9.5 h，test 结论约在 T0 + 11.5 h**。卡被其他评测占着时会推迟。
- 第一条消息里给的"约 7.5 h / 9 h"偏乐观：当时还不知道每步行数多 23%、train16 多 35%。以本表为准。

## 8. 脚本与启动命令（都只用 `bash -n` 和 DRY_RUN / `--dry` 检查过，没有试跑）

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=en2zh_v2p_$(date -u +%Y%m%dT%H%M%SZ); C=0,1,2,3      # 训练用的 4 张卡由协调者指定（0,1,2,3 或 4,5,6,7）
RUN_ID=$R CARDS=$C setsid nohup bash experiments/mist_w3po_v1/scripts/en2zh_train_thinker.sh > logs/mist_w3po_v1/en2zh/train_thinker.log 2>&1 < /dev/null &
RUN=$R TCARDS=$C setsid nohup bash experiments/mist_w3po_v1/scripts/en2zh_ckpt_proxy.sh > logs/mist_w3po_v1/en2zh/ckpt_proxy.log 2>&1 < /dev/null &
RUN=$R TALKER_CARDS=$C setsid nohup bash experiments/mist_w3po_v1/scripts/en2zh_after_select.sh > logs/mist_w3po_v1/en2zh/after_select.log 2>&1 < /dev/null &
RUN=$R setsid nohup bash experiments/mist_w3po_v1/scripts/en2zh_test.sh > logs/mist_w3po_v1/en2zh/test.log 2>&1 < /dev/null &
```

- 四个脚本可以同时启动，各自等待自己的事件。
- 关键日志行：
  - `EN2ZH_THINKER_TRAINING_STARTED`
  - `EN2ZH_THINKER_SELECTED <step>`
  - `EN2ZH_TALKER_SELECTED`
  - `EN2ZH_DEV_DONE`
  - `EN2ZH_TEST_DONE`
  - 需要人决定时：`EN2ZH_STOP …` 或 `EN2ZH_PROXY_FAILED …`
- 新增代码都只做加法，默认值不变，配有单元测试：
  - `data/build_online_windows_en2zh.py`、`data/chunk_windows_en2zh.py`；
  - `crf/build_sched_windows.py` 的 en2zh 选项；
  - `crf/merge_read_segments.py --sep`；
  - `crf/heldout_loss.py`；
  - `evaluation/si_ctc_scorer.py --lang zh`、`evaluation/zh_units.py`、`evaluation/en2zh_lat.py`、`evaluation/talker_cer_zh.py`、`evaluation/ctc_zh_selfcheck.py`；
  - `talker/talker_rows.py --lang cmn --units char`；
  - `scripts/en2zh_orch.py`、`scripts/en2zh_talker_orch.py`、`scripts/en2zh_select.py` 和上面四个 shell 脚本。
- `en2zh_orch.py` 对每条命令都做检查：不能残留 zh→en 的路径或参数（zh_mix_v8、nemotron_lowlat_zh8、`--dir/--direction/--set zh2en`、zh→en 的参考集和 PL-EM 导出、直接调用 speech_latency_lit）；test 命令里不能出现 dev 路径。

## 9. 已知风险

1. **每步更慢**：pass 1 行数是 zh→en 的 1.37 倍，最大的一步有 703 行。用的仍是 120 GiB 的分配器上限和 32000 的 token 预算；第一步之后核对显存和步时。
2. **中文 CTC 的识别力**：在混音片段上 CER 是 0.32。实际的 talker 语音更干净，但 f 的量级比英文大，PL-EM 的 λ 会自动适应预算。照配方，不调参。
3. **Paraformer 的分词**：中文 ASR-BLEU 由 sacreBLEU 的 zh 分词决定。跨方向的 BLEU 只作并列，不作比较。
4. **dev5 选择噪声**：dev5 的文本 tcpBLEU 在检查点之间可能只差 ±0.3。规则已经写死，不另作解释。


## 修正 1（2026-11-13 07:20 布拉格，执行者；只改卡的放置，不改选择规则、门槛或任何训练 / 解码配置）

用户（约 06:50、06:55 布拉格）："都改成叠加跑dev5速查"；0–3 叠 (a) 的 dev5 速查和 PL-EM 训练。

1. **PL-EM 开训条件**（`en2zh_after_select.sh` 的 `STACK_PLEM=1`，默认关闭时行为不变；提交 26ac3caf3）：
   - 和原来一样，先用 noclobber 占 slotA。占槽后 `sync_train_exclude` 会排除 0–3，新的作业进不来。
   - 原条件"0–3 上没有任何非占卡进程"改为：0–3 每卡实际占用 + 56 GiB（PL-EM 峰值）+ 3 GiB（余量）≤ 135 GiB，也就是 ≤ 76 GiB，相当于每卡最多还有 1 个解码在跑；同时卡上没有别的训练。正在跑的解码不停。
2. **PL-EM 峰值的来源**：三次 zh→en 的 PL-EM 训练（talker_crf_v1/v2/v3），rank 0 的 max reserved 分别是 51606 / 51186 / 43820 MiB，最大约 50.4 GiB；再加 CUDA 上下文、NCCL 和其他 rank 之间的差异，按 56 GiB 计。
3. **训练中继续叠放**：PL-EM 日志出现第 400 步存盘（之前三次的峰值都在这之前达到）后，写 `logs/mist_w3po_v1/slots/stack_ok_cards_a`（"0 1 2 3"）。`sync_train_exclude`（新环境变量 `STACK_OK_FILE`）不再排除这几张卡，新的解码按 --cap135 以实际占用叠到 135 GiB。训练结束后这个文件改名释放。
4. 两个等待进程已按 PID 停掉，用相同的 argv、env、cwd 重启，只加了上面的环境变量，日志接着原来的文件追加：
   - `en2zh_after_select.sh`：3192301 → 1222510；
   - `sync_train_exclude.sh`：1255324 → 1222453。
