# 鲁棒第十四波：ERM-WiSE 换用 UniST 训练数据（2026-11-20 15:30 CET，写在任何数据构建、训练和评测之前）

## 起因
- 第十三波确认，ERM 在干净中→英上掉分的原因是 CoVoST 中→英训练集太窄：只有 7004 条、22 个说话人，训了约 10.5 个 epoch。证据：训练子集上 +13.31，dev 上 −3.63；离线翻译 −2.77。
- 不含 QABC 的 LA2 基线中，Talker（T1c）是在 UniST 上训练的。UniST 是开源数据集（HuggingFace cmots/UniST）。本机有它的 15 分片 pilot：中→英 683341 条，约 1361 h；英→中 816659 条，约 1502 h，音频齐全。
- 用户 15:20 要求不要让卡空着。按"有需要点头的事按最优选择决定"的授权，开始本波。

## 唯一的变化：训练数据来源
- 两个方向都改从 UniST pilot 取：`data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source`；参考译文用 manifest 的 `translation` 字段。
- 每个方向按 sha1("20261120|w14|<dir>|<id>") 排序取前 120000 条。英→中与第五波相同规模；中→英从 7004 条增加到 120000 条。两个方向都约 0.6 个 epoch。
- 预处理同第五波：`bench_build.trim_norm`（切边、−26 dBFS、1.5–12 s），失败的行丢弃。
- **排除（构建时断言并记录数量）：**
  - UniST 留出验证句：`crc32(id) % 50 == 0`；
  - 与任何评测或测试条目重叠的行：S1 dev 的 (a)/(b) 条目、Idea 2 的 cvt 流、RealSI、ACL、AliMeeting、OOD-3、电竞。按源转写文本归一化后完全相同，或音频文件名、Common Voice clip id 相同来判定。重点是 UniST 英→中里的 CommonVoice_EN 与 CoVoST dev/test 的重叠；
  - bench/HELDOUT_SPEECH.txt 中的说话人（能对上的话）。
- 其余全部与第五波 ERM 相同：
  - 8 个环境（两个方向 × 干净/音乐/人群/音效），噪声库、BVR 取值、限幅相同；
  - 2300 步，全局批量 64，LoRA rank 16、alpha 32，q/k/v/o；
  - AdamW 1e-4，余弦调度，种子 20261119；行号和噪声的 sha1 键沿用 "20261119|w5m|..."。
- **并行方式：** world 4（卡 0–3），每个 rank 16 条，一个微批。只影响梯度累积切分，与第八波实现修正 2 的论证相同。
- **最终系统：** unist_wise05 = W0 + 0.5·Δ（alpha 减半），与 ERM-WiSE 的协议完全相同。unist（α = 1）作为辅助一并评测。

## 评测与判定（写定）
- 链与打分同第五、十、十一波：si_thinker_stream la2 w1 → T1c（4 个分片，种子 20261021）→ 只对语音做 ASR；用 3 次底座运行、噪声底、bootstrap。
- **主判定：**
  1. unist_wise05 相对底座，用统一规则：
     - 所有 (a)/(b) × 干净/带噪单元，text BLEU 和 ASR-BLEU 的点估计 ≥ −0.3，且 CI 上界 ≥ 0；
     - 至少一个带噪单元 text 增益的 CI 下界 > 0，且超过噪声底。
  2. unist_wise05 相对 erm_wise05 的增量：
     - 至少一个单元（干净或带噪）text BLEU 差的 CI 下界 > 0，且点估计超过噪声底；
     - 没有任何 (a)/(b) 单元的点估计 < −0.3 且 CI 上界 < 0（text 和 ASR 都查）。
- **预先说明：**
  - UniST 的译文是合成的，风格与 CoVoST 的人工参考不同，(a) 单元可能因此吃亏。所以 unist（α = 1）在中→英干净上的结果另列，作为第十三波诊断的延续：数据变宽后 α = 1 还掉不掉分。
  - 若第 1 条过、第 2 条不过，记为"换数据对 ERM-WiSE 没有增量"。
- **两条都过时：** 跑完整系统组合（+ V1 音色 + T1c，`wave5_m_combo_*` 的同一协议），再与 ERM-WiSE 的完整系统比较。
- **辅助：** LAAL。

## 卡与时间
- 卡 0–3。数据构建（CPU）约 30 分钟，训练预计 2–3 小时，评测约 1.5 小时。

## 实现说明（2026-11-20 15:28 CET，写在任何数据统计、训练和评测数字之前）
1. **并行方式（按主会话 15:35 的指示改为 8 卡）：** 第八波已结束，卡 4–7 已交还，所以训练用卡 0–7，world 8，每个 rank 8 条，`--mb 8`，一个微批。全局批量仍是 64；每个 slot j 的样本仍由 "20261119|w5m|row|s|j" 和 "20261119|w5m|mix|s|j" 决定，与 world 无关。world 只决定 slot 落在哪个 rank 上（rank r 取 (j % 8) % W == r 的 slot），各 rank 的梯度相加后除以 world，所以损失仍是 64 条样本的平均 CE。论证同第八波实现修正 2。冒烟如果显示 world 8 比 world 4 / `--mb 16` 更慢，或者显存不够，就退回卡 0–3、world 4、`--mb 16`，并在这里补记。
2. **数据（`experiments/mist_w3po_v1/voicecond/wave14_data.py`）：**
   - manifest 每行自带 `source_audio` 的绝对路径（`parts/train-XXXXX/source_audio/<全局序号>_<id>.flac`，16 kHz 单声道 PCM16），直接用这个字段，不靠文件名去推。
   - 方向：`src_lang` 为 cmn 的行算 zh2en，eng 的行算 en2zh；源文本用 `transcription`，参考译文用 `translation`；两者任一为空就丢弃，同第五波。
   - 已知事实：pilot manifest 里所有行的 `audio_origin` 都是 `bicodec_reconstructed`，即源音频是 BiCodec 重建的语音，不是原始录音。这一点照实记录；按预登记仍用这份数据。
   - 处理顺序：先做排除，再按 sha1("20261120|w14|<dir>|<id>") 升序取前 120000 条，然后过 trim_norm，失败的丢弃（第五波也是先取行再丢 trim_norm 失败的）。
   - 输出 `eval_outputs/mist_w3po_v1/wave14/data/{rows_<dir>.jsonl, clips/<dir>/<id>.wav, stats.json}`，字段与第五波一致。UniST 没有说话人字段，`spk` 记为 null。
3. **重叠判定：**
   - 文本归一化：NFKC，转小写，只保留 `str.isalnum()` 为真的字符（字母、数字、汉字），去掉其余所有标点和空白；判定为完全相等。
   - 测试/评测文本库的来源：
     - CoVoST 2 两个方向的 dev（HF validation）和 test 全集的 `sentence` 与 `translation`；
     - S1 的 `wave2_S1/data/*.jsonl` 和 `wave1_K2K3K5/data/*.jsonl`；
     - OpenBroadcast-SI 的 bench 和 bench_dev 下全部 json/jsonl；
     - Idea 2 cvt 的 index 和 refs/*/info.json；
     - Idea 2 ood3 的 *.jsonl；
     - RealSI 的 json；
     - ACL 60/60 dev 的 jsonl；
     - AliMeeting（alimeeting_si_v1、alimeeting_eval_si_v1、diarist_alimeeting_refs_v1）的 json/jsonl；
     - AMI（ami_multispk_si_v1 顶层和 index 下的 jsonl）；
     - 电竞（game_esports_si_v1 下全部 json/jsonl）。
   - 取文本的方式偏保守：取这些文件里的所有字符串值（递归），只跳过含 "/" 的路径和形如 id 的无空格 token（只由字母、数字和 `_.:-` 组成，且含数字或下划线）。所以译文字段一旦和 UniST 的源文本相同，也会触发排除。每一类来源分别计入 stats.json，一行命中多类时每类都计。
   - 音频文件名和 Common Voice clip id 无法比对：UniST 的 id 是它自己的内部序号（例如 CommonVoice_EN_0000499569），manifest 里也没有原始文件名或 client_id，所以对 CommonVoice_EN 只能靠文本比对。按上面的文本库，CoVoST dev/test 两个方向全集都在其中。
   - bench/HELDOUT_SPEECH.txt：说话人对不上，因为 UniST 没有说话人字段，这一点在 stats.json 中写明。它列出的 clip 都是 CV test 的条目，由 CoVoST test 文本覆盖；在 CoVoST dev/test 中找不到的 clip 数也会写进 stats.json。
   - UniST 留出验证句：crc32(id 的 UTF-8 字节) % 50 == 0 的行排除。
   - 构建完成后断言：选中的每一行都不在文本库中，也不满足 crc32 留出条件。
4. **训练：**
   - `wave14_train.py` 是 `wave5_m_train.py` 的副本，只改了 DATA 目录（另加说明文字），arm 用 erm。
   - 启动用 `wave14_run.sh`，watcher 是 `wave14_noplaceholder.sh`，有自己的 PIDS 和 STOP 文件。
   - 整条链由 `wave14_chain.sh`（setsid nohup）自动接续：训练 → WiSE → 合并 → 评测 → 打分。
5. **WiSE：**
   - `run_unist_wise05/adapters_final.pt` 是 `run_unist/adapters_final.pt` 的硬链接。
   - json 与 `wave5/M/run_erm_wise05/adapters_final.json` 同格式，alpha 为 16，即 W0 + 0.5·Δ。
6. **评测：**
   - `wave14_eval.sh` / `wave14_eval_orch.sh` 是第十波的副本。每个 arm 用 4 张卡：unist_wise05 用卡 0–3，unist 用卡 4–7，两条并行。
   - Thinker：en2zh 在第一张卡，zh2en 在第二张卡。
   - Talker：每个方向 4 个分片（种子 20261021，`--num-shards 4`），分片 0–1 和 2–3 分到两张卡上，结果与在一张卡上跑相同。
   - ASR：在第一、二张卡上。
7. **打分（`wave14_score.py`，`wave10_score.py` 的副本）：**
   - 系统：base（S1 + 2 次 Thinker 重跑）、erm_wise05、unist_wise05、unist。
   - 第 1 条：同第五波统一规则。text 对 3 次底座的均值，ASR-BLEU 对 S1。不降的范围是方向 × {a, b} × {干净, 带噪}。增益看方向 × {a, b, b_no_crosstalk} × 带噪的 text，要求 CI 下界 > 0 且点估计大于该单元的底座噪声底（3 次底座两两差的最大值）。
   - 第 2 条：
     - 增量的单元范围是方向 × {a, b, b_no_crosstalk} × {干净, 带噪}，看 text 的 unist_wise05 − erm_wise05，要求 CI 下界 > 0 且点估计大于该单元的噪声底；
     - 不伤的范围是方向 × {a, b} × {干净, 带噪} × {text, ASR}，不能有点估计 < −0.3 且 CI 上界 < 0 的单元。
   - unist（α = 1）只报告相对底座和相对 erm_wise05 的差，不设门；zh2en/a/clean 单独列出。
   - 自检：复算 erm_wise05 相对底座的结果，须与 `wave5/M/summary_erm_wise05.json` 完全一致。

### 实现说明更正（2026-11-20 15:34 CET，追加；上面第 1、6 条不改动，以此为准）
- 主会话 15:45 的消息撤回了"用 8 张卡"的指示。第十四波按原预登记执行：训练和评测都只用卡 0–3，world 4，每个 rank 16 条，`--mb 16`，一个微批，全局批量 64 不变。卡 4–7 不碰。
- 已经发生的事：在更正前跑过一次 12 步的 world 8 冒烟（`wave14/smoke_w8`，卡 0–7，约 15:32–15:33 CET 结束，不留 checkpoint）。它只确认了数据读取、标签和显存没有问题，不参与任何结果。冒烟期间卡 4–7 上的占位进程被停过，冒烟结束后交还 guard。
- world 4 / `--mb 16` 没有另跑冒烟。第八波 AUL 用同一个模型、同一种切分跑完过 2300 步；正式训练的头几步就当作这一配置的冒烟，显存和每步秒数记在"训练进度"里。
- 评测：unist_wise05 用卡 0–1（en2zh 在卡 0，zh2en 在卡 1，每张卡 4 个 Talker 分片），unist 用卡 2–3，两条并行。`wave14_eval_orch.sh` 在调用时把 CC=CA、CD=CB 传进去，这样就退回第十波的两卡分法。
- 打分脚本自检已通过（尚无 unist 的数字）：复算的 erm_wise05 相对底座的 180 个数值和第 1 条的门，与 `wave5/M/summary_erm_wise05.json` 完全一致。

## 训练进度
- 2026-11-20 15:34 CET 启动 `run_unist`：卡 0–3，world 4，`--mb 16`，torchrun 的 pid 是 2046230。后续链条 `wave14_chain.sh` 的 pid 是 2047792。
- 前 30 步：没有跳过的步，峰值显存 69.4 GiB/卡，第 10–30 步平均每步 3.20 s。按这个速度，2300 步约 2.05 h，预计 17:40 CET 前后结束。
- 之后链条自动接续：WiSE → 合并（CPU，约 15 分钟）→ 评测（约 1.5 h）→ 打分。预计 19:30 CET 前后出结果。
- 对照：world 8 / `--mb 8` 冒烟是每步 2.9 s，与 world 4 相差不大（瓶颈在数据加载）。

## 数据统计（`eval_outputs/mist_w3po_v1/wave14/data/stats.json`，构建于 15:29 CET，用时约 3 分钟）
| | zh2en | en2zh |
|---|---|---|
| manifest 行 | 683341 | 816659 |
| UniST 留出（crc32 % 50 == 0） | 13602 | 16388 |
| 与评测/测试文本重叠（任一类） | 53 | 103791 |
| 　其中 CoVoST dev/test | 14 | 94785（其中 CommonVoice_EN 94498） |
| 　S1 dev | 0 | 218 |
| 　OB-SI bench / bench_dev | 1 | 8460 |
| 　Idea 2 cvt | 0 | 6596 |
| 　OOD-3 | 9 | 1389 |
| 　RealSI | 5 | 472 |
| 　ACL 60/60 | 1 | 80 |
| 　AliMeeting | 27 | 1673 |
| 　AMI | 0 | 3683 |
| 　电竞 | 21 | 2017 |
| 空文本 | 0 | 0 |
| 排除合计（去重） | 13655 | 118101 |
| 可选 | 669686 | 698558 |
| 按 sha1 取前 | 120000 | 120000 |
| trim_norm 失败丢弃 | 14747 | 16764 |
| **最终保留** | **105253（179.6 h）** | **103236（161.8 h）** |

- 表中各类重叠是分类计数，一行可以同时命中多类，所以各类相加大于"任一类"。
- 构成：zh2en 主要来自 emilia_zh（102190 条），另有 HQ-Conversations 1940 条、NCSSD_R_ZH 1123 条。en2zh 主要来自 emilia_zh 的英文部分（37347 条）、CommonVoice_EN（22706 条）、LibriSpeech（5359 条）、VCTK（4619 条）、DailyTalk（2406 条）和 NCSSD_R_EN（616 条）；其余约 3 万条是 Emilia 风格 id 为 EN_B… 的英文段，stats.json 的 corpus_mix 把它们按完整 id 前缀拆得很碎，这只是统计口径的问题，不影响数据。
- CommonVoice_EN 中有 94498 条，源文本与 CoVoST dev/test 的某句相同。Common Voice 的同一句话会由不同的人朗读，所以文本相同的录音也可能来自训练集说话人；这里一律排除，以防测试文本泄漏进训练。
- HELDOUT_SPEECH 中的 2609 个 clip 都能在 CoVoST dev/test 的 id 中找到，所以已经按文本覆盖。553 个说话人无法比对，因为 UniST 没有说话人字段。
- 断言全部通过：选中的行不在测试文本库中，不满足留出条件，也没有空文本。
