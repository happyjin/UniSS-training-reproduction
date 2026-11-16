# E9：同训练数据的"先识别、后归属"级联 S2ST 基线（Table 1 第 S3 行）

- 登记：`reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md` §1 S3、§4（含修订 1）。
- 测试集只做推理和打分，每个集合只解码一次，不调任何参数；不训练任何东西。
- 时间一律为布拉格时间（CET = UTC+1）。

## 1. 可行性判定（§4，1 小时 CPU 准备 + 3 个 dev 窗口冒烟）：**GO**

判定时间：2026-11-16 18:26（布拉格），准备从 18:08 开始，约 18 分钟；测试链 18:26 启动。

### (a) 检查点：找到，补全后可用

- 模型就是 prereg_v8 的 C1（`docs/mist_prereg_v8.zh-CN.md` §1 "Relabel" 行）：
  - `eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_zh2en_v8/thinker_merged`（AMI Zh→En、AliMeeting 共用）
  - `eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged`（AMI En→Zh）
  - 以 o2nosd 为基座，NOSD=1（说话人流置零），数据、种子 20261026、3000 次迭代、GBS 64 都与 MIST-Simul 相同（作业 `reports/gstar_si_v8/jobs/c1_train.json`）。
- **问题**：换服务器后两个目录都不完整，缺 `model-00014/00015-of-00015.safetensors` 以及 config、tokenizer、index 等 10 个小文件；S3 上的副本也缺同样的文件（实时同步只上传普通文件，这些文件当时应是链接）。
- **为什么能无损补全**：
  - 合并日志 `logs/multispk/v8/thinker_gstar_s2dlnosd_*_v8.log` 写明 `"shards_rewritten": 13`；
  - 按 release 的 index，第 1–13 个分片正好装着被改动的注意力层、embed 和 lm_head；
  - 其余文件由 `merge_thinker_gstar.py` 从基座 o2nosd 原样复制（`shutil.copy2`），而基座里这些文件与 release 逐字节相同（`cmp`）。
- **处理**：只新增、不覆盖。在两个目录里为缺失的文件建立相对符号链接，指向 `thinker_gstar_o2nosd_20261029b/thinker_merged/` 中的同名文件；已有的 13 个分片、`sd_proj.pt`、`gstar_tokens.json` 不动。冒烟测试中模型正常加载（2034/2034 个权重）。
- 注意：S3 恢复会再次丢掉这些链接，届时按本节重建即可。

### (b) B 的解码与合成方式（照搬）

- **入口**：`joint_test.sh`（BASE_TALKER=t1c）→ `joint_orch.py` → `final_test_orch.py` / `en2zh_orch.py` → `thinker_t_eval_orch.py`。
- **解码**：`decode_crf_ckpt_finalpost --crf-ckpt <identity>`，identity 检查点不折叠任何适配器，等价于直接用合并后的 Thinker。参数如下：
  - c2g06 在线测试窗口，1.0 s 读取；
  - `--final-read-post close_detect --policy wait --lag-ms 2000 --context 0 --batch 16 --max-gib 120`；
  - 循环保护 `--loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50`；
  - 按会议分 8 个分片。
- **语音**：
  - T1c `si_infer`，参数 `--speed-policy fixed --history-ban 8 --gpu-mem 0.1`，每个种子分 8 个分片；
  - 两个 TTS 种子：默认种子和 20261109；
  - talker 输入由 `gstar_talker_v6 prep --causal --blocks --preset lowlat` 生成（参数按集合设置），再用 `drop_long_streams` 过滤（400 块）。
- **识别与打分**：仅语音部分做 ASR（英文 Whisper-large-v3，中文 Paraformer-zh），再经 `asr_hyp` → `score_run --asr`（tcpASR-BLEU）。
- **延迟**：chunk 协议下按 `joint_test.sh` 的 `lat` 计算；Table 1 的延迟单元由 `table1_lat.sh` 计算（LAAL_block / LAAL_spread / StartOffset / EndOffset）。

### (c) 解码与重标的接入点

- **解码**：
  - 现在的 `online_decode` 引擎没有 `--zero-sd`。
  - 新增一个很薄的包装 `evaluation/online_decode_finalpost_zerosd.py`：把加载好的说话人投影器的输出换成同形状的零张量，其余全部沿用 `online_decode_finalpost`。
  - 这与训练时的置零方式（`thinker_gstar_model.py`：`use_sd=False` → `zeros_like`）一致，也与 v8 重叠研究解码这两个模型时用的方式一致（`gstar_si_stream_online_v6s --zero-sd`）。
- **因果重标**：
  - v8 重叠研究是在打分器里重标的（`overlap_score.score(relabel=True)` 调用 `relabel_causal`）；E7 的 `build_e7.py` 也以同样方式调用它。
  - 新增 `evaluation/relabel_causal_run.py`：用同一个函数、同样的在线提交时刻（≥ open_detect；读到窗口末尾时 ≥ close_detect）和 `post.lowlat`，生成一个新的解码目录 `<name>_rel`。现有的打分、talker、延迟工具都可以原样读取它。
  - 规则本身不改：取片段区间内平均后验最大的槽位；没有可用帧时，取 [a − 2 s, 提交 − 1.04 s] 内最活跃的槽位；仍然没有就记为 `unk`。
- **`unk` 的合成**：
  - `gstar_talker_v6 prep` 遇到 `unk` 会报错，因此新增 `evaluation/gstar_talker_v6_unk.py`：流 id 写成 `<窗口>__Sunk`（与 E7 及 `asr_hyp` 的查找方式一致），音色提示用 prep 自带的回退（窗口开头到该流第一次提交之间的音频）。
  - 没有 `unk` 时，输出与原 prep 逐字节相同（冒烟数据上的 items、blocks 和音色 wav 都已核对）。
- **编排**：
  - 新增 `scripts/e9_orch.py`：完全复用 `joint_orch.py`，只在最后替换上面两个模块名，并检查每条命令。
  - 解码命令与 B 的逐项比对，只有模块名、`--crf-ckpt` 和 `--thinker` 不同。

### 冒烟测试（3 个 dev 窗口，卡 4，2026-11-16 18:21–18:24）

- **窗口**：zh→en c2g06 dev 的 ES2011a_on_0022、0026、0054。卡 4 的占位训练先按 PID 停掉（supervisor 769513、worker 770908，事先核对过命令行）。
- **各步结果**：
  - 解码：含加载共 63 s，日志打印 `ZERO_SD`，3 个窗口全部完成，单卡峰值约 66.5 GiB；
  - 重标：16 个片段，`unk` 为 0，5 个片段的标签与模型自己输出的不同；
  - prep：5 条流；
  - T1c：rc 0；
  - ASR：5/5；
  - `score_run --asr`：rc 0；
  - chunk 协议语音延迟：rc 0。
- 输出在 `/opt/dlami/nvme/jasonleeeli/work/e9_smoke/`，不入库。

### GPU

- 协调者分配卡 4–7（另一个代理用 0–3），E8 不在这几张卡上，因此不触发 §4 的"放不下"条件。

## 2. 测试运行配置

- **启动**：`scripts/e9_test.sh`。三个集合并行，间隔 120 s 启动，各自在卡 4–7 上运行（`--exclude 0 1 2 3`）。
- **命名**：
  - 解码：`decode/jtest_<set>_cascade_blind_s03000`；
  - 重标后：`decode/jtest_<set>_cascade_blind_s03000_rel`；
  - 语音：`decode|speech/jtest_<set>_cascade_blind_s03000_rel__t1c[_tts2]`；
  - set 取 ami、ali、en。
- **汇总**：`eval_outputs/mist_w3po_v1/e9_cascade/e9_summary_v1.json`（`evaluation/e9_summary.py`）。
- **与 B 的差别**（只有以下几处，均写明原因）：
  1. **说话人流置零**：模型本身就是按置零训练的，这是该模型唯一正确的用法。
  2. **重标后的标签作用于 talker 的块**：一个片段若跨多次读取，其中较早写出的块也用片段结束时定下的标签（按 v8 规则，标签在片段提交时确定，保证文本和语音的说话人一致）。这类块的数量见 `relabel_stats.json` 的 `blocks_before_segment_close`。
  3. **`unk` 流的音色提示**：用 prep 自带的回退。
  4. **规划上限**：每卡 129 GiB（本次的规则：已用 + 估计 + 6 GiB ≤ 135 GiB；B 当时是 135 GiB）。解码的估计仍是 64 GiB，实测一个解码稳定在 62–68 GiB，所以只有当卡上已有的解码不超过 65 GiB 时才会叠加第二个。这只影响速度，不影响输出。
  5. **检查点补全**：见 §1(a)。

- **占位训练**：每个作业上卡前，由编排器（`talker_crf_orch.YieldOrch`）按 PID 停掉该卡的占位训练。它先核对 nvidia-smi 进程所在卡和命令行中的 `train_mist_workload.py`，再停 worker 及其子进程，supervisor 随后自行退出。guard 未被停止，测试结束后 guard 已把卡 4–7 补回占位训练。

## 3. 结果（测试集，一次运行，18:26–23:50 布拉格）

- 6 个编排器都正常结束，记录为 `DONE; failed/skipped []`；测试链最后打印 `E9_TEST_DONE`。
- 汇总：`eval_outputs/mist_w3po_v1/e9_cascade/e9_summary_v1.json`。
- 运行目录：
  - `decode/jtest_{ami,ali,en}_cascade_blind_s03000`：说话人无关模型的原始解码，说话人标签是模型自己猜的；
  - `decode/jtest_{ami,ali,en}_cascade_blind_s03000_rel`：重标后的文本；
  - `decode|speech/jtest_{ami,ali,en}_cascade_blind_s03000_rel__t1c[_tts2]`：两个种子的语音；
  - 编排日志：`logs/mist_w3po_v1/e9_cascade/`。
- 延迟都按 chunk 协议计算，单位为 s。语音指标是两个 TTS 种子的均值；B 一列取自同一批文件，用同一套函数计算。

### 3.1 主结果（级联 E9 对 B）

| 集合 | 系统 | tcpBLEU 全部 / ov / 1spk（文本） | tcpASR-BLEU（`score_run --asr`，两种子均值） | 文本 LAAL | LAAL_block | LAAL_spread | StartOffset | EndOffset |
|---|---|---|---|---|---|---|---|---|
| AMI Zh→En | **E9** | 23.51 / 15.65 / 26.93 | **18.37**（18.41, 18.33） | 3.347 | 3.510 | 4.397 | 3.391 | 4.546 |
| | B | 25.02 / 19.97 / 27.10 | 18.06 | 2.986 | 3.119 | 4.014 | 2.986 | 4.083 |
| AliMeeting Zh→En | **E9** | 19.94 / 14.50 / 22.20 | **15.91**（15.69, 16.12） | 3.054 | 3.551 | 4.585 | 3.272 | 4.866 |
| | B | 21.43 / 16.91 / 23.05 | 16.73 | 2.823 | 3.279 | 4.276 | 3.052 | 4.565 |
| AMI En→Zh | **E9** | 39.96 / 30.87 / 43.19 | **30.97**（30.89, 31.05） | 2.543 | 2.803 | 3.510 | 2.740 | 3.720 |
| | B | 41.15 / 36.16 / 42.51 | 31.64 | 2.368 | 2.539 | 3.210 | 2.434 | 3.484 |

- **语音转写的 ov / 1spk**（两种子均值，E9 对 B）：

  | 集合 | ov（E9 / B） | 1spk（E9 / B） | 说话人覆盖 CR（E9 / B） |
  |---|---|---|---|
  | AMI | 11.16 / 12.48 | 21.20 / 20.39 | 0.47 / 0.69 |
  | Ali | 10.28 / 11.69 | 18.50 / 18.74 | 0.46 / 0.54 |
  | En→Zh | 21.73 / 24.69 | 35.10 / 34.72 | 0.54 / 0.70 |

- **重标前**：说话人标签是模型自己猜的，文本 tcpBLEU 只有 AMI 10.91、Ali 11.42、En→Zh 24.38，说明归属几乎全靠因果重标。重标改动的片段比例分别为 69%、64%、78%。

### 3.2 Table 1 口径（探索性，`table1_analysis.py --extra`，不是正式的 Table 1 文件）

- 文件：`eval_outputs/mist_w3po_v1/e9_cascade/table1_analysis_with_e9_v1.json`。
- cpASR-BLEU（E9 / B / MIST）：

  | 集合 | E9 | B | MIST |
  |---|---|---|---|
  | AMI | 24.88 | 24.72 | 26.84 |
  | Ali | 21.81 | 22.60 | 24.04 |
  | En→Zh | 35.07 | 35.83 | 36.49 |

- `score_ext` 算出的 tcpASR（两个种子合并计算）：AMI 18.37，Ali 16.16，En→Zh 30.97。Ali 的 `score_run` 两种子均值是 15.91，两者口径不同。
- **MIST − E9**（会议级配对检验）：

  | 集合 | cp 差 | tcp 差 | p_Holm（cp / tcp） |
  |---|---|---|---|
  | AMI | +1.97 | +1.98 | 0.0003 / 0.0002 |
  | Ali | +2.23 | +1.92 | 2.3e-5 / 2.3e-5 |
  | En→Zh | +1.41 | +1.26 | 0.0005 / 0.0003 |

  四项语音延迟上 MIST 都显著更早，符号翻转检验 p ≤ 6e-5。
- 正式 Table 1 由主线生成。参数：`--extra "cascade=Cascade E9:ami=jtest_ami_cascade_blind_s03000_rel__t1c,jtest_ami_cascade_blind_s03000_rel__t1c_tts2;ali=...;en=..."`；延迟单元已由 `table1_lat.sh` 写入 `joint/table1_v14/lat/`。

### 3.3 计数

| 集合 | 窗口 | 片段 | unk 片段 | 块 | 片段提交前已说出的块 | 归属不确定且标签不同的块 | 无片段的块 | Talker 流（每种子） | 丢弃 |
|---|---|---|---|---|---|---|---|---|---|
| AMI | 3973 | 12036 | 1 | 12039 | 8 | 23（0.19%） | 0 | 5619 | 0 |
| Ali | 4065 | 31376 | 18 | 31426 | 44 | 395（1.26%） | 18 | 7158 | 0 |
| En→Zh | 5126 | 20651 | 0 | 20666 | 9 | 80（0.39%） | 3 | 7192 | 0 |

- **归属不确定的块**：块所属片段的键（模型的说话人和起点）在同一窗口里重复出现，而且几个候选片段在同一次读取中结束，就无法判定这个块属于哪个片段。这类块取第一个候选片段的标签，只影响它用谁的音色合成，不影响文本打分。
- **`unk` 流**：AMI 有 1 条，Ali 有 2 条，用 prep 自带的回退音色合成；打分时 `unk` 不参与说话人匹配。

### 3.4 解读（只陈述数字）

- **与 B 比**：同训练数据的"先识别、后归属"级联在文本上全面低于 B（tcpBLEU −1.2 至 −1.5）。差距集中在重叠区：ov −4.3 / −2.4 / −5.3，1spk 只差 −0.9 至 +0.7。并发说话人召回 CR 也明显更低（文本 0.54 / 0.65 / 0.64，B 0.83 / 0.86 / 0.85）。
- **语音 tcpASR-BLEU**：AMI 比 B 高 0.31（ov 仍低 1.3，1spk 高 0.8），Ali 低 0.82，En→Zh 低 0.67。本次没有做 E9 对 B 的配对检验，登记里也没有这一项。
- **延迟**：四项语音延迟都比 B 晚 0.2–0.5 s（LAAL_block +0.39 / +0.27 / +0.26 s）。文本 LAAL 也比 B 晚 0.17–0.36 s。
- **MIST 对 E9**：两族 BLEU 在三个集上都显著更高（Holm 后 p < 0.001），延迟显著更早。

## 4. 运行中的问题（如实记录）

- **CPU 超载**：约 19:30–22:00 布拉格（UTC 约 18:30–21:00），整机 CPU 严重超载（load 约 750–820，共 192 核），来源是另一条线的 E8 X-Voice 服务（未限线程）。这段时间 AliMeeting 解码约比 B 当时慢 3 倍。已报告协调者，协调者在 E8 的 Ali 部分结束后停掉了这些服务，此后恢复正常。这只影响耗时，不影响输出：延迟按 chunk 协议计算，不计墙钟。
- **两个解码叠在同一张卡上**：每卡上限 129 GiB 之下，只要第一个解码不超过 65 GiB，就会放入第二个。实测两个解码合计 127–129 GiB，加 6 GiB 余量后 ≤ 135，没有发生 OOM，也没有触发 headroom 停止。
