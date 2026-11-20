# 鲁棒第十六波：NCWI，噪声条件的权重插值（2026-11-20 15:55 CET，写在任何 b̂、拟合和评测数字之前）

方法与判定见主线方案 4.2（44823eb32），逐字有效。本文件只追加实现说明和结果。

## 实现说明（2026-11-20 15:55 CET，写在任何 b̂、拟合和评测数字之前）

### 1. 噪声估计器 `experiments/mist_w3po_v1/voicecond/wave16_noise_est.py`（CPU，venv simuls2s）
- VAD：silero-vad 6.2.3（pip 包 `silero_vad` 自带的 JIT 权重 `silero_vad.jit`，本机已有，开源、预训练，不训练、不调参）。
- 读音频的方式与 si_thinker_stream 的 `--items-file` 完全相同（soundfile float32、多声道取均值、截到 dur_ms），只取前 16000 个样本（前 1 s；不足 1 s 的条目取全部）。
- 帧：512 个样本（32 ms，silero 在 16 kHz 下的固定窗长），从头切 31 帧，不补零；每条重置 VAD 状态，逐帧送入。语音帧 = 概率 ≥ 0.5（silero 默认阈值），其余是非语音帧。
- 帧能量 = 该帧样本平方的均值。b̂ = 非语音帧能量均值 / 语音帧能量均值。
- 退化情况（写定，不调）：
  - 没有语音帧：语音能量取能量最大的一帧，非语音能量取其余帧的均值；
  - 没有非语音帧：非语音能量取能量最小的一帧，语音能量取其余帧的均值。
  - 两种情况都在输出里打标记并计数。
- 输出 `eval_outputs/mist_w3po_v1/wave16/noise/bhat_<dir>.jsonl`：每条 dev 条目（S1 的 (a)/(b) × 干净/带噪）一行，含 id、b̂、b̂ 的 dB 值、语音帧数、退化标记，以及 si_thinker_stream 提示音频前 1 s 的 sha1（推理时按它找条目）。
- 真实 BVR（辅助指标的参照）：
  - (a) 带噪：`wave1_K2K3K5/data/manifest.jsonl` 的 `bvr_db`（逐句）；
  - (b) 带噪：bench_dev 的 id2 混音 = clean 孪生混音 + `bg_track` 混音（已核对，逐样本残差 ≤ 3.1e-5），所以逐条真实 BVR = 该条切片区间内 bg_track 能量 / clean 孪生混音能量，切片位置与 `wave2_s1_prep.py` 相同（refs_windows 的 start_s、dur_ms）；
  - 干净条目没有背景，真实 BVR 记为 −∞。
- 报告：每个方向在 (a) 带噪、(b) 带噪上 b̂_dB 与真实 BVR_dB 的 Spearman 和 Pearson 相关；以及在全部条目上用 b̂ 区分干净和带噪的 AUC。只作辅助，不参与任何选择。

### 2. 拟合 `wave16_fit.py`（CPU，qwen3omni python）
- **两半：** half = int(sha1(pair).hexdigest(), 16) % 2，pair 是 S1 meta 的 `pair` 字段，即干净/带噪孪生共用的句子 id。4.2 写的是"按 sha1(id)"；这里取孪生共用的 id，是为了让同一句话（同一参考）的干净和带噪版本落在同一半，不在一半拟合、另一半评估时泄漏。这只比逐条 id 更严，不改变方法。
- **分箱：** 对拟合半，在该方向的 (a) 条目（干净 + 带噪，约 300 条）上取 b̂ 的 25/50/75 分位（`np.quantile`，线性插值）为边界；箱号 = `searchsorted(edges, b̂, side="right")`，即 0–3。另一半的条目（(a) 和 (b) 都算）按这组边界归箱。
- **f(α)：** 拟合半里，该箱 (a) 条目的句级 text BLEU 均值。
  - 句级 BLEU：sacrebleu `BLEU(tokenize=zh|13a, effective_order=True).sentence_score`，默认 exp 平滑；文本先过 `normalize_for_bleu`，与打分口径相同。hyp 是 Thinker 各块拼起来的全文。
  - f(0)：第五波 3 次底座运行（S1、base_rerun1、base_rerun2）的逐句 BLEU 先平均，再对箱内条目取均值；f(0.5)：第十四波 `eval_unist_wise05`；f(1)：第十四波 `eval_unist`。
- **闭式解：** f(α) = a0 + gα − cα² 过三点：a0 = f(0)，c = 2(2f(0.5) − f(0) − f(1))，g = f(1) − f(0) + c。c > 0 时 α* = clip(g / 2c, 0, 1)；c ≤ 0 时取三个端点里 f 最大的一个，并列时优先 0.5，再 0，再 1。α* 保留 4 位小数，作为适配器的名字。
- 每个方向 2 个拟合半 × 4 个箱，最多 8 个不同的 α*，正好不超过 8 个适配器。
- 输出 `eval_outputs/mist_w3po_v1/wave16/fit/`：`fit.json`（边界、每箱 f 值、a0/g/c、α*、条目数）和 `assign_<dir>.jsonl`（每条：id、提示音频 sha1、所在半、用哪一半拟合的箱号、α*）。
- **自检（`--selftest`，玩具数据）：** 从已知的 a0、g、c 生成三点，要求恢复 a0、g、c 和 α* 到 1e-9；另外覆盖 c ≤ 0、g/2c < 0、g/2c > 1、并列端点这几种情况，以及分箱和两半的划分（每条只属于一半，边界条目归箱正确）。

### 3. 推理路径
- **首选：vLLM 多 LoRA。** 共享文件 `si_thinker_stream.py` 不改。`wave16_si_thinker_ncwi.py` 在进程内包装（同 `wave9_si_thinker_anchor.py` 的做法）：
  - 包装 `vllm.LLM.__init__`，加上 `enable_lora=True, max_loras=8, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"]`。LoRA 只在 LLM 文本层的注意力上（与训练一致），MoE 专家层不挂 LoRA，所以不走 vLLM 的 MoE LoRA。
  - 包装 `vllm.LLM.generate`：按每条提示音频前 1 s 的 sha1 查 `assign_<dir>.jsonl` 得到 α*，传每条请求自己的 `LoRARequest`。α* = 0 时不传 LoRA，即原始底座。查不到的提示直接报错退出。
  - `--thinker` 是原始 Qwen3-Omni（底座），其余参数与第十四波的 Thinker 阶段相同（`--windows 1 --policy la2 --gpu-mem 0.85`）。
  - 适配器 `wave16_make_lora.py`（CPU）：从 `wave14/run_unist/adapters_final.pt` 转成 PEFT 格式。键为 `base_model.model.thinker.model.<层>.lora_A/B.weight`，A、B 原样（fp32，vLLM 载入时转成模型 dtype），`r` = 16，`lora_alpha` = 32·α*，所以缩放 = 2α* = α* × 训练时的缩放（α* = 0.5 即 unist_wise05 的 alpha 16）。每个 α* 一个目录，A、B 都相同，只有 `lora_alpha` 不同。
  - 转换自检（CPU）：拿第五波 `run_erm/adapters_final.pt` 转出的 α = 0.5 适配器，在 4 个层上算 (lora_alpha / r)·B·A，与 `wave5/M/merged_erm_wise05` 减底座的权重差比较，相对误差须 < 1%（bf16 舍入）。第十四波合并完成后，链条里对 unist 的适配器和 `merged_unist_wise05` 再做一次同样的比较。
  - **一致性自检（GPU，与 NCWI 并行）：** 把所有条目的 α 都设为 0.5（`--const-alpha 0.5`），在全部 dev 上跑 LA2，与第十四波 `eval_unist_wise05` 的 Thinker 输出逐条比。参照第五波 3 次底座运行两两的一致比例（从已有文件算出，不是新结果）：

    | 方向 | 全文相同的最小比例 | 块（文本 + 时刻）完全相同的最小比例 |
    |---|---|---|
    | en2zh | 0.9333 | 0.9035 |
    | zh2en | 0.9453 | 0.9095 |

    两个方向的两个比例都不低于上表，就算一致，用 LoRA 路径的 NCWI 输出。
- **退路：每个 α* 合并一个模型。** 以下任一情况就整体（两个方向一起）改用合并路径，LoRA 输出不用于判定，只在结果里记录：LoRA 引擎启动或生成报错；或一致性自检任一方向、任一比例低于上表。
  - 合并：α* ∈ {0, 0.5, 1} 直接用原始底座、第十四波的 `merged_unist_wise05`、`merged_unist`；其他 α* 用 `wave5_m_merge.py`（适配器硬链接，json 的 alpha 写成 32·α*）合并出新副本。
  - 按 α* 把条目分组，每组用对应的模型在该组的条目子集上跑 LA2，再按原顺序拼回一个 blocks 文件。LA2 对每条条目独立，所以与一个进程里按请求选适配器数值上等价，只差 vLLM 本身的不确定性。
  - 交叉拟合下每个方向最多有 8 个不同的 α*（任务说明里写的"每个方向最多 4 个"是按不交叉拟合估的）；有几个就合并几个。

### 4. 评测与打分
- `wave16_eval.sh`（`wave14_eval.sh` 的副本）、`wave16_orch.sh`：
  - 卡 0：NCWI en2zh，卡 1：NCWI zh2en，卡 2：常数 0.5 自检 en2zh，卡 3：常数 0.5 自检 zh2en，四条并行；
  - Talker：每个方向 4 个分片（种子 20261021，`--num-shards 4`），en2zh 分片 0–1 在卡 0、2–3 在卡 2，zh2en 分片 0–1 在卡 1、2–3 在卡 3；
  - 只对语音做 ASR：en2zh 在卡 0，zh2en 在卡 1。
  - 输出根目录 `eval_outputs/mist_w3po_v1/wave16/eval_ncwi/`，结构同 wave2_S1。
- `wave16_score.py`（`wave14_score.py` 的副本）：系统为 base（3 次底座运行）、erm_wise05、unist_wise05、ncwi，另带 unist（α = 1，只列数）。交叉拟合两半的输出已拼成一份，判定在合并后的整份 dev 上做。
  - **主判定（4.2，对 unist_wise05）：**
    1. 不伤：方向 × {a, b} × {干净, 带噪} × {text, ASR-BLEU}，不能有 ncwi − unist_wise05 的点估计 < −0.3 且 CI 上界 < 0 的单元；
    2. 增益：方向 × {a, b, b_no_crosstalk} × {干净, 带噪} 中，至少一个单元 text 的 ncwi − unist_wise05 CI 下界 > 0，且点估计大于该单元的底座噪声底（3 次底座两两差的最大值）。
    - 两条都过为 PASS，否则 FAIL。
  - CI 是配对 bootstrap 95%（2000 次，种子 0），单元为句子 (a) 或单说话人流 (b)，与第五、十四波相同。
  - 辅助（不设门）：ncwi 相对底座的统一规则；每个箱的 α*；b̂ 与真实 BVR 的相关；D1、D2 LAAL；按两半分别列出的 ncwi − unist_wise05。
  - 打分自检：复算的 erm_wise05 相对底座须与 `wave5/M/summary_erm_wise05.json` 完全一致；unist_wise05 相对底座、相对 erm_wise05 的数值须与第十四波 `summary.json` 完全一致。

### 5. 卡与链条
- 等待脚本 `wave16_wait.sh`（setsid nohup）：等第十四波的 `CHAIN_DONE` 出现，并且 `logs/mist_w3po_v1/wave14/w14_workers.pids` 里没有活着的进程；出现 `CHAIN_FAILED`，或缺 `eval_unist`/`eval_unist_wise05` 的 `EVAL_DONE`，就写 `WAIT_FAILED` 退出，不占卡。
- 之后自动：适配器转换与自检 → 拟合 → 起自己的 watcher（`wave16_noplaceholder.sh`，PIDS 和 STOP 文件独立）→ 卡 0–3 上的评测（含退路）→ touch STOP，把卡交还 guard → 打分 → `DONE`。
- 卡 4–7 不碰。不碰 guard 2726951、gpu_watch 52045、sync 1222453。只按 PID 停进程。

### 实现补记（2026-11-20 16:04 CET，仍在任何端点和评测数字之前；上面各条不改，以此补充）
1. **合并副本带 bf16 舍入。** 用第五波 `run_erm` 的适配器做转换自检时发现：合并副本是 bf16(W0 + δ)，它相对精确 δ 的误差有 7–15%（Frobenius，8 个投影）。原因是 δ 比 W0 小得多，bf16 存 W0 + δ 时的舍入相对 δ 不小。所以 α = 0.5 的 LoRA（不合并，δ 不经舍入）和合并的 unist_wise05 不是逐位相同的模型。
   - 转换自检因此改为：用 `wave5_m_merge.py` 的同一算法，把适配器重新合并成 bf16(W0 + δ)，必须与合并副本逐元素相同的比例 ≥ 99.9%。对第五波 erm_wise05 的 8 个投影都是 100%，说明键名、A/B 方向和缩放都对。上面第 3 节写的"相对误差 < 1%"作废。
   - LoRA 路径能不能用，仍按第 3 节的一致性自检判（与第五波底座重跑的一致比例比较）；舍入差异如果大到超出 vLLM 不确定性，就按写好的规则退到合并路径。退路的各 α* 副本与端点副本用同一种合并，舍入方式一致。
2. **一致性门槛**由 `wave16_tools.py agree` 每次从第五波 3 次底座文件现算，与第 3 节表中的数相同（en2zh 0.9333 / 0.9035，zh2en 0.9453 / 0.9095）。
3. **合并退路的分卡：** 各 (方向, α*) 组按轮转分到卡 0–3，每张卡上顺序跑。
4. 用 DRY_RUN 检查 `wave16_eval.sh` 时，脚本建了空的 `wave16/eval_ncwi`、`eval_lora05`、`eval_ncwi_merged` 目录（只有 data 软链接），不影响后面运行。
5. 拟合和打分的代码路径已用旧系统冒烟（输出在会话的临时目录，不进仓库，不涉及第十四波的任何数字）：
   - 拟合：把 α = 0.5、1 两个端点临时换成第五波 base_rerun1/2；
   - 打分：把 ncwi 换成 erm_wise05、unist_wise05 换成 erm。erm_wise05 相对底座的 180 个数值和门与第五波 summary 完全一致；zh2en/a/clean 的 erm_wise05 − erm 为 +3.97，与第十三波记录相同。
   - 合并退路的分组和拼装用 S1 的输出冒烟：拆开再拼回，与原文件逐条相同。

## 噪声估计结果（辅助，与端点无关；`eval_outputs/mist_w3po_v1/wave16/noise/report.json`）
| | en2zh | zh2en |
|---|---|---|
| 条目数 | 1274 | 2012 |
| 前 1 s 没有语音帧（退化情况 1） | 240 | 295 |
| 前 1 s 全是语音帧（退化情况 2） | 21 | 61 |
| (a) 带噪：b̂_dB 与真实 BVR_dB 的 Spearman / Pearson（n = 300） | 0.324 / 0.325 | 0.452 / 0.436 |
| (b) 带噪：同上（n = 337 / 706） | 0.341 / 0.336 | 0.304 / 0.341 |
| 用 b̂ 区分干净和带噪的 AUC：(a) / (b) / 全部 | 0.706 / 0.676 / 0.689 | 0.694 / 0.658 / 0.665 |

- 结论：只看前 1 s 的 b̂ 与真实 BVR 只有中等偏弱的相关（0.30–0.45），区分干净和带噪的 AUC 约 0.66–0.71。前 1 s 里约 19%（en2zh）和 15%（zh2en）的条目 VAD 找不到语音，这时用的是写定的退化规则。这会限制 NCWI 能拿到的增益：箱与真实噪声水平只是部分对应。
- 真实 BVR：(a) 用 manifest 的 `bvr_db`；(b) 用 bg_track 与（混音 − bg_track）的能量比。已核对 id2 混音 = limiter × clean 孪生 + bg_track（zh2en 的 dbz018 limiter 为 0.9569），最大残差 8.1e-5。

## 进度
- 2026-11-20 16:05 CET 启动等待脚本 `wave16_wait.sh`（setsid nohup，pid 2380461，日志 `logs/mist_w3po_v1/wave16/wait.log`）。它等第十四波的 `CHAIN_DONE` 和第十四波进程退出，之后自动拟合、在卡 0–3 上评测、打分。

### 实现说明更正 2：提速（2026-11-20 16:12 CET，按主会话 16:15 的要求；仍在任何端点和评测数字之前）
- **可以叠到卡 4–5。** 每到需要卡的阶段，`wave16_tools.py extracards` 用 nvidia-smi 当场判断卡 4、5 能不能用：
  - 卡上不能有训练进程（torchrun、`*_train.py` 之类）。guard 的无检查点占位（`scripts/gpu_load/`）不算训练，它占的显存算作可用，启动前照常按 PID 停掉；
  - Talker 分片要求可用显存 ≥ 60 GB；合并退路的 Thinker（`--gpu-mem 0.85`，约 122 GB）要求 ≥ 125 GB；
  - 卡 6–7（第十七波训练）永远不用。`wave16_eval.sh` 只接受卡 0–5。每次判断的结果记在 `logs/mist_w3po_v1/wave16/extracards.log`。
- **Talker 和 ASR：**
  - zh2en 的 4 个分片放在卡 1、3 加上可用的额外卡（卡 4、5 都可用时每卡一个），en2zh 的 4 个分片放在卡 0、2；
  - 每个方向自己的 4 个分片一跑完，就在该方向的卡上（en2zh 卡 0，zh2en 卡 1）开始 ASR，不再等另一个方向。
  - `--num-shards` 仍是 4，种子不变，所以输出与原来的分法相同。
- **ASR 不拆分：** Whisper（zh2en 的 ASR）在整份文件内按时长排序后组批，拆开会改变批次组成，进而可能改变转写。为了与 unist_wise05 的 ASR 口径完全一致，不拆。按第五波的耗时（Thinker 约 4 分钟，Talker 约 8 分钟，zh2en ASR 约 13 分钟），LoRA 路径全程约 25–30 分钟，在 45 分钟以内。
- **合并退路：** Thinker 的通道数为卡 0–3 加上当时 ≥ 125 GB 可用的卡 4、5，最多 6 个通道。
- **第一阶段**（4 个 LoRA Thinker）仍在卡 0–3，每个约 4 分钟，不再拆。
- **怎么换的：** 等待脚本 `wave16_wait.sh`（pid 2380461）没有写死卡号，所以没有动它。它还在等第十四波，尚未调用编排脚本。`wave16_orch.sh` 和 `wave16_eval.sh` 都是先写成新文件、检查 `bash -n`，再用 mv 替换；替换时两者都不在运行，旧版本在 git 的 f6cc54b36 里。

## 变更 B（主会话，2026-11-20 18:08 CET，写在任何 ncwi_erm 数字之前；只追加）
- **起因：** 第十四波 unist_wise05 两条主判定都没过，12 个单元相对 erm_wise05 全部变差。UniST pilot 的源音频是 BiCodec 重建的，译文是合成的。所以以 UniST 为端点的 NCWI 主要在检验一个失败的底座，不能回答"按噪声条件选插值系数是否有效"。原定的 UniST 端点照常跑完并报告，不删除。
- **新增一组 ncwi_erm：** 方法与 4.2 逐字相同，只把端点换成 CoVoST 的 ERM，三个端点都有现成的 dev 输出：
  - α = 0：base（第五波 3 次底座运行）；
  - α = 0.5：erm_wise05（`wave5/M/eval_erm_wise05`）；
  - α = 1：erm（`wave5/M/eval_erm`）。
  - 推理用 `wave5/M/run_erm` 的 LoRA 按 α* 缩放。
- **噪声估计、分箱和交叉拟合：** 与原定相同，直接复用已算好的 b̂ 和箱边界。
- **主判定：** 与 4.2 相同，参照改为 erm_wise05。
  - 没有任何 (a)/(b) 单元的 text BLEU 或 ASR-BLEU 点估计 < −0.3 且 CI 上界 < 0；
  - 至少一个单元 text 增益的 CI 下界 > 0，且超过噪声底。

### 变更 B 的实现说明（2026-11-20 18:09 CET，写在任何 ncwi_erm 数字之前）
- **UniST 组的 LoRA 路径已退到合并路径（18:05 CET）。** 原因不是一致性不够，而是引擎起不来：vLLM 实际加载的是整个 `Qwen3OmniMoeForConditionalGeneration`（vllm-omni 的外壳），报 "does not support LoRA yet"。只有其中 Thinker 子类声明了 LoRA 支持。4 个 LoRA Thinker 都在约 1 分钟内退出，`agree.json` 记为未通过，按第 3 节写好的规则整体改走合并路径。所以本波实际用的都是"每个 α* 合并一个模型、按箱分组跑"，数值上与端点副本同一种合并。
- **ncwi_erm 的文件都是副本，不改正在运行的 UniST 组的脚本：**
  - `wave16e_fit.py`：调用 `wave16_fit.py`，只把端点换成 `wave5/M/eval_erm_wise05`（α = 0.5）和 `wave5/M/eval_erm`（α = 1）；α = 0 仍是 3 次底座。b̂ 直接读 `wave16/noise`，两半划分相同，所以箱边界相同；脚本断言边界和各箱条目数与 `wave16/fit/fit.json` 完全一致。
  - 适配器：`wave16_make_lora.py`，输入 `wave5/M/run_erm`，对 `wave5/M/merged_erm_wise05` 做重新合并自检。
  - `wave16e_tools.py`、`wave16e_eval.sh`、`wave16e_orch.sh`：分别由 `wave16_tools.py`、`wave16_eval.sh`、`wave16_orch.sh` v2 用 sed 生成。根目录是 `wave16/erm`，日志在 `logs/mist_w3po_v1/wave16/erm`，PIDS 和 STOP 文件独立；一致性参照改为 `eval_erm_wise05`；合并退路的 α = 0.5、1 用 `wave5/M/merged_erm_wise05`、`merged_erm`。
  - LoRA 路径照写好的流程先试一次。预计同样起不来，约 1 分钟后退到合并路径；保留这一步，是为了与 UniST 组的流程完全相同。
  - `wave16e_score.py`：系统为 base（3 次）、erm_wise05、erm、ncwi，主判定以 erm_wise05 为参照，规则与 4.2 相同。
    - 自检：erm_wise05 相对底座须与 `summary_erm_wise05.json` 一致；凡是 `wave5/M/summary.json` 里也有的数值都须一致。
    - 用旧系统冒烟过（临时把 ncwi 换成 fishr）：两项自检都通过，分别是 180 个值加门，以及 312 个值。
- **卡：**
  - 等 UniST 组的 GPU 部分结束（它的 watcher STOP 出现，且它的进程都已退出）才上卡。Thinker 先用卡 1、2（NCWI），一致性自检用卡 0、3；合并退路的通道和 Talker 用卡 0–3。
  - 卡 4–5：`extracards` 在 `combo/ctrl_pool_base/DONE` 出现前一律不给（按主会话 18:15 的更正，这条同时加到了 UniST 组的 `wave16_tools.py`）；出现后，仍按 ≥ 60 / ≥ 125 GB 的规则判断。卡 6–7 不用。
- 等待脚本 `wave16e_wait.sh`（setsid nohup）先在 CPU 上做拟合和适配器转换，再等卡。

## 结果一：UniST 端点（原定组，ncwi 对 unist_wise05；2026-11-20 18:33 CET）
- **运行：**
  - 18:02 开始，18:30 卡 0–3 交还 guard，18:32 打分完成，GPU 部分共 28 分钟。
  - 走的是合并路径：LoRA 引擎起不来，见变更 B 的实现说明。
  - 合并出 α* = 0.2616、0.8009 两个副本（各约 1 分钟）；端点 α* 用原始底座、`merged_unist`。
  - 打分自检：erm_wise05 相对底座与第五波一致（180 个值加门）；516 个值与第十四波 `summary.json` 一致。
- **每箱 α\*（箱 0–3 按 b̂ 从低到高；"拟合半 h"给另一半用）：**

  | 方向 | 拟合半 0 | 拟合半 1 |
  |---|---|---|
  | en2zh | 0, 0.2616, 0, 0 | 0, 0, 0, 0 |
  | zh2en | 0, 0, 0, 0 | 0.8009, 0, 1.0, 0 |

  - 条目数：en2zh α = 0 有 1073 条、0.2616 有 201 条；zh2en α = 0 有 1560 条、0.8009 有 231 条、1.0 有 221 条。
  - 也就是说，16 个箱里 12 个选了 α = 0（原始底座）。原因是 unist_wise05 本身比底座差（第十四波）。
- **主判定（4.2，对 unist_wise05）：PASS。** 没有伤害单元；有 5 个单元的 text 增益显著且超过噪声底。

  | 单元 | text：unist_wise05 → ncwi，差 [95% CI]，噪声底 | ASR-BLEU 差 [CI] | D2 LAAL 差 (s) |
  |---|---|---|---|
  | en2zh/a/clean | 34.67 → 37.86，**+3.19 [+0.85, +5.72]**，0.31 | +0.25 [−1.37, +1.89] | +0.05 |
  | en2zh/a/noisy | 34.47 → 37.72，**+3.25 [+1.02, +5.74]**，0.30 | +0.05 [−1.63, +1.75] | +0.18 |
  | en2zh/b/clean | 37.03 → 36.82，−0.21 [−1.65, +1.37] | −0.84 [−2.33, +0.84] | +0.00 |
  | en2zh/b/noisy | 35.38 → 35.03，−0.35 [−2.10, +1.40] | −0.91 [−2.57, +0.80] | +0.10 |
  | zh2en/a/clean | 28.67 → 31.17，**+2.51 [+0.79, +4.10]**，0.15 | **+2.40 [+0.60, +4.40]** | −0.03 |
  | zh2en/a/noisy | 29.56 → 29.43，−0.12 [−2.34, +1.79] | −0.31 [−2.87, +2.37] | +0.01 |
  | zh2en/b/clean | 25.32 → 27.40，**+2.07 [+0.45, +3.17]**，0.32 | +1.34 [−0.40, +2.70] | −0.00 |
  | zh2en/b/noisy | 25.28 → 25.16，−0.11 [−2.53, +1.88] | +0.60 [−1.79, +1.97] | +0.07 |
  | zh2en/b_no_crosstalk/clean | **+1.87 [+0.09, +3.42]**，0.49 | +1.82 [+0.00, +3.69] | |
  | en2zh/b_no_crosstalk/noisy（不在不伤的范围内） | −2.13 [−4.72, +0.49] | −2.80 [−5.22, −0.37] | +0.12 |

- **怎么读这个 PASS（重要）：**
  1. 增益几乎全部来自"退回底座"：大多数箱选了 α = 0，而 unist_wise05 比底座差。所以这是在一个失败的端点上避开它，不是噪声条件插值的证据。增益最大的单元（en2zh/a、zh2en/a 干净）正是第十四波 unist 掉分最多的地方。
  2. 相对底座（辅助，统一规则）不过：zh2en/a/clean text −1.98 [−3.44, −0.66]，另有 7 个单元点估计 < −0.3（CI 跨 0）。ncwi 比只用底座还差，说明箱和 α* 的选择本身没带来额外收益。
  3. 拟合半 1 的 en2zh α* 全是 0，en2zh/a_half1 的增益 +4.69 / +4.84 比 half0 的 +1.54 / +1.47 大，也是同一个原因。
  4. en2zh/b_no_crosstalk/noisy 的 ASR-BLEU −2.80 [−5.22, −0.37] 显著变差。它不在 4.2 不伤判定的范围内（只看 a、b），照实列出。
  5. 带噪单元的 D2 LAAL 慢了 0.07–0.18 s（CI 都在 0 以上）。
- **结论：** 按预登记规则判为 PASS，但不能作为 NCWI 有效的证据，因为端点 unist_wise05 本身失败了。决定性的检验是变更 B 的 ncwi_erm。

### 实现说明更正 3：ncwi_erm 的卡位冲突与续跑（2026-11-20 18:53 CET，写在任何 ncwi_erm 数字之前）
- **出了什么事：**
  - ncwi_erm 第一轮 18:30 开始。LoRA 路径与 UniST 组一样起不来，18:33 改走合并路径，合并 12 个副本，15 个 (方向, α*) 组分到卡 0–5 的 6 个通道上。
  - 18:46 链条停在 "merged part en2zh a0.5926: 0 of 167 rows"。15 组里有 3 组在启动时失败：
    - en2zh a0.5926（卡 3）和 zh2en a0.2304（卡 3）是 CUDA OOM，当时卡上出现了别的进程；
    - en2zh a1.0000（卡 0）是 vLLM 初始化时 `aten::new_empty` 失败，紧跟在同卡上一组之后。
  - 主会话 18:50 指出：uniss-de 的组合臂 C 从 18:47 起在卡 0、3 上跑（到 `combo/C/DONE` 为止）。我 18:42 的 thinker_merged_zh2en_a0.3813 落在卡 0，挤爆了它的 Talker；我这边的 OOM 很可能也是同一个冲突。对组合臂 C 造成的影响，在此照实记录。
- **续跑 1**（`wave16e_resume.sh`，18:48）：只重跑这 3 组，分到卡 0、1、2。它违反了"卡 0、3 不用"的更正，所以 18:50 按 PID 停掉了：
  - 停了续跑脚本 21045 和卡 0 上的 thinker 23568。它的 EngineCore 子进程随之退出，事后核对卡 0 上只剩组合臂 C 的进程。
  - 被停的 en2zh a0.5926 其实已经写完了 167 行（停的时候正在拆引擎）。它跑在不该用的卡上，所以输出改名为 `parts/thinker_en2zh_a0.5926.superseded_v2`，日志改名为 `*.superseded_v2`，不用，重跑。
  - 卡 1、2 上的两组（zh2en a0.2304 pid 31143，en2zh a1.0000 pid 39903）不受影响，继续跑，由续跑 2 接管。
- **启动脚本 v2**（`wave16e_eval.sh`，写新文件再 mv）：停掉卡上的占位后，不再固定等 8 s，而是每张卡最多等 120 s，直到满足以下两条才启动；120 s 内不满足就拒绝启动：
  - 卡上没有 `scripts/gpu_load/` 的 train_mist_workload.py / run_mist_workload.sh 进程（按 argv 字段匹配，不会匹配到检查命令自身）；
  - nvidia-smi 的空闲显存够这一阶段用：Thinker 125000 MiB，Talker / ASR 30000 MiB。
- **续跑 2**（`wave16e_resume2.sh`，18:51 起，setsid nohup）：
  - 只用卡 1、2、4、5。卡 4–5 由 extracards 当场核准：ctrl_pool_base 已结束，卡上只有占位。卡 0、3、6、7 不用。
  - en2zh a0.5926 在卡 4 重跑，并等卡 1、2 上的两组跑完；
  - 然后检查行数、拼装，Talker（zh2en 分片在卡 1、4，en2zh 分片在卡 2、5），ASR（zh2en 卡 1，en2zh 卡 2），打分；
  - 有自己的 watcher STOP 文件，结束时 touch。
- 第一轮的 `ORCH_FAILED` / `CHAIN16E_FAILED` 改名为 `*.superseded_v1`。

### 实现说明更正 4：续跑 3（2026-11-20 18:59 CET，写在任何 ncwi_erm 数字之前）
- **续跑 2 的进展：** Thinker 15 组全部完成，en2zh a0.5926 在卡 4 重跑。行数检查和拼装都通过，zh2en 的 Talker（卡 1、4）随后正常开始。
- **续跑 2 的问题：** en2zh Talker 的分片 0 没能在卡 2 上启动。guard 在启动脚本停掉占位后几秒内又起了新的占位（103344、106965），120 s 内卡 2 只剩约 14 GB 空闲，启动脚本按 v2 的规则拒绝启动，于是续跑 2 的 en2zh 一路停了，zh2en 一路照常继续。
- **启动脚本 v3**（写新文件再 mv）：在最多 120 s 的等待里，每轮都按 PID 停掉 guard 新起的 gpu_load 占位（argv 重新核对，受保护的 PID 不碰）。
- **续跑 3**（`wave16e_resume3.sh`，18:57 起）：只跑 en2zh 一路，Talker 分片在卡 2、5，ASR 在卡 2。然后等续跑 2 退出：
  - 续跑 2 如果已经看到两份 ASR 并完成打分（DONE），续跑 3 就不再做什么；
  - 否则由续跑 3 写 EVAL_DONE 和 NCWI_ROOT，再打分。
- 卡 0、3（组合臂 C）、6、7 都没有用。
