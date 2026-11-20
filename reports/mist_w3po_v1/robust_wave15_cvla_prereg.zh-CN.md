# 鲁棒第十五波：CV-LA 跨视角一致提交 预登记

方法与判定见主线方案 4.1（44823eb32），逐字有效：`reports/mist_w3po_v1/robust_mainline_plan_20261120.zh-CN.md`。本文件只追加实现说明和结果。

## 实现说明（执行者追加，2026-11-20 15:55 CET，写于本波任何数字之前）

### 增强模型的选择（写在任何数字之前）
- **选用 DeepFilterNet3**（Schröter et al., Interspeech 2023；MIT 许可；`deepfilternet` 0.5.6 发布的官方预训练权重 `model_120.ckpt.best`）。
  - 权重本来就在本机（`/root/.cache/DeepFilterNet/DeepFilterNet3`，X-Translator 基线用过），原样复制到 `pretrained_models/DeepFilterNet3/`：
    - `checkpoints/model_120.ckpt.best` sha256 `23b92884…c4c171e6003`；
    - `config.ini` sha256 `415eb925…b4290`。
  - 代码用已有的 `venvs/xtranslator`（装有 deepfilternet 0.5.6），没有安装任何包；按 `serve_xvoice.py` 的做法，给已从 torchaudio 删除的 `torchaudio.backend.common.AudioMetaData` 一个空壳类，`enhance()` 用不到它。
- **理由：**
  1. 它是专门做**语音降噪**的模型（DNS 挑战数据训练），目标正是保留语音、压掉背景，与本波的噪声类型（音乐、人群、音效）对应；
  2. 它是**实时因果**模型：48 kHz，帧长 20 ms、帧移 10 ms，conv / DF 前瞻各 2 帧，算法延迟约 30 ms，可以只看已读到的音频；
  3. 开源、权重公开、本机已有，不需要训练或调参；
  4. 其余候选不合适：torchaudio 的 HDemucs 是音乐分离模型、双向、长上下文（第一、三波用过，替换输入伤干净），ConvTasNet-Libri2Mix 是说话人分离，SQUIM 只做质量评估；`pretrained_models/` 里没有别的增强模型。
- **用法（一个设置，全部用库默认值）：** `init_df()` 默认配置，`enhance(pad=True)`，不设 `atten_lim_db`（不限制衰减）。16 kHz → 48 kHz（`df.io.resample`，默认 sinc_fast）→ 增强 → 16 kHz，截齐或补零到原长度。

### 因果性
- 第 t 次读（t = 1000, 2000, … ms，且 t < 句长）时，v1 = DF3(x[: t])：**只把已读到的前缀**送进模型，每次读都从头重算（模型状态每次复位），所以 v1 的任何一个采样点都不依赖 t 之后的音频。重采样滤波也只作用在这段前缀上。
- 末次读（t = 句长）按规则直接提交 v0 的完整草稿，不需要 v1，所以不算。
- 视角预先离线算好（CPU，每个进程 1 线程），存成每条一个 float32 `.npy`，依次拼接各次读的 v1 前缀，索引里记录输入音频的 sha1；Thinker 读入时核对 sha1 与长度（= t × 16 个采样点）。这样做与在线逐读计算完全等价，只是把计算挪到了推理之前。
- **每个读窗的计算时间**（辅助指标）：记录每次读对整段前缀重算的 CPU 单线程时间；另报实时率（处理时间 / 前缀长度）。因为模型本身是流式的，部署时每次只需处理新读到的 1 s，所以实时率乘以 1 s 才是部署时每个读窗的代价。

### CV-LA 实现
- **文件：** `experiments/mist_w3po_v1/voicecond/wave15_si_thinker_cvla.py`。共享的 `si_thinker_stream.py` 不改。
  - 这是 la2 w1 路径的副本：`units`、`join`、`QWEN3`、`THINKER_EOS` 从共享文件导入；条目读取、提示词、`LLM(...)` 与 `SamplingParams(...)` 的参数、每轮的草稿拼接、末次读、输出的 blocks 后处理都逐行照抄；
  - 不带 snap、hygiene、ban-wait 等任何选项，种子 20261021，`--windows 1`。
- **每轮：** 先放所有在读条目的 v0 提示，后放其中非末次读条目的 v1 提示（相同的 head + 已提交文本，只换音频），一次 `llm.generate` 同批算完。
  - d0 = v0 的草稿（与 LA2 相同的拼接方式），d1 = v1 的草稿（同样的拼接方式）；
  - agreed = LCP(prev, d0)，prev 仍然只记 v0 的草稿，与 LA2 相同；
  - cv = min(agreed, LCP(d0, d1))，即 LCP(A_t, d1)；提交 d0[n : cv]（cv > n 时）。
- 每次读的记录里多存 `draft_v1`、`la2_new`（LA2 在同一状态下会提交几个单元）、`cv_new`（实际提交几个），供提交统计。
- **代码内断言：** d1 = d0 时 cv_new 必须等于 la2_new。

### 自检（v1 = v0，必须在 cvla_base 之前通过）
- `--view1 same`：v1 直接用 v0 的同一段音频，其余不变，两个方向都跑完整 dev。
- **参照**（第五波 base_rerun 与 S1 底座三次运行两两之间、逐条 blocks 完全一致的比例，已有文件算出，不是本波结果）：英→中 0.9035 / 0.9372 / 0.9239，中→英 0.9299 / 0.9120 / 0.9095。
- **通过条件（写在跑之前）：** 每个方向，same 运行与三次底座运行的 blocks 逐条一致比例的均值 ≥ 该方向三对底座运行之间的最小值（英→中 0.9035，中→英 0.9095）。另报同一运行内 d1 ≠ d0 的非末次读比例（只有 vLLM 自身的批内差异）。不通过就停，不跑 cvla_base。

### 评测链与打分
- **脚本：** `wave15_enh_views.py`（视角）、`wave15_eval.sh`、`wave15_eval_orch.sh`、`wave15_noplaceholder.sh`、`wave15_selfcheck.py`、`wave15_score.py`、`wave15_commit.py`、`wave15_chain_g1.sh`、`wave15_wait_g2.sh`。
  - 前四个照第十一 / 十四波的同名脚本复制，路径、日志、PIDS、STOP 都独立；
  - 链与第五波相同：la2 w1 Thinker → T1c export_1500 Talker（4 个分片，种子 20261021）→ 只含语音的 ASR。
- **输出与日志：** `eval_outputs/mist_w3po_v1/wave15/`，`logs/mist_w3po_v1/wave15/`。
- **卡：** 只用卡 4、5（英→中在 4，中→英在 5，Talker 分片也各在本方向的卡上）。
  - 用自己的 noplaceholder watcher，两组各有一个 STOP 文件（`w15_noplaceholder_g1.stop` / `_g2.stop`）；
  - 一组跑完立即 touch，把卡还给 guard。
- **组 1：** cvla_base（底座 Thinker）对 la2_base。
  - text BLEU 对三次底座运行的均值，ASR-BLEU 对 S1；
  - 噪声底 = 三次底座运行两两差的最大值；
  - 不退化看 方向 × {a, b} × {干净, 带噪} × {text, ASR}，提升看 方向 × {a, b, b_no_crosstalk} 的带噪 text（第五波口径）。
- **组 2：** cvla_unist（merged_unist_wise05 Thinker）对 unist_wise05。
  - 对照是第十四波自己的 LA2 运行，text 和 ASR 都对它；
  - 噪声底沿用三次底座运行的噪声底（第十一波规则 2 的口径）；
  - 不退化与提升单元同组 1。
  - `wave15_wait_g2.sh`（setsid nohup）先等 `wave14/merged_unist_wise05` 合并完成（index 文件存在，且合并日志里有 "adapted"），再起 watcher 和评测链；
  - 打分再等 `wave14/eval_unist_wise05/EVAL_DONE`。
- **打分自检：** `wave15_score.py` 照 `wave14_score.py`，复算 erm_wise05 相对底座的每个数字和规则 1 判定，必须与 `wave5/M/summary_erm_wise05.json` 完全一致，否则脚本停下。
- **提交统计：** 照 `wave8_aul_commit.py` 的口径。非末次块里，参考译文不支持的单元比例（1 − LCS(提交, 参考) / |提交|），以及非末次提交占比；配对 bootstrap 给出 cvla − 对照的差。另报 v1 截短提交的读次比例。
