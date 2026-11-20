# 鲁棒第十二波（诊断）：ERM 的干净中→英掉分是不是过拟合（2026-11-20 12:40 CET，写在任何数字之前）

## 事实（训练数据和日志，不是评测数字）
- 第五波训练集（CoVoST 2 train）：zh2en 只有 7004 条、22 个说话人；en2zh 118983 条、9157 个说话人。
- 每步每个方向 32 条，共 2300 步，所以每个方向各看了 73600 条：zh2en 约 10.5 个 epoch，en2zh 约 0.62 个 epoch。
- ERM 训练 CE（rank 0，每 100 步平均）：
  - zh2en 四个环境：1.04–1.09（第 0–100 步）→ 0.73–0.77（第 1000–1100 步）→ 0.53–0.60（第 2200–2300 步）；
  - en2zh 四个环境：0.84–0.90 → 0.77–0.87 → 0.78–0.83。
- 在 zh2en 内部，干净和带噪环境的 CE 下降幅度相同。
- 干净数据占训练的 1/4（每个方向 4 个环境里有一个是干净），所以不是"只用带噪数据训练"。

## 假设
- zh2en 干净掉分（ERM α = 1 时 −3.63）主要来自对 7004 条、22 个说话人的 zh2en 训练集多 epoch 过拟合，而不是带噪训练遗忘了干净输入。
- 预测：zh2en (a) 干净的 text BLEU 相对底座的差，随检查点步数（约 2.3 / 4.6 / 6.9 / 9.1 / 10.5 个 epoch）单调变差；en2zh (a) 干净不出现这种趋势。

## 方法（固定）
- 检查点：ERM 的 resume_00500、01000、01500、02000 和 final，都用 α = 1 合并（与 run_erm 相同的合并脚本）。不做 WiSE，不训练任何新东西。
- 评测：只跑 Thinker（si_thinker_stream la2 w1），单元为 (a)，两个方向 × 干净/带噪，text BLEU。打分用第五波的 bootstrap 和 3 次底座运行。
- 卡 0–1。

## 判定（只用于诊断，不改主系统）
- 支持过拟合：zh2en (a) 干净的 text 差从 500 步到 final 单调下降（允许一处 ≤ 0.3 的回升），并且 final 与 500 步之差的 bootstrap CI 上界 < 0。
- 若 500 步时 zh2en 干净就已显著掉分：记为"早期就有的偏移（风格或领域）"，不支持纯过拟合。
- en2zh 干净作为对照照列。
- 辅助：zh2en 带噪随步数的曲线，看带噪增益是否早于干净掉分出现。

## 实现说明（2026-11-20 13:05 CET，写在任何数字之前）
- **适配器提取**：resume_XXXXX.pt 是 `{step, adapters, opt, ema, ema_n}`，只取 `adapters`（384 个 fp32 张量，键和形状与 adapters_final.pt 相同），存为 `eval_outputs/mist_w3po_v1/wave12/ckpt_XXXXX/adapters.pt`，并按 adapters_final.json 的格式写 `adapters.json`（rank 16，alpha 32，即 α = 1，scale = 2.0；step 取 resume 里的 step）。脚本 `experiments/mist_w3po_v1/voicecond/wave12_extract.py`（CPU）。
- **合并**：原样用 `wave5_m_merge.py`，输出 `wave12/merged_ckpt_XXXXX`。合并日志的 scale 应为 2.0。
- **final 复用**：`wave5/M/eval_erm` 的 Thinker 输出由 `wave5/M/merged_erm` 产生（合并日志 scale 2.0，即 α = 1；vLLM 日志 model = merged_erm）。直接复用其 (a) 单元的 blocks，不重跑。
- **评测**：`wave12_eval.sh` / `wave12_eval_orch.sh`（第十一波的副本，只保留 merge 和 thinker 两个阶段）。Thinker 命令与第五波相同（si_thinker_stream --windows 1 --policy la2 --gpu-mem 0.85，seed 默认 20261021），只是 items 文件换成只含 (a) 单元的子集 `wave12/data/items_<dir>_a.jsonl`（从 wave2_S1 的 items 中按 meta 的 set = a 抽取，每个方向 600 条 = 300 干净 + 300 带噪）。卡 0 跑 en2zh、卡 1 跑 zh2en，四个检查点依次跑。占位进程由 `wave12_noplaceholder.sh`（独立的 PIDS 和 STOP 文件）停。
  - 已知差异：final 的输出来自完整 items 文件（a + b 一起跑），检查点只跑 (a)。vLLM 批次组成不同，可能带来与底座重跑同量级的逐条差异（底座三次运行在这些单元的最大两两差为 0.15–0.52）。
- **打分**：`wave12_score.py`，沿用 `wave11_score.py` / `wave5_m_score.py` 的 cell()：normalize_for_bleu，sacrebleu（zh / 13a），单元 = 句子（set a 的 stream），bootstrap 2000 次，seed 0；text BLEU 相对 3 次底座运行（wave2_S1、base_rerun1、base_rerun2）的均值。另算 final − ckpt_00500 的配对 bootstrap CI。
  - 自检：复算 erm（final）的 (a) 单元四个格子的 erm − base_mean，必须与 `wave5/M/summary.json` 一致（点估计和 CI 到小数点后 3 位），否则不出结果。
- **判定的程序化写法**（按预登记，写在数字之前）：
  - 曲线 = zh2en (a) 干净 text 差（相对底座均值），顺序为 500、1000、1500、2000、final。"单调下降（允许一处 ≤ 0.3 的回升）"= 相邻 4 步里至多一步上升，且这一步上升 ≤ 0.3。
  - "500 步时就已显著掉分"= ckpt_00500 − base_mean 的 CI 上界 < 0。
  - 判定顺序：若 500 步时已显著掉分，记"早期就有的偏移（风格或领域），不支持纯过拟合"（同时照列趋势条件是否成立）；否则，若单调条件和"final − 500 的 CI 上界 < 0"都成立，记"支持过拟合"；否则记"不支持过拟合"。
  - en2zh 干净、两方向带噪照列，不进判定。
