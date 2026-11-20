# 鲁棒第八波：Agreement-Unlikelihood（AUL）预登记（2026-11-20 07:55 CET，写在任何训练和评测之前）

用户 2026-11-20 早上说"开始"。约束：
- 严格非 QABC 的 LA2 底座，中→英和英→中共用一个模型；
- 不做教师–学生蒸馏，没有后验阈值；
- 一个设置、一个种子，不扫参；
- 推理开销不变。

## 动机
- 第三波 oracle 诊断：中→英 CoVoST 带噪的损失约 70% 来自提交，即 LA2 在噪声上让两次连续的草稿认同了错误的前缀。
- ERM-WiSE 已经通过统一规则（第五波变更 4）。本波检验：直接把"认同但错误"的前缀当作负样本训练，能否在 ERM-WiSE 之上再降低提交错误。
- 借鉴：Unlikelihood Training（Welleck et al., ICLR 2020）。
- 据我们所知，这是第一次把同传提交策略的失败模式写进训练目标。

## 方法（固定）
- 底座、数据、环境、步数、采样随机数、LoRA 结构、优化器都与第五波 ERM（λ_Fishr = 0）完全相同：2300 步，全局批量 64，即 8 个环境 × 8。
- **变化：** world 4（卡 4–7，第五波 ERM 是 world 2）。全局样本不变，只有每张卡的微批量分配不同。
- **AUL 样本：** 每个 rank 每步、每个环境取本 rank 的第一个 slot。每步 world 4 × 8 个环境 = 32 个样本，含干净环境。对每个样本：
  1. 读窗 w = 1 s（与评测的 `--windows 1` 一致）。k 从 {1, …, ⌊dur/1 s⌋ − 1} 中按 sha1("20261120|w8aul|cut|s|j") 选；没有可选的 k 就跳过。
  2. 在 no_grad、eval 模式下，用相同的 prompt，对音频 x[:k·w] 和 x[:(k+1)·w] 各做一次贪心续写。助手前缀为空，相当于 LA2 从空提交开始；上限为 max(参考单元数 + 8, 16) 个 token。
  3. 按 LA2 的单元切分（英文按词，去标点并转小写后比较；中文按字），取两次草稿的最长公共前缀 A，与参考 R 逐单元比较，找出第一个不一致的位置 j。若 j < |A|，就是认同但错误：A_j 是负样本。
  4. UL 项：上下文是音频 x[:(k+1)·w]，助手前缀是 R 的前 j 个单元（等于 A 的前 j 个单元）的文本，对 A_j 的第一个 token y⁻ 计
     ℓ_UL = −log(1 − p_θ(y⁻ | ·))。
- **总损失：** L = mean_i CE_i + λ · mean_{有负样本的 AUL 样本} ℓ_UL，λ = 1.0（Welleck 的默认值；固定，不扫）。没有负样本的步只用 CE。
- **WiSE-FT：** 与 ERM-WiSE 相同的协议，W0 + 0.5·Δ_AUL，即 alpha 减半。最终比较的是 aul_wise05。
- 不是蒸馏：负样本来自当前模型自己的贪心草稿，正样本只用参考译文；没有教师模型，也不拟合任何模型输出。
- 不是 QABC：没有后验，没有阈值，推理仍是原样的 LA2。

## 评测（与第五波相同的链）
- 合并 aul_wise05 → `si_thinker_stream --policy la2 --windows 1` → T1c（4 个分片，种子 20261021）→ 只对语音做 ASR。打分用 `wave5_m_score.py`：三次底座运行、噪声底、bootstrap 都与第五波相同。
- **主判定（全部写定）：**
  1. 相对底座的统一规则：所有 (a)/(b) × 干净/带噪单元，text BLEU 和 ASR-BLEU 的点估计 ≥ −0.3，且 CI 上界 ≥ 0；至少一个带噪单元的 text 增益 CI 下界 > 0，且超过噪声底。
  2. 相对 ERM-WiSE 的增量：
     - 至少一个带噪单元的 aul_wise05 − erm_wise05 text BLEU CI 下界 > 0，且点估计超过该单元的噪声底；
     - 同时没有任何 (a)/(b) 单元的 aul_wise05 − erm_wise05 点估计 < −0.3 且 CI 上界 < 0。
  - 第 1 条过、第 2 条不过：记为"AUL 没有带来增量"，主系统仍用 ERM-WiSE。
- **辅助（照列，不作判定）：**
  - LAAL D1、D2；
  - 提交统计（`wave3_d_cpu.py` 的口径：提交后被参考否定的单元比例），把 aul_wise05 和 erm_wise05 并列。
- **两条都过时：** 用 aul_wise05 的 Thinker 加 V1 音色加 T1c，跑与第五波"完整系统组合"相同的评测（`wave5_m_combo_*`），再与 ERM-WiSE 的完整系统比较。

## 风险（事先写明）
- 中→英的参考有语序重排，第一个不一致的单元可能其实是正确的意译，这会把正确的提交也压低。只罚第一个不一致的单元，能减轻这个问题。
- 每步要多做 2 × 8 次贪心续写，训练会更慢。

## 实现说明与实现修正（2026-11-20 08:00 CET，冒烟之后、正式训练结果和任何评测数字之前）
λ、AUL 样本、步数、WiSE 都没有改。以下是对方法描述中未写明之处的实现选择，以及冒烟中发现的一个实现错误。
- **代码：** `voicecond/wave8_aul_train.py`（import `wave5_m_train.py` 的 Sampler / build_model，不修改它；CE 部分逐行同第五波 ERM）；启动 `wave8_aul_run.sh`，占位 watcher `wave8_aul_noplaceholder.sh`，评测 `wave8_aul_eval.sh` / `wave8_aul_eval_orch.sh` / `wave8_aul_chain.sh`，打分 `wave8_aul_score.py`（第五波打分的副本，加 aul / aul_wise05 臂和 aul_wise05 − erm_wise05 增量规则），辅助提交统计 `wave8_aul_commit.py`。
- **单元比较：** 两次草稿的最长公共前缀和与参考的比较都在 norm() 之后（英文词去 Unicode 标点并转小写；中文字原样，标点字符也按字比较）。A 取后一次草稿（x[:(k+1)w]）的原始单元，与 LA2 从当前草稿提交一致。
- **截断：** 草稿到达自身上限 max(参考单元数 + 8, 16) 仍未输出 <|im_end|> 时，丢掉最后一个（可能被截断的）单元。
- **y⁻：** 取"草稿从 A_j 起的续写文本"（英文 j > 0 时带前导空格）分词后的第一个 token；这样中文多字 token（如"挥手"）与模型实际生成的一致。若 y⁻ 等于参考续写的第一个 token（共享的 BPE 前缀，例如 " un"），该样本不算负样本，避免 UL 压低参考 token。j = |R| 时参考续写的第一个 token 记为 <|im_end|>。
- **生成：** 每个 rank 每步把 8 个样本 × 2 个截断拼成一个批次（左填充）做一次 HF greedy generate（eval 模式、no_grad、KV cache），每条输出再截到自己的上限，与逐条生成等价。
- **实现修正（冒烟发现）：** Qwen3-Omni 的 processor 不论 tokenizer.padding_side 都在左侧填充；第一次冒烟（`smoke.superseded_v1`）的 UL 前向按右填充取"最后一个位置"，较短的行取错位置（p(y⁻) ≈ 0）。已改为取 attention mask 中最后一个 1 的位置；第二次冒烟中 p(y⁻) = 0.33–0.99，与模型自己的贪心草稿一致。
- **冒烟（卡 4–7，world 4，第二次 6 步）：** 每步 13–16 s（生成 4–8 s），显存峰值 65.8 GiB；每步 29–32 个 AUL 样本中 9–13 个有负样本（约 37%）；ℓ_UL 0.87–1.90，CE ≈ 1.0。2300 步预计约 9.5 h，未超过 10 h，生成上限不需要改。
- **正式训练：** 2026-11-20 07:57 CET 在卡 4–7 启动（pid 1282229，`eval_outputs/mist_w3po_v1/wave8/AUL/run_aul`，日志 `logs/mist_w3po_v1/wave8/aul_run_aul.log`）。训练结束后 `wave8_aul_chain.sh` 自动执行：写 run_aul_wise05（alpha 减半）→ aul_wise05 在卡 4–5、aul（辅助）在卡 6–7 跑完整评测链 → 打分 → 提交统计。
