# v11 P2 预登记：流畅性约束的播放对齐 Talker（PA-F）

- 写于 2026-11-15 布拉格时间约 10:00（CET），在任何 PA-F 训练和评测之前提交。
- 依据：docs/mist_v11_plan.zh-CN.md §3 P2；用户 2026-11-15 授权按最优选择决定（记忆 autonomy-20261115-v11）。
- 动机：审稿 M4。PA Talker 在测试上比基线快 45%（Zh→En）/ 29%（En→Zh），≥0.5 s 停顿/分钟约翻倍。

## 1. 方法（一个设置，不扫参）

PA 的后验是 λ=0 播放 CRF 在延迟约束 E_q[Σ l_n]/ΣW ≤ b 上的 KL 投影。PA-F 再加一个期望约束：

  E_q[Σ_n d_{c_n}] / Σ_n W_n ≥ s_min（每词 / 每字的语音秒数下界，即语速上界）。

两个约束下的投影仍是线性链 CRF：ψ_n(b,c) = 0.5·l̄(c) + 5·f(c) − λ·L_n(b,c) + μ·d_c，λ, μ ≥ 0 为对偶变量，
用精确的对偶坐标上升求解（固定 μ 二分 λ 使延迟约束成立；固定 λ 二分 μ 使时长约束成立；交替至两者都成立）。
前后向在积压状态上仍是精确的。代码：experiments/mist_w3po_v1/talker/fluency_rows.py（test_fluency_rows.py：
与穷举一致、μ=0 时与 PL-EM 逐位一致、单调性、约束满足）。

## 2. 预算（只用训练集统计，训练前定死）

- 延迟预算 b 不变：规则路径自身延迟 − 0.3 s（与 PA 相同）。
- 语速上界：r_cap = r_0 + 0.5·(r_PA − r_0)，s_min = 1/r_cap。r_0 = 训练候选中基础语速（speed_base）候选的语速，
  r_PA = PA（PL-EM）后验下的期望语速。即"把 PA 相对基础语速的提速砍掉一半"。两方向各用自己的训练统计。
- Zh→En 训练统计（eval_outputs/mist_w3po_v1/talker_crf/rows_fluency/stats_zh2en.json）：r_0 = 2.971 词/秒，
  r_PA = 3.422，r_cap = 3.196，s_min = 0.3129 s/词。En→Zh（stats_en2zh.json，每字）：r_0 = 4.431 字/秒，r_PA = 5.023，r_cap = 4.727，s_min = 0.2116 s/字。
- 不可行时的规则（事先定）：若两个约束不能同时满足（坐标上升 50 轮内未收敛或某对偶到上限 50），保留语速约束，
  延迟预算放宽为规则路径自身延迟（减 0 s）；仍不可行则停止，不训练，如实报告。

## 3. 训练（与 PA 完全相同）

- rows：fluency_rows --rate-frac 0.5（Zh→En 用 train16_crf_v2p_s3000 候选；En→Zh 用 train16_en2zh_v2p s2000 候选，
  --lang cmn --units char），格式与 plem rows 相同（mode plem，variant fluency）。
- 训练：run_talker_crf.sh MODE=plem，T1c 起点，1200 步，4 卡，其余超参不变。
- 检查点不做新选择：沿用 PA 的选择——Zh→En 用第 1200 步（export），En→Zh 用第 400 步（export_i0400）。

## 4. dev 评测与门槛

- 文本：严格基线 B（joint v8h6k s6000）已有的 dev 解码，读级合并（_rl），两个方向全 dev；语音链与 E2 的 rlplem
  完全相同，只换 Talker（PA-F），两个 TTS 种子。
- 对照：E2 的 B+PA dev 输出（jdev_b{z,e}_…__rlplem 及 _tts2）。
- **硬门槛（每个方向）**：gate_2seed 的语音项：ASR-BLEU 点估计 ≥ B+PA − 0.2，且按会议配对 bootstrap
  P(差 ≤ 0) < 0.95（与以往所有阶段门槛相同）。文本相同，文本项自动满足。
- 报告（不作门槛）：≥0.5 s 停顿/分钟、句中停顿/分钟（talker_block_pauses）、语速、LAAL_block、LAAL_spread、
  speech start、EndOffset（chunk 协议）、UTMOS（种子 1）。
- 主张成立的描述性标准：停顿/分钟和语速都低于 B+PA。

## 5. 测试

- 只有通过 dev 门槛的方向才在测试上各跑一次：Zh→En Talker → AMI Zh→En 和 AliMeeting；En→Zh Talker → AMI En→Zh。
- 同一链、两个种子，与 B+PA 测试输出比较；测试上不做任何选择。
- 若 dev 门槛不通过：不测，作为负结果写入报告。
