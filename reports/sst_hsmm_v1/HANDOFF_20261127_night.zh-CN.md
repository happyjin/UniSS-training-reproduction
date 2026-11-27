# HANDOFF 2026-11-27 夜间（en2zh 根因修正 + 迭代轮次）

用户 21:25 CET 去睡觉，授权：
- 需要点头的事按最佳选项自主决定，到 11-28 上午为止；
- 最多 8 轮"分析 → 新的、ICLR 级 novel 方案 → 训练 + 测试"；
- 早上交付全部方法相对 baseline 的 benchmark、是否有效、结论和分析（中文）。

## 背景（必读）
- `reports/sst_hsmm_v1/en2zh_failure_analysis_20261127.zh-CN.md`
  - en2zh 失败的根因：预判 LoRA 在 UniST 参考上训练，带来两种风格偏移，即英文专有名词被音译、语体变书面。这个损失在整句离线解码时也存在。
  - dev 用的原始 sacrebleu 看不出这个损失。
- prereg：`reports/sst_hsmm_v1/prereg.zh-CN.md`，change 57（A / B1 / B2）、change 58（C1 / C2）。

## 运行中的链路（全部 detached）
- **第 1 轮（change 57）**
  - b1tr、b2tr 训练 → `pv4_chain.sh`（M=b1/b2）：G0（`pv4_g0.sh`，结果 `pv4/g0_<m>.json` + `G0_<m>_PASS/FAIL`）→ 通过才跑流式 pfn1 / pff1 → `pv4/CHAIN_<m>_DONE`。
  - `pv4_final.sh`：两条链都结束后，算 COMET 并生成 `reports/sst_hsmm_v1/pv4_tables.zh-CN.md`。
- **第 2 轮（change 58）**
  - C1：c1tr 训练 → G0 → pfi1。
  - C2：HCG 冒烟（`pv5/HCG_SMOKE_OK`）→ pfg1（hsc23）。
  - 链路是 `pv5_chain.sh`（M=c1/g1），结束标志 `pv5/CHAIN_<m>_DONE`。
- 等待脚本放在 scratchpad 的 `pv4_wait.sh`；后续轮次要扩展到 pv5+。

## 规则
- 每轮先分析，再在 prereg 登记 change 59+，然后才启动。
- 每个方法一个设置；无蒸馏；只用开放数据；RealSI / OBSI 只测试。
- 判定：相对 p3hk，(i) ASR ≥ −0.2 且 D2 ≤ −0.10 s，或 (ii) 比 p3hk–p3hk2 连线高 ≥ 0.2。两个方向分别判定。
- 不删除文件；按 PID 停进程；不碰 guard 2726951、gpu_watch 52045、sync 1222453。
- 时间一律用布拉格时间：写之前先运行 `TZ=Europe/Prague date`。
- 早上的报告要汇总所有轮次：表格（COMET + 全指标）、判定、分析，以及对新颖性的诚实评估。

## 进度记录（23:28 CET 更新）
| 轮次 | 方法 | G0 en2zh（门槛 40.81） | G0 zh2en（w17 30.64） | 状态 |
|---|---|---|---|---|
| 1 | B1 保留术语 | 38.43 | 27.56 | 未过 |
| 1 | B2 只学预判 | 35.60 | 25.02 | 未过 |
| 2 | C1 完整输入 KL 锚定 | 40.15 | 30.71 | 未过 |
| 2 | C2' HCG v2（推理期，无 G0） | – | – | pfg2 流式 dev 中（`pv5_chain2.sh`） |
| 3 | SAA 按支持分配目标 | 40.43 | 30.58 | 未过 |
| 4 | EGA 耗尽门控预判（尾部 = SAA） | 待出 | 待出 | 冒烟 + G0-EGA 中（`pv6_ega.sh` → `pvx_chain.sh` M=q1，流式 TAG pfq1） |

- 分析脚本：scratchpad 下 `g0split.py`（按参考是否含英文拆分 G0）。
- 等待脚本：scratchpad 下 `px_wait2.sh`，比 px_wait.sh 多一个 FILES 环境变量。

## 进度记录（00:26 CET 更新）
| 轮次 | 方法 | 状态 |
|---|---|---|
| 2 | C2' HCG v2（pfg2） | dev 不过（en2zh 同设置 −3.26）；测试 p3pfg2 运行中 |
| 4 | EGA（pfq1，change 60） | G0 通过（41.33）；dev 不过（ΔD2 +0.02/+0.07）；测试运行中 |
| 5 | CCA（pfr1，change 61） | dev 不过；测试运行中 |
| 6 | SAA 流式，越过 G0（pfs2，change 62） | rollout 中 |
| 7 | HCD token 级视野对比解码（pfk1，change 63） | 冒烟通过（zh2en 34 个草稿改道 10 次，en2zh 0 次）；rollout 中 |

- 关键机制发现（change 62，`en2zh_analysis/a11_draft_cmp.py`）：预判起草器的时延收益几乎全部来自已听到内容的前缀一致性（E1 每读多提交约 2 个单位，其中只有 0.15 个超出 p3hk 草稿）。EGA / CCA 的预判尾部本身没有用。
- 等结果：各链路结束会生成 `pvx/CHAIN_<m>_DONE`（q1 / r1 / s2 / k1）和 `pv5/CHAIN_g2_DONE`。之后用 `pv8_final.sh`（NEWS = p3pfg2 p3pfq1 p3pfr1 p3pfs2 p3pfk1，EXTRA_SYS 加上 pfs2 / pfk1 的名字）算 COMET 和表格。
