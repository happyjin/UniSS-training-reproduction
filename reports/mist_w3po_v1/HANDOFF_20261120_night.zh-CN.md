# 交接：鲁棒线夜间（2026-11-20 22:15 CET）

用户已睡，到 11-21 白天都按最优选择决定（memory overnight-autonomy-20261120-robust）。回复用中文，时间写 CET。

## 当前主线
- 分析和方案见 `robust_analysis_and_ideas_v3_20261120.zh-CN.md`，含末尾的更正。
- 唯一的 motivation：噪声鲁棒适配会把"声学不变性"和"输出分布偏移"纠缠在一起，有效的方法应当只学不变性。
- 训练只用同一批 UniST 数据。dev 用 UniST 同域 dev：`eval_outputs/mist_w3po_v1/devU/data`，加 bench_dev (b)。
- 不跑 AMI / AliMeeting；先看点估计，不做 bootstrap。

## 正在运行（都脱离会话）
| 方案 | 卡 | 完成标记 | 预计出结果 |
|---|---|---|---|
| R1 任务向量差 τ_noisy − τ_clean | 1、2、3、7 | `eval_outputs/mist_w3po_v1/r1/DONE` 或 `FAILED` | 11-21 约 03:00 |
| R2 不看译文的编码器对比学习 | 4、5 | `r2/DONE` 或 `FAILED` | 约 23:00 |
| R3 噪声子空间约束的编码器 LoRA | 0、6 | `r3/CHAIN_DONE` 或 `CHAIN_FAILED` 或 `EVAL_FAILED` | 约 01:30 |
| R4 一致性路由，推理期 | 等 R2 或 R3 交出的卡 | `r4/DONE` 或 `FAILED` | R2 交卡后约 1 h |

- 等待器：`scratchpad/wait_r.sh`，四个方案都结束或任一失败时唤醒。
- 测试：test_robust 里 UE / UW 已跑完，打点估计用 `test_quick.py UE:v1 UW:v1 B:b C:v1`。各预登记在 `reports/mist_w3po_v1/robust_r{1,2,3,4}_*prereg.zh-CN.md`。

## 下一步
1. 按各预登记的点估计规则判读，写进"结果"一节，记入决策日志，然后 commit 并 push。
2. 无效的方法：逐个分析原因，再从 `robust_novel_ideas_v1` 的参考文献和 v3 分析出发，设计几个新方案，先写预登记，再并行跑。
   - 委托工作用新开的子代理，给简洁的 brief，不用 fork，省 token。
3. 有效的方法：在测试上补跑，测试只跑一次，要先补写预登记的变更。

## 不能碰的
- 进程：guard 2726951、gpu_watch 52045、sync 1222453。只按 PID 停进程。
- 不用 rm，拒绝覆盖。不就地修改正在运行的脚本。
