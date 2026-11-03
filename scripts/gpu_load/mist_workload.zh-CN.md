# MIST 训练负载与 Claude 实验完成后的自动启动

训练入口为 `train_mist_workload.py`，启动脚本为 `run_mist_workload.sh`。旧的
`run_train_no_checkpoint.sh` 和 Python 入口继续兼容，执行时会转入新名字的进程。
使用当前 MIST 模型和真实数据进行前向、反向、AdamW 更新；训练结果只在内存中，不保存 checkpoint。
历史原文、数据路径和原始实现说明见 `train_no_checkpoint.zh-CN.md`。

## 已配置的后台等待程序

入口：`after_claude_workload.py`。
配置：`after_claude_20261102.json`。
独立状态目录：`logs/mist_workload_after_claude_20261102/`。

2026-11-03 用户确认立即启动后，已重新核对实验收尾和八卡空闲，当前监督程序使用
该状态目录中的 `run_now_config.json`。此运行配置跳过原先的额外稳定等待，每 5 秒检查一次；
实验完成条件、启动前 GPU 空闲检查、单卡 12 步验证及自动恢复仍保留。
下列 10 分钟和 120 秒等待是原始预约配置的设置。

它每 30 秒只读检查当前的 **MIST v7 和 QABC v11**，须同时满足：

1. `logs/multispk/v7/queue3/state.json` 没有 `wait` / `run` 作业，`add/*.json` 没有尚未被队列读取的作业。
2. 两个最终回复文件已写完并提交：MIST 的 `docs/response_to_review_v6.zh-CN.md`、QABC 的 `docs/response_to_review_v10.zh-CN.md`。草稿占位符不算完成。
3. 两篇论文的 `main.pdf` 在本次请求之后重新生成，晚于最新 LaTeX/BibTeX 源码，论文源码和 PDF 没有未提交修改。
4. `paper/bayes_simul_s2st_v11/generated/v11_cont.json` 的双向结果均无缺项；当前 `cont_tasks*.jsonl` 中的语音任务均有完整 ASR 摘要与足量转写。
5. 上述完成状态持续稳定 10 分钟，8 张卡随后持续空闲 120 秒。

旧失败/取消的作业不会单独阻止启动，因为 Claude 已通过替代作业重跑；最终论文、完整结果和队列清空共同作为收尾依据。共享队列可以继续常驻，不要求 Claude 写 `FINISH` 或退出应用。如果 Claude 改了最终文件命名或未产出这些收尾文件，等待程序会继续等待，原因写入 `status.json`；它不会猜测任务已完成。

只有确认实验收尾后，如果卡上剩下的全是本仓库原有 `target_gpu_util.py` 矩阵负载，才按核实的父 PID 发 SIGTERM，为这次新负载让位。进程工作目录、脚本路径、GPU 范围和 PID 启动时间都会核对；其他训练、解码和其他人的进程都不停止。

## 启动后的验证和自动调整

- 先在第一张指定卡做 12 步真实训练验证，通过后才开 8 卡。
- 自动模式设置 `--max-used-mib 0`，显存目标 95%，目标预留 8192 MiB。目标按 CUDA 报告的可用总容量计算；本机单卡验证的目标为 **134966 MiB**，实际占用约 **135217 MiB（约 94%）**，会随训练激活和分配器缓存小幅波动。
- 每个 worker 用最近 64 步的训练峰值校准填充。旧的大峰值离开窗口后可以补充显存；OOM 后先释放填充，必要时将 micro batch 从 4 降到 2、再降到 1，同时增加梯度累积。稳定后逐步回收 OOM 预留余量。
- 后台程序在进程级失败时按错误类型调整 batch、预留显存、学习率或 DataLoader worker 数，并从只读底座重新启动。至多 6 次自动恢复，不写实验代码，不安装依赖。
- 如果显存连续一段时间低于目标 4 GiB 以上，自动缩短峰值统计窗口重新校准，至多 2 次；启动/加载阶段不据此判定不足。
- 出现新的实验作业时释放自身负载；训练入口检测到同卡其他 GPU 进程时也会退出，等待卡再次空闲。
- 不认识的异常、恢复次数耗尽或 batch=1 仍无法容纳时，状态记为 `failed`，保留对应日志并停止。这里实现的是明确的自动恢复规则，不是一个能在无人查看时修复任意新错误的代码代理。

## 查看和停止

```bash
cat logs/mist_workload_after_claude_20261102/status.json
tail -n 30 logs/mist_workload_after_claude_20261102/supervisor.log
```

状态包括 `waiting`（等待实验完成）、`validating`（单卡验证）、`running`（全卡运行）、`failed` 和 `stopped`。
训练日志为同目录下独立的 `training_<时间戳>.log`，不覆盖实验日志。

停止整个等待程序和它启动的训练：

```bash
touch logs/mist_workload_after_claude_20261102/STOP
```

也可以对 `supervisor.pid` 中的 PID 发 SIGTERM；程序会回收自己启动的训练进程。服务未配置开机自启，服务器重启后不会自动恢复。

手动运行新名字的训练入口：

```bash
bash scripts/gpu_load/run_mist_workload.sh \
  --devices 0,1,2,3,4,5,6,7 --wait-idle --max-used-mib 0
```

## 其他人能看到什么

`nvidia-smi` 通常显示 Python 可执行文件路径及 PID、显存；某些界面会截断路径。
具备系统读取权限的人查看 `ps` 或 `/proc/<PID>/cmdline`，能看到类似：

```text
.../bin/python -B -u .../scripts/gpu_load/train_mist_workload.py --devices ... --worker 0
```

后台等待程序本身显示 `after_claude_workload.py`。DataLoader 子进程也可能显示 Python multiprocessing 的启动命令。
改名只改变入口名称，不隐藏进程、脚本内容或用途；日志中保留训练负载及不保存 checkpoint 的说明。

## 验证

```bash
OMP_NUM_THREADS=2 /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -B \
  -m unittest discover -s scripts/gpu_load -p 'test_*.py' -v
python3 -B scripts/gpu_load/after_claude_workload.py \
  --config scripts/gpu_load/after_claude_20261102.json --check-once
```

2026-11-03 05:38 UTC，完整模型的单卡 12 步训练验证通过，loss 均有限，显存填充正常；
日志确认结束时未保存 checkpoint。随后已启动八卡持续训练。05:42 UTC，八卡均已完成至少
10 步更新，未出现 OOM 或运行错误；`nvidia-smi` 实测每卡占用 134835–135275 MiB，约 94%。
此次启动验证不代表全卡长期稳定性测试。
