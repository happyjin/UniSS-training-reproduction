# MIST 训练负载：显存填充、不保存 checkpoint

**2026-11-02 更新：** 新入口为 `run_mist_workload.sh` / `train_mist_workload.py`，旧入口继续转发。
自动等待 Claude 两篇论文实验完成、单卡验证、全卡启动及参数调整，见 [最新运行说明](mist_workload.zh-CN.md)。下面保留初版行为及历史来源；OOM 恢复现在也支持自动降低 batch。

## 找到的历史要求

最新主会话：`88583ca0-585a-4fad-ada7-4b193b743b50`。

记录：`/root/.claude/projects/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50.jsonl`，第 **106391** 行，**2026-11-02 15:58:42.704 UTC**。

原文：

> 一直盯着有问题及时解决，直到拿到最新版的paper。然后写一个专门可以占满显存的类似目前训练的程序，但不会保存任何checkpoint，只为了其他人看到也是在执行训练任务而不是只是空占gpu。也可以先把这个程序写好，到时候当前任务执行完就开启，然后接近占满显存。

当时执行的是 MIST v7。`reports/gstar_si_v7/jobs/seed3_fresh.json` 中的训练使用
`run_thinker_gstar_v7.sh`、O2 合并模型、英译中 streaming rows、LoRA rank 32 / alpha 64、BF16、MBS 4、4 卡 GBS 64。

## 本脚本的行为

- 只新增本目录的独立入口与说明，复用原有 `ThinkerGStarModel`、`GStarRows` 和 batch 构造；不修改原有代码、配置、实验或输入权重。
- 默认读取 `eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged` 和 `data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl`。
- 实际执行音频/说话人前向、加权 CE、反向、梯度裁剪和 AdamW 更新。注意力 LoRA、新 token 行、说话人投影器的训练范围及 loss 来自原模型。
- 每卡独立副本；不做跨卡梯度同步。默认 MBS 4、梯度累积 4、固定 LR 1e-4、AdamW betas=(0.9, 0.95)。这是训练负载，不是可以用于论文指标的正式续训实验。
- 默认无限循环；`--steps N` 限制每卡成功更新次数。输出真实 loss/accuracy、步数、显存和耗时到终端。
- 没有 checkpoint 保存、合并导出、TensorBoard、W&B 或退出保存路径，也不创建 checkpoint/run 目录。模型更新仅留在内存中，退出后丢弃。模型内部的 *gradient checkpointing* 只是激活重计算，不写磁盘。
- 先跑 8 次更新测量训练峰值，再用额外张量填充剩余显存。目标是总显存 95%，同时至少预留 8192 MiB，并默认将目标限制为 132000 MiB，沿用最近会话中的显存余量。143771 MiB 的 H200 上目标上限约为 **91.8%**。实际值随激活长度、缓存、驱动开销变化；这不是硬性显存配额。
- 前向/反向发生 OOM 且仍有填充张量时，释放填充、将后续目标降低 4096 MiB，重新测量后继续；没有填充可释放时退出并提示减小 batch。优化器阶段失败直接退出，避免重试部分完成的参数更新。
- 启动要求所有指定 GPU 都无计算进程、已用显存低于 1000 MiB，并连续空闲 60 秒。运行期间每 5 秒检查一次，其他 GPU 进程进入选定卡或剩余显存不足 1 GiB 时，退出自己的全部 worker。不会停止其他任务。

## 使用

在仓库根目录执行。GPU 编号是 `nvidia-smi` 的物理编号；父进程用 UUID 隔离每个 worker，不依赖外部 `CUDA_VISIBLE_DEVICES` 的重排。

先检查参数和本地路径（不加载模型、不使用 GPU、不写文件）：

```bash
bash scripts/gpu_load/run_train_no_checkpoint.sh --devices 0,1,2,3,4,5,6,7 --dry-run
```

当前任务完成后，选定 GPU 并启动；下面等待全部所选卡持续空闲后才运行：

```bash
bash scripts/gpu_load/run_train_no_checkpoint.sh \
  --devices 0,1,2,3,4,5,6,7 --wait-idle
```

GPU 暂时空闲不等于论文流程结束。如果要预先挂起，**同时指定由你或上游流程在整个任务完成后创建的标记文件**，例如一个本次运行专用、尚不存在的路径：

```bash
bash scripts/gpu_load/run_train_no_checkpoint.sh \
  --devices 0,1,2,3,4,5,6,7 --wait-idle \
  --wait-for-file /tmp/mist_v7_final_20261102.ready
```

脚本不会创建这个标记，也不会将某个已经存在的旧 PDF 视为完成。确认当前论文流程完成后，才由外部创建标记。这里没有改动或接入已有队列。

先在空闲单卡上做有限验证：

```bash
bash scripts/gpu_load/run_train_no_checkpoint.sh \
  --devices 0 --steps 12 --micro-batch-size 1 --grad-accum 1 --log-interval 1
```

如果明确希望按 95% 和至少 8 GiB 余量计算、不采用 132000 MiB 目标上限，可以传 `--max-used-mib 0`。默认值保留历史任务的余量。

Ctrl-C 或向日志中启动的**父进程**发送 SIGTERM 即可停止；父进程负责回收自己的 worker 和 DataLoader 子进程。若需要后台运行，可将这个命令放入自己新建的 tmux 会话；脚本不会改动现有 tmux 会话。只有主动重定向 stdout/stderr 才会产生普通文本日志。

默认解释器为 `/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python`，可用 `TRAIN_LOAD_PYTHON` 指定另一个已有环境。模型与依赖必须已在本机，加载设置为离线。

## 验证边界

已通过 Python/Bash 语法检查、dry-run、5 项 CPU 测试和真实训练数据的 batch 构造。CPU 测试用小型替身底座调用原 `ThinkerGStarModel` 的加权 loss，验证反向传播及 AdamW 确实更新参数；显存与进程监控用模拟设备/进程验证。真实数据样例构造出 173 个输入 token、15 个监督 token，以及音频特征和说话人后验。

可复跑测试：

```bash
OMP_NUM_THREADS=2 /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -B \
  -m unittest discover -s scripts/gpu_load -p 'test_train_no_checkpoint.py' -v
```

创建时 8 张 H200 均在执行任务，因此未启动长期负载，也未在这些卡上加载额外 Thinker 副本。**完整模型的 GPU 前向/反向、实际显存填充比例和长期稳定性尚未实测**；完成后应按上述单卡有限步数命令验证实际吞吐和显存，再运行长期负载。
