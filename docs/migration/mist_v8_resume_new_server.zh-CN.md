# MIST v8：换到新服务器后如何继续

写于 2026-11-04 05:05 UTC（布拉格 06:05）。

本文只讲 MIST v8 剩下的工作。环境、软链接和写死路径的问题，见同目录的 `new_server_run_guide.zh-CN.md` 第 0 节；那一节的原则这里同样适用。

记号：
- **J** = `/opt/dlami/nvme/jasonleeeli`
- **R** = `J/projects/UniSS`
- **S3** = `s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli`
- **PY_EVAL** = `J/conda_envs/uniss-eval/bin/python`
- **PY_OMNI** = `J/conda_envs/qwen3omni/bin/python`

方案原文：`R/docs/mist_v8_plan.zh-CN.md`（含修正 1）。预注册：`R/paper/gstar_si_v8/docs/prereg_v8.zh-CN.md`（含修正 1、2）。

---

## 1. 旧服务器上停在哪里

**中断**：
- 11-04 约 04:02–04:06 UTC，v8 的所有解码进程、两个启动器（`gpuq8.py`、`memq8b.py`）和执行代理挂的后台监控都消失了，日志里没有退出记录；
- 约 04:40 UTC 起，8 张卡被另一个 K8s 作业（`swift sft`，`/nvme/code/liveinterpret_training_pipelines`）占满。

**已完成**（输出都在 `R/eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/`，已同步到 S3）：
- **A 阶段**（CPU）全部完成：
  - overlap 评测模块 `reports/gstar_si_v8/overlap_score.py`，带单元测试；
  - 用新指标重打分；
  - 延迟排队递推 `listener_latency_v8.py`；
  - 三种分母；
  - 声音提示冷启动；
  - 标签与翻译的贡献拆分。
- **预注册** prereg_v8 已提交：在任何 v8 测试解码之前提交，之后的修正 1、2 也都在相关结果出来之前提交。
- **C1 训练**完成，两个方向的 speaker-blind 前缀模型都已合并：
  - `R/eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_{en2zh,zh2en}_v8/thinker_merged`，各 66 GB，S3 上已有。
- **B1、B5、B7、B8、C2** 都已完成，结果已写进 v8 草稿附录。
- **整场在线解码**，13 组中 12 组的窗口已经齐全（`runs/` 下）：

| 组 | 窗口 | 状态 |
|---|---|---|
| `on_{en2zh,zh2en,ali}_S2dlnosd`（C1，E-ov1/E-ov2 要用） | 3387 / 3365 / 2837 | 齐全 |
| `on_{en2zh,zh2en,ali}_S2dlLA2`（同模型 LA2，E-pol 要用） | 3387 / 3365 / 2837 | 齐全 |
| `on_en2zh_S2dl_seed2`、`seed3`，`on_zh2en_S2dl_seed2`，`on_ali_S2dl_seed2`（B4） | 全部 | 齐全 |
| `on_{en2zh,zh2en}_S2dllag2`（固定滞后 2 s，只跑子集会议） | 1584 / 1643 | 齐全 |
| **`on_ali_S2dllag2`** | **747 / 1439** | **还差 692 个窗口** |

窗口"齐全"指各分片、续跑和辅助进程的 `hyp_*.jsonl` 按 window id 合起来覆盖了全部窗口。合并成 `hyp.jsonl` 由第 3 节的 R3 完成。

**论文**：`R/paper/gstar_si_v8` 是独立 git 仓库，在 `v8` 分支上，最新提交是 `38617ec`。现在的 `main.pdf` 是 11-03 10:07 UTC 的草稿：正文 8 页，没有 `??`，没有 Overfull，但还没有最终 v8 数字。`generated/v8_tests.json` 也是那时的版本。

**主仓库**：v8 的代码都已提交并推送到私有仓库 `main`（`ed037b79` 之后还有本文档这一次提交）。

---

## 2. 新服务器上要恢复的东西

最省事的做法是把 J 整个从 S3 同步回来：

```bash
S5=J/tools/s5cmd_dl/bin/s5cmd     # 若新机还没有 J，先单独下载 s5cmd
AWS_REGION=ap-southeast-3 $S5 --numworkers 256 sync "$S3/jasonleeeli_backup/*" /opt/dlami/nvme/jasonleeeli/
bash R/docs/migration/restore_symlinks.sh        # 恢复软链接，详见 new_server_run_guide
```

如果只想恢复 v8 需要的部分，至少要有下面这些（路径都相对 R，S3 上对应 `$S3/jasonleeeli_backup/projects/UniSS/<同一路径>`）：

| 内容 | 路径 | 大小 |
|---|---|---|
| 代码 | 整个仓库（`git clone` 私有仓库，再用 S3 补上未跟踪的 `reports/`、`paper/`） | — |
| 测试窗口 | `data/online_v6_20261031/` | 2.7 GB |
| v8 输出 | `eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/` | 5.6 GB |
| 固定滞后和 B6 用的模型 | `eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged` | 66 GB |
| B6 用的模型 | `thinker_gstar_s2dl_20261029b/thinker_merged`、`thinker_gstar_o2_20261029b/thinker_merged`（都在 `eval_outputs/uniss_qwen3_omni_v1/` 下），以及作业 `needs` 里列出的 `thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged/sd_proj.pt` | 每个模型 66 GB |
| C1 模型（只在要重跑 C1 解码时需要） | `eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_{en2zh,zh2en}_v8/thinker_merged` | 各 66 GB |
| 环境 | `J/conda_envs/qwen3omni`、`J/conda_envs/uniss-eval`、`J/venvs/vllm_omni_030`，以及 `J/tools/`（tectonic、PyMuPDF） | 约 19 GB |
| 音频 | 窗口 jsonl 里引用的绝对路径（AMI、AliMeeting 源音频） | 见 new_server_run_guide |

路径必须和旧服务器完全相同，否则窗口 jsonl 里的绝对路径都会失效。

---

## 3. 剩下的任务（按顺序）

### R1 AliMeeting 固定滞后：补完 692 个窗口

先生成缺口清单。用 `open(..., "x")`：文件已存在时直接报错，不会覆盖。

```bash
cd R
$PY_EVAL - <<'EOF'
import json, glob
d = "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2"
sub = set(json.load(open("data/online_v6_20261031/subset_meetings.json"))["ali"])
W = [json.loads(l) for l in open("data/online_v6_20261031/windows_test_online_ali.jsonl")]
got = set()
for f in glob.glob(d + "/hyp_*.jsonl"):
    for l in open(f):
        try: got.add(json.loads(l)["id"])
        except Exception: pass
miss = [w for w in W if w["meeting"] in sub and w["id"] not in got]
with open(d + "/resume3.windows.jsonl", "x") as fh:
    fh.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in miss)
print(len(miss))   # 应为 692
EOF
```

然后在 4 对卡上各跑一片。卡上没有别的作业时，每片用 2 卡、`--max-gib 34`、batch 16。batch 32 在 AliMeeting 长窗口上单进程会涨到约 57 GB，只有一张卡上没有别的作业时才可以用。

```bash
cd R; export PATH=J/venvs/vllm_omni_030/bin:$PATH
D=eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2
for k in 0 1 2 3; do
  CVD=$((2*k)),$((2*k+1))
  CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=.:third_party/Megatron-LM setsid nohup $PY_OMNI -m \
    experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s \
    --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged \
    --windows $D/resume3.windows.jsonl --out $D/hyp_r3s$k.jsonl --shard $k --num-shards 4 \
    --policy lag --lag-ms 2000 --window 1 --post-key lowlat --dir zh2en --batch 16 --max-gib 34 \
    > $D/infer_r3s$k.log 2>&1 < /dev/null &
  sleep 60
done
```

- 日志里出现 CPU 卸载（offload）时，说明显存设置不对，必须停掉重来，否则会慢几十倍。
- 预计 2–3 小时。
- 新输出文件名用 `hyp_r3s*`，不会和已有文件冲突；R3 会按 window id 合并所有 `hyp_*`。

### R2 B6 墙钟回放（6 个作业）

清单在 `R/reports/gstar_si_v8/jobs/b6.json`，内容是 3 组会议 × {MIST-Simul, 离线 + LA2}，batch 1，`--time-reads`。

**必须在没有任何其他作业的卡上跑**，否则测出的 RTF 不可信（prereg_v8 修正 2）。每对卡一个作业，可以 4 对卡并行：

```bash
cd R; export PATH=J/venvs/vllm_omni_030/bin:$PATH
$PY_EVAL - <<'EOF'
import json, subprocess, time
jobs = json.load(open("reports/gstar_si_v8/jobs/b6.json"))
pairs = ["0,1", "2,3", "4,5", "6,7"]
for i, j in enumerate(jobs[:4]):
    subprocess.Popen(["setsid", "bash", "-c", j["cmd"].replace("$CVD", pairs[i])], stdin=subprocess.DEVNULL)
    time.sleep(60)
EOF
# 前 4 个结束后，同样再跑 jobs[4:]（不要和前 4 个叠在同一对卡上）
```

"重叠下约 2 s 延迟"这个主张，以 B6 测出的 RTF < 1 为前提。

### R3 合并、重打分、生成表格

```bash
cd R && $PY_EVAL reports/gstar_si_v8/after_v8.py
```

它会：
1. 把窗口齐全的每组 `hyp_*` 合并成 `hyp.jsonl`（已有的 `hyp.jsonl` 不会被覆盖）；
2. 运行 `rescore_existing.py` 重打分；
3. 运行 `paper/gstar_si_v8/scripts/make_tables_v8.py`，重写宏、表格和 `generated/v8_tests.json`。

**B6 结果**：B6 的分析脚本还没有写，现有的 `after_v8.py`、`make_tables_v8.py` 和 `listener_latency_v8.py` 都不读 `b6/`。要新写一个 `reports/gstar_si_v8/b6_wallclock.py`：
- 读 `b6/*_hyp.jsonl` 里每次读取的 `read_ms`；
- 按 1 s 节拍做排队递推（每次读取的开始时间 = max(计划时刻, 上一次读取结束)）；
- 报 RTF、延迟超过 5/10/30 s 的比例、超时和失败数、墙钟文本延迟。

结果写进 `gstar_v8_20261103/b6/b6_summary.json`，再由 `make_tables_v8.py` 读入。

### R4 统计检验

`make_tables_v8.py` 按 prereg_v8 做全部 12 项检验（E-ov1、E-ov2、E-lat、E-pol 及其子项）：
- 会议级精确符号翻转检验；
- 10k 次会议重抽样 bootstrap，每次叠加一次解码噪声；
- Holm 校正。

检验结果有持平或失败时，按 prereg_v8 里事先写好的措辞写进论文，不能改口径。

### R5 论文定稿（方案阶段 D）

- 正文按四组证据改写：
  1. 重叠多人会议的翻译；
  2. 同一模型上的提交策略；
  3. 整场会议和评测覆盖；
  4. 用户实际听到的内容。
- 编译：在 `paper/gstar_si_v8` 下运行 `J/tools/tectonic/tectonic --keep-logs main.tex`。
- 检查：正文 ≤ 8 页，没有 `??`，没有 Overfull，没有未定义的引用。用 PyMuPDF 检查第 8、9 页的内容。

### R6 回复审稿文档

新建 `R/docs/response_to_review_v7.zh-CN.md`，逐条回应：
- 第一份 ACL 审稿 M01–M25；
- 再审 M01–M18。

### R7 提交与推送

```bash
cd R/paper/gstar_si_v8 && git add -A . && git commit -m "v8: final"       # 论文仓库，v8 分支，不推 GitHub
cd R && git add docs/response_to_review_v7.zh-CN.md reports/gstar_si_v8 && git commit -m "gstar v8: final"
git pull --rebase private main && git push private HEAD:main                # 只推 private，不推 origin
```

---

## 4. 用 Claude 接着做

**恢复 Claude 的记忆和会话**（Claude 每 3 分钟同步到 S3 一次）：

```bash
AWS_REGION=ap-southeast-3 $S5 sync "$S3/claude_backup/root_dot_claude/*" /root/.claude/
```

- **记忆**：在 `/root/.claude/projects/-opt-dlami-nvme-jasonleeeli-projects-UniSS/memory/`。目录名由工作目录决定，所以新机上仓库也必须在 `/opt/dlami/nvme/jasonleeeli/projects/UniSS`，记忆才会自动加载。
- **旧会话**：`88583ca0-585a-4fad-ada7-4b193b743b50`，记录已有 922 MB，`--resume` 很可能加载不了或者极慢，所以**建议开新会话**。
- 执行代理的记录在 `…/88583ca0-585a-4fad-ada7-4b193b743b50/subagents/agent-a888b5f017c96eac3.jsonl`，只作查阅。

新会话里可以直接这样说：

> 读 docs/migration/mist_v8_resume_new_server.zh-CN.md 和 docs/mist_v8_plan.zh-CN.md，按第 3 节 R1–R7 完成 MIST v8。GPU 空闲时叠加作业并行，B6 除外。

---

## 5. 不能忘的规则

- 不做 `rm` 等破坏性操作；不覆盖旧实验，新文件新建，已有输出只移动、不删除。
- 只按 PID 停自己启动的进程。
- 长作业用 `setsid nohup`。
- vLLM 和推理作业要把 `J/venvs/vllm_omni_030/bin` 放进 PATH，启动之间间隔 60 s。
- RealSI、ACL、AliMeeting 只用于测试。
- 和流畅度、延迟有关的改动不能降低 ASR-BLEU。
- 每次修改都提交；最后只推送到 `private main`。
- 任务完成后**不要启动任何 GPU 占用程序**（`holder_on.sh`、`scripts/gpu_load/*`）。
- 旧服务器上还有一个自动占卡监视进程 `scripts/gpu_load/after_claude_workload.py`（PID 1805231）。新服务器上不要启动它。
