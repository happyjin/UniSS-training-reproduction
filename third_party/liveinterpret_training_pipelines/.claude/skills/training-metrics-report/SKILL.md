---
name: training-metrics-report
description: 从 ktrain 训练任务（deepspeed/megatron 后端，单机 / ray / pytorch 直连多机）拉取 HF Trainer logging 训练指标（loss/acc/grad_norm/s_it/显存/耗时），解析成结构化数据，并写入/更新 A/B/C/... 多列对照 Markdown 文档（如 GBS4096_LIGER_200STEP_TRAINING_METRICS.md 这类逐 step 对照报告）。默认不采集/不使用 mfu_log.jsonl 里的 MFU/TFLOPs 数据（计算方法未验证，不作为指标）。当用户要求"拉一下这个任务的结果""跑到 200 步了记录一下""把这一列加进对照文档""看看是不是 OOM 了"时使用。
---

# training-metrics-report

把一次训练任务的运行结果（逐 step 的 loss/acc/grad_norm/s_it/显存/耗时）从远程拉下来、
解析成干净的结构化数据，再整理成 Markdown 多列对照表里的一整列（新增列 or 更新已有列），
并补齐结论段落与采集历史记录。**默认不采集/不使用 `mfu_log.jsonl`**（见下方 §0）。

`SKILL_DIR` = 本 SKILL.md 所在目录。脚本调用写法：`bash <SKILL_DIR>/scripts/<脚本>` /
`python3 <SKILL_DIR>/scripts/<脚本>`。

与 `ktrain-job-ops` skill 的边界：提交 / 停止 / 删除 / 节点排查 / 代码同步一律用
`ktrain-job-ops`；本 skill 只管"任务跑起来之后，怎么把结果变成文档里的一列"，前置的
`ktrain job status` / `ktrain job stop --force` 仍按 `ktrain-job-ops` 的规范调用。

## 何时触发

- 用户说"看一下这个任务跑到多少步了""拉一下结果""记录一下""写进文档"。
- 用户给了一个多列对照文档路径（如 `docs/*_200STEP_TRAINING_METRICS.md`），要求把某个任务的
  结果加进去成为新的一列（或更新已有列）。
- 用户问"是不是 OOM 了""stall 了吗""这个任务数值对不对"——本质是同一套日志拉取/解析流程。

## ⚠️ 0. 不采集 / 不使用 `mfu_log.jsonl`

**默认不拉取、不解析、不填写 MFU/TFLOPs/tokens_per_sec 相关字段**，跟
`training-optimization` skill §0 的结论一致：MFU 依赖的"理论 FLOPs"分母
是项目自己手算/估算出来的公式，容易在具体实现里悄悄算错而不报错，比
s/it 这种直接测量的墙钟耗时危险得多。本 skill 的默认核心指标是
**s/it（HF Trainer 的 `train_speed(s/it)`，累计口径）、显存(GiB)、
elapsed/remaining_time**，不依赖 `mfu_log.jsonl`。

`fetch_logs.sh`/`parse_metrics.py` 仍保留 `mfu_log.jsonl` 的拉取/解析能力
（`--mfu-log` 参数），**但只有用户明确要求"顺便看一下 MFU/TFLOPs"时才使用**，
默认流程（Step 1-4）不传 `OUTPUT_DIR`/`--mfu-log`。文档里 1.2 表格的稳态
MFU/TFLOPs/tokens_per_s 单元格默认填**"不采用"**（不是"未采集"——"不采用"
明确表示这是本 skill 的方法论选择，不是碰巧没采集到）。

## 核心流程（4 步）

### Step 1 — 确认 Pod 名与拉取方式（pytorch 直连 vs ray 两种模式不同）

先判断任务的 `framework`（从任务 yaml 的 `spec.framework` 或 `ktrain job status` 输出的
`framework` 字段确认）：

- **`framework: pytorch`**（torchrun 直连，含单机/多机）：主日志 pod 是
  `<JOB_NAME>-master-0`。`kubectl logs -n <NS> <JOB_NAME>-master-0` 即可拿到该 pod
  自己的完整 HF Trainer logging 输出（每个 rank 的日志分散在各自的 pod，通常只看
  `master-0` 即够用于统计，因为 HF Trainer 的 logging 只在 rank0 打印）。
- **`framework: ray`**（KubeRay RayJob）：**不要**去 head/worker pod 找日志——那些
  pod 里是 ray 集群自身的启动信息，不含训练输出。训练输出全部聚合在**任务提交器 pod**
  （名字形如 `<JOB_NAME>-<hash5位>`，**没有** `-head-` 或 `-worker-` 后缀，用
  `kubectl get pods -n <NS> | grep <JOB_NAME>` 排除 head/worker 即可定位），日志里每行
  前缀带 `(_driver pid=..., ip=...)` 或 `(SwiftSft pid=..., ip=...)`，本质是所有 actor
  日志的聚合流，直接 `kubectl logs -n <NS> <提交器pod>` 拉取即可。

两种模式的 HF Trainer logging 格式完全一致（都是 `{'loss': ..., ...}` 这种 python dict
字面量），差异只在"去哪个 pod 拉日志"。

### Step 2 — 拉取原始数据到本地

```bash
bash <SKILL_DIR>/scripts/fetch_logs.sh <JOB_NAME> <NAMESPACE> <POD_NAME>
```

- `POD_NAME`：Step 1 判断出的日志 pod（pytorch 用 `master-0`，ray 用提交器 pod）。
- **默认不传 `OUTPUT_DIR`/`ANY_POD_FOR_EXEC`**——按本文件 §0，默认不拉取 `mfu_log.jsonl`。
  只有用户明确要求"顺便看一下 MFU/TFLOPs/tokens_per_sec"时，才补上这两个参数（`OUTPUT_DIR`
  = 任务 yaml `spec.env.OUTPUT_DIR` 的值；`ANY_POD_FOR_EXEC` = 任一还在运行的训练 pod）。

脚本输出到 `/tmp/training-metrics-report/<JOB_NAME>/`：`raw_log.txt`（完整 `kubectl logs`
输出）、`mfu_log.jsonl`（若拉取）。全程只读，不修改远程任何状态。

### Step 3 — 解析成结构化 JSON

```bash
python3 <SKILL_DIR>/scripts/parse_metrics.py \
  --raw-log /tmp/training-metrics-report/<JOB_NAME>/raw_log.txt \
  [--steps 1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] \
  > /tmp/training-metrics-report/<JOB_NAME>/report.json
```

**默认不传 `--mfu-log`**（见 §0）。脚本自动：

1. 从 `raw_log.txt` 用正则提取所有 HF Trainer logging 字典行，按 `global_step/max_steps`
   （deepspeed 链路）或 `iteration`（megatron 链路）字段取出 step 号，**同 step 只保留最后
   一次出现的记录**（覆盖任务被停止后重新提交、日志里出现多段训练的情况）。
2. 自动识别字段口径（deepspeed 输出 `token_acc/token_text_acc/token_acoustic_token`；
   megatron 输出 `codec_loss/codec_token_acc/codec_residual_loss`，二者互斥，脚本按实际
   出现的字段名自动判断，不需要手动指定后端）。
3. 若某个 step 没有采集到记录（比如任务在中途 OOM，没跑到 200 步），对应字段留空
   （JSON 里为 `null`），不臆造数值。

（`--mfu-log`/`--steady-window` 参数仍存在，仅用户明确要求看 MFU 时才加上，行为见
`references/field_mapping.md` 的 mfu_log 章节。）

`report.json` 结构详见 `references/field_mapping.md`；出现 OOM / 崩溃时的处理方式见
`references/markdown_table_guide.md` 的"负面结果记录"章节（**不要**把没跑完的残缺数据
硬凑成表格的一整列）。

### Step 4 — 写入 / 更新 Markdown 对照文档

参照 `references/markdown_table_guide.md`：先读一遍目标文档，确认现有列结构（表头顺序、
已有几列），再用 `report.json` 里的数据逐节补齐（1.1 配置表 / 1.2 性能表 / 1.3.x 逐指标表 /
2.1 数学结果 / 2.2 效率结论 / 2.3 结论小结 / 3 采集历史），**新增列一律追加在表格最右侧**，
不要打乱已有列的顺序。每个 `replace_in_file` 优先合并相邻（20 行内）的表格编辑，减少调用
次数（本 skill 见过的真实案例：七八列 × 十几张表格 × 逐行编辑，容易因为单次编辑范围太大
超时——宁可拆成多次中等大小的编辑，也不要一次塞进整份文档）。

## 关键踩坑记录

- **ray 模式绝对不要去 head/worker pod 找训练日志**——那里只有 ray 集群自身日志，训练输出
  在提交器 pod。
- **`mfu_log.jsonl` 默认不用**（见 §0）；若用户明确要求才拉取解析，注意它是追加写入的：
  如果同一个任务被停止后用**同一个 `OUTPUT_DIR`** 重新提交过，文件里会有多段"step 从 1
  开始"的历史，必须只取最后一段，否则统计会把两次不同运行的数据混在一起。
- **RayJob 名称长度上限 47 字符**：ray 任务命名要提前数字符，超限会在 `ktrain job submit`
  后直接报 `RayJob metadata is invalid` 且状态显示 `Unknown`。
- **同为单机 DP8+liger，deepspeed 与 megatron 的字段名不同**：不要把 `codec_token_acc`
  和 `token_acc` 当成同一个数值直接比较（口径不同，只能看趋势），写文档时要显式标注。
- **稳态 MFU/TFLOPs/tokens_per_sec 字段默认在表格里写"不采用"**（不是"未采集"，见 §0），
  不要用 HF Trainer 的累计 `train_speed(s/it)` 冒充稳态瞬时值（口径不同，会误导后续对比）。
- **任务失败（OOM/崩溃）时**：只在结论段落和采集历史里记录负面结果，不要为了凑列而在正式
  对照表格里塞入不完整/无效的数据行。
