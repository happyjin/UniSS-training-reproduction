# KtrainJob 失败诊断

> 触发词：`失败` `FAILED` `挂了` `报错` `error` `traceback` `日志` `排查` `诊断`

## 诊断纪律

- 一律先把**全量**状态与日志下载到本地再分析；禁止用 grep 过滤后的片段下结论，禁止先射箭再画靶。
- 根因未形成闭环证据链（时间线、首发失败点、上游真实错误 vs 下游表象）之前，禁止提出任何绕过 / 修复方案。

## Step 1 — 确定任务名与命名空间

同 `progress.md` Step 1。

## Step 2 — 收集全量证据

```bash
bash <SKILL_DIR>/scripts/collect_logs.sh <JOB_NAME> <NAMESPACE>
```

产物落在 `/tmp/ktrain-collect/<JOB_NAME>/`（`status.txt` / `logs.txt` / `summary.txt`），stdout 输出 `COLLECT_OK` / `JOB_STATE` / `LOG_LINES` / `OUT_BUNDLE`。

`ktrain job logs` 默认只取尾部 100 行（`--tail 100`），collect_logs.sh 内部已自动追加 `--tail 5000` 重试。若仍不够，用 `--tail` 指定更大值：

```bash
ktrain job logs <JOB_NAME> --namespace <NAMESPACE> --tail -1 > /tmp/<JOB_NAME>-full.log 2>&1
```

指定某个 worker 的日志：

```bash
ktrain job logs <JOB_NAME> --namespace <NAMESPACE> --worker <POD名> --tail -1
```

已结束任务的持久化日志用 `--offset`（0 起始行偏移）向前翻页，对 running 任务的容器日志无效：

```bash
ktrain job logs <JOB_NAME> --namespace <NAMESPACE> --tail 2000 --offset 2000
```

机器可读状态用全局 `-o json`：`ktrain -o json job status <JOB_NAME> --namespace <NAMESPACE>`。

`COLLECT_OK=no` 时（ktrain CLI 不可达或任务不存在），退到 kubectl 直拉（ktrain CLI 不覆盖此场景）：

```bash
kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide
kubectl -n <NAMESPACE> logs <POD> --all-containers --tail=-1 > /tmp/<JOB_NAME>-<POD>.log 2>&1
```

## Step 3 — 本地分析

1. 任务状态与 Pod 列表从 `ktrain job status` 输出读取（含每 Pod 状态与重启次数）。ktrain 不覆盖时用 `kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide`。
2. 日志末尾 60 行：任务怎么死的。
3. 首个 `Traceback` / 首个 `ERROR` 前后各 60 行：根因。
4. 多 Pod 时按时间线对齐，找**首发**失败 Pod；其余 Pod 的 `Connection refused` / `NCCL abort` 是次生级联，不是根因。
5. 按下表归类，输出证据行（带行号引用本地日志文件）。

## 通用失败类别

| 日志关键字 | 类别 | 排查方向 |
|---|---|---|
| `CUDA out of memory` | 单卡显存不足 | 首发 OOM 的 rank；batch / 序列长度 / 并行切分 |
| `Connection refused` / `No space left on device` | 次生级联 | 回时间线找首发失败 rank，忽略本表象 |
| `RuntimeError` / 张量维度不匹配 | 训练代码 / 数据 | 首个 Traceback 指向的代码行；交还用户改代码 |
| `No ... backend is available` / attention 后端报错 | 注意力后端 | 后端与 mask 形式的匹配 |
| `Cannot copy out of meta tensor` | 模型未正确加载 | 模型文件是否在节点就位（查提交时同步日志） |
| `unexpected EOF` / arrow 读失败 | 节点本地缓存损坏 | 节点上数据集缓存目录是否有截断文件 |
| `/dev/shm` 相关 NCCL 报错 | 共享内存不足 | 任务 yaml 的 shm 配置 |
| 跨节点 `NCCL abort` / EFA 报错 | 网络 / EFA | EFA 设备与插件、节点间连通性 |
| 同步 Job 失败（提交阶段） | 数据未就绪 | submit 输出的 sync Pod 日志；S3 源路径 / 凭证 |

## Step 4 — 输出模板

```markdown
## 诊断结论 — <JOB_NAME>

| 项 | 值 |
|---|---|
| **Pod 状态** | <每 Pod STATUS / RESTARTS> |
| **失败阶段** | <提交同步期 / 调度期 / 启动期 / 训练中> |
| **根因** | <一句话> |
| **首发证据** | <本地日志文件:行号 + 关键行> |
| **次生表象** | <级联报错列表，标注"非根因"> |
| **时间线** | <关键事件按时序排列> |

### 修复建议
<基于根因的具体步骤；涉及训练代码 / 超参的改动交还用户，不代改>
```

## 禁止事项

- 禁止 cherry-pick 日志片段作为结论依据。
- 禁止在 Pod 内执行云 CLI 排查对象存储权限；权限问题查 CSI 卷 / ServiceAccount。
- 禁止以换节点、改白名单作为修复手段。
- 训练代码 / 超参的修改不属于本 skill，诊断到此为止。
