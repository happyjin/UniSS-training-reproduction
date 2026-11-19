# PyTorchJob 失败诊断

> 触发词：`失败` `FAILED` `挂了` `排查` `报错` `error` `traceback` `日志`

---

## ⚠️ 诊断纪律（通用，强烈建议遵守）

一律**下载全量日志到本地**分析，**禁止用命令过滤 cherry-pick 日志、禁止先射箭再画靶**。

```bash
# 下载全量日志到本地（master + 各 worker）
K8S_NS=<NAMESPACE> bash <skill>/scripts/collect_logs.sh <JOB_NAME> <NAMESPACE>
# 日志落在 /tmp/pytorchjob-collect/<JOB_NAME>/
```

或用 kubectl 直接下载：

```bash
kubectl -n <NAMESPACE> logs <JOB_NAME>-master-0 -c pytorch > /tmp/<JOB_NAME>-master.log 2>&1
kubectl -n <NAMESPACE> logs <JOB_NAME>-worker-0 -c pytorch > /tmp/<JOB_NAME>-worker.log 2>&1
```

---

## 通用失败类别（按现象归类，不绑定具体项目代码）

| 现象（日志关键字） | 通用根因方向 | 排查 / 修复方向 |
|---|---|---|
| `CUDA out of memory` | 单卡显存不足（激活 / 序列过长 / 未并行切分） | 减小 batch / 序列长度；对相关前向加 activation checkpoint；用 TP/PP 切分；确认未遗漏重算 |
| 分布式通信 `Connection refused` / `No space left on device` | 通常是某 rank **死后**的**次生级联**，非首发根因 | 先定位首发失败 rank 的根因（多为 OOM / 代码错误）；必要时强制删卡死 pod 后重提 |
| `RuntimeError: ... ambiguous` / 张量维度不匹配 | 代码对 Tensor 误用标量语义、标签长度与输入不对齐 | 改显式 None 判断；对齐 labels 长度（逐样本 padding 破坏对齐时关掉 padding-free） |
| 自定义注意力 / 后端 `No ... backend is available` | TE/flash 注意力不接受自定义 mask | 切到 unfused 后端或调整 mask 形式 |
| `Dynamo failed to run FX node` / torch.compile 失败 | 动态 shape / 动态分组 GEMM 追踪失败 | 关闭 torch.compile（`TORCH_COMPILE_DISABLE=1` 等） |
| `Cannot copy out of meta tensor` | 模型/配置未正确加载到设备 | 确认模型文件已就位、配置类存在；启动脚本正确 cp 到本地 |
| `/dev/shm` 相关 NCCL 报错 | 共享内存未挂 emptyDir | yaml 挂 `emptyDir` Memory |
| 跨节点 NCCL abort | 网络插件名不匹配 / EFA 配置 | 启动脚本建立所需符号链接或修正插件名 |

> 上述为**通用方向**。具体修复要结合项目代码与日志，属于项目排错史，不在本 skill 范围。

---

## 诊断流程

1. 收集全量日志（master + 各 worker）到本地。
2. 看 pod 状态：`kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide`（Restart 数、节点、状态）。
3. 末尾 60 行看任务怎么死的；首个 Traceback 前后 60 行看根因。
4. 按上表匹配通用类别，套用排查方向；未知模式深入日志。
5. 区分**根因 vs 次生级联**：某 rank OOM → 通信 refused 是级联，先修 OOM 而非网络。

---

## 输出模板

```markdown
## 诊断结论 — <JOB_NAME>

| 项 | 值 |
|---|---|
| **Pod 状态** | master-0 <Running/Error>, worker-0 <...>, Restarts=<N> |
| **失败阶段** | <OOM / 配置 / 网络级联> |
| **根因** | <描述> |
| **是否次生级联** | <是(通信 refused)/否> |
| **证据** | <关键日志行> |

### 修复建议
<具体步骤>
```

---

## 注意事项

1. 下载全量日志本地分析，禁止 cherry-pick。
2. Pod 若无云凭证，排查对象存储权限看 CSI 而非 pod 内云 CLI。
3. 节点白名单固定，worker 崩溃首选查训练侧根因（如 OOM）而非换节点。
