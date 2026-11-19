# 停止 / 删除 KtrainJob

> 触发词：`停止` `stop` `停掉` `暂停` `释放资源` `kill` `cancel` `清理` `删除任务`

## 规则

- 默认只做**停止（stop）**：Pod 删除、GPU 释放，任务对象与日志保留。
- **删除（delete）不是常规清理手段**，仅当用户明确说「删除任务对象」时才执行。
- 用户指令含明确任务名时直接执行，不二次确认；任务名缺失或有歧义时先问清楚。

## Step 1 — 确定任务名与命名空间

- `<JOB_NAME>`：从用户输入提取；未提供时列出候选让用户选：`ktrain job list --namespace <NAMESPACE> --all`
- `<NAMESPACE>`：用户输入 > 任务 yaml `metadata.namespace` > 提交时汇报记录。ktrain / kubectl 不传 `--namespace` 会用 `default`，必须显式传。

## Step 2 — 执行停止（ktrain CLI ≥ 1.7 已覆盖）

`ktrain job stop` 释放资源（删 Pod）、保留任务记录与日志；server 端处理，无 `--framework` 参数：

```bash
ktrain job stop <JOB_NAME> --namespace <NAMESPACE> --force
```

兜底（CLI 低于 1.7 或 `job stop` 报错时）：停止作用于 operator 渲染出的**同名** PyTorchJob，用 kubectl patch：

```bash
kubectl -n <NAMESPACE> patch pytorchjob <JOB_NAME> --type merge \
  -p '{"spec":{"runPolicy":{"stop":true}}}'
```

## Step 3 — 验证

```bash
# 任务状态（ktrain CLI）
ktrain job status <JOB_NAME> --namespace <NAMESPACE>

# stop 字段是否生效（ktrain CLI 不暴露此字段，用 kubectl）
kubectl -n <NAMESPACE> get pytorchjob <JOB_NAME> -o jsonpath='{.spec.runPolicy.stop}{"\n"}'   # 应为 true

# Pod 是否退出（ktrain CLI 不提供 Pod 列表，用 kubectl）
kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME>
```

## 恢复（ktrain CLI 无 resume，用 kubectl）

```bash
kubectl -n <NAMESPACE> patch pytorchjob <JOB_NAME> --type merge \
  -p '{"spec":{"runPolicy":{"stop":false}}}'
```

## 异常处理

- stop 后 Pod 卡 `Terminating`：
  ```bash
  kubectl -n <NAMESPACE> delete pod <POD名> --force --grace-period=0
  ```
- 对 KtrainJob 重新 submit / 编辑后 operator 重渲染可能把 `stop` 重置为 `false` 导致任务重新拉起；发现被重置就再执行一次停止（`ktrain job stop` 或 patch `stop=true`）。

## 彻底删除（仅用户明确要求时）

```bash
ktrain job delete <JOB_NAME> --namespace <NAMESPACE> --force
```

删除移除任务对象及状态历史，不可恢复；仅执行用户明确要求的删除指令。
