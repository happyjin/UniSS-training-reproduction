# 终止 / 清理 PyTorchJob

> 触发词：`终止` `kill` `cancel` `删` `清理` `delete` `teardown`
> ⚠️ **删除操作变更集群状态，需用户二次确认**

---

## 清理方式

### 方式 A：按项目清单删除整个 PyTorchJob（推荐）

```bash
kubectl -n <NAMESPACE> delete -f <pytorchjob.yaml 渲染后的清单>
```

### 方式 B：按名删除

```bash
kubectl -n <NAMESPACE> delete pytorchjob <JOB_NAME>
```

### 卡死 pod 强制删除

```bash
kubectl -n <NAMESPACE> delete pod <JOB_NAME>-worker-0 --force --grace-period=0
```

### 清理重复 / 抢 GPU 的 job

某些平台存在外部自动化会**重建带后缀的重复 PyTorchJob**（与主 job 同名前缀，争抢 GPU）。排查并清理：

```bash
kubectl -n <NAMESPACE> get pytorchjob
# 若发现多个同名前缀的 job，清理重复实例（保留用户想要的主 job）
kubectl -n <NAMESPACE> delete pytorchjob <重复 JOB_NAME>
```

---

## 二次确认模板

```json
{
  "questions": [{
    "question": "确认清理 PyTorchJob <JOB_NAME>（ns <NAMESPACE>）？\n\n清理后：\n- 训练状态将丢失\n- GPU 资源释放\n- 此操作无法撤销（PVC / 代码 / checkpoint 存储不受影响）",
    "header": "确认清理",
    "multiSelect": false,
    "options": [
      {"label": "确认清理", "description": "执行 kubectl delete pytorchjob"},
      {"label": "取消", "description": "不执行任何操作"}
    ]
  }]
}
```

---

## 执行步骤

1. 确认 PyTorchJob 名（从 `kubectl -n <NAMESPACE> get pytorchjob` 读取，或用户提供）。
2. 展示二次确认，等待确认。
3. 执行删除。
4. 验证：`kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME>`。

---

## 注意事项

- 删除 PyTorchJob 会释放 GPU；PVC 与对象存储（S3 / FSx 等）上的代码 / 模型 / checkpoint 不受影响。
- 若出现重复 job，仅删除实例不够，需排查外部 controller / 自动重试根源。
- 终止前确认输出 / checkpoint 存储上的数据是否需保留。
