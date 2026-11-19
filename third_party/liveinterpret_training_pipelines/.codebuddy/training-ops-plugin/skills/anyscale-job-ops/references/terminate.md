# 终止任务

> 触发词：`终止` `kill` `cancel` `terminate` `停掉` `停止任务`
> ⚠️ **终止操作变更云端状态，需用户二次确认**

---

## 命令模板

### 终止任务

```bash
anyscale job terminate --id <PRODJOB_ID>
```

### 验证终止状态

```bash
anyscale job status --id <PRODJOB_ID>
```

---

## 二次确认模板

终止任务后无法恢复，请与用户确认：

```json
{
  "questions": [{
    "question": "确认终止任务 <PRODJOB_ID>？\n\n终止后：\n- 训练状态将丢失\n- 云端资源会被释放\n- 此操作无法撤销",
    "header": "确认终止",
    "multiSelect": false,
    "options": [
      {"label": "确认终止", "description": "执行 anyscale job terminate 命令"},
      {"label": "取消", "description": "不执行任何操作"}
    ]
  }]
}
```

| 选择 | 操作 |
|---|---|
| 确认终止 | 执行终止命令 |
| 取消 | 停止操作 |

---

## 执行步骤

1. 提取任务 ID（从用户输入或上下文）
2. 展示二次确认模板，等待用户确认
3. 用户确认后，执行终止命令
4. 验证终止状态

---

## 注意事项

- 终止操作会变更云端状态，执行前必须获得用户明确确认
- 终止后无法恢复任务执行
- 建议在终止前确认是否需要保存检查点或其他数据
- 终止后资源会在数分钟内释放
