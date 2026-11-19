# 任务诊断流程指南

本文档描述诊断失败任务的完整流程和分析逻辑。

## 诊断流程

### Step 1: 从用户输入提取任务标识符

从用户消息中识别并提取 `<PRODJOB_ID>`（任务唯一标识符）。

### Step 2: 收集日志和状态

调用日志收集脚本：

```bash
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/collect_logs.sh <PRODJOB_ID>
```

**分析脚本输出**：
- 确认日志是否成功获取（`LOGS_COLLECTED=true/false`）
- 如果 `LOGS_COLLECTED=false`，记录错误信息并考虑是否需要手动检查

### Step 3: 定位失败

调用失败定位脚本：

```bash
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/locate_failure.sh <PRODJOB_ID>
```

**分析脚本输出的 KEY=VALUE**：

| 键值 | 说明 |
|------|------|
| `FAILURE_STAGE` | 失败发生的阶段（提交、集群启动、初始化、运行等） |
| `FAILURE_CATEGORY` | 失败分类（A: 配置错误、B: 资源问题、C: 代码错误） |
| `FIRST_ERROR` | 日志中第一个错误信息 |

### Step 4: 匹配失败模式

查阅失败模式速查表：

```bash
# 参考文件
.codebuddy/training-ops-plugin/skills/anyscale-job-ops/references/failure_patterns.md
```

根据 `FAILURE_STAGE` 和日志关键字在速查表中匹配已知失败模式。

### Step 5: 综合分析和修复建议

结合以下信息源进行根因分析：
1. `collect_logs.sh` 输出的日志内容
2. `locate_failure.sh` 输出的 `FAILURE_STAGE`、`FAILURE_CATEGORY`、`FIRST_ERROR`
3. `failure_patterns.md` 中匹配的模式和修复建议

**如果匹配到已知模式**：
- 直接应用速查表中的修复步骤
- 提供具体的配置或代码修改建议

**如果是未知模式**：
- 提取更多上下文信息（如调用 `inspect_p5_nodes.sh` 检查集群状态）
- 分析日志中的异常模式
- 提供通用的排查方向

### Step 6: 向用户汇报

生成结构化诊断报告，包含：
- 失败阶段
- 根因分析
- 修复步骤

## 分析逻辑

### 问题严重程度判断

根据 `FAILURE_STAGE` 和 `FAILURE_CATEGORY` 判断问题严重程度：

| FAILURE_CATEGORY | 严重程度 | 说明 |
|------------------|----------|------|
| A (配置错误) | 高 | 配置问题导致任务无法启动，需立即修复 |
| B (资源问题) | 中 | 资源不足或调度失败，可能需要调整配置 |
| C (代码错误) | 低 | 应用代码问题，需修复业务逻辑 |

### 使用失败模式速查表

`failure_patterns.md` 包含常见失败模式的三列信息：
1. **Grep 模式** - 日志中搜索的关键字或正则表达式
2. **根因** - 导致失败的根本原因
3. **修复方向** - 具体的修复步骤或建议

**匹配流程**：
1. 提取 `FIRST_ERROR` 和日志关键段落
2. 在 `failure_patterns.md` 中搜索匹配的 Grep 模式
3. 如果找到匹配，应用对应的修复方向
4. 如果未找到匹配，记录为未知模式并深入日志分析

### 未知模式处理

当遇到未知失败模式时：

1. **检查集群状态**：
   ```bash
   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh <PRODJOB_ID>
   ```

2. **提取更多日志上下文**：
   - 查看错误发生前后的日志行
   - 检查是否有权限、网络、存储相关错误
   - 分析是否有级联 failure（一个错误导致后续错误）

3. **通用排查方向**：
   - 配置检查：环境变量、密钥、权限
   - 资源检查：CPU、内存、GPU、网络
   - 依赖检查：镜像、Python 包、系统库
   - 代码检查：应用日志、异常处理

## 诊断报告模板

```markdown
## 诊断结论 — `<PRODJOB_ID>`

| 项 | 值 |
|---|---|
| **失败阶段** | `<FAILURE_STAGE>` |
| **失败分类** | `<FAILURE_CATEGORY>` |
| **根因** | `<根因描述>` |
| **证据** | `<关键日志行>` |

### 错误原文

```
<日志中的错误片段>
```

### 修复建议

<具体修复步骤>

### 预防措施

<如何避免类似问题>
```

## 注意事项

1. 所有诊断操作应通过提供的脚本完成，避免直接调用底层 CLI
2. 日志分析应专注于通用错误模式，而非项目特定路径
3. 修复建议应针对根因而非症状
4. 对于复杂问题，可能需要多次迭代诊断和修复
5. 如果脚本执行失败，应记录错误信息并考虑替代诊断路径
