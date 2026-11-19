# 提交 Anyscale Job

> 触发词：`提交` `submit` `跑训练` `跑 ETL` `启动` `重新提交` `重跑`

---

## 提交流程

### Step 1：确认 Job YAML 路径

如果用户未提供 `<JOB_YAML>`，询问用户要提交的 Job 配置文件路径。

`<JOB_YAML>` 是本地 yaml 文件路径，内含入口命令、环境变量等配置。

---

### Step 2：调用脚本执行预检（DRY_RUN）

调用以下命令获取预检信息，**不实际提交**：

```bash
DRY_RUN=1 bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/submit_helper.sh <JOB_YAML>
```

脚本会依次检查：
1. AWS 凭证是否可用（`aws sts get-caller-identity`）
2. Anyscale cloud 是否可见
3. Compute config 是否存在（通过 `anyscale compute-config get`）
4. Image URI 是否配置

---

### Step 3：分析预检输出的 KEY=VALUE

脚本输出为纯 KEY=VALUE 格式，逐行解析以下关键字段：

| KEY | 含义 | 异常处理 |
|---|---|---|
| `PRECHECK_OK` | 预检是否通过 | `no` → 停止，查看 `FAIL_REASON` |
| `JOB_NAME` | 自动生成的三段式任务名 | 直接使用，禁止询问 |
| `AWS_ACCOUNT_ID` | 当前 AWS 账号 | 为空 → 提示 AWS 凭证问题 |
| `COMPUTE_CONFIG` | 使用的 compute config 名称 | 为空 → 提示用户需要在 yaml 中指定 |
| `IMAGE_URI` | 使用的镜像 URI | 为空 → 提示用户需要在 yaml 中指定 |
| `FAIL_REASON` | 失败原因 | 仅在预检不通过时输出 |

**预检未通过时**，将 `FAIL_REASON` 和相关上下文报告给用户，询问如何修复，**不要继续提交**。

---

### Step 4：向用户展示提交计划，请求确认

预检通过后，**先判断用户意图是否明确**：

**意图明确的场景（跳过确认，直接执行 Step 5）：**
- 用户明确给出了 Job YAML 路径，如 `提交任务 anyscale/train/job.yaml`
- 用户使用 `bypassPermissions` 模式调用
- 用户输入是执行指令，不含疑问词（如何、能不能、是否可以、帮我看看等）

**意图不明确的场景（展示计划，等待用户确认）：**
- 用户问 `我想提交训练任务，该怎么做？`
- 用户问 `帮我看看这个配置能不能提交？`
- 用户说 `提交任务` 但未指定 YAML 路径，且对默认路径有疑问

**意图明确时**，不展示确认计划，直接执行 Step 5。

**意图不明确时**，展示以下确认计划，等待用户回复后再执行 Step 5：

```markdown
## 提交计划

以下信息来自预检结果，请确认无误后执行提交：

| 项 | 值 |
|---|---|
| **任务名** | <JOB_NAME 值> |
| **Job 配置** | <JOB_YAML> |
| **Compute Config** | <COMPUTE_CONFIG 值> |
| **镜像 URI** | <IMAGE_URI 值> |
| **AWS 账号** | <AWS_ACCOUNT_ID 值> |

**提交操作将在云端创建资源，确认提交？**

- **确认提交** — 执行正式提交
- **我要改参数** — 等待用户指定修改项后重新展示计划
```

---

### Step 5：执行正式提交

`submit_helper.sh` 支持以下参数：

```bash
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/submit_helper.sh \
    <JOB_YAML> \
    [--name <job-name>] \
    [-- <extra-anyscale-flags>...]
```

**参数说明**：

| 参数 | 作用 |
|---|---|
| `<JOB_YAML>` | Job 配置文件路径（必需） |
| `--name <job-name>` | 完全覆盖自动生成的任务名 |
| `--` 之后的参数 | 原样传给 `anyscale job submit` |

**任务名生成规则**（三段式 `<模型名>-<user>-<4位随机字符>`）：

1. 如果传了 `--name <job-name>`，直接使用该值
2. 否则，模型名按以下优先级获取：
   - `MODEL_NAME` 环境变量
   - 兜底：`etl` 或 `train`（根据 YAML 文件名判断）
3. user 来自 `git config user.name`（fallback: `whoami`）
4. 4 位随机字符来自 `/dev/urandom`

**环境变量**：所有 env 已在 `job.yaml` 的 `env_vars` 中静态配置，无需额外注入。

---

### Step 6：分析提交结果，提取 PRODJOB_ID 并汇报

脚本提交成功后输出 KEY=VALUE，解析以下字段：

| KEY | 含义 | 处理 |
|---|---|---|
| `SUBMIT_OK` | 提交是否成功 | `yes` → 继续；`no` → 查看 `FAIL_REASON`，报告给用户 |
| `PRODJOB_ID` | 提交成功的 Job ID | 提取此值，拼接 Console URL |
| `JOB_URL` | Job 的 Anyscale Console 链接 | 直接展示给用户 |
| `JOB_NAME` | 提交时使用的任务名 | 展示给用户 |
| `STATUS` | 提交后的初始状态 | 展示给用户（通常为 `STARTING` 或 `QUEUED`） |
| `FAIL_REASON` | 失败原因 | 仅在 `SUBMIT_OK=no` 时输出 |

**提交成功时**，向用户报告：

```markdown
## 提交结果

| 项 | 值 |
|---|---|
| **Prodjob ID** | <PRODJOB_ID 值> |
| **任务名** | <JOB_NAME 值> |
| **Console 链接** | <JOB_URL 值> |
| **初始状态** | <STATUS 值> |
| **Compute Config** | <COMPUTE_CONFIG 值> |
| **镜像** | <IMAGE_URI 值> |

后续操作：
- 查进度：`<PRODJOB_ID> 进度如何`
- 查日志：`<PRODJOB_ID> 日志`
- 排查失败：UI 显示 FAILED 后说 `<PRODJOB_ID> 失败了`
```

**提交失败时**，将 `FAIL_REASON` 和详细错误报告给用户。

---

## 脚本输出 KEY=VALUE 字段说明

`submit_helper.sh` 输出的所有字段（stdout，每行一个 KEY=VALUE）：

| KEY | 来源 | 说明 |
|---|---|---|
| `DRY_RUN` | 预检阶段 | `yes` 表示仅预检，未实际提交 |
| `PRECHECK_OK` | 预检阶段 | `yes`/`no` |
| `SUBMIT_OK` | 提交阶段 | `yes`/`no` |
| `PRODJOB_ID` | 提交阶段 | 成功时返回，格式 `prodjob_xxx` |
| `JOB_URL` | 提交阶段 | `https://console.anyscale.com/jobs/<PRODJOB_ID>` |
| `JOB_NAME` | 提交阶段 | 任务名 |
| `STATUS` | 提交阶段 | Job 初始状态 |
| `AWS_ACCOUNT_ID` | 预检阶段 | AWS 账号 ID |
| `AWS_USER_ARN` | 预检阶段 | AWS 用户 ARN |
| `COMPUTE_CONFIG` | 预检/提交阶段 | Compute config 名称 |
| `IMAGE_URI` | 预检/提交阶段 | 镜像 URI |
| `OUT_BUNDLE` | 始终 | 临时输出目录路径 |
| `FAIL_REASON` | 失败时 | 失败原因代码 |

---

## 注意事项

- `submit_helper.sh` 是**自包含**的：不依赖仓库内 `anyscale/` 目录下的任何文件。
- 所有环境变量已在 `job.yaml` 的 `env_vars` 中静态配置，提交时无需额外注入。
- `OUT_BUNDLE` 目录包含详细的日志文件，排查问题时可参考。
