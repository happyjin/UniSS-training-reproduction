# training-ops 插件

训练平台运维插件，覆盖 **Anyscale / ktrain / Kubernetes PyTorchJob** 三类平台的训练任务全生命周期运维：
提交、进度查询、日志诊断、节点资源排查、代码/模型/数据文件同步、终止清理。

## 组件

### Agent（限定 hy3 模型）
- `agents/training-ops.md` — `training-ops` agent，固定使用 **hy3** 模型。
  当用户消息涉及 anyscale / prodjob / ktrain / PyTorchJob / 训练任务 / 节点健康 / 训练日志 /
  提交训练 / 终止任务 / 同步代码模型数据 时，主线程会委派本 agent 执行。
  它会根据意图加载并遵循下面三个 skill 的流程与输出模板。

### Skills（命名空间 `training-ops`）
| Skill | 平台 | 能力 |
|---|---|---|
| `anyscale-job-ops` | Anyscale (Ray prodjob) | 提交 / 进度 / 节点健康 / 失败诊断 / 终止（由原有 `.claude/skills/anyscale-job-diagnose` 重命名迁移而来） |
| `ktrain-job-ops` | ktrain (PyTorchJob, L4) | 任务操作 / 文件同步 / 日志查询 / 资源管理 |
| `pytorchjob-ops` | 纯 kubectl PyTorchJob (H200) | 任务操作 / 日志查询 / 资源管理（节点白名单、重复 -209 job） |

每个 skill 含 `SKILL.md` + `references/`（分场景流程与输出模板）+ `scripts/`（自包含诊断脚本，输出 KEY=VALUE 结构化字段）。

## 目录结构
```
.codebuddy/training-ops-plugin/
├── .codebuddy-plugin/plugin.json     # 插件清单
├── agents/training-ops.md            # hy3 agent
└── skills/
    ├── anyscale-job-ops/   {SKILL.md, references/, scripts/}
    ├── ktrain-job-ops/     {SKILL.md, references/, scripts/}
    └── pytorchjob-ops/     {SKILL.md, references/, scripts/}
```

## 本地测试

```bash
# 从仓库根目录加载插件（脚本路径以 .codebuddy/training-ops-plugin/ 前缀引用）
codebuddy --plugin-dir ./.codebuddy/training-ops-plugin

# 校验插件格式
codebuddy plugin validate ./.codebuddy/training-ops-plugin

# 热重载
/reload-plugins

# 查看已注册 agent / skill
/agents
```

## 触发示例
- "提交 anyscale 训练任务 `anyscale/train/job.yaml`" → 委派 training-ops agent → anyscale-job-ops 提交流程
- "ktrain 集群 L4 节点有哪些、有 NVMe 吗" → ktrain-job-ops 节点排查
- "qwen3-omnispeak-bridge-align-h200 进度如何" → pytorchjob-ops 进度查询
- "speech-h200 节点白名单是否正常、有没有重复 -209 job" → pytorchjob-ops 节点健康

## 约束
- 所有诊断信息通过 `anyscale` / `ktrain` / `kubectl` / `aws` 远程获取，不读本地仓库文件作为诊断依据。
- 变更类操作（提交、终止、节点恢复、覆盖式 sync）执行前需二次确认，除非用户给出明确执行指令。
- skill 脚本内的命令需在具备对应 CLI（anyscale / ktrain / kubectl / aws）与集群凭证的环境运行。
