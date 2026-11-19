# CODEBUDDY.md

**强制规则：你的所有 agent 的思维链、响应、回复、提问都必须是中文，专业术语除外。**
**强制规则：禁止将所有文档改写成英文，除非用户明确指定。**
**强制规则：排查 / 诊断问题时，必须先收集确凿证据，把根因和完整来龙去脉（时间线、谁触发谁、上游真实错误 vs 下游表象、端口 / 进程状态）查清并形成闭环证据链；在原因未查清之前，不得提出或尝试任何绕过（workaround）、修复、优化方案。**
**强制规则：所有训练参数的设置，禁止更新或新增到任务 yaml（`pytorch_job/*.yaml`）中，必须写到训练脚本里。详见下方「训练参数归属规则」。**
**强制规则：尽量阅读更多的信息后再动手——不凭记忆或表面现象臆测，先把相关代码、日志、配置、历史文档读全再下结论或改动。**
**强制规则：尽量做最少的代码改动和变动——只改完成当前任务所必需的最小集合，不做顺手重构、不顺带改无关文件。**
**强制规则：禁止将注释当作 changelog 编写至膨胀——注释只写「为什么 / 约束 / 前置条件」这类必要说明，不记录修改历史、版本更替、谁在何时改了什么。**

## 训练参数归属规则

训练参数与运行环境必须严格分层，**单一事实源不得跨层**：

| 层 | 载体 | 只放什么 |
| --- | --- | --- |
| **训练参数层** | `pipelines/bridge_alignment/common_args_megatron.sh`、各 `train_*_megatron.sh` | 所有影响训练行为与收敛的参数 |
| **运行环境层** | `pytorch_job/*.yaml` | 硬件与环境，**不含任何训练超参** |

### 训练参数层（唯一允许修改处）
`SCENE` 预设及并行度（`TP`/`PP`/`CP`/`EP`/`SEQUENCE_PARALLEL`）、batch（`MICRO_BATCH_SIZE`/`GLOBAL_BATCH_SIZE`）、
`MAX_LENGTH`、`LR`/`MIN_LR`/`LR_WARMUP_FRACTION`/`NUM_TRAIN_EPOCHS`、
`RECOMPUTE_*`、`DATALOADER_*`、`DATASET_NUM_PROC`、`SPLIT_DATASET_RATIO`、
`EVAL_STEPS`/`SAVE_STEPS`/`LOGGING_STEPS`/`SAVE_TOTAL_LIMIT`、
`ATTENTION_BACKEND`/`PACKING`/`PADDING_FREE`、freeze 策略、loss 权重等。

### 运行环境层（任务 yaml 允许放的东西）
镜像、命名空间、节点白名单 / `affinity`、`resources`、卷与挂载、`initContainers`、
网络与 NCCL env（`NCCL_*`）、工件路径（`MODEL_PATH`/`QWEN3_TTS_TOKENIZER`/`BRIDGE_ALIGNMENT_DATA`/
`ARTIFACT_AUDIO_SHARDS`/`NVME_MOUNT`/`FSX_MOUNT`）、任务身份（`TRAIN_SCRIPT`/`OUTPUT_DIR`/
`WANDB_PROJECT`/`WANDB_EXP_NAME`/`WANDB_MODE`）。

### 约定
1. **禁止**在任务 yaml 里用 env 覆盖任何训练超参——即使只是临时试验，也要改训练脚本。
2. 单 / 双节点等不同规模需要不同超参时，在训练脚本内用 `SCENE` 预设，或按 `WORLD_SIZE`/`NNODES`/
   `NPROC_PER_NODE` 派生，**不得**在 yaml 侧分别覆盖。
3. 改动训练脚本里的默认值时，必须同步更新其上方注释里的实测依据（如显存实测值、OOM 边界），
   避免出现"预设值与注释自相矛盾"。

## 训练平台运维路由规则

当 agent 涉及以下训练平台 / 任务的运维操作时，**必须优先加载并使用对应的 skill**（来自 `.codebuddy/training-ops-plugin` 插件），不得凭空臆测命令或直接使用本地仓库脚本作为诊断依据：

| 操作涉及的平台 / 关键词 | 启用的 skill | 覆盖范围 |
| --- | --- | --- |
| Anyscale / Ray / prodjob | `training-ops:anyscale-job-ops` | 任务提交、进度查询、节点健康、失败诊断、终止 |
| ktrain / Kubeflow PyTorchJob（nlp-h100 / L4） | `training-ops:ktrain-job-ops` | 任务操作、文件同步、日志查询、资源管理 |
| Kubernetes PyTorchJob / H200 / 纯 kubectl（speech-h200-eks） | `training-ops:pytorchjob-ops` | 任务操作、日志查询、资源管理 |

### 触发条件
只要用户消息包含以下任一意图，即启用对应 skill：
- 提交 / 启动 / 终止 / 查询 / 诊断一个训练任务（anyscale、ktrain、pytorchjob、job、prodjob）
- 节点健康排查、资源（GPU/节点）管理
- 训练日志拉取与失败分析
- 代码 / 模型 / 数据在 ktrain 上的同步

### 执行约定
1. 路由到 skill 后，严格遵循对应 `SKILL.md` 中的意图路由表、执行流程与输出模板。
2. 所有远程诊断信息必须通过 CLI（`anyscale` / `ktrain` / `kubectl` / `aws`）远程获取，仓库内脚本仅作为执行入口调用，不得当作诊断依据。
3. 变更类操作（提交、终止、节点恢复、覆盖式同步）执行前必须向用户二次确认，除非用户已给出明确执行指令。
4. 任务名、镜像 tag、compute config 版本、节点数量等一律从远程实际状态解析，不得硬编码。

## 可用 skills（插件命名空间 `training-ops`）
- `/training-ops:anyscale-job-ops` — Anyscale prodjob 全生命周期
- `/training-ops:ktrain-job-ops` — ktrain PyTorchJob：任务操作、文件同步、日志查询、资源管理
- `/training-ops:pytorchjob-ops` — 纯 kubectl PyTorchJob（H200）：任务操作、日志查询、资源管理
