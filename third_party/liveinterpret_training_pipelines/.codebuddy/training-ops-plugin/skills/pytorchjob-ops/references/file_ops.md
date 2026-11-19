# PyTorchJob 文件操作（代码同步到对象存储 CSI）

> 触发词：`同步代码` `sync_code` `代码` `更新代码`

PyTorchJob 的 pod 通过对象存储 CSI（S3 CSI）挂载只读获取代码与数据。代码每次提交由 `submit.sh` 强制同步到同区域对象存储；模型 / tokenizer / manifest / 原音频则由 `submit.sh` 的**数据就绪门控**在提交前经 4 个串行 sync Job（`completions=N` 门控）同步到节点 NVMe（每个工件跑一次 `aws s3 sync --size-only`，同步 Job 由 `derive_sync_jobs.py` 从训练 yaml 推导，业务仓库不维护同步任务配置）。

---

## 同步代码到对象存储

skill 提供 `scripts/sync_code.sh`，把仓库代码强制同步到 data bucket
（先 `aws s3 rm --recursive` 清空目标前缀，再全量上传，保证与本地严格一致）。
目标 bucket **不硬编码**：从训练 yaml（`pytorch_job/pytorchjob.yaml`）中 `s3all` volume 的
`persistentVolumeClaim.claimName` 解析出 PVC，再经 `kubectl` 查 `PVC -> PV ->
csi.volumeAttributes.bucketName` 得到；代码前缀为 `<git_user_name>/<repo_name>`（实时从
git 配置解析，如 `deferzhou/liveinterpret_training_pipelines`）：

```bash
bash .codebuddy/skills/pytorchjob-ops/scripts/sync_code.sh [pytorchjob.yaml 路径]
```

`submit.sh` 内部已调用此脚本（传入训练 yaml），无需单独跑。训练入口 `TRAIN_SCRIPT` 即该 bucket 下的 S3 路径，容器内通过 S3 CSI 直接执行，无需拷贝到本地。

> ⚠️ 清空目标前缀后全量上传属于变更操作，执行前需确认。

---

## 同步模型 / tokenizer / manifest / 原音频到节点 NVMe

这些工件由 `submit.sh` 在提交前自动处理（见 references/submit.md 步骤 4–7），无需手动同步。其 S3 源全部在
`pytorch_job/pytorchjob.yaml` 的 env 声明：

- 模型 + tokenizer：`s3://iegg-speech-model-models/origin_models/...`（模型桶）
- manifest + 原音频：`s3://iegg-speech-model-data-.../data/...`（数据桶）

`derive_sync_jobs.py` 解析训练 yaml 的 env（如 `MODEL_PATH` / `QWEN3_TTS_TOKENIZER` / `BRIDGE_ALIGNMENT_DATA` / `ARTIFACT_AUDIO_SHARDS`），把其中的 `s3://` URL 直接作为 sync 源，经 `aws s3 sync --size-only`（s3 直连模式）同步到节点 NVMe（`/nvme/<s3-key>`，与 S3 key 1:1）。`aws s3 sync` 自带大小比对、已一致则跳过（幂等）；不依赖任何 S3 CSI PVC 挂载或 rsync。

---

## 哪些改动需要重新 sync

| 改动 | 是否需 sync | 说明 |
|---|---|---|
| 训练代码（`custom/` 等） | 是 | 走对象存储 CSI，必须重新 `sync_code` |
| pipeline / 超参脚本（`*.sh`） | 是 | 训练超参 / 环境变量在此设置 |
| 训练启动脚本（`launch_train.sh`） | 否（走 ConfigMap） | `submit.sh` 自动刷新 ConfigMap |
| `pytorchjob.yaml` | 否 | apply / 重新 `submit.sh` 即生效 |
| 模型 / tokenizer / manifest / 音频 | 否（节点 NVMe 持久） | 仅首次缺失时同步；已存在提交时跳过 |
| 镜像内依赖 | 否 | 需重建镜像 |

---

## 注意事项（通用）

1. 代码随容器退出清理（本地临时路径），每次提交都重新 sync 保证最新。
2. 模型 / tokenizer / manifest / 音频由 submit.sh 同步到持久 NVMe（跨重启持久），不进代码同步范围。
3. 容器路径采用「/nvme + S3 key 去头」1:1 映射，无隐式路径转换潜规则。
4. 改动不生效时先确认 `submit.sh` 是否真正跑了 `sync_code.sh`，以及超参配置是否同步。
