# 代码 / 工件同步

> 触发词：`同步代码` `sync_code` `更新代码` `同步模型` `同步工件`

## 1. 同步代码到代码 bucket

```bash
bash <SKILL_DIR>/scripts/sync_code.sh \
  --yaml <任务yaml> --repo <本地代码库根> \
  [--pvc <代码PVC>] [--bucket <bucket名>] [--dest s3://<bucket>/<前缀>] \
  [--cluster <EKS集群> --region <区域>] [--exclude <模式>]...
```

脚本行为：

1. 从任务 yaml 解析命名空间、`spec.env.CODE_PREFIX`（缺失即报错中断）。
2. 确定代码 CSI 卷：`--pvc` 指定 > render-extra 中唯一的 PVC 卷自动采用 > 多个 PVC 卷且未指定则报错中断。
3. 经 `kubectl` 解析 `PVC → PV → csi.volumeAttributes.bucketName` 得到 bucket（`--bucket` 或 `--dest` 可跳过 kubectl 解析直给）。
4. 目的地 = `s3://<bucket>/<CODE_PREFIX>`；先 `aws s3 rm --recursive` 清空该前缀，再 `aws s3 sync` 全量上传（内置排除 `.git` / `.codebuddy` / `.claude` / `.cache` / `__pycache__` / `*.pyc` / 模型权重扩展名；额外排除用 `--exclude`）。

`submit.sh` 内部已调用本脚本，走提交流程时无需单独执行。

## 2. 同步模型 / tokenizer / manifest / 音频到节点 NVMe

工件同步不是独立操作，由 `submit.sh` 的数据就绪门控完成（见 `submit.md` 内部步骤 4–6）：每个含 `s3://` / `s3mnt://` 值的 env 派生一个串行同步 Job，落到 `<NVME_MOUNT>/<key>`（`s3` 走 `aws s3 sync --size-only`，`s3mnt` 走本地增量同步）；`fsx://` 工件训练经 FSX CSI 直读，不同步。

- 同步均按大小增量比对，数据已一致则幂等跳过，重复提交无需手工干预。
- 禁止手工在节点上拷数据代替同步 Job；工件未就绪导致的训练失败，回到提交流程重跑。

## 3. 改动与重同步对照

| 改动内容 | 需要重新同步代码吗 | 操作 |
|---|---|---|
| 训练代码 / pipeline 脚本（`.py` / `.sh`） | 是 | 重跑 `submit.sh`（内含代码同步） |
| 任务 yaml | 否 | 重跑 `submit.sh` 即生效 |
| 入口脚本（`--launcher` 指向的文件） | 否（走 ConfigMap） | 重跑 `submit.sh` 自动刷新 |
| 模型 / tokenizer / manifest / 音频 | 否（节点 NVMe 持久） | 仅源路径变化时重跑 `submit.sh` |
| 镜像内依赖 | 否 | 需重建镜像并更新任务 yaml `spec.image`，不属本 skill |

## 4. ktrain storage 快路径（ad-hoc 文件操作，不替代训练工件同步）

对已注册文件系统（`ktrain fs list` 查看，路径写法 `<fs名>:/<路径>`）：

| 命令 | 用途 |
|---|---|
| `ktrain storage ls / cp / mkdir / rm / mv` | 浏览与单点文件操作 |
| `ktrain storage sync <本地目录> <fs>:/<路径> [--delete] [--dry-run]` | rsync 式增量 push（经 storage-browser Pod，受上传大小限制；适合脚本 / 配置目录反复推送） |
| `ktrain storage s3-sync <src> <dst> [--delete] [--dry-run]` | GB 级目录快路径，走本地 AWS 凭证 client↔S3 线速（src / dst 其一必须是已注册 S3 文件系统） |
| `ktrain storage s3-cp` | 单文件（或可选递归）快路径 |
| `ktrain storage presign-upload / presign-download` | 批量生成 S3 预签名 URL，client↔S3 直连上传 / 下载 |

边界：以上仅为 ad-hoc 文件操作；训练工件（模型 / 数据）就绪仍必须走 `submit.sh` 的同步 Job 门控。

## 禁止事项

- 禁止把代码同步目标 bucket / 前缀写死在命令里；一律由任务 yaml 的 PVC 与 `CODE_PREFIX` 解析（或显式 `--dest` 参数）。
- 代码改动未重新同步就提交，Pod 内跑的是旧代码；提交前确认 `submit.sh` 的代码同步步骤已执行。
- 禁止用 `ktrain storage` 快路径（cp / sync / s3-sync）往节点 NVMe 拷数据替代 `submit.sh` 的同步 Job。
