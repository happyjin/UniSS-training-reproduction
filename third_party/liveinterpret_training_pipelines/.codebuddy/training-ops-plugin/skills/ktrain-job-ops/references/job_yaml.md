# 任务定义管理（validate / manifest / schema / explain / export）

> 触发词：`导出` `export` `克隆` `校验` `validate` `渲染` `manifest` `schema` `字段` `explain`

除重提外均为只读 / 本地操作，无集群副作用。

## 校验 yaml

```bash
ktrain job validate -f <任务yaml>   # exit 0 = 合法，exit 1 = 有错误；无副作用
```

## 预览渲染结果（不 apply）

```bash
ktrain job manifest -f <任务yaml> [--cluster <集群>]
```

输出完整渲染的 PyTorchJob / RayJob YAML + 平台注入清单（文件系统挂载等）。提交前审查 operator 渲染行为用；`ktrain job submit -f <yaml> --dry-run` 等价于提交入口的 dry-run。

## 查字段含义 / spec 结构

```bash
ktrain explain ktrainjob.spec.numNodes                 # 类 kubectl explain
ktrain explain ktrainjob.spec --recursive              # 递归打印所有子字段
ktrain job schema | jq '.properties.spec.properties'   # TrainJob spec 的 JSON Schema（ktrain.oit-svc.com/v1alpha2）
ktrain job schema > job-schema.json                    # 落盘供 IDE 校验
```

## 导出运行中任务为 yaml

```bash
ktrain job export <JOB_NAME> --namespace <NAMESPACE>                # 输出到 stdout
ktrain job export <JOB_NAME> --namespace <NAMESPACE> -o rerun.yaml  # 落盘
ktrain job export <JOB_NAME> --namespace <NAMESPACE> --raw          # 不过滤（调试用）
```

默认输出已清洗（剔除 apiserver 簿记、kopf annotation、operator 内部字段与 status），可直接编辑重提（submit 支持 `-f -` 从 stdin 读）：

```bash
ktrain job export <JOB_NAME> --namespace <NAMESPACE> | sed 's/<旧名>/<新名>/' | ktrain job submit -f -
```

## 边界

- 含 `s3://` / `s3mnt://` 工件或代码同步需求的训练任务，重提必须走 `submit.sh`（数据就绪门控 + 代码同步）；`export → submit` 仅适用于无同步需求的场景，禁止用它绕过 `submit.sh`。
- 导出的 yaml 是 operator 当前 CR 的清洗版，与原始提交 yaml 可能不完全一致；差异以 export 输出为准。
