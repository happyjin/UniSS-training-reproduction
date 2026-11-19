# dev workspace（code-server 开发机）

> 触发词：`开发机` `workspace` `code-server` `远程开发` `web IDE` `ssh-keys`

ktrain ≥ 1.7 的交互式开发机：code-server Web IDE + 可选 GPU + 空闲自动停止。适合交互式调试 / 探查数据；**不是训练任务的替代品**，训练仍走 KtrainJob 提交流程（`submit.md`）。

## 生命周期

```bash
ktrain workspace create <NAME> [--namespace <NS>] [--filesystem <FS>]... [--no-filesystem] \
  [--cpu 4] [--memory 16Gi] [--gpus <N>] [--idle-timeout <分钟>]
ktrain workspace list [--namespace <NS>]
ktrain workspace status <NAME>
ktrain workspace open <NAME> [--no-browser]   # 浏览器打开 IDE；--no-browser 只打印 URL
ktrain workspace stop <NAME>                  # scale 到 0（保留盘）
ktrain workspace start <NAME>                 # scale 回 1
ktrain workspace delete <NAME>
```

- `--filesystem` 可重复，指定挂载的注册文件系统（`ktrain fs list` 查看）；缺省按集群自动挂载；`--no-filesystem` 只用 emptyDir 项目目录。
- `--idle-timeout`：空闲自动停止分钟数，默认 120。

## SSH 进入 workspace

```bash
ktrain workspace ssh <NAME> [--user coder]    # 经 ktrain WebSocket 隧道开 SSH shell
```

前置：`ktrain login` 已自动完成 keypair 生成、公钥上传与 `~/.ssh/config` Host 块（`ktrain.<env>.<name>`）；缺失时 `workspace ssh` 会幂等补齐。登录时可用 `ktrain login --no-ssh-setup` 跳过该设置。

公钥管理（workspace remote-SSH 用）：

```bash
ktrain ssh-keys list
ktrain ssh-keys add ~/.ssh/id_ed25519.pub [--name <标签>]
ktrain ssh-keys rm <名称>
```

## 边界

- workspace 只做交互式开发；禁止把长时训练任务塞进 workspace 跑，训练一律 `submit.sh` 提交 KtrainJob。
- workspace 的 create / stop / start / delete 是变更操作，用户指令明确（含名称、资源规格）时直接执行，参数缺失先问清楚。
