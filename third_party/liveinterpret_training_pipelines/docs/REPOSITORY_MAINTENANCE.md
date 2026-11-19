# 仓库整理记录

## 2026-10-31：入口导航与安全清理

检查范围为当前 Git 跟踪文件、顶层目录大小、完全相同的文件、临时/缓存产物，以及推理相关文档的入口说明。
本轮检查发现约 902 个跟踪文件；除打包重复外，未发现大小至少 200 字节且内容完全相同的跟踪文件。
这不是未使用代码的完整证明，不据此批量删训练脚本或迁移模块。

### 已整理

- 新增 [推理入口导航](INFERENCE_ENTRYPOINTS.md)：四条链路、核心实现、官方资产路径、权重阶段、启动参数与结果位置。
- 新增 [文档索引](README.md)，在根 README、Streaming SI README 和根 `inference/README.md` 增加入口。
- 修正 Streaming SI README 中“S5 流式引擎未迁移”的过期描述，补充共享模型和 Graph 探针入口。
- 给 Graph 过程记录标明最新状态与历史阶段，避免读到早期失败记录便误以为后续全量未完成。

### 删除项与依据

| 项目 | 依据 | 影响 |
|---|---|---|
| `test_script/wandb_runs_export/wandb_configs.tgz` | 压缩包内 180 个普通文件全部与同目录展开文件逐字节一致，且展开文件已由 Git 跟踪；仓库文本中未发现对此压缩包的引用 | 删除 344622 字节的重复归档；所有原始记录保留，Git 历史仍可恢复归档 |
| 本地 `__pycache__` / `.pyc` | Python 可重新生成的缓存，已被 `.gitignore` 排除；不属于源码 | 清理本地缓存，不改变跟踪代码 |

避免再提交同一归档，在 `.gitignore` 中精确排除上述 `wandb_configs.tgz`，不泛化忽略所有压缩包。
实际清理缓存 25 个文件、12 个目录，共 844942 字节；连同重复归档共释放约 1.13 MiB。
归档删除前已在 `/tmp/liveinterpret-cleanup-h5gp559h/` 留存本地备份，该临时备份不属于版本库。

文档检查：57 个新增文档中的本地 Markdown 链接均可解析；`docs/` 下所有 Markdown 文档均已纳入索引；`git diff --check` 通过。

### 检查后保留的内容

| 目录或文件 | 保留原因 |
|---|---|
| `custom/simuls2st_omni/` 与 `custom/streaming_si/` | 前者是 Omni 训练/迁移实现，后者是 S4/S5 与评测包装；不是重复目录 |
| 根 `inference/` | qwen_omnispeak CLI，和 S4/S5/官方 Omni 跑测面向不同模型 |
| `custom/streaming_si/__init__.py` 的旧模块映射 | 服务历史 adapter 的模块路径与外部脚本；无引用搜索不能覆盖外部依赖 |
| `test_script/` 与 WandB 展开记录 | 训练探针、回归测试和历史实验依据；没有证据证明全部可删 |
| `vendor/SimulEval/` | 仓库内代码/测试可引用；资产目录还可能有官方版本，需分别管理 |
| `ktrain/`、`pytorch_job/`、`anyscale/` | 不同调度与启动方式，不按目录名称相似判定重复 |
| `pipelines/streaming_si/docs/2609. Paralinguistic Information.pdf` | 约 4MB 的研究参考资料，有独立内容；本轮保留 |
| 根架构 HTML、历史 handoff、优化计划及过程报告 | 文档用途与实验溯源；没有完整替代关系时保留 |
| `.claude/`、`.codebuddy/` 等本地工具目录 | 被忽略的本地工作环境，不属于可随意删除的项目生成物 |

本轮不移动/重命名训练或推理源码，避免破坏集群 YAML、外部脚本和历史 checkpoint 中的路径。
后续若要缩减历史实验资产，宜单独制定归档范围并校验 S3 备份，再删除仓库中的唯一副本。
