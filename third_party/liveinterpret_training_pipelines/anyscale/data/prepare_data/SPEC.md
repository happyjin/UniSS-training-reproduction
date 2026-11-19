# prepare_data 滚动预热 + data_maker 流水线 Spec

> 目录：`anyscale/data/prepare_data/`
> 适用任务：Anyscale/Ray 上的 `prepare_data` 数据预处理（生成 `bridge_alignment_joint` 训练 jsonl）。
> 维护者：deferzhou

本文件记录该流水线的**设计意图、组件职责、关键不变量与已知坑**。代码以本目录下的 `worker.sh`（编排器）、`prewarm.py`（FSx CLI）、`tools.py`（非 FSx 辅助 CLI）、`prewarm_task_jsonl.py`（最终预热）为准；`dataset_maker/` 下的 maker 脚本为被调用方，本文只描述与其的接口契约。

---

## 1. 设计原则

1. **Shell 是唯一编排者**（`worker.sh`）：负责发现输入、任务开始**全量预热**、每个阶段单次 maker 调用（占满核心）、拼接、每阶段**上传 S3**、合并、审计、最终预热。控制流与命令拼接在 bash。
2. **Python 只做 CLI 工具、且每个工具单一职责、无状态**：
   - `prewarm.py`：FSx 相关——`preload`（任务开始全量预热目录树）、`extract-paths` / `produce` / `ready` / `release`（保留）/ `audit`。
   - `tools.py`：非 FSx 的「管道」操作——`find-inputs` / `concat` / `manifest-diff` / `manifest-save` / `ready` / `count-lines`。
   - 每个子命令独立进程、独立退出，**不跨调用共享内存状态**。
3. **单 CPU head 节点、无多节点分发**：数据预处理是 CPU/IO 密集型，不需要 GPU。节点规格与 `anyscale/config/compute_config_h100.yaml` 的 head 完全一致（`8CPU-32GB`，仅挂载 fsx / s3 / efs），compute config 只有 head 节点、无 worker 节点。各阶段产物临时写在节点本地 ephemeral-storage（`LOCAL_WORK_DIR`），阶段结束即上传 S3，节点本身不保留（无 NVMe）。
4. **任务开始即全量预热（full preload）**：在跑 maker 之前，对 `TTS_DATA_PATH` / `S2S_DATA_PATH` / `TTS_RAW_PATH_PREFIX` 三个根目录下**全部音频**执行 `lfs hsm_restore`（按 AWS HSM-DRA 预加载方式，目录级批量提交），召回进 FSx 缓存，使后续 maker 读热数据。预热 best-effort（受 `PREWARM_FULL_WAIT_TIMEOUT` 限制，超时则继续，可能冷读）；缓存小于全量音频时 Lustre HSM 按 LRU 覆盖，不阻塞流程。
5. **并发占满核心**：每个阶段以**单次** maker 调用处理该阶段全部增量 parquet，传 `--num-workers = nproc`（默认占满节点所有 CPU 核心），由 maker 内部 `ProcessPoolExecutor` 把行组均匀分发各核心。阶段间顺序执行（TTS → S2S → Merge），每个阶段都跑满核心。

**降级行为**：当 `lfs` 不可用（容器非 privileged）或 HSM 不可达时，`prewarm.py preload` / `prewarm_task_jsonl.py` 自动降级为 no-op，流程照常——只是失去预热加速，maker 读文件时触发 on-demand HSM 召回（一次读一个文件、按需淘汰 LRU，天然有界）。

---

## 2. 组件职责

| 文件 | 职责 |
|------|------|
| `worker.sh` | **唯一编排器**。env 配置 → 任务开始全量 `preload` → 各阶段 `process_stage`（`find-inputs`→`manifest-diff`→单次 maker 调用 `--num-workers`→`concat`→`manifest-save`）→ 每阶段 `s3_upload` → `merge` → `audit` → 最终预热。 |
| `prewarm.py` | FSx HSM 原语 + CLI：`preload`（遍历根目录树、`lfs hsm_restore` 全量召回并轮询）、`extract-paths` / `produce` / `ready` / `release`（保留）/ `audit`（音频总字节审计）。 |
| `tools.py` | 非 FSx 辅助 CLI：`find-inputs`（导入 maker 枚举 parquet）、`concat`（part 追加/分片）、`manifest-diff`/`manifest-save`（增量 manifest）、`ready`/`count-lines`（shell 门禁）。 |
| `prewarm_task_jsonl.py` | 主流程后由单节点把 `PREWARM_TARGETS` 阶段 jsonl 引用的全部音频预热进 FSx（复用 `extract_audio_paths_from_jsonl` + `warm_paths`）。 |
| `dataset_maker/make_text_to_speech.py` | TTS maker，被 `worker.sh` 以单次子进程调用（`--input-files` 全部增量 + `--num-workers`）。 |
| `dataset_maker/make_speech_to_speech_translation.py` | S2S maker，同上。 |
| `dataset_maker/merge_datasets.py` | 按 5:1 把 TTS/S2S 合并为 `bridge_alignment_joint`，末尾调用一次。 |
| `job.yaml` | Anyscale 任务定义，env_vars 直接注入全部 knobs；compute config `tokyo-prepare-data-cpu` 为 head-only `8CPU-32GB`，容器 `privileged: true`（HSM ioctl 需要）。 |

> 已删除：`run_prepare.py`、`dispatch.py`（head-only 模式不再多 worker 分发）。

---

## 3. 端到端流程（`worker.sh`）

```
(0/4) 全量预热: prewarm.py preload --roots TTS_DATA_PATH S2S_DATA_PATH TTS_RAW_PATH_PREFIX
      -> lfs hsm_restore 三个根目录下全部音频到 FSx 缓存 (best-effort, 受超时限制)
(1/4) TTS : find-inputs -> manifest-diff(delta) -> 单次 make_text_to_speech
            (--input-files 全部增量, --num-workers=nproc) -> concat -> 上传 S3
            s3://<S3_BUCKET>/datasets/<TASK_NAME>/text_to_speech/
(2/4) S2S : 同上, make_speech_to_speech_translation
            -> s3://<S3_BUCKET>/datasets/<TASK_NAME>/speech_to_speech_translation/
(3/4) Merge: merge_datasets.py --ratios ${BRIDGE_RATIOS} (默认 1 5 => TTS:S2S=5:1)
            -> bridge_alignment_joint[.jsonl] (shuffle)
            -> s3://<S3_BUCKET>/datasets/<TASK_NAME>/bridge_alignment_joint/
(4/4) Audit: prewarm.py audit --jsonl final -> 日志 + 可选报告文件
(5)   最终预热: 按 PREWARM_TARGETS 把对应阶段 jsonl 引用的全部音频
            lfs hsm_restore 进 FSx (单节点, 受 FSX_MOUNT 原子标记互斥)
```

`SHARD_MAX_LINES>0` 时，TTS/S2S/最终产物均为**分片子目录**（`text_to_speech/` …）；等于 0 时均为单文件 `*.jsonl`。

**产物与上传**：各阶段结果先写到 `LOCAL_WORK_DIR`（节点本地 ephemeral-storage），阶段结束即 `aws s3 cp`/`sync` 到 `s3://<S3_BUCKET>/datasets/<TASK_NAME>/<stage>/`。S3 是持久副本；本地临时目录随节点回收。凭借 `TASK_NAME`（默认时间戳）保证每次运行目录独立、可幂等覆盖。

---

## 4. 预热设计（历史/参考）

> 注：本流水线已改为**任务开始全量预热**（`prewarm.py preload` 对三个根目录树 `lfs hsm_restore`）+ **按 `PREWARM_TARGETS` 最终预热**（`prewarm_task_jsonl.py`），不再使用下文的「逐批滚动预热 + 释放」模型。本节保留的是 HSM 吞吐与目录级批量 `lfs hsm_restore` 的核心推理，对两种模型都适用。

### 4.1 为什么是「目录级批量 + 预热」

HSM 召回（archive→FSx 缓存的拷贝）是**服务端吞吐上限**，客户端并发 `lfs hsm_restore` **不会**加快召回——瓶颈在 FSx 容量 × 每 TiB 速率（见 §12）。因此提速的杠杆是：

- **批量 / 目录级 restore**：一次 `lfs hsm_restore` 带上整目录的音频路径，让 HSM 代理以满吞吐把整目录拷进缓存，而不是逐文件 RPC（早期逐文件 `hsm_action` 轮询约 5 万次进程调用/轮，远慢于 `warm_fsx` 的 100 files/call）。
- **滚动预热（批次间重叠）**：每批 `produce` 预热本批目录，消费者处理已就绪目录并释放，下一批预热与处理重叠 I/O 与计算。

而**不是**靠客户端并发（`PREWARM_PARALLEL` 已移除）：流水线按目录批量提交 `restore`，靠 HSM 服务端吞吐；`release` 也固定并发、无 env 旋钮。

### 4.2 就绪判定：纯文件查询，绝不调 `lfs hsm_action`（消费者侧）

消费者要等「本批目录已预热」才能调 maker。但 `lfs hsm_action` 看到的是**整块 FSx 缓存状态**，分不清「这批文件是本流水线预热进来的，还是别的租户 / 别的任务遗留的」。所以用 `lfs hsm_action` 判定就绪会**误判**。

解决办法：**就绪状态由生产者写入 state 文件**，消费者只做集合查询：

```
produce:  for each dir:  lfs hsm_restore (目录级批量) -> 轮询该目录文件是否全 NOOP
                       -> 全 NOOP 则把 dir 路径 append 到 ready_dirs (state 文件)
ready:    needed = {dirname(p) for p in 本批音频路径}
          read ready_dirs -> ready 集合
          exit 0  iff  needed <= ready          # 纯文件查询, 不调 lfs hsm_action
```

这样消费者完全不受其它租户缓存状态干扰。生产者即便在超时（或 `lfs` 不可用）时也把目录写入 state（带告警），因此消费者**最终一定会继续**，不会死锁。

### 4.3 working set 有界：处理完即释放，无需容量门控

预热**不做剩余容量门控**（原 `PREWARM_RESERVE_GB` 已移除）：`lfs hsm_restore` 召回时若 FSx 缓存已满，Lustre HSM 自动按 LRU 逐出旧缓存（即「直接覆盖」），无需手动节流或启动时全量释放。消费者每批处理完即 `release` 该批音频（默认行为，无开关）。于是：

- 每批音频只在「处理期间」驻留缓存，处理完即逐出；
- 批次间预热与处理重叠，但单批在途量受限（缓存满时 HSM 自动 LRU 回收），working set 始终有界，**不会写满 FSx**，也无需常驻水位守护进程。

### 4.4 每批三步（消费者侧）

```bash
# 每个 batch (滚动预热, 默认开启):
python3 prewarm.py extract-paths --kind $kind --parquet-file $BATCH > paths.txt
python3 prewarm.py produce --audio-list paths.txt --dir-state ready_dirs \
    --dir-batch $DIR_BATCH --poll-interval $PRODUCER_POLL --wait-timeout $WAIT_TIMEOUT
while ! python3 prewarm.py ready --audio-list paths.txt --dir-state ready_dirs; do
    sleep $CONSUMER_POLL
done
python3 $MAKER --input-files ${BATCH[@]} --input-dir $DIR --output $PART --shard-max-lines 0 ...
python3 prewarm.py release --list paths.txt   # 处理完即释放 (默认)
```

- **`produce` 触发归档→FSx 缓存拷贝**（服务端、批量），就绪目录写 state。这是**提前**发生的 I/O。
- **maker 处理时「当场读文件」不是预热**：那只是一次缓存命中（或冷召回）。因为预热已提前完成，maker 读到的是已缓存文件。
- **`release`** = `lfs hsm_release` 把该批音频从缓存**逐出**（释放数据块，保留 inode 元数据）。
- `lfs` 缺失 / 容器非 privileged 时，`produce` 把所有目录直接记就绪（no-op 降级），`ready` 立即放行，流程照常（只是失去预热加速，maker 触发现场冷召回）。

### 4.5 为什么不用容量水位 / lookahead 守护进程

- 容量水位 + 生产者/消费者模型需要跨批次持久状态（`_ready`、`_cached_bytes`），必须常驻守护进程；早期实现曾因「必须凑满整批 16 个 parquet 才处理」在水位模式下死等直到 2h 超时。
- 本模型把「就绪」状态记录在 **state 文件**里（不是内存记账），**不需要常驻进程**，也无死锁风险（缓存满时 HSM 自动 LRU 回收，不靠容量门控）。
- 调参只需 `PREWARM_DIR_BATCH`（单目录 restore 批量）、`PREWARM_PRODUCER_POLL` / `PREWARM_CONSUMER_POLL`（轮询间隔）、`PREWARM_WAIT_TIMEOUT`（单目录召回上限），详见 §13。切块粒度为 worker 内固定值（非 env 配置）。

### 4.6 调参权衡

| 变量 | 越大 | 越小 |
|------|------|------|

| `PREWARM_DIR_BATCH` | 单目录 restore 调用更少、HSM 更易满吞吐 | 单目录 restore 调用更多（大目录拆更细） |

| `PREWARM_WAIT_TIMEOUT` | 单目录召回等待上限更宽 | 召回卡住时更快 best-effort 继续（可能冷读） |


---

## 5. 增量更新（incremental update）

当 `TTS_DATA_PATH` / `S2S_DATA_PATH` 下**新增** parquet 时，只处理新文件、把新数据**追加**到已有 jsonl，绝不重读已处理音频，也只在新数据出现时才重算 `bridge_alignment_joint` 与音频审计。

### 5.1 机制：输入 manifest

每个数据集维护 `<out_base>.inputs_manifest.json`：`{"inputs": [已处理 parquet 绝对路径集合]}`。

`worker.sh` 流程（`process_dataset` 函数）：
```
current  = find-inputs (当前 input_dir 下全部 parquet, 绝对路径)
delta    = manifest-diff(current) = current - 已处理
```
- **`delta` 非空**：把 `delta` 按固定批次切块，每批抽取音频路径并交给 `produce`、逐批(`ready`→处理→`release`) 产物经 `tools.py concat` **追加**到既有 `out_base`（旧行保留）。
- **`delta` 为空且 `ready(out_base)`**：直接跳过（保留旧产物，不重算）。
- **manifest 存在但产物缺失**：视同半成品/丢失，删除 manifest 并以 `current` 为 delta 全量重算（先清旧残留）。
- **新鲜运行（manifest 丢失）**：清掉旧 `out_base`，避免追加到陈旧/遗留文件上。

### 5.2 追加不覆盖（append-only）

- 单文件 `SHARD_MAX_LINES=0`：`concat` 以 `"a"` 模式追加新行。
- 分片 `SHARD_MAX_LINES>0`：`concat` 的 `_append_parts_sharded` 追加到最后一个分片，满 `shard_max_lines` 才滚动新分片。
- manifest 在 **concat 成功之后**才 `manifest-save`（原子 `os.replace`）。崩溃只导致「该 delta 重算一次」（at-least-once），不丢数据；`FORCE_UPDATE` 用于从任何不一致中恢复。

### 5.3 条件合并与审计（避免无谓重算）

仅当 `need_merge = tts_new or s2s_new or not ready(final_out)` 时：重跑 `merge_datasets.py`（只读 tts/s2s jsonl）+ 跑 `audit`。无新数据时两者整体跳过，日志打印 `skip`。

### 5.4 强制全量：`FORCE_UPDATE`

`FORCE_UPDATE=1`（`worker.sh`）删除 `text_to_speech[.jsonl]`、`speech_to_speech_translation[.jsonl]`、`bridge_alignment_joint[.jsonl]` **及两个 `*.inputs_manifest.json`** 和最终预热标记。manifest 必须清掉，否则增量逻辑会误判「已处理」而跳过全部数据。

---

## 6. 逐批处理 & 不覆盖不变量（重要）

**不变量**：每个 maker 子进程写入**带批次号后缀的唯一文件** `${out_base}.part_NNNNN.jsonl`，最终由 `concat` **追加**进单一产物（增量语义，见 §5）；批次之间**不会互相覆盖**。

- `worker.sh` 每批：`part="$(printf '%s.part_%05d.jsonl' "$out_base" "$idx")"`，`--output "$part" --shard-max-lines 0`。
- maker 内部 `shardio.ShardWriter(args.output, 0)` 写到 `part` 本身（保留后缀）；`>0` 时以 stem 建子目录。`concat` 再把它追加进 `out_base`。
- maker 的默认文件名只在**未传 `--output`** 时出现；本流水线总是显式传 `--output`，不会走到默认分支。

---

## 7. 硬时长/格式过滤

- maker 内部 `duration_filter_reason(...)`：优先用 parquet 预计算 `duration_ms`（无需读音频），缺失时回退读音频头（wave → soundfile → ffmpeg），支持 mp3/aac/flac/wav/opus 等。返回 `None`（通过）/`audio_too_long`/`audio_duration_unknown`。
- `worker.sh` 传 `--max-audio-duration-sec`（默认 120），并**默认丢弃未知时长音频**（硬条件，不再暴露开关）。**硬条件**：未知时长/超长音频一律不进训练集。

---

## 8. 配置参考（环境变量）

全部由 `job.yaml` env_vars 直接注入 `worker.sh`（不再经 dispatch 透传）。默认值见 `worker.sh` 顶部 `${VAR:-default}`。

| 变量 | 默认 | 含义 |
|------|------|------|
| `TTS_DATA_PATH` | （必填） | TTS parquet 根目录 |
| `S2S_DATA_PATH` | （必填） | S2S parquet 根目录 |
| `TTS_RAW_PATH_PREFIX` | （必填） | 原始 TTS 音频前缀（任务开始全量预热的根之一） |
| `FSX_MOUNT` | `/mnt/fsx` | S3 镜像挂载点 |
| `LOCAL_WORK_DIR` | `/tmp/prepare_data_<TASK_NAME>` | 各阶段产物本地临时目录（节点 ephemeral-storage） |
| `TASK_NAME` | 时间戳 | S3 上根目录名：`datasets/<TASK_NAME>/<stage>/`；默认时间戳保证独立 |
| `S3_BUCKET` | `anyscale-poc-v2-bucket-124355679795-ap-northeast-1` | 结果上传目标桶 |
| `S3_REGION` | `ap-northeast-1` | 结果上传 region |
| `SAMPLING_RATE` | `16000` | TTS 写入采样率 |
| `LATENCY_BUDGET_MS` | `2000` | S2S 延迟预算 |
| `SMALL_DATA_LIMIT` | `0` | `>0` 仅处理前 N 行（小数据调试） |
| `MAX_AUDIO_DURATION_SEC` | `120` | 硬时长上限（秒） |
| `SHARD_MAX_LINES` | `0` | `>0` 时分片输出 |


| `PREP_PARALLEL` | `nproc` | 每个阶段 maker 的 `--num-workers`，默认占满节点所有 CPU 核心 |
| `PREWARM_DIR_BATCH` | `2000` | 单个 `lfs hsm_restore` 调用最多带的文件数；目录内文件超此值则拆多段 |
| `PREWARM_POLL_INTERVAL` | `15` | 全量预热轮询目录就绪间隔（秒） |
| `PREWARM_FULL_WAIT_TIMEOUT` | `7200` | 任务开始全量预热的等待上限（秒，超时 best-effort 继续） |
| `PREWARM_PARALLEL` | `16` | 最终预热 `lfs hsm_restore` 客户端并发度 |
| `PREWARM_PRODUCER_POLL` | `15` | 最终预热轮询间隔（秒） |
| `PREWARM_WAIT_TIMEOUT` | `7200` | 最终预热单批等待上限（秒） |
| `AUDIO_SIZE_REPORT` | （空） | 审计报告路径；空则只打日志 |
| `BRIDGE_RATIOS` | `1 5` | 合并比例，对应 `merge_datasets.py --inputs` 顺序 (S2S, TTS)；`1 5` => TTS:S2S=5:1 |
| `PREWARM_TASK_JSONL` | `auto` | 最终预热：`auto` = 自动预热 `PREWARM_TARGETS` 列出的产物；留空=跳过；也可直接填 jsonl 文件/分片目录路径 |
| `PREWARM_TARGETS` | `bridge_alignment_joint` | `auto` 模式下的产物名（逗号组合） |
| `FORCE_UPDATE` | （空） | 非空则清掉产物+manifest+最终预热标记，强制全量重算 |

---

## 9. 已知坑 / 维护注意

1. **`lfs hsm_restore/release` 需要 privileged 容器**（HSM ioctl）。`compute_config` 的 ray 容器须 `securityContext.privileged: true`，否则 `lfs` 不可用、预热降级为 no-op。
2. **`lfs hsm_release` 只释放数据、保留元数据**；后续 `audit` 的 `os.path.getsize` 仍可用，不会重新召回。
3. **不要改回「进程内调用 maker」**：maker 内部用 `ProcessPoolExecutor`（fork），与预热无关但子进程隔离是最安全选择；本流水线始终以子进程调用 maker。
4. **消费者就绪判定必须走 state 文件，不要调 `lfs hsm_action`**：FSx 缓存状态混有其它租户/任务的文件，`hsm_action` 无法区分「本批是否本流水线预热」，会误判就绪（或漏判）。`ready` 子命令是纯集合查询，天然规避该问题。
5. **不要恢复容量水位/常驻生产者-消费者状态机**：它需要跨批次持久内存状态，且早期曾因「必须凑满整批 16 个 parquet 才处理」死等直到 2h 超时。本模型把就绪记录在 state 文件，生产者 self-throttle 于空闲空间，无死锁。
6. **`REPO_ROOT` 路径深度陷阱**：`prewarm.py` / `prewarm_task_jsonl.py` / `tools.py` 位于 `anyscale/data/prepare_data/`，故 `REPO_ROOT` 必须用 **3 层** `..`（`os.path.join(__file__, "..", "..", "..")`）才能指向仓库根，从而 `import shardio` 与 `dataset_maker.*`。误用 2 层会解析到 `anyscale/`，导致 `ModuleNotFoundError: No module named 'shardio'`。`worker.sh` 用同一规则算 `REPO_ROOT`。
7. **调用 maker 必须传 `--input-dir`**：`worker.sh` 除 `--input-files` 外**还**传 `--input-dir`（按 `kind` 取 `TTS_DATA_PATH`/`S2S_DATA_PATH`）。maker 的 pre-flight 要求 `--input-dir` 目录存在，且 S2S maker 用它解析相对音频路径；缺失会报 `[ERROR] input directory not found` 并退出（错误码 1）。
8. **`find-inputs` 依赖导入 maker 模块**：`tools.py` 启动即 `from dataset_maker.make_* import find_input_parquets`，要求 `REPO_ROOT`/`dataset_maker` 在 `sys.path`（已设置）。修改 maker 的 `find_input_parquets` 签名需同步此处。
9. **`SMALL_DATA_LIMIT` 仅作用于首批次**：`worker.sh` 只在 `first_batch` 给 maker 加 `--limit`，其余批次不限——避免小数据验证时只产出一个微型 jsonl。
10. **`merge` 的 `--inputs` 顺序为 (S2S, TTS)**，与 `BRIDGE_RATIOS` 顺序一致；`BRIDGE_RATIOS` 为两个正数（如 `1 5`），长度必须与 `--inputs` 相同，否则 `merge_datasets.py` 直接报错退出。
11. **`produce` 标记就绪是「全目录文件 NOOP」**：一个目录里只要有文件仍在召回，该目录就不会写入 state，消费者同目录的后续批次会等到整目录就绪。这是正确且安全的（避免冷读）；但意味着目录是预热/释放的最小粒度，超大单目录（接近或超过 FSx 容量）需靠 `RESERVE_GB` 背压 + 逐批 release 切分 working set。

---

## 10. 端到端验证建议

- 小数据冒烟：`SMALL_DATA_LIMIT>0` + `PREWARM_BATCH_SIZE=4` 跑一轮，核对日志出现：后台 `producer started (pid=...)` → `batch N waiting for prewarm` → `dir ready: ...` → `maker batch ...` → `releasing N files` → `concat` → 合并 → 审计总字节数；并确认 `wait 生产者` 正常收尾。
- 审计产出：`AUDIO_SIZE_REPORT` 文件含 `audio_files_unique / audio_bytes_unique / audio_bytes_total_referenced / audio_references / audio_missing`。
- 增量验证：首轮正常产出后，向 `TTS_DATA_PATH`/`S2S_DATA_PATH` **新增**若干 parquet 再跑。确认日志出现 `incremental: N new / M total inputs` 且只处理新增批次；旧 jsonl 行数应只增不减（append）；`*.inputs_manifest.json` 内容随已处理输入增长；无新数据时日志打印 `no new inputs; skip` 且 `bridge unchanged` 跳过合并/审计。
- 强制全量验证：`FORCE_UPDATE=1` 重跑，确认两个 manifest 被删、jsonl 从头再生、行数与首轮一致。
- 背压验证（大数据集）：观察生产者日志在 FSx 空闲低于 `RESERVE_GB` 时出现 `FSx free XGiB < reserve YGiB; throttling`，且消费者持续 `release` 后生产者恢复——说明背压闭环成立、working set 有界。

---

## 11. 最终任务 jsonl 预热（single-node prewarm）

主流程结束后，由**一个节点**把配置的 task jsonl 引用的全部音频 `lfs hsm_restore` 进 FSx 缓存，使下游训练直接命中缓存、避免冷 HSM 召回。

- 配置 `PREWARM_TASK_JSONL`（默认 `auto`）：`auto`/`1` 预热 `PREWARM_TARGETS` 列出的产物（默认 `bridge_alignment_joint`）；留空跳过；其它非空值直接作为 jsonl 文件/分片目录路径。
- 单/多文件无需单独配置：整条链路形态由唯一全局开关 `SHARD_MAX_LINES` 决定；`worker.sh` 在 `auto` 模式只传逻辑名 `${OUTPUT_DIR}/<name>`，由 `prewarm_task_jsonl.py` 按「目录优先、否则 `<name>.jsonl`」自动识别。
- 互斥靠共享 `FSX_MOUNT` 上的原子标记 `.prewarm_task_jsonl_done`：某节点以 `set -C; : > 标记` 创建成功即获执行权，其余跳过。标记一经创建长期有效（只预热一次）；`FORCE_UPDATE=1` 会清掉它触发重新预热。

---

## 12. 当前 FSx 配置（来自仓库 + 运行时观测）

| 项 | 值 | 来源 |
|----|----|------|
| PVC 名称 | `aigc-anyscale-fsx-pvc` | `warm_fsx/compute_config_warm_fsx.yaml` |
| 挂载点 | `/mnt/fsx` | `compute_config` + `worker.sh` |
| 容量 | ≈ **1.12 TiB**（1,230,344,290,304 B） | 前期 job 运行时日志观测 |
| HSM 召回吞吐上限 | **50–200 MB/s/TiB**（FSx for Lustre 官方经验带宽） → 本容量下 ≈ **56–224 MB/s** | FSx 官方文档经验值 × 实测容量 |
| compute | `tokyo-prepare-data-head`（head-only，无独立 worker 节点） | `job.yaml` / `compute_config` |
| 容器镜像 | `anyscale/image/warm-fsx:5` | `job.yaml` |
| 特权 | ray 容器 `privileged: true`（HSM ioctl 必需） | `compute_config` |
| `warm_fsx` 参考配置 | `WARM_PARALLEL=16`、`WARM_POLL_INTERVAL=30`、`WARM_WAIT_TIMEOUT=7200` | `warm_fsx/job.yaml` |
| `warm_fsx` 批量轮询 | `xargs -a list -n 100 -P 16 lfs hsm_action \| grep -vc NOOP`（100 files/call，远优于逐文件） | `warm_fsx/warm_cache.sh` |

**关键推论**：

- 召回吞吐由 FSx 容量决定、服务端上限封顶；**客户端并发 `lfs hsm_restore` 无法突破该上限**。本流水线只启动**一个**后台生产者即足够，提速靠「目录级批量 restore + 生产者/消费者重叠」。
- `warm_fsx` 只预热**过滤后最终集合**，体量可控；`prepare_data` 预热的是**过滤前全集**，体量远超 1.12 TiB，故**不可能一次性全部预热进缓存**。靠「有界 working set = `RESERVE_GB` 余量 + 逐批 `release`」来解决（见 §4.3 / §13）。
- 逐文件 `lfs hsm_action` 轮询约 5 万次进程调用/轮，是早期 prewarm 慢的主因之一；本模型生产者用 `DIR_BATCH=2000` 的批量 restore、`warm_fsx` 用 100 files/call 批量轮询，消费者侧 `ready` 完全不调 `lfs`。

---

## 13. 建议的生产者设置

| 设置 | 建议值 | 理由 |
|------|--------|------|
| 生产者进程数 | **1**（单后台 `produce`） | HSM 召回是服务端吞吐，多客户端并发不加速；一个进程足以维持 HSM 队列满载 |
| `PREWARM_DIR_BATCH` | **2000** files/call | 单目录 restore 批量上限；目录内文件更多时自动拆段。越大 restore 调用越少、HSM 越易满吞吐 |
| `PREWARM_RESERVE_GB` | **64**（≈容量 5%；1.12 TiB 对应 ~56 GiB，取整 64） | 生产者 self-throttle 余量；与消费者 `release` 形成背压，working set 有界 |
| `PREWARM_PRODUCER_POLL` | **15** s | 生产者批量轮询目录就绪的间隔（每目录独立 7200s 上限） |
| `PREWARM_CONSUMER_POLL` | **15** s | 消费者 `ready` 纯文件查询轮询间隔（不含 `lfs` 调用开销） |
| `PREWARM_WAIT_TIMEOUT` | **7200** s/目录 | 单目录召回等待上限；超时 best-effort 继续（可能冷读），不阻塞整轮 |
| `PREWARM_PARALLEL` | **16** | 仅 `release` 的客户端 RPC 并发，与 HSM 召回吞吐无关 |
| `PREWARM_BATCH_SIZE` | **16**（音频极大时调 4~8） | maker 切块粒度；也决定 `release` 的松散度与单批缓存占用 |
| `PREWARM_RELEASE` | **1**（开启） | 处理完即释放该批音频，腾空间供生产者继续；关掉则 working set 增长，仅当 FSx 足以容纳全量时使用 |
| 消费者就绪判定 | **state 文件纯查询**（`ready`） | 不调 `lfs hsm_action`，规避与其它租户缓存状态混淆 |

**一句话**：一个后台生产者按目录批量 `lfs hsm_restore` 并写就绪 state；消费者按批等就绪→处理→释放；容量上限下的吞吐靠目录级批量 + 生产者/消费者重叠，而非客户端并发；`RESERVE_GB` 背压把 working set 锁在 FSx 容量以内。
