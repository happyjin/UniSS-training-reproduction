# NCCL 分布式死锁分析（初始化阶段 · 模型加载 barrier 死锁）

> **一句话结论**：模型目录无 `.py` 但 `config.json` 残留 `auto_map`，导致 `get_model` 走进 HF 动态模块加载路径去找本地不存在的 `.py` 而抛 `OSError`；这段加载被 ms-swift 的 `safe_ddp_context` 套了一个 barrier，且抛异常的位置(yield)之前——**rank0 因是全局 master 被豁免、没有 barrier，得以走到抛异常那一行并靠 except 逃生走本地兜底；其余 15 个 rank 在抛异常之前先撞上 barrier 卡住，永远到不了抛异常那一步**，而 rank0 又不来凑齐 barrier，于是永久卡死。16 卡都缺同一个文件、本会抛同样的异常，差别只在"抛异常那行之前有没有 barrier"。修复：`get_model` 缺 `.py` 时清空 `config.auto_map` 并直接 import 本地类，使加载不再进入该路径。
>
> **根因修复**：2026-06-25 `bdd649d`（`e4fc13b`）｜ **对照日志**：死锁 `prodjob_hvvsq4mwi11vszmv7pm7c5mmka.log`、成功 `prodjob_lcct3bkkvnkq8866vgk33s9msa.log`（同模型 / 同拓扑，唯一变量=修复前后）
>
> 📌 **本文档结构**：前四章（现象 / 正常原理 / 异常流程 / 修复）**永远保持最新准确**。历史排查、被推翻的推断、其他死锁形态一律归入**附录**，不污染前文。

---

## 集群拓扑（固定）

```
[entrypoint] Detected 2 GPU node(s), 8 GPU(s) per node, 16 total GPUs
[entrypoint] USE_RAY=true   DEEPSPEED=zero2
[entrypoint] RAY_DEVICE_GROUPS={"nproc_per_node":8,"default":{"device":"GPU","ranks":[0..15],"workers":["default"]}}
```

- 1 个 CPU head node + 2 个 8-GPU worker node，共 **16 GPU = 16 rank**。
- swift driver 被钉在某个 worker 上（num_gpus=0 的 ray actor），由它 fan-out 拉起 16 个训练 actor（每 actor 独占 1 进程 1 GPU）。
- Ray 模式下每个 actor 的 `LOCAL_RANK=0`、`local_world_size=1`（是常态，不是 bug）。

---

## 一、确切现象

| 指标 | 观察值 | 含义 |
|------|--------|------|
| Job 状态 | RUNNING → 长时间无进度 → 超时/终止 | 进程不崩溃，卡死 |
| 卡死位置 | 各 rank 打完**模型结构**、`get_config` 完成后，卡在 `get_model` 内 | 模型加载阶段，训练循环从未开始 |
| `model_parameter_info` | 全日志**仅出现 1 次，且为 rank0** | 16 个 rank 里只有 rank0 走完 `get_model` |
| 其余 rank | rank1~15 停在加载阶段，无后续输出 | 卡在 barrier |
| CPU / GPU | **16 个 rank 全部 100%** | 全部卡在 NCCL barrier 忙等自旋（spin-wait），非空闲干等 |
| 持续时间 | 50+ 小时毫无进度 | barrier 无人配对的死等，非"加载慢"，永不恢复 |
| 典型末尾 | 模型结构打印 → `Gradient accumulation steps mismatch`（仅走通的 rank0）→ 零输出 | rank0 独自往前，其余早已卡住 |

**关键判据**：`via fallback`（`get_config` 完成）在死锁/成功任务里都是 16；而 `model_parameter_info`（`get_model` 返回后）死锁任务=**1**、成功任务=**16**。分叉点精确落在 **`get_config` 之后、`get_model` 内部**。

---

## 二、正常原理与流程

### 2.1 `swift sft --use_ray true` 初始化到训练（真实顺序）

> 顺序**纯依据 ms-swift 源码控制流**（单个 actor / 单进程视角）梳理，不依赖日志（16 个 rank 的日志交错，行号先后 ≠ 单 rank 执行先后）。

```
【进程入口】swift/cli/sft.py  __main__
 │
 ├─ try_use_single_device_mode()
 ├─ try_init_ray()
 │    └─ RayHelper.initialize(device_groups)
 │         └─ ray.init() + ResourceManager（按 nproc_per_node 建 placement group）
 └─ sft_main()  →  SwiftSft(args).main()

【类装饰】SwiftSft 被 @RayHelper.worker(group=['default']) 装饰
 │    __init__ 被替换为 new_init：
 │      · driver 进程：new_init → RayHelper._create_workers() 拉起 16 个 actor（rank 0~15），
 │                     每 actor 注入 RANK / WORLD_SIZE=16 / LOCAL_RANK=0 / MASTER_ADDR / CUDA_VISIBLE_DEVICES
 │      · actor 进程：直接执行原始 __init__（下方【A】）
 │
 ▼
【A · __init__ 阶段】每个 actor 各自执行（swift/pipelines/train/sft.py:26-31）
 │
 ├─ A1  super().__init__(args)   →  SwiftPipeline.__init__（pipelines/base.py:17）
 │        ├─ _parse_args()  →  BaseArguments.__post_init__（arguments/base_args/base_args.py:167）
 │        │     └─ get_dist_setting()：读 RANK/WORLD_SIZE/LOCAL_RANK（此处仅读环境变量，不建 PG）
 │        └─ _set_seed()
 │
 ├─ A2  _prepare_model_tokenizer()   ← @RayHelper.function，actor 内直接执行（sft.py:48）
 │        └─ args.get_model_processor()  →  get_model_processor()（model/register.py:516）
 │             └─ ModelLoader.load()（register.py:473）
 │                  with patch_get_dynamic_module(), patch_tp_plan(), ...:   ← ★关键 patch 生效范围
 │                    ├─ config = get_config(model_dir)（register_qwen_omnispeak.py）
 │                    │     └─ 目录无 .py → fallback：从本地 custom 包实例化 QwenOmniSpeakConfig
 │                    │        （⚠️ 此 config 仍带 config.json 里的 auto_map ── 埋雷点）
 │                    └─ model = get_model(model_dir, config, ...)   ← ★★模型加载，死锁在此，见第三章
 │
 ├─ A3  _prepare_template()   ← @RayHelper.function（sft.py:64）
 │        └─ use_model=True（Qwen3OmniTemplate）→ template.model = self.model
 │
 └─ A4  _prepare_flash_ckpt()

【B · main() → run() 阶段】driver 通过 execute_all_sync 分发，16 actor 并行执行
 │                            （SwiftPipeline.main → SwiftSft.run，sft.py:159-184）
 │
 ├─ B1  train_dataset, val = _prepare_dataset()（sft.py:97-123）
 │        ├─ _get_dataset()  →  load_dataset()（含 safe_ddp_context 文件级同步）
 │        └─ _encode_dataset()（sft.py:297）
 │             ├─ template.model = None            ← 编码期不碰 model（Avoid serializing）
 │             ├─ AddLengthPreprocessor / dataset.map(num_proc=8)   ← 仅 tokenize + 算长度
 │             └─ template.model = 恢复
 │
 ├─ B2  prepare_model()（TunerMixin，tuner.py:339）
 │        └─ tuner=full：requires_grad_(True) → freeze_parameters(全冻) → activate_parameters(bridge / talker.layers[0-1])
 │           （只改 requires_grad，不加载权重）
 │
 ├─ B3  logger.info('model: ...')  +  get_model_parameter_info()   ← 每个 rank 都打印（无 is_master 门控）
 │
 ├─ B4  trainer = TrainerFactory.get_trainer_cls(args)(model, args, template, dataset...)
 │        └─ Trainer.__init__ → accelerate PartialState/_setup_devices（惰性初始化 default PG 的候选点之一）
 │
 └─ B5  self.train(trainer)  →  trainer.train()
          └─ deepspeed.initialize() → 建 DeepSpeedEngine（内部 new_group 建 DP 子组）→ 训练循环
```

**要点（均为源码控制流事实）**：

1. **`get_model`（模型加载）在【A · __init__ 阶段】**，早于【B · run 阶段】的数据集/训练。死锁发生在 A2 的 `get_model` 内，训练循环从未开始。
2. **两个"prepare 模型"是不同的事**：A2 `_prepare_model_tokenizer`（**加载权重进显存**）vs B2 `prepare_model`（**冻结/激活参数，不加载**）。driver 自身两者都不做（`@RayHelper.function` 在 driver 的 __init__ 期返回 None）。
3. **数据集编码（B1）与 model 解耦**：`_encode_dataset` 期间 `template.model` 被临时置 None，只做 tokenize/算长度，不跑 model.forward。
4. **`patch_get_dynamic_module` 的生效范围覆盖整个 `load()`**（含 get_config + get_model），这是第三章 barrier 的来源。
5. **default process group 在 get_model 之前已初始化（✅ 实测坐实）**：swift 的 SFT+Ray 路径**没有显式 `init_process_group`**（仅 infer/export/megatron 有），dist 由底层库（transformers/accelerate/deepspeed）在更早处间接初始化。**实测证据**：在 `custom/plugin.py` 与 `modeling_qwen_omnispeak.py` 的 `__init__` 里 monkey-patch 抓取，`dist.is_initialized()=True`、`dist.get_world_size()` 25/25 次全返回 16（任务 `prodjob_s55ccbcp52119jn6tm8cj2u7vp`，见 `OOM_Analysis_and_Optimization.md`）。**结论**：get_model 执行时 dist 已初始化 → `safe_ddp_context` 走 **barrier 分支（有 NCCL 跨卡通讯）**，而非 FileLock。（精确到"哪一行代码初始化 dist"仍未坐实，但不影响本结论；见附录 5.1）

### 2.2 关键机制：`safe_ddp_context` 是一个"master 先干活、其余等待"的临界区（正常不死锁）

ms-swift 在 `ModelLoader.load()` 外层加了 `patch_get_dynamic_module()`（`swift/model/patcher.py:508`），把 HF 加载动态模块代码的函数 `get_cached_module_file` 包了一层 `safe_ddp_context`：

```python
# swift/model/patcher.py
def new_get_cached_module_file(name, ...):
    with safe_ddp_context(hash_id=str(name)):    # ← 临界区
        return origin_get_cached_module_file(...)
```

```python
# swift/utils/torch_utils.py:95-108（barrier 分支，dist 已初始化时走这里）
if not is_master():        dist.barrier()   # A 进入
if not is_local_master():  dist.barrier()   # B 进入
yield                                        #   （执行 get_cached_module_file）
if is_master():            dist.barrier()   # C 退出
if is_local_master():      dist.barrier()   # D 退出
```

**这段设计正常、对称，绝不是低级 BUG**。它的作用是保护共享资源的写操作（下载/编译模块文件、写所有 rank 共享的 `HF_MODULES_CACHE` 缓存目录），实现"**master 先独占执行、其余 rank 等待、做完再放行**"的临界区：

- **正常情况每个 rank 恰好 2 次 barrier**（单机多卡 rank==local_rank：rank0 走 C+D，其余走 A+B），完全对称，配成两轮全员同步。
- **执行语义**：其余 rank 先在 A 处 barrier **阻塞等待** → master 独占执行 `yield`（下载/写缓存这种"只能一个人做"的事，避免并发写冲突）→ master 做完后在 C 处 barrier **放行** → 全体继续。
- 只要 `yield` 正常返回，master 的 C/D 就会和其余 rank 的 A/B 正确配对，**永不死锁**。

⚠️ **它唯一的缺陷是"异常不安全"**：`@contextmanager` 的 `yield` **没有 `try/finally` 包裹**。一旦 `yield` 内部抛异常，master 就会**跳过它本该在 C 处发出的"放行 barrier"**——而其余 rank 早已在 A 处阻塞等待放行。这个"yield 抛异常"的情况，正常使用中几乎不会发生（所以这个上游项目长期没暴露该问题），但被我们的导出配置触发了（见第三章）。

### 2.3 什么时候会走进这个临界区

HF 的 `AutoModelForCausalLM.from_pretrained` 是否走"动态模块加载"，只看一个条件（`transformers/models/auto/auto_factory.py:208`）：

```python
has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
if has_remote_code and trust_remote_code:        # :223
    model_class = get_class_from_dynamic_module(...)   # :228 → get_cached_module_file
```

即：**只要 `config.auto_map` 存在且包含该类名**，就会走进 `get_cached_module_file`，从而触发 §2.2 那个 barrier。是否本地、是否禁用 HF Hub 都不影响这个判定。

---

## 三、异常发生的流程（根因）

### 3.1 触发链（每一环都有源码出处）

```
① 模型目录（本地 NVMe）无 modeling_qwen_omnispeak.py，但 config.json 残留 auto_map
   （custom/qwen_omnispeak get_config 的 fallback 只补了 config 对象，未清 config.auto_map）
      ↓
② 修复前 get_model 走 super().get_model()
   → HF AutoModelForCausalLM.from_pretrained(config, trust_remote_code=True)
      ↓
③ auto_factory.py:208  has_remote_code=True  → :228 get_class_from_dynamic_module
   → :612 get_cached_module_file(model_dir, "modeling_qwen_omnispeak.py")
      ↓
④ dynamic_module_utils.py:410 is_local=os.path.isdir(model_dir)=True
   :421 cached_file(model_dir, "modeling_qwen_omnispeak.py") → 文件不在本地 → 抛 OSError（:436-438）
      ↓
⑤ 该函数被 patch_get_dynamic_module 的 safe_ddp_context 包着
   OSError 在 safe_ddp_context 的 yield 内部抛出；yield 无 try/finally → 退出 barrier 被跳过
   （前提：该刻 dist 已初始化 → 走 barrier 分支；✅ 已由实测坐实 `dist.is_initialized()=True`、`get_world_size()=16`，见 §2.1 要点5）
      ↓
⑥ rank0 与非 rank0 进入 barrier 不对称 → 15 个 rank 永久卡死（见 3.2）
```

### 3.2 为什么只有 rank0 走异常流程、其余 rank 卡死

**这段临界区的规则是"排队 + 开门"**：barrier 规定 **rank0 先进去读文件（去填所有卡共享的缓存），其余 15 卡先在门口等；rank0 读完发一个"放行 barrier"把门打开，其余卡才进去读缓存**。目的是避免 16 张卡同时抢写同一个缓存文件。

```python
if not is_master():        dist.barrier()   # A ← 非 rank0 在这里排队等"开门"
if not is_local_master():  dist.barrier()   # B
yield                        # ← 真正读 .py 文件在这里；文件缺失 → 抛 OSError ★
if is_master():            dist.barrier()   # C ← rank0 在这里"开门"放行别人
if is_local_master():      dist.barrier()   # D
```

**正常（文件存在）——两条流程正确配对，不死锁：**
```
rank0        : 跳过A、B → yield(读文件、填缓存，成功) → C、D 发barrier"开门" ✅
rank1~15     : A 处 barrier 排队等"开门" ……（rank0 开门）…… → 进入 yield(读缓存) ✅
              rank0 的 C/D  ⟷  其余的 A/B  正好配对，全体放行
```

**异常（文件缺失）——rank0 溜走时忘了开门：**
```
rank0        : 跳过A、B → yield(读文件→缺失→抛 OSError ★)
               ↑ 因 yield 无 try/finally，异常直接跳出，C、D 根本不执行 → 没开门！
               → 异常被外层 try/except 接住 → 走兜底本地加载 → ✅ 成功
rank1~15     : A 处 barrier 排队等"开门" …… 永远等不到 rank0 开门 …… 🔒 卡死
               （它们还堵在门口，压根没走到 yield，没机会读文件、没机会抛异常）
```

**逐条对应你的疑问：**

1. **rank0 为什么去读那两个 .py、rank1~15 为什么"不读"？**
   不是不读，是**排队**。设计让 rank0 先读（去填共享缓存），rank1~15 先在 A 处 barrier 等；等 rank0 读完开门才轮到它们。所以 rank0 是**第一个、失败时也是唯一一个真正碰到文件的**。

2. **rank0 为什么没 finally、导致没发出那两个 barrier？**
   代码是 `yield` 后直接跟 C、D，**没有 `try: yield finally: barrier()`**。Python 中 yield 一抛异常，控制流立刻离开这个上下文管理器，**yield 之后的 C、D 被跳过**。rank0 本该用 C、D 给等待的卡"开门"，异常让它逃走时没做这件事 —— 这就是这段代码的**异常安全缺陷**。

3. **为什么只有 rank0 走异常流程、其余走正常流程被卡死？**
   因为 rank0 是**唯一先走到 yield（读文件那行）的**，只有它真正尝试读、真正撞上 OSError、真正走异常逃生。rank1~15 **还堵在 A 处的 barrier，根本没走到读文件那一步**，自然没机会抛异常；它们只是"正常地排队等开门"，而开门信号永远不来 → 卡死。

**一句话**：同一段代码，barrier 规定"rank0 先进去读、其他人门口等"。文件缺失时，先进去的 rank0 撞墙后从异常出口溜走了，**却忘了开门（发 C/D 放行 barrier）**，把门外等着的 15 个人永远锁在外面。

> 这不是低级 BUG：正常（文件存在、yield 不抛异常）时两条流程完美配对，绝不死锁——这正是该上游项目长期没暴露此问题的原因。触发它需要一个不常见前置条件：config 带 auto_map 但本地缺 .py（由我们导出模型时留了 auto_map 却没放 .py 造成，见 §3.1）。`is_local_master()`（B/D，依赖 local_rank）在此次要；决定"谁先进、谁门口等"的是 rank0 的**全局 master** 身份。Ray 令 local_rank 全 0 只是背景，不是死锁原因——成功任务 local_rank 同样全 0。

**这不是"后门"或低级 BUG，而是临界区的异常安全缺陷**（见 §2.2）：`if not is_master(): barrier()` 让其余 rank 先等待、master 独占 yield 干活，本是正确的"master 先干活"设计；只是没料到 yield 会抛异常，导致 master 跳过了本该发出的放行 barrier。正常（yield 不抛异常）时它完全对称、不死锁——这正是该上游项目长期没暴露此问题的原因。**触发它需要一个不常见的前置条件：config 带 auto_map 但本地缺 .py（由我们的导出配置造成，见 §3.1）。**

> `is_local_master()`（B/D 两行，依赖 local_rank）在这里是次要的；决定"谁在 yield 前等待、谁独占 yield"的是 rank0 的**全局 master** 身份。**Ray 令 local_rank 全 0 只是背景，不是死锁原因**——成功任务的 local_rank 同样全 0。

### 3.3 与现象、证据吻合

- **可证伪预测**：若机制成立，唯一能走通的**必定是 rank0**。→ 日志命中：死锁任务里唯一打出 `model_parameter_info` 的进程 `rank: 0`；其余 rank1~15 全部卡住。✅
- **50+ 小时无进度**：barrier 无人配对的死等，非加载慢，永不恢复。
- **16 rank CPU/GPU 全 100%**：`dist.barrier()` 底层是 NCCL AllReduce，卡住的 rank 在 GPU 上忙等自旋。
- **双日志计数铁证**（同模型同拓扑，唯一变量=修复前后）：

| 计数项 | 死锁（修复前） | 成功（修复后） |
|--------|------|------|
| `via fallback`（get_config 完成） | 16 | 16 |
| `model_parameter_info`（get_model 返回） | **1（rank0）** | **16** |
| `'loss':`（进入训练） | **0** | 71 |

---

## 四、修复方案

**修复对象**：`custom/qwen_omnispeak/register_qwen_omnispeak.py` 的 `QwenOmniSpeakLoader.get_model()`
**提交**：`bdd649d`（`e4fc13b`，2026-06-25）

### 4.1 修复前（触发源）

```python
def get_model(self, model_dir, config, processor, model_kwargs):
    from transformers import AutoModelForCausalLM
    try:
        self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        model = super().get_model(model_dir, config, processor, model_kwargs)  # ← config 带 auto_map → 走 HF 动态模块 → OSError → barrier
    except Exception:
        from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration
        model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=self.torch_dtype, trust_remote_code=True, **model_kwargs)
    return model
```

### 4.2 修复后

```python
def get_model(self, model_dir, config, processor, model_kwargs):
    from transformers import AutoModelForCausalLM
    has_modeling_py = os.path.isfile(os.path.join(model_dir, 'modeling_qwen_omnispeak.py'))
    has_config_py  = os.path.isfile(os.path.join(model_dir, 'configuration_qwen_omnispeak.py'))
    if has_modeling_py and has_config_py:
        # checkpoint 自带 .py：正常走 HF trust_remote_code
        self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        model = super().get_model(model_dir, config, processor, model_kwargs)
    else:
        # checkpoint 无 .py：直接从本地 custom 包 import，并清空 auto_map
        from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration
        saved_auto_map = getattr(config, 'auto_map', None)
        if saved_auto_map:
            config.auto_map = {}          # ← 关键：使 has_remote_code=False
        try:
            model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
                model_dir, config=config, torch_dtype=self.torch_dtype,
                trust_remote_code=True, **model_kwargs)
        finally:
            if saved_auto_map is not None:
                config.auto_map = saved_auto_map
        if getattr(model, '_auto_class', None) is None:
            model._auto_class = 'AutoModelForCausalLM'
    return model
```

### 4.3 为什么解死锁

清空 `config.auto_map` → HF `has_remote_code=False`（`auto_factory.py:208`）→ **不调 `get_class_from_dynamic_module` → 不调 `get_cached_module_file` → 不去找本地缺失的 .py → 不抛 OSError → 不进入 `safe_ddp_context` barrier**。没有"barrier + 异常"的组合，rank0 与非 rank0 的不对称就不存在。16 个 rank 走同一条无跨进程同步的本地 import 路径 → 全部走通（`model_parameter_info` 16/16）→ 进入训练。

> 该修复未改 Ray 拓扑、也未改 ms-swift 那个有缺陷的 `safe_ddp_context`（它在 Ray 单进程 local_rank=0 + 异常场景下本就脆弱），而是从触发条件下手：让加载不再踏进这条会抛异常的 barrier 路径。

---

## 五、附录

> 以下为历史排查、被推翻的推断、其他死锁形态与工具方法，仅供追溯，**不代表当前结论**。当前结论以第一~四章为准。

### 5.1 根因认知的三版演进（含被推翻的推断）

| 版本 | 当时结论 | 为何被推翻 |
|------|---------|-----------|
| v1 | "各 rank 加载耗时错位 → 传导到 load_dataset barrier → barrier↔new_group 死锁" | 模型全本地、HF 禁用、加载本身很快；且 `model_parameter_info` 只有 1 个 rank，说明根本没到 dataset 阶段 |
| v2 | "HF 去 Hub 拉取远程代码，无 Hub 直连而 hang" | 模型全程本地、HF 已禁用，从不联网；`auto_map` 触发的是**本地动态模块加载代码路径**，非下载 |
| v3 | "Ray 令 local_rank 全 0 → barrier 角色/次数错乱" | 逐 rank 算账每个都是 2 次 barrier，次数不乱；成功任务 local_rank 同样全 0 却不死锁 |
| **v4（当前）** | **OSError 在 barrier 中间抛出 + rank0 进入豁免 → 不对称死锁**（第三章） | 有可证伪预测"走通的必是 rank0"并被日志命中 |

**这一环已由实测坐实（v4 不再有缺口）**：第三章要成立，需要 `get_model`（A2 阶段）执行时 `dist.is_initialized()=True`（`safe_ddp_context` 才走 barrier 分支而非 FileLock）。**实测证据**（任务 `prodjob_s55ccbcp52119jn6tm8cj2u7vp`，在 `custom/plugin.py` 与 `modeling_qwen_omnispeak.py` 的 `__init__` 里 monkey-patch）：`dist.is_initialized()=True`、`dist.get_world_size()` 25/25 次全返回 16。故 get_model 时 dist 确已初始化 → 走 barrier 分支。
- 曾有静态源码分析推测"dist 到 run 阶段 Trainer(`_setup_devices`/`PartialState`) 才初始化"——**此推测被上述实测推翻**：dist 在模型 `__init__` 期就已 initialized。swift SFT 路径无显式 `init_process_group`，实际由底层库（transformers/accelerate/deepspeed）在更早处间接初始化。
- **仍未坐实的只剩"精确到哪一行代码初始化 dist"**（本地 transformers 版本与线上不一致，跨 accelerate/deepspeed 库行为无法静态断定），但这不影响结论——实测已证明"到 get_model 时 dist 就是初始化好的"。


### 5.2 早期记录的其他死锁形态（历史，多为 v1/v2 时期的误判或已修复项）

- **barrier↔new_group 循环死锁（旧假说）**：曾认为 rank0 数据加载快、抢先进 `deepspeed.initialize()→new_group()`，其余卡在 load_dataset barrier。py-spy（`prodjob_pldld9x4`）曾显示 rank0 在 `_new_process_group_helper`、其余在 `barrier`。→ 现认为该现象是"只有 rank0 走通"的下游表现，真正卡点在更早的 get_model（见第三章）。
- **TORCHELASTIC TCPStore Hang（已修复）**：`common_args.sh` 在 `USE_RAY=true` 时误 export `NNODES/NODE_RANK/MASTER_*` → swift 误判 `use_torchrun()` → 污染所有 actor env → 全 rank connect 不存在的 master 而 hang。修复：USE_RAY guard 跳过这些变量。现已确认防御生效（`MASTER_ADDR/RANK/WORLD_SIZE` 均 unset）。
- **debug_hooks.py 的 dist.barrier patch（已移除）**：曾在 `custom/debug_hooks.py` hook `init_process_group` 并 patch `dist.barrier`（含 `threading.Lock()`），一度被列为死锁原因。该调试代码已移除，非当前根因。
- **CUDA_VISIBLE_DEVICES / local_world_size=1**：Ray runtime_env 不透传 `CUDA_VISIBLE_DEVICES`（Ray issue #49147），曾怀疑导致进程组错误。对照发现成功任务同样如此，已排除为死锁原因。

### 5.3 已排除的错误假设

| 假设 | 排除依据 |
|------|----------|
| 依赖版本 / 镜像不一致 | 成功与失败任务版本相同 |
| 数据 JSONL 问题 | 同一数据集 `bridge_alignment_all`，格式一致 |
| RAY_DEVICE_GROUPS 格式错误 | JSON 解析 OK，成功任务格式相同 |
| CUDA_VISIBLE_DEVICES 未传 | 成功任务也未设置 |
| local_world_size=1 | 成功任务也是 1 |
| 某 rank OOM 静默崩溃 | 进程全 alive，加载阶段显存远未满 |
| Flash Attention 3 kernel bug | 死锁在加载阶段，未进入 forward；FA3/FA2 差异是修复后暴露的后续问题，与本死锁无关 |
| DDP init `_verify_param_shape` 不同步 | 本问题未到 DDP init 即死锁 |

### 5.4 诊断方法论

**分类决策树**：
```
GPU 利用率？
├─ 100% flat → NCCL collective 自旋（本案：barrier 卡死）
└─ 0% → CPU 级 hang
        ├─ 显存 ~17%（有模型） → Gloo monitoredBarrier / DDP init hang
        └─ 显存极低（无模型） → TCPStore hang
```

**标准流程**：① 日志定位最后输出 → 卡在哪个阶段；② `nvidia-smi` 看 GPU/显存/CPU；③ py-spy 取所有 rank 栈帧；④ 对比 rank0 与其余是否处于不同同步点；⑤ 关键计数（`via fallback` / `model_parameter_info` / `'loss'`）判断分叉点。

**常用命令**：
```bash
# 进 worker 看 GPU
kubectl exec -n anyscale-operator <worker-pod> -c ray -- nvidia-smi
# py-spy 抓某 actor 栈帧（含 native）
kubectl exec -n anyscale-operator <worker-pod> -c ray -- bash -c \
  "/home/ray/anaconda3/bin/py-spy dump --native --pid \$(pgrep -f 'SwiftSft' | head -1)"
# 看 actor 环境变量
cat /proc/<pid>/environ | tr '\0' '\n' | grep -E '^(NCCL|FI_|MASTER_|RAY_|LOCAL_)' | sort
```

### 5.5 NCCL collective 死锁的数学本质

N 个 rank，每 rank 通信序列 `S_i`。可达一致终态需：① 全序匹配（每步同一 collective）；② 无交叉依赖（barrier 不被跨 context 混淆匹配）；③ wait-for graph 无环。本案破坏的是 ①：rank0 因异常跳过了其余 rank 已发出的进入 barrier，参与者数量不齐（15≠16），barrier 永远无法完成。

### 5.6 关键 Commit / 文件

| Commit | 日期 | 说明 |
|--------|------|------|
| `bdd649d`（`e4fc13b`） | 2026-06-25 | ✅ 修复本死锁：`get_model` 缺 `.py` 时清空 `auto_map`、直接 import 本地类 |
| `be48e27` | 2026-06-26 | FA3/FA2 兼容 + DataLoader 调整（**修复死锁后暴露的后续问题**，与本死锁无关） |
| `ed15210` | 2026-06-18 | 回退 L4 head node 策略 |

| 文件 | 相关内容 |
|------|---------|
| `custom/qwen_omnispeak/register_qwen_omnispeak.py` | `QwenOmniSpeakLoader.get_config / get_model`（修复点） |
| `swift/model/register.py:473` | `ModelLoader.load()` 外层 `patch_get_dynamic_module()` 生效范围 |
| `swift/model/patcher.py:508` | `patch_get_dynamic_module`：给 `get_cached_module_file` 包 `safe_ddp_context` |
| `swift/utils/torch_utils.py:91-108` | `safe_ddp_context` barrier 分支（yield 无 try/finally） |
| `transformers/dynamic_module_utils.py:410,421,436` | `get_cached_module_file`：本地缺 .py 抛 OSError |
| `transformers/models/auto/auto_factory.py:208` | `has_remote_code` 判定 |
| `anyscale/train/dispatch_train.py` | driver 钉在 worker[0]、clean_env 防御 |

### 5.7 验证记录

| Job ID | 日期 | 模型 | 状态 | 备注 |
|--------|------|------|------|------|
| `prodjob_hvvsq4mwi11vszmv7pm7c5mmka` | 修复前 | qwen3_omnispeak_d | 死锁 | `model_parameter_info`=1(rank0)，`'loss'`=0 |
| `prodjob_lcct3bkkvnkq8866vgk33s9msa` | 修复后 | qwen3_omnispeak_d | 成功 | `model_parameter_info`=16，`'loss'`=71，正常训练 |
| `prodjob_s99r9w2f88pwe1vlc9mkd1xv7j` | 2026-06-25 | qwen2_5_3b_omnispeak_d | 成功 | 日志全开，16 rank 全走通并进入训练 |
| `prodjob_z25yehia4xk6t75i9k1mwy1f2x` | 2026-06-20 | qwen2_5_3b_omnispeak_d | 死锁 | 单节点 8 卡复现，同一 get_model barrier 卡点 |
