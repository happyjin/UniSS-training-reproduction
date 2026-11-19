# SimulS2ST-Omni（Qwen2.5-Omni-3B + DualCodec talker）三阶段训练与流式同传推理

迁移自 `SimulS2ST-Omni-stage123`。源项目用自研的 `scripts/train.py` + HF Trainer 训练，这里改为 `swift sft` 启动：
源训练代码原样迁入 `custom/simuls2st_omni/train/`，新增的接入层 `custom/simuls2st_omni/integration/` 只负责把它接到
swift 上（model_type / template / 数据集 / tuner / checkpoint 格式），不改训练语义。

单卡下与源项目**逐步一致**（同一库版本下 loss、grad_norm、eval_loss、每个可训张量逐位相同），多卡下修正了源项目
梯度不同步的问题（§6.1）。验证方法与结果见 §7。

| 阶段 | 训练范围 | 数据 | 默认产出（相对数据根） |
|---|---|---|---|
| Stage 1 talker warmup | thinker 冻结，talker 全参 | Emilia DualCodec TTS | `outputs/checkpoint/stage1_talker_warmup` |
| Stage 2 多任务 | thinker LoRA r32 + talker 全参；ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5 | 5 类任务，按步抽任务 | `outputs/checkpoint/stage2_multitask_full` |
| Stage 3 流式 | 先合并 Stage 2，再挂 thinker / talker LoRA r8（talker 含 `thinker_to_talker_proj`）；m ∈ {1..12} | 1000ms WAIT/WRITE 轨迹 | `outputs/checkpoint/stage3_streaming` |

---

## 1. 文件清单

### `custom/simuls2st_omni/`

| 路径 | 内容 |
|---|---|
| `train/` | 源 `src/train/` 原样迁入（模型包装、阶段策略、数据集、collator、sampler、prompt），只改了 import 前缀 |
| `train/native_train.py` | 源 `scripts/train.py` 原样迁入，作对拍参照；相对路径改为相对 `SIMULS2ST_ROOT` |
| `inference/`、`agents/`、`voicebox/`、`model/` | 源推理代码原样迁入：离线 TTS、SimulEval 流式同传 agent、VoiceBox / Vocos / DualCodec |
| `integration/register.py` | swift 接入：`simuls2st_omni` model_type / template / tuner、指针行数据集、源格式 checkpoint 的保存与续训 |
| `integration/plan.py` | 把源 batch sampler 按 rank 物化为「一行 = 一个 batch」的指针行 |
| `integration/swift_args.py` | 阶段配置的 `training:` 段 → swift 命令行参数（与源 `build_training_args` 同规则） |
| `integration/stage_config.py`、`latency.py` | 阶段配置读取与交接路径覆盖；Stage 3 延迟倍数的稳定抽取 |
| `scripts/probe_stage1_overfit.py` | 源 Stage 1 监督位置过拟合探针 |
| `make_smoke_fixture.py` | tiny 随机模型 + 源目录布局的迷你语料（冒烟 / 对拍用） |
| `tests/` | 源单测（32 项）+ 接入层单测；`tests/parity/` 三阶段对拍工具 |
| `requirements-infer.txt` | 推理额外依赖（含源清单漏列的包） |

### `pipelines/simuls2st_omni/`

| 路径 | 内容 |
|---|---|
| `configs/stage{1,2,3}.yaml` | 与源项目 `configs/` **逐字相同**，训练行为的唯一事实源 |
| `common_args.sh` | 路径解析、运行场景（prod / smoke）、分布式环境、启动 swift |
| `train_stage{1,2,3}.sh` | 各阶段入口 |
| `ktrain_entry.sh` | ktrain 集群入口（§3.3） |
| `run_infer.sh` | 流式同传推理（与源 `scripts/run_infer.sh` 参数相同） |

其他：`ktrain/ktrainjob_simuls2st_omni_stage{1,2,3}_2node.yaml`（KtrainJob 模板），`vendor/SimulEval/`（钉在 536de825，
源项目漏拷的 `simuleval/data/` 已从上游同一提交补齐），`custom/plugin.py` 加了一行注册。

---

## 2. 配置与数据

### 2.1 阶段配置

`configs/stage{1,2,3}.yaml` 的格式与源项目相同：

* `model` / `data` / `sampler` 段由接入层直接读取（`SIMULS2ST_CONFIG` 指向该文件）；
* `training` 段由 `swift_args.py` 翻译成 swift 参数。规则与源 `build_training_args` 一一对应：只取同一份 `supported` 键，
  `min_lr_ratio` → `lr_scheduler_kwargs.min_lr_rate`，`per_device_*_batch_size=1`（batch 由 sampler 决定），
  有 `eval_steps` 且有验证集时按步评估，Stage 2 不评估。
* swift 改过一批 TrainingArguments 默认值（`adam_beta2` 0.95、`weight_decay` 0.1、`lr_scheduler_type` cosine、
  `logging_first_step` 等），翻译时全部钉回源（HF）的取值；`adam_beta2` 不钉住时第 2 步起参数更新就与源不同。
* 优化器固定 `adamw_torch`：torch ≥ 2.8 时 HF 默认改选 fused 实现，源环境（torch 2.6）用的是非 fused。

配置里的相对路径（`models/...`、`data/...`、`outputs/...`）与源项目一样相对「数据根」解析，数据根取 `SIMULS2ST_ROOT`
（缺省本仓库根）。试跑可用环境变量覆盖：`MAX_STEPS`、`NUM_TRAIN_EPOCHS`、`LEARNING_RATE`、`SAVE_STEPS`、
`LOGGING_STEPS`、`EVAL_STEPS`、`DATALOADER_NUM_WORKERS`。

### 2.2 目录布局（相对数据根，与源项目 docs/TRAINING.md 相同）

```
models/SimulS2ST-Omni/offline/          # 基座（HA-SA-ki/SimulS2ST-Omni 的 offline/）
data/emilia/dualcodec/                  # Stage 1，以及 Stage 2 的 TTS：manifest/<LANG>/*.jsonl + codes/<LANG>/*.npy
data/manifests/{asr_librispeech, asr_common_voice, s2tt_wenetspeech, s2tt_gigast, s2tt_covost, mt_wmt17}/
data/asr/librispeech/  data/asr/common_voice/  data/wenetspeech/audio_tar/  data/gigaspeech/  data/s2tt/covost/hf/
data/s2st/dualcodec_rechunk_d/          # Stage 2 离线 S2ST
data/s2st/dualcodec_rechunk_d_2kh/      # Stage 3 流式轨迹
data/si_s4/audio/                       # S2ST 源语音
```

### 2.3 权重

* 基座：`hf download HA-SA-ki/SimulS2ST-Omni --local-dir models/SimulS2ST-Omni`（offline/ 约 12 GB；推理另需
  voicebox/、dualcodec/、w2v/ 约 5 GB）。ktrain 上用要先上传到 S3。
* 源项目训完的三个阶段（ap-northeast-1）：
  `s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_share/Qwen2.5_3B_DualCodec/`
  下 `stage1_talker_warmup/`、`stage2_multitask_full/`、`stage3_mix_v2/`。checkpoint 格式与本仓库产出相同，
  可直接作为 Stage 2 / Stage 3 的起点或推理输入。

---

## 3. 训练

### 3.1 本地 / 单机

```bash
bash pipelines/simuls2st_omni/train_stage1.sh
bash pipelines/simuls2st_omni/train_stage2.sh      # 默认从 outputs/checkpoint/stage1_talker_warmup 接 talker
bash pipelines/simuls2st_omni/train_stage3.sh      # 默认合并 outputs/checkpoint/stage2_multitask_full

# 数据不在仓库根下
SIMULS2ST_ROOT=/data/simuls2st bash pipelines/simuls2st_omni/train_stage1.sh
# 改交接目录（例如接源项目训完的 Stage 1）
SIMULS2ST_INIT_TALKER_FROM=/data/Qwen2.5_3B_DualCodec/stage1_talker_warmup bash pipelines/simuls2st_omni/train_stage2.sh
# 续训（目录里需有 trainer_state.json）
RESUME=outputs/checkpoint/stage1_talker_warmup/checkpoint-2000 bash pipelines/simuls2st_omni/train_stage1.sh
```

卡数自动检测（`NPROC_PER_NODE` 可覆盖）；多机时 `NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT` 由外部导出。

### 3.2 产出

每个 `checkpoint-N/` 与训练结束后的 `output_dir` 根目录都是源格式：

| 文件 | 内容 |
|---|---|
| `talker_components.pt` | talker 全量 state_dict（有 talker LoRA 时为 PEFT 包装后的 state_dict + LoRA 配置） |
| `thinker_lora/`、`talker_lora/` | PEFT adapter（有对应 LoRA 时） |
| tokenizer / processor 文件、`simuls2st_meta.json` | 阶段、配置路径 |
| `trainer_state.json`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth` | 仅 `checkpoint-N/`，续训用 |

源项目的推理脚本、下一阶段的 `init_talker_from` / `merge_adapters_from` 都能直接读。

### 3.3 ktrain 集群

模板：`ktrain/ktrainjob_simuls2st_omni_stage{1,2,3}_2node.yaml`（2 节点 × 8 H200，`framework: pytorch`）。
提交方式与 streaming_si 相同，见 `ktrain/SUBMIT.md`。

`ktrain_entry.sh` 按 `SIMULS2ST_MAP_*` 在节点 NVMe 上用软链拼出 §2.2 的目录树，并令 `SIMULS2ST_ROOT` 指向它，
所以阶段配置**原样使用、不改任何路径**：

```yaml
SIMULS2ST_STAGE: "stage1"
MODEL_PATH: "s3://<bucket>/<key>/SimulS2ST-Omni/offline/"          # 自动映射为 models/SimulS2ST-Omni/offline
SIMULS2ST_MAP_EMILIA: "data/emilia/dualcodec s3://<bucket>/<key>/emilia/dualcodec/"   # "<相对路径> <来源>"
OUTPUT_ROOT: "/nvme/output/<你>/simuls2st_omni"                      # 映射为 outputs/
```

* `SIMULS2ST_MAP_*` 第二个 token 是 `s3://` 时，submit.sh 自动同步到 `/nvme/<key>`（同步任务名 `simuls2st-map-<名字>`）；
  也可以写 NVMe 绝对路径或 `fsx://`。
* Stage 2 / Stage 3 默认从 `OUTPUT_ROOT/checkpoint/stage{1,2}_...` 接上一阶段（需在同一组节点上，多节点时自动
  `--save_on_each_node`）；`SIMULS2ST_INIT_TALKER_FROM` / `SIMULS2ST_MERGE_ADAPTERS_FROM` 可写 `s3://`，会被同步。
* 源项目各阶段按单机 8 卡写。16 卡时每步的 batch 数翻倍：Stage 1 / 3 每 epoch 步数减半（学习率调度按步数缩放），
  Stage 2 的 `max_steps` 固定，总 token 是源的 2 倍，要对齐就设 `MAX_STEPS=11719`。

### 3.4 冒烟

```bash
python custom/simuls2st_omni/make_smoke_fixture.py --meta_dir <含 offline/*.json 的目录> --out /tmp/simuls2st_smoke
python custom/simuls2st_omni/tests/parity/make_parity_configs.py --fixture /tmp/simuls2st_smoke --out /tmp/p --sides sw
for n in 1 2 3; do SCENE=smoke SIMULS2ST_CONFIG=/tmp/p/cfg/sw/stage$n.yaml bash pipelines/simuls2st_omni/train_stage$n.sh; done
```

`--meta_dir` 只要发布仓库的配置与 tokenizer 小文件：
`hf download HA-SA-ki/SimulS2ST-Omni --include 'offline/*.json' 'offline/*.txt' 'offline/*.jinja' --local-dir <dir>`。
`SCENE=smoke` 固定 CPU 单进程、3 步。

---

## 4. swift 接入方式

| 源 | swift 侧 |
|---|---|
| `load_base_model`（启用 talker、删 token2wav / visual、重建 DualCodec 词表） | `SimulS2STOmniLoader.get_model` 直接调用源函数 |
| `STAGE_BUILDERS` + `build_wrapped_model`（合并 → 接入 talker → 挂 LoRA → 冻结 → 包装） | `--tuner_type simuls2st_omni`：`prepare_model` 调用源函数 |
| batch sampler：token-budget 打包 / Stage 2 按步抽任务，sampler 内做 DDP 分片 | `plan.py` 按 rank 物化为指针行 `row[s*W + r]`；swift 以 batch_size=1、不打乱读取，`BatchSamplerShard` 的 `range(rank, N, W)` 恰好还原每个 rank 的序列 |
| dataset `__getitem__` + collator | template 的 `_encode` 取源 dataset 样本，`data_collator` 调源 collator（swift 的 encode 在 inference_mode 里，张量必须在 collator 里建） |
| `compute_loss`：四个跳批阈值、分任务 loss | `template.compute_sft_loss`；分任务 loss 写进 swift 的 `custom_metrics`（`loss/<task>`、`loss/text`、`loss/audio`、`task_frac/<task>`） |
| `save_model`：talker_components.pt / LoRA 目录 | `template.save_callback`（swift `_save_model` 的委托钩子）；训练结束另导出到 output_dir 根目录 |
| `_load_from_checkpoint` | 补丁 `SwiftMixin._load_from_checkpoint`，只对本模型生效 |

另外两处必须声明的行为：template 的 `model_accepts_loss_kwargs = False`（swift 默认 True，会把 loss 乘以进程数、
且 HF 不再按梯度累积步数除 loss）；梯度检查点在 tuner 里按源做法开启（swift 自己的开关会改写 decoder 层 forward，
统一 `--gradient_checkpointing false`）。

**不要加 `--deepspeed`**。ZeRO 会把未参与反传的参数梯度补成 0，AdamW 随之更新它们（动量衰减 + 权重衰减），
与源不同；这里与源一样用 DDP。

---

## 5. 与源项目的差异

### 5.1 修正（单卡下不出现）

1. **多卡梯度同步**。源 `compute_loss` 调 `model.module(...)`，绕过了 DDP 的 forward，梯度从未 all-reduce：
   多卡训练时每张卡独立训练、只存 rank0。实测源 2 卡 Stage 1 训 4 步后两卡 talker 参数和分别为 131.39 / 135.22。
   这里经 DDP forward，两卡参数逐位相同。
2. **`ddp_find_unused_parameters`**：Stage 2/3 设为 true。文本任务 batch 不经过 talker；thinker LoRA 按名字也命中了
   音频塔的 q/k/v，在无音频 batch 上不参与计算。不设时，遇到一个有参数未参与的 batch 后，下一步即报
   "Expected to have finished reduction"。
   未参与的参数梯度为 None，AdamW 跳过它们，与单卡源行为一致。
3. **LoRA 初始化可复现**：源挂 LoRA 前没有播种，每次运行的 LoRA A 不同；这里固定用训练 seed。
4. **Stage 3 延迟倍数 m**：源用 `hash((seed, index, "m"))`，字符串哈希随进程随机化，且 Python 3.11 起算法变了，
   同一份数据每次运行（以及续训前后）抽到的 m 都不同。这里改用 sha256，分布相同、跨进程与 Python 版本可复现。
   `SIMULS2ST_LATENCY_HASH=python` 恢复源行为。
5. **多卡跳批**：源在超阈值时不做 forward、直接返回 0；多卡下这会让其他 rank 卡死在 all-reduce。这里仍做 forward、
   loss 乘 0（单卡仍与源相同）。

### 5.2 保留的源行为（对拍需要逐样本一致，未改）

* **验证集取错下标**：源 `build_sampler(indices=...)` 产出的是子集内位置，DataLoader 直接拿它当下标，Stage 1 / 3
  的验证集实际评的是前 `eval_samples` 条（训练样本）；多源 TTS（`sources:`）时训练集也会错位。
  `SIMULS2ST_MAP_SUBSET_INDICES=1` 改为按真实下标取样。
* **每个 epoch 的 batch 顺序相同**：源 sampler 从未收到 `set_epoch`。
* **Stage 3 不读源音频**：源 `StreamingS2STDataset` 不产出 `audio_chunks`，collator 的 `feature_extractor=None`，
  thinker 只看到 `<|AUDIO|>` 占位 token，`si_root` 未被使用。源仓库桶里的 `stage3_mix_v2` 是另一份配置训的，
  不确定是否同样如此；按本配置训 Stage 3 前建议先确认这是否符合预期。

### 5.3 观测上的差异

* swift 日志里的 `learning_rate` 是下一步的值（比源日志早一位）；每步实际生效的学习率逐位相同（已核对）。
* 步数不是 `eval_steps` 整数倍时，swift 在最后一步多评估一次。
* 分任务 loss 是跨卡均值（源只记 rank0 本地）。
* `ddp_timeout` 用 swift 的默认值（源为 1800 秒，Stage 1 全量数据集构造可能超过）。

---

## 6. 推理

```bash
pip install -r custom/simuls2st_omni/requirements-infer.txt && pip install -e vendor/SimulEval --no-deps
SOURCE=source.txt TARGET=target.txt \
  BASE=models/SimulS2ST-Omni/offline MERGE=<Stage 2 产出> CKPT=<Stage 3 产出> \
  bash pipelines/simuls2st_omni/run_infer.sh
```

`MERGE` / `CKPT` 可以是本仓库训练的 output_dir 或 checkpoint-N，也可以是源项目的 `stage2_multitask_full` /
`stage3_mix_v2`。离线 TTS：`python custom/simuls2st_omni/inference/infer_omni_talker_tts.py --help`
（需 `PYTHONPATH=custom` 与 `SIMULS2ST_MODEL_ROOT`）。

`requirements-infer.txt` 补上了源清单漏列、实测导入需要的 `hyperpyyaml`、`ruamel.yaml`、`setuptools<81`
（`pkg_resources`）与 SimulEval 的 `pandas`。`descript-audiotools` 要求 `protobuf<3.20`，会降级训练环境的 protobuf，
推理建议单独建环境。agent 的 `--dtype` 与 SimulEval 的同名参数冲突，不要传（agent 默认 bf16）。

---

## 7. 验证

### 7.1 对拍（CPU，tiny 夹具，`tests/parity/run_parity.sh all`）

每个阶段跑三份：src（源代码 + 源环境 Python 3.10 / transformers 4.52.3 / peft 0.18.1 / torch 2.6）、
nat（迁入代码 + HF Trainer + 本仓库环境 3.12 / 4.57.3 / 0.15.2 / 2.8）、sw（swift sft）。阶段间各用自己上一阶段的产物。

| 对比 | Stage 1（20 步） | Stage 2（12 步） | Stage 3（7 步） |
|---|---|---|---|
| nat vs swift（只差接入层） | 逐位一致 | 逐位一致 | 逐位一致 |
| src vs swift（另含库版本差异） | batch 序列一致；grad_norm ≤ 4.3e-5、参数 ≤ 6.4e-4 | ≤ 4.8e-5 / 9.4e-4 | ≤ 2.4e-5 / 7.2e-4 |
| 2 进程：nat（经 DDP forward）vs swift | 逐位一致 | 逐位一致 | 逐位一致 |
| 2 进程：每个 rank 每步的 batch vs 源 sampler | 一致 | 一致 | 一致 |

逐位一致指 loss、grad_norm、eval_loss、每个可训张量的和全部相同；比较的是每步的 (rank, task, 样本 id)、
log_history 与训练结束时的参数。对拍时固定 `PYTHONHASHSEED` 与 BLAS 单线程（多线程 BLAS 的归约顺序随进程变化，
会带来 fp32 舍入级差异，实测梯度中位相对差 1.7e-7）。

```bash
SRC_PY=<源环境 python> SRC_ROOT=<SimulS2ST-Omni-stage123> SW_PY=<本仓库环境 python> \
  FIXTURE=/tmp/simuls2st_smoke bash custom/simuls2st_omni/tests/parity/run_parity.sh all
```

### 7.2 其他

* **续训**：Stage 1 从 checkpoint-10、Stage 3（thinker + talker LoRA）从 checkpoint-3 恢复，后续每步与不中断训练逐位相同。
* **推理**：swift 训出的 Stage 1 / 2 / 3 产物被源推理代码直接加载，流式同传 agent 端到端产出 wav
  （tiny 模型 + 发布版 VoiceBox / Vocos / DualCodec / w2v-BERT，CPU）。
* **单测**：`python -m pytest custom/simuls2st_omni/tests -q`（46 项：源单测 32 + 接入层 14，含采样计划与源 sampler 在
  world_size 1/2/3 下逐 rank 一致）。
* **ktrain**：三份 yaml 通过 `ktrain job validate`；`ktrain_entry.sh` 在模拟 NVMe 上用生产版 Stage 1 配置跑通。

### 7.3 未验证

没有在 GPU、真实权重与真实语料上训练过；bf16 下的数值、显存与吞吐未测。Stage 2 所需的 manifest 与音频语料
在 S3 上的位置未知，三份 yaml 里的数据路径是占位符。

---

## 8. 常见报错

| 报错 | 原因 / 处理 |
|---|---|
| `需要 SIMULS2ST_CONFIG` | 直接调了 swift；用 `train_stage*.sh`，或导出 `SIMULS2ST_CONFIG` |
| `--seed ... 与 training.seed 不一致` | 源用同一个 seed 驱动 sampler 与 Trainer；不要单独改 `--seed` |
| `指针行的采样计划 ... 不一致` | 配置 / 数据 / world_size 变了却复用了旧的数据集缓存；保持 `--load_from_cache_file false` |
| `一行即一个完整 batch，需 --per_device_train_batch_size 1` | 覆盖了 batch size；batch 大小由 `sampler.token_budget` 控制 |
| `Expected to have finished reduction in the prior iteration` | 多卡 Stage 2/3 没开 `ddp_find_unused_parameters`（swift_args 已默认开启，检查是否被覆盖） |
| `Stage 2 的 init_talker_from 下没有 talker_components.pt` | Stage 1 未完成或交接目录不对；设 `SIMULS2ST_INIT_TALKER_FROM` |
| `merge source not found` | Stage 3 的 `merge_adapters_from` 不对；相对路径按 `SIMULS2ST_ROOT` 解析，建议写绝对路径 |
| `映射目标不存在`（ktrain） | `SIMULS2ST_MAP_*` 的同步任务未完成或路径写错 |
| `AutoVideoProcessor requires the Torchvision library` | 环境缺 torchvision（镜像已带 0.23）|
| `No module named 'simuleval.data'` | 用了源项目的 external/SimulEval（漏拷了该子包）；用 `vendor/SimulEval` |
