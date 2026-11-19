# streaming_si 训练管线

从 `streaming-si-0.6b` 迁移到 ms-swift 4.4.0（transformers 4.57.3 / peft 0.15.2）的流式同传训练管线。
模型结构：Voxtral 流式音频编码器（冻结）→ Thinker（文本 LLM）→ GatedBridge + Fusion → Talker（CosyVoice
FSQ speech token 自回归解码器）。源项目的全部训练阶段已迁移（本仓库编号 S1-S5，与源项目的对应见第 3 节）；离线整句推理与 S5 流式推理均已提供，目标波形由 Fun-CosyVoice3 flow/HiFT 生成。

四条评测链路的入口、调用关系、权重组成和运行方式见 [推理入口导航](../../docs/INFERENCE_ENTRYPOINTS.md)。

---

## 1. 模型变体

两种变体各对应一个 swift `model_type`。它们共用 template（`streaming_si`）、数据集、S5 tuner 和本目录全部训练脚本，只有结构不同（定义见 `custom/streaming_si/model/variants.py`）：

| model_type | Thinker | Talker | 参数量（thinker / talker） | S5 LoRA 目标数 |
|---|---|---|---|---|
| `streaming_si` | Qwen3-0.6B（28 层，1024d） | 0.1B：8 层 Qwen3 风格 decoder（1024d，16/4 头） | 0.60B / 0.14B | 252 |
| `streaming_si_4b` | Qwen3.5-4B（32 层，2560d；24 层 Gated DeltaNet + 8 层门控全注意力） | 0.6B：Qwen3-0.6B 结构（28 层，1024d，16/8 头） | 4.21B / 0.47B | 396 |

- 变体在导出时由 `--variant` 决定，写入 config 的 `thinker_arch`（`qwen3` / `qwen3_5`）和 `talker_preset`（`0.1b` / `0.6b`）。
- 训练脚本按 `MODEL_PATH/config.json` 的 `thinker_arch` 自动选 model_type，也可用 `SI_MODEL_TYPE` 手动指定；两者不一致时，加载阶段直接报错。
- 旧导出目录（config 里没有 `thinker_arch`）按 `streaming_si` 处理，行为不变。
- 两种变体都会额外保留一份冻结的 `talker_text_embed`：Thinker embedding 的副本，形状与 Thinker 词表相同。4B 变体的副本为 0.64B 参数，bf16 约 1.3GB。

---

## 2. 文件清单

### `custom/streaming_si/`

按职责分 6 个子目录。依赖方向：`model` ← `data` ← `integration` / `inference` / `scripts`（`model` 不依赖其他子目录，`data` 只依赖 `model`）。

| 文件 | 作用 |
|---|---|
| **`model/`** | 网络结构与 loss |
| `model/model_si.py` / `model/base_model.py` | 源项目模型本体与配置（SI、SITalker、GatedBridge、TalkerFusion、音频注入、tokenizer 构造） |
| `model/modeling_streaming_si.py` | HF 包装 `StreamingSIForConditionalGeneration`：forward 内部做 shift 并算 loss；按 `thinker_arch` 构建 Thinker；S5 目标分支 |
| `model/variants.py` | 变体与 Talker 规格预设 |
| `model/voxtral_compat.py` | Voxtral realtime 编码器、投影器与 mel 前端的 4.57 兼容实现 |
| `model/qwen3_5_compat.py` | Qwen3.5 文本塔的 4.57 兼容实现（复用 qwen3_next 组件）。训练前向 + 增量推理：`ThinkerCache` 存线性注意力的卷积/递归状态与注意力层 KV，`ThinkerRunner` 支持一次追加多个 token，CUDA 上可叠 CUDA Graph |
| `model/s4_objective.py` / `model/s4_losses.py` | S5 loss（译文 / 决策 / 步边界分项 CE + 样本加权 speech CE）与逐 token 指标、决策 macro-F1；`s4_losses` 从源项目原样迁入 |
| `model/speech_decoder.py` | 把 Fun-CosyVoice3 的 flow/HiFT 打包进模型目录 `speech_decoder/`，推理时定位 |
| **`data/`** | 样本编码与 S5 数据侧（不依赖 swift） |
| `data/si_encode.py` | S1-S4 样本编码（Emilia / P2）与 collate |
| `data/s4_streaming_dataset.py` / `data/s4_context_views.py` / `data/si_sampler.py` / `data/si_traj_ops.py` | S5 数据侧，**从源项目原样迁入**，只改写 import |
| `data/s4_pipeline.py` | S5 采样计划、指针行构建、按指针行编码 |
| **`integration/`** | swift 接入 |
| `integration/register_streaming_si.py` | swift 注册：两个 model_type、template、`streaming_si_s1/s2/s3` 数据集（Emilia / P2 S2TT / P2 S2ST）；`register_all()` 由 `custom/plugin.py` 调用 |
| `integration/register_streaming_si_s4.py` | S5 数据集 `streaming_si_s4` / `streaming_si_s4_val` 注册 |
| `integration/s4_peft.py` | S5 PEFT 边界 + swift 自定义 tuner `streaming_si_s4` |
| **`inference/`** | 推理 |
| `inference/infer_s2st.py` | 离线整句推理 / 评测：源语音 → 译文 + speech token → 24kHz 波形（S1-S4，见第 8 节） |
| `inference/infer_s5_streaming.py` | S5 流式同传：640ms 分块驱动 WAIT / WRITE，每段 WRITE 立即生成语音 |
| `inference/infer_tts.py` | 文本 → 语音（S1 / S3 checkpoint） |
| **`scripts/`** | 命令行工具 |
| `scripts/export_streaming_si.py` | 基座（Thinker + Voxtral + 可选源 `.pth`）→ swift 可加载的 HF 目录 |
| `scripts/merge_stage_checkpoints.py` | S3-w 起点合并：S2 的 thinker / audio_proj + S1 的 talker |
| `scripts/s3w_bridge_gate.py` / `scripts/s4w_bridge_gate.py` | 桥接预热的门 A（full vs text_only 的 speech CE） |
| `scripts/make_smoke_fixture.py` | CPU 冒烟夹具生成（tiny 随机模型 + 迷你 Emilia / P2 / 流式 records 数据） |
| **`eval/`** | 对比评测（RealSI / ACL 60/60，多系统多卡并行，见第 10 节） |
| `eval/prep_eval_sets.py` / `eval/make_shards.py` | 切句级测试集；按时长均衡装箱切 N 片 |
| `eval/run_worker.sh` / `eval/stage_to_fsx.sh` | 单卡跑一片（各系统串行）；模型与数据搬到共享 FSX |
| `eval/run_asr.py` / `eval/score_eval.py` | Whisper-large-v3 转写；BLEU / chrF / COMET 打分 |
| `eval/run_utmos.py` / `eval/run_para_metrics.py` | UTMOS；SIM-o / ESim / AutoPCP（含随机配对基线） |
| `eval/latency_eval.py` / `eval/stream_segments.py` | LAAL / LAAL-EOW / StartOffset / EndOffset（CA 与非 CA） |
| `eval/run_omni_offline.py` / `eval/omni_simuleval_wrap.py` | SimulS2ST-Omni 的离线与流式（L1～L4）入口包装 |
| `eval/omni_shared_run.py` / `eval/omni_graph_probe.py` | Omni 双方向共享模型入口与 static/graph 对照探针 |
| `eval/run_rtf.sh` / `eval/profile_s4.py` | 单卡独占 RTF 复测；S4 推理耗时分解 |
| `eval/make_report.py` | 汇总成 Markdown 报告 |
| **`tests/`** | 单测与跨版本对拍（`python -m pytest custom/streaming_si/tests -q`，见第 9 节） |

**旧模块名**：2026-10 之前这些模块平铺在 `custom/streaming_si/` 下。`streaming_si/__init__.py` 把旧名（如
`streaming_si.modeling_streaming_si`）映射到新位置的同一个模块对象，已有 adapter 的 `auto_mapping.parent_library`
与仓库外脚本不用改；新代码请用新路径。

### `dataset_maker/streaming_si/`

| 文件 | 作用 |
|---|---|
| `build_s4_val_split.py` | 按**源文本指纹分组**切 S5 held-out 验证集（同一句在库里重复出现，按 id 切会让验证集泄漏到训练集） |
| `build_record_index.py` | 构建 S5 分层采样索引（NIR × 句长；prefix-drop 资格判定也用它） |

### `pipelines/streaming_si/`

`common_args.sh`（SCENE 预设、阶段冻结映射、学习率调度、swift 启动封装），以及按任务划分的训练脚本：`train_tts.sh`（S1 / S3-w / S3）、`train_st.sh`（S2）、`train_s2st.sh`（S4-w / S4）、`train_simul.sh`（S5）。按仓库规则，影响训练行为的参数只写在这些脚本里，不写进任务 yaml。

`ktrain_entry.sh` 是 ktrain 集群上的入口（见 3.4 节）。

---

## 3. 训练流程

```
导出基座 ──► S1（talker, text_only）──┐
        └─► S2（audio_proj + thinker）─┴─► 合并 ──► S3-w（桥接预热）──► S3（talker, full）
            ──► S4-w（跨池桥接预热）──► S4（joint）──► S5（LoRA，流式）
```

编号按执行顺序；每组「-w / 正式」都是先只训桥接（`thinker_proj` + `fusion`），再全部放开。

| 阶段 | 任务 | 入口（`SI_STAGE`） | 可训参数 | MODEL_PATH | 数据 |
|---|---|---|---|---|---|
| S1 | TTS | `train_tts.sh s1` | 整个 talker（`cond_mode=text_only`，thinker 路恒为 0） | 导出的基座目录 | `SI_S1_DATA` |
| S2 | 语音→文本翻译 | `train_st.sh`（`s2`） | audio_proj + thinker | 导出的基座目录（可与 S1 并行） | `SI_P2_DATA` |
| S3-w | TTS | `train_tts.sh s3-w` | 只训 `talker.thinker_proj` + `talker.fusion`（桥接预热，`cond_mode=full`） | `merge_stage_checkpoints.py` 的产出 | `SI_S1_DATA` |
| S3 | TTS | `train_tts.sh s3` | 整个 talker（`cond_mode=full`，去掉 prompt 中重复的原文） | S3-w 的 checkpoint（不预热时为合并产出） | `SI_S1_DATA` |
| S4-w | 语音→语音翻译 | `train_s2st.sh s4-w` | 只训 `talker.thinker_proj` + `talker.fusion`（跨池桥接预热，只算语音 loss，默认 2500 步） | S3 的 checkpoint | `SI_P2_DATA` |
| S4 | 语音→语音翻译 | `train_s2st.sh s4` | audio_proj + thinker + talker | S4-w 的 checkpoint（不预热时为 S3 的） | `SI_P2_DATA` |
| S5 | 流式读写时机 | `train_simul.sh`（`s5`） | Thinker/Talker LoRA + 4 行流式 token + thinker_proj/fusion 全量（audio_proj、text_proj 冻结） | S4 的 checkpoint | `SI_S4_*` |

所有阶段中 `audio_encoder`（Voxtral）和 `talker_text_embed` 始终冻结。如果冻结前缀写错把它们打开，训练第一步会直接报错。

**与源项目 / 改名前编号的对应**（2026-10 改名）：

| 本仓库 | 源项目 / 改名前 | 输出子目录（改名前） |
|---|---|---|
| S1 | S1-0 | `s1_tts/`（`s1-0/`） |
| S2 | S2 | `s2_st/`（不变） |
| S3-w | S1-1w | `s3w_tts/`（`s1-1w/`） |
| S3 | S1-1a（新版源项目合并为单段 S1-1） | `s3_tts/`（`s1-1a/`） |
| S4-w | S3-w | `s4w_s2st/`（`s3_w/`） |
| S4 | S3 | `s4_s2st/`（`s3_s2st/`） |
| S5 | S4 | `s5_simul_<arm>_<chunk>/`（`s4_<arm>_<chunk>/`） |

- **旧 `SI_STAGE` 值**：`s1-0` / `s1-1w` / `s1-1a` / `s1-1` 与新名不冲突，入口自动映射并打印警告；旧 `s3-w` / `s3` / `s4` 与新名冲突，无法自动识别。旧 yaml 会因数据变量对不上（如旧 `s3` 没有 `SI_S1_DATA`）在入口直接报错，须手动改成 `s4-w` / `s4` / `s5`。
- **沿用源项目编号、未改名的**：数据集名与行内 `si_stage`（`streaming_si_s1/s2/s3/s4`，表示样本格式）、`SI_S4_*` / `S4_ARM` / `S4_P_DROP` 变量、tuner `streaming_si_s4`、`s4_*.py` 模块、`si_s4` 数据目录，以及 `custom/streaming_si/` 模型 / 数据代码注释里的阶段名。
- **S3 桶产物目录已改成新编号**（`trained_streaming_si_4b/` 下，2026-10-12）：`S1-1_init/` → `S3-w_init/`，`S1-1w/` → `S3-w/`，`S1-1a/` → `S3/`，原 `S3-w/`（S4-w 产物）→ `S4-w/`；`S1/`、`S2/`、`S4/` 本来就是新编号。注意新 `S3-w/` 与改名前的 `S3-w/` 内容不同，旧记录里的 `S3-w/` 路径应对应现在的 `S4-w/`。

### 3.1 端到端命令

两个基座可以用 `pipelines/streaming_si/export_base_models.sh` 一键导出并上传。脚本从 HuggingFace 下载 Qwen3-0.6B、Qwen3.5-4B、Voxtral（只取 HF 格式的 `model.safetensors`）和 Fun-CosyVoice3（只取 flow/hift/campplus/yaml），下载量约 20GB，全程约需 40GB 磁盘。已下载、已导出的会跳过：

```bash
S3_DEST=s3://<bucket>/<prefix>/streaming_si_swift bash pipelines/streaming_si/export_base_models.sh
# -> ${S3_DEST}/base_streaming_si/、${S3_DEST}/base_streaming_si_4b/
```

单个变体的手动导出如下：

```bash
# 0) 导出基座（二选一）
python custom/streaming_si/scripts/export_streaming_si.py \
    --thinker_dir <Qwen3-0.6B> --voxtral_dir <Voxtral-Mini-4B-Realtime-2602> \
    --out_dir models/streaming_si_base
python custom/streaming_si/scripts/export_streaming_si.py --variant streaming_si_4b \
    --thinker_dir <Qwen3.5-4B> --voxtral_dir <Voxtral-Mini-4B-Realtime-2602> \
    [--talker_init_dir <Qwen3-0.6B>] --out_dir models/streaming_si_4b_base
# 可选：--source_ckpt <源项目 .pth>（可重复，后者覆盖前者）、--add_stream_tokens、--dtype、
#       --cosyvoice_dir <Fun-CosyVoice3-0.5B>（把声码器打包进模型目录，之后每个 S1-S4 checkpoint 都自带）

BASE=models/streaming_si_base
OUT=output/streaming_si_output          # 默认 OUTPUT_ROOT

# 1) S1 与 S2（可并行）
MODEL_PATH=$BASE SI_S1_DATA='/data/s1_emilia_fsq_v3/*.parquet' bash pipelines/streaming_si/train_tts.sh s1
MODEL_PATH=$BASE SI_P2_DATA='/data/unist/machine_1/p2_fsq_v3/*.parquet' bash pipelines/streaming_si/train_st.sh

# 2) 合并：S2 的 thinker + audio_proj，S1 的 talker
python custom/streaming_si/scripts/merge_stage_checkpoints.py \
    --thinker_from $OUT/s2_st/<run>/checkpoint-XXX --talker_from $OUT/s1_tts/<run>/checkpoint-XXX \
    --out_dir models/streaming_si_s3w_init

# 3) S3-w → S3（Emilia TTS 上接入 thinker 语义）
MODEL_PATH=models/streaming_si_s3w_init SI_S1_DATA=... bash pipelines/streaming_si/train_tts.sh s3-w
python custom/streaming_si/scripts/s3w_bridge_gate.py --model $OUT/s3w_tts/<run>/checkpoint-XXX \
    --data /data/s1_emilia_fsq_v3 --baseline models/streaming_si_s3w_init
MODEL_PATH=$OUT/s3w_tts/<run>/checkpoint-XXX SI_S1_DATA=... bash pipelines/streaming_si/train_tts.sh s3

# 4) S4-w → S4（离线 S2ST）
MODEL_PATH=$OUT/s3_tts/<run>/checkpoint-XXX SI_P2_DATA=... bash pipelines/streaming_si/train_s2st.sh s4-w
python custom/streaming_si/scripts/s4w_bridge_gate.py --model $OUT/s4w_s2st/<run>/checkpoint-XXX \
    --data '/data/unist/machine_1/p2_fsq_v3/*.parquet' --wav_dir /data/unist/machine_1/p2_source_wav   # 门 A
MODEL_PATH=$OUT/s4w_s2st/<run>/checkpoint-XXX SI_P2_DATA=... bash pipelines/streaming_si/train_s2st.sh s4

# 5) S5（数据准备见第 4.3 节）
MODEL_PATH=$OUT/s4_s2st/<run>/checkpoint-XXX SI_S4=/data/si_s4 \
    SI_S4_VAL_DIR=/data/s4_val SI_S4_INDEX_DIR=/data/s4_index bash pipelines/streaming_si/train_simul.sh
```

- 所有脚本都可以在末尾追加 swift 参数，例如 `--resume_from_checkpoint <ckpt>`、`--max_steps N`。
- swift 保存的 `checkpoint-*` 目录可以直接作为下一阶段的 `MODEL_PATH`。保存时会复制 `voxtral_processor_config.json`，保证 mel 前端配置随 checkpoint 走。

#### S5 首训结果（default 臂 / 640ms，2026-10-13）

3 节点 24 卡，`ktrainjob_streaming_si_s5.yaml` 默认配置，1 epoch = 17309 步，约 6.4 小时，Succeeded。
验证集是 `build_s4_val_split.py` 按源文本指纹分组切出的 held-out（见第 4.3 节），每条只出现在一个集合里。

| 指标（末段，验证集） | 数值 | 第 2000 步 |
|---|---|---|
| `dec_macro_f1`（选 best 的指标） | 0.786 | 0.775 |
| `wait_f1` / `write_f1` | 0.812 / 0.760 | 0.800 / 0.714 |
| `bnd_acc`（步边界） | 0.943 | 0.933 |
| `dec_acc`（等 / 说） | 0.789 | 0.765 |
| `text_loss` / `text_acc` | 0.785 / 0.792 | 0.886 / 0.772 |
| `speech_loss` / `speech_acc10` | 2.918 / 0.721 | 2.957 / – |

- **读写时机在第 2000 步就基本学完**（占全程 12%），之后 15000 步只有约 0.01 的提升：决策与步边界的标签由采样计划给出，
  规律较确定。后半程的变化主要在译文与语音（`text_loss` 0.886 → 0.785）。
- **adapter 已上传 S3**：`.../trained_streaming_si_4b/S5/s5_simul_default_640/checkpoint-{16000,17309}/`
  （`checkpoint-16000` 是 `dec_macro_f1` 最高的 best；只含 adapter 权重 + `s4_metadata.json` + `args.json`，各约 62 MB）。
  加载方式：`PeftModel.from_pretrained(StreamingSIForConditionalGeneration.from_pretrained(S4 底座), <adapter 目录>)`。
- 只跑了 default 臂（640ms）。`prefix_drop` / `prefix_drop_control` 两臂尚未训练。
- **流式推理**：`custom/streaming_si/inference/infer_s5_streaming.py`（S4 底座 + S5 adapter，640ms 分块驱动 WAIT / WRITE，
  每段 WRITE 立即生成语音；带塌缩检测与换 seed 重采）。thinker 走增量缓存 + CUDA Graph，talker 走 StaticCache + CUDA Graph，
  单卡 H200 上 RTF 约 0.45（含声码器）；源音频上限 30s（训练 `SI_MAX_AUDIO_SEC`）。每步记 `elapsed_sec`，可算计算感知延迟。
  RealSI / ACL 60/60 上的对比评测（含 LAAL、UTMOS）见 `inference_outputs/eval_realsi_acl6060_20261016/`。

```bash
PYTHONPATH=custom python -m streaming_si.inference.infer_s5_streaming --model_dir <S4 checkpoint-24980> \
    --adapter <S5 checkpoint-17309> --input_jsonl eval.jsonl --out_dir out --cosyvoice_repo <CosyVoice>
```

### 3.2 运行场景（`SCENE`）

| SCENE | 行为 |
|---|---|
| `prod`（默认） | bf16 + DeepSpeed ZeRO-2；`NPROC_PER_NODE` 缺省按 `nvidia-smi` 自动检测；导出 `NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT` 给 swift 拉起 torchrun |
| `smoke` | 单进程 CPU、float32、3 步即停、`SAVE_ONLY_MODEL=true`；不导出分布式变量（macOS 的 MPS 不支持 c10d） |

公共默认值：`WEIGHT_DECAY=0.01`、`MAX_GRAD_NORM=1.0`、`SAVE_STEPS=2000`（S5 为 500）、`SAVE_TOTAL_LIMIT=5`、`LOGGING_STEPS=50`、`DATALOADER_NUM_WORKERS=4`、`REPORT_TO=none`。

### 3.3 学习率

- 源项目各阶段都是余弦衰减、无 warmup，对应 transformers 的 `cosine_with_min_lr`：
  - S1-S4：下限为峰值的 0.1 倍（`MIN_LR_RATE=0.1`）。
  - S5：绝对下限 2e-6（`MIN_LR=2e-6`，会覆盖 `MIN_LR_RATE`）。
- 与源项目差一个步位：源项目第 k 次更新取 t=k（从 1 开始），HF 取 t=k-1（从 0 开始）。因此 HF 首步是峰值 lr，末步略高于下限；曲线形状相同。

### 3.4 在 ktrain 集群上多机训练

提交命令、参数含义、各阶段日志与常见报错见 [`ktrain/SUBMIT.md`](../../ktrain/SUBMIT.md)，本节只讲 streaming_si 特有的部分。

每个阶段一份 yaml，文件名只含阶段号（`ktrainjob_streaming_si_s<N>.yaml`），使用 `framework: pytorch`。streaming_si 还没有适配 Ray，不能用 `framework: ray`。

**节点数只是配置**：由 yaml 里的节点白名单（`render-extra` 的 `affinity`）和 `numNodes` 决定，两者必须一致；文件名不区分节点数。全局 batch = 每卡 batch × 8 × `numNodes`，改节点数时步数、warmup 步数随之变化，学习率是否跟着调见各 yaml 的训练参数注释。提交时 `--job-name-prefix` 也只写阶段（如 `si-s4`），yaml 的 `metadata.name` 统一为 `__JOB_NAME__`。

| yaml | 阶段（`SI_STAGE`） | 当前节点数 | 数据（S3） | 每节点需同步 |
|---|---|---|---|---|
| `ktrain/ktrainjob_streaming_si_s1.yaml` | S1 | 2 | `open_data_0803_fortrain/TTS/emilia_fsq_v3/s1_emilia_fsq_v3/` | 20.5 GiB |
| `ktrain/ktrainjob_streaming_si_s2.yaml` | S2 | 2 | `open_data_0803_fortrain/machine_{0,1,2}_shard/` 的 `p2_fsq_v3/` 与 `p2_source_wav/` | 约 2.1 TiB |
| `ktrain/ktrainjob_streaming_si_s3.yaml` | S3-w / S3（改 `SI_STAGE`） | 3 | 同 S1 | 同 S1 |
| `ktrain/ktrainjob_streaming_si_s4.yaml` | S4-w / S4（改 `SI_STAGE`） | 3 | 同 S2 | 同 S2（跑过 S2 的节点不再下载） |
| `ktrain/ktrainjob_streaming_si_s5.yaml` | S5 | 2 | `si_s4_rechunk_D/records/`（640ms）+ `si_s4/audio/_tars/` | records 48 GiB + 音频 tar 5.4 TiB（解压后再占约 5.4 TiB） |

数据相关的环境变量：

- **S1 / S3-w / S3**：`SI_S1_DATA` 写 parquet 目录。
- **S2 / S4-w / S4**：`SI_P2_DATA_DIRS` 与 `SI_P2_WAV_DIRS` 各写多台机器的目录，按顺序一一对应。入口会在节点上用软链把它们合并成一个 parquet 目录和一个 tar 目录；tar 或 parquet 跨机器重名时直接报错。
- **S5**：`SI_S4_RECORDS` 写各方向的 records 目录，`SI_S4_AUDIO_TARS` 写音频 tar 目录。records 和音频可以在不同位置（rechunk_D 本身不含音频）。节点上已有解压好的音频时，改用 `SI_S4_AUDIO_ROOT`。

yaml 里的 `TRAIN_SCRIPT` 指向 `pipelines/streaming_si/ktrain_entry.sh`，它负责三件事：

1. 把 `SI_*` 变量里的 `s3://` 地址转换成节点上的 `/nvme/<key>`。`launch_train.sh` 只转换固定几个变量，不处理 `SI_*`。
2. S5 数据在节点上的准备：解压音频 tar、切验证集、建分层索引。这些步骤可以重复执行，结果保存在节点 NVMe 上，后续任务直接复用。
3. 按 `SI_STAGE` 调用对应阶段的训练脚本。`launch_train.sh` 调用入口时不传参数，所以阶段只能通过环境变量指定。

**第一次在一组新节点上训练 S5 时，要分两次提交。** 先设 `SI_PREP_ONLY: "1"` 提交一次，各节点并行准备数据后退出；再改回 `"0"` 提交正式训练。原因是：如果把数据准备和训练放在同一个任务里，各节点的准备耗时可能相差几小时，先准备完的节点会在 torchrun 等待其他节点时超时。所以多节点训练时入口只检查数据，不做准备；缺数据会直接报错退出，不会白占 GPU。单节点训练时不受此限制，入口会先准备数据再开始训练。

其他说明：

- **多节点保存 checkpoint**：多节点时入口会自动加上 `--save_on_each_node true`，每个节点都保留一份完整 checkpoint。如果 `OUTPUT_ROOT` 在多节点共享的存储上，设置 `SI_SHARED_OUTPUT=1` 关掉这个行为。
- **续训**：在 yaml 里设 `SI_EXTRA_ARGS: "--resume_from_checkpoint <路径>"`。
- **各阶段的 `MODEL_PATH`**：必须是 HF 格式目录（含 `config.json`）。源项目的 `.pth` 要先用 `export_streaming_si.py --source_ckpt` 转换再上传。
- **镜像**：各份 yaml 统一使用 `p5-cuda12.8-torch2.8-te-v9-streaming-si-v1`，0.6B 与 4B 两个变体通用。它由 `ktrain/image/Dockerfile.streaming-si` 在 v9 上叠一层构建，额外带有：
  - fla / causal-conv1d 加速核：`streaming_si_4b` 的线性注意力靠它们提速，没有时走纯 torch，很慢；
  - CosyVoice 仓库，并配好 `COSYVOICE_REPO`：在 Pod 里用 `infer_s2st.py` 合成音频时需要。

  torch / TE / EFA 与 v9 相同，训练环境不变。依赖清单见 `ktrain/image/requirements-streaming-si.txt`；要出新版本，用 `bash ktrain/image/build_image_streaming_si.sh <新 tag>` 构建（ECR 的 tag 不可覆盖）。
- **S3-w**：训练起点需要先用 `merge_stage_checkpoints.py` 合并出来，这一步不在入口里。

---

## 4. 数据

### 4.1 Emilia（`streaming_si_s1`，S1 / S3-w / S3）

- **格式**：Emilia-FSQ parquet，字段 `conversations`（JSON 字符串）、`answer_speech_tokens`、`ref_speech_tokens`、`duration`。
- **必须是 v3 代 FSQ**（CosyVoice3 tokenizer + 正确的 mel 前端）。旧代码本的数据从内容上无法区分，训错也不会报错；生成后先用源项目 `data_prep/verify_fsq_data.py` 验收。
- **相关变量**：`SI_S1_DATA`（文件或 glob）。

### 4.2 P2（`streaming_si_s2` / `streaming_si_s3`，S2 / S4-w / S4）

- **格式**：UniST P2 parquet，字段 `source_transcript`、`translation`、`src_lang`、`tgt_lang`、`source_wav`（`tar名::偏移:长度`）、`target_fsq` 等。
- **源音频 tar 目录**：默认取 `<parquet 目录>/../p2_source_wav`。
- **一个数据集只能对应一个 tar 目录**：datasets 读多个 parquet 的顺序不可靠，行和 tar 的对应可能错位而不报错，所以跨目录时直接报错。machine_1 / machine_2 的 tar 同名但内容不同，需要分别训练，或把 tar 合并到一个目录后用 `SI_P2_WAV_DIR` 指定。
- **相关变量**：`SI_P2_DATA`、`SI_P2_WAV_DIR`。

### 4.3 流式 records（S5，`streaming_si_s4` / `streaming_si_s4_val`）

记录格式为 `si-rw-m1-v0`（640 ms 基础块轨迹 + `target_codec_codes`），布局为 `si_s4/records/<dir>/chunk_640ms/{train,val}/records/part-*.jsonl`，音频根为 `si_s4/audio`。

一次性数据准备（每个方向各跑一次，`en2zh` / `zh2en`）：

```bash
python dataset_maker/streaming_si/build_s4_val_split.py \
    --records $SI_S4/records/en2zh/chunk_640ms/val/records \
    --direction en2zh --n_val 1500 --out $VAL_DIR --data_root $SI_S4/audio
python dataset_maker/streaming_si/build_record_index.py \
    --records $SI_S4/records/en2zh/chunk_640ms/train/records \
    --direction en2zh --data_root $SI_S4/audio --out $INDEX_DIR --workers 16
```

两个脚本的产物：

- **验证集划分**（`build_s4_val_split.py`）：`<dir>_val.jsonl` 加 `<dir>_val_keys.json`。训练集按这里的源文本指纹剔除整组，避免同一句同时出现在训练集和验证集。
- **分层索引**（`build_record_index.py`）：`<dir>_index.npz` 加 `<dir>_index_meta.json`，schema 2。records 目录需带 `_manifest.json`，脚本会校验 profile、行数和 part 顺序。

**swift 侧的设计：指针行。** 一条 S5 记录约 4.5KB，两方向共约 5.8GB，转成 arrow 缓存会再占一份磁盘，所以 swift 数据集里每行只存一个指针：`(split, 记录下标, 视图类型, draw_seed, 采样计划 digest)`。template 在 `_encode` 里用指针调用源项目的 `S4StreamingDataset.__getitem__`，得到与源项目逐 token 相同的样本。

采样在加载时一次性定好，训练中不再有 sampler 状态，`--resume_from_checkpoint` 可以精确续训：

| 源项目行为 | 本实现 |
|---|---|
| 均匀 shuffle（`SI_S4_STRATIFIED=0`） | 每条记录一行，由 swift 打乱 |
| `WeightedRandomSampler(w, len, replacement=True)` | 用同一权重（`si_sampler.weights_for_dataset`）有放回地抽 len 行 |
| prefix-drop 调度 `MixedContextViewSampler` | 直接调用源类（world_size=1，全局一份精确配额）；调度长度取整到全局 batch，并受 `max_steps` 截断 |
| 延迟倍数 m 用每个 dataset 实例一条顺序 RNG 流 | 每行一个独立 `draw_seed`：分布相同，且与 worker 数、续训无关 |
| 训练集剔除 val 源文本组 | 相同（`exclude_keys`） |

loader 进程和 DataLoader worker 各自重建采样计划，用 digest 核对一致；环境变量在两者之间变了会直接报错。

**tokenizer**：S1-S4 的导出目录只注册了 `<|audio_pad|>`。S5 加载时会按源项目 `build_qwen_tokenizer(add_stream_tokens=True)` 的顺序追加 38 个流式 token 并逐个校验 id：

| 变体 | `[wait, W, chunk, src_end]` |
|---|---|
| `streaming_si` | `[151670, 151671, 151672, 151707]` |
| `streaming_si_4b` | `[248077, 248078, 248079, 248114]`（`<|audio_pad|>` 直接用词表里已有的 248076） |

---

## 5. 各阶段关键参数（默认值）

| 参数 | S1 | S2 | S3-w | S3 | S4-w | S4 | S5 |
|---|---|---|---|---|---|---|---|
| `SI_TEXT_LOSS_WEIGHT` / `SI_SPEECH_LOSS_WEIGHT` | 0 / 1 | 1 / 0 | 0 / 1 | 0 / 1 | 0 / 1 | 1 / 1 | 1 / 1 |
| `SI_COND_MODE` | text_only | full | full | full | full | full | full |
| `SI_STRIP_PROMPT_TEXT` | false | – | true | true | – | – | – |
| 每卡 batch × 时长 | 32 × 2 epoch | 8 × 1 epoch | 32 × `MAX_STEPS`（yaml 给出） | 32 × 2 epoch | 16 × 2500 步 | 16 × 1 epoch | 48 × 1 epoch |
| 学习率 | 1e-4 | 1e-4 | 3e-4 | 1e-4 | 3e-4 | 5e-5 | 3e-5 → 2e-6 |
| `MAX_LENGTH` | 512 | 768 | 512 | 512 | 768 | 768 | 1024 |

其他样本口径：

- **S1**：`SI_MAX_SPEECH_LEN=750`（约 30s）、`SI_MAX_PROMPT_SPEECH_LEN=125`、`SI_REF_DROP_RATIO=0.5`。
- **S2 / S4-w / S4**：`SI_MAX_AUDIO_SEC=30`、`SI_MIN_AUDIO_SEC=0.5`、`SI_MAX_TARGET_LEN=256`、`SI_ST_TASK=translate`、`SI_DROP_OVERLONG=1`；S4-w / S4 另有 `SI_MAX_SPEECH_LEN=750`。
- **S5**：`SI_MAX_AUDIO_SEC=30`、`SI_MAX_SPEECH_LEN=750`。

与源项目的差异：S2/S4/S5 的源音频上限从 20s 放宽到 30s（Voxtral 编码器 `max_position_embeddings=1500` 帧 @50Hz，正好 30s），`MAX_LENGTH` 相应调大；超长样本**丢弃**而不截断。源项目 S2/S3 是截断：音频超过 20s 截掉尾部但保留完整译文，译文超过 256 token 截掉，导致音频与译文对不上。S5 源项目本来就是跳过。

- S2/S4 丢弃条件：音频 > `SI_MAX_AUDIO_SEC`、译文 > `SI_MAX_TARGET_LEN`、总长 > `MAX_LENGTH`、S4-w / S4 的 `target_fsq` > `SI_MAX_SPEECH_LEN`（`si_encode.st_overlong_reason`）。
- 经 ktrain 提交时，`ktrain_entry.sh` 先用 `p2_parquet_prep.py` 给 parquet 加元数据列（音频采样点数、token 数、`lengths`），超长样本在数据集预处理阶段就被过滤掉，日志里能看到过滤前后的行数；编码时再检查一次兜底。
- `SI_DROP_OVERLONG=0` 回到源口径（截断）；要完全复现源项目，同时设 `SI_MAX_AUDIO_SEC=20`、`MAX_LENGTH=512`。
- `GROUP_BY_LENGTH=true`：按 `lengths` 列把长度相近的样本组进同一个 batch（swift `--group_by_length`），P2 数据随机组 batch 时约 60% 的计算花在 padding 上。batch / 样本长度 / 推理窗口的调参依据见 `docs/tuning_batch_and_context.md`；不要用 `--packing`（见该文档第 2 节）。

S5 专属变量（`train_simul.sh` 中设置）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SI_S4` 或 `SI_S4_DATA` / `SI_S4_DATA_ROOT` | – | si_s4 根目录；或显式给出 records 目录（逗号分隔）和音频根 |
| `SI_S4_VAL_DIR` / `SI_S4_INDEX_DIR` | – | 验证集划分产物 / 分层索引产物；缺 val 时没有 held-out 指标，也不剔除 val 源文本组（会打印警告） |
| `SI_S4_DECISION_WEIGHT` | 1 | 决策与步边界 CE 的权重，调参区间 [0.5, 2.0]。**设置该变量即启用 S5 目标** |
| `SI_S4_STRATIFIED` / `SI_S4_MAX_REPEAT` / `SI_S4_MIN_SOURCE_MS` | 1 / 2.0 / 0 | 分层加权采样 |
| `SI_S4_LATENCY_MULTIPLIER` / `SI_S4_M_MAX` / `SI_S4_M_SCHEDULE` | 1 / 4 / uniform | 延迟倍数；`sample` 为随机抽样 |
| `SI_S4_USE_SRC_END` / `SI_S4_STRICT_TEXT` | 1 / 0 | 同源项目 |
| `SI_S4_TRAIN_SCOPE` | joint | `joint` \| `thinker` \| `talker` |
| `S4_ARM` / `S4_P_DROP` | default / – | 实验臂 `prefix_drop` / `prefix_drop_control`（见 6.3） |
| `SI_S4_PREFIX_CONTEXT_TURNS` | 16 | prefix-drop 保留的上下文轮数 |
| `SI_S4_SEED` | 1337 | 采样种子；同时作为 swift `--seed`（token 行重置种子） |
| `VAL_INTERVAL` / `VAL_BATCHES` | 2000 / 20 | 验证间隔；每个方向取前 `VAL_BATCHES × 每卡 batch × 卡数` 条 |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_DROPOUT` | 8 / 16 / 0 | 在脚本内固定 |

---

## 6. S5 细节

### 6.1 PEFT（`--tuner_type streaming_si_s4`）

- **LoRA 目标**：Thinker 与 Talker 的 transformer 层。
  - 全注意力层（7 个）：`q/k/v/o_proj` 与 `gate/up/down_proj`。
  - Qwen3.5 的线性注意力层（6 个）：`in_proj_qkv / in_proj_z / out_proj` 与 MLP 三个投影。`in_proj_b` / `in_proj_a` 每头只输出一个标量，不挂 LoRA。
- **token 行**：`trainable_token_indices` 放开 4 行流式 token，embedding 与 lm_head 共享同一组 token adapter。
  - peft 0.15.2 没有 `ensure_weight_tying`，这个绑定靠它的自动判定（`tie_word_embeddings=True` 且 `_tied_weights_keys` 为列表）。
  - 判定一旦失效，lm_head 会停留在旧行而 embedding 学新行，决策 logit 不随训练变化，而且不报错。所以 attach 和 load 之后都会硬校验，未绑定就报错。
- **全量训练**：`talker.thinker_proj` 与 `talker.fusion`（`modules_to_save`）。
- **训练前**：用固定种子把 4 个 token 行重置为可区分的随机方向，范数对齐到已用词表行（源 `reinit_stream_token_rows`）。
- **checkpoint**：只保存 adapter，格式与源项目一致（含 `s4_metadata.json`：format、stream_token_ids、train_scope、base_model_dir），源项目训出的 adapter 目录可以直接加载。
  - 续训：`--resume_from_checkpoint <ckpt>`。
  - 只加载 adapter 权重、从第 0 步开始：`--adapters <adapter 目录>`。加载时会核对 token id 是否与当前 tokenizer 一致。
- **scope**：
  - `thinker`：Thinker LoRA + token 行。
  - `talker`：Talker LoRA + thinker_proj + fusion。
  - `joint`：以上全部。
  - 实际可训集合会与期望逐项比对，多一个或少一个都报错。

### 6.2 Loss 与指标

```
loss = text_w · split_total + speech_w · weighted_speech_ce + aux
split_total = (Σ译文CE + w_dec·(Σ决策CE + Σ步边界CE)) / (n_txt + w_dec·(n_dec + n_bnd))
```

- `split_losses` 从源项目原样拷贝。样本加权的 speech CE 复用源 `s4_losses.weighted_speech_ce`（prefix-drop 的 crop 视图会按样本改变 speech 权重）。
- 不对整张 logits 做 `.float()` 拷贝：按默认 batch 48、序列 640、词表 15 万算，约省 19GB 显存。精度交给 autocast 下 `cross_entropy` 内部的 fp32 计算，与源项目一致。
- **日志指标**：
  - `text_loss/acc`（译文）、`dec_loss/acc`（等/说决策）、`bnd_loss/acc`（步边界）、`speech_loss/acc/acc10`。
  - `dec_macro_f1`、`wait_f1`、`write_f1`、`wait_recall`、`write_recall`、`dec_pred_other`。
- **跨卡聚合**：指标以逐 token 向量上报，swift 的 MeanMetric 按元素累加并跨卡 all_reduce，得到的就是源 `evaluate()` 的 token 加权均值。
- **macro-F1**：不能按 token 平均，因此上报 6 个混淆矩阵单元的逐决策 token 指示向量，由 `install_s4_metric_patch` 在汇总后还原成 F1（与源 `macro_f1_from_cm` 同式）。
- **选 best**：`--metric_for_best_model dec_macro_f1`。决策两类严重不平衡，按 acc 会把「恒说」选成 best。

### 6.3 实验臂（prefix-drop，对应源 `--p_drop` / `--paired_crop_control`）

```bash
S4_ARM=prefix_drop         S4_P_DROP=0.25 ... bash pipelines/streaming_si/train_simul.sh   # H1 实验臂：真实 crop
S4_ARM=prefix_drop_control S4_P_DROP=0.25 ... bash pipelines/streaming_si/train_simul.sh   # 配对对照：同调度、不 crop
```

- **约束**（与源项目相同，违反即报错）：`SI_S4_TRAIN_SCOPE=thinker`、`SI_SPEECH_LOSS_WEIGHT=0`、`SI_S4_LATENCY_MULTIPLIER=1`，并且需要 `SI_S4_INDEX_DIR`。
- **`--strict true`**：`train_simul.sh` 固定传入，样本编码失败即中止，与源项目一致。swift 默认会吞掉异常并随机换一行，这会静默破坏 crop 配额。

---

## 7. 兼容层

| 模块 | 解决的问题 | 对拍（tiny 随机权重） |
|---|---|---|
| `voxtral_compat.py` | transformers 4.57 没有 Voxtral realtime 编码器 | 对 transformers 5.8 官方实现：整段、分块流式、投影、mel 四项 max\|Δ\| < 1e-6 |
| `qwen3_5_compat.py` | transformers 4.57 没有 Qwen3.5 | 对 transformers 5.17 官方 `Qwen3_5ForCausalLM`：logits 与全部层 hidden max\|Δ\| < 1e-6（含跨 64 chunk 边界、右 padding、eager/sdpa、tied lm_head） |

Qwen3.5 兼容层要点：

- **复用**：4.57 的 qwen3_next 组件（门控全注意力、`(1+w)` RMSNorm、门控 RMSNorm、MLP、torch 版 gated delta rule）。
- **线性注意力输入投影**：改为 Qwen3.5 的分开形式（`in_proj_qkv / in_proj_z / in_proj_b / in_proj_a`）。
- **RoPE**：官方是 interleaved mRoPE，纯文本输入时退化为 1D 部分旋转 RoPE（partial_rotary_factor=0.25），按 1D 实现。
- **权重导出**：官方发布为多模态 `Qwen3_5ForConditionalGeneration`，导出只取 `model.language_model.*`，丢弃 `model.visual.*` 与 `mtp.*` 并打印计数。
- **加速核**：GPU 上检测到 `flash-linear-attention` / `causal-conv1d` 时走加速核，否则回退纯 torch 实现（正确但很慢）。
- **增量推理**：`ThinkerCache` 存两类状态——线性注意力层（24/32 层）的卷积状态与递归状态、全注意力层（8/32 层）的 KV。
  `ThinkerRunner` 一次可追加多个 token（卷积把上一次末尾 `kernel-1` 帧前置，delta rule 以上一次递归状态为初始态），
  这是流式按块推进所必需的——4.57 的 `qwen3_next` 只在追加 1 个 token 时才走增量路径，多 token 会把状态清零重算。
  CUDA 上对定长单 token 步叠 CUDA Graph。`tests/test_qwen3_5_compat.py` 对拍增量与整段重算的逐步 logits。
  bf16 下两者存在数值差，约 8% 的句子译文会有个别 token 不同。

Mel 前端的 `global_log_mel_max` 取 1.5：这是官方 Voxtral 配置在源训练中实际生效的值。源代码里默认的 2.0 被官方配置覆盖了，并不生效。

---

## 8. 推理（离线整句 S2ST）

`custom/streaming_si/inference/infer_s2st.py` 把一句源语音翻成译文和 24kHz 目标语音，用于 S4（离线 S2ST joint）及之前阶段的完整模型目录：导出目录、merge 产物、swift 的 S1-S3 `checkpoint-*`。S5 adapter 会被拒绝，S5 用 `inference/infer_s5_streaming.py`。

### 8.1 声码器（flow / HiFT）为什么不在训练模型里

- 训练目标是 speech token，数据里的目标 token 已经离线算好，训练前向不经过 flow/HiFT，它们的权重也从不更新。
- 源项目同样不把它们存进 checkpoint：`attach_speech_decoder` 用 `object.__setattr__` 挂载，不进 `state_dict`；迁移后保持一致，加载时忽略 `speech_decoder.*` 键。
- 放进训练模型的代价：CosyVoice 不是 pip 包（要 `git clone --recursive` 拉 Matcha-TTS，再装 hyperpyyaml / conformer / diffusers），训练镜像里没有；DeepSpeed 会给这部分分配显存。

为了一个目录就能推理，导出时可以把**声码器文件**原样打包进模型目录（仍然不进权重）：

```bash
python custom/streaming_si/scripts/export_streaming_si.py ... --cosyvoice_dir <Fun-CosyVoice3-0.5B>
# -> <out_dir>/speech_decoder/{flow.pt, hift.pt, cosyvoice3.yaml, campplus.onnx, manifest.json}
```

- 之后 swift 保存的 S1-S4 checkpoint 会整目录带上 `speech_decoder/`（模型注册的 `additional_saved_files`），`merge_stage_checkpoints.py` 也会复制它。代价是每个 checkpoint 多一份声码器文件。
- S5 只保存 adapter，不复制；推理时会按 `s4_metadata.json` 的 `base_model_dir` 回去找。
- `campplus.onnx`（CAM++ 说话人向量）可选，缺失时用零向量，音色是中性的。

### 8.2 用法

```bash
# 依赖：CosyVoice 官方仓库（--recursive）+ hyperpyyaml / conformer / diffusers / onnxruntime
export COSYVOICE_REPO=<CosyVoice 仓库>

# 单条
python custom/streaming_si/inference/infer_s2st.py --model_dir <S4 checkpoint> \
    --audio src.wav --tgt_lang zh --out_dir out/demo
# 批量评测：jsonl 每行 {"id", "audio", "tgt_lang", 可选 "reference"}
python custom/streaming_si/inference/infer_s2st.py --model_dir <S4 checkpoint> \
    --input_jsonl eval.jsonl --out_dir out/eval
```

- **输出**：`<id>.wav`（24kHz）、`results.jsonl`（译文、speech token、停止原因、耗时）、`summary.json`（成功/失败数、停止原因分布；带 reference 且装了 sacrebleu 时按目标语言报文本 BLEU，中文用 zh 分词）。ASR-BLEU 需要额外的 ASR 模型，不在脚本内。
- **声码器目录**：`--cosyvoice_dir` 优先，否则用 `<model_dir>/speech_decoder/`。
- **不装 CosyVoice 时**：`--no_vocoder` 只输出译文和 speech token；`--text_only` 只出译文（S2 checkpoint 用，它的 talker 没训练过）。
- **默认参数**：Thinker greedy，最多 256 个 token；Talker RAS 采样（top_k 25 / top_p 0.8），最多 750 个 token；flow ODE 5 步（源项目评测口径，官方默认 10）；源音频超过 30s 截断（训练上限，`--max_audio_sec`；源项目口径训的模型传 20）。
- **加速**（默认开启，与 S5 同一实现）：Thinker 增量缓存（Qwen3.5 走 `ThinkerRunner`：线性注意力状态 + KV，CUDA 上叠 CUDA Graph），Talker 单 token 步 StaticCache + CUDA Graph。单卡 H200 RTF 从约 1.05 降到约 0.3。`--no_thinker_cache` / `--no_thinker_graph` / `--no_talker_graph` 退回旧路径作对照；bf16 下增量与整段重算存在数值差，约 8% 的句子译文会有个别 token 不同。

### 8.3 与训练布局的对齐

| 步骤 | 推理 | 训练 |
|---|---|---|
| Thinker 输入 | `audio_pad×n + 指令`，自回归到 eos | `audio_pad×n + 指令 + 译文 + eos`（`encode_st`） |
| Talker 条件 | 对 `prompt + 译文`（不含 eos）整段重跑 thinker，取译文位置 | shift 后 `X = ids[:-1]`、`answer_mask[:-1]`，覆盖的正是译文、不含 eos |
| Talker 序列 | `[sos] + cond + [task]` 后自回归到 eos | `[sos] + cond + [task] + speech + eos`（`SITalker.build_inputs`） |

`test_infer_s2st.py` 对这三点逐位验证：推理算出的 Talker 条件与训练前向 `torch.equal`；Talker 的 KV cache 逐步生成与 teacher forcing 整段前向的 argmax 逐位一致；Thinker 的 KV cache 生成与每步整段重算一致（`thinker_cache=False` 为后者）。

---

## 9. 验证与测试

### 9.1 与源项目的一致性

| 项目 | 结论 |
|---|---|
| S1-S4 样本编码（源 S1/S2/S3 dataset） | input_ids / labels / answer_mask / speech token / mel 与源 dataset 逐 token 一致（真实 Qwen3 tokenizer） |
| S4 loss | 与源 `train_s3_s2st.py` 逐位一致 |
| 冻结范围 | 三种组合在 swift 实际日志中核实 |
| 梯度累积 | `accepts_loss_kwargs=False`，否则 HF Trainer 会跳过 GA 除法；实测 `bs=2/ga=1` 与 `bs=1/ga=2` 首步 loss 与 grad_norm 一致 |
| S5 数据侧 | 源项目 80 个相关单测对准迁移后的模块全部通过 |
| S5 对拍（`test_s4_pipeline.py`，14 项） | 4 种采样方式（分层 / 随机延迟倍数 / prefix-drop / 对照臂）的采样计划、逐条样本编码、验证集、collate + 模型内 shift、loss、macro-F1 均与源项目一致 |
| S5 流式 token id | 与源 `build_qwen_tokenizer(add_stream_tokens=True)` 一致（两种变体都已验证） |

### 9.2 运行测试

```bash
# 单测（需 peft 的用例在无 peft 环境自动跳过）
python -m pytest custom/streaming_si/ -q

# S5 与源项目对拍（需要源仓库与原始 Qwen3-0.6B tokenizer，以及含 s4/ 的冒烟夹具）
SI_SRC_ROOT=<streaming-si-0.6b> SI_QWEN3_TOKENIZER=<Qwen3-0.6B> SI_SMOKE_DIR=/tmp/si_smoke \
    python -m pytest custom/streaming_si/tests/test_s4_pipeline.py -q

# 跨版本对拍：先在 transformers>=5.2 的环境生成参考，再在本仓库环境（4.57.3）比较
python custom/streaming_si/tests/test_voxtral_compat.py reference /tmp/voxtral_ref.pt    # tf5 环境
python custom/streaming_si/tests/test_voxtral_compat.py compare   /tmp/voxtral_ref.pt    # 本仓库环境
python custom/streaming_si/tests/test_qwen3_5_compat.py reference /tmp/qwen3_5_ref.pt    # tf5 环境
python custom/streaming_si/tests/test_qwen3_5_compat.py compare   /tmp/qwen3_5_ref.pt    # 本仓库环境
```

### 9.3 CPU 冒烟

```bash
# 生成夹具：tiny streaming_si 模型 + Emilia / P2 / 流式 records 迷你数据（流式部分调用 dataset_maker 的两个脚本）
python custom/streaming_si/scripts/make_smoke_fixture.py --tokenizer_dir <Qwen3-0.6B> --out /tmp/si_smoke
# 追加 tiny streaming_si_4b 模型（数据共用）
python custom/streaming_si/scripts/make_smoke_fixture.py --out /tmp/si_smoke --only model_4b \
    --qwen35_tokenizer_dir <Qwen3.5-4B 的 tokenizer 目录>

# 用正式脚本跑，每阶段 3 步
export SCENE=smoke SI_S1_DATA=/tmp/si_smoke/s1/fsq-00000.parquet \
       SI_P2_DATA=/tmp/si_smoke/p2/p2_fsq_v3/fsq-00000.parquet
MODEL_PATH=/tmp/si_smoke/model_4b bash pipelines/streaming_si/train_tts.sh s1
# ... S2 → 合并 → S3-w → S3 → S4-w → S4 同第 3.1 节；S5 的 SI_S4_* 取值由 make_smoke_fixture.py 打印
```

已验证（改名前，按源项目编号）：两种变体的 S1-0 → S2 → 合并 → S1-1 → S3 → S4 全链路，S4 三个实验臂，S4 续训与 `--adapters` 加载。旧格式导出目录重跑五阶段，末步 loss 与改动前逐位相同。

冒烟 checkpoint 并不小：`streaming_si` 的 tiny 模型仍含 Qwen3 全词表 embedding，float32 单份约 0.6GB，保存 optimizer 状态时约为 3 倍。磁盘紧张时，跑完一个阶段就删掉上一阶段的产物。

---

## 10. 评测（RealSI / ACL 60/60）

脚本都在 `custom/streaming_si/eval/`。已完成的一次完整评测见 `inference_outputs/eval_realsi_acl6060_20261016/README.md`（1193 句、7 张 H200 并行，S4 / S5 / SimulS2ST-Omni 离线与流式 L1～L4 共 7 个系统）。

### 10.1 测试集

| 数据集 | 方向 | 句数 | 切分方式 |
|---|---|---|---|
| RealSI | en→zh / zh→en | 346 / 431 | 按官方 json 的 `segment` 边界切句 |
| ACL 60/60 eval | en→zh | 416 | 用官方 gold 切分（`*.gold.seg.*`） |

`prep_eval_sets.py` 负责切句并统一成 jsonl（`id` / `audio` / `src_lang` / `tgt_lang` / `src_text` / `reference` / `source_sec` / `ds`）；
`make_shards.py` 把三个数据集合并，按源音频时长做均衡装箱切 N 片，使每片的总时长接近（避免长句都挤在一张卡上）。

### 10.2 评测工具

| 环节 | 用的模型 / 工具 | 细节 |
|---|---|---|
| **ASR（算 ASR-BLEU 用）** | `openai/whisper-large-v3` | 中英都用同一个模型，`language` 按目标语**强制指定**（不让它自动判语种，否则中文输出偶尔被识别成日文）。chunk 30s / stride 5s，greedy（`num_beams=1`），`condition_on_prev_tokens=False` 关掉跨 chunk 条件以免幻觉传播。`eval/run_asr.py` |
| **BLEU / chrF** | `sacrebleu` 2.x | 中文 `tokenize='zh'`（按字），英文 `13a`（按词）。**所以中英两列的 chrF 不能横向比**。假设与参考做同样的轻归一化：去标点、英文小写、压空白。`eval/score_eval.py` |
| **COMET** | `Unbabel/wmt22-comet-da` | 三元输入 `src` / `mt` / `ref`，用**未归一化**的原文（COMET 自带分词，归一化反而掉分）。batch 64 |
| **UTMOS（语音自然度）** | `sarulab-speech/UTMOS22` strong | 1～5 分。原计划用 `utmosv2`，但它要求 `torch<2.5`（见 10.5）。16kHz 单声道整句打分。同时给源音频打一遍作参照。`eval/run_utmos.py` |
| **SIM-o（音色相似度）** | `microsoft/wavlm-base-plus-sv` | x-vector 余弦，「源音频 vs 生成语音」配对。同时报**随机配对基线**（同数据集内打乱配对的均值） |
| **ESim（情绪相似度）** | `emotion2vec_plus_large` | utterance-level embedding 余弦，同样配对 + 随机基线。实测区分度很弱（见 10.5） |
| **AutoPCP（韵律一致性）** | SeamlessM4T `AutoPCP-multilingual-v2` | 回归分数，近似人工 1～4 分的韵律一致性评分。没有 pip 包，从 `facebookresearch/stopes` 取权重自己写推理入口。两段音频都要 16kHz 单声道。`eval/run_para_metrics.py` |
| **LAAL 系列（延迟）** | SimulEval `latency_scorer`（语音口径） | 见 10.3 |
| **RTF** | 自测墙钟 | `wall / 源音频时长`，只覆盖模型前向 + 声码器，不含读文件。并发测的值偏乐观，要单卡独占复测（见 10.4） |

### 10.3 延迟指标

流式系统（S5、Omni-L1～L4）才有。每条句子记一串「第 k 段输出语音」的三元组：开始/结束时刻、当时已消费的源音频秒数、这一段的计算耗时 `elapsed_sec`。

| 指标 | 含义 |
|---|---|
| **LAAL** | Length-Adaptive Average Lagging。按每段输出语音的**结束时刻**对齐源音频消费进度；长度自适应分母取 `max(源时长, 输出时长)`，避免输出过长或过短把 AL 拉偏 |
| **LAAL (CA)** | Computation-Aware：把实际计算耗时算进延迟。不带 CA 的是「理想无限算力」下的纯策略延迟，带 CA 的接近真实体验 |
| **LAAL-EOW** | End-Of-Word 变体，按每段最后一个词的结束时刻而不是整段结束时刻对齐 |
| **StartOffset / EndOffset** | 首段输出开始 / 末段输出结束相对源音频起止的偏移，各有 CA 版。StartOffset 就是常说的「首包延迟」 |

**CA 与非 CA 的差值直接反映算力压力**：S5 差 0.6s（RTF 0.46），Omni 差 1.6～1.9s（RTF 1.22～1.34）。

要算 CA 就得有逐段耗时，所以给 `infer_s5_streaming.py` 加了逐段 `elapsed_sec` 记录；Omni 侧用 agent 自带的 `delays` 加墙钟差（`eval/latency_eval.py`、`eval/stream_segments.py`）。

### 10.4 并行方式与 RTF 口径

7 张 H200，每张卡跑一片，片内各系统串行（`eval/run_worker.sh`）。模型与数据都放共享 FSX（`/fsx/stevehguo/eval/`，`eval/stage_to_fsx.sh` 负责搬），所以 7 张卡可以分布在不同节点。打分阶段同样切 7 份并行（ASR 最重）。

**RTF 要分两种口径**：7 进程并发时各进程挤同一张卡的 SM，单进程得到的算力不足一张卡，但进程内墙钟被 GPU 排队摊平，测出来偏乐观。所以另做**单卡独占**复测：按时长分层抽 30 句（覆盖 1.5～22.7s），独占一张卡顺序跑（`eval/run_rtf.sh`）。**对外引用单卡值。**

### 10.5 评测中碰到的问题

| 问题 | 原因 | 处理 |
|---|---|---|
| `utmosv2` 装不上 | 它依赖 `torch<2.5`，容器是 2.8；强装会拆掉 flash-attn | 退回官方 `UTMOS22` strong。两者口径不同，**不可与论文里 utmosv2 的数值直接比** |
| COMET 出现 `nan` | 空译文送进 COMET，模型对空串输出 nan，污染整体均值 | 打分跳过空串，「空译文」单列统计；BLEU/chrF 对空串按 0 分计入 |
| `AutoPCP` 没有 pip 包 | 是 SeamlessM4T `stopes` 的子模块，只在源码里 | 从 `facebookresearch/stopes` 拉权重自己写推理入口 |
| ESim 几乎不区分系统 | `emotion2vec_plus_large` 给任意两段同语种语音的余弦都在 0.85～0.92，动态范围太小 | 配对值与随机基线一起报，让区分度可见；结论里标注不建议作主指标 |
| Whisper 对极短输出幻觉 | 0.3s 以下的片段会被转写出完整句子 | 短于 0.3s 的生成语音按空处理，计入「空语音」列 |
| Whisper 把中文识别成日文 | 自动判语种在短中文上不稳 | `language` 按目标语强制指定 |
| S5 有 2 条 talker 序列超长失败 | `StaticCache` 上限写死 2048，RealSI zh→en 两条 7.5s 句子的 speech token 超了 | 按失败计入，未修（见第 11 节限制） |
| Omni 流式 L3/L4 有 5 条失败 | 官方 agent 在极少数句子上 talker 生成到上限仍未出 eos，`finish_write` 抛错 | 按失败计入，不改官方代码（保持「官方默认参数」口径） |
| SimulS2ST-Omni 官方脚本每条都重载模型 | 官方 `run_offline.py` 是单条 CLI 设计 | 只改成「加载一次循环跑」（`eval/run_omni_offline.py`），推理参数与官方默认完全一致 |
| 第一轮 S4 的 RTF 失真 | 第一轮用的是优化前的推理路径（无增量缓存），RTF 约 1.05 | 第二轮重跑 S4/S5，报告以第二轮为准；旧表保留对照 |
| 官方 Omni 要 transformers 4.52.3 | 容器是 4.57，`Qwen2_5Omni` 的接口有差异 | 把 4.52.3 与缺的依赖装到独立目录用 `PYTHONPATH` 前置，不动容器环境 |

### 10.6 复现

```bash
# 1. 切句级测试集
python custom/streaming_si/eval/prep_eval_sets.py <ds_root> <work_dir>
# 2. 合并并按时长均衡切 7 片
python custom/streaming_si/eval/make_shards.py <work_dir> 7
# 3. 搬模型与数据到 FSX，每张卡跑一片
bash custom/streaming_si/eval/stage_to_fsx.sh
bash custom/streaming_si/eval/run_worker.sh <片号> 7 <GPU 号>
# 4. 合并 -> 7 卡 ASR -> 打分（BLEU/chrF/COMET/UTMOS/SIM-o/ESim/AutoPCP/LAAL）
bash custom/streaming_si/eval/run_score.sh merge
bash custom/streaming_si/eval/run_score.sh asr 0 7 7
bash custom/streaming_si/eval/run_score.sh score
# 5. 单卡独占 RTF 复测
bash custom/streaming_si/eval/run_rtf.sh <GPU 号>
# 6. 汇总
python custom/streaming_si/eval/make_report.py <report_dir> REPORT.md
```

---

## 11. 与源项目的差异和已知限制

**行为差异**

- 多 epoch 时，分层抽样结果在各 epoch 之间复用（源项目每个 epoch 重抽）；默认 1 个 epoch 时两者分布相同。
- S5 验证只报两个方向合并后的 ALL 口径：没有分方向指标，也没有 prefix-drop 的 full/crop 配对验证。
- 学习率调度与源项目相差一个步位（见 3.3）。
- S1-S4 的续训与选 best 使用 swift 自带机制，与源项目的实现不完全相同。
- 导出比源项目更严格：shape 不匹配、非 `vision_proj.*` 的多余键一律报错，而不是跳过。

**限制**

- 训练只有 CPU tiny 模型冒烟：尚未用真实权重在多卡 / ZeRO-2 上训练，也没有和源项目的训练曲线比对。推理侧已用真实 S4 `checkpoint-24980` 在 H200 上跑过完整评测（第 10 节）。
- `infer_s2st.py` 只做离线整句推理，不接受 S5 adapter；S5 用 `inference/infer_s5_streaming.py`（chunk 级 WAIT/WRITE）。
- S5 的 talker 静态缓存上限 2048：极少数句子（RealSI zh→en 2/431）会超出而失败，尚未处理。
- `streaming_si_4b` 各阶段的 batch size 沿用 0.6B 变体的默认值，未针对 4B 调过；248k 词表下大概率需要调小，上集群前先探测显存。
- `flash-linear-attention` 与 `causal-conv1d` 不在 `requirements.txt` 里，4B 变体在 GPU 上训练需要自行安装。

**上集群前建议**：先用真实 S4 checkpoint 在单机多卡上跑 50~100 步，确认两件事：ZeRO-2 下 LoRA 与 token 行能否正常保存和恢复；loss 量级是否与源项目同一起点相近。

---

## 12. 常见报错

| 报错 | 原因 / 处理 |
|---|---|
| `... 是 streaming_si_4b（thinker_arch=qwen3_5），与 --model_type streaming_si 不符` | `SI_MODEL_TYPE` 与导出目录不一致；去掉 `SI_MODEL_TYPE` 让脚本自动识别 |
| `tokenizer 中 <|audio_pad|> -> ...，与 config.audio_ids 不一致` | 导出目录的 tokenizer 被替换过；用 `export_streaming_si.py` 重新导出 |
| `新增 N 个流式 token 超出 embedding 空位` | 词表空位不足（Qwen3-0.6B 与 Qwen3.5 的空位都够用，一般是换了基座） |
| `S4 token 行未与 lm_head 绑定` | `config.tie_word_embeddings` 被改成了 False，或 peft 版本变了；token 行绑定依赖 peft 0.15.2 的自动判定 |
| `adapter token ids ... 与当前 tokenizer ... 不一致` | 加载的 adapter 来自另一种变体或另一份 tokenizer |
| `指针行的采样计划与本进程重建的不一致` | loader 与 DataLoader worker 看到的 `SI_S4_*` 环境变量不同 |
| `S4ViewBuildError: ... admission_exhausted` | prefix-drop 候选及替补全部不可用（常见于数据太少或候选恰好都在 val 剔除组里）；检查索引与 val 划分 |
| `prefix-drop 训练要求 ...` | 实验臂约束不满足（见 6.3） |
| P2 跨 tar 目录报错 | 见 4.2：分机器训练，或合并 tar 后用 `SI_P2_WAV_DIR` 指定 |
| `Qwen3.5 增量缓存要求 fla / causal-conv1d` | 4B 变体的增量推理走加速核；装 `flash-linear-attention` 与 `causal-conv1d`，或用 `--no_thinker_cache` 回退整段重算（慢约 1.9 倍） |
| COMET 打分出现 `nan` | 空译文送进了 COMET；`score_eval.py` 会跳过空串，若自行调用需要先过滤 |
| `utmosv2` 安装报 `torch<2.5` | 容器是 torch 2.8，强装会拆掉 flash-attn；用官方 `UTMOS22` strong（见 10.5） |
| 训练首步报 `audio_encoder` / `talker_text_embed` 可训 | 冻结前缀写错；这两部分必须冻结 |
