# UniSS Phase 1/2 UniST Train Shard 四卡跑通计划

日期：2026-07-15 UTC

本计划只针对当前这一次受限实验：

- Phase 1 和 Phase 2 都先只用 `cmots/UniST` 的 `train-*.parquet` shard。
- 暂不使用论文 Phase 1 的 77.1k 小时 speech alignment 语料、WMT17，也暂不跑 Phase 3。
- 除了数据源按用户要求临时替换为 UniST train shard，模型、loss、packing、优化器、batch、学习率、Megatron-LM 训练方式尽量严格按 UniSS 论文 Implementation Details 执行。
- 使用后四张 GPU：`CUDA_VISIBLE_DEVICES=4,5,6,7`。
- 每一步先做可验证 smoke，再继续下一步；没有通过验证不进入真实训练。

参考来源：

- UniSS paper: https://arxiv.org/pdf/2509.21144
- UniST dataset: https://huggingface.co/datasets/cmots/UniST
- UniSS model/tokenizer: https://huggingface.co/cmots/UniSS
- Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Megatron Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge

## 1. 当前本地状态

项目目录：

```bash
/opt/dlami/nvme/jasonleeeli/projects/UniSS
```

所有安装、下载、缓存、临时文件都必须放在：

```bash
/opt/dlami/nvme/jasonleeeli
```

当前 conda env：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train
```

当前检查到的 UniST 本地数据：

```text
data/raw/UniST
  train shard 数：13
  train row 数：1,300,000
  损坏 parquet：0
```

完整 UniST train split 预期为：

```text
198 train parquet shards
19,785,924 train rows
```

当前 `tmux list-sessions` 只看到：

```text
uniss_assets_curl
```

也就是说 UniST 下载需要在后续执行阶段确认是否已经完整；如果没有完整，需要重新用 tmux 续跑。第一轮 pipeline bring-up 可以先使用已存在的 `train-*.parquet`，但文档和日志必须标注这是 current-shard run，不是 full-UniST run。

当前已实现并验证过的相关代码：

- `training/constants_uniss.py`：UniSS vocab / token id 常量。
- `training/sample_builders.py`：ASR、S2TT、TTS、MT、Quality、Performance、Direct S2ST sample 构造。
- `training/prepare_phase1_alignment.py`：从 UniST parquet 生成 Phase 1 ASR/S2TT/TTS。
- `training/prepare_unist_s2st.py`：从 UniST parquet 生成 Phase 2 Quality/Performance/Direct S2ST。
- `training/mix_sample_jsonl.py`：按整数比例混合 JSONL。
- `training/pack_sequences.py`：next-token shift、loss mask、18k packing、packed sample boundary。
- `training/megatron_uniss_dataset.py`：packed JSONL -> Megatron tensors。
- `training/pretrain_uniss_megatron.py`：复用 Megatron-LM GPT model/forward/loss/training loop。
- `training/initialize_uniss_hf_checkpoint.py`：Qwen2.5 HF checkpoint -> UniSS vocab HF checkpoint。
- `scripts/train_phase1.sh`、`scripts/train_phase2.sh`：Megatron-LM 启动脚本。

尚未完成：

- Qwen2.5-1.5B-Instruct base checkpoint 下载。
- UniSS-vocab HF checkpoint 到 Megatron checkpoint 的完整转换。
- GPU-visible shell 中 Megatron Bridge import / conversion 验证。
- 真实 Megatron optimizer step。
- Phase 1 的 UniST-only MT proxy 样本还需要补到预处理脚本中，原因见第 4.2 节。

## 2. 论文要求到本实验的映射

| 项目 | 论文 Implementation Details | 本次 UniST-only 四卡实验 |
| --- | --- | --- |
| Backbone | Qwen2.5-1.5B-Instruct | 相同 |
| Transformer 结构 | 不修改 architecture | 相同，不加 audio encoder，不改 decoder |
| Vocab | 扩展到 `180,407` | 相同，用 UniSS tokenizer/config |
| Training framework | Megatron-LM Framework | 相同，使用 `third_party/Megatron-LM` + 当前 repo adapter |
| Sequence packing | pack 到 `18k` token sequence | 相同，`--seq-length 18000` |
| Global batch | `2.3M` tokens，`128` packed sequences | 相同，`128 * 18000 = 2,304,000` tokens |
| Optimizer | AdamW | 相同 |
| Adam betas | `(0.9, 0.95)` | 相同 |
| Weight decay | `0.1` | 相同 |
| Precision | large-model bf16 训练 | 使用 `--bf16` |
| Phase 1 data | speech alignment data + WMT17 MT | 改为只从 UniST train shard 派生 ASR/S2TT/TTS/MT proxy |
| Phase 1 schedule | 3 epochs，LR `8e-4`，warmup 1 epoch | 相同 schedule shape，iteration 按本地 packed count 计算 |
| Phase 2 data | UniST General + Phase 1 replay，`2:1` | UniST train S2ST + UniST-derived Phase 1 replay，`2:1` |
| Phase 2 schedule | 1 epoch，LR `2e-4`，warmup 5% epoch | 相同 schedule shape，iteration 按本地 packed count 计算 |
| Loss | autoregressive next-token CE | 相同，masked next-token cross entropy |

这次实验的性质：

- 这是 pipeline bring-up，不是 paper-quality reproduction。
- 指标不能和论文对齐，因为 Phase 1 数据源被替换，并且可能只用了部分 UniST train shards。
- 只要不触碰用户的数据限制，其他实现细节都按论文靠齐。

## 3. GPU 与并行计划

使用后四卡：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
```

第一轮推荐并行配置：

```bash
export TP=1
export PP=1
export MICRO_BATCH_SIZE=1
```

此时：

```text
visible GPUs = 4
tensor parallel = 1
pipeline parallel = 1
data parallel = 4
global batch = 128 packed sequences
micro batch = 1 packed sequence / GPU
gradient accumulation = 128 / (4 * 1) = 32
tokens per optimizer step = 128 * 18000 = 2,304,000
```

如果 18k context OOM：

- 先保持 `seq_length=18000` 和 `global_batch_size=128` 不变。
- 先确认 `--bf16`、`--use-flash-attn`、`--recompute-activations` 生效。
- 再考虑 `TP=2`，此时 data parallel size 变为 2，gradient accumulation 改为 `128 / (2 * 1) = 64`。
- 只有机械 smoke 才允许临时缩短 seq length；严格实验 run 不允许缩短 18k。

GPU preflight：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=4,5,6,7 nvidia-smi
```

如果当前 shell 仍然看不到 CUDA/NVML，不启动训练；必须换到 GPU-visible shell 或修正环境。

## 4. 数据计划

### 4.1 UniST 下载与校验

目标路径：

```bash
data/raw/UniST
```

只使用 train shards：

```bash
data/raw/UniST/train-*.parquet
```

不使用 validation/test shards，不使用额外 WMT17 或外部 speech corpus。

如果需要继续下载 UniST，用 tmux 后台跑，并且所有 cache/tmp 都指向用户目录：

```bash
tmux new-session -d -s unist_download -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "env USER_ROOT=/opt/dlami/nvme/jasonleeeli \
       ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train \
       HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
       HUGGINGFACE_HUB_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub \
       TRANSFORMERS_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/transformers \
       TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp \
       HF_MAX_WORKERS=1 \
       HF_HUB_DISABLE_XET=1 \
       scripts/download_hf_assets.sh unist"
```

下载监控：

```bash
tmux list-sessions
tail -n 80 /opt/dlami/nvme/jasonleeeli/logs/unist_download_jasonleeeli.log
find data/raw/UniST -maxdepth 1 -type f -name 'train-*.parquet' | sort | wc -l
```

schema 与 row count 校验：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

required = {
    "id", "transcription", "translation",
    "source_glm", "source_bicodec", "target_bicodec",
    "bicodec_global", "src_lang", "tgt_lang",
}

paths = sorted(Path("data/raw/UniST").glob("train-*.parquet"))
assert paths, "no UniST train parquet shards"
rows = 0
for path in paths:
    pf = pq.ParquetFile(path)
    rows += pf.metadata.num_rows
    cols = set(pf.schema_arrow.names)
    missing = required - cols
    assert not missing, f"{path} missing columns: {sorted(missing)}"
print({"shards": len(paths), "rows": rows})
PY
```

本次执行策略：

- smoke run 使用当前已下载的 `train-*.parquet`。
- 如果正式预处理开始前有更多 train shard 下载完成，则 glob 会自动包含它们。
- 每次预处理都记录 shard 数、row 数、task count，避免后续不知道到底训练了多少数据。

### 4.2 Phase 1 数据：UniST-only 但保留论文四类 objective

论文 Phase 1 包含：

- ASR
- TTS
- S2TT
- MT

为了满足“只用 UniST train shard”同时尽量保留论文 objective，本次 Phase 1 这样构造：

| 任务 | 输入来源 | 目标来源 | 是否论文同源 |
| --- | --- | --- | --- |
| ASR | `source_glm` + `bicodec_global` | `transcription` | objective 相同，数据源替换 |
| S2TT | `source_glm` + `bicodec_global` | `translation` | objective 相同，数据源替换 |
| TTS | `transcription` + `bicodec_global` | `source_bicodec` semantic tokens | objective 相同，数据源替换 |
| MT proxy | `transcription` text | `translation` text | objective 相同，WMT17 替换为 UniST text pair |

当前 `training/prepare_phase1_alignment.py` 只生成 ASR/S2TT/TTS。因此执行前需要先补一个小改动：

- 方案 A：给 `prepare_phase1_alignment.py` 增加 `--include-mt-proxy`。
- 方案 B：新增 `training/prepare_phase1_unist.py`，统一输出 ASR/S2TT/TTS/MT proxy。

建议方案 A，改动小，也复用现有验证。

Phase 1 每条 UniST row 生成样本：

```text
1 ASR sample
1 S2TT sample
1 TTS sample
1 MT proxy sample
```

如果某行缺失 `source_bicodec`，只跳过该行 TTS，并记录 skipped count；ASR/S2TT/MT proxy 仍可生成。初始任务比例采用每行一条，不额外猜论文未公开的任务采样权重。

Phase 1 prompt/target 约定：

```text
ASR prompt:
  <|task_asr|> <src_lang> global_tokens source_glm_tokens
  <|write_generate|> <src_lang> <|start_content|>
ASR target:
  transcription_text <|end_content|> <|im_end|>

S2TT prompt:
  <|task_s2t_translation|> <tgt_lang> global_tokens source_glm_tokens
  <|write_generate|> <tgt_lang> <|start_content|>
S2TT target:
  translation_text <|end_content|> <|im_end|>

TTS prompt:
  <|task_tts|> <src_lang> global_tokens
  <|start_content|> transcription_text <|end_content|>
  <|write_generate|> <src_lang> <|speed_9|> <|start_semantic_token|>
TTS target:
  source_bicodec_semantic_tokens <|end_semantic_token|> <|im_end|>

MT proxy prompt:
  <|task_t2t_translation|> <tgt_lang>
  <|start_content|> transcription_text <|end_content|>
  <|write_generate|> <tgt_lang> <|start_content|>
MT proxy target:
  translation_text <|end_content|> <|im_end|>
```

Phase 1 生成命令目标形态：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/prepare_phase1_alignment.py \
  --input data/raw/UniST/train-*.parquet \
  --tokenizer pretrained_models/UniSS \
  --tasks asr s2tt tts \
  --include-mt-proxy \
  --output data/processed/phase1_unist_only/phase1_alignment_mt_proxy.jsonl
```

如果脚本还没有 `--include-mt-proxy`，第一步就是实现它，并单测验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python -m unittest discover training/tests -v
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/prepare_phase1_alignment.py --help
```

Phase 1 数据验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("data/processed/phase1_unist_only/phase1_alignment_mt_proxy.jsonl")
assert path.exists() and path.stat().st_size > 0
counts = Counter()
with path.open() as f:
    for line in f:
        row = json.loads(line)
        counts[row["task"]] += 1
        assert row["prompt_ids"]
        assert row["target_ids"]
        assert row["prompt_length"] == len(row["prompt_ids"])
        assert row["target_length"] == len(row["target_ids"])
print(counts)
assert {"asr", "s2tt", "tts", "mt"} <= set(counts)
PY
```

### 4.3 Phase 2 数据：UniST train S2ST

Phase 2 使用 UniST train shard 生成论文中的三种 S2ST prompting modes：

| 模式 | Prompt mode | Target |
| --- | --- | --- |
| Quality | slow mode | transcription -> translation -> target semantic speech tokens |
| Performance | balance mode | translation -> target semantic speech tokens |
| Direct S2ST | fast mode | target semantic speech tokens |

每条 UniST row 生成：

```text
1 quality sample
1 performance sample
1 direct_s2st sample
```

命令：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/prepare_unist_s2st.py \
  --input data/raw/UniST/train-*.parquet \
  --phase phase2 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase2_unist_only/phase2_s2st.jsonl
```

Phase 2 数据验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("data/processed/phase2_unist_only/phase2_s2st.jsonl")
assert path.exists() and path.stat().st_size > 0
counts = Counter()
with path.open() as f:
    for line in f:
        row = json.loads(line)
        counts[row["task"]] += 1
        assert row["prompt_ids"]
        assert row["target_ids"]
print(counts)
assert {"quality", "performance", "direct_s2st"} <= set(counts)
PY
```

### 4.4 Phase 2 mix：严格保留论文 2:1

论文 Phase 2 使用 UniST General 与 Phase 1 data 以 `2:1` 混合。本次映射为：

```text
UniST train Phase2 S2ST samples : UniST-derived Phase1 replay samples = 2 : 1
```

命令：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/mix_sample_jsonl.py \
  --group unist=2:data/processed/phase2_unist_only/phase2_s2st.jsonl \
  --group phase1=1:data/processed/phase1_unist_only/phase1_alignment_mt_proxy.jsonl \
  --output data/processed/phase2_unist_only/phase2_mix_2to1.jsonl
```

验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("data/processed/phase2_unist_only/phase2_mix_2to1.jsonl")
counts = Counter()
with path.open() as f:
    for line in f:
        row = json.loads(line)
        counts[row.get("phase", "unknown")] += 1
        counts[row["task"]] += 1
print(counts)
assert counts["phase2"] > 0
assert counts["phase1"] > 0
PY
```

### 4.5 18k sequence packing

Phase 1 packing：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/pack_sequences.py \
  --input data/processed/phase1_unist_only/phase1_alignment_mt_proxy.jsonl \
  --output data/megatron/phase1_unist_only/packed_train.jsonl \
  --seq-length 18000 \
  --drop-overlong
```

Phase 2 packing：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/pack_sequences.py \
  --input data/processed/phase2_unist_only/phase2_mix_2to1.jsonl \
  --output data/megatron/phase2_unist_only/packed_train.jsonl \
  --seq-length 18000 \
  --drop-overlong
```

packing 原则：

- 每条 raw sample 先做 next-token shift。
- 多条 shifted sample 串接到一个固定长度 `18000` packed sequence。
- 每条 packed sequence 保留 `sample_boundaries`。
- `position_ids` 在每条 sample 内从 0 重置。
- prompt 与 padding 的 `loss_mask=0`。
- target 的 `loss_mask=1`。
- packed sample 之间 attention 不允许泄漏，Megatron 训练走 `--sft` / `cu_seqlens` 路径。

packing 验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
from training.megatron_uniss_dataset import UniSSPackedJsonlDataset

for path in [
    "data/megatron/phase1_unist_only/packed_train.jsonl",
    "data/megatron/phase2_unist_only/packed_train.jsonl",
]:
    ds = UniSSPackedJsonlDataset(path, seq_length=18000)
    assert len(ds) > 0, path
    item = ds[0]
    assert item["tokens"].shape == (18000,)
    assert item["labels"].shape == (18000,)
    assert item["loss_mask"].shape == (18000,)
    assert item["position_ids"].shape == (18000,)
    assert item["cu_seqlens"].shape == (18001,)
    assert float(item["loss_mask"].sum()) > 0
    print(path, {"packed": len(ds), "first_loss_tokens": float(item["loss_mask"].sum())})
PY
```

## 5. Loss 计划

### 5.1 总体 objective

UniSS 训练是普通 autoregressive next-token prediction。LLM 不直接输出 waveform，也不训练 BiCodec decoder 或 GLM tokenizer。

构造方式：

```text
input_ids = prompt_ids + target_ids
tokens = input_ids[:-1]
labels = input_ids[1:]

loss_mask[i] = 1 if labels[i] belongs to target_ids else 0
loss_mask[i] = 0 for prompt labels and padding labels
```

训练 loss：

```text
ce_i = CrossEntropy(logits[i], labels[i])
loss = sum_i(ce_i * loss_mask[i]) / sum_i(loss_mask[i])
```

分布式训练时：

```text
global_loss = all_reduce(sum_loss) / all_reduce(num_active_loss_tokens)
```

不使用：

- waveform reconstruction loss
- mel/STFT loss
- speaker embedding loss
- CTC loss
- contrastive loss
- distillation loss
- 单独的 BiCodec/GLM tokenizer loss

### 5.2 Phase 1 loss

Phase 1 的总 loss 是任务混合后的 masked CE：

```text
L_phase1 = E_batch[masked_next_token_CE(sample)]
```

不是把四个 loss 固定加权相加；权重来自数据采样比例。本次初始比例为每条 UniST row 生成 ASR/S2TT/TTS/MT proxy 各一条，后续如果要改比例必须明确记录。

ASR：

```text
loss on:
  transcription_text
  <|end_content|>
  <|im_end|>

no loss on:
  task token
  language token
  global tokens
  source GLM tokens
  <|write_generate|>
```

S2TT：

```text
loss on:
  translation_text
  <|end_content|>
  <|im_end|>
```

TTS：

```text
loss on:
  source_bicodec_semantic_tokens
  <|end_semantic_token|>
  <|im_end|>

no loss on waveform or acoustic decoder outputs
```

MT proxy：

```text
loss on:
  translation_text
  <|end_content|>
  <|im_end|>
```

建议日志额外拆分：

```text
loss/phase1_total
loss/asr_text
loss/s2tt_text
loss/tts_semantic
loss/mt_proxy_text
tokens/active_loss_tokens
tokens/packed_utilization
```

主训练只需要总 masked CE；细分 loss 用于诊断。

### 5.3 Phase 2 loss

Phase 2 的总 loss：

```text
L_phase2 = E_batch[
  2/3 * masked_CE(UniST_S2ST_sample)
  + 1/3 * masked_CE(Phase1_replay_sample)
]
```

实现上不直接写这个公式加权，而是通过 `mix_sample_jsonl.py` 生成 `2:1` 混合数据，让 Megatron 的标准 LM loss 在 batch 上求平均。

Quality mode：

```text
loss on generated target sequence:
  transcription_text
  <|end_content|>
  generated translation-control tokens
  translation_text
  <|end_content|>
  <|start_semantic_token|>
  target_bicodec_semantic_tokens
  <|end_semantic_token|>
  <|im_end|>
```

Performance mode：

```text
loss on generated target sequence:
  translation_text
  <|end_content|>
  <|start_semantic_token|>
  target_bicodec_semantic_tokens
  <|end_semantic_token|>
  <|im_end|>
```

Direct S2ST：

```text
loss on generated target sequence:
  target_bicodec_semantic_tokens
  <|end_semantic_token|>
  <|im_end|>
```

Phase 1 replay：

```text
same loss rules as Phase 1
```

建议日志额外拆分：

```text
loss/phase2_total
loss/quality_transcription_text
loss/quality_translation_text
loss/quality_semantic
loss/performance_translation_text
loss/performance_semantic
loss/direct_semantic
loss/phase1_replay_total
tokens/active_loss_tokens
tokens/packed_utilization
```

如果暂时没有实现 segment-level logging，不能阻塞训练；但必须至少记录 Megatron reduced LM loss、active loss tokens、task counts。

## 6. Checkpoint 与模型初始化计划

需要本地资产：

```text
pretrained_models/UniSS/tokenizer.json
pretrained_models/UniSS/config.json
pretrained_models/Qwen2.5-1.5B-Instruct/config.json
pretrained_models/Qwen2.5-1.5B-Instruct/model*.safetensors
```

Qwen2.5 下载命令：

```bash
tmux new-session -d -s qwen25_download -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "env USER_ROOT=/opt/dlami/nvme/jasonleeeli \
       ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train \
       HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
       HUGGINGFACE_HUB_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub \
       TRANSFORMERS_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/transformers \
       TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp \
       HF_MAX_WORKERS=1 \
       HF_HUB_DISABLE_XET=1 \
       scripts/download_hf_assets.sh qwen"
```

初始化 UniSS vocab HF checkpoint：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python training/initialize_uniss_hf_checkpoint.py \
  --base-model pretrained_models/Qwen2.5-1.5B-Instruct \
  --uniss-tokenizer pretrained_models/UniSS \
  --output checkpoints/qwen2_1p5b_uniss_vocab_hf \
  --dtype bfloat16
```

验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM

path = "checkpoints/qwen2_1p5b_uniss_vocab_hf"
cfg = AutoConfig.from_pretrained(path)
assert cfg.vocab_size == 180407
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
emb = model.get_input_embeddings().weight
assert emb.shape[0] == 180407
print({"vocab_size": cfg.vocab_size, "embedding": tuple(emb.shape)})
PY
```

Megatron checkpoint conversion 首选 Megatron Bridge：

```bash
PYTHONPATH=third_party/Megatron-Bridge/src:third_party/Megatron-LM:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
from megatron.bridge import AutoBridge

AutoBridge.import_ckpt(
    "checkpoints/qwen2_1p5b_uniss_vocab_hf",
    "checkpoints/qwen2_1p5b_uniss_vocab_megatron",
)
PY
```

已知风险：

- 当前 shell 曾经因为看不到 CUDA driver/NVML，导入 Megatron Bridge 时触发 Triton driver error。
- conversion 必须在 GPU-visible shell 再验证。
- 如果 Bridge 仍失败，fallback 是实现 Qwen2 HF -> Megatron 参数映射；但训练本身仍必须通过 Megatron-LM，而不是改用纯 Transformers 训练。

conversion gate：

- 先用 tiny Qwen2 checkpoint 做 HF -> Megatron -> HF 或 logits smoke。
- 再转换完整 Qwen2.5 UniSS-vocab checkpoint。
- 完整转换后确认 Megatron 能 load checkpoint 并跑 1 step dry/smoke。

## 7. Megatron-LM 使用方式

本项目不把 UniSS repo 搬进 Megatron-LM 里，也不直接 fork 大量 Megatron 文件。训练方式是：

- `third_party/Megatron-LM` 作为外部训练框架。
- 当前 repo 的 `training/pretrain_uniss_megatron.py` 作为 adapter。
- adapter 复用 Megatron-LM 的：
  - distributed initialization
  - GPT model provider
  - forward step
  - masked LM loss
  - AdamW optimizer
  - LR scheduler
  - checkpointing
  - tensor/pipeline/data parallel
- adapter 只替换 dataset provider，让 Megatron 读取 UniSS packed JSONL。

关键参数：

```bash
--sft
--uniss-packed-train <packed_train.jsonl>
--uniss-strict-paper-config
--tokenizer-type NullTokenizer
--vocab-size 180407
--seq-length 18000
--global-batch-size 128
```

`--uniss-strict-paper-config` 会检查：

```text
seq_length == 18000
global_batch_size == 128
```

因此这次不是“看起来像 Megatron”，而是实际通过 Megatron-LM 的训练 loop 跑。

## 8. 训练 schedule 与启动命令

通用环境变量：

```bash
export USER_ROOT=/opt/dlami/nvme/jasonleeeli
export ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub
export TRANSFORMERS_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/transformers
export PIP_CACHE_DIR=/opt/dlami/nvme/jasonleeeli/cache/pip
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export TP=1
export PP=1
export MICRO_BATCH_SIZE=1
export PYTHONPATH=/opt/dlami/nvme/jasonleeeli/projects/UniSS/third_party/Megatron-LM:/opt/dlami/nvme/jasonleeeli/projects/UniSS:$PYTHONPATH
```

计算 current-shard iteration：

```bash
PHASE1_PACKED=$(wc -l < data/megatron/phase1_unist_only/packed_train.jsonl)
PHASE1_EPOCH_ITERS=$(( (PHASE1_PACKED + 127) / 128 ))
PHASE1_TRAIN_ITERS=$(( 3 * PHASE1_EPOCH_ITERS ))
PHASE1_WARMUP_ITERS=${PHASE1_EPOCH_ITERS}

PHASE2_PACKED=$(wc -l < data/megatron/phase2_unist_only/packed_train.jsonl)
PHASE2_EPOCH_ITERS=$(( (PHASE2_PACKED + 127) / 128 ))
PHASE2_TRAIN_ITERS=${PHASE2_EPOCH_ITERS}
PHASE2_WARMUP_ITERS=$(( (PHASE2_EPOCH_ITERS + 19) / 20 ))
if [ "${PHASE2_WARMUP_ITERS}" -lt 1 ]; then PHASE2_WARMUP_ITERS=1; fi

echo "phase1 packed=${PHASE1_PACKED} epoch_iters=${PHASE1_EPOCH_ITERS} train_iters=${PHASE1_TRAIN_ITERS} warmup=${PHASE1_WARMUP_ITERS}"
echo "phase2 packed=${PHASE2_PACKED} epoch_iters=${PHASE2_EPOCH_ITERS} train_iters=${PHASE2_TRAIN_ITERS} warmup=${PHASE2_WARMUP_ITERS}"
```

如果 packed sequence 太少，10-step smoke 可以强制 `TRAIN_ITERS=10`，但必须标注为 smoke。正式 current-shard run 使用上面计算的 epoch schedule。

### 8.1 Phase 1 smoke

目的：确认 GPU、Megatron、checkpoint、dataset、loss、optimizer step 都能跑通。

```bash
tmux new-session -d -s uniss_phase1_smoke -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase1_unist_only/packed_train.jsonl \
   LOAD_CHECKPOINT=checkpoints/qwen2_1p5b_uniss_vocab_megatron \
   SAVE_DIR=checkpoints/uniss_phase1_unist_only_smoke \
   TRAIN_ITERS=10 \
   LR_WARMUP_ITERS=1 \
   SAVE_INTERVAL=10 LOG_INTERVAL=1 \
   scripts/train_phase1.sh"
```

通过标准：

- 10 optimizer steps 完成。
- loss 是 finite number。
- checkpoint 成功保存。
- 没有 CUDA OOM、NCCL hang、dataset shape error。

### 8.2 Phase 1 current-shard run

论文-aligned hyperparameters：

```text
epochs = 3
LR = 8e-4
warmup = 1 epoch
seq_length = 18000
global_batch_size = 128
AdamW betas = 0.9, 0.95
weight_decay = 0.1
bf16 = true
```

启动：

```bash
tmux new-session -d -s uniss_phase1_unist -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase1_unist_only/packed_train.jsonl \
   LOAD_CHECKPOINT=checkpoints/qwen2_1p5b_uniss_vocab_megatron \
   SAVE_DIR=checkpoints/uniss_phase1_unist_only \
   TRAIN_ITERS=${PHASE1_TRAIN_ITERS} \
   LR_WARMUP_ITERS=${PHASE1_WARMUP_ITERS} \
   SAVE_INTERVAL=100 LOG_INTERVAL=1 \
   scripts/train_phase1.sh"
```

### 8.3 Phase 2 smoke

Phase 2 必须从 Phase 1 checkpoint 继续：

```bash
tmux new-session -d -s uniss_phase2_smoke -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase2_unist_only/packed_train.jsonl \
   LOAD_CHECKPOINT=checkpoints/uniss_phase1_unist_only_smoke \
   SAVE_DIR=checkpoints/uniss_phase2_unist_only_smoke \
   TRAIN_ITERS=10 \
   LR_WARMUP_ITERS=1 \
   SAVE_INTERVAL=10 LOG_INTERVAL=1 \
   scripts/train_phase2.sh"
```

通过标准同 Phase 1 smoke。

### 8.4 Phase 2 current-shard run

论文-aligned hyperparameters：

```text
epochs = 1
LR = 2e-4
warmup = 5% epoch
phase2_new_data : phase1_replay = 2 : 1
seq_length = 18000
global_batch_size = 128
AdamW betas = 0.9, 0.95
weight_decay = 0.1
bf16 = true
```

启动：

```bash
tmux new-session -d -s uniss_phase2_unist -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  "CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase2_unist_only/packed_train.jsonl \
   LOAD_CHECKPOINT=checkpoints/uniss_phase1_unist_only \
   SAVE_DIR=checkpoints/uniss_phase2_unist_only \
   TRAIN_ITERS=${PHASE2_TRAIN_ITERS} \
   LR_WARMUP_ITERS=${PHASE2_WARMUP_ITERS} \
   SAVE_INTERVAL=100 LOG_INTERVAL=1 \
   scripts/train_phase2.sh"
```

## 9. 验证 gate

每个 gate 失败都停止，不继续下一阶段。

1. 环境 gate
   - conda env 可用。
   - `python -m unittest discover training/tests -v` 通过。
   - `bash -n scripts/train_phase1.sh` 通过。
   - `bash -n scripts/train_phase2.sh` 通过。
   - `CUDA_VISIBLE_DEVICES=4,5,6,7 nvidia-smi` 通过。

2. 数据 gate
   - `train-*.parquet` 数量大于 0。
   - parquet schema 包含必需列。
   - row count 与日志一致。

3. Phase 1 预处理 gate
   - `--include-mt-proxy` 实现并有单测。
   - 输出 JSONL 非空。
   - task count 包含 `asr/s2tt/tts/mt`。
   - 每条样本 `prompt_ids/target_ids` 非空。

4. Phase 2 预处理 gate
   - 输出 JSONL 非空。
   - task count 包含 `quality/performance/direct_s2st`。

5. Mix gate
   - Phase 2 mix 中 `phase2:phase1` 约为 `2:1`。
   - 混合脚本 deterministic，重复运行输出一致。

6. Packing gate
   - packed JSONL 非空。
   - 每条 packed item 长度为 `18000`。
   - `loss_mask.sum() > 0`。
   - `cu_seqlens.shape == (18001,)`。
   - overlong dropped count 被记录。

7. Checkpoint gate
   - UniSS-vocab HF checkpoint 可加载。
   - vocab size 为 `180407`。
   - Megatron checkpoint conversion 完成。
   - Megatron 可 load checkpoint。

8. Phase 1 smoke gate
   - 10 steps 完成。
   - loss finite。
   - checkpoint saved。

9. Phase 2 smoke gate
   - 从 Phase 1 smoke checkpoint load。
   - 10 steps 完成。
   - loss finite。
   - checkpoint saved。

10. Current-shard run gate
   - Phase 1 按 3 epoch-equivalent 完成。
   - Phase 2 按 1 epoch-equivalent 完成。
   - final checkpoint reload test 通过。

## 10. 输出目录与日志

数据输出：

```text
data/processed/phase1_unist_only/phase1_alignment_mt_proxy.jsonl
data/processed/phase2_unist_only/phase2_s2st.jsonl
data/processed/phase2_unist_only/phase2_mix_2to1.jsonl
data/megatron/phase1_unist_only/packed_train.jsonl
data/megatron/phase2_unist_only/packed_train.jsonl
```

checkpoint 输出：

```text
checkpoints/qwen2_1p5b_uniss_vocab_hf
checkpoints/qwen2_1p5b_uniss_vocab_megatron
checkpoints/uniss_phase1_unist_only_smoke
checkpoints/uniss_phase1_unist_only
checkpoints/uniss_phase2_unist_only_smoke
checkpoints/uniss_phase2_unist_only
```

建议每次 run 额外保存 metadata：

```json
{
  "date_utc": "2026-07-15",
  "data_source": "cmots/UniST train shards only",
  "num_train_shards": 13,
  "num_train_rows": 1300000,
  "phase1_tasks": {"asr": 0, "s2tt": 0, "tts": 0, "mt": 0},
  "phase2_tasks": {"quality": 0, "performance": 0, "direct_s2st": 0},
  "seq_length": 18000,
  "global_batch_size": 128,
  "visible_gpus": "4,5,6,7",
  "megatron_lm_commit": "see training/MEGATRON_COMMIT",
  "megatron_bridge_commit": "see training/MEGATRON_BRIDGE_COMMIT"
}
```

实际 task count 在预处理后填入真实值。

## 11. 主要风险

1. 数据偏离论文：
   - Phase 1 的 WMT17 被 UniST MT proxy 替代。
   - Phase 1 speech alignment corpus 被 UniST train 替代。
   - 结果只能证明流程跑通，不能代表论文指标。

2. 当前可能只有部分 UniST shard：
   - current-shard run 可以先跑。
   - full-UniST run 需要等 198 train shards 下载完整。

3. Megatron Bridge / CUDA import：
   - 之前在非 GPU-visible shell 触发过 Triton driver error。
   - 必须在 `CUDA_VISIBLE_DEVICES=4,5,6,7` 可见的 shell 重测。

4. 18k context 显存：
   - 先用 recompute + flash attention。
   - 如 OOM，再改 TP/accumulation，不先改论文的 18k 和 2.3M batch。

5. Segment-level loss logging：
   - 不是训练 correctness 的硬依赖。
   - 如果时间不够，先保证 total masked CE 正确，再补细分日志。

## 12. 立即执行顺序

1. 确认 UniST train shard 下载状态；必要时用 tmux 续跑。
2. 下载 Qwen2.5-1.5B-Instruct 到本地 `pretrained_models/`。
3. 补 `prepare_phase1_alignment.py --include-mt-proxy`，并写单测。
4. 运行完整 unit tests。
5. 用当前 UniST train shards 生成 Phase 1 JSONL。
6. 用当前 UniST train shards 生成 Phase 2 JSONL。
7. 按 `2:1` 混合 Phase 2 与 Phase 1 replay。
8. pack 到 18k。
9. 初始化 UniSS-vocab HF checkpoint。
10. 用 Megatron Bridge 转 Megatron checkpoint。
11. 后四卡跑 Phase 1 10-step smoke。
12. 后四卡跑 Phase 2 10-step smoke。
13. 两个 smoke 都通过后，跑 current-shard Phase 1 3 epoch-equivalent。
14. Phase 1 完成后，跑 current-shard Phase 2 1 epoch-equivalent。

## 13. 当前 13-shard 数据产物

本次实验固定只使用下面 13 个 train shard：

```text
data/raw/UniST/train-00000.parquet
...
data/raw/UniST/train-00012.parquet
```

即使后台继续下载 `train-00013.parquet` 之后的 shard，本次 run 不混入新增文件。

已完成产物：

```text
data/processed/phase1_unist13_sharded/train-00000.jsonl ... train-00012.jsonl
  5,200,000 samples
  asr=1,300,000
  s2tt=1,300,000
  tts=1,300,000
  mt=1,300,000

data/processed/phase2_unist13_sharded/train-00000.jsonl ... train-00012.jsonl
  3,900,000 samples
  quality=1,300,000
  performance=1,300,000
  direct_s2st=1,300,000

data/processed/phase2_unist13_mix/phase2_mix_2to1.jsonl
  total=5,850,000
  unist=3,900,000
  phase1_replay=1,950,000
```

Packed 18k 数据：

```text
data/megatron/phase1_unist13/packed_train.jsonl
  packed_sequences=54,144

data/megatron/phase2_unist13_mix/packed_train.jsonl
  packed_sequences=133,681
```

UniST dev validation packed 数据：

```text
data/megatron/validation_unist_dev/phase1_valid_packed.jsonl
  packed_sequences=241

data/megatron/validation_unist_dev/phase2_valid_packed.jsonl
  packed_sequences=571
```

13-shard schedule：

```text
Phase 1:
  epoch_iters = ceil(54144 / 128) = 423
  train_iters = 3 * 423 = 1269
  warmup_iters = 423

Phase 2:
  epoch_iters = ceil(133681 / 128) = 1045
  train_iters = 1045
  warmup_iters = ceil(0.05 * 1045) = 53
```

## 14. Validation/Test 音频生成与完成判定

论文和 README 报告的核心评估维度包括：

- Translation Fidelity: Speech-BLEU, Text-BLEU.
- Prosody Preservation: A.PCP.
- Duration Consistency: SLC 0.2, SLC 0.4.
- Speech Quality: UTMOS.
- 主观评测：voice similarity, emotion similarity, speech naturalness / MOS.

本次 13-shard bring-up 不声称复现论文指标，但训练期间仍需要保存可听的 validation 音频，最后在 test split 上生成音频。

### 14.1 什么时候判断训练完成

严格 schedule 完成条件：

```text
Phase 1 完成：
  从 Qwen2.5-1.5B-Instruct UniSS-vocab checkpoint 开始
  跑满 1269 optimizer steps
  LR=8e-4, warmup_iters=423
  dev validation loss finite，没有持续 NaN/Inf

Phase 2 完成：
  从 Phase 1 final checkpoint 开始
  跑满 1045 optimizer steps
  LR=2e-4, warmup_iters=53
  dev validation loss finite
```

不要只用 training loss 判断完成。training loss 正常下降只能说明优化在工作；完成必须同时满足固定 step schedule、checkpoint 可 reload、dev validation loss 可跑、固定 validation 样本能生成非空 semantic tokens 和 wav。

如果需要 early stop 或选择 best checkpoint：

- 先按论文 schedule 跑满。
- 每个 save/eval interval 记录 dev validation loss。
- 每个 save interval 对固定 dev 样本生成 wav，保存 metadata。
- 在 Phase 2 中优先选择：
  - dev validation loss 不再明显下降的后段 checkpoint；
  - generation success rate 高；
  - semantic token count 合理；
  - 合成音频没有明显爆音、静音、重复、截断；
  - translation 文本与 reference translation 大体一致；
  - duration 不明显偏离 reference/source。

当前没有完整自动指标链路时，最终 checkpoint 至少要通过人工听感 spot check。后续可接入 ASR、BLEU、UTMOS、speaker similarity 模型，把上述人工项自动化。

### 14.2 Megatron validation loss

训练脚本支持传入 `VALID_DATA`：

```bash
VALID_DATA=data/megatron/validation_unist_dev/phase1_valid_packed.jsonl \
EVAL_INTERVAL=100 EVAL_ITERS=10 \
scripts/train_phase1.sh

VALID_DATA=data/megatron/validation_unist_dev/phase2_valid_packed.jsonl \
EVAL_INTERVAL=100 EVAL_ITERS=10 \
scripts/train_phase2.sh
```

validation loss 仍然是 masked next-token CE，不解码音频。

### 14.3 每个 interval 生成 validation 音频

新增脚本：

```text
training/convert_uniss_checkpoint.py
scripts/convert_uniss_checkpoint.sh
training/generate_unist_eval_audio.py
scripts/generate_unist_audio_eval.sh
scripts/watch_uniss_checkpoint_audio_eval.sh
```

音频生成使用 HF checkpoint。Megatron checkpoint 必须先导出到 HF checkpoint 目录，然后运行：

```bash
HF_CHECKPOINT=checkpoints/exported_hf/phase2_iter_000100 \
STEP_NAME=phase2_iter_000100 \
SPLIT=dev \
LIMIT_RECORDS=8 \
MODES="quality performance direct_s2st" \
EVAL_CUDA_VISIBLE_DEVICES=4 \
scripts/generate_unist_audio_eval.sh
```

输出：

```text
eval_outputs/phase2_iter_000100_dev/
  results.jsonl
  summary.json
  wav/*.wav
  reference_wav/*.wav
```

`results.jsonl` 会记录：

- sample id / mode / language
- reference transcription / translation
- generated raw text / cleaned text
- generated semantic token count
- generated wav path
- reference wav path
- error field
- checkpoint path

如果训练占满 GPUs 4-7，音频生成不能同时抢同一批 GPU。实际执行有两种方式：

1. 分段训练：每 `SAVE_INTERVAL` 停一次，导出 HF，生成 validation wav，再从 checkpoint 继续训练。
2. 训练连续跑：只在每个 checkpoint 保存后排队，等训练阶段结束后批量导出并生成所有 interval 的 validation wav。

为了先跑通流程，本次建议：

- smoke run 后立即生成一次 validation wav。
- Phase 1 final 后生成一次 validation wav。
- Phase 2 每 100 step 保存 checkpoint，训练结束后对关键 checkpoint 生成 wav。
- Phase 2 final 后生成 dev + test wav。

已实现的自动 watcher 可以在训练期间旁路监控新 checkpoint，并在 checkpoint 目录稳定后自动执行：

```text
Megatron iter_* checkpoint
  -> scripts/convert_uniss_checkpoint.sh export
  -> Hugging Face checkpoint under checkpoints/exported_hf/
  -> scripts/generate_unist_audio_eval.sh
  -> eval_outputs/<run>_<iter>_dev/*.wav
```

Phase 1 watcher 示例：

```bash
tmux new-session -d -s uniss_phase1_audio_watch \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'CHECKPOINT_DIR=checkpoints/uniss_phase1_unist13_full \
   HF_REFERENCE=checkpoints/qwen2_1p5b_uniss_vocab_hf \
   RUN_NAME=phase1_unist13 \
   SPLIT=dev LIMIT_RECORDS=8 MODES="quality performance" \
   EVAL_CUDA_VISIBLE_DEVICES=4 \
   POLL_SECONDS=60 STABILITY_SECONDS=120 \
   scripts/watch_uniss_checkpoint_audio_eval.sh \
   > logs/phase1_audio_watch.log 2>&1'
```

Phase 2 watcher 示例：

```bash
tmux new-session -d -s uniss_phase2_audio_watch \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'CHECKPOINT_DIR=checkpoints/uniss_phase2_unist13_full \
   HF_REFERENCE=checkpoints/qwen2_1p5b_uniss_vocab_hf \
   RUN_NAME=phase2_unist13 \
   SPLIT=dev LIMIT_RECORDS=8 MODES="quality performance direct_s2st" \
   EVAL_CUDA_VISIBLE_DEVICES=4 \
   POLL_SECONDS=60 STABILITY_SECONDS=120 \
   scripts/watch_uniss_checkpoint_audio_eval.sh \
   > logs/phase2_audio_watch.log 2>&1'
```

注意：watcher 会使用 `scripts/convert_uniss_checkpoint.sh export`，因此训练后生成音频前必须已经完成 Megatron Bridge 依赖安装，并且 base HF reference checkpoint `checkpoints/qwen2_1p5b_uniss_vocab_hf` 存在。

### 14.4 最终 test 音频生成

最终 Phase 2 checkpoint 导出 HF 后运行：

```bash
HF_CHECKPOINT=checkpoints/exported_hf/phase2_final \
STEP_NAME=phase2_final \
SPLIT=test \
LIMIT_RECORDS=23369 \
MODES="quality performance direct_s2st" \
EVAL_CUDA_VISIBLE_DEVICES=4 \
scripts/generate_unist_audio_eval.sh
```

如果先做小规模 test smoke：

```bash
HF_CHECKPOINT=checkpoints/exported_hf/phase2_final \
STEP_NAME=phase2_final_test_smoke \
SPLIT=test \
LIMIT_RECORDS=32 \
MODES="quality performance" \
scripts/generate_unist_audio_eval.sh
```

最终 Phase 2 checkpoint 如果还没有 HF export，先执行：

```bash
scripts/convert_uniss_checkpoint.sh export \
  --hf-model checkpoints/qwen2_1p5b_uniss_vocab_hf \
  --megatron-path checkpoints/uniss_phase2_unist13_full \
  --hf-output checkpoints/exported_hf/phase2_final
```

最终 test 产物用于：

- 人工听感检查。
- translation 文本 BLEU/COMET 计算。
- 对生成 wav 跑 ASR 后计算 Speech-BLEU。
- 计算 duration consistency / SLC。
- 计算 UTMOS 或其他 speech quality proxy。
- 如接入 speaker/emotion 模型，再计算 voice similarity / emotion similarity。

## 15. 2026-07-16 实际执行校正

本节记录在真实四卡 smoke 中发现并修正的执行细节。后续 full run 以本节命令为准。

### 15.1 已通过验证

环境与 checkpoint：

- `pybind11==3.0.4` 已安装到 `/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train`，Megatron dataset helper 已能编译。
- `checkpoints/qwen2_1p5b_uniss_vocab_hf` 已创建，vocab size 为 `180407`。
- `checkpoints/qwen2_1p5b_uniss_vocab` 已转换为 Megatron torch_dist checkpoint。

训练 smoke：

- Phase 1 `TP=2` smoke 已跑完 10 iter、2 次 validation、保存 checkpoint：`checkpoints/uniss_phase1_smoke_tp2/iter_0000010`。
- Phase 2 `TP=2` smoke 已从 Phase 1 smoke checkpoint 以 `--finetune` 重置 iteration 后跑完 10 iter、2 次 validation、保存 checkpoint：`checkpoints/uniss_phase2_smoke_tp2/iter_0000010`。
- 单元测试：`python -m unittest discover training/tests -v`，62 tests OK。

关键 smoke 结果：

```text
Phase 1 smoke:
  iteration 10 train lm loss: 1.076855E+01
  iteration 10 validation lm loss: 9.815033E+00
  final validation lm loss: 9.796430E+00

Phase 2 smoke:
  iteration 10 train lm loss: 9.511981E+00
  iteration 10 validation lm loss: 9.161410E+00
  final validation lm loss: 9.119370E+00
```

### 15.2 必须保留的 Megatron 参数

四卡 H100 上 `TP=1` 会在 18k context、180k vocab 的 cross entropy fp32 logits 处 OOM。严格 run 保持论文的 `seq_length=18000` 和 `global_batch_size=128`，使用：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export TP=2
export PP=1
export MICRO_BATCH_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`TP=2` 时 Megatron 要求：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

训练脚本现在默认设置该环境变量。

Qwen2.5 checkpoint 结构要求：

```bash
--disable-bias-linear
--add-qkv-bias
```

Packed SFT 训练要求：

```bash
--sft
--no-create-attention-mask-in-dataloader
```

阶段切换或从 HF 初始化 checkpoint 开始训练时，必须只加载模型权重并重置 iteration：

```bash
--no-load-optim
--no-load-rng
--finetune
```

如果是同一个 phase 的断点续训，才使用：

```bash
FINETUNE=0 LOAD_OPTIM=1 LOAD_RNG=1
```

### 15.3 数据长度与 validation

Megatron 会把 train/valid/test 的目标样本数传给 dataset provider。当前 adapter 已按目标长度 repeat finite packed JSONL：

- Phase 1 train 13-shard packed：`54144 * 3 = 162432`，对应 `1269 * 128` samples。
- Phase 2 train 13-shard mix packed：`133681`，对应 `1045 * 128 = 133760` samples，最后少量样本 repeat。
- Validation packed 文件小于 `eval_iters * global_batch_size` 时也 repeat，避免 periodic validation `StopIteration`。

这不会改变单条样本 loss；只是在当前 13-shard bring-up 中复用有限 dev/train packed rows。

### 15.4 Full Phase 1 后台命令

```bash
tmux new-session -d -s uniss_phase1_full \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'PATH=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin:$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=2 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase1_unist13/packed_train.jsonl \
   VALID_DATA=data/megatron/validation_unist_dev/phase1_valid_packed.jsonl \
   LOAD_CHECKPOINT=checkpoints/qwen2_1p5b_uniss_vocab \
   SAVE_DIR=checkpoints/uniss_phase1_unist13_full \
   TRAIN_ITERS=1269 LR_WARMUP_ITERS=423 \
   SAVE_INTERVAL=100 EVAL_INTERVAL=100 EVAL_ITERS=10 LOG_INTERVAL=10 \
   scripts/train_phase1.sh \
   > logs/uniss_phase1_unist13_full.log 2>&1'
```

### 15.5 Full Phase 2 后台命令

Phase 1 full 完成后执行：

```bash
tmux new-session -d -s uniss_phase2_full \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'PATH=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin:$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=2 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase2_unist13_mix/packed_train.jsonl \
   VALID_DATA=data/megatron/validation_unist_dev/phase2_valid_packed.jsonl \
   LOAD_CHECKPOINT=checkpoints/uniss_phase1_unist13_full \
   SAVE_DIR=checkpoints/uniss_phase2_unist13_full \
   TRAIN_ITERS=1045 LR_WARMUP_ITERS=53 \
   SAVE_INTERVAL=100 EVAL_INTERVAL=100 EVAL_ITERS=10 LOG_INTERVAL=10 \
   scripts/train_phase2.sh \
   > logs/uniss_phase2_unist13_full.log 2>&1'
```
