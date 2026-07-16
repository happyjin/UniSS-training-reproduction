# Qwen2.5-0.5B-Instruct UniSS 快速验证计划

日期：2026-07-16 UTC

目标：在不影响当前 `Qwen2.5-1.5B-Instruct` UniSS 训练实验的前提下，新增一条独立的 `Qwen2.5-0.5B-Instruct` 小模型验证链路，用于快速检查数据处理、Megatron-LM 训练、checkpoint 转换、validation、test/audio generation 是否端到端工作。

该计划不是替代论文严格复现。论文严格复现主线仍然使用 `Qwen/Qwen2.5-1.5B-Instruct`。小模型只作为快速验证、调试和 ablation。

## 1. 不影响 1.5B 主实验的硬约束

当前 1.5B 主实验相关目录必须保持不变：

```text
pretrained_models/Qwen2.5-1.5B-Instruct
checkpoints/qwen2_1p5b_uniss_vocab_hf
checkpoints/qwen2_1p5b_uniss_vocab
checkpoints/uniss_phase1_unist13_full
checkpoints/uniss_phase2_unist13_full
logs/uniss_phase1_unist13_full.log
logs/uniss_phase2_unist13_full.log
scripts/train_phase1.sh
scripts/train_phase2.sh
scripts/train_phase3.sh
```

小模型实验只能写入独立路径：

```text
pretrained_models/Qwen2.5-0.5B-Instruct
checkpoints/qwen2_0p5b_uniss_vocab_hf
checkpoints/qwen2_0p5b_uniss_vocab
checkpoints/uniss_qwen0p5b_phase1_smoke
checkpoints/uniss_qwen0p5b_phase2_smoke
checkpoints/uniss_qwen0p5b_phase3_smoke
checkpoints/uniss_qwen0p5b_phase1_unist13_full
checkpoints/uniss_qwen0p5b_phase2_unist13_full
checkpoints/uniss_qwen0p5b_phase3_unist13_full
checkpoints/exported_hf/qwen0p5b_*
logs/uniss_qwen0p5b_*.log
runs/uniss_qwen0p5b_*
```

执行小模型训练前必须检查：

```bash
tmux list-sessions
nvidia-smi
cat checkpoints/uniss_phase1_unist13_full/latest_checkpointed_iteration.txt || true
```

如果 `uniss_phase1_full` 或 `uniss_phase2_wait` / Phase 2 full 正在使用 GPU `4,5,6,7`，不要启动小模型训练。小模型训练默认也使用 `CUDA_VISIBLE_DEVICES=4,5,6,7`，应等待主实验释放后再跑。除非明确确认其他 GPU 空闲并得到同意，不要抢占 `0,1,2,3`。

## 2. 小模型选择与结构

模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

官方 model card 说明该模型是 instruction-tuned Qwen2.5 0.5B，架构为 RoPE、SwiGLU、RMSNorm、Attention QKV bias、tied word embeddings，参数约 `0.49B`，非 embedding 参数约 `0.36B`，`24` layers，GQA 为 `14` Q heads / `2` KV heads，context length `32768`。

来源：

```text
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

下载后必须以本地 `config.json` 为准，预期关键字段如下：

```json
{
  "model_type": "qwen2",
  "num_hidden_layers": 24,
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "vocab_size": 151936
}
```

扩展到 UniSS vocab 后：

```text
base vocab: 151936
target vocab: 180407
added tokens: 28471
embedding shape: [180407, 896]
estimated total params: about 0.52B
```

## 3. 与论文一致/不一致边界

保持一致：

- UniSS tokenizer vocab size 仍为 `180407`。
- GLM4 linguistic token、BiCodec global speaker token、BiCodec semantic token 的 ID offset 不变。
- Phase 1/2/3 prompt 结构不变。
- Loss 仍为 masked autoregressive next-token CE。
- Sequence length 仍为 `18000`。
- Global batch size 仍为 `128` packed sequences，也就是 `2,304,000` tokens / optimizer step。
- Optimizer 保持 AdamW，betas `(0.9, 0.95)`，weight decay `0.1`，bf16。
- Phase 1 LR `8e-4`，Phase 2 LR `2e-4`，warmup 比例保持当前 13-shard 计划。
- Megatron-LM training framework 不变。

不一致：

- Backbone 从论文的 `Qwen2.5-1.5B-Instruct` 换成 `Qwen2.5-0.5B-Instruct`。
- 当前本地只有 UniST 13-shard 的 Phase 1 packed 数据和 Phase 2 mix packed 数据；小模型 Phase 3 先复用当前 Phase 2 mix 数据跑通流程，后续拿到独立 Phase 3 数据后再替换 `TRAIN_DATA`。
- 因此该实验不能作为论文主结果，只能作为 fast validation / ablation。

## 4. 需要新增或修改的代码

优先采用“新增脚本，不改 1.5B 默认脚本”的策略。

### 4.1 下载脚本

在 `scripts/download_hf_assets.sh` 中新增 target：

```bash
qwen0p5b)
  download_model "Qwen/Qwen2.5-0.5B-Instruct" \
    "${REPO_ROOT}/pretrained_models/Qwen2.5-0.5B-Instruct"
  ;;
```

不要改变现有 `qwen` target，避免影响 1.5B 路线。

下载命令：

```bash
USER_ROOT=/opt/dlami/nvme/jasonleeeli \
ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train \
HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
HUGGINGFACE_HUB_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub \
TRANSFORMERS_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/transformers \
TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp \
HF_MAX_WORKERS=1 \
HF_HUB_DISABLE_XET=1 \
scripts/download_hf_assets.sh qwen0p5b
```

验证：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("pretrained_models/Qwen2.5-0.5B-Instruct/config.json")
cfg = json.loads(p.read_text())
expected = {
    "model_type": "qwen2",
    "num_hidden_layers": 24,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}
for k, v in expected.items():
    assert cfg[k] == v, (k, cfg.get(k), v)
assert cfg["tie_word_embeddings"] is True
print("qwen0p5b config OK")
PY
```

### 4.2 初始化 UniSS vocab HF checkpoint

现有 `training/initialize_uniss_hf_checkpoint.py` 已支持 `--base-model` 和 `--output`，无需修改。

命令：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
training/initialize_uniss_hf_checkpoint.py \
  --base-model pretrained_models/Qwen2.5-0.5B-Instruct \
  --uniss-tokenizer pretrained_models/UniSS \
  --output checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --seed 1234 \
  --overwrite
```

验证：

```bash
cat checkpoints/qwen2_0p5b_uniss_vocab_hf/uniss_init_summary.json
```

必须满足：

```json
{
  "base_vocab_size": 151936,
  "target_vocab_size": 180407,
  "added_tokens": 28471,
  "input_embedding_shape": [180407, 896],
  "output_embedding_shape": [180407, 896],
  "tied_word_embeddings": true
}
```

同时确认没有读取作者训练后 UniSS LM 权重：

```bash
rg -n "pretrained_models/UniSS/model.safetensors" logs checkpoints || true
```

### 4.3 转换为 Megatron checkpoint

使用现有转换脚本，但显式传入 0.5B 路径：

```bash
USER_ROOT=/opt/dlami/nvme/jasonleeeli \
ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train \
HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
HUGGINGFACE_HUB_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub \
TRANSFORMERS_CACHE=/opt/dlami/nvme/jasonleeeli/cache/huggingface/transformers \
TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp \
scripts/convert_uniss_checkpoint.sh import \
  --hf-model checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --megatron-path checkpoints/qwen2_0p5b_uniss_vocab \
  --torch-dtype bfloat16 \
  --no-gradient-accumulation-fusion
```

验证：

```bash
cat checkpoints/qwen2_0p5b_uniss_vocab/latest_checkpointed_iteration.txt
find checkpoints/qwen2_0p5b_uniss_vocab -maxdepth 2 -type f | head
```

`latest_checkpointed_iteration.txt` 应为 `0`。

### 4.4 新增 0.5B 训练脚本

新增三个脚本，不修改现有 1.5B 脚本：

```text
scripts/train_phase1_qwen0p5b.sh
scripts/train_phase2_qwen0p5b.sh
scripts/train_phase3_qwen0p5b.sh
```

它们从现有 phase 脚本复制，但替换默认 checkpoint 与模型结构参数：

```bash
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab}"  # phase1
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase1}"
```

Phase 2:

```bash
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase1}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2}"
```

Phase 3:

```bash
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase3}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/data/megatron/phase2_unist13_mix/packed_train.jsonl}"
```

模型结构参数：

```bash
--num-layers 24
--hidden-size 896
--ffn-hidden-size 4864
--num-attention-heads 14
--group-query-attention
--num-query-groups 2
--normalization RMSNorm
--swiglu
--disable-bias-linear
--add-qkv-bias
--position-embedding-type rope
--rotary-base 1000000
--seq-length 18000
--max-position-embeddings 32768
--vocab-size 180407
```

训练超参保持论文/当前 13-shard 计划：

Phase 1:

```bash
--global-batch-size 128
--lr 8e-4
--min-lr 8e-4
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.95
--bf16
```

Phase 2:

```bash
--global-batch-size 128
--lr 2e-4
--min-lr 2e-4
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.95
--bf16
```

Phase 3 保持现有 Phase 3 脚本的论文复现实验设置，只替换 backbone 形状和 checkpoint 路径：

```bash
--global-batch-size 128
--lr 5e-5
--min-lr 5e-6
--lr-warmup-iters 0
--lr-decay-style cosine
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.95
--bf16
```

并保留：

```bash
--sft
--uniss-strict-paper-config
--tokenizer-type NullTokenizer
--use-flash-attn
--no-create-attention-mask-in-dataloader
--no-gradient-accumulation-fusion
--recompute-activations
--no-load-optim
--no-load-rng
--finetune
```

脚本必须支持 `--dry-run`，用于验证命令行。

## 5. 单元测试计划

新增或扩展测试：

```text
training/tests/test_train_scripts_qwen0p5b.py
```

最低测试内容：

1. `scripts/train_phase1_qwen0p5b.sh --dry-run` 输出包含：
   - `--num-layers 24`
   - `--hidden-size 896`
   - `--ffn-hidden-size 4864`
   - `--num-attention-heads 14`
   - `--num-query-groups 2`
   - `--vocab-size 180407`
   - `--seq-length 18000`
   - `--global-batch-size 128`
   - `checkpoints/qwen2_0p5b_uniss_vocab`

2. `scripts/train_phase2_qwen0p5b.sh --dry-run` 默认 load/save 不指向 1.5B。

3. 现有 1.5B 脚本 dry-run 仍保持原参数：
   - `--num-layers 28`
   - `--hidden-size 1536`
   - `--ffn-hidden-size 8960`
   - `--num-attention-heads 12`
   - `--num-query-groups 2`
   - `checkpoints/qwen2_1p5b_uniss_vocab`

执行：

```bash
PYTHONPATH=third_party/Megatron-LM:$PYTHONPATH \
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
-m unittest discover training/tests -v
```

必须 `OK` 后才能启动 smoke training。

## 6. Smoke 训练计划

前置条件：

```bash
tmux list-sessions
nvidia-smi
```

如果 1.5B 主实验仍在 GPU `4,5,6,7` 上运行，不启动。

### 6.1 Phase 1 smoke

```bash
tmux new-session -d -s uniss_qwen0p5b_phase1_smoke \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'PATH=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin:$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase1_unist13/packed_train.jsonl \
   VALID_DATA=data/megatron/validation_unist_dev/phase1_valid_packed.jsonl \
   LOAD_CHECKPOINT=checkpoints/qwen2_0p5b_uniss_vocab \
   SAVE_DIR=checkpoints/uniss_qwen0p5b_phase1_smoke \
   TRAIN_ITERS=10 LR_WARMUP_ITERS=3 \
   SAVE_INTERVAL=10 EVAL_INTERVAL=5 EVAL_ITERS=2 LOG_INTERVAL=1 \
   scripts/train_phase1_qwen0p5b.sh > logs/uniss_qwen0p5b_phase1_smoke.log 2>&1'
```

验证：

```bash
tail -n 120 logs/uniss_qwen0p5b_phase1_smoke.log
cat checkpoints/uniss_qwen0p5b_phase1_smoke/latest_checkpointed_iteration.txt
```

通过标准：

- 日志显示 loaded checkpoint from `checkpoints/qwen2_0p5b_uniss_vocab`。
- 至少完成 `10` iterations。
- `lm loss` 是有限数值。
- `number of skipped iterations: 0`。
- `number of nan iterations: 0`。
- 保存 `iter_0000010`。

### 6.2 Phase 2 smoke

```bash
tmux new-session -d -s uniss_qwen0p5b_phase2_smoke \
  -c /opt/dlami/nvme/jasonleeeli/projects/UniSS \
  'PATH=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin:$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=4,5,6,7 \
   NPROC_PER_NODE=4 TP=1 PP=1 MICRO_BATCH_SIZE=1 \
   TRAIN_DATA=data/megatron/phase2_unist13_mix/packed_train.jsonl \
   VALID_DATA=data/megatron/validation_unist_dev/phase2_valid_packed.jsonl \
   LOAD_CHECKPOINT=checkpoints/uniss_qwen0p5b_phase1_smoke \
   SAVE_DIR=checkpoints/uniss_qwen0p5b_phase2_smoke \
   TRAIN_ITERS=10 LR_WARMUP_ITERS=2 \
   SAVE_INTERVAL=10 EVAL_INTERVAL=5 EVAL_ITERS=2 LOG_INTERVAL=1 \
   scripts/train_phase2_qwen0p5b.sh > logs/uniss_qwen0p5b_phase2_smoke.log 2>&1'
```

通过标准同 Phase 1 smoke，并确认加载的是 `uniss_qwen0p5b_phase1_smoke`，不是 1.5B checkpoint。

## 7. Mini 功能验证计划

Smoke 通过后，可以跑一个更有意义的小规模 functional run：

Phase 1 mini:

```bash
TRAIN_ITERS=200
LR_WARMUP_ITERS=67
SAVE_INTERVAL=50
EVAL_INTERVAL=50
EVAL_ITERS=5
SAVE_DIR=checkpoints/uniss_qwen0p5b_phase1_mini
```

Phase 2 mini:

```bash
TRAIN_ITERS=200
LR_WARMUP_ITERS=10
SAVE_INTERVAL=50
EVAL_INTERVAL=50
EVAL_ITERS=5
LOAD_CHECKPOINT=checkpoints/uniss_qwen0p5b_phase1_mini
SAVE_DIR=checkpoints/uniss_qwen0p5b_phase2_mini
```

通过标准：

- training loss 和 validation loss 不要求绝对值接近 1.5B，但应总体下降或至少保持有限稳定。
- 没有 NaN/skip。
- 每 50 iter checkpoint 能保存。
- Phase 2 能从 Phase 1 mini checkpoint 启动。

## 8. Full 13-shard 小模型验证计划

如果 mini 通过，且 GPU 空闲，可以用当前 13-shard snapshot 跑完整小模型 Phase 1/2/3：

Phase 1:

```bash
TRAIN_ITERS=1269
LR_WARMUP_ITERS=423
SAVE_INTERVAL=100
EVAL_INTERVAL=100
EVAL_ITERS=10
SAVE_DIR=checkpoints/uniss_qwen0p5b_phase1_unist13_full
```

Phase 2:

```bash
TRAIN_ITERS=1045
LR_WARMUP_ITERS=53
SAVE_INTERVAL=100
EVAL_INTERVAL=100
EVAL_ITERS=10
LOAD_CHECKPOINT=checkpoints/uniss_qwen0p5b_phase1_unist13_full
SAVE_DIR=checkpoints/uniss_qwen0p5b_phase2_unist13_full
```

Phase 3:

```bash
TRAIN_ITERS=4341
LR_WARMUP_ITERS=0
SAVE_INTERVAL=100
EVAL_INTERVAL=100
EVAL_ITERS=10
TRAIN_DATA=data/megatron/phase2_unist13_mix/packed_train.jsonl
VALID_DATA=data/megatron/validation_unist_dev/phase2_valid_packed.jsonl
LOAD_CHECKPOINT=checkpoints/uniss_qwen0p5b_phase2_unist13_full
SAVE_DIR=checkpoints/uniss_qwen0p5b_phase3_unist13_full
```

Phase 1/2 iteration count 与当前 1.5B 13-shard 计划一致，便于比较同数据、同 schedule 下的小模型 loss 走势。Phase 3 当前用于端到端流程验证，数据先复用 Phase 2 mix。

## 9. Export 与音频生成验证

小模型 Phase 2 smoke/mini/full checkpoint 完成后，导出到 HF：

```bash
scripts/convert_uniss_checkpoint.sh export \
  --hf-model checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --megatron-path checkpoints/uniss_qwen0p5b_phase2_mini \
  --hf-output checkpoints/exported_hf/qwen0p5b_phase2_mini_hf \
  --no-progress
```

注意：

- `--hf-model` 必须是 `checkpoints/qwen2_0p5b_uniss_vocab_hf`。
- 不允许用 `pretrained_models/UniSS` 作为 LM model path。
- `pretrained_models/UniSS` 只允许作为 tokenizer/speech-tokenizer asset path。

生成固定 validation/test 样本：

```bash
HF_CHECKPOINT=checkpoints/exported_hf/qwen0p5b_phase2_mini_hf \
SPEECH_TOKENIZER=pretrained_models/UniSS \
INPUT='data/raw/UniST/dev-00000.parquet' \
OUTPUT_DIR=runs/uniss_qwen0p5b_phase2_mini_audio_eval \
LIMIT=8 \
SAVE_REFERENCE_AUDIO=1 \
scripts/generate_unist_audio_eval.sh
```

通过标准：

- `manifest.jsonl` 存在。
- 每条样本有 generated text 或 semantic token 审计字段。
- 至少部分样本 `semantic_token_count > 0`。
- 能写出 `.wav` 文件。
- reference audio 也能用同一 BiCodec decoder 解码，证明 speech tokenizer asset 正常。

## 10. 监控指标

训练日志最少记录：

```text
iteration
consumed samples
learning rate
lm loss
grad norm
skipped iterations
nan iterations
validation lm loss
validation PPL
checkpoint save status
GPU memory
```

小模型额外记录：

```text
model_variant=qwen2_0p5b
num_layers=24
hidden_size=896
ffn_hidden_size=4864
num_attention_heads=14
num_query_groups=2
load_checkpoint=...
save_dir=...
```

建议把 Phase 1/2 validation loss 汇总为 CSV：

```text
iteration,phase,model_variant,train_lm_loss,valid_lm_loss,valid_ppl
```

## 11. 风险与处理

### 11.1 Megatron Bridge 不支持 0.5B config

现象：HF -> Megatron import 报 Qwen provider/config mapping 错误。

处理：

1. 先确认 local HF config 字段。
2. 检查 Megatron-Bridge Qwen provider 是否硬编码 1.5B 形状。
3. 如需 patch，只在 conversion wrapper 中为 0.5B 提供显式 config override，不修改 1.5B 路线。
4. Patch 后必须重跑 1.5B dry-run/unit tests，确认主实验不受影响。

### 11.2 TP=1 仍 OOM

0.5B 理论上应明显低于 1.5B 显存。若 OOM：

1. 保持 `seq_length=18000` 不变。
2. 保持 `micro_batch_size=1`。
3. 尝试 `TP=2 PP=1`。
4. 仅在 mechanical smoke 中允许临时降 `seq_length`，并明确标记为非论文形状验证。

### 11.3 小模型 loss 比 1.5B 差

这是预期风险。0.5B 只能验证训练链路和趋势，不能要求达到论文质量。

### 11.4 误加载作者 UniSS 训练后权重

禁止：

```text
AutoModelForCausalLM.from_pretrained("pretrained_models/UniSS")
```

允许：

```text
AutoTokenizer.from_pretrained("pretrained_models/UniSS")
UniSSTokenizer.from_pretrained("pretrained_models/UniSS")
```

检查：

```bash
rg -n 'AutoModelForCausalLM.from_pretrained\\("pretrained_models/UniSS"|model_path = "pretrained_models/UniSS"' .
```

如果后续修改 eval/infer，必须强制 LM checkpoint 由 `--model` 或 `HF_CHECKPOINT` 显式传入。

## 12. 最小执行顺序

严格顺序如下，每一步失败都停止，不进入下一步：

1. 等 1.5B 当前训练释放 GPU。
2. 新增 `qwen0p5b` 下载 target 和 0.5B train scripts。
3. 跑 unit tests，确认 1.5B dry-run 仍保持原参数。
4. 下载 `Qwen2.5-0.5B-Instruct`。
5. 验证 local config。
6. 初始化 `checkpoints/qwen2_0p5b_uniss_vocab_hf`。
7. 验证 `uniss_init_summary.json`。
8. 转换到 `checkpoints/qwen2_0p5b_uniss_vocab`。
9. Phase 1 smoke 10 iter。
10. Phase 2 smoke 10 iter。
11. Export smoke checkpoint 到 HF。
12. Generate 2-8 条 validation audio。
13. Phase 1/2 mini 200 iter。
14. 若 mini 正常，再跑完整 13-shard 小模型 Phase 1/2/3。

## 13. 完成判据

小模型快速验证完成需要满足：

- 1.5B 主线 checkpoint/log/script 未被覆盖。
- `git status` 中没有误加入 `data/`、`checkpoints/`、`pretrained_models/` 大文件。
- 0.5B HF init summary 正确。
- 0.5B Megatron checkpoint iteration 0 可加载。
- Phase 1 smoke 和 Phase 2 smoke 都完成 10 iter。
- smoke/mini 训练无 NaN、无 skipped iteration。
- validation loss 可计算。
- export 到 HF 成功。
- 至少一批 validation 样本能生成非空 semantic tokens。
- 至少一批 `.wav` 能由 BiCodec decoder 写出。

## 14. 建议提交拆分

建议分三次 commit，便于回滚：

1. `Add Qwen2.5-0.5B UniSS training scripts`
   - 下载 target
   - 0.5B train scripts
   - dry-run tests

2. `Initialize Qwen2.5-0.5B UniSS checkpoint`
   - 只提交代码/文档，不提交 checkpoint 权重
   - checkpoint 保持本地

3. `Document Qwen0.5B validation results`
   - 训练结果摘要
   - loss 表格
   - audio eval manifest 摘要

不要 commit：

```text
data/
checkpoints/
pretrained_models/
logs/
runs/
```
