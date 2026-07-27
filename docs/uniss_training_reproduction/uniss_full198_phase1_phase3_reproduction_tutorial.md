# UniSS UniST-198（full198）Phase1–Phase3 完整复现教程

> 适用范围：当前仓库的公开数据可复现路线，使用 `cmots/UniST` 的全部 198 个 train parquet shard、Qwen2.5-0.5B-Instruct、Megatron-LM 和 8 张 GPU。
>
> 本教程对应当前实际稳定完成并用于评估的 checkpoint 链路：Phase1 recovery B1 v2 → Phase2 from-Phase1 v4 → Phase3 after-Phase2 v4。
> 最后核对日期：2026-07-27。

## 1. 先理解复现边界

这条路线是“UniST-198 full-data public reproduction”，不是论文严格的完整训练语料复现。

当前实现与论文目标一致的部分：

- 使用 UniSS 的任务 token、speech token 和 prompt/target 格式；
- Phase1 包含 ASR、S2TT、TTS 和 MT 四类目标；
- Phase2 包含 Quality、Performance、Direct S2ST，并按 `Phase2 : Phase1 replay = 2 : 1` 混合；
- Phase3 只保留 Quality 和 Performance；
- 使用 Megatron-LM、BF16、FlashAttention、18,000 token sequence packing；
- Phase2/Phase3 使用全局 shuffle 和完整 validation；
- 训练结束后可以导出 Hugging Face checkpoint，生成语音并计算 Text-BLEU、Speech-BLEU、SLC、UTMOS、AutoPCP。

与论文严格复现不同的部分：

- 本路线使用 Qwen2.5-0.5B，而论文主模型是 1.5B；
- Phase1 的 MT 使用 UniST transcription→translation 作为 MT proxy，而不是完整 WMT17 2.3B tokens；
- Phase1 speech data 也来自公开 UniST，而不是论文约 77.1k 小时的完整 speech alignment 组合；
- Phase3 使用全部 UniST train 的 Quality/Performance 样本，没有额外重建论文未公开的 high-quality 筛选；
- 当前 dev/test 是 UniST dev/test，不能直接和论文 CVSS-T Table 1 排名。

因此实验名称应保留 `unist198`、`full198` 或 `public reproduction`，不要把它描述成论文原始 1.5B 模型的严格复现。

## 2. 最终要得到什么

完整流程如下：

```text
UniST 198 train parquet + dev/test parquet
  ├─ Phase1: ASR + S2TT + TTS + MT proxy
  ├─ Phase2: Quality + Performance + Direct S2ST
  └─ Phase3: Quality + Performance
          ↓
Phase2 与 Phase1 replay 按 2:1 混合
          ↓
固定长度 18,000 token Megatron packed JSONL
          ↓
Phase1：基础训练到有效 iteration 3300，再低 LR + shuffle 完成剩余 15465 steps
          ↓
Phase2 v4：从干净 Phase1 权重重新开始，15381 steps
          ↓
Phase3 v4：从 Phase2 最终权重重新开始，9075 steps
          ↓
Megatron → Hugging Face 导出
          ↓
UniST dev/test：Q/P 推理、音频解码、客观指标、聚合报告
```

当前成功结果的关键路径：

| 阶段 | 最终 checkpoint | 本阶段 iteration |
| --- | --- | ---: |
| Phase1 | `checkpoints/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2/iter_0015465` | 15465 |
| Phase2 | `checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/iter_0015381` | 15381 |
| Phase3 | `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075` | 9075 |

Phase1 recovery 的 `15465` 是从原始 Phase1 iteration 3300 之后重新计数的本地 step。有效训练预算为：

```text
3300 + 15465 = 18765 steps
```

## 3. 硬件、空间和软件要求

推荐配置：

- 8 张 NVIDIA H200；
- 单机 8 进程，`TP=1`、`PP=1`；
- 每卡 micro batch 2；
- global batch 128；
- 训练 sequence length 18000；
- 数据处理阶段建议至少 2 TB 可用空间；
- checkpoint、评估音频和指标还需要额外空间。

当前数据规模参考：

| 目录 | 大小参考 |
| --- | ---: |
| `data/raw/UniST` | 约 29 GB |
| `data/processed/phase1_unist198_sharded` | 约 115 GB |
| `data/processed/phase2_unist198_sharded` | 约 220 GB |
| `data/processed/phase3_unist198_sharded` | 约 151 GB |
| `data/processed/phase2_unist198_mix` | 约 261 GB |
| `data/megatron/phase1_unist198` | 约 335 GB |
| `data/megatron/phase2_unist198_mix` | 约 750 GB |
| `data/megatron/phase3_unist198` | 约 385 GB |

本机已验证的核心版本：

```text
Python 3.12.13
PyTorch 2.6.0+cu124
CUDA runtime 12.4
Transformers 4.53.1
PyArrow 25.0.0
Transformer Engine 2.6.0.post1
```

Transformer Engine 会提示当前 FlashAttention `2.8.3.post1` 高于其声明的上限 `2.8.1`，但本机训练链路已经实际跑通。换机器时仍建议优先复现当前环境，而不是随意升级依赖。

## 4. 设置路径并激活训练环境

后续命令默认从仓库根目录执行：

```bash
export USER_ROOT=/opt/dlami/nvme/jasonleeeli
export REPO_ROOT=${USER_ROOT}/projects/UniSS
cd "${REPO_ROOT}"
```

激活当前恢复好的环境：

```bash
source /opt/dlami/nvme/jasonleeeli/env_recovery/uniss-train-20260721/activate_uniss.sh
```

确认 Python、GPU、CUDA 库和 Transformer Engine：

```bash
python - <<'PY'
import ctypes
import platform
import torch
import transformers
import pyarrow

ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("pyarrow", pyarrow.__version__)
print("gpu count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
assert torch.cuda.device_count() == 8
PY
```

如果 `python` 不存在，通常是没有执行 activation；不要直接使用系统 Python。

## 5. 检查仓库和关键脚本

```bash
git status --short --branch
git rev-parse HEAD

for path in \
  training/prepare_phase1_alignment.py \
  training/prepare_unist_s2st.py \
  training/mix_sample_jsonl.py \
  training/pack_sequences.py \
  training/pack_sequences_parallel.py \
  training/validate_packed_jsonl.py \
  scripts/pack_unist198_full.sh \
  scripts/run_qwen0p5b_unist198_phase1_recovery_b.sh \
  scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh \
  scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh; do
  test -f "${path}" || { echo "missing ${path}"; exit 1; }
done
```

建议先执行仓库测试：

```bash
python -m unittest discover training/tests -v
python -m unittest discover evaluation/tests -v
```

## 6. 准备 tokenizer 和 Qwen-0.5B 初始化 checkpoint

### 6.1 必需资产

必须存在：

```text
pretrained_models/UniSS/
pretrained_models/Qwen2.5-0.5B-Instruct/
```

检查：

```bash
test -f pretrained_models/UniSS/tokenizer.json
test -f pretrained_models/UniSS/config.json
test -f pretrained_models/Qwen2.5-0.5B-Instruct/config.json
```

如果 Qwen-0.5B 尚未下载：

```bash
USER_ROOT=${USER_ROOT} \
ENV_ROOT=${USER_ROOT}/conda_envs/uniss-train \
HF_HOME=${USER_ROOT}/cache/huggingface \
HUGGINGFACE_HUB_CACHE=${USER_ROOT}/cache/huggingface/hub \
TRANSFORMERS_CACHE=${USER_ROOT}/cache/huggingface/transformers \
TMPDIR=${USER_ROOT}/tmp \
HF_MAX_WORKERS=1 \
HF_HUB_DISABLE_XET=1 \
scripts/download_hf_assets.sh qwen0p5b
```

UniSS tokenizer/codec 资产如果缺失，可以使用：

```bash
scripts/download_hf_assets.sh uniss
```

### 6.2 初始化扩展到 UniSS vocab 的 HF checkpoint

```bash
python training/initialize_uniss_hf_checkpoint.py \
  --base-model pretrained_models/Qwen2.5-0.5B-Instruct \
  --uniss-tokenizer pretrained_models/UniSS \
  --output checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --seed 1234
```

如果目标目录已存在，脚本默认拒绝覆盖。确认确定要重建时才加 `--overwrite`。

验证：

```bash
cat checkpoints/qwen2_0p5b_uniss_vocab_hf/uniss_init_summary.json
```

关键值应为：

```text
base_vocab_size=151936
target_vocab_size=180407
added_tokens=28471
embedding hidden size=896
tied_word_embeddings=true
```

### 6.3 转成 Megatron checkpoint

```bash
scripts/convert_uniss_checkpoint.sh import \
  --hf-model checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --megatron-path checkpoints/qwen2_0p5b_uniss_vocab \
  --torch-dtype bfloat16 \
  --no-gradient-accumulation-fusion
```

验证：

```bash
cat checkpoints/qwen2_0p5b_uniss_vocab/latest_checkpointed_iteration.txt
find checkpoints/qwen2_0p5b_uniss_vocab/iter_0000000 -maxdepth 1 -type f -printf '%f\n' | sort
```

tracker 应为 `0`。

## 7. 放置并验证 UniST 数据包

假设下载后的数据已经解压到：

```text
data/raw/UniST/
```

至少需要：

```text
data/raw/UniST/train-00000.parquet ... train-00197.parquet
data/raw/UniST/dev-00000.parquet
data/raw/UniST/test-00000.parquet
```

检查 shard 数、行数和字段：

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path("data/raw/UniST")
train = sorted(root.glob("train-*.parquet"))
assert len(train) == 198, len(train)
rows = sum(pq.ParquetFile(path).metadata.num_rows for path in train)
print("train shards", len(train))
print("train rows", rows)
assert rows == 19_785_924, rows

required = {
    "id", "transcription", "translation", "source_glm",
    "source_bicodec", "target_bicodec", "bicodec_global",
    "src_lang", "tgt_lang",
}
columns = set(pq.ParquetFile(train[0]).schema_arrow.names)
missing = required - columns
assert not missing, missing

for split, expected in [("dev-00000.parquet", 7965), ("test-00000.parquet", 23369)]:
    path = root / split
    assert path.is_file(), path
    actual = pq.ParquetFile(path).metadata.num_rows
    print(split, actual)
    assert actual == expected, (split, actual)
PY
```

如果数据包版本不同，不要直接修改教程中的 expected count 继续训练。应先确认数据版本、记录新计数，并使用新的实验名称和 checkpoint 目录。

## 8. 从 198 个 parquet 生成 Phase1–Phase3 task JSONL

### 8.1 任务构成

每条 train row 生成：

| 阶段 | 每条 row 生成的任务 | 样本数/row |
| --- | --- | ---: |
| Phase1 | `asr`, `s2tt`, `tts`, `mt` | 4 |
| Phase2 | `quality`, `performance`, `direct_s2st` | 3 |
| Phase3 | `quality`, `performance` | 2 |

预期总数：

```text
raw train rows: 19,785,924
Phase1 samples: 79,143,696
Phase2 samples: 59,357,772
Phase3 samples: 39,571,848
```

Phase1 的 `mt` 是 UniST transcription→translation 的 MT proxy。

### 8.2 创建输出目录

```bash
mkdir -p \
  data/processed/phase1_unist198_sharded \
  data/processed/phase2_unist198_sharded \
  data/processed/phase3_unist198_sharded \
  data/processed/phase2_unist198_mix \
  runs/unist198_preprocess/logs
```

### 8.3 并行处理 198 个 shard

下面命令每个 worker 依次构造同一个 shard 的 Phase1/2/3，最多并行 8 个 shard。输出先写临时文件，行数正确后才原子发布：

```bash
export PYTHON=${USER_ROOT}/conda_envs/uniss-train/bin/python
export TOKENIZER=${REPO_ROOT}/pretrained_models/UniSS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

seq 0 197 | xargs -P 8 -I '{}' bash -c '
set -euo pipefail
index=$(printf "%05d" "$1")
raw="data/raw/UniST/train-${index}.parquet"
p1="data/processed/phase1_unist198_sharded/train-${index}.jsonl"
p2="data/processed/phase2_unist198_sharded/train-${index}.jsonl"
p3="data/processed/phase3_unist198_sharded/train-${index}.jsonl"
rows=$(${PYTHON} -c "import pyarrow.parquet as pq,sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" "${raw}")

build_and_check() {
  output=$1
  expected=$2
  shift 2
  if [[ -s "${output}" && $(wc -l < "${output}") -eq ${expected} ]]; then
    echo "[${index}] reuse ${output}"
    return
  fi
  tmp="${output}.tmp.${BASHPID}"
  rm -f "${tmp}"
  "$@" --output "${tmp}"
  actual=$(wc -l < "${tmp}")
  [[ ${actual} -eq ${expected} ]] || {
    echo "count mismatch ${tmp}: expected=${expected} actual=${actual}" >&2
    exit 1
  }
  mv "${tmp}" "${output}"
}

build_and_check "${p1}" $((rows * 4)) \
  "${PYTHON}" training/prepare_phase1_alignment.py \
  --input "${raw}" --tokenizer "${TOKENIZER}" \
  --tasks asr s2tt tts --include-mt-proxy

build_and_check "${p2}" $((rows * 3)) \
  "${PYTHON}" training/prepare_unist_s2st.py \
  --input "${raw}" --phase phase2 --tokenizer "${TOKENIZER}"

build_and_check "${p3}" $((rows * 2)) \
  "${PYTHON}" training/prepare_unist_s2st.py \
  --input "${raw}" --phase phase3 --tokenizer "${TOKENIZER}"

echo "[${index}] complete"
' _ '{}'
```

检查文件和总行数：

```bash
for phase in phase1 phase2 phase3; do
  find "data/processed/${phase}_unist198_sharded" -maxdepth 1 \
    -type f -name 'train-*.jsonl' | wc -l
done

wc -l data/processed/phase1_unist198_sharded/train-*.jsonl | tail -n 1
wc -l data/processed/phase2_unist198_sharded/train-*.jsonl | tail -n 1
wc -l data/processed/phase3_unist198_sharded/train-*.jsonl | tail -n 1
```

总数应分别为 `79143696`、`59357772`、`39571848`。

## 9. 生成 Phase2 的 2:1 replay mix

这里的 `2:1` 是：

```text
Phase2 S2ST task samples : Phase1 replay task samples = 2 : 1
```

不是把原始 UniST parquet 简单复制三份。Phase2 组本身每条 raw row 有三个 task，Phase1 replay 从 Phase1 的四类 task 流中按顺序取样。

```bash
phase1_files=(data/processed/phase1_unist198_sharded/train-*.jsonl)
phase2_files=(data/processed/phase2_unist198_sharded/train-*.jsonl)
phase1_csv=$(IFS=,; echo "${phase1_files[*]}")
phase2_csv=$(IFS=,; echo "${phase2_files[*]}")

python training/mix_sample_jsonl.py \
  --group "unist=2:${phase2_csv}" \
  --group "phase1=1:${phase1_csv}" \
  --max-records 89036658 \
  --output data/processed/phase2_unist198_mix/phase2_mix_2to1.jsonl
```

验证：

```bash
wc -l data/processed/phase2_unist198_mix/phase2_mix_2to1.jsonl

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("data/processed/phase2_unist198_mix/phase2_mix_2to1.jsonl")
counts = Counter()
with path.open() as handle:
    for line in handle:
        row = json.loads(line)
        counts[row["mix_group"]] += 1
print(counts)
assert counts["unist"] == 59_357_772
assert counts["phase1"] == 29_678_886
assert counts["unist"] == 2 * counts["phase1"]
PY
```

## 10. 准备 Phase1、Phase2、Phase3 validation

validation 使用同一个 `dev-00000.parquet`，但每个 phase 的任务组成必须与训练阶段一致。

### 10.1 生成未 packed validation JSONL

```bash
mkdir -p data/processed/validation_unist_dev data/megatron/validation_unist_dev

python training/prepare_phase1_alignment.py \
  --input data/raw/UniST/dev-00000.parquet \
  --tokenizer pretrained_models/UniSS \
  --tasks asr s2tt tts \
  --include-mt-proxy \
  --output data/processed/validation_unist_dev/phase1_dev.jsonl

python training/prepare_unist_s2st.py \
  --input data/raw/UniST/dev-00000.parquet \
  --phase phase2 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/validation_unist_dev/phase2_dev.jsonl

python training/mix_sample_jsonl.py \
  --group unist=2:data/processed/validation_unist_dev/phase2_dev.jsonl \
  --group phase1=1:data/processed/validation_unist_dev/phase1_dev.jsonl \
  --output data/processed/validation_unist_dev/phase2_dev_mix_2to1.jsonl

python training/prepare_unist_s2st.py \
  --input data/raw/UniST/dev-00000.parquet \
  --phase phase3 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/validation_unist_dev/phase3_dev.jsonl
```

预期未 packed 样本数：

```text
Phase1 dev: 31,860
Phase2 dev before replay: 23,895
Phase2 dev 2:1 mix: 35,842
Phase3 dev: 15,930
```

### 10.2 pack validation

```bash
python training/pack_sequences.py \
  --input data/processed/validation_unist_dev/phase1_dev.jsonl \
  --output data/megatron/validation_unist_dev/phase1_valid_packed.jsonl \
  --seq-length 18000 --drop-overlong

python training/pack_sequences.py \
  --input data/processed/validation_unist_dev/phase2_dev_mix_2to1.jsonl \
  --output data/megatron/validation_unist_dev/phase2_valid_packed.jsonl \
  --seq-length 18000 --drop-overlong

python training/pack_sequences.py \
  --input data/processed/validation_unist_dev/phase3_dev.jsonl \
  --output data/megatron/validation_unist_dev/phase3_valid_packed.jsonl \
  --seq-length 18000 --drop-overlong
```

验证：

```bash
for phase in phase1 phase2 phase3; do
  python training/validate_packed_jsonl.py \
    --input "data/megatron/validation_unist_dev/${phase}_valid_packed.jsonl" \
    --seq-length 18000
done
```

当前参考 packed count 是：

```text
Phase1 validation: 241
Phase2 validation: 571
Phase3 validation: 331
```

## 11. pack 全量训练数据到 18,000 tokens

### 11.1 为什么分两步

当前历史 Phase1 packed 文件由单进程 packer 生成，count 为 `800632`。Phase2/Phase3 使用 16 worker 并行 packer，分别为 `1968716` 和 `1161587`。并行 packer 在 worker byte-range 边界会新开 packed sequence，最多增加 `workers-1` 条 padding record，因此要复现当前 count，应保持相同策略。

### 11.2 Phase1 单进程 packing

```bash
mkdir -p data/megatron/phase1_unist198 runs/unist198_preprocess
phase1_inputs=(data/processed/phase1_unist198_sharded/train-*.jsonl)
phase1_tmp=data/megatron/phase1_unist198/packed_train.jsonl.tmp.$$

python training/pack_sequences.py \
  --input "${phase1_inputs[@]}" \
  --output "${phase1_tmp}" \
  --seq-length 18000 \
  --drop-overlong

python training/validate_packed_jsonl.py \
  --input "${phase1_tmp}" --seq-length 18000

phase1_count=$(wc -l < "${phase1_tmp}")
[[ ${phase1_count} -eq 800632 ]]
mv "${phase1_tmp}" data/megatron/phase1_unist198/packed_train.jsonl
printf '%s\n' "${phase1_count}" \
  > data/megatron/phase1_unist198/packed_train.jsonl.count
```

### 11.3 Phase2/Phase3 并行 packing

Phase1 正式文件和 `.count` 已存在后，统一 runner 会跳过 Phase1，使用 16 workers pack Phase2/Phase3，并重新验证所有产物：

```bash
PACK_WORKERS=16 \
bash scripts/pack_unist198_full.sh \
  --config configs/experiments/uniss_qwen0p5b_unist198_full_v1.env \
  --start-phase phase1
```

成功后应出现：

```text
runs/unist198_preprocess/PACKING_COMPLETE_V1
```

核对 count：

```bash
for path in \
  data/megatron/phase1_unist198/packed_train.jsonl \
  data/megatron/phase2_unist198_mix/packed_train.jsonl \
  data/megatron/phase3_unist198/packed_train.jsonl; do
  printf '%s: ' "${path}"
  cat "${path}.count"
done
```

当前精确参考：

```text
Phase1: 800632
Phase2: 1968716
Phase3: 1161587
```

任何 count 不一致都要先解释 packer worker 数、输入 shard 顺序或数据版本差异，不要直接绕过 Phase2/Phase3 runner 的 count 检查。

## 12. 三阶段共同训练配置

| 参数 | 值 |
| --- | ---: |
| backbone | Qwen2.5-0.5B-Instruct + UniSS vocab |
| vocab size | 180407（Megatron padded embedding 180480） |
| layers | 24 |
| hidden size | 896 |
| FFN size | 4864 |
| attention heads | 14 |
| query groups | 2 |
| sequence length | 18000 |
| max position | 32768 |
| BF16 | 开启 |
| FlashAttention | 开启 |
| recompute activations | 开启 |
| GPUs | 8 |
| TP / PP | 1 / 1 |
| micro batch | 2 |
| global batch | 128 |
| weight decay | 0.1 |
| Adam betas | 0.9 / 0.95 |

新 phase 的正确加载语义：

```text
FINETUNE=1
LOAD_OPTIM=0
LOAD_RNG=0
```

也就是只加载上一阶段模型权重，重置 optimizer、RNG、scheduler 和本阶段 iteration。

同一个 phase 断点续训才使用：

```text
FINETUNE=0
LOAD_OPTIM=1
LOAD_RNG=1
LOAD_CHECKPOINT=<本阶段 SAVE_DIR>
```

不要在 Phase1→Phase2 或 Phase2→Phase3 时加载上阶段 optimizer。

## 13. Phase1 完整训练

### 13.1 为什么当前稳定链路包含 recovery

最早的 Phase1 full198 使用：

```text
LR=8e-4
constant LR
3 epochs = 18765 steps
```

该实验在后段出现明显 loss/grad 爆炸。经过日志分析，iteration 3300 被保留为稳定候选；随后采用：

- 从 iteration 3300 只加载模型权重；
- 重置 optimizer/RNG；
- 改用 `dataloader-type=cyclic` 做 shuffle；
- LR 改为 `1e-4 → 1e-5 cosine`；
- warmup 200；
- 前 500 step 完成稳定性验证；
- 再保持 optimizer/RNG/data cursor 继续到本地 iteration 15465。

因此当前 Phase2 的正确来源不是原始 Phase1 最后一个异常 checkpoint，而是 recovery B1 v2 的 `iter_0015465`。

### 13.2 先生成原始 Phase1 iteration 3300 checkpoint

用独立 tmux 启动，仅跑到 3300：

```bash
tmux new-session -d -s uniss_phase1_prefix_3300 \
  "cd ${REPO_ROOT} && \
   source ${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh && \
   PHASE1_TRAIN_ITERS=3300 \
   PHASE1_LR_WARMUP_ITERS=6255 \
   bash scripts/run_qwen0p5b_unist198_all_phases.sh \
     --config configs/experiments/uniss_qwen0p5b_unist198_full_v1.env \
     --start-phase phase1 --end-phase phase1"
```

说明：

- `6255 = ceil(800632 / 128)`，即一个 packed-data epoch；
- 原始计划总预算 `3 × 6255 = 18765`；
- 这里只保存 iteration 3300 候选，不继续使用高 LR 跑完整 18765。

查看进度：

```bash
tmux attach -t uniss_phase1_prefix_3300
tail -f logs/uniss_qwen0p5b_phase1_unist198_full_v1.log
cat checkpoints/uniss_qwen0p5b_phase1_unist198_full_v1/latest_checkpointed_iteration.txt
```

必须得到：

```text
checkpoints/uniss_qwen0p5b_phase1_unist198_full_v1/iter_0003300
```

### 13.3 从 iteration 3300 执行稳定 recovery

历史运行先做了 500-step stability pilot，再从同一 recovery checkpoint 保留 optimizer/RNG/data cursor 续训到 15465。对于全新复现，可以直接把 recovery target 设为 15465；训练数据、seed、shuffle 和学习率日程相同，同时避免人工拆分：

```bash
tmux new-session -d -s uniss_phase1_recovery_b1_v2 \
  "cd ${REPO_ROOT} && \
   source ${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh && \
   TRAIN_ITERS=15465 \
   bash scripts/run_qwen0p5b_unist198_phase1_recovery_b.sh \
     --config configs/experiments/uniss_qwen0p5b_unist198_phase1_recovery_b1_v2.env"
```

该 runner 会：

1. 确认原始 Phase1 iteration 3300 有 8 个 distributed checkpoint shard；
2. 在 `checkpoints/candidates/` 创建 hard-link candidate，不修改原始 checkpoint；
3. `FINETUNE=1`，只加载模型权重；
4. 使用 8 GPU、micro batch 2、global batch 128；
5. 使用 cyclic shuffle；
6. 使用 `LR=1e-4`、`min LR=1e-5`、warmup 200、cosine decay 500；
7. 保存到隔离目录 `uniss_qwen0p5b_phase1_unist198_recovery_b1_v2`。

如果需要严格保留历史的 500-step 人工检查边界，先不设置 `TRAIN_ITERS` 运行同一 runner；默认会在 500 结束。确认 validation、grad norm、NaN/skipped 都正常后，再以 `LOAD_CHECKPOINT=<recovery SAVE_DIR>`、`FINETUNE=0`、`LOAD_OPTIM=1`、`LOAD_RNG=1` 和 `TRAIN_ITERS=15465` 调用 `scripts/train_phase1_qwen0p5b.sh`。不要在 500 处重新初始化 optimizer，否则与历史 continuation 不同。

查看：

```bash
tail -f logs/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2.log
cat checkpoints/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2/latest_checkpointed_iteration.txt
```

最终必须是 `15465`，并确认最后 validation 有限、`nan iterations=0`、`skipped iterations=0`。

当前历史 final validation 参考：

```text
iteration 15465 validation loss ≈ 5.309781
PPL ≈ 202.3059
```

### 13.4 Phase1 TensorBoard

```bash
tmux new-session -d -s uniss_phase1_tensorboard \
  "source ${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh && \
   tensorboard \
     --logdir ${REPO_ROOT}/runs/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2/tensorboard \
     --host 0.0.0.0 --port 6007"
```

浏览器：

```text
http://<服务器IP>:6007
```

SSH 转发：

```bash
ssh -L 6007:localhost:6007 root@<服务器IP>
```

## 14. Phase2 v4 完整训练

### 14.1 配置摘要

Phase2 使用：

```text
train data: phase2_unist198_mix/packed_train.jsonl
train packed records: 1,968,716
source: Phase1 recovery iter_0015465
train iterations: 15,381（约一个 packed-data epoch）
LR: 1e-5 → 1e-6 cosine
LR warmup: 400
LR decay iters: 3000
clip grad: 0.5
cyclic shuffle: enabled
--no-data-sharding: enabled
full validation: enabled
eval interval: 100
pilot inspection boundary: 3000
```

`3000` 不是 early stop。runner 会在 iteration 3000 正常退出一次、执行健康 gate，然后从同一 Phase2 checkpoint 加载 optimizer/RNG/data cursor，继续到 15381。

### 14.2 先 dry-run

```bash
bash scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh \
  --config configs/experiments/uniss_qwen0p5b_unist198_phase2_from_phase1_v4.env \
  --dry-run
```

重点确认输出中包含：

```text
source iteration=15465
train target=15381
DATALOADER_TYPE=cyclic
--no-data-sharding
--full-validation
FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0
```

### 14.3 后台启动 Phase2

```bash
tmux new-session -d -s uniss_phase2_from_phase1_v4 \
  "cd ${REPO_ROOT} && \
   bash scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh \
     --config configs/experiments/uniss_qwen0p5b_unist198_phase2_from_phase1_v4.env"
```

监控：

```bash
tmux attach -t uniss_phase2_from_phase1_v4
tail -f logs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4.log
watch -n 2 nvidia-smi
```

完成条件：

```bash
cat checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/latest_checkpointed_iteration.txt
cat runs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/PILOT_GATE.json
cat runs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/FINAL_GATE.json
test -f runs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/TRAINING_COMPLETE
```

tracker 应为 `15381`，两个 gate 的 `status` 都应为 `pass`。

当前 final validation 参考：

```text
best/last validation loss ≈ 4.288134
```

### 14.4 Phase2 TensorBoard

```bash
tmux new-session -d -s uniss_phase2_v4_tensorboard \
  "source ${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh && \
   tensorboard \
     --logdir ${REPO_ROOT}/runs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/tensorboard \
     --host 0.0.0.0 --port 6014"
```

地址：

```text
http://<服务器IP>:6014
```

## 15. Phase3 v4 完整训练

### 15.1 配置摘要

Phase3 使用：

```text
train data: phase3_unist198/packed_train.jsonl
train packed records: 1,161,587
source: Phase2 v4 iter_0015381
train iterations: 9075
LR: 1e-5 → 1e-6 cosine
warmup: 200
clip grad: 0.5
cyclic shuffle: enabled
--no-data-sharding: enabled
full validation: enabled
eval interval: 100
```

### 15.2 推荐自动衔接方式

可以在 Phase2 训练时就启动 Phase3 waiter。它不会抢 GPU，会等待 Phase2 final checkpoint、completion marker 和 final health gate：

```bash
tmux new-session -d -s uniss_phase3_after_phase2_v4 \
  "cd ${REPO_ROOT} && \
   bash scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh \
     --config configs/experiments/uniss_qwen0p5b_unist198_phase3_after_phase2_v4.env"
```

如果 Phase2 已经完成，也可以直接执行同一命令。

监控：

```bash
tmux attach -t uniss_phase3_after_phase2_v4
tail -f logs/uniss_qwen0p5b_phase3_unist198_after_phase2_v4.log
cat runs/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/wait_and_train.log
```

完成条件：

```bash
cat checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/latest_checkpointed_iteration.txt
test -f runs/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/TRAINING_COMPLETE
```

tracker 应为 `9075`。

### 15.3 Phase3 TensorBoard

```bash
tmux new-session -d -s uniss_phase3_v4_tensorboard \
  "source ${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh && \
   tensorboard \
     --logdir ${REPO_ROOT}/runs/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/tensorboard \
     --host 0.0.0.0 --port 6015"
```

地址：

```text
http://<服务器IP>:6015
```

## 16. 训练状态和异常检查

通用检查：

```bash
rg 'iteration +[0-9]+/|validation loss at iteration|number of skipped iterations|number of nan iterations' \
  logs/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4.log | tail -n 50

rg 'iteration +[0-9]+/|validation loss at iteration|number of skipped iterations|number of nan iterations' \
  logs/uniss_qwen0p5b_phase3_unist198_after_phase2_v4.log | tail -n 50
```

正常标准：

- loss 有短时波动，但整体不能持续爆炸；
- validation 不应连续显著回退；
- grad norm 偶发尖峰可以审计，但不能连续出现；
- `number of nan iterations` 必须为 0；
- `number of skipped iterations` 必须为 0；
- checkpoint tracker 不得超过预期 target；
- Phase2/Phase3 的 TensorBoard 必须同时有 `lm loss` 和 validation loss。

Phase2/Phase3 每 100 steps 做一次 validation；到达 final iteration 还会做最终 validation。

## 17. 断点续训规则

### 17.1 同一阶段中断

不要重新建立新的 source candidate。原 runner 会检测本阶段 `latest_checkpointed_iteration.txt` 并从本阶段 `SAVE_DIR` 恢复：

```text
FINETUNE=0
LOAD_OPTIM=1
LOAD_RNG=1
```

重新执行同一个 runner 即可。

### 17.2 新阶段开始

Phase1→Phase2 或 Phase2→Phase3 必须：

```text
FINETUNE=1
LOAD_OPTIM=0
LOAD_RNG=0
```

否则会把上阶段 optimizer/scheduler 带入新任务，破坏阶段学习率和 iteration 语义。

### 17.3 不覆盖旧结果

所有 runner 都使用隔离目录。若要做新实验，至少覆盖实验名和 save/run/log 路径，例如：

```bash
EXPERIMENT_NAME=uniss_qwen0p5b_phase2_unist198_ablation_v1 \
SAVE_DIR=${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2_unist198_ablation_v1 \
RUN_DIR=${REPO_ROOT}/runs/uniss_qwen0p5b_phase2_unist198_ablation_v1 \
LOG_PATH=${REPO_ROOT}/logs/uniss_qwen0p5b_phase2_unist198_ablation_v1.log \
bash scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh
```

不要删除或复用本教程列出的成功 checkpoint 目录。

## 18. 完整测试与评估环境

训练环境与评估环境分开：

```text
training: /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train
evaluation: /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval
```

创建/修复评估环境：

```bash
experiments/evaluation/uniss_full198_phase2_phase3/setup_eval_environment.sh
experiments/evaluation/uniss_full198_phase2_phase3/prepare_metric_models.sh
```

这会准备：

- vLLM 0.8.5.post1；
- FunASR/ModelScope ASR；
- sacreBLEU；
- UTMOS；
- AutoPCP/stopes；
- 所需缓存和 metric model。

## 19. 准备 dev/test manifest

```bash
experiments/evaluation/uniss_full198_phase2_phase3/prepare_manifests.sh
```

生成：

```text
experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_dev_all.jsonl
experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_test_all.jsonl
experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_dev_smoke_3.jsonl
experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_dev_listen_50.jsonl
```

其中：

- smoke：快速验证格式、生成和音频解码；
- listen：少量音频供人工试听；
- dev/test all：完整指标评估。

## 20. 导出 Phase2/Phase3 为 Hugging Face checkpoint

导出过程需要至少一张可见 GPU 用于 Megatron/Triton import，但权重转换本身主要在 CPU 完成。

```bash
CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh phase2

CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh phase3
```

输出：

```text
checkpoints/exported_hf/qwen0p5b_phase2_unist198_iter_0015381_hf
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

脚本会验证：

```text
logical tokenizer size = 180407
HF/Megatron padded model vocab size = 180480
source checkpoint iteration 正确
```

已有导出目录时脚本拒绝覆盖。需要重导出时使用新的 `HF_OUTPUT` 路径，不要删除已审计结果。

## 21. 先做 smoke 和试听测试

HF smoke：

```bash
EVAL_CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/run_hf_matrix.sh smoke
```

50 条试听：

```bash
EVAL_CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/run_hf_matrix.sh listen
```

输出目录会包含：

```text
source_audio/
reference_audio/
generated_audio/
results.jsonl
summary.json
metrics/
```

在 full dev/test 之前，必须确认：

- Quality 和 Performance 两种 mode 都有生成；
- 生成音频可打开；
- 没有 dummy padded vocab token；
- semantic token 非空；
- Text-BLEU/SLC smoke 能成功计算。

## 22. 一键执行完整 dev/test 评估

先做只读 preflight：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_ONLY=1 RUN_ID=${RUN_ID} \
experiments/evaluation/uniss_full198_phase2_phase3/run_complete_evaluation.sh
```

完整后台评估：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)

tmux new-session -d -s uniss_full198_evaluation \
  "cd ${REPO_ROOT} && \
   RUN_ID=${RUN_ID} \
   DEV_PHASE2_GPUS=0,1,2,3 \
   DEV_PHASE3_GPUS=4,5,6,7 \
   TEST_PHASE2_GPUS=0,1,2,3 \
   TEST_PHASE3_GPUS=4,5,6,7 \
   ASR_BATCH_SIZE=32 \
   experiments/evaluation/uniss_full198_phase2_phase3/run_complete_evaluation.sh"

echo "RUN_ID=${RUN_ID}"
```

默认行为：

1. 等待训练 GPU 空闲；
2. Phase2/Phase3 HF smoke；
3. smoke objective metrics；
4. vLLM smoke + BiCodec decode；
5. 试听集；
6. Phase2 dev 和 Phase3 dev 并行；
7. 各自 dev 完成后自动进入对应 test；
8. Phase2 使用 GPU 0–3，Phase3 使用 GPU 4–7；
9. 计算 Text-BLEU、Speech-BLEU、SLC、UTMOS、AutoPCP；
10. 生成聚合 Markdown/JSON 报告。

监控：

```bash
tmux attach -t uniss_full198_evaluation

tail -f "eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}/pipeline.log"
cat "eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}/status.txt"
cat "eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}/phase2_status.txt"
cat "eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}/phase3_status.txt"
```

完成标记：

```bash
test -f "eval_outputs/uniss_full198_phase2_phase3_${RUN_ID}/COMPLETE"
```

## 23. 评估指标解释

| 指标 | 含义 | 越大是否越好 |
| --- | --- | --- |
| Text-BLEU | 模型生成翻译文本与 reference translation 的 BLEU | 是 |
| Speech-BLEU | 生成语音经 ASR 后与 reference translation 的 BLEU | 是 |
| SLC-0.2 / SLC-0.4 | 生成与参考语音时长一致性 | 是 |
| UTMOS | 预测语音自然度/质量 | 是 |
| AutoPCP | 多语言韵律/表达相似度代理指标 | 是 |

报告需要按以下维度拆分：

- Phase2 / Phase3；
- dev / test；
- Quality / Performance；
- ZH→EN / EN→ZH。

不能把 UniST test 的数值直接和论文 CVSS-T Table 1 计算差值或排名。只有本地也完成同版本 CVSS-T test 后才可以直接比较。

## 24. 评估输出和报告路径

控制目录：

```text
eval_outputs/uniss_full198_phase2_phase3_<RUN_ID>/
```

主要报告：

```text
report/aggregate_report.md
report/aggregate_report.json
report/phase2_phase3_detailed_evaluation_report.md
```

试听音频在各 run 目录的：

```text
generated_audio/
source_audio/
reference_audio/
```

当前已完成的历史报告已经复制到仓库文档：

```text
docs/uniss_training_reproduction/uniss_full198_phase2_phase3_detailed_evaluation_report.md
```

## 25. 当前已完成结果的快速核对

```bash
for root in \
  checkpoints/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2 \
  checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4 \
  checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4; do
  printf '%s: ' "${root}"
  cat "${root}/latest_checkpointed_iteration.txt"
done
```

应输出：

```text
15465
15381
9075
```

当前完整 test 生成规模：

```text
Phase2 test decoded: 46,738，failed/no-semantic: 80
Phase3 test decoded: 46,738，failed/no-semantic: 47
```

当前 test 内部比较：Phase3 在 24 个 higher-is-better 指标单元中提高 21 项、下降 3 项；下降主要出现在部分 UTMOS 单元，翻译和语义指标总体提高。详细数字见配套评估报告。

## 26. 常见问题

### 26.1 TensorBoard 没有 validation loss

先看当前 iteration 是否已经达到 `EVAL_INTERVAL`。Phase2/Phase3 当前配置是每 100 step validation；Simul-UniSS 的某些 stage 使用 500 step，不要混淆。

```bash
rg 'iteration +[0-9]+/|validation loss at iteration' <训练日志> | tail
```

### 26.2 GPU 利用率为 0

模型初始化、checkpoint load、dataset index 构建期间 GPU 可能暂时为 0。只有日志进入真实 `iteration 1/...` 后才能判断训练吞吐。持续为 0 时检查：

```bash
pgrep -af 'torchrun|pretrain_uniss_megatron'
tmux capture-pane -pt <session> -S -100
nvidia-smi
```

### 26.3 `libcudnn_graph.so.9` 找不到

先 source activation：

```bash
source /opt/dlami/nvme/jasonleeeli/env_recovery/uniss-train-20260721/activate_uniss.sh
```

runner 还会自动把 CUDA 和 pip `nvidia/*/lib` 加入 `LD_LIBRARY_PATH`。

### 26.4 Phase2 loss 再次突然上升

确认没有回到旧配置：

```text
source 必须是 Phase1 recovery iter_0015465
LR 必须是 1e-5 → 1e-6
clip grad 必须是 0.5
dataloader-type 必须是 cyclic
--no-data-sharding 必须开启
full validation 必须开启
```

不要从旧 Phase2 recovery iteration 4600 继续，也不要使用非 shuffle 的 `single` dataloader。

### 26.5 评估中途失败如何恢复

完整 dev/test 的 generation、decode 和 objective metrics 支持 resume。保留原目录和 `RUN_ID`，先阅读对应日志，再用相同配置重新启动。HF smoke/listen 如果留下不完整目录，主 runner 会要求使用新的 `RUN_ID`，以避免覆盖不可审计的半成品。

### 26.6 为什么 full evaluation 的 GPU 不总是 100%

自回归生成包含不同长度序列、CPU 调度、音频解码和写盘；ASR/UTMOS/AutoPCP 又是不同模型和 batch 形态，所以不能用训练时的稳定满载作为唯一标准。当前脚本通过四路 data parallel、duration bucket、batch decode 和指标分片提高吞吐，但不改变样本和指标定义。

## 27. 最小复现检查清单

开始训练前：

- [ ] 198/198 train parquet；
- [ ] train rows = 19,785,924；
- [ ] dev rows = 7,965；
- [ ] test rows = 23,369；
- [ ] UniSS tokenizer size = 180,407；
- [ ] base Megatron tracker = 0；
- [ ] Phase1/2/3 task JSONL count 正确；
- [ ] Phase2 replay mix = 59,357,772 : 29,678,886；
- [ ] packed count = 800,632 / 1,968,716 / 1,161,587；
- [ ] 三个 validation packed 文件通过 validator；
- [ ] 8 张 GPU 可见；
- [ ] dry-run 命令中的 load/save 路径正确。

训练完成后：

- [ ] Phase1 recovery tracker = 15,465；
- [ ] Phase2 tracker = 15,381；
- [ ] Phase2 pilot/final gate = pass；
- [ ] Phase3 tracker = 9,075；
- [ ] 三阶段 final validation 有限；
- [ ] NaN iteration = 0；
- [ ] skipped iteration = 0；
- [ ] checkpoint、log、TensorBoard、manifest 和 completion marker 都存在。

评估完成后：

- [ ] Phase2/Phase3 HF export 验证通过；
- [ ] smoke、listen、dev、test 都有 summary；
- [ ] generated/source/reference audio 可读取；
- [ ] Text-BLEU、Speech-BLEU、SLC、UTMOS、AutoPCP 文件存在；
- [ ] `COMPLETE` 标记存在；
- [ ] Markdown/JSON 聚合报告生成；
- [ ] 报告明确写出 UniST 与 CVSS-T 的比较边界。

## 28. 关键配置和入口索引

```text
数据与 packing：
  training/prepare_phase1_alignment.py
  training/prepare_unist_s2st.py
  training/mix_sample_jsonl.py
  training/pack_sequences.py
  training/pack_sequences_parallel.py
  training/validate_packed_jsonl.py
  scripts/pack_unist198_full.sh
  configs/experiments/uniss_qwen0p5b_unist198_full_v1.env

Phase1：
  scripts/run_qwen0p5b_unist198_all_phases.sh
  scripts/run_qwen0p5b_unist198_phase1_recovery_b.sh
  configs/experiments/uniss_qwen0p5b_unist198_phase1_recovery_b1_v2.env

Phase2：
  scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh
  configs/experiments/uniss_qwen0p5b_unist198_phase2_from_phase1_v4.env

Phase3：
  scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh
  configs/experiments/uniss_qwen0p5b_unist198_phase3_after_phase2_v4.env

评估：
  experiments/evaluation/uniss_full198_phase2_phase3/README.md
  experiments/evaluation/uniss_full198_phase2_phase3/prepare_manifests.sh
  experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh
  experiments/evaluation/uniss_full198_phase2_phase3/run_complete_evaluation.sh
  evaluation/aggregate_report.py
```
