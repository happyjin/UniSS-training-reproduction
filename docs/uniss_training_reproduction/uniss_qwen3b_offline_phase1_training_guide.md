# Qwen2.5-3B Offline 训练手册

> 3B offline 现在是 **两阶段**：Phase1 对齐，然后 Phase2（Quality + Performance + Phase1 replay）。不再训 Direct S2ST，也不再单开 Phase3。
>
> 这份文档覆盖 Phase1，以及已经准备好、等 Phase1 结束后再启动的 Phase2。
>
> 仓库：`/opt/dlami/nvme/neuhao/UniSS`
>
> 机器：本机 8×H200，用户根目录 `USER_ROOT=/opt/dlami/nvme/neuhao`
>
> 最后核对：2026-09-22。Phase1 高 LR 在约 8000 step 后爆炸，已从 iter 7000 用 `1e-4 → 1e-5` cosine 做 recovery。Phase2 数据和启动脚本已准备，等 recovery 到 11765 后再开。

相关文档（0.5B 历史，不要拿来直接跑 3B）：

- 0.5B 完整复现教程：`docs/uniss_training_reproduction/uniss_full198_phase1_phase3_reproduction_tutorial.md`
- 0.5B 项目总览：`docs/uniss_training_reproduction/uniss_project_offline_phase1_3_and_streaming_stage_a_b_colleague_overview.md`

---

## 1. 先读这 12 条硬规则

1. **这是 UniST-198 公开复现，不是论文原配。** 论文主模型是 1.5B、约 77.1k 小时 speech alignment、WMT17 2.3B tokens。当前路线用公开 UniST 198 shard + MT proxy，实验名必须带 `unist198`。
2. **3B 指** `Qwen/Qwen2.5-3B-Instruct`**，**`model_type=qwen2`**。** 不是 Qwen3，也不是 0.5B。
3. **绝对不要覆盖 0.5B 的 checkpoint / 日志 / TensorBoard。** 3B 资产全部用独立目录，见第 4 节。
4. **Phase1 训练数据是 packed JSONL，不是 parquet，也不是 processed JSONL。** Megatron 入口只吃 `data/megatron/phase1_unist198/packed_train.jsonl`。
5. **3B 必须** `TP=2` **+** `--sequence-parallel`**。** `TP=1` 会在 vocab CE 上 OOM。0.5B 当时是 `TP=1`、`micro-batch=2`。
6. **正式 Phase1 目标是 3 packed epoch = 18765 step。** 一个 packed epoch = 6255 step。不要把 15465 当成 3B 的训练长度。
7. **新阶段启动：**`FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0`**。** 只加载模型权重，重置 Adam / RNG / scheduler / iteration。
8. **同一阶段断点续训：**`FINETUNE=0 LOAD_OPTIM=1 LOAD_RNG=1`**，**`LOAD_CHECKPOINT` **改成本阶段** `SAVE_DIR`**。**
9. **学习率沿用 0.5B 论文配方** `8e-4` **constant。** 0.5B 曾经用这个 LR 在 iter 3300 后炸过。3B 是否会炸还不知道；见第 12 节。
10. **TensorBoard 必须开。** wandb 这轮先不做。
11. **conda 环境用本机** `conda_envs/uniss-train`**。** 激活脚本见第 6.2 节；不要用 `conda activate` 去找不存在的 `base`。
12. `configs/experiments/uniss_qwen3b_unist198_full_v1.env` **里默认仍是** `TP=1`**。** 正式训练时必须覆盖成 `TP=2`，或改这个文件。直接 source 后开跑会 OOM。

---



## 2. Phase1 在整条 offline 链路里是什么

3B offline 两阶段：

```text
Qwen2.5-3B-Instruct
  -> 扩词表 HF checkpoint
  -> Megatron iter_0000000
  -> Phase1  ASR / S2TT / TTS / MT-proxy
  -> Phase2  Quality + Performance + Phase1 replay 2:1
```

0.5B 当时还有 Direct S2ST，并在 Phase3 里把它拿掉。3B 直接不训 Direct，Phase2 就是最后一阶段。Quality 是慢模式（转录 + 翻译 + 目标语音），Performance 是平衡模式（翻译 + 目标语音），Direct 是跳过文本、源语音直接出目标语音。评测只看 Quality 和 Performance。

Phase1 的训练目标是让 3B 解码器学会 **speech token ↔ text** 对齐。每条 UniST train row 生成 4 个样本：


| 任务     | 输入                 | 输出                  | 说明                           |
| ------ | ------------------ | ------------------- | ---------------------------- |
| `asr`  | 源语音 GLM token      | 源语转录                | 语音理解                         |
| `s2tt` | 源语音 GLM token      | 目标语翻译               | 跨语文本                         |
| `tts`  | 文本 + speaker token | 目标语音 semantic token | 语音生成                         |
| `mt`   | 源语转录               | 目标语翻译               | UniST 文本对做 MT proxy，不是 WMT17 |


Phase1 不训练 Quality / Performance。Phase2 才训练这两项，并混入 Phase1 replay。Direct S2ST 和 Phase3 都不再做。

## 2.1 Phase2 数据和启动（等 Phase1 recovery 到 11765 再开）

数据脚本已经在准备，不占 GPU：

```bash
bash scripts/prepare_qwen3b_phase2_qp_replay.sh
```

产物：

```text
data/processed/phase2_qwen3b_qp_replay_sharded/train-00000.jsonl ... train-00197.jsonl
data/megatron/phase2_qwen3b_qp_replay/packed_train.jsonl
data/megatron/phase2_qwen3b_qp_replay/packed_train.jsonl.count
```

混合比例是 Quality/Performance : Phase1 replay = 2:1。Quality/Performance 用本机已有的 Phase3 JSONL（本来就没有 Direct）。Phase1 replay 从本地 198 个 parquet 现做，不从 S3 拉。验证集用现成的 `phase3_valid_packed.jsonl`，只含 Quality/Performance。

一个 packed epoch 的步数由 runner 计算：`ceil(packed_count / 128)`。

Phase1 到 11765 之后再启动，不要提前开：

```bash
cd /opt/dlami/nvme/neuhao/UniSS
bash scripts/run_qwen3b_unist198_phase2.sh
```

脚本会拒绝启动，除非 Phase1 recovery 的 tracker 正好是 `11765`。配方沿用 0.5B Phase2：`lr 1e-5 -> 1e-6 cosine`，warmup 400，clip 0.5，cyclic shuffle，只加载 Phase1 权重。并行仍是 TP=2、micro-batch 1、global batch 128。TensorBoard 端口 `6008`。本机仍只留最新 3 个 ckpt，每 500 step 备份到：

```text
${S3_PROJECT_PREFIX}/checkpoints/uniss_qwen3b_phase2_unist198_qp_replay_v1/
```

`S3_PROJECT_PREFIX` 是团队内部的 S3 项目前缀，不写进仓库；启动前 `export S3_CHECKPOINT_PREFIX=${S3_PROJECT_PREFIX}/checkpoints/<实验名>`，否则备份脚本会直接报错退出。

续训：

```bash
bash scripts/run_qwen3b_unist198_phase2.sh --resume
```

---



## 3. 数据：到底用了什么、没用什么



### 3.1 训练真正吃进去的文件（必须有）


| 用途                | 本机路径                                                           | 规模                 | 从哪来                                   |
| ----------------- | -------------------------------------------------------------- | ------------------ | ------------------------------------- |
| Phase1 packed 训练集 | `data/megatron/phase1_unist198/packed_train.jsonl`             | **259G，800,632 行** | 从 S3 拉的现成 packed 数据，**不是本机重新 pack 的** |
| Phase1 packed 验证集 | `data/megatron/validation_unist_dev/phase1_valid_packed.jsonl` | 已存在                | UniST dev 按 Phase1 任务 pack            |


这两份文件的 tokenizer 是 UniSS 180407 词表。**packed 数据跨模型尺寸可复用**：0.5B 和 3B 吃同一份 packed JSONL，因为 packing 只依赖 tokenizer，不依赖 hidden size。

一行 packed JSONL 是一条 **最长 18000 token** 的 packed 序列，里面可能拼了多条 ASR/S2TT/TTS/MT 样本。所以：

```text
raw UniST train rows     = 19,785,924
Phase1 任务样本          = 19,785,924 × 4 = 79,143,696
packed 训练序列          = 800,632
global batch             = 128 条 packed 序列 / step
1 packed epoch           = ceil(800632 / 128) = 6255 step
3 packed epoch（正式）   = 6255 × 3 = 18765 step
tokens / step            = 128 × 18000 = 2,304,000
```

`TRAIN_ITERS=18765` 的含义就是：**把 800,632 条 packed 序列过 3 遍**。不是 18765 个 epoch，也不是 18765 条原始音频。

### 3.2 本机已有、但 Phase1 Megatron 训练不会直接读的文件


| 路径                                                           | 作用                                | Phase1 训练要不要                                                    |
| ------------------------------------------------------------ | --------------------------------- | --------------------------------------------------------------- |
| `data/raw/UniST/train-00000.parquet` … `train-00197.parquet` | 198 个原始 shard，约 29G，19,785,924 行  | 不直接读。只有你要从零 `prepare_phase1` + pack 才用                          |
| `data/raw/UniST/dev-00000.parquet`、`test-00000.parquet`      | 验证/评测原始数据                         | 训练不读；评测才用                                                       |
| `pretrained_models/UniSS/`                                   | UniSS tokenizer + GLM-4 + BiCodec | 扩词表、以后推理需要；Megatron 训练用 `NullTokenizer` + `--vocab-size 180407` |
| `pretrained_models/Qwen2.5-3B-Instruct/`                     | HF 基座                             | 只用于扩词表初始化，训练时不读                                                 |




### 3.3 本机没有、Phase1 训练也不需要的文件


| 资产                                | S3 前缀                                                      | 何时才需要拉                  |
| --------------------------------- | ---------------------------------------------------------- | ----------------------- |
| Phase1 processed JSONL（79M 行任务样本） | `data/processed/phase1_unist198_sharded/`                  | 只有 packed 丢了、要从零 pack 时 |
| Phase2/3 processed / packed       | `data/processed/phase2*`、`data/megatron/phase2*`、`phase3*` | Phase2/3，现在不要拉          |
| 0.5B checkpoint                   | `checkpoints/uniss_qwen0p5b_*`                             | 不要拿来初始化 3B              |


S3 bucket 前缀（本机 IAM 只能列这个前缀，不能 `ListAllMyBuckets`）：

```text
${S3_PROJECT_PREFIX}/    # 团队内部前缀，向项目负责人获取
```

拉 packed 数据时用过：

```bash
aws s3 sync \
  ${S3_PROJECT_PREFIX}/data/megatron/phase1_unist198/ \
  /opt/dlami/nvme/neuhao/UniSS/data/megatron/phase1_unist198/ \
  --exclude '*tmp*' --exclude '*.tmp*'
```

跳过了约 82G 的临时文件。不要把 `*.tmp*` 也 sync 下来。

### 3.4 如果 packed 丢了，要从零造数据（正常训练不要走这条）

完整 0.5B 教程第 8–11 节写了 parquet → JSONL → pack。3B 复用同一套命令，只是输出目录不要覆盖 0.5B。关键脚本：


| 脚本                                         | 作用                                        |
| ------------------------------------------ | ----------------------------------------- |
| `training/prepare_phase1_alignment.py`     | 每个 parquet shard 生成 ASR/S2TT/TTS/MT JSONL |
| `training/pack_uniss_sequences.py`         | 把 JSONL pack 成 18000 token 序列             |
| `training/prepare_validation_alignment.py` | UniST dev → Phase1 验证 JSONL，再 pack        |


预期计数必须对上再训练：

```text
raw train rows     19,785,924
Phase1 JSONL 行数  79,143,696
packed 行数        800,632
```

计数不对就停，不要改 expected 继续训。

---



## 4. 3B 资产隔离清单（路径必须独立）

所有 3B 路径都在 `/opt/dlami/nvme/neuhao/UniSS` 下，不要写到 jasonleeeli 的 0.5B 目录。


| 资产                  | 路径                                                                 | 状态（2026-09-17）                         |
| ------------------- | ------------------------------------------------------------------ | -------------------------------------- |
| HF 基座               | `pretrained_models/Qwen2.5-3B-Instruct/`                           | 已下载                                    |
| 扩词表 HF ckpt         | `checkpoints/qwen2_3b_uniss_vocab_hf/`                             | 已生成。embedding `180407×2048`，新增 28471 行 |
| Megatron 初始化 ckpt   | `checkpoints/qwen2_3b_uniss_vocab/`                                | `iter_0000000`，TP1 保存，约 3.14B params   |
| Smoke v2 ckpt       | `checkpoints/uniss_qwen3b_phase1_unist198_smoke_v2/`               | `iter_0000050`，**不要当正式训练起点**           |
| 正式 Phase1 ckpt（尚未训） | 建议：`checkpoints/uniss_qwen3b_phase1_unist198_full_v1/`             | 空                                      |
| Smoke 日志            | `logs/uniss_qwen3b_phase1_unist198_smoke_v2.log`                   | 已完成                                    |
| Smoke GPU CSV       | `logs/uniss_qwen3b_phase1_unist198_smoke_v2_gpu_power_utility.csv` | 已完成                                    |
| Smoke TensorBoard   | `runs/uniss_qwen3b_phase1_unist198_smoke_v2/tensorboard/`          | 已有 events                              |
| 正式训练日志 / TB         | 见第 10 节建议命名                                                        | 未创建                                    |


HF 扩词表后的关键数字：

```text
原始 Qwen2.5-3B vocab     151936
UniSS tokenizer           180407
新增 token 行             28471
embedding 形状            [180407, 2048]
Megatron 参数量           ~3.14B（transformer block 日志写 2.77B，不含 vocab）
```

---



## 5. 和 0.5B Phase1 对比


| 项                 | 0.5B 已完成路线                                                        | 3B 当前路线                                            |
| ----------------- | ----------------------------------------------------------------- | -------------------------------------------------- |
| 基座                | `Qwen2.5-0.5B-Instruct`                                           | `Qwen2.5-3B-Instruct`                              |
| 层 / hidden / FFN  | 24 / 896 / 4864                                                   | **36 / 2048 / 11008**                              |
| heads / KV groups | 14 / 2                                                            | **16 / 2**                                         |
| 词表                | 180407                                                            | 同左，packed 数据可共用                                    |
| seq length        | 18000                                                             | 同左                                                 |
| GPU               | 8                                                                 | 同左（本机 H200）                                        |
| TP / PP           | **1 / 1**                                                         | **2 / 1**（TP=1 会 OOM）                              |
| sequence parallel | 关                                                                 | **必须开**（TP≠1 时）                                    |
| micro batch       | **2**                                                             | **1**                                              |
| DP                | 8                                                                 | 4（8 GPU / TP2）                                     |
| grad accum        | 8                                                                 | 32                                                 |
| global batch      | 128                                                               | 同左                                                 |
| tokens/step       | 2.304M                                                            | 同左                                                 |
| lr                | 先 8e-4 炸了，从 iter 3300 用 1e-4 cosine 救回来                           | **先按 8e-4 constant 开正式训练**；若再炸，复用 0.5B recovery 思路 |
| warmup            | 原配方 1 epoch = 6255                                                | 同左                                                 |
| 目标步数              | 原计划 18765；高 LR 跑到 3300 后切 recovery 再跑 15465，合计仍是 3.0 packed epoch | 18765，尚未开跑                                         |
| 实测稳态 step 时间      | 本机没有留下 0.5B 的 ms/step 日志                                          | **约 13.3–13.4 s/step**                             |
| 吞吐                | —                                                                 | 约 **406 TFLOP/s/GPU**                              |
| 功耗                | —                                                                 | median ~688W，p90 ~698W，偶发 716W（TDP 700W）           |


3B 比 0.5B 重在哪里：

- 参数大约 6 倍（0.5B → 3.14B）。
- hidden 896 → 2048，FFN 4864 → 11008。
- vocab CE 的 FP32 logits 是 `[seq, vocab] = [18000, 180407]`，单份约 12.1 GiB。TP=1 时这份 tensor 无法切开，所以 3B 必须 TP=2 把 vocab 维对半。
- micro-batch 从 2 降到 1，是为了省显存，不是改有效 batch。有效 batch 仍由 `global-batch-size=128` 决定。

---



## 6. 环境、脚本、每个文件干什么



### 6.1 目录角色

```text
UniSS/
  pretrained_models/Qwen2.5-3B-Instruct/   HF 基座
  pretrained_models/UniSS/                 tokenizer / GLM-4 / BiCodec
  data/raw/UniST/                          原始 parquet
  data/megatron/phase1_unist198/           Phase1 packed 训练数据
  data/megatron/validation_unist_dev/      packed 验证
  checkpoints/qwen2_3b_uniss_vocab/        Megatron 初始化（iter 0）
  checkpoints/uniss_qwen3b_phase1_*        Phase1 训练输出
  configs/experiments/uniss_qwen3b_*.env   实验参数
  scripts/train_phase1_qwen3b.sh           真正拼 torchrun 命令的入口
  scripts/run_qwen3b_unist198_phase1_smoke.sh  smoke 包装器
  training/pretrain_uniss_megatron.py      Megatron 训练主程序
  logs/                                    文本日志 + GPU CSV
  runs/<experiment>/tensorboard/           TensorBoard events
  third_party/Megatron-LM/                 Megatron 源码，必须在 PYTHONPATH 里
```



### 6.2 运行环境


| 项          | 值                                                                                              |
| ---------- | ---------------------------------------------------------------------------------------------- |
| Python env | `/opt/dlami/nvme/neuhao/conda_envs/uniss-train`                                                |
| 激活脚本       | `/opt/dlami/nvme/neuhao/env_recovery/uniss-train-20260721/activate_uniss.sh`                   |
| 作用         | 把 conda env 的 `bin/` 插到 PATH 最前，设 `HF_HOME` / `TMPDIR`，`unset PYTHONPATH`，`PYTHONNOUSERSITE=1` |
| 不要做的事      | `conda activate uniss-train`（这台机 conda hook 会去找不存在的 `base`）                                    |


激活后立刻补 CUDA / pip NVIDIA `.so` 路径，并确认 TransformerEngine 能 import。smoke 包装器已经做了这件事；手动跑时复制第 9.3 节。

关键环境变量：

```text
USER_ROOT=/opt/dlami/nvme/neuhao
HF_HOME=${USER_ROOT}/cache/huggingface
TMPDIR=${USER_ROOT}/tmp
PYTHONPATH=${REPO}/third_party/Megatron-LM:${REPO}
CUDA_DEVICE_MAX_CONNECTIONS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HF_HUB_OFFLINE=1          # 训练时建议开，避免联网
TRANSFORMERS_OFFLINE=1
```



### 6.3 脚本和配置逐个说明


| 文件                                                              | 作用                                                                                           | 什么时候用                                       |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `scripts/download_hf_assets.sh`                                 | 从 Hugging Face 拉基座。`qwen3b` 目标拉 `Qwen/Qwen2.5-3B-Instruct`                                   | 基座缺失时。已经拉过就不要再跑                             |
| `training/initialize_uniss_hf_checkpoint.py`                    | 把 Qwen embedding resize 到 180407，保存 HF 格式                                                    | 扩词表。已生成 `qwen2_3b_uniss_vocab_hf`           |
| `training/convert_hf_to_megatron.py`（或仓库里对应 convert 入口）         | HF → Megatron dist ckpt，写出 `iter_0000000`                                                    | 已完成 `checkpoints/qwen2_3b_uniss_vocab`      |
| `scripts/train_phase1_qwen3b.sh`                                | **3B Phase1 的真正训练入口**。写死 36L/2048/11008/16 heads，拼 `torchrun ... pretrain_uniss_megatron.py` | smoke 和正式训练都走它                              |
| `scripts/run_qwen3b_unist198_phase1_smoke.sh`                   | smoke 包装：source env、激活 conda、补 LD_LIBRARY_PATH、检查 8 GPU 和 TE、拒绝覆盖已有 SAVE_DIR、tee 日志          | 只用于 50-step smoke                           |
| `configs/experiments/uniss_qwen3b_unist198_phase1_smoke_v2.env` | smoke v2 参数：TP=2，50 iter，LR warmup 10                                                        | 复现 smoke 或对照                                |
| `configs/experiments/uniss_qwen3b_unist198_phase1_smoke_v1.env` | **作废。** TP=1，iter 1 后 OOM                                                                    | 不要再跑                                        |
| `configs/experiments/uniss_qwen3b_unist198_full_v1.env`         | 正式三阶段路径模板。**当前默认 TP=1，正式 Phase1 必须改成 2**                                                     | 正式训练前先改 / 覆盖                                |
| `scripts/start_unist198_tensorboard.sh`                         | 读 env 里的 `TENSORBOARD_ROOT` 起 TB。默认配置还是 0.5B env                                             | 3B 更建议直接 `tensorboard --logdir <本实验 TB 目录>` |
| `training/pretrain_uniss_megatron.py`                           | Megatron 主程序，读 packed JSONL，SFT loss                                                         | 不要直接手敲，除非 debug                             |
| `training/prepare_phase1_alignment.py`                          | parquet → Phase1 JSONL                                                                       | 从零造数据才用                                     |
| `training/pack_uniss_sequences.py`                              | JSONL → 18000 packed                                                                         | 从零造数据才用                                     |


`train_phase1_qwen3b.sh` 的职责边界：

- 它 **写死** 3B 结构：`--num-layers 36 --hidden-size 2048 --ffn-hidden-size 11008 --num-attention-heads 16 --num-query-groups 2`。
- 它从环境变量读：`TRAIN_DATA VALID_DATA LOAD_CHECKPOINT SAVE_DIR TP PP MICRO_BATCH_SIZE TRAIN_ITERS LR_WARMUP_ITERS SAVE_INTERVAL EVAL_* LOG_INTERVAL TENSORBOARD_* FINETUNE LOAD_OPTIM LOAD_RNG MASTER_PORT NPROC_PER_NODE`。
- 它默认 `lr=8e-4`、`global-batch-size=128`、`seq-length=18000`、`--bf16 --use-flash-attn --recompute-activations`。
- 额外 Megatron flag 通过脚本后面的 `"$@"` 传入，例如 `--sequence-parallel`、`--attention-backend fused`。

---



## 7. 训练超参（Phase1 正式配方）

下面是 **3B Phase1 正式训练应该使用的值**。Smoke 只改了 iter / interval / warmup，结构相同。

### 7.1 模型与并行


| 参数                                             | 值             | 来源                 |
| ---------------------------------------------- | ------------- | ------------------ |
| `--num-layers`                                 | 36            | Qwen2.5-3B         |
| `--hidden-size`                                | 2048          | 同左                 |
| `--ffn-hidden-size`                            | 11008         | 同左                 |
| `--num-attention-heads`                        | 16            | 同左                 |
| `--group-query-attention --num-query-groups`   | 2             | 同左                 |
| `--normalization`                              | RMSNorm       | Qwen2              |
| `--swiglu`                                     | 开             | Qwen2              |
| `--disable-bias-linear --add-qkv-bias`         | 开             | Qwen2              |
| `--position-embedding-type rope --rotary-base` | 1000000       | Qwen2.5            |
| `--seq-length`                                 | 18000         | UniSS packing      |
| `--max-position-embeddings`                    | 32768         | Qwen2.5            |
| `--tokenizer-type`                             | NullTokenizer | packed 已是 token id |
| `--vocab-size`                                 | 180407        | UniSS              |
| `--tensor-model-parallel-size` (`TP`)          | **2**         | smoke 证明必须         |
| `--pipeline-model-parallel-size` (`PP`)        | 1             | 单机 8 卡够            |
| `--sequence-parallel`                          | **TP≠1 时必须加** | 省 activation 显存    |
| `--micro-batch-size`                           | 1             | 3B 显存              |
| `--global-batch-size`                          | 128           | 与 0.5B / 论文配方对齐    |
| `--nproc_per_node`                             | 8             | 8 卡                |
| `--sft --uniss-strict-paper-config`            | 开             | UniSS SFT          |


并行怎么算：

```text
world_size = 8
TP = 2, PP = 1
DP = 8 / (2 * 1) = 4
grad_acc = global_batch / (micro_batch * DP) = 128 / (1 * 4) = 32
```

每个 step：4 个 DP replica 各做 32 次 micro-batch=1 的 18000 token 前向+反向，再 all-reduce。

### 7.2 优化器与日程


| 参数                                                        | 正式值                      | Smoke v2 值                |
| --------------------------------------------------------- | ------------------------ | ------------------------- |
| `--lr` / `--min-lr`                                       | `8e-4` / `8e-4`          | 同左                        |
| `--lr-decay-style`                                        | `constant`               | 同左                        |
| `--lr-warmup-iters`                                       | **6255**（1 packed epoch） | 10（只为了 50 step 内能爬到 8e-4） |
| `--train-iters`                                           | **18765**                | 50                        |
| `--weight-decay`                                          | 0.1                      | 同左                        |
| `--adam-beta1 / --adam-beta2`                             | 0.9 / 0.95               | 同左                        |
| `--bf16`                                                  | 开                        | 同左                        |
| `--use-flash-attn`                                        | 开                        | 同左                        |
| `--recompute-activations`                                 | 开                        | 同左                        |
| `--no-gradient-accumulation-fusion`                       | 开（这版 Megatron 需要）        | 同左                        |
| `--attention-backend`                                     | `fused`                  | smoke 显式传了                |
| `--rerun-mode`                                            | `disabled`               | smoke 显式传了；防 TE 数值检查拖慢/误报 |
| `--cross-entropy-loss-fusion --cross-entropy-fusion-impl` | `native`                 | smoke 显式传了；减轻 vocab CE 显存 |
| `--save-interval`                                         | 100                      | 50                        |
| `--eval-interval`                                         | 100                      | 50                        |
| `--eval-iters`                                            | 10                       | 2                         |
| `--log-interval`                                          | 10                       | 1                         |
| TensorBoard log interval                                  | 10                       | 1                         |


加载语义（新阶段从 iter 0 Megatron ckpt 起步）：

```text
FINETUNE=1
LOAD_OPTIM=0
LOAD_RNG=0
LOAD_CHECKPOINT=checkpoints/qwen2_3b_uniss_vocab
```

这会让 Megatron：

- 加载 iter 0 的模型权重（允许 TP1 ckpt → TP2 运行，smoke 已验证）；
- **不**加载 Adam 状态（iter 0 也没有可用的训练态）；
- **不**加载 RNG；
- 把当前 iteration 当成新阶段的 0，从头 warmup。



### 7.3 不要改的东西（除非你在做新实验）

- `global-batch-size=128` 和 `seq-length=18000`：改了就不再和 0.5B / 论文 token budget 可比。
- `vocab-size=180407`：必须和 packed 数据、UniSS tokenizer 一致。
- 模型结构数字：必须和 `Qwen2.5-3B-Instruct` 一致，否则 convert 出来的权重对不上。
- 不要用 0.5B 的 `scripts/train_phase1_unist198.sh` 跑 3B，那份脚本写死了 24L/896。

---



## 8. 时间估算（基于 smoke v2 实测）



### 8.1 Smoke v2 实测

实验：`uniss_qwen3b_phase1_unist198_smoke_v2`，8×H200，TP=2，50 step。


| 指标                                   | 值                                     |
| ------------------------------------ | ------------------------------------- |
| 进程启动 → 第 1 个 step 结束                 | 约 6 分钟（建模型 + 扫 259G JSONL 建 iterator） |
| iteration 1                          | 72.4 s（含首次编译/warmup）                  |
| iteration 2                          | 44.5 s                                |
| iteration 3 起稳态                      | **13.3–13.4 s/step**                  |
| 稳态吞吐                                 | ~406 TFLOP/s/GPU                      |
| train loss                           | 12.81 → 8.36                          |
| validation loss（iter 50，2 eval iter） | ~8.59                                 |
| skipped / NaN iter                   | 0 / 0                                 |
| 存 ckpt                               | iter 50，约 16 s                        |
| GPU 功耗                               | median ~688W，p90 ~698W，max ~716W      |
| 结论                                   | **成功。** 正式训练用同一并行配置                   |




### 8.2 正式 18765 step 墙钟

按稳态 **13.4 s/step** 算纯训练：


| 范围             | step  | 纯训练墙钟              |
| -------------- | ----- | ------------------ |
| 1 packed epoch | 6255  | **23.3 h**         |
| 3 packed epoch | 18765 | **69.8 h ≈ 2.9 天** |


额外开销（粗算）：


| 项                         | 估算                        |
| ------------------------- | ------------------------- |
| 启动扫数据                     | 每次启动约 4–6 min             |
| eval：每 100 step 跑 10 iter | 187 次 × ~65 s ≈ **3.4 h** |
| save：每 100 step           | 187 次 × ~16 s ≈ **0.8 h** |
| 合计                        | **大约 3.1–3.3 天** 不间断 8 卡  |


和 0.5B 比：3B 每 step 明显更慢（TP=2、模型更大、mbs=1 但 accum=32）。0.5B 本机没有留下可引用的 ms/step；不要用 3B 的 13.4 s 去反推 0.5B。3B 正式 Phase1 按 **三天出头** 预留机器。

如果中途挂了再 resume，每次重启都要重新扫 packed JSONL（几分钟），但不会重算已完成的 step。

---



## 9. 手动训练流程

全程在 **tmux** 里做。训练进程不要放前台 SSH。建议窗口：

```text
tmux new -s qwen3b
  0: train
  1: tensorboard
  2: gpu-monitor
  3: log-tail
```

下面所有命令默认：

```bash
export USER_ROOT=/opt/dlami/nvme/neuhao
export REPO_ROOT=${USER_ROOT}/UniSS
cd "${REPO_ROOT}"
```



### 9.1 一次性资产（已经做完，核对即可）

```bash
# 1) 基座
ls pretrained_models/Qwen2.5-3B-Instruct/config.json

# 2) 扩词表 HF
python - <<'PY'
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    "checkpoints/qwen2_3b_uniss_vocab_hf", local_files_only=True)
print(m.get_input_embeddings().weight.shape)  # expect [180407, 2048]
PY

# 3) Megatron iter 0
cat checkpoints/qwen2_3b_uniss_vocab/latest_checkpointed_iteration.txt
# expect: 0

# 4) packed 数据
wc -l data/megatron/phase1_unist198/packed_train.jsonl
# expect: 800632
ls data/megatron/validation_unist_dev/phase1_valid_packed.jsonl
```

如果基座缺失：

```bash
USER_ROOT=/opt/dlami/nvme/neuhao \
  bash scripts/download_hf_assets.sh qwen3b
```

如果还要重新扩词表 / 转 Megatron，命令以当时实际使用的 convert 脚本为准，输出目录必须仍是：

```text
checkpoints/qwen2_3b_uniss_vocab_hf
checkpoints/qwen2_3b_uniss_vocab
```

不要写进 `qwen2_05b_*`。

### 9.2 开跑前检查 GPU 和端口

```bash
nvidia-smi
# 8 张 H200 都应该空闲。不要和别的训练抢卡。

ss -ltnp | grep -E '29521|29711|29722|6007' || true
# master_port 和 TensorBoard port 不能被占用。
```

正式训练建议：

```text
MASTER_PORT=29711
TENSORBOARD_PORT=6007
```

Smoke 用过 `29722` / `6007`。如果 6007 还被旧 TB 占着，换 6008 或先杀掉旧 TB。

### 9.3 正式 Phase1 启动（推荐手动命令）

不要直接 source `uniss_qwen3b_unist198_full_v1.env` 后开跑，除非你已经把里面的 `TP` 改成 2。

```bash
source /opt/dlami/nvme/neuhao/env_recovery/uniss-train-20260721/activate_uniss.sh

# 补 CUDA / pip NVIDIA 库，确认 TE
python - <<'PY'
import os, site, ctypes, torch
from pathlib import Path
dirs = []
sp = Path(site.getsitepackages()[0])
dirs += [str(p) for p in sp.glob("nvidia/*/lib") if p.is_dir()]
os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH","")])
ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print("TE ok, ngpu=", torch.cuda.device_count())
PY

export EXPERIMENT_NAME=uniss_qwen3b_phase1_unist198_full_v1
export TRAIN_DATA=${REPO_ROOT}/data/megatron/phase1_unist198/packed_train.jsonl
export VALID_DATA=${REPO_ROOT}/data/megatron/validation_unist_dev/phase1_valid_packed.jsonl
export LOAD_CHECKPOINT=${REPO_ROOT}/checkpoints/qwen2_3b_uniss_vocab
export SAVE_DIR=${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}
export TENSORBOARD_DIR=${REPO_ROOT}/runs/${EXPERIMENT_NAME}/tensorboard
export LOG_PATH=${REPO_ROOT}/logs/${EXPERIMENT_NAME}.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TP=2
export PP=1
export MICRO_BATCH_SIZE=1
export TRAIN_ITERS=18765
export LR_WARMUP_ITERS=6255
export SAVE_INTERVAL=100
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export LOG_INTERVAL=10
export MASTER_PORT=29711
export TENSORBOARD_LOG_INTERVAL=10
export TENSORBOARD_MEMORY_INTERVAL=10
export FINETUNE=1
export LOAD_OPTIM=0
export LOAD_RNG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 拒绝覆盖已有正式目录
[[ ! -e "${SAVE_DIR}" ]] || { echo "SAVE_DIR exists: ${SAVE_DIR}"; exit 1; }
mkdir -p "${SAVE_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_PATH}")"

env \
  USER_ROOT="${USER_ROOT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  TP="${TP}" PP="${PP}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
  TRAIN_DATA="${TRAIN_DATA}" \
  VALID_DATA="${VALID_DATA}" \
  LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" \
  SAVE_DIR="${SAVE_DIR}" \
  TRAIN_ITERS="${TRAIN_ITERS}" \
  LR_WARMUP_ITERS="${LR_WARMUP_ITERS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_ITERS="${EVAL_ITERS}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  MASTER_PORT="${MASTER_PORT}" \
  TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
  TENSORBOARD_LOG_INTERVAL="${TENSORBOARD_LOG_INTERVAL}" \
  TENSORBOARD_MEMORY_INTERVAL="${TENSORBOARD_MEMORY_INTERVAL}" \
  FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${REPO_ROOT}/scripts/train_phase1_qwen3b.sh" \
  --lr 8e-4 --min-lr 8e-4 --lr-decay-style constant \
  --attention-backend fused \
  --rerun-mode disabled \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --sequence-parallel \
  2>&1 | tee -a "${LOG_PATH}"
```

`--dry-run` 只打印命令不训练：

```bash
bash scripts/train_phase1_qwen3b.sh --dry-run
```

包装脚本同样支持：

```bash
bash scripts/run_qwen3b_unist198_phase1_smoke.sh --dry-run \
  --config configs/experiments/uniss_qwen3b_unist198_phase1_smoke_v2.env
```



### 9.4 复现 50-step smoke（可选）

```bash
bash scripts/run_qwen3b_unist198_phase1_smoke.sh \
  --config configs/experiments/uniss_qwen3b_unist198_phase1_smoke_v2.env
```

注意：包装器遇到已存在的 `SAVE_DIR` 会直接拒绝。要重跑必须先换实验名，或手动挪走旧目录。**不要删正式 checkpoint 来重跑 smoke。**

### 9.5 同一阶段断点续训

训练已经写过 `SAVE_DIR/iter_XXXXXXX` 之后挂了，从该阶段继续：

```text
FINETUNE=0
LOAD_OPTIM=1
LOAD_RNG=1
LOAD_CHECKPOINT=${SAVE_DIR}          # 不再指向 qwen2_3b_uniss_vocab
SAVE_DIR=${SAVE_DIR}                 # 同一个目录
TRAIN_ITERS=18765                    # 仍是目标总步数，不是“再训 18765”
```

Megatron 会读 `latest_checkpointed_iteration.txt`，从下一个 iteration 继续。`TRAIN_ITERS` 是 **结束步数**，不是“再跑多少步”。

如果只想再跑 N 步，把 `TRAIN_ITERS` 设成 `当前 iteration + N`。

### 9.6 启动 TensorBoard 和 GPU 监控

TensorBoard（本机 data-server 二进制不可执行，必须关 fast loader）：

```bash
source /opt/dlami/nvme/neuhao/env_recovery/uniss-train-20260721/activate_uniss.sh
tensorboard \
  --logdir "${REPO_ROOT}/runs/${EXPERIMENT_NAME}/tensorboard" \
  --host 0.0.0.0 \
  --port 6007 \
  --load_fast=false
```

浏览器：`http://<机器IP>:6007`。SSH 隧道：

```bash
ssh -L 6007:127.0.0.1:6007 <host>
```

GPU 功耗 / 利用率（正式训练建议一直挂着）：

```bash
GPU_LOG=${REPO_ROOT}/logs/${EXPERIMENT_NAME}_gpu_power_utility.csv
echo "timestamp,gpu_index,power_w,util_gpu,util_mem,mem_used_mb" > "${GPU_LOG}"
while true; do
  ts=$(date -u +%FT%TZ)
  nvidia-smi --query-gpu=index,power.draw,utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits \
    | awk -v ts="$ts" -F',' '{gsub(/ /,"",$0); print ts","$0}' >> "${GPU_LOG}"
  sleep 10
done
```



### 9.7 怎样算“正式训练在正常跑”

启动后 6 分钟内应看到：

```text
building and compiling schedule
> learning rate decay style: constant
iteration        1/   18765 | ...
```

iter 3 之后：

```text
elapsed time per iteration (ms): 13xxx     # 约 13.3 s
throughput per GPU (TFLOP/s/GPU): 40x      # 约 406
number of skipped iterations:   0
number of nan iterations:   0
lm loss: 从 ~12 降到个位数（前几十 step）
learning rate: warmup 期间线性爬到 8e-4，之后保持 8e-4
```

iter 100 应出现 validation 和 `saving checkpoint`。

---



## 10. 日志怎么看



### 10.1 文本训练日志

路径：`logs/<EXPERIMENT_NAME>.log`

关键行：

```text
iteration     N/   18765 | consumed samples:  N*128 | elapsed time per iteration (ms): ... |
throughput per GPU (TFLOP/s/GPU): ... | learning rate: ... | global batch size: 128 |
lm loss: ... | grad norm: ... | number of skipped iterations: 0 | number of nan iterations: 0
```

字段含义：


| 字段                                | 含义                                | 异常时                           |
| --------------------------------- | --------------------------------- | ----------------------------- |
| `iteration`                       | 当前全局 step / 目标总 step              | 续训后应从 ckpt 的下一 step 开始，不应回到 1 |
| `consumed samples`                | 已吃掉的 packed 序列数 = iteration × 128 | 对不上说明 batch 设错                |
| `elapsed time per iteration (ms)` | 这个 step 墙钟。稳态应 ~13300–14000       | 长期 >20000 可能是 I/O 或通信问题       |
| `throughput per GPU`              | 理论 FLOP 估算，稳态 ~400                | 掉到几十通常是在做 eval/save，或卡住       |
| `learning rate`                   | warmup 内线性；之后 8e-4                | 若突然变成 0 或 NaN，停               |
| `lm loss`                         | 当前 step 的 LM loss                 | 连续爆炸（几十、几百）或 NaN：见第 12 节      |
| `grad norm`                       | 梯度范数。前几 step 到 100+ 可以出现          | 持续几千/inf 是炸的前兆                |
| `skipped iterations`              | 因 inf/nan 跳过的更新                   | 必须保持 0                        |
| `nan iterations`                  | 同左                                | 必须保持 0                        |


看最新进度：

```bash
rg "iteration +[0-9]+/" logs/${EXPERIMENT_NAME}.log | tail
```

看是否 OOM：

```bash
rg -n "out of memory|CUDA OOM|RuntimeError" logs/${EXPERIMENT_NAME}.log | tail
```

看 validation：

```bash
rg "validation loss|lm loss" logs/${EXPERIMENT_NAME}.log | tail
```



### 10.2 TensorBoard 曲线

events 在 `runs/<EXPERIMENT_NAME>/tensorboard/`。`train_phase1_qwen3b.sh` 在设置了 `TENSORBOARD_DIR` 后会打开：

- `--log-timers-to-tensorboard`
- `--log-validation-ppl-to-tensorboard`
- `--log-memory-to-tensorboard`
- `--log-world-size-to-tensorboard`
- `--log-throughput`

重点看：`lm loss`、`learning-rate`、`grad-norm`、`throughput`、`memory`。本机必须 `--load_fast=false`，否则 TB 起不来。

### 10.3 GPU CSV

列：`timestamp,gpu_index,power_w,util_gpu,util_mem,mem_used_mb`

H200 TDP 700W。smoke 时 median ~688W 说明算力吃满，偶发 716W 是瞬时超标，可接受。如果 8 卡长期 <200W 且 util 很低，训练可能卡在数据加载或已经挂了。

### 10.4 Checkpoint 目录结构

```text
checkpoints/<EXPERIMENT_NAME>/
  latest_checkpointed_iteration.txt
  iter_0000100/
  iter_0000200/
  ...
```

`latest_checkpointed_iteration.txt` 是 resume 的权威来源。保存间隔 100，18765 step 大约会有 187 个 ckpt 目录，磁盘要预留（3B TP2 dist ckpt 比 0.5B 大，按实际 `du -sh iter_*` 估，建议至少留数 TB 余量后再开正式训）。

---



## 11. 故障排查



### 11.1 `TP=1` OOM（已经踩过）

现象：iter 1 能跑，随后：

```text
torch.OutOfMemoryError: CUDA out of memory
Tried to allocate 12.10 GiB
File ".../cross_entropy.py", in vocab_parallel_cross_entropy
    logits_max = logits.float().max(dim=-1)
```

原因：vocab CE 把 `[18000, 180407]` logits 转 FP32，约 12.1 GiB。TP=1 无法切 vocab 维。

处理：`TP=2` **+** `--sequence-parallel` **+** `--cross-entropy-fusion-impl native`**。** 不要再试 TP=1。

### 11.2 TransformerEngine / libcudnn 找不到

现象：`libcudnn_graph.so.9: cannot open shared object file`，或 `import transformer_engine` 失败。

处理：按第 9.3 节把 `site-packages/nvidia/*/lib` 加进 `LD_LIBRARY_PATH`。必须用 `activate_uniss.sh` 提供的 env python，不要用系统 python。

### 11.3 conda activate 失败

现象：`conda activate` 报找不到 `base`。

处理：不要用 conda hook。只用：

```bash
source /opt/dlami/nvme/neuhao/env_recovery/uniss-train-20260721/activate_uniss.sh
```



### 11.4 TensorBoard 起不来

现象：`tensorboard: Permission denied` 或 data-server 无法执行。

处理：加 `--load_fast=false`。不要用 0.5B 的 `start_unist198_tensorboard.sh` 除非改了 `--logdir` 和 env。

### 11.5 `Refusing to overwrite existing smoke dir`

smoke 包装器保护机制。换 `EXPERIMENT_NAME` / `SAVE_DIR`，或把旧目录改名。

### 11.6 端口占用

`torchrun` 报 `address already in use`：换 `MASTER_PORT`。TB 端口占用：换 `--port`。

### 11.7 加载 TP1 ckpt 到 TP2

这是正常路径。iter 0 是 TP1 保存的。Megatron dist ckpt 可以在不同 TP 下加载。smoke v2 已经验证。

### 11.8 训练在 “building dataset / compiling” 停很久

259G JSONL 第一次建 iterator 要几分钟。超过 20 分钟仍无 `iteration 1` 再查磁盘和 CPU。不要此时 Ctrl-C 除非确认卡死。

### 11.9 loss 不降或 grad norm 爆炸

前 10–20 step loss 从 ~12 降到 ~8 是 smoke 的正常现象。如果正式训练在 warmup 结束后（iter >6255）出现：

- `lm loss` 变成几十/几百/NaN
- `grad norm` 持续爆炸
- `skipped/nan iterations` > 0

立刻停，保留最后一个健康 ckpt，走第 12 节 recovery，**不要继续用 8e-4 硬撞**。

---



## 12. 0.5B 高 LR 爆炸，以及 3B 该怎么防



### 12.1 0.5B 实际发生了什么

0.5B 正式 Phase1 也曾用 `lr=8e-4 constant`。在 **iteration 3300** 之后 loss/grad 爆炸。3300 是高 LR 阶段最后一个健康 ckpt。

之后的 recovery：

```text
从 iter 3300 只加载模型权重（FINETUNE=1, LOAD_OPTIM=0, LOAD_RNG=0）
lr = 1e-4, min-lr = 1e-5, cosine, warmup 200
再跑 15465 step
3300/6255 ≈ 0.53 packed epoch（作废的高 LR 进度，权重还在）
15465/6255 ≈ 2.47 packed epoch
合计 3.00 packed epoch 的 token budget
```

所以 15465 **不是** “Phase1 要训 15465 个 epoch”，而是 **recovery 段的步数**。3B 正式训练仍以 **18765** 为目标。

### 12.2 3B 建议

1. 先按论文/0.5B 原配方 `8e-4 constant + 6255 warmup` 开跑（与 smoke 一致）。
2. 盯 TensorBoard 的 loss / grad-norm。重点窗口：warmup 末尾（~6255）以及 2000–4000（0.5B 炸点附近）。
3. `SAVE_INTERVAL=100` 不要改大。爆炸后要能回退到最近健康 iter。
4. 若再炸：复制 0.5B recovery 语义——新实验目录、从最后健康 iter 加载、降 lr 到 `1e-4` cosine、`FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0`、剩余步数补齐到合计 3.0 packed epoch。

3B 比 0.5B 大，**不保证** 8e-4 一定炸，也不保证一定不炸。smoke 只有 50 step，不能当稳定性证据。

---



## 13. 3B 还没做、不要误当成已完成的事


| 项                             | 状态                                               |
| ----------------------------- | ------------------------------------------------ |
| HF 基座 + 扩词表 + Megatron iter 0 | 完成                                               |
| Phase1 packed 数据就位            | 完成                                               |
| 8 卡 smoke 50 step             | 完成（TP=2）                                         |
| 正式 Phase1 18765 step          | **未开始**                                          |
| Phase2 / Phase3 数据与训练         | 未开始                                              |
| wandb                         | 明确推迟                                             |
| 导出 HF、UniST dev/test 评测       | Phase1 完成后再做                                     |
| 用 smoke iter 50 初始化正式训练       | **不要。** 正式训练从 `qwen2_3b_uniss_vocab` iter 0 重新起步 |


---



## 14. FAQ

**Q: Phase1 用的是哪份数据？**  
A: 只使用 `data/megatron/phase1_unist198/packed_train.jsonl`（800,632 条 18k packed 序列）和 `data/megatron/validation_unist_dev/phase1_valid_packed.jsonl`。原始 198 parquet 和 79M processed JSONL 训练时不读。

**Q: 要不要重新 pack？**  
A: 不要。packed 已从 S3 拉齐，和 0.5B 共用同一 tokenizer。

**Q: 18765 / 6255 / 15465 / 3300 分别是什么？**  
A: 6255 = 1 packed epoch；18765 = 3 epoch，3B 正式目标；3300 = 0.5B 高 LR 最后健康 ckpt；15465 = 0.5B recovery 又跑的步数。3B 不要把 15465 当目标。

**Q: 为什么 3B 必须 TP=2，0.5B 却是 TP=1？**  
A: 3B 的 vocab CE FP32 logits 约 12.1 GiB，TP=1 切不开。0.5B hidden/激活更小，TP=1 + micro-batch 2 能放下。

**Q: smoke 的 iter 50 能不能接着训到 18765？**  
A: 不建议。smoke 的 warmup 只有 10 step，和正式 6255 warmup 不是同一条 LR 曲线。正式训练从 iter 0 Megatron ckpt 重新开。

**Q: 手动训练最小闭环是什么？**  
A: 激活 env → 确认 8 卡空闲 → `TP=2` 调 `scripts/train_phase1_qwen3b.sh` 并带上 `--sequence-parallel` 和 CE fusion → tee 到 `logs/` → 另开 TB `--load_fast=false`。

**Q: 训练脚本入口到底是哪一个？**  
A: `scripts/train_phase1_qwen3b.sh`。smoke 只是它的包装器。

**Q: 和论文 1.5B 比算复现吗？**  
A: 不算严格复现。这是 UniST-198 公开数据上的 3B 放大实验。

**Q: 功耗 716W 超过 700W TDP 要紧吗？**  
A: smoke 里是瞬时峰值，median 688W。正式训练继续记 CSV。若长期 >700W 且掉卡，再降频或查散热。

**Q: 验证集是全量还是采样？**  
A: 当前 Megatron 这条路径是 `--eval-iters N` 采样，不是 full validation。正式训练 `EVAL_ITERS=10`。0.5B recovery 文档提过这版 Megatron 的 `--full-validation` 有 bug，3B 先不要开。