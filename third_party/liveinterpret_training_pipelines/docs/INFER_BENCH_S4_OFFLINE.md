# S4-offline 推理跑测：不同 GPU 单卡对比

S4-offline = 自研句级（离线整句）同传模型：`streaming_si_4b` S4 `checkpoint-24980`，推理入口
`custom/streaming_si/inference/infer_s2st.py`（thinker 增量缓存 + CUDA Graph、talker CUDA Graph、
flow ODE 5 步、源语音声学 prompt）。

本文档每种 GPU 一列，基线列取自 `README_self.md`（stevehguo 第二轮评测）的 S4-offline 列。

> **列的含义**：`README baseline` = `/data/workspace/README_self.md` 公布的该系统的数值（基线）；`H200 / A10G / L40S` = 本次单卡实测。README_self.md 表格里没展示的指标（Text-chrF、显存峰值、p90 等）沿用其底层 `out3/_report` 报告的数值。

同系列文档：S5-streaming、Repro-Omni-offline、Repro-Omni-L1～L4 各一份（待补）。

| GPU | 状态 | 跑测目录 |
|---|---|---|
| H200（p5e，141GB） | ✅ 2026-10-21 完成 | `/mnt/fsx/junyaoyang/infer_bench/20261021-194857/` |
| A10G（g5.2xlarge，23GB） | ✅ 2026-10-22 完成 | `iegg-speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261022-213001-a10-s4/` |
| L40S（g6e.xlarge，46GB） | ✅ 2026-10-26 完成 | `speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261026-193402-l40s-s4/` |

---

## 1. 跑测口径

与 README_self 完全相同，只改「怎么跑」，不改「跑什么」：

| 项 | 取值 |
|---|---|
| 测试集 | RealSI en→zh 346 句 / zh→en 431 句、ACL 60/60 en→zh 416 句，共 1193 句、142.0 分钟源音频（stevehguo `work/manifests/all.jsonl`，只读复用） |
| 模型 | `/mnt/fsx/stevehguo/eval/models/s4_ck`（S4 checkpoint-24980，已去优化器状态），只读 |
| 推理参数 | `infer_s2st.py` 全部默认值：thinker greedy、talker top_k 25 / top_p 0.8、flow ODE 5 步、源语音 prompt 开、seed 0（每句重设） |
| 预热 | 清单最前面复制 1 句（`_warmup_` 前缀），用于录 CUDA Graph，打分时剔除 |
| 打分 | 同一套 `custom/streaming_si/eval/`：Whisper-large-v3 ASR → BLEU/chrF（SacreBLEU）、COMET（wmt22-comet-da）、UTMOS22、SIM-o / ESim / AutoPCP |
| 运行方式 | **单卡独占、1193 句串行**（README_self 是 7 卡各跑一片） |

入口：`pipelines/streaming_si/bench_entry.sh`（`SYSTEM=s4 GPU_TAG=<卡>`），任务 yaml
`ktrain/ktrainjob_infer_bench.yaml`。步骤与产物见第 5 节。

---

## 2. 结论（H200）

1. **质量指标与 README_self 逐位相同。** 三个数据集的 ASR/Text 的 BLEU、chrF、COMET，UTMOS，
   SIM-o / ESim / AutoPCP 全部一致到小数点后 4 位：同权重、同参数、同 seed 下推理是确定的，
   单卡串行与 7 卡分片不影响结果。之后换卡时，质量指标的差异就只来自硬件数值（bf16 内核差异）。
   抽查 `en2zh-01-tech_0003`：本次输出 wav 与 README_self 那一轮（`out2/s4/shard0/`）的 md5 相同（`656e470d…`），
   即连合成语音也逐字节一致。
2. **单卡 RTF 0.300（模型内计时），单句平均 2.14s、p90 3.15s、最长 6.11s。**
   各数据集 RTF 0.280～0.318，与 README_self 的 0.285～0.324 相差 ≤0.006。
   README_self 正文说该列是「7 进程并发」测的，实际 7 个进程各占一张卡，并不争抢同一张卡，
   所以数字本来就接近单卡独占值——本次单卡结果印证了这一点。
3. **算上读文件 / 写 wav 的整体墙钟 RTF 为 0.345**（2939s / 142 分钟），I/O 约占 13%。
   在线服务里没有写盘这一步，以 0.300 为准。
4. **稳定性：1193 句 0 失败、0 空译文、0 空语音。**
5. **显存峰值 74.7GB（nvidia-smi 口径）。** 单句服务实测稳态约 33GB；长跑中 PyTorch 缓存分配器
   不释放已申请的块，显示值随最长句子逐步涨高。这是「占用」不是「必需」——A10G 上实测峰值只有
   15.6GB（23GB 可用），见第 6 节。

## 2.1 A10G 补充结论

6. **质量与 H200 / README_self 基本一致，差距在采样噪声量级。** 三个数据集 ASR-BLEU 26.90 / 23.41 / 47.81，
   与 H200 相差 ≤0.4；COMET、UTMOS、SIM-o 相差 ≤0.01。bf16 matmul 在不同 GPU 架构上舍入不同，
   不会像同款硬件那样逐位相同（对照 H200 vs README_self 的「逐位相同」）。
7. **RTF 0.478，约为 H200（0.300）的 1.6 倍。** 各数据集 RTF 0.440～0.525，单句均值 3.25～3.53s。
   A10G 算力与显存带宽远低于 H200，该比例符合预期。
8. **显存峰值 15.6GB，仅为 23GB 可用显存的 68%，无 OOM、0 失败。** 权重 bf16 约 14GB 是主要占用，
   长跑未把缓存分配器撑到接近上限——A10G 单卡跑 S4-offline 有安全余量。

## 2.2 L40S 补充结论

9. **RTF 0.332，介于 H200（0.300）与 A10G（0.478）之间，是 H200 的 1.11 倍。** 单句均值 2.37s、p90 3.55s，
   推理 3442s。L40S 的算力/带宽约为 A10G 的 1.4～1.5 倍，落在预期位置。
10. **显存峰值 15.9GB（46GB 卡的 35%），无 OOM、0 失败**，与 A10G 的 15.6GB 几乎一样——
   说明 S4-offline 的常驻占用就是权重 + 声码器，与卡型无关，余量充足。
11. **质量与 H200 / A10G 基本一致**：ASR-BLEU 26.76 / 23.26 / 47.56（H200 26.48 / 23.50 / 47.62，
   A10G 26.90 / 23.41 / 47.81），COMET、UTMOS、SIM-o 相差 ≤0.01，AutoPCP 与 README_self 正文一致。
   三张卡的差异都在 bf16 舍入的噪声带内。

---

## 3. 对比表

### 3.1 效率与资源（全部 1193 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| RTF（模型内计时，Σwall / Σ源时长） | ≈0.30（各数据集 0.285～0.324） | **0.300** | **0.478** | **0.332** |
| RTF（整体墙钟，含 I/O） | - | 0.345 | 0.552 | 0.404 |
| 单句耗时 均值 / 中位 | 2.10～2.23s（各数据集均值） | 2.14s / 2.01s | 3.41s / 3.23s | 2.37s / 2.27s |
| 单句耗时 p90 / 最大 | - | 3.15s / 6.11s | 5.26s / 9.77s | 3.55s / 5.79s |
| 显存峰值（nvidia-smi） | - | 74.7GB | 15.6GB | 15.9GB |
| 失败 / 空译文 / 空语音 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

### 3.2 RealSI en→zh（346 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 26.48 | 26.48 | 26.90 | 26.76 |
| ASR-chrF | 23.85 | 23.85 | 24.21 | 24.01 |
| ASR-COMET | 0.7335 | 0.7335 | 0.7313 | 0.7344 |
| Text-BLEU | 28.25 | 28.25 | 28.38 | 28.33 |
| Text-chrF | 25.39 | 25.39 | 25.52 | 25.49 |
| Text-COMET | 0.7838 | 0.7838 | 0.7854 | 0.7841 |
| UTMOS | 3.274 | 3.274 | 3.268 | 3.269 |
| SIM-o | 0.813 | 0.813 | 0.815 | 0.813 |
| SIM-o delta | 0.162 | 0.162 | 0.165 | 0.159 |
| ESim delta | 0.016 | 0.016 | 0.013 | 0.013 |
| AutoPCP delta | 1.186 | 1.186 | 1.204 | 1.188 |
| RTF（语料级） | 0.302 | 0.303 | 0.471 | 0.328 |
| RTF（中位） | 0.314 | 0.314 | 0.488 | 0.341 |
| 单句耗时均值（s） | 2.23 | 2.24 | 3.48 | 2.42 |
| 时长比 | 0.952 | 0.952 | 0.948 | 0.955 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

### 3.3 RealSI zh→en（431 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 23.50 | 23.50 | 23.41 | 23.26 |
| ASR-chrF | 51.14 | 51.14 | 50.91 | 50.41 |
| ASR-COMET | 0.7438 | 0.7438 | 0.7427 | 0.7348 |
| Text-BLEU | 23.33 | 23.33 | 23.37 | 23.25 |
| Text-chrF | 52.66 | 52.66 | 52.68 | 52.66 |
| Text-COMET | 0.7603 | 0.7603 | 0.7603 | 0.7602 |
| UTMOS | 3.676 | 3.676 | 3.674 | 3.656 |
| SIM-o | 0.896 | 0.896 | 0.898 | 0.885 |
| SIM-o delta | 0.146 | 0.146 | 0.147 | 0.143 |
| ESim delta | 0.018 | 0.018 | 0.022 | 0.021 |
| AutoPCP delta | 1.083 | 1.083 | 1.062 | 1.051 |
| RTF（语料级） | 0.324 | 0.318 | 0.525 | 0.360 |
| RTF（中位） | 0.340 | 0.335 | 0.540 | 0.375 |
| 单句耗时均值（s） | 2.18 | 2.14 | 3.53 | 2.42 |
| 时长比 | 1.030 | 1.030 | 1.059 | 1.055 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

### 3.4 ACL 60/60 en→zh（416 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 47.62 | 47.62 | 47.81 | 47.56 |
| ASR-chrF | 39.82 | 39.82 | 40.01 | 39.89 |
| ASR-COMET | 0.7897 | 0.7897 | 0.7873 | 0.7917 |
| Text-BLEU | 49.86 | 49.86 | 49.92 | 49.89 |
| Text-chrF | 41.62 | 41.62 | 41.70 | 41.68 |
| Text-COMET | 0.8285 | 0.8285 | 0.8295 | 0.8286 |
| UTMOS | 3.341 | 3.341 | 3.356 | 3.345 |
| SIM-o | 0.948 | 0.948 | 0.949 | 0.947 |
| SIM-o delta | 0.105 | 0.105 | 0.105 | 0.104 |
| ESim delta | 0.012 | 0.012 | 0.010 | 0.011 |
| AutoPCP delta | 1.023 | 1.023 | 1.012 | 1.026 |
| RTF（语料级） | 0.285 | 0.280 | 0.440 | 0.310 |
| RTF（中位） | 0.297 | 0.288 | 0.457 | 0.325 |
| 单句耗时均值（s） | 2.10 | 2.06 | 3.25 | 2.28 |
| 时长比 | 0.882 | 0.882 | 0.886 | 0.892 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

---

## 4. 换卡跑测要注意的

| 项 | H200 | A10G（sm_86） | L40S（sm_89） |
|---|---|---|---|
| 镜像 kernel 架构 | 基础镜像 `TORCH_CUDA_ARCH_LIST="8.9;9.0"` | **不在列表里**，但实测 flash-attn / fla / causal-conv1d 在 A10G（sm_86）上均可用（torch 自带 PTX JIT 兜底），S4 推理未受影响 | 在列表里 |
| 显存 | 141GB | 23GB 可用；实测峰值 15.6GB（权重 bf16 约 14GB + 声码器 1.3GB），有余量 | 46GB 可用；实测峰值 15.9GB |
| 集群 / 调度 | speech-h200（ktrain） | `iegg-speech-model-data-eks`，`curion` 命名空间，`curion-runtime` SA，裸 Pod + kubectl（非 ktrain） | `speech-model-data-eks`（共享 HyperPod），同样 `curion` ns + `curion-runtime` SA；g6e.xlarge 只有 4 vCPU，`OMP_NUM_THREADS=3`，requests 3 核 / 20Gi |
| 模型与数据 | FSX 直读 | 已搬运至 S3（`junyaoyang/infer_bench_assets/`，29.8GB，经 `speech-model-data-eks-s3-data-pvc` 挂载），`ASSETS_ROOT` 指向挂载路径即可复用 | 同一个桶、同一个 PVC，`ASSETS_ROOT` 不变 |

换卡时保持第 1 节口径不变，只改 `GPU_TAG` 与运行环境；结果填进第 3 节对应列。

---

## 5. 复现

```bash
# 提交（单卡；submit.sh 要求白名单节点数 = numNodes，提交前把白名单改成一台有空卡的节点）
bash .codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/submit.sh \
    --yaml ktrain/ktrainjob_infer_bench.yaml --repo . --launcher ktrain/launch_train.sh \
    --pvc speech-h200-s3-pvc --job-name-prefix bench-s4-h200 --aws-creds-secret aws-local-creds
```

`bench_entry.sh` 的 7 步（每步写 `<OUT>/status.log`）：

| 步骤 | 内容 | H200 耗时 |
|---|---|---|
| 1 环境 | 检查模型 / 清单，复制 HF 与 torch.hub 缓存到自己目录（不写 stevehguo 目录） | <1 分钟 |
| 2 清单 | 1193 句 + 1 预热句 | 秒级 |
| 3 推理 | `infer_s2st.py` 单卡串行 | 49 分钟 |
| 4 合并 | 剔除预热句，补 `tgt_lang` / `source_sec` | 秒级 |
| 5 GPU 指标 | Whisper ASR、UTMOS（合成 + 源）、SIM-o / ESim / AutoPCP | 5、6 两步合计约 12 分钟 |
| 6 打分 | 三个数据集各一份 `score_eval.py` 报告（BLEU / chrF / COMET） | （同上） |
| 7 汇总 | `summary.md`：本机 vs README_self 逐项对照 | 秒级 |

只重跑打分（复用已有推理与 GPU 指标）：yaml 里加 `RESUME_FROM: "score"` 和 `OUT: "<已有目录>"`。

产物：`infer/`（每句 wav + `results.jsonl` + `merged.jsonl`）、`asr/`、`utmos/`、`para/`、
`report/{realsi_en2zh,realsi_zh2en,acl6060_en2zh}.json`、`logs/`（含 `gpu_infer.csv` 显存曲线）、`summary.md`。

### 跑测中修过的问题

| 问题 | 原因 | 处理 |
|---|---|---|
| COMET 报 `OfflineModeIsEnabled` | xlm-roberta 分词器会联网查补丁，`HF_HUB_OFFLINE=1` 下直接抛异常 | 不开离线；HF / torch.hub 缓存复制到本次输出目录再用 |
| 首次汇总 ASR-BLEU 为 0、UTMOS / 副语言为空 | 拼文件列表时 glob 已是完整路径又拼了一次目录 | 修正 `j()`，用 `RESUME_FROM=score` 只重跑打分 |
| `LIMIT` 冒烟时分数全 0 | 打分用全量清单，缺的句子按空译文计 | `LIMIT` 模式下打分清单只保留实际跑过的句子 |
