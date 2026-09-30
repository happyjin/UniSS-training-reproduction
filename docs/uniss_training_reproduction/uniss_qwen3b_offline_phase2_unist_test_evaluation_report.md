# Qwen2.5-3B Offline UniST test 评测报告

> 评测日期：2026-09-28
>
> 对照对象：0.5B Offline Phase3 v4 `iter_0009075` 的 UniST test 结果。
> 0.5B 数字来自 `docs/uniss_training_reproduction/uniss_project_offline_phase1_3_and_streaming_stage_a_b_colleague_overview.md`。
>
> 3B 这条路线没有训 Direct S2ST，也没有单独的 Phase3。评测的是 Phase2 最终 checkpoint 的 Quality 和 Performance 两种模式。

## 1. 评测的模型

| 项 | 值 |
| --- | --- |
| 实验 | `uniss_qwen3b_phase2_unist198_qp_replay_v1` |
| Megatron checkpoint | `checkpoints/uniss_qwen3b_phase2_unist198_qp_replay_v1/iter_0010645` |
| Hugging Face 导出 | `checkpoints/exported_hf/qwen3b_phase2_unist198_iter_0010645_hf` |
| 训练内容 | Quality + Performance + Phase1 replay 2:1 |
| 学习率 | `1e-5 → 1e-6` cosine，10645 step，一个 packed epoch |
| 初始化 | Phase1 recovery `iter_0011765`，该权重来自第一次训练的健康 iter 7000 |
| 测试集 | UniST test，`experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_test_all.jsonl`，23369 句 |
| 生成 | 8 卡 vLLM 0.8.5，`temperature=0.7`，`top_p=0.8`，`repetition_penalty=1.1`，`max_new_tokens=1500` |
| 结果目录 | `eval_outputs/qwen3b_phase2_unist198_iter_0010645_unist_test_vllm/` |
| 汇总指标（已入库） | `reports/uniss_qwen3b_offline_phase2_unist_test_v1/` |

两种模式是同一个模型：

- Quality：先转写，再翻译，再出目标语音。
- Performance：跳过转写，直接出翻译和目标语音。

生成结果 46738 条（23369 × 2）。其中 26 条没有产出 semantic token，没有音频。Text-BLEU 覆盖 46731 条，SLC 和 UTMOS 覆盖 46712 条，Speech-BLEU 覆盖 46345 条。

## 2. 和 0.5B 的总表

0.5B 是 Phase3，3B 是 Phase2。两边都在同一份 UniST test 上、用同一套指标。

| 模式 | 方向 | 指标 | 3B | 0.5B | 差值 |
| --- | --- | --- | ---: | ---: | ---: |
| Performance | 中文→英文 | Text-BLEU | 45.63 | 32.45 | +13.18 |
| Performance | 中文→英文 | Speech-BLEU | 12.41 | 19.38 | -6.97 |
| Performance | 中文→英文 | SLC-0.2 | 0.527 | 0.632 | -0.105 |
| Performance | 中文→英文 | SLC-0.4 | 0.734 | 0.878 | -0.144 |
| Performance | 中文→英文 | UTMOS | 3.51 | 3.67 | -0.16 |
| Performance | 英文→中文 | Text-BLEU | 54.12 | 40.54 | +13.58 |
| Performance | 英文→中文 | Speech-BLEU | 51.17 | 39.00 | +12.17 |
| Performance | 英文→中文 | SLC-0.2 | 0.717 | 0.718 | -0.001 |
| Performance | 英文→中文 | SLC-0.4 | 0.926 | 0.942 | -0.016 |
| Performance | 英文→中文 | UTMOS | 3.33 | 3.35 | -0.02 |
| Quality | 中文→英文 | Text-BLEU | 49.75 | 39.38 | +10.37 |
| Quality | 中文→英文 | Speech-BLEU | 12.75 | 22.73 | -9.98 |
| Quality | 中文→英文 | SLC-0.2 | 0.515 | 0.636 | -0.121 |
| Quality | 中文→英文 | SLC-0.4 | 0.710 | 0.871 | -0.161 |
| Quality | 中文→英文 | UTMOS | 3.51 | 3.67 | -0.16 |
| Quality | 英文→中文 | Text-BLEU | 57.06 | 48.17 | +8.89 |
| Quality | 英文→中文 | Speech-BLEU | 53.68 | 46.31 | +7.37 |
| Quality | 英文→中文 | SLC-0.2 | 0.721 | 0.725 | -0.004 |
| Quality | 英文→中文 | SLC-0.4 | 0.929 | 0.935 | -0.006 |
| Quality | 英文→中文 | UTMOS | 3.34 | 3.36 | -0.02 |
| Performance | 中文→英文 | AutoPCP | 2.43 | 2.91 | -0.48 |
| Performance | 英文→中文 | AutoPCP | 3.22 | 3.28 | -0.06 |
| Quality | 中文→英文 | AutoPCP | 2.31 | 2.89 | -0.58 |
| Quality | 英文→中文 | AutoPCP | 3.21 | 3.28 | -0.07 |

AutoPCP 第一次因为环境里没有 `stopes` 失败，装上 `stopes` 后已补完，结果在 `metrics/autopcp.json`。

## 3. 怎么读这些数字

英文→中文是明确变好的。Text-BLEU 高 9 到 14 个点，Speech-BLEU 高 7 到 12 个点，时长和 UTMOS 与 0.5B 基本持平。

中文→英文的文本也更好，Text-BLEU 高 10 到 13 个点。语音四项都比 0.5B 低，原因见第 3.1 节：大约五分之一的样本语音停不下来。

### 3.1 中文→英文 Speech-BLEU 偏低的原因

中文→英文有 2940 条 Performance、3266 条 Quality 样本（约 21% 和 23%）没有生成 `END_SEMANTIC`，一直跑到 `max_new_tokens=1500`。这些样本最后几百个 token 基本是同一个 semantic token 在重复（周期 1 的 3328 条，周期 3 的 1953 条），出现最多的是 7645 和 7239。参考语音里没有这种长尾，7645 在参考语音结尾的连续长度中位数是 0，所以这不是训练数据的时长问题，是推理时掉进了重复循环。

这批样本的文本翻译是正常的，坏的只是语音：

| 模式 | 子集 | 条数 | Text-BLEU | Speech-BLEU | UTMOS | AutoPCP | SLC-0.2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Performance | 正常结束 | 11317 | 45.42 | 33.49 | 3.75 | 3.12 | 0.664 |
| Performance | 停不下来 | 2940 | 42.57 | 0.29 | 2.57 | -0.24 | 0.000 |
| Quality | 正常结束 | 10991 | 48.93 | 36.62 | 3.75 | 3.13 | 0.669 |
| Quality | 停不下来 | 3266 | 48.90 | 0.30 | 2.72 | -0.46 | 0.000 |

Speech-BLEU 列里的条数比表中略少，因为 ASR 文本为空的样本不计分。正常结束的子集是挑过的，不能直接当成整体结果去和 0.5B 比，但它说明除了停止以外，翻译、音质、韵律都不比 0.5B 差。

同样的解码参数下，0.5B Phase3 的停止失败率是：中文→英文 Performance 5.0%、Quality 5.7%，英文→中文 0.7% 和 0.6%。3B 分别是 20.6%、22.9%、3.1%、3.1%。解码参数两边一样，所以这是 3B 这版训练在“何时停”上的退化，不是评测脚本的问题。

可能的原因，按可信度排序：

1. 停止监督不足。`END_SEMANTIC` 在每个样本里只有一个 token，前面是几百个 semantic token。0.5B 总共看过两遍 Quality/Performance（Phase2 一遍、Phase3 一遍），外加 Direct S2ST 这一类同样以 `END_SEMANTIC` 结尾的样本。3B 只在 Phase2 看过一遍 Quality/Performance，没有 Direct，也没有 Phase3。
2. 中文→英文的来源数据以 magicdata 短句为主，目标英文语音短，模型更容易在说完后停在静音 token 上出不来。

UTMOS 在中文→英文低大约 0.16，英文→中文几乎一样。音质下降同样主要来自停不下来的那批样本。

## 4. 指标文件

| 指标 | 文件 |
| --- | --- |
| Text-BLEU | `eval_outputs/qwen3b_phase2_unist198_iter_0010645_unist_test_vllm/metrics/text_bleu.json` |
| Speech-BLEU | 同目录 `metrics/speech_bleu.json` |
| SLC | 同目录 `metrics/slc.json` |
| UTMOS | 同目录 `metrics/utmos.json` |
| 逐条 ASR | 同目录 `metrics/asr_results.jsonl`，46712 条，其中 367 条 ASR 文本为空 |
| 生成结果 | 同目录上一级 `results.jsonl` |

Speech-BLEU 的 ASR 与 0.5B 相同：英文用 `openai/whisper-large-v3`，中文用 Paraformer `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`。UTMOS 是 `tarepan/SpeechMOS` 的 `utmos22_strong`。这些权重从团队 S3 前缀（`${S3_PROJECT_PREFIX}`，不入库）的 `evaluation_models/`、`cache/huggingface/`、`cache/modelscope/`、`cache/torch/hub/` 拉到了 `${USER_ROOT}` 下的同名目录。

## 5. AutoPCP

0.5B 用的是 Facebook `stopes` commit `a4e75e8`，比较器 `AutoPCP-multilingual-v2`，编码器 `facebook/wav2vec2-large-xlsr-53`。权重已经在：

```text
/opt/dlami/nvme/neuhao/evaluation_models/AutoPCP-multilingual-v2
/opt/dlami/nvme/neuhao/evaluation_models/wav2vec2-large-xlsr-53
```

`stopes` commit `a4e75e8` 已安装到训练环境。补跑已经开始，不要重跑 ASR。如果进程掉了，用下面这条命令从断点继续：

```bash
cd /opt/dlami/nvme/neuhao/UniSS
CUDA_VISIBLE_DEVICES=0 \
/opt/dlami/nvme/neuhao/conda_envs/uniss-train/bin/python -m evaluation.autopcp_metrics \
  --input eval_outputs/qwen3b_phase2_unist198_iter_0010645_unist_test_vllm/results.jsonl \
  --output-dir eval_outputs/qwen3b_phase2_unist198_iter_0010645_unist_test_vllm/metrics \
  --comparator-path /opt/dlami/nvme/neuhao/evaluation_models/AutoPCP-multilingual-v2 \
  --encoder-model /opt/dlami/nvme/neuhao/evaluation_models/wav2vec2-large-xlsr-53 \
  --device cuda:0 \
  --pick-layer 9 \
  --symmetrize \
  --batch-size 16 \
  --num-shards 1 \
  --shard-index 0
```

0.5B 的 AutoPCP 对照值：Performance 中文→英文 2.91、英文→中文 3.28；Quality 中文→英文 2.89、英文→中文 3.28。
