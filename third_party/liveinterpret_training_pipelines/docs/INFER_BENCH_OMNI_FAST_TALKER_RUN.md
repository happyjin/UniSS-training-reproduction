# L40S L4 FAST_TALKER 对照跑测

目标：保持 L4、thinker beam 4、8 步 BF16 和 1193 句串行口径，验证 talker penalty/ngram 向量化能否把含 I/O RTF 从 0.659 降至 0.6 以下。尚未实测。

## 环境

复用之前 L40S 跑测 Pod 的镜像与挂载：

- EKS context：`arn:aws:eks:ap-southeast-3:124355679795:cluster/speech-model-data-eks`
- namespace：`curion`；service account：`curion-runtime`
- PVC：`speech-model-data-eks-s3-data-pvc`
- 单张 L40S 独占，沿用原跑测的 CPU/内存配额；文档记录 requests 3 CPU / 20Gi，`OMP_NUM_THREADS=3`。
- 代码必须包含当前 `patch_omni_repo.py`：ngram 的 `_win` 要截取到 `[: _seq.numel() - _n + 1]`，避免匹配末尾 prefix 后越界。

本地没有 GPU/模型。首次访问 EKS 被网络限制拦截，放开网络后已提交并完成推理和质量打分，结果见下文。仓库通用 `ktrainjob_infer_bench.yaml` 是 H200 配置，不能直接拿来提交 L40S。

## 每轮的 Pod 内命令

每轮使用全新 Pod。`bench_entry.sh` 固定在 `/tmp/omni_repo_local` 打补丁，复用 Pod 后将 FAST_TALKER 设为 0 不会撤销补丁；profile 补丁也可能残留。

进入已挂载资产、包含当前代码的原跑测镜像后，先设置下面四项。路径取 Pod 的真实挂载位置；ASSETS_ROOT 应同时包含 `eval/` 与 `tts_bench/`。

```bash
cd /path/to/liveinterpret_training_pipelines
export ASSETS_ROOT=/actual/pvc/mount/junyaoyang/infer_bench_assets
export RUN_ID="$(date +%Y%m%d-%H%M%S)-l40s-fast-l4-smoke"
export OUT="/tmp/infer_bench/${RUN_ID}"
export PUBLISH_TO="/actual/pvc/mount/junyaoyang/infer_bench/${RUN_ID}"

test -f "$ASSETS_ROOT/eval/work/manifests/all.jsonl" || exit 1
test ! -e /tmp/omni_repo_local || {
  echo '请使用全新 Pod，避免已打补丁的仓库污染对照' >&2
  exit 1
}

unset SIMULS2ST_PROFILE SIMULS2ST_PROFILE_TALKER
unset SIMULS2ST_FM_CFG SIMULS2ST_FM_RESCALE_CFG
unset RESUME_FROM BENCH_REEXEC BENCH_ORIG_SCRIPT
export SYSTEM=rep_L4 GPU_TAG=l40s BENCH_EPOCHS=1 THINKER_BEAMS=4
export OMP_NUM_THREADS=3 LAT_BATCH=4
export SIMULS2ST_FM_STEPS=8 SIMULS2ST_VOICEBOX_DTYPE=bf16
export SIMULS2ST_FAST_TALKER=1
export LIMIT=30 STOP_AFTER=infer
bash pipelines/streaming_si/bench_entry.sh
```

`LIMIT=30` 只用于检查补丁锚点、越界、空输出和耗时趋势；它取清单前 30 句，不保证覆盖两个方向。不要用冒烟 RTF 判断是否达标。`STOP_AFTER=infer` 不生成最终 summary。

## 正式对照

冒烟通过后，分别在全新 Pod 上执行上面的命令，各跑一轮：

| 轮次 | RUN_ID 后缀 | FAST_TALKER | LIMIT / STOP_AFTER |
|---|---|---|---|
| A：8 步 BF16 基线 | `l40s-bf16-8-l4-control` | `0` | 都 unset |
| B：向量化 | `l40s-bf16-8-l4-fast` | `1` | 都 unset |

正式命令前使用 `unset LIMIT STOP_AFTER`，并给每轮独立的 RUN_ID / OUT / PUBLISH_TO。保持镜像、CPU 配额、线程数、资产路径、缓存条件和 GPU 独占一致；不要启用 profiler。优先 A/B 串行，以免同时读 S3 互相干扰。

全量执行包含推理、合并和打分；无需改 bench_entry.sh。结束后查看各轮 `summary.md`、`status.log` 和 `logs/infer.1.*.log`。

## 验收

1. B 的日志应出现 `[patch] fast talker`，且实际有手工 talker 解码执行；仅出现补丁消息不能证明热点被覆盖。
2. 最终保留 1193 句；两个方向正常完成，无新增失败、空译文、空语音。
3. 按 ID 对齐 `infer/merged.jsonl` 比较文本；如有差异，定位首个不同句子的 semantic codes。当前标准产物不保证保存 codes，需要另做诊断才能确认 codes 一致性。
4. 比较 ASR-BLEU、Text-BLEU、UTMOS、SIM-o 和 LAAL；语音随机性不能直接证明所有质量差异都属于噪声。
5. 达标看 summary 的 `推理总墙钟 / 源音频秒数 < 0.6`，保留模型加载、源音频读取、wav 写入和两条预热的原有口径。142 分钟音频对应墙钟需低于约 5112 秒；精确阈值以本轮源音频秒数为准。
6. 若接近阈值，重复 A/B 确认缓存与调度波动。最终发布拷贝原本在推理计时外，不计入这里的 RTF。

若没有足够收益，再对小样本单独启用 `SIMULS2ST_PROFILE=1` / `SIMULS2ST_PROFILE_TALKER=1`，定位前向与 logits 后处理占比。诊断轮使用新 Pod、新输出目录，其耗时不作为性能成绩。

## 全量结果与下一步（2026-10-29）

Pod：`curion/bench-fast-l4-full-20261028-205204`。产物发布位置：PVC 内 `junyaoyang/infer_bench/20261028-205204-l40s-fast-l4-full`。检查时打分、summary 已完成，Pod 正在发布产物。

- 1193 句，源音频 8518.919 秒，零失败、零空译文、零空语音。
- 推理墙钟 5453 秒，含 I/O RTF **0.640105**；句内墙钟 4495.2238 秒，RTF **0.527675**。
- ASR-BLEU：RealSI en→zh / zh→en、ACL 分别 26.14 / 19.65 / 42.74；Text-BLEU 28.04 / 20.06 / 45.20。
- 相比文档的 8 步 BF16 历史结果 0.659，墙钟 RTF 降约 2.9%。这不是同批 A/B，不能把全部差值归因于 FAST_TALKER。
- 0.6 的精确墙钟阈值是 5111.3514 秒，需再节省超过 341.6486 秒。

结合 Kubernetes 主日志时间戳和方向日志，分解如下（UTC；时间取整存在秒级误差）：

| 阶段 | en2zh | zh2en |
|---|---|---|
| 启动方向进程 | 21:51:44.809 | 22:48:26.790 |
| 首条 tokenizer 加载日志 | 21:55:40 | 22:52:26 |
| 开始评测 | 21:57:41 | 22:54:31 |
| 启动至开始评测 | 约 356 秒 | 约 364 秒 |
| tokenizer 日志至开始评测 | 121 秒 | 125 秒 |
| tqdm 评测总时长 | 48:19 | 26:42 |
| 方向进程退出 | 22:48:26.676 | 23:22:37.187 |

两次评测约 4501 秒，与剔除预热后的句内 4495 秒接近；因此剩余约 958 秒不能称为 wav 写入时间。其中约 720 秒在开始评测之前，约 232 秒落在评测结束之后及其他边界处理，具体责任函数仍需细分计时。首条 tokenizer 日志前的约 235 / 239 秒包括导入、processor 初始化等，不能仅凭日志断言全部来自 S3 Python 导入。

优先方案：单进程跑两个方向，共享一份 `OmniTalkerSemanticTTSInference`。每个方向重新创建 agent / evaluator，重新生成语言 prompt，独立重置句子状态与 KV cache，保留两条预热和原墙钟起止点。模型构造参数不包含语言，因此可以尝试复用同一加载结果；还需审查推理对象上的可变字段，验证换方向后无缓存污染。仅消除第二次约 364 秒的启动，按当前数据估算 RTF 约为 **0.597**，余量仅约 22 秒，应再留出波动余量。进程退出后处理若也能减少，可能进一步受益，但尚未量化验证。

其次：给 wrapper 的 import、processor、模型各组件、evaluator、退出边界加 `perf_counter` 标记，再决定是否将 `site_omni` 等依赖缓存到本地。缓存复制若属于本轮准备，需额外报告耗时，不能通过移出计时掩盖冷启动开销。降低 beam、CFG 或反扩散步数属于质量相关实验，暂不优先。

## 共享模型实验

新增 `custom/streaming_si/eval/omni_shared_run.py`，通过 `SIMULS2ST_SHARED_MODEL=1` 打开；默认仍用原双进程入口。共用同一个模型加载结果，每方向新建 agent/evaluator，保留每方向预热、原解码参数、原墙钟起止点。加载参数不一致时拒绝复用；每方向结束后 reset 状态并释放 agent/evaluator。

双方向冒烟：`bench-shared-l4-smoke-20261029-004249`，每方向 3 句 + 1 预热，两个方向均输出 4/4，进程 rc=0，推理墙钟 385 秒。

- wrapper 导入与 Stage 2 钩子安装：207.374 秒。
- 第二方向日志明确显示 `[shared-model] reuse loaded inference`。
- 第二方向 agent 构建 0.012 秒、evaluator 构建 0.719 秒。
- en2zh / zh2en 的评测耗时分别 10.359 / 8.028 秒。
- Python 主体结束时间 382.966 秒，与 shell 墙钟 385 秒接近。

全量 Pod：`curion/bench-shared-l4-full-20261029-004840`。使用当前脚本 ConfigMap 快照；启动前将冒烟的 6 条非预热译文按 ID 与上一轮全量对齐，若存在空文本或文本不一致则退出，不进入全量。全量保留 1193 句及全部质量打分。

全量结果已确认：推理墙钟 **5044s**，含 I/O RTF **0.592094**，句内 RTF **0.524099**。
1193 句零失败、零空译文、零空语音，所有 ID 及文本与上轮全量完全一致。
ASR-BLEU 为 26.50 / 19.70 / 42.62，全部打分已成功，正在发布产物。
第二方向 agent 构建 0.013s；首次导入与钩子安装 184.289s，模型加载 152.961s；
evaluator 构建分别 145.757 / 84.498s。相比上轮省 409 秒，距离 0.6 阈值约有 67 秒余量，尚未重复跑验证稳定性。
