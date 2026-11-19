# Omni 与 S4/S5 推理入口导航

本页说明跑测实际启动哪些文件；指标以四份 `INFER_BENCH_*.md` 为准。
这里的 **S4 streaming 在仓库中叫 S5-streaming**：S4 基座加 S5 流式 adapter；`SYSTEM=s4` 是整句离线推理。

## 统一入口与四条链路

统一启动脚本：[bench_entry.sh](../pipelines/streaming_si/bench_entry.sh)，负责环境检查、清单、推理、合并、打分与发布。

| 系统 | SYSTEM | Python 推理入口 | 权重组成 |
|---|---|---|---|
| Omni offline | `rep_off` | [run_omni_offline.py](../custom/streaming_si/eval/run_omni_offline.py) | 官方 offline 基座 + 复现 Stage 2 |
| Omni streaming，L1～L4 | `rep_L1`～`rep_L4` | 共享模式：[omni_shared_run.py](../custom/streaming_si/eval/omni_shared_run.py) | 官方 offline 基座 + 合并复现 Stage 2 + 复现 Stage 3v2 |
| S4 offline | `s4` | [infer_s2st.py](../custom/streaming_si/inference/infer_s2st.py) | S4 导出模型，默认资产 `models/s4_ck` |
| S5 streaming | `s5` | [infer_s5_streaming.py](../custom/streaming_si/inference/infer_s5_streaming.py) | S4 导出模型 + `models/s5_adapter` |

Omni 的 L 是 WAIT/WRITE 决策档位，四档共用权重。

## Omni offline

```text
bench_entry.sh：SYSTEM=rep_off
  → custom/streaming_si/eval/run_omni_offline.py
    → 官方 src/inference/run_offline.py：load_audio
    → 官方 src/inference/infer_omni_talker_tts.py
      → OmniTalkerSemanticTTSInference.from_pretrained(...)
      → e2e(...)：译文、semantic codes、目标波形
```

模型只加载一次，循环输入清单。结果为 `infer/results.00.jsonl`，波形在 `infer/wav/`。
该跑测挂复现 Stage 2，不挂 streaming 的 Stage 3v2。

## Omni streaming

```text
bench_entry.sh：SYSTEM=rep_L1～rep_L4
  ├─ SIMULS2ST_SHARED_MODEL=1 → omni_shared_run.py
  │   单进程；每方向新建 agent/evaluator，共享 inference 模型
  └─ 未开启共享模型 → 每方向单独启动 omni_simuleval_wrap.py

共同使用：
  omni_simuleval_wrap.py：补 elapsed / wall_sec，安装 Stage 2 合并钩子
  omni_merge.py：先合 Stage 2，再加载 Stage 3v2
  官方 src/agents/simuleval_omni_talker_s2st_agent.py：WAIT/WRITE、分块和历史状态
  官方 src/inference/infer_omni_talker_tts.py：thinker / talker 推理
```

| 辅助文件 | 作用 |
|---|---|
| [omni_simuleval_wrap.py](../custom/streaming_si/eval/omni_simuleval_wrap.py) | SimulEval 计时包装、原双进程入口 |
| [omni_merge.py](../custom/streaming_si/eval/omni_merge.py) | Stage 2 → Stage 3 权重加载钩子 |
| [patch_omni_repo.py](../pipelines/streaming_si/patch_omni_repo.py) | 为官方仓库本地副本安装运行期补丁 |
| [omni_talker_accel.py](../pipelines/streaming_si/omni_talker_accel.py) | 静态 KV、CUDA Graph、容量分档、超限回退 |
| [omni_graph_probe.py](../custom/streaming_si/eval/omni_graph_probe.py) | eager/static/graph 探针；设置 `SIMULS2ST_PROBE_MODES` 时替换共享入口 |

结果在 `infer/run_en2zh/` 和 `infer/run_zh2en/`：`instances.log`、`text_predictions.jsonl`、`wavs/`。
共享日志为 `logs/infer.1.shared.log`；双进程为 `logs/infer.1.en2zh.log` / `infer.1.zh2en.log`。

**官方资产与本库实现的区别**：上述 `src.*` 来自 `${SRC}/code/SimulS2ST-Omni-main`，
不等于本库 `custom/simuls2st_omni/` 的训练/迁移实现。只改后者不会自动改变官方跑测。

设置 `ASSETS_ROOT` 时 `${SRC}=${ASSETS_ROOT}/eval`，否则为 `${FSX_MOUNT:-/mnt/fsx}/stevehguo/eval`。
A10/L40S 跑法把官方仓库暂存到 `/tmp/omni_repo_local` 后打补丁，helper 复制到副本的 `src/inference/`；原资产不修改。
当前补丁安装位于设置 `ASSETS_ROOT` 的暂存分支内；每轮使用新 Pod 或干净副本，避免旧补丁残留。

## S4 offline 与 S5 streaming

```text
bench_entry.sh：SYSTEM=s4
  → infer_s2st.py → S2STPipeline：整句源语音 → 译文 / speech tokens → 波形

bench_entry.sh：SYSTEM=s5
  → infer_s5_streaming.py → S4 模型 + S5 adapter
    → ThinkerRunner：640ms 分块，WAIT/WRITE，每次 WRITE 生成语音
```

共用组件：[modeling_streaming_si.py](../custom/streaming_si/model/modeling_streaming_si.py)（HF 包装）、
[model_si.py](../custom/streaming_si/model/model_si.py)（模型、TalkerStreamState、TalkerDecodeGraph）、
[source_prompt.py](../custom/streaming_si/inference/source_prompt.py)（参考音频）、
[speech_decoder.py](../custom/streaming_si/model/speech_decoder.py)（decoder 资产）。
S4/S5 的 TalkerDecodeGraph 与 Omni helper 面向不同模型，不能直接互换。

## 运行方式与结果

在已准备好的 GPU 镜像、模型和测试资产环境中，从仓库根目录运行。路径变量见各跑测文档；S3 PVC 场景 `OUT` 要放本地盘，再发布到 `PUBLISH_TO`。

```bash
SYSTEM=rep_off bash pipelines/streaming_si/bench_entry.sh
SYSTEM=s4 bash pipelines/streaming_si/bench_entry.sh
SYSTEM=s5 bash pipelines/streaming_si/bench_entry.sh

# Omni streaming 当前优化组合；L 可换档，L1 尚未全量验证该组合
SYSTEM=rep_L4 SIMULS2ST_FM_STEPS=8 SIMULS2ST_VOICEBOX_DTYPE=bf16 \
SIMULS2ST_FAST_TALKER=1 SIMULS2ST_SHARED_MODEL=1 SIMULS2ST_TALKER_ACCEL=graph \
bash pipelines/streaming_si/bench_entry.sh
```

`STOP_AFTER=infer` 跳过合并/打分/最终汇总；`LIMIT=N` 取总清单前 N 条，不保证覆盖双方向。
共享模式的双方向探针使用 `SHARED_LIMIT_PER_DIRECTION=N`，配合 `STOP_AFTER=infer`。

后处理：[stream_segments.py](../custom/streaming_si/eval/stream_segments.py)、
[run_asr.py](../custom/streaming_si/eval/run_asr.py)、[run_utmos.py](../custom/streaming_si/eval/run_utmos.py)、
[run_para_metrics.py](../custom/streaming_si/eval/run_para_metrics.py)、
[latency_eval.py](../custom/streaming_si/eval/latency_eval.py)、[score_eval.py](../custom/streaming_si/eval/score_eval.py)。
最终看 `summary.md`、`status.log`、`report/*.json`。
含 I/O RTF 包含推理阶段加载、源音频读取、写 wav 和预热；不包含后续打分与最终发布拷贝。

- [Omni offline 跑测](INFER_BENCH_REPRO_OMNI_OFFLINE.md)
- [Omni streaming 跑测与优化](INFER_BENCH_REPRO_OMNI_STREAMING.md)
- [S4 offline 跑测](INFER_BENCH_S4_OFFLINE.md)
- [S5 streaming 跑测](INFER_BENCH_S5_STREAMING.md)
- [共享模型与 FAST_TALKER 实验](INFER_BENCH_OMNI_FAST_TALKER_RUN.md)
- [静态 cache / Graph 实验](INFER_BENCH_OMNI_STATIC_GRAPH_PROBE.md)

根目录 [inference/](../inference/README.md) 是另一套 `qwen_omnispeak` 模型的 CLI，不是本页四项跑测的入口。
