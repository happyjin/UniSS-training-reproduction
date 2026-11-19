# S5 流式推理验证

2026-11-18。验证使用 S4 checkpoint-93767 与 S5 checkpoint-10000。L40S 候选按 1 倍速输入，模型语义窗口为 10 秒，声学参考随已经消费的源音频滚动更新。全量和试听均显式关闭 VAD 分段与 30 秒硬切。

| 验证项 | 结果 | 详细记录 |
|---|---|---|
| 本次提交的状态机、PCM、Web 和评测入口回归 | 44 项通过。测试使用替身模型验证状态与协议 | [复现命令](#复现回归检查) |
| L40S 日韩 1596 句首包 | 五组各 1596 句已核对。1920 ms 候选实际 PCM 首包均值 6.735 秒、p95 12.963 秒，分别比冻结 640 ms 基线早 1.225 秒、1.963 秒 | [五组全量首包](S5_NO30_FULL_COMPARISON_REPORT.md#完整1596句实际首个pcm) |
| L40S 日韩 50 句同样本 | 五个子集各 10 句，当前五组均有译音。报告包含 ASR-BLEU、COMET 以及 CA、实际 PCM 的 StartOffset、EndOffset、LAAL-BOW／EOW | [日韩 50 句对比](S5_日韩50句同样本对比.md) |
| L40S 640 ms 上下文 A/B | 同一段 377.8 秒英译中，10 秒窗与 32/16 的 Thinker 重放为 574／35 次，逐块计算 p95 为 1.257／1.149 秒，实际 PCM 词首 LAAL 为 2.405／4.082 秒 | [长流窗口 A/B](S5_STREAMING_INFERENCE_OPTIMIZATION.md) |
| L40S Web 试听 | 英译日文件上传分别使用默认 1920 ms 与请求切换的 640 ms，两次均返回 4 块 PCM，返回的 chunk 与请求一致 | [推理与评测操作](S5_STREAMING_BENCH_RUNBOOK.md) |

冻结 640 ms 基线有 2 句无 PCM，640 ms 候选有 3 句无 PCM。2560 ms 候选有 1 句在产出首包后触发 Talker 收尾上限。对应样本 ID 和统计分母见全量报告。五组全量 CA、实际 PCM 词级 LAAL 和 1596 句质量仍待补齐。[全量结果与失败说明](S5_NO30_FULL_COMPARISON_REPORT.md)

## GPU 数据对应的版本

- 1596 句首包、50 句对比和 Web 试听使用冻结候选快照，源码包 SHA-256 为 `6e2b36164454a9836055ec098ebd943feb930366ba98c88c052809788c5e4973`。本次提交在该快照基础上增加了显式的 32/16 上下文选项。
- 10 秒窗与 32/16 的 A/B 快照中，文件推理入口、Web 入口和页面、会话、分段器及 PCM 解码器与本次提交逐文件一致。源码包 SHA-256 为 `1d7348492a446de30abaa460b0df4c3864079e7612182d6f7fd07db728419775`。20/16 的数字来自独立实验快照，当前可用的窗口参数是秒数窗口和 `--context_window_32_16`。
- Qwen、OpenAI 和旧 S5 的历史值沿用原日韩报告，50 句子集使用其原始逐句记录重算。外部墙钟包含网络和外部服务耗时。新版实际 PCM 时间线使用本地服务端产出时刻。[原日韩同传对比](日韩同传对比_Qwen_OpenAI_我们.md)

## 复现回归检查

在仓库根目录执行以下命令，Python 环境需有项目的推理依赖、pytest 和 Flask。

```bash
PYTHONPATH=custom python -m pytest -q \
  tests/test_s5_session.py \
  test_script/streaming_si/test_s5_pcm_decoder.py \
  test_script/demo/test_web_demo_s5.py \
  custom/streaming_si/eval/tests/test_s5_streaming_bench.py
```

GPU 权重、WAV、逐句 JSONL 和监控日志存放在本地实验目录。跑测入口、配置和报告口径见 [S5 跑测操作](S5_STREAMING_BENCH_RUNBOOK.md)。
