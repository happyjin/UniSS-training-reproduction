# 原 S5 报告质量协议

`SOURCE.json`记录从Git提交`dfe4f887509131387315801943ce53f9fb24213e`恢复的原文件及SHA256。四个恢复文件保持原样，`run.py`负责接入当前实际PCM输出，启动时校验这些文件。该提交可恢复原库，但原报告没有记录11月7日运行环境的完整版本，不能声称执行环境逐字节一致。

- 中文目标：Qwen3-ASR-1.7B，英文目标：Whisper-large-v3；时间戳仍由Whisper生成。
- 日语/韩语目标也使用原库已有Qwen3-ASR映射；BLEU分别使用原库的ja-mecab/ko-mecab。编排器已放开四语目标，日韩缺少MeCab时直接报错，禁止悄悄回退char。安装`sacrebleu[ja,ko]==2.5.1`；语言对按manifest实际源文、目标参考和数据集拆分，不从中英成绩外推。
- 原OpenCC t2s、文本归一化、SacreBLEU中文zh/英文13a；原COMET使用原始src/mt/ref，batch32。
- manifest中的失败或缺失结果补为空假设，保留评分分母。非空音频必须有实际PCM事件，检查offset连续和输出时长。
- 原评分器保留生成时间戳的能力；新编排器只发布质量，不把加速文件输入的LAAL当作实时延迟。
- ASR/BLEU/COMET完成仍不代表音质、说话人相似度、流利度、边界听感或完整原报告全部完成。

模型推理进程退出、GPU释放后运行。ASR解释器需原`requirements-asr.txt`及Qwen依赖；COMET环境可独立配置，避免污染推理环境。所有模型运算在GPU上执行。

```bash
"$ASR_PYTHON" custom/streaming_si/eval/report_protocol/run.py \
  --manifest "$MANIFEST" --results "$RESULTS" --events "$PCM_EVENTS" \
  --whisper_model "$WHISPER_MODEL" --qwen_model "$QWEN_MODEL" \
  --comet_python "$COMET_PYTHON" --site_comet "$SITE_COMET" \
  --comet_model "$COMET_CHECKPOINT" --out "$NEW_OUTPUT_DIR"
```

`--skip_comet`仅用于明确标记的ASR诊断；正式对照需要COMET。离线对照也写实际PCM事件，但整段文件只在合成完成后作为单块可用，不能用token摊派伪造发包时刻。模型、样本、chunk、seed、采样、重试策略均须锁定并记录，报告原来的1920ms与新640ms结果不能直接归因于滑窗。

A10实测：原默认ASR batch16在Whisper词时间戳阶段OOM；失败记录保留且不发布质量分。小显存验证可显式加入`--asr_batch 1`，两组对照必须相同并记录与原batch16的差异；ASR模型、精度和解码参数保持原库设置。该batch调整仍需在结果中说明，不能称执行配置完全一致。ASR日志出现原库的“单条失败”时编排器直接判失败，防止空转写被误报为有效成绩。
