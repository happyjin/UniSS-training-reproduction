# 05 · Indexed-WDS + EOS/History 训练配方

公开入口：

`pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh`

该入口用于从已有模型 checkpoint 权重 warm-start，并以新版 indexed-WDS 数据继续训练
reply-add Thinker+Talker。它组合了四项必要更新：

1. 根据 `target_lang` 统一生成训练时的标准 S2S system prompt；
2. 从 `wds://...?...offset=...&length=...&format=...` 引用按 byte range 读取源音频；
3. main-codec stop/EOS 标签使用 10× CE 权重；
4. 训练阶段以 5% 概率扰动 teacher-forced codec 历史，eval/inference 不扰动。

public 版不内置周期 AR rollout。常规 validation 仍记录 text/main-codec/acoustic-token
accuracy，以及 EOS boundary accuracy 和 premature EOS rate；完整 AR 生成在训练后用独立
evaluation 工具执行。

## 1. 数据要求

调用方提供两个目录：

- `WDS_MANIFEST_DIR`：包含 `manifest-*.jsonl` 和完成标记 `_COMMITTED.json`；

WDS 音频 tar shards 通过容器 volumeMount 映射（pytorchjob.yaml），
`wds://` URI 中的 netloc 直接对应容器内挂载路径（如 `wds://curated_audio/...` → `/curated_audio/...`）。

每条 manifest 至少应提供训练所需的 source/target 文本、`source_lang`、`target_lang`、
目标 codec，以及一个 indexed-WDS 源音频引用。典型引用格式：

```text
wds://<logical-prefix>/process_version=<version>/shard-00000.tar?offset=123&length=456&format=wav
```

resolver 不解析完整 tar，也不会先执行 `torch.load`；它定位本地 shard 后只读取指定
byte range，将音频 bytes 交给现有解码器。manifest 不需要 `source_xvector`，speaker
embedding 在模型 forward 中由源音频在线提取。

## 2. 启动与预检

先只运行预检：

```bash
MODEL_PATH=/path/to/checkpoint \
WDS_MANIFEST_DIR=/path/to/manifests \
PYTHON_BIN=/path/to/training-env/python \
PREFLIGHT_ONLY=true \
  bash pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh
```

预检会确认：

- checkpoint 和 manifest 目录存在；
- 使用 whole talker + unfrozen thinker；
- EOS 权重为 10、history perturbation 为 0.05；
- manifest 的中英两个方向都映射为正确的标准 prompt；
- 没有误用 `RESUME_FROM` 恢复旧 optimizer/scheduler。

预检通过后移除 `PREFLIGHT_ONLY=true` 即可启动训练：

```bash
MODEL_PATH=/path/to/checkpoint \
WDS_MANIFEST_DIR=/path/to/manifests \
PYTHON_BIN=/path/to/training-env/python \
NPROC_PER_NODE=8 \
  bash pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh
```

`MODEL_PATH` 是权重 warm-start；脚本有意 unset `RESUME_FROM`，因此 optimizer、scheduler
和 dataloader 状态从新训练任务开始。`OUTPUT_DIR`、batch size、GPU 数、cache、端口、
wandb 名称等都可通过同名环境变量覆盖。

## 3. 兼容旧数据

底层 resolver 仍支持以下模式：

- `WDS_AUDIO_MODE=off`：只接受普通本地音频路径；
- `WDS_AUDIO_MODE=auto`：普通路径直通，`wds://` 引用走 byte-range resolver；
- `WDS_AUDIO_MODE=required`：只接受 `wds://`，新版入口默认使用此模式并 fail fast。

因此旧数据可以继续调用
`train_speech_to_speech_translation_upstream.sh`，或显式设置 `WDS_AUDIO_MODE=auto`；不需要
为了支持 WDS 修改 model/template 配置。

## 4. 验证命令

```bash
bash -n pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh

python -m unittest \
  test_script.test_speaker_encoder_short_audio \
  test_script.test_s2s_prompt_pipeline

PYTHONPATH=. python test_script/wds_resolver/test_wds_audio.py
```

测试必须使用已安装项目依赖（PyTorch、ms-swift 等）的训练环境 Python。
