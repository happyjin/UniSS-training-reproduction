# UniST train/dev streaming regression

试听时优先比较每个子目录中的 `source.wav`、`reference_target.wav`、
`offline_phase3.wav`。安全模式下没有生成 `streaming_translation.wav`；
`streaming_stereo.wav` 的右声道为空，表示质量门拒绝了不安全输出。

## 结论

当前 pilot15 **在训练域和 dev validation 域都不是正常 streaming**：8/8 样本的
`support_bucket` 始终为0，8/8 没有自然 WRITE，安全模式下8/8都没有可播放的
streaming翻译语音。这个结果排除了“只对用户上传的外部音频失败”的解释。

Phase3-v4 offline 对照在相同8条 BiCodec 重建源音频上8/8都生成了可播放语音，
其中 `train_en_zh_01` 的文本相似度为0.978，`dev_zh_en_02`为1.000。
因此源数据、BiCodec解码、Phase3基础模型均可工作，失败集中在pilot15的
streaming policy、causal frontend和semantic训练监督。

## 推荐试听顺序

1. [训练集英→中源音频](train_en_zh_01/source.wav)
2. [该条数据集目标语音](train_en_zh_01/reference_target.wav)
3. [该条Phase3 offline输出](train_en_zh_01/offline_phase3.wav)
4. [dev中→英源音频](dev_zh_en_02/source.wav)
5. [该条数据集目标语音](dev_zh_en_02/reference_target.wav)
6. [该条Phase3 offline输出](dev_zh_en_02/offline_phase3.wav)

为了能够听到被安全门阻止的旧行为，另生成了明确标注为不安全诊断的
[UNSAFE_FORCED_REPORT.md](UNSAFE_FORCED_REPORT.md)。这些输出最多只有0.24秒，
覆盖率仅3.8%–7.4%，部分RMS接近零，不能当作正常同传结果。

| sample | split | direction | streaming text | natural/forced | coverage | gate | offline text similarity |
|---|---|---|---|---:|---:|---|---:|
| [train_en_zh_01](train_en_zh_01/) | train | 英文 → 中文 | (empty) | 0/1 | 0.0% | FAIL | 0.978 |
| [train_en_zh_02](train_en_zh_02/) | train | 英文 → 中文 | (empty) | 0/2 | 0.0% | FAIL | 0.679 |
| [train_zh_en_01](train_zh_en_01/) | train | 中文 → 英文 | (empty) | 0/2 | 0.0% | FAIL | 0.474 |
| [train_zh_en_02](train_zh_en_02/) | train | 中文 → 英文 | (empty) | 0/2 | 0.0% | FAIL | 0.440 |
| [dev_en_zh_01](dev_en_zh_01/) | dev | 英文 → 中文 | (empty) | 0/1 | 0.0% | FAIL | 0.667 |
| [dev_en_zh_02](dev_en_zh_02/) | dev | 英文 → 中文 | (empty) | 0/2 | 0.0% | FAIL | 0.545 |
| [dev_zh_en_01](dev_zh_en_01/) | dev | 中文 → 英文 | (empty) | 0/2 | 0.0% | FAIL | 0.841 |
| [dev_zh_en_02](dev_zh_en_02/) | dev | 中文 → 英文 | (empty) | 0/1 | 0.0% | FAIL | 1.000 |

## Aggregate

- Samples: 8
- Streaming quality passed: 0/8
- Streaming with natural WRITE: 0/8
- Streaming with playable audio: 0/8
- Offline Phase3 with playable audio: 8/8
