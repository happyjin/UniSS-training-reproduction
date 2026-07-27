# Simul-UniSS full198 Stage3 / Stage4 / Stage6 simultaneous-streaming 评估计划

> 文档日期：2026-07-27 UTC  
> 适用范围：当前已完成的 full198 Stage3、Stage4、Stage6 及其后续真实流式组件接入验证  
> 目标：建立同时覆盖翻译质量、流式策略、墙钟延迟、语音连续性、音色保持和实时性的可复现 dev/eval 协议  
> 原则：不覆盖历史 checkpoint、数据、日志或评估输出；所有新产物进入独立、带版本和时间戳的目录

## 1. 执行摘要

当前 full198 已经完成：

| Stage | 训练内容 | 最终 iteration | 当前可评估范围 |
| --- | --- | ---: | --- |
| Stage3 | WAIT/WRITE action-only SFT | 4753 | action logits、WAIT/WRITE 行为、premature WRITE、unnecessary WAIT、final flush |
| Stage4 | phrase-level interleaved S2ST SFT | 4753 | WAIT/WRITE、目标文本短语、BiCodec semantic token、离线/伪流式生成 |
| Stage6 | 低学习率 interleaved refinement | 1189 | 与 Stage4 相同，重点验证低学习率 refinement 是否改善 Pareto 前沿且未破坏离线能力 |

精确 checkpoint：

```text
checkpoints/simul_uniss_v7_full198_seq18000_mbs2_gbs128_stage3/
  stage03_action_sft/iter_0004753

checkpoints/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/
  stage04_interleaved_s2st/iter_0004753
  stage06_joint_refinement/iter_0001189
```

当前评估必须保留以下边界说明：

1. 当前 full198 Stage3/4/6 使用 `pseudo_proportional_token_alignment`，不是由真实词时间戳或 full198 streaming audio student/CTC heads 产生的因果边界。
2. 当前所谓 Stage6 是 Qwen 侧低学习率 interleaved refinement；full198 Stage1/2/5 尚未接入，所以不能称为完整的 student + CTC + Qwen + streaming BiCodec 联合 Stage6。
3. 当前代码已有 schedule proxy 评估和 oracle schedule replay，但尚没有把 Stage3/4/6 Qwen checkpoint 连接到 `StreamingController` 的完整自由运行 adapter。
4. 因此结果必须分为 `teacher_forced_proxy`、`free_running_pseudo_streaming` 和 `real_audio_wall_clock_streaming` 三个层级，禁止混为一个“端到端 simultaneous”结果。

建议最终采用如下评估主线：

```text
结构与权重检查
→ teacher-forced action/token 评估
→ free-running Simul-S2TT
→ free-running semantic-token streaming
→ BiCodec waveform streaming
→ computation-aware 墙钟评估
→ dev 选择 Pareto 配置
→ test/eval 一次性冻结评估
→ 生成报告、曲线和试听目录
```

## 2. 文献与官方代码调研结论

### 2.1 SimulS2S-LLM

论文：

- [SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation](https://arxiv.org/abs/2504.15509)
- ACL 2025 DOI：[10.18653/v1/2025.acl-long.817](https://doi.org/10.18653/v1/2025.acl-long.817)

方法关键点：

- 使用 streaming acoustic encoder 和 CIF 获得接近文本粒度的 boundary-aware speech prompts。
- LLM 仍以 offline 方式训练，推理时使用 test-time wait-k 解锁 simultaneous 能力。
- speech generator 使用多层 LLM hidden states，并以 causal Transformer + CTC 预测离散语音 token。
- 使用 incremental beam search 扩大每个 chunk 的语音 token 搜索空间，同时避免额外等待完整输入。
- encoder chunk 为 32 个 frame，论文口径对应理论平均 320 ms。
- CVSS-C 评估语言为 Es→En、Fr→En、De→En。

论文评估协议：

- Simul-S2ST 质量：ASR-BLEU，生成语音先经 ASR，再计算 SacreBLEU。
- Simul-S2ST 延迟：ATD；附录同时报告 StartOffset、EndOffset 和 computation-aware 版本。
- Simul-S2TT：BLEU + speech word-level AL。
- 语义/语音质量：BLASER 2.0 Unsupervised、QE、Ref。
- 形成 wait-k quality-latency 曲线，而不是只报告一个 operating point。
- computation-aware latency 单独报告，不能只报理论策略延迟。

论文数值参考，仅用于同一 CVSS-C 语言和同一口径复现后比较：

| 方向 | wait-k | ASR-BLEU | ATD | StartOffset | EndOffset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Es→En | 5 | 23.2 | 2533 ms | 3109 ms | 1722 ms |
| Es→En | 8 | 26.3 | 3440 ms | 4191 ms | 2209 ms |
| Fr→En | 5 | 22.5 | 2103 ms | 2659 ms | 1659 ms |
| Fr→En | 8 | 26.9 | 2677 ms | 3378 ms | 1914 ms |
| De→En | 11 | 18.4 | 2819 ms | 3855 ms | 2182 ms |
| De→En | 17 | 21.6 | 3684 ms | 4931 ms | 2838 ms |

对 Simul-UniSS 的启示：

- 必须同时保留 boundary-aware 与 boundary-unaware/pseudo schedule 的对照。
- 当前 Stage3/4/6 最重要的第一组对照是：固定 schedule、模型自由 action、固定 wait-k 三种策略。
- 必须画 quality-latency Pareto 曲线，单独一个 640 ms、wait-k=2 配置不能证明 simultaneous 方法有效。
- Stage4/6 应比较 greedy 与有限 beam/search，但搜索计算必须计入 computation-aware latency。

论文未给出独立的官方训练仓库；实现说明基于 ESPnet-ST 和 SimulEval。因此本计划引用其论文协议，但不假定存在可直接复用的官方 SimulS2S-LLM evaluation script。

### 2.2 Hibiki：High-Fidelity Simultaneous S2ST

论文与代码：

- [High-Fidelity Simultaneous Speech-to-Speech Translation](https://arxiv.org/abs/2502.03382)
- [kyutai-labs/hibiki](https://github.com/kyutai-labs/hibiki)
- 调研 commit：`f1cf9293e35c1dceffbe60dd325bdd702bc8305e`

方法关键点：

- 基于 Moshi multistream decoder-only 架构，同时建模 source audio、target audio 和 target text。
- audio/text 都按固定 frame rate 连续生成；README 给出的输出帧率为 12.5 Hz。
- contextual alignment 根据“目标词何时能被源前缀预测”确定安全输出时间。
- 训练数据通过 silence insertion 或 alignment-aware TTS 构造。
- 使用 speaker-similarity conditioning 和 classifier-free guidance 控制 voice transfer。
- 2.2B Temporal Transformer + Depth Transformer，完整模型约 2.7B；40 秒上下文。
- 训练序列最长约 120 秒，但推理上下文使用滑动/局部上下文。

论文评估协议：

- short-form：CVSS-C Fr→En test，99% 样本短于 10 秒。
- long-form：Audio-NTREX，10 名双语 speaker，每种语言约 10 小时，平均约 50 秒。
- real interpretation：VoxPopuli 的 90 条真人同传。
- 质量：ASR-BLEU；同时报告模型直接输出文本的 BLEU。
- 延迟：End Offset 与 LAAL。
- 音色：WavLM Large speaker embedding cosine similarity。
- 人评：audio quality、speaker similarity、naturalness；每个系统30条样本，每条15名评分者，1–5 MOS。
- 速度：批量 RTF；H100 上测到大 batch 仍快于实时。

Hibiki 的核心评估贡献是把“翻译正确”与“听起来像连续自然的同传”分开。对 Simul-UniSS 必须增加：

- 长音频评估，不能只看 UniST 短句。
- natural pauses、abrupt cuts、stuttering 和 speaker drift 的专项指标与试听集。
- batch throughput 与 batch=1 实时延迟分开报告。
- speaker similarity 不能用同一模型既作为训练条件又作为唯一评价；应补充盲听或第二个 speaker encoder。

官方 Hibiki 仓库主要提供推理入口，实际 PyTorch/MLX 核心实现位于 Moshi 生态中；它没有提供论文完整 evaluation pipeline。因此本计划复用其数据分层和指标定义，而不是直接复制仓库命令。

### 2.3 Hibiki-Zero：Without Aligned Data

论文与代码：

- [Simultaneous Speech-to-Speech Translation Without Aligned Data](https://arxiv.org/abs/2602.11072)
- [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero)
- 调研 commit：`871e89d078202c7d9d18d0924bd76cf161cd6606`

方法关键点：

- 不依赖 word-level alignment，只要求句子级 source/target 映射。
- 对每个目标句子随机加入与源句时长相关的 delay，并在标点处插入随机 silence，形成 coarse causal training pairs。
- 再使用带自然 pause 控制和 voice conditioning 的 TTS 改善边界自然度。
- 使用 GRPO；process reward 由当前输入已支持的参考前缀 BLEU 和最终 BLEU 共同组成。
- 过程 reward 每若干 source words 计算一次，论文默认 `n_w=8`。
- 正式 RL 配置：group size 4、2000 updates、temperature 0.8、top-k 250、`alpha=0.4`、clip epsilon 0.2。
- 论文明确表明：如果 SFT 从未探索过句中提前输出，RL 很难仅靠奖励把高延迟模型变成低延迟模型。

评估协议：

- short-form：Europarl-ST，2–20 秒；valid/test 每个 source language 1024 条。
- long-form：Audio-NTREX-4L；Fr/Es/Pt/De→En，每个 source language 300 个文本，多个商业 TTS，平均约45秒。
- 文本质量：BLEU、XCOMET-XL。
- 语音质量：ASR-BLEU、ASR-COMET。
- 延迟：End Offset、LAAL。
- 音色：WavLM speaker cosine similarity。
- 人评：audio quality、speaker similarity、naturalness，0–100 MOS；每种语言50条、20名评分者。

对 Simul-UniSS 的启示：

- Stage3/4/6 的 dev 评估必须计算 prefix translation quality，不能只算最终 BLEU。
- 对每次 WRITE，参考应是当前 source prefix 所支持的目标前缀，而不是整句最终参考，否则会奖励过早猜测。
- Stage7 前要保存 Stage3/4/6 的 prefix BLEU/COMET、premature-write rate 和 LAAL 基线。
- 当前 pseudo proportional alignment 可作为 coarse-alignment baseline，但必须在报告中明确它不是 word-level ground truth。
- 正式 Stage7 需要评估 RL 前后同一输入、同一解码 seed 的 paired difference。

官方 Hibiki-Zero 仓库提供：

- 实时 websocket server；
- batch file generation；
- 逐 codec frame 的 streaming inference；
- batch throughput/real-time factor 日志。

但官方仓库目前不包含论文所用的完整 BLEU/COMET/LAAL/MOS evaluation pipeline。因此本计划借鉴其输出事件记录和 batch inference 方式，同时在本项目内统一计算指标。

### 2.4 StreamSpeech：Multi-task Learning

论文与代码：

- [StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning](https://arxiv.org/abs/2406.03049)
- [ictnlp/StreamSpeech](https://github.com/ictnlp/StreamSpeech)
- 调研 commit：`220273fd13aaae648f7aaae83282df1f60246f93`

方法关键点：

- chunk-based Conformer：chunk 内双向，chunk 间单向。
- Source CTC 负责确认是否识别出新 source token。
- Target CTC 估计当前 source prefix 支持的目标 token 数。
- 只有 source token 数增长且 supported target count 大于已输出数时才 WRITE。
- AR-S2TT 负责流畅目标文本，NAR-S2TT CTC 主要作为 alignment/policy guide。
- NAR text-to-unit CTC 同步生成语音 unit，再由冻结 HiFi-GAN 解码。
- 总损失包含 S2UT、AR-S2TT、ASR、NAR-S2TT。
- multi-chunk training 在训练时随机改变 chunk size，使单一模型支持不同 latency operating points。

官方 evaluation 命令直接使用 SimulEval，并报告：

```text
ASR_BLEU
AL AP DAL StartOffset EndOffset LAAL ATD
NumChunks
DiscontinuitySum DiscontinuityAve DiscontinuityNum
RTF
```

同时开启 `--computation-aware`，并提供去除插入 silence 后重新计算 ASR-BLEU 的脚本。

论文的重要发现：

- 小 chunk 下 ASR-BLEU 下降不一定代表翻译内容错误，可能主要来自 chunk 间 silence、stuttering 和 ASR 对非连续语音的不适配。
- 因此必须同时报告原始 ASR-BLEU、silence-removed ASR-BLEU、Discontinuity 和 unit/text 质量。
- 320/640/960/1280/… ms 应形成连续 quality-latency 曲线。

对 Simul-UniSS 的启示：

- 当前 Stage3 的 action policy 必须与 Source/Target eligibility 分开评估。
- 当前 full198 尚未完成 Stage1/2，所以 Stage3 action accuracy 只能对 pseudo eligibility 评估，不能证明真实声学 gate 正确。
- Stage4/6 需要新增 `NumChunks`、`Discontinuity*` 和 computation-aware 全套指标。
- 未来接入 full198 Stage1/2 后，同一评估 manifest 可直接重跑，对比 pseudo policy 与真实 CTC policy。

### 2.5 NAST-S2x：Non-autoregressive Generation

论文与代码：

- [A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation](https://arxiv.org/abs/2406.06937)
- [ictnlp/NAST-S2x](https://github.com/ictnlp/NAST-S2x)
- 调研 commit：`2330e90c90bdba86cc136dcc7db3efc465014e2f`

方法关键点：

- chunk-based streaming encoder + chunk-based NAR decoder。
- CTC blank/repeat 允许每个 chunk 动态生成不同数量的 text/unit token。
- non-monotonic latent alignment loss 与 two-step glancing 改善 NAR 训练。
- 通过 chunk size 和 lookahead 控制 latency；论文发现直接增大 chunk 通常优于固定小 chunk 再增加 lookahead。
- offline S2ST 报告约 28 倍于 autoregressive baseline 的推理加速。

论文 S2ST 评估：

- 数据：CVSS-C Fr→En。
- chunk：320、640、1280、1920、2560 ms。
- 质量：ASR-BLEU、ASR-BLEU Silence Removed、Unit-BLEU、S2T-BLEU、BLASER 2.0。
- 延迟：AL、AL_EOW、AL_BOW、StartOffset、EndOffset；并报告 computation-aware 版本。
- 速度：每个 chunk 的 Average Computation Time，ACT。
- 连续性：DCNum、DCAve、DCSum。

代表性观察：

| chunk | ASR-BLEU | Silence-removed ASR-BLEU | DCNum | DCSum |
| ---: | ---: | ---: | ---: | ---: |
| 320 ms | 19.67 | 24.90 | 7.3 | 2220 ms |
| 640 ms | 19.15 | 25.67 | 4.7 | 1952 ms |
| 1280 ms | 20.20 | 25.71 | 2.1 | 1420 ms |
| 2560 ms | 24.88 | 26.14 | 0.4 | 395 ms |

这说明内容 unit 的正确性可能没有随小 chunk 显著下降，而语音播放不连续严重拉低了 ASR-BLEU。

对 Simul-UniSS 的启示：

- Stage4/6 必须同时计算 semantic-token 层和 waveform 层指标。
- BiCodec overlap/cross-fade 评估应使用同一 semantic token 输出，避免把 Qwen 改变与 decoder 改变混在一起。
- Stage8 的进入条件应来自 p95 RTF、ACT、buffer growth，而不是仅凭 GPU 利用率或主观感觉。

### 2.6 Textless Streaming S2ST using Semantic Speech Tokens

论文：

- [Textless Streaming Speech-to-Speech Translation using Semantic Speech Tokens](https://arxiv.org/abs/2410.03298)
- ICASSP 2025 DOI：[10.1109/ICASSP49660.2025.10889740](https://doi.org/10.1109/ICASSP49660.2025.10889740)

本次未发现论文作者提供的独立官方 GitHub evaluation repository，评估细节以论文为准。

方法关键点：

- RNN-Transducer 直接把 source acoustic features 转换为 target semantic speech tokens。
- RNN-T blank 决定何时继续读取 source，从结构上形成 streaming alignment。
- semantic tokenizer 单码本大小4096，输出约25 Hz。
- AcousticLM 将 semantic tokens 转为多层 acoustic tokens，DAC 解码 waveform。
- latency 由 RNN-T segment/right context 和 AcousticLM inference buffer 共同控制。

评估协议：

- 数据：CVSS-C Es→En、Fr→En、De→En。
- 质量：ASR-BLEU、BLASER 2.0 QE/Ref。
- 延迟：AL。
- AcousticLM buffer 取10/30/50形成质量–延迟曲线。
- 论文显示 buffer 变小时 ASR-BLEU 可能下降，但 BLASER 语义分数相对稳定，说明下降可能来自声学质量而非翻译语义。

对 Simul-UniSS 的启示：

- 需要把 semantic content quality 与 waveform intelligibility 分开。
- 同一 Stage4/6 semantic output 应在多个 BiCodec left-context/holdback/overlap 配置下解码，形成 codec quality-latency 曲线。
- BLASER 2.0 QE 可用于没有 reference target audio 的样本；Ref 仅用于有 reference speech 的数据。

## 3. 跨论文统一后的评估维度

所有论文共同指向以下六个不可替代的维度：

| 维度 | 核心问题 | 主指标 |
| --- | --- | --- |
| 最终翻译质量 | 整句翻译是否正确 | Text-BLEU、chrF、COMET、ASR-BLEU、ASR-COMET |
| 前缀质量/策略正确性 | 当前 WRITE 是否被 source prefix 支持 | prefix BLEU/COMET、WRITE precision/recall、premature WRITE |
| 理论策略延迟 | 忽略计算时，策略本身等待多少 | AL、AP、DAL、LAAL、ATD、StartOffset、EndOffset |
| 真实墙钟延迟 | 模型计算后是否仍能实时 | `*_CA`、ACT、first-audio wall time、p50/p95 RTF、buffer growth |
| 流式连续性 | 语音是否卡顿、断裂或重复 | NumChunks、DCSum、DCAve、DCNum、boundary jump、silence-removed ASR-BLEU |
| 语音保真度 | 音质、韵律、音色是否保持 | BLASER 2.0、WavLM speaker similarity、UTMOS、AutoPCP、SLC、MOS |

任何最终报告都必须至少包含：

```text
质量–延迟 Pareto 图
原始/去静音 ASR-BLEU 对照
computation-aware 与 non-computation-aware 对照
Stage3/Stage4/Stage6 paired comparison
短句/长句分层
ZH→EN 与 EN→ZH 分开
失败样本和可试听音频目录
```

## 4. 本地评估数据状态与使用边界

### 4.1 UniST dev/test：当前主内部评估集

| split | 样本数 | 方向 | 当前状态 |
| --- | ---: | --- | --- |
| UniST dev | 7,965 | cmn→eng 6,531；eng→cmn 1,434 | 已有完整 pseudo streaming schedules |
| UniST test | 23,369 | cmn→eng 14,257；eng→cmn 9,112 | parquet 已有；尚未建立独立 multi-chunk Simul-UniSS test schedules |

当前 dev schedule：

```text
data/processed/simul_uniss_v1/validation_dev/
```

统计：

```text
records:      7,965
events:      62,688
WAIT events: 49,767
WRITE events:12,921
chunk_ms:    640
wait_k:      2 chunks
alignment:   pseudo_proportional_token_alignment
```

UniST parquet 不含原始 WAV，但含 source/target BiCodec token，因此可以重建：

```text
source waveform:    bicodec_global + source_bicodec
reference waveform: bicodec_global + target_bicodec
```

这足够用于内部 streaming 评估和试听，但必须在报告注明它们是 BiCodec reconstruction，不是原始录音。

### 4.2 CVSS-T zh→en：可做 UniSS 论文口径，但当前不能直接做真实 streaming input

当前本地已有：

```text
/opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0/
```

其中 test 共 4,897 条 target English WAV 和 translation text。

限制：CVSS 下载包只含合成 target speech 和文本；source Common Voice 4 audio 需要单独下载并按文件名配对。当前没有发现对应 Common Voice 4 source archive。因此在 source audio 补齐前：

- 可以继续做 target/reference 音频审计；
- 不能声称完成 CVSS-T 真实 source-audio simultaneous inference；
- 不能计算真实 StartOffset/LAAL/ATD。

### 4.3 CVSS-C：论文协议对齐集

当前 CVSS-C Fr/Es/De/zh archives 已下载，但尚未形成可直接使用的 source-target paired evaluation manifest；同样缺少 Common Voice 4 source speech。

更重要的是，当前 UniSS full198 主要训练方向为中文与英文。SimulS2S-LLM、StreamSpeech、NAST-S2x 和 Textless Streaming 的主要公开结果是 Fr/Es/De→En，因此：

- 当前模型不能与这些数字做公平直接排名；
- 可复用相同指标定义和 SimulEval 协议；
- 只有未来模型支持相同语言、使用相同 CVSS-C split、相同 ASR 与 normalization 后才可计算差值。

### 4.4 长音频评估集

Hibiki/Hibiki-Zero 表明短句结果不能代表真实同传。当前模型不支持 Audio-NTREX-4L 的 Fr/Es/Pt/De→En，因此建立两个独立内部长音频集合：

1. `unist_longform_natural_v1`：从 UniST dev/test 中筛选本来就较长的样本，按 `<=10s`、`10–20s`、`20–40s`、`>40s` 分桶。
2. `unist_longform_concat_v1`：仅用于系统压力测试，将同方向样本按固定 seed 拼接成30/60/120秒输入，句间加入固定200/400ms silence；报告中标为 synthetic concatenation，不与论文 long-form 数据比较。

长音频重点评估：

- context/cache 是否随时间漂移；
- speaker/global token 是否保持；
- source buffer 是否持续增长；
- first-audio、EndOffset、p95 RTF 是否恶化；
- output 是否提前 EOS、漏句、重复、卡顿。

## 5. 必须建立的评估 baseline

同一 manifest、同一 reference、同一 normalization 下至少评估：

| Baseline | 用途 |
| --- | --- |
| Offline Phase3 base | 翻译/语音质量上界，衡量 simultaneous 带来的质量损失 |
| Oracle schedule replay | 使用数据构造 schedule 和 reference semantic token，验证 controller/BiCodec/metric 管道 |
| Stage0 fixed wait-k | 不依赖 Stage3 learned policy 的低成本策略基线 |
| Stage3 action policy | 判断 learned WAIT/WRITE 是否优于固定策略 |
| Stage4 interleaved | 判断 phrase+semantic 联合训练是否形成完整流式输出 |
| Stage6 refinement | 判断低LR refinement 是否改善 Stage4 而不损害离线上界 |

每个 model/policy 必须在相同 chunk grid 上运行，不能用不同 chunk 后只比较单点。

## 6. 评估 operating points

### 6.1 Chunk grid

主 grid：

```text
320, 640, 960, 1280, 1920, 2560 ms
```

其中：

- `640 ms + wait_k=2` 是当前训练分布内基准点；
- 320 ms 测低延迟极限；
- 960/1280 ms 测质量–延迟中间区域；
- 1920/2560 ms 用于观察接近 offline 时的质量上限和连续性变化。

### 6.2 Policy grid

每个 chunk 至少比较：

```text
oracle pseudo schedule
fixed wait-k: k = 1, 2, 3, 5
Stage3 learned action
Stage4 learned action
Stage6 learned action
learned action + hard eligibility gate
```

在 full198 Stage1/2 尚未完成前，`hard eligibility gate` 只能使用 pseudo source/target capacity，指标名必须带 `_proxy`。未来接入真实 Source/Target CTC 后，用同一 manifest 重跑并移除 `_proxy`。

### 6.3 Decoding grid

先在 dev 上使用：

```text
greedy/deterministic
temperature = 0.7, 0.8
top-k = 50, 250
semantic beam/search = 1, 5（仅在实现支持且计算时间完整计入时）
```

test/eval 只运行 dev 选定的最多3个 Pareto operating points：

- low-latency；
- balanced；
- high-quality。

采样配置至少运行3个固定 seed；greedy 配置运行1次。

### 6.4 BiCodec streaming grid

对固定 semantic output，单独扫描：

```text
left_context_tokens = 25, 50
holdback_tokens = 5, 10
overlap_ms = 40, 80, 120
cross_fade = equal_power
```

这一层只改变 BiCodec streaming decode，不重新运行 Qwen，从而把 codec 边界问题与语言模型问题隔离。

## 7. 分阶段评估设计

### 7.1 Stage3：WAIT/WRITE action-only SFT

Stage3 的训练 loss 只在 WAIT/WRITE token 上有权重，不能把它当成完整 S2ST checkpoint 来评价 ASR-BLEU。

#### 7.1.1 Teacher-forced action evaluation

在每个 action position 读取 Qwen logits，计算：

- action CE / perplexity；
- overall accuracy；
- WAIT precision/recall/F1；
- WRITE precision/recall/F1；
- confusion matrix；
- premature WRITE rate：reference 为 WAIT、模型为 WRITE；
- unnecessary WAIT rate：reference 为 WRITE、模型为 WAIT；
- final WAIT rate；
- first WRITE chunk error；
- predicted write count 与 reference write count 差值；
- 按方向、时长、chunk index、source coverage 分桶。

由于 WAIT 类明显多于 WRITE，禁止只看 accuracy；主指标应为 WRITE F1、premature WRITE 和 macro-F1。

#### 7.1.2 Free-running action evaluation

模型逐 chunk 自回归选择 action，已提交内容不允许回滚，记录：

- 每次 READ/WAIT/WRITE 的 source timestamp；
- action probability 和 entropy；
- first WRITE；
- 连续 WAIT 长度；
- source final 后是否强制 flush；
- 是否进入死循环、永远 WAIT 或反复 WRITE 空内容；
- policy latency：AL/AP/DAL/LAAL/ATD proxy；
- 与 oracle/fixed wait-k 的 paired difference。

Stage3 的通过条件：

- invalid action token 为0；
- final flush success 为100%；
- WRITE F1 明显高于 majority WAIT baseline；
- premature WRITE 不因降低 first-write latency而显著增加；
- dev 上形成至少一个不劣于 fixed wait-k 的质量–延迟点。

### 7.2 Stage4：phrase-level interleaved S2ST

Stage4 要同时评价 action、text phrase 和 semantic chunk。

#### 7.2.1 结构完整性

- WAIT 后不得生成目标内容；
- WRITE 必须有合法 target phrase 或 final flush；
- text phrase 只能 append，不允许修改已提交前缀；
- semantic chunk 只能 append；
- semantic token 必须位于 `[0, 8191]`；
- 32个 speaker/global token 保持不变；
- source final 后必须生成完整 tail 并 EOS；
- 记录 empty semantic、duplicate chunk、missing phrase、invalid delimiter。

#### 7.2.2 Simul-S2TT 质量

- final Text-BLEU；
- chrF/chrF++，尤其用于中文；
- COMET/XCOMET；
- prefix BLEU、prefix chrF、prefix COMET；
- prefix coverage：已提交目标长度/最终参考长度；
- hallucinated-prefix rate；
- repetition rate；
- final translation 与 offline Phase3 的差值；
- append-only revision rate，理论上应为0。

#### 7.2.3 Semantic-token 质量

只有 reference semantic token 可比时计算：

- semantic token accuracy；
- semantic edit distance；
- semantic BLEU/n-gram overlap；
- duration/token-count ratio；
- duplicate/repeated-run ratio；
- per-WRITE semantic token 数分布；
- chunk boundary token transition statistics。

semantic token 不是唯一正确答案，因此 token exact-match 只能作为诊断指标，不能作为最终语音质量指标。

#### 7.2.4 Waveform S2ST

- ASR-BLEU；
- ASR-BLEU Silence Removed；
- ASR-COMET；
- BLASER 2.0 Unsupervised/QE/Ref；
- UTMOS；
- AutoPCP；
- SLC-0.2、SLC-0.4；
- source/generated WavLM speaker cosine；
- reference/generated duration ratio；
- listening subset。

### 7.3 Stage6：低学习率 refinement

Stage6 使用与 Stage4 完全相同的 manifest、seed、decode 和 metric models，做 paired comparison。

核心研究问题：

1. Stage6 是否降低 validation loss，但自由运行质量没有改善？
2. Stage6 是否减少 early WRITE、repetition 或 incomplete flush？
3. Stage6 是否改善语义 token/音频质量，却增加 latency？
4. 0.25 epoch 是否足以形成稳定收益，还是仅产生很小随机波动？
5. Stage6 是否破坏 offline Phase3 replay 能力？

选择 Stage6 的标准不是单个 loss 更低，而是：

- 在相同 latency 下质量更高；或
- 在相同质量下 latency 更低；或
- 同时改善连续性/失败率且质量下降处于置信区间内；
- offline Phase3 quality 不出现显著回退。

若 Stage6 不支配 Stage4，则 Stage4 仍应保留为正式候选，不因编号更高而自动替换。

## 8. 指标定义与实现要求

### 8.1 文本与语音翻译质量

#### Text-BLEU

使用模型直接输出目标文本，按方向分别规范化后计算 SacreBLEU。必须保存 SacreBLEU signature。

#### ASR-BLEU

```text
generated waveform
→ 固定 ASR 模型转录
→ 与 target translation 计算 SacreBLEU
```

必须保存：ASR 模型、revision、language、decoding 参数、文本 normalizer 和 SacreBLEU signature。

建议：

- 英文目标：Whisper medium/large-v3 固定其一为论文对齐主结果；另一模型做稳健性审计。
- 中文目标：固定 Paraformer/FunASR 或已验证中文 ASR；同时报告字符级 CER，避免英文 ASR 口径套用中文。

#### COMET/ASR-COMET

- Text-COMET：模型文本 vs reference translation。
- ASR-COMET：生成语音ASR转录 vs reference translation。
- 模型固定为 XCOMET-XL 或项目已缓存的明确版本。

#### BLASER 2.0

- Unsupervised：source/generated speech embedding cosine，0–1。
- QE：无需 target reference speech，1–5。
- Ref：需要 target reference speech，1–5。

所有 UniST source/reference 均为 BiCodec reconstruction，报告中单独标记。

### 8.2 延迟指标

每个指标同时输出：

```text
NCA: non-computation-aware，只按策略/音频时间轴
CA:  computation-aware，加入真实模型与codec计算时间
```

必须实现：

- `StartOffset`：第一段目标语音开始播放的时间；
- `EndOffset`：最后目标语音结束时间减 source 结束时间；
- `AL`：平均输出相对理想同步进度的滞后；
- `AP`：输出时读取 source 的平均比例；
- `DAL`：对不连续读写更稳定的 differentiable AL；
- `LAAL`：修正 over-generation 的 length-adaptive AL；
- `ATD`：输出子段相对对应输入子段的平均 token/segment delay；
- `first_audio_latency`：从 source 开始到首个可播放 waveform sample；
- `ACT`：每次 chunk 的模型+codec计算时间；
- `RTF`：必须明确采用 `compute_time/audio_duration`，小于1表示快于实时；若外部代码使用相反定义，字段名必须转换后再汇总。

中文 LAAL 同时报告：

- waveform-frame LAAL；
- ASR character-emission LAAL；
- 若使用中文分词器，再额外报告 word LAAL，但不得和英文 whitespace word LAAL 无说明混合。

### 8.3 流式连续性

必须记录：

- NumChunks；
- DCSum、DCAve、DCNum；
- 每次 WRITE 输出时长；
- boundary amplitude jump mean/p95/max；
- boundary RMS/energy jump；
- boundary spectral/STFT distance；
- click detector rate；
- repeated waveform overlap；
- dropped/duplicated semantic token；
- ASR-BLEU 与 silence-removed ASR-BLEU 差值。

如果 silence-removed ASR-BLEU 显著高于原始 ASR-BLEU，而 Text-BLEU/semantic 指标稳定，应把问题归因到播放连续性或 codec 边界，不能直接归因到翻译模型。

### 8.4 音色、韵律和自然度

客观指标：

- WavLM Large speaker cosine；
- 第二 speaker encoder 的交叉验证分数；
- AutoPCP；
- UTMOS；
- SLC-0.2、SLC-0.4；
- F0 continuity、energy continuity；
- voiced/unvoiced boundary error。

主观盲听建议：

- 每个 Stage/operating point 至少30条平衡样本；
- 中文→英文、英文→中文分开；
- 质量、自然度、speaker similarity、卡顿感分别评分；
- 随机化文件名，不显示 Stage；
- 同一评测者不同时看到 reference method 名称；
- 报告均值、95% CI 和评分者一致性。

## 9. 推理与评估实现缺口

当前已有：

- `training/simul_uniss/stage0_eval.py`：schedule proxy 统计；
- `training/simul_uniss/latency_metrics.py`：token-level proxy AL/AP/ATD；
- `training/simul_uniss/replay_streaming.py`：oracle schedule 经过 `StreamingController` 和 BiCodec replay；
- `uniss/streaming/`：prefix commit、policy gate、BiCodec overlap decoder；
- Phase2/Phase3 的 text/audio/UTMOS/AutoPCP/SLC evaluation components。

当前缺少：

1. Stage3/4/6 Megatron checkpoint 的独立 HF export/验证 manifest。
2. `QwenStreamingAdapter`：逐 chunk 构造 prompt、解析 WAIT/WRITE、生成 phrase/semantic、维护 append-only state。
3. 真实墙钟 event logger。
4. multi-chunk dev/test schedule builder。
5. SimulEval-compatible S2ST agent 或等价 scorer adapter。
6. AL/DAL/LAAL/ATD 的 waveform 和 computation-aware 完整实现。
7. discontinuity、silence-removed ASR-BLEU 和长音频压力测试。
8. 统一 Stage3/4/6 paired report generator。

计划新增独立目录：

```text
experiments/evaluation/simul_uniss_stage3_stage4_stage6_v1/
  README.md
  experiment.env
  prepare_manifests.sh
  export_exact_checkpoints.sh
  run_teacher_forced_actions.sh
  run_streaming_s2tt.sh
  run_streaming_s2st.sh
  run_codec_grid.sh
  run_metrics.sh
  aggregate_report.sh

evaluation/simul_uniss/
  manifest.py
  qwen_streaming_adapter.py
  event_logger.py
  latency_metrics.py
  continuity_metrics.py
  stage_metrics.py
  aggregate_report.py
```

这些均为计划新增文件；在实际实现前不能把上面的命令当成当前已经可运行的脚本。

## 10. 输出目录与不可覆盖规则

每次运行创建：

```text
eval_outputs/
  simul_uniss_stage3_stage4_stage6_v1_<UTC_TIMESTAMP>/
    environment/
    manifests/
    exports/
    stage3/
      teacher_forced/
      free_running/
    stage4/
      s2tt/
      semantic/
      s2st/
    stage6/
      s2tt/
      semantic/
      s2st/
    baselines/
    metrics/
    curves/
    listening/
    failures/
    report.md
    COMPLETE
```

每个 sample 保存一行 JSONL，至少包含：

```json
{
  "sample_id": "...",
  "dataset": "unist",
  "split": "dev",
  "direction": "cmn-eng",
  "stage": "stage4",
  "checkpoint_iteration": 4753,
  "chunk_ms": 640,
  "policy": "learned_action",
  "seed": 42,
  "source_duration_ms": 0,
  "events": [],
  "final_text": "...",
  "semantic_tokens": [],
  "generated_wav": "...",
  "failure": null
}
```

冻结信息：

- git commit、git diff/status；
- checkpoint tracker 和 shard hash；
- manifest hash；
- metric model revision；
- conda/pip freeze；
- GPU 型号、CUDA、PyTorch；
- decoding config；
- random seed；
- wall-clock timestamps。

## 11. 8 GPU 执行策略

### 11.1 质量评估

可将固定 manifest 按 `sample_id hash % 8` 分为8个互斥 shard，每卡独立生成，最后严格按 sample_id 合并。

要求：

- 每个 sample 只由一张卡处理；
- shard manifest 和 seed 固定；
- 合并时检查无丢失、无重复；
- 各卡写独立目录，禁止多个进程同时写同一个 JSONL/WAV。

### 11.2 真实延迟评估

computation-aware 结果必须另跑：

- batch size = 1；
- 单卡、独占 GPU；
- 预热固定次数；
- 不与其他 GPU-intensive 任务并行；
- 分别记录 tokenizer、Qwen、BiCodec 和音频 I/O 时间；
- 报告 p50、p90、p95、p99，而不只报均值。

8卡并行吞吐不能替代 batch=1 实时延迟。两者分别报告：

- latency benchmark：1 GPU / batch1；
- throughput benchmark：1/2/4/8 GPU，多 batch size。

## 12. Dev → Eval 的完整执行顺序

### Phase A：冻结输入与 checkpoint

1. 读取3个 final tracker。
2. 验证所有 distributed checkpoint shard。
3. 记录 git/environment。
4. 生成固定 dev/test/listening/long-form manifest。
5. 对所有 manifest 计算 SHA256。

### Phase B：metric/controller 单元验证

1. 使用 oracle schedule replay 100条。
2. 检查 final flush、append-only、WAV长度、event timestamp。
3. 用手工构造事件验证 AL/AP/DAL/LAAL/ATD。
4. 用已知 silence 片段验证 discontinuity scorer。
5. 用相同 waveform 验证 silence removal 不改变非静音内容。

### Phase C：Stage3 dev

1. 全量 teacher-forced action metrics。
2. 100条 free-running smoke。
3. 全量 dev policy sweep。
4. 与 oracle/fixed wait-k 做 Pareto 比较。
5. 选择 Stage3 policy 的 low/balanced/high-quality operating points。

### Phase D：Stage4/6 dev

1. 先跑 Simul-S2TT，排除 codec 干扰。
2. 再跑 semantic-token 评估。
3. 最后跑 waveform S2ST。
4. Stage4/6 使用相同 seed 做 paired bootstrap。
5. 扫 chunk/policy；codec grid 只在入选 semantic 输出上运行。
6. 形成 dev Pareto 曲线。

### Phase E：长音频与墙钟性能

1. natural long-form。
2. 30/60/120秒 synthetic concatenation stress。
3. batch1 computation-aware latency。
4. 多 batch/GPU throughput。
5. 记录 source buffer growth 和 OOM。

### Phase F：锁定配置

在 dev 上最多选3个 operating points，写入只读 lock file：

```text
selected_operating_points.json
```

选择后禁止根据 test 结果重新调 chunk、wait-k、temperature、overlap 或 ASR normalization。

### Phase G：test/eval 一次性评估

1. UniST test 全量运行。
2. 只跑锁定 operating points。
3. 生成 per-sample 和 aggregate metrics。
4. 计算 paired bootstrap 95% CI。
5. 抽取固定 listening/failure subset。
6. 写最终报告并创建 `COMPLETE` marker。

### Phase H：CVSS 外部评估

仅当 Common Voice 4 source audio 补齐并完成配对审计后执行：

1. CVSS-T zh→en：与原 UniSS 离线协议对照，并增加 simultaneous 指标。
2. CVSS-C zh→en：使用 StreamSpeech/SimulS2S-LLM 指标口径，但不与 Fr/Es/De 数值直接比较。
3. 若未来模型支持 Fr/Es/De→En，再运行完全同 split 的论文数值对比。

## 13. 统计检验和结果选择

每个 aggregate 指标报告：

- mean；
- median；
- p90/p95；
- 95% bootstrap CI；
- 有效样本数和失败数。

Stage4 vs Stage6 使用 paired bootstrap，至少1000次 resampling。质量–延迟选择基于 Pareto dominance，不使用简单加权总分掩盖 trade-off。

需要单独报告：

- 全量；
- cmn→eng；
- eng→cmn；
- source dataset；
- duration bins；
- low/high duration ratio；
- first-write early/late groups；
- generation failure groups。

## 14. 通过门槛与停止条件

### 14.1 硬门槛

- checkpoint 可重复导出且 logits round-trip 通过；
- invalid token rate = 0；
- final flush success = 100%；
- append-only committed output 不回滚；
- per-sample manifest 无丢失、无重复；
- metric merge 数量与 manifest 完全一致；
- 所有失败必须有 reason code，禁止静默跳过。

### 14.2 simultaneous 有效性门槛

- 至少一个 learned-policy operating point 位于 fixed wait-k Pareto 前沿上或之外；
- first-audio latency 降低不能伴随显著 premature WRITE 增加；
- test final quality 相对 offline upper bound 的下降必须明确量化；
- computation-aware p95 RTF < 1 才能称为目标硬件上的实时系统；
- source buffer 不随长音频持续无界增长。

### 14.3 waveform 连续性门槛

- silence-removed 与原始 ASR-BLEU 差距不能仅靠移除长 silence 掩盖；
- DCNum/DCSum 随 chunk 减小的恶化必须可见并进入 Pareto 选择；
- boundary click、重复音频、漏播任何一项明显恶化时，不得只凭 BLEU 选择模型。

## 15. 最终报告结构

最终 `report.md` 应包含：

1. checkpoint、代码和数据冻结信息；
2. 当前实现边界：pseudo、free-running 或 real audio；
3. Stage3 action confusion/premature-write 分析；
4. Stage4/6 Text-BLEU、COMET、prefix quality；
5. semantic-token 诊断；
6. ASR-BLEU/ASR-COMET/BLASER/UTMOS/AutoPCP/SLC；
7. AL/AP/DAL/LAAL/ATD/StartOffset/EndOffset；
8. computation-aware 指标与 RTF；
9. discontinuity 和 silence-removed 对照；
10. short/long-form 分层；
11. Stage4 vs Stage6 paired CI；
12. 与论文相同数据/语言/口径时的对比表；
13. 不能直接比较的论文结果及原因；
14. 失败样本、试听目录和下一步建议。

推荐主图：

- ASR-BLEU vs ATD；
- ASR-BLEU vs LAAL；
- Text-BLEU vs LAAL；
- ASR-BLEU vs computation-aware LAAL；
- ASR-BLEU 与 silence-removed ASR-BLEU 差值 vs chunk；
- DCSum/DCNum vs chunk；
- speaker similarity/UTMOS vs latency；
- Stage3 premature WRITE vs first-audio latency。

## 16. 论文结果比较边界

只有同时满足以下条件才允许在表格中直接计算“高/低多少”：

```text
相同数据集版本
相同语言方向
相同 split
相同 source/target audio 定义
相同 ASR 与文本 normalization
相同 latency 定义
相同 computation-aware 设置
```

否则只能列为 protocol/reference，不计算胜负。

当前可直接做的是：

- Stage3 vs Stage4 vs Stage6 的同 manifest 内部比较；
- learned policy vs fixed wait-k/oracle 的同 checkpoint 比较；
- Stage4 vs Stage6 的 paired difference；
- UniST dev/test 的短长句、方向和数据来源分层。

当前不能直接做的是：

- 当前中文/英文 UniSS 与 CVSS-C Fr/Es/De→En 论文数字排名；
- UniST BiCodec reconstructed audio 与 Audio-NTREX 真人录音直接排名；
- proxy AL/ATD 与真实墙钟 computation-aware latency 混合比较；
- Stage3 action-only checkpoint 与完整 S2ST 模型按 ASR-BLEU排名。

## 17. 推荐的第一轮最小闭环

为了最快得到可信结论，第一轮只做：

```text
数据：UniST dev 全量 + test smoke 200条
checkpoint：Stage3 4753、Stage4 4753、Stage6 1189
chunk：320/640/1280/2560 ms
policy：fixed k=2、learned、oracle
decode：greedy
codec：left_context=50、holdback=5、overlap=80ms
```

输出：

- Stage3 action metrics；
- Stage4/6 Simul-S2TT；
- Stage4/6 200条真实 BiCodec S2ST；
- NCA/CA latency；
- continuity metrics；
- 试听集；
- 初版 Pareto 图。

第一轮通过后再扩展全 test、采样 seed、codec grid 和 long-form。这样可以先确认 Qwen streaming adapter 与 metric pipeline 正确，避免在全量生成后才发现时间戳或 action parsing 错误。

## 18. Stage4 full-dev 端到端执行补充（2026-07-27）

当前第一轮 Stage4 正式执行固定为一个训练分布内 operating point：

```text
dataset = UniST dev 7,965 raw records
checkpoint = Stage4 iter_0004753
source schedule = 640 ms, pseudo proportional alignment, wait-k training distribution 2
policy = Stage4 free-running learned WAIT/WRITE
Qwen decode = greedy, repetition_penalty 1.1, max WRITE 700 tokens
Stage4 training context boundary = 18,000; native Qwen inference envelope = 32,768; no truncation
BiCodec = left_context 50, holdback 5, overlap 80 ms, equal-power cross-fade
quality/throughput GPUs = 0,1,2,3 data parallel
```

所有输出进入新的时间戳目录：

```text
eval_outputs/simul_uniss_stage4_streaming_v1/<RUN_ID>/
```

Stage3、offline Phase2/Phase3、Stage4/6 checkpoint 和历史 eval_outputs 不允许被覆盖。

### 18.1 两条独立评估轨道

为了同时满足高 GPU 吞吐和真实 latency 定义，禁止把一个运行同时解释成两者：

| 轨道 | Batch | 用途 | 允许解释的结果 |
| --- | ---: | --- | --- |
| Full-dev quality/throughput | 每 GPU 最多512 active sequences | 跑完整7,965条、生成全部WAV、GPU利用率和公共质量指标 | corpus quality、失败率、批量吞吐、批量服务CA latency |
| Batch=1 latency audit | 每 GPU 一条独立stream | 排除大batch排队/调度影响 | request TTFT、每chunk ACT、first-audio、CA RTF、CA Start/EndOffset |

正式执行前的负载审计比较了每 GPU 512 与 1,024 active records。1,024 配置没有提高
H200 利用率，且单位样本吞吐略有下降；原因是0.5B Qwen的逐chunk单action解码和Python/CPU
调度占主导，而不是显存或KV cache不足。因此正式点固定512，不允许通过重复样本或重复计算
伪造高功率。GPU利用率、功率及其峰值必须作为实测结果如实报告。

负载审计还发现，自由运行生成历史可能长于teacher-forced reference：至少一条dev样本达到
4,108 tokens。保守上界审计表明，若模型在每个chunk都WRITE，最长样本可能达到约27,568
tokens。Stage4训练序列上限仍为18,000，但Qwen原生上下文为32,768，因此正式评估使用
`max_model_len=32768`防止崩溃，并逐样本记录`max_prompt_tokens`和
`training_context_exceeded`。超过18,000必须在报告中作为out-of-training-context告警，
不能把长样本截断、跳过或仅在4,096上下文内报告选择性结果。

正式报告必须把两条轨道分栏，不能用 full-dev batch 吞吐声称单会话 latency，也不能用
batch=1 低 GPU 利用率否定 full-dev 吞吐实现。

### 18.2 Stage4 实际自由运行协议

每条样本只保留一个 append-only prompt：

```text
streaming header
→ append source chunk
→ model generates one action
→ WAIT: append WAIT and read next chunk
→ WRITE: autoregressively generate target phrase + BiCodec semantic chunk
→ append next source chunk
→ final source forces a recorded final-WRITE recovery only when model仍WAIT
```

任何 invalid action、缺少 content/semantic delimiter、达到 max-write-token 上限、空文本、空
semantic 或重复循环都必须保留 reason code。允许为后续 prompt 规范化 delimiter 以继续评估，
但该样本仍计入 structural recovery/failure，不能静默当作成功。

### 18.3 本轮必须输出的 streaming 指标

策略与结构：

- WAIT/WRITE accuracy、Macro-F1、WAIT/WRITE precision/recall/F1；
- premature WRITE、unnecessary WAIT、forced final flush；
- invalid action、structural recovery、empty text/semantic、write count；
- prefix append-only、prefix BLEU、final Text-BLEU；
- semantic length ratio、aligned token accuracy、unigram/bigram F1、重复 run。

NCA/CA 延迟：

- first-WRITE、StartOffset、EndOffset；
- AL、AP、DAL、LAAL、ATD token/schedule proxy；
- action/write request TTFT、queue time、request wall time；
- per-WRITE BiCodec ACT；
- Qwen RTF、codec RTF、联合 RTF；
- batch throughput 与 batch=1 latency 分开。

连续性：

- NumChunks；
- playback gap count/sum/mean，NCA 和 CA 分开；
- boundary amplitude jump mean/p95/max；
- boundary RMS jump、spectral distance、click rate；
- semantic dropped/duplicated/repeated diagnostics；
- 原始和 silence-removed ASR-BLEU 在生成真实 gap waveform 后分开报告。

### 18.4 与 offline Phase3 完全相同的指标

Streaming 最终 WAV 仍可使用 offline 完全相同的指标。对比必须使用相同 7,965 个 raw ID、
相同 target reference、相同 ASR、相同 normalizer 和相同 metric model：

| 公共指标 | Streaming 输入 | Offline Phase3 输入 | 是否直接计算差值 |
| --- | --- | --- | --- |
| Text-BLEU | Stage4 最终 committed text | Phase3 Q/P generated translation | 是，分别对Q/P作差 |
| Speech-BLEU | Streaming BiCodec WAV→同ASR | Offline WAV→同ASR | 是 |
| SLC-0.2/0.4 | Streaming WAV/source WAV duration | Offline 同口径 | 是 |
| UTMOS | Streaming WAV | Offline WAV | 是 |
| AutoPCP | source/streaming WAV | source/offline WAV | 是 |
| Speaker similarity | source/streaming WAV | source/offline WAV | 只有同一encoder完成两侧后才比较 |

Offline Phase3 是质量上界，不参与 learned WAIT/WRITE、AL/LAAL/ATD 排名。最终必须报告：

```text
ΔText-BLEU
ΔSpeech-BLEU
ΔUTMOS
ΔAutoPCP
ΔSLC-0.2 / ΔSLC-0.4
quality-retention vs NCA/CA latency Pareto
```

### 18.5 当前外部模型边界

本机已缓存并验证 Whisper large-v3、中文 Paraformer、UTMOS 和 AutoPCP，可直接用于全量。
XCOMET/COMET、BLASER 2.0、第二 speaker encoder 当前未安装或缓存；在这些模型被固定版本、
下载校验并对 offline/streaming 两侧同时重跑前，报告中必须标记为 `not measured`，不能用
其他指标冒名替代。该边界不影响 Text/Speech-BLEU、SLC、UTMOS、AutoPCP 和全部内部
streaming event/latency/continuity 指标。

## 19. 参考文献与代码

1. SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation. ACL 2025. [arXiv:2504.15509](https://arxiv.org/abs/2504.15509)
2. High-Fidelity Simultaneous Speech-to-Speech Translation. [arXiv:2502.03382](https://arxiv.org/abs/2502.03382). [GitHub](https://github.com/kyutai-labs/hibiki)
3. Simultaneous Speech-to-Speech Translation Without Aligned Data. [arXiv:2602.11072](https://arxiv.org/abs/2602.11072). [GitHub](https://github.com/kyutai-labs/hibiki-zero)
4. StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning. ACL 2024. [arXiv:2406.03049](https://arxiv.org/abs/2406.03049). [GitHub](https://github.com/ictnlp/StreamSpeech)
5. A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation. ACL 2024. [arXiv:2406.06937](https://arxiv.org/abs/2406.06937). [GitHub](https://github.com/ictnlp/NAST-S2x)
6. Textless Streaming Speech-to-Speech Translation using Semantic Speech Tokens. ICASSP 2025. [arXiv:2410.03298](https://arxiv.org/abs/2410.03298)
7. SimulEval: An Evaluation Toolkit for Simultaneous Translation. [GitHub](https://github.com/facebookresearch/SimulEval)
8. BLASER 2.0 / Seamless communication evaluation resources. [GitHub](https://github.com/facebookresearch/seamless_communication)
