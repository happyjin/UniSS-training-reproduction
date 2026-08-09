# UniSS Emformer 各阶段训练动机与 StreamSpeech 原文方案移植审计

> 审计日期：2026-08-05  
> 当前工程：`experiments/uniss_streamspeech_ctc_v1/`  
> 对照论文：*StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning*，ACL 2024，[arXiv:2406.03049](https://arxiv.org/pdf/2406.03049)  
> 对照代码：[ictnlp/StreamSpeech](https://github.com/ictnlp/StreamSpeech)  
> 审计原则：排除 Conformer/Emformer、Fairseq/Megatron、mHuBERT/BiCodec 等不能直接相同的底层架构因素，只比较可以移植的训练数据、监督任务、objective、联合训练方式、流式策略和评估方法。

## 1. 核心结论

### 1.1 当前没有一个 Stage 完整按照 StreamSpeech 原文训练

严格结论是：**没有。**

当前工程已经移植了 StreamSpeech 的若干关键思想，但没有完成原文最关键的完整训练闭环：

```text
共享流式语音编码器
  + Source ASR CTC
  + Target NAR-S2TT CTC
  + CTC 对齐约束下的 AR-S2TT
  + Target-text → Target-unit NAR CTC
  + multi-chunk 随机训练
  + 所有任务在同一次端到端训练中联合优化
```

当前最接近原文的阶段有三个，但各自只覆盖一部分：

1. **Stage03b** 最接近原文的“文本侧三任务 objective”：实现了 `4×ASR CTC + 4×NAR-S2TT CTC + 8×AR-S2TT CE`。
2. **Stage05** 最接近原文的“CTC 计数 READ/WRITE 推理策略”：source CTC 出现新 token 且 target CTC 具有新 token 时 WRITE。
3. **Stage08 Step1** 最接近原文的“共享编码器联合训练原则”：三项文本损失通过同一次 Emformer forward 联合优化，并额外连接冻结 Phase3 的下游 NLL。

但是：

- Stage03b 和 Stage08 的 AR-S2TT 训练都没有使用原文由 CTC 对齐产生的 `g(i)` 流式 attention mask；
- 当前全部正式训练固定使用 `160 ms segment + 80 ms right context`，没有原文 multi-chunk 随机训练；
- 当前没有训练原文的 NAR Text-to-Unit / Unit CTC objective；
- 当前目标语音由 Phase3 Qwen 自回归 semantic token + BiCodec 生成，不是原文 target-text hidden → NAR unit CTC；
- 当前大多数训练是分阶段冻结/解冻，而原文 StreamSpeech 的正式模型是一次多任务端到端联合训练；
- Stage08 Step1-R 和 Step2 都未通过预注册质量门，后续 Stage09–13 只是 research-only 流水线验证。

### 1.2 如果只问“哪个阶段最像原文”

| 问题 | 当前最接近的阶段 | 准确判断 |
|---|---|---|
| 原文 ASR CTC objective | Stage03、Stage03b、Stage08 Step1 | 已移植 |
| 原文 NAR-S2TT CTC objective | Stage03、Stage03b、Stage08 Step1 | 已移植 |
| 原文 AR-S2TT CE 权重 `8.0` | Stage03b、Stage08 Step1 | 损失类型和权重已移植，但训练 attention 范围未移植 |
| 原文四任务 joint objective | 无 | 缺少 NAR S2UT/Unit CTC |
| 原文 CTC-count READ/WRITE | Stage05、Stage09 | 策略主体已移植，并增加稳定性保护 |
| 原文 multi-chunk training | 无 | 尚未实现 |
| 原文 NAR Text-to-Unit | 无 | Stage06 名称含 `nar`，但实际不是原文 NAR T2U |
| 原文 all-in-one 端到端联合训练 | 无 | Stage08 Step1 只是最接近的部分联合训练 |
| 原文 SimulEval 完整评估 | 无 | 当前仅实现部分延迟/质量字段 |

因此，最准确的表述应是：

> 当前 UniSS Emformer 路线是“StreamSpeech-inspired / partially transplanted”，而不是 StreamSpeech 的严格复现。Stage03b 复现了文本侧损失组合，Stage05/09 复现了 CTC-count policy，Stage08 Step1 复现了部分 joint-training 原则；原文最重要的 policy-conditioned AR training、multi-chunk training 和 NAR unit CTC 尚未实现。

## 2. StreamSpeech 原文真正训练了什么

### 2.1 数据监督四元组

原文把每条训练数据定义为：

\[
\mathcal{D}=\{(X,A,Y,S)\},
\]

其中：

- `X`：源语音；
- `A`：源语音 transcription；
- `Y`：目标翻译文本；
- `S`：目标语音；
- `U`：从目标语音 `S` 提取的离散 target speech units。

原文 CVSS-C 数据处理包括：

- 源语音统一为 16 kHz；
- 源语音提取 80 维 filterbank，并进行全局 CMVN；
- 每个编码器 feature 约对应 40 ms；
- 目标语音为 22.05 kHz；
- 目标语音经 mHuBERT layer 11 + k-means 得到 1000 类 units；
- source/target text 分别训练 6000 词 unigram SentencePiece；
- HiFi-GAN vocoder 预训练后冻结。

这些具体声学表示可以替换，但四类监督不可随架构变化而消失：

```text
源语音 X
源文本 A
目标文本 Y
目标语音离散表示 U
```

UniST 当前实际具备对应信息：

| StreamSpeech 字段 | UniST 当前字段 | 可移植性 |
|---|---|---|
| `X` source speech | `source_audio` | 已具备 |
| `A` source transcription | `transcription` | 已具备 |
| `Y` target translation | `translation` | 已具备 |
| `U` target units | `target_bicodec`，另有 `bicodec_global` | 已具备类似监督，但尚未按 NAR Unit CTC 使用 |

因此，数据不需要强制改成 CVSS-C TSV 或 mHuBERT km1000；真正需要移植的是这四种监督关系。

### 2.2 原文是 two-pass architecture，不是“两阶段分别训练”

原文的 two-pass 指推理计算图：

```text
Pass 1：source speech → target text hidden / AR target text
Pass 2：target text hidden → target speech units → vocoder
```

它不表示先把 Pass 1 单独训完，再冻结后训练 Pass 2。论文第 3.2 节明确说明所有任务通过 multi-task learning 端到端联合优化。

这点和当前 Stage02→03→03b→04→06→08 的逐级冻结式训练不同。当前分阶段设计适合低风险工程诊断，但不能等同于原文正式训练过程。

### 2.3 原文四个 objective

#### ASR CTC

从共享流式 encoder hidden `H` 预测源文本 `A`：

\[
L_{ASR}=CTC(CTCDec_A(H), A).
\]

Motivation：学习 source speech → source text 对齐，判断当前是否识别出了新的源 token。

#### NAR-S2TT CTC

从相同 `H` 非自回归预测目标文本 `Y`：

\[
L_{NAR-S2TT}=CTC(CTCDec_Y(H), Y).
\]

Motivation：它不负责提供最流畅的最终翻译，而是学习 source speech → target text 的单调软对齐和“当前证据能支持多少目标 token”。论文特别指出 NAR-S2TT 的 BLEU 可能不高，但 unigram accuracy 足以指导 policy。

#### Policy-conditioned AR-S2TT

原文不是普通的 full-context teacher-forcing AR 翻译。它先根据两个 CTC 路径的期望 token 数得到：

- `N_asr(j)`：听到 source prefix `X≤j` 后识别出的 source token 数；
- `N_nar-s2tt(j)`：同一 source prefix 所支持的 target token 数。

第 `i` 个目标 token 能看到的 source 边界是：

\[
g(i)=\arg\min_{j:\,N_{asr}(j-1)<N_{asr}(j)}
\{j\mid N_{nar-s2tt}(j)\ge i\}.
\]

然后训练：

\[
L_{AR-S2TT}=-\frac{1}{|Y|}\sum_i
\log p(y_i\mid X_{\le g(i)},Y_{<i}).
\]

Motivation：NAR CTC 负责决定“什么时候可以写”，AR decoder 负责生成流畅翻译。**如果 AR decoder 训练时能看完整源语音，推理时却只给 prefix，就会产生明显 train/inference mismatch。**

#### NAR Text-to-Unit CTC / S2UT

AR text decoder hidden 经过 T2U encoder，再按比例上采样 `r`，由 unit CTC decoder 预测目标 speech units：

\[
L_{S2UT}=CTC(CTCDec_U(D_{text}),U).
\]

论文设置 `r=25`。这一部分使用 NAR 的原因是 text 与 speech unit 基本单调对应，而且 unit 序列较长；NAR 能显著降低合成计算量。

Motivation：学习 target text → target speech unit 对齐，并让已经生成的 target text prefix 能同步产生对应的语音，而不是每次自回归生成很长的 semantic token 序列。

### 2.4 原文完整 objective 与权重

论文主文写作：

\[
L=L_{S2UT}+L_{AR-S2TT}+L_{ASR}+L_{NAR-S2TT}.
\]

Appendix H 和官方 YAML/训练脚本给出的实际权重为：

\[
L=
1\cdot L_{S2UT}
+8\cdot L_{AR-S2TT}
+4\cdot L_{ASR}
+4\cdot L_{NAR-S2TT}.
\]

官方仓库对应配置：

- `target_unigram`：Transformer AR-S2TT，weight `8.0`；
- `source_unigram`：ASR CTC，weight `4.0`；
- `ctc_target_unigram`：NAR-S2TT CTC，weight `4.0`；
- primary speech-to-unit CTC：weight `1.0`。

### 2.5 原文 multi-chunk training

论文理论描述是在训练时随机采样 chunk size：

\[
C\sim U(1,|X|),
\]

其中 `C=|X|` 等价于 offline。

官方代码的实际离散实现为每个训练 batch 从以下集合随机选择：

```text
C ∈ {8, 16, 24, 32, offline}
```

每个 feature 对应约 40 ms，因此近似为：

```text
320 / 640 / 960 / 1280 ms / offline
```

Motivation：一个 checkpoint 同时适应不同 latency，推理时只改变 chunk size，不需要为每种延迟训练不同模型。论文 Table 4 表明 multi-chunk 模型在所有测试 chunk 下都优于只使用固定 chunk 训练的模型。

**multi-chunk 是训练方法而不是底层架构，因此即使把 Conformer 换成 Emformer，这一原则仍然应该移植。**

### 2.6 原文推理 policy

每次收到新 source chunk：

1. ASR CTC 解码当前 source token path；
2. NAR-S2TT CTC 解码当前 target token path；
3. source CTC 新增 token 且 target CTC 比已生成 target 更长时 WRITE；
4. AR decoder 只生成 target CTC 当前支持的新 target tokens；
5. NAR T2U CTC 为新增 target text hidden 生成 units；
6. 冻结 vocoder 合成新增语音；
7. 否则 READ 下一个 source chunk。

原文 policy 的关键不是固定 wait-k，而是三个对齐链：

```text
source speech → source text
source speech → target text
target text   → target units/speech
```

### 2.7 论文和官方训练脚本自身的配置差异

为了审计严谨，需要注意论文 Appendix 与公开脚本也并非每个数值完全一致：

| 项目 | 论文 Appendix H | 官方 `train.simul-s2st.sh` |
|---|---:|---:|
| GPU | 4×RTX 3090 | 4 张可见 GPU |
| optimizer | Adam | Adam `(0.9,0.98)` |
| LR | `1e-3` | `1e-3` |
| scheduler | inverse square root | inverse square root |
| warmup | 4000 updates | 10000 updates |
| max tokens | 160k | 22000，`update-freq=2` |
| precision | 未在表中单列 | FP16 |
| multi-chunk | 是 | `{8,16,24,32,offline}` |

因此本报告判断“是否按原文”时优先比较方法本身，而不要求 UniSS 复制 3090 数量或 Fairseq batch 参数。

## 3. 当前 Emformer 路线的前置来源

### 3.1 Causal Student v2/v3 的 motivation

当前 Stage03 并不是从随机 Emformer 开始，而是从 Stage-B-v3 causal Emformer checkpoint 初始化。

这条前置路线的 motivation 是：

- 用真正有 cache 的 causal Emformer 替代不断重算 prefix 的 WhisperVQ；
- 每 160 ms 处理新音频；
- 使用 80 ms right context；
- 蒸馏到原 UniSS WhisperVQ/GLM codebook 空间；
- 让后续 Phase3 可以尽量复用原 source speech token 接口。

这不是 StreamSpeech 原文训练。原文直接为 ASR、NAR-S2TT、AR-S2TT、S2UT 学习共享表示；Student v2/v3 首先蒸馏 WhisperVQ 表示，属于 UniSS 的兼容性预训练。

其价值是提供一个可缓存、真流式、已经过声学预训练的 Emformer 初始化；其风险是优化目标先服务于 WhisperVQ token agreement，而不是原文四任务联合最优。

## 4. 当前 Stage00–13 的逐阶段 motivation 与原文一致性

下表中的分类含义：

- **严格对应**：可移植的方法、objective 和训练语义基本相同；
- **部分对应**：复现了某个 loss/policy，但缺少原文上下文或联合训练；
- **诊断/工程扩展**：为 UniSS 接口或排错而新增，不属于原文；
- **未训练**：只做数据、policy、runtime、评估或 Demo。

| Stage | 是否训练 | 当前 motivation | 当前 objective/动作 | 与原文关系 |
|---|:---:|---|---|---|
| 前置 Student v2/v3 | 是 | 把 WhisperVQ 前端换为 causal cached Emformer | hidden/codebook 蒸馏 + CTC/capacity/stability 辅助 | UniSS 自研前置，不是原文 |
| Stage00 | 否 | 确认 15-shard 数据、offset、字段、音频和长度合法 | 只读 audit | 工程准备 |
| Stage01 | 部分 | 构造 source/target CTC 文本监督 | 训练 EN/ZH SentencePiece；生成四路 CTC targets | 数据思想部分对应 |
| Stage02 | 是 | 低成本判断冻结 latent 是否线性可分 | 四个 linear CTC heads，`4 ASR + 4 NAR` | 原文 CTC loss 的诊断 ablation，不是原文正式训练 |
| Stage03 | 是 | Stage02 失败后，联合更新 Emformer 和 CTC heads | `4 ASR CTC + 4 NAR-S2TT CTC` | 原文四任务的两个子任务 |
| Stage03b | 是 | 加入流畅翻译监督，避免仅靠 NAR CTC | `4 ASR + 4 NAR + 8 AR CE` | 当前最接近原文文本侧 objective，但 AR 未使用 `g(i)` mask |
| Stage04 | 是 | 把 Emformer hidden 接回冻结 UniSS Phase3 | codebook bridge + frozen Phase3 NLL + entropy | UniSS 接口桥接，非原文 |
| Stage05 | 否 | 用两个 CTC path 决定 READ/WRITE | CTC count + 2 次确认 + append-only + lagging | policy 主体最接近原文，加入安全扩展 |
| Stage06 | 是 | 在离散 B2 bridge 上加连续小残差，提高 Phase3 质量 | Phase3 NLL + `1e-4` residual MSE | 非原文；不是原文 NAR T2U |
| Stage07 | 否 | 用固定样本比较 Stage06 checkpoint | BLEU/RTF/首 token gate | 工程 checkpoint gate |
| Stage08 Step1 | 是 | 让 Emformer 同时承受 endpoint 与 Phase3 梯度 | `4 ASR + 4 NAR + 8 AR + 0.5 Phase3 NLL + residual` | 最接近原文 joint principle，但仍缺三项核心训练机制 |
| Stage08 Step1-R | 是 | 修复中英/英中方向失衡和 Phase3 梯度弱 | balanced sampling，Phase3 weight `2.0`，ZH→EN weight `1.25` | UniSS 自研修复 |
| Stage08 Step2 | 是 | 让 Qwen 适配 streaming embeddings，同时 replay offline Phase3 | `0.7 streaming NLL + 0.3 offline replay NLL`，Q/V LoRA | UniSS 自研，原文没有 Qwen/LoRA/replay |
| Stage09 | 否 | 把 Emformer、CTC policy、B1/Phase3 接成真 chunk runtime | 160 ms + 80 ms，真实 cache，append-only | 推理 policy 部分对应 |
| Stage10 | 否 | Qwen KV-cache Micro-WRITE，避免重编码历史 | CTC WRITE 触发 Qwen 增量生成 | UniSS 自研后端 |
| Stage11 | 否 | 把 semantic spans 增量解码成 BiCodec 音频 | anti-collapse、holdback、crossfade | UniSS 自研后端；非 NAR unit CTC |
| Stage12 | 否 | 汇总双向质量、NCA/CA 和 fallback | 两样本 research evaluation | 只覆盖原文评估的一小部分 |
| Stage13 | 否 | 公网 Gradio 展示真流式链路 | 上传/麦克风、音频、timeline、stereo | Demo，不是训练阶段 |

## 5. 各训练阶段的详细分析

### 5.1 Stage01：数据准备

#### Motivation

不改写 150 万条 15-shard 源 manifest，只增加可版本控制的文本 tokenizer 和 CTC sidecar。

#### 已完成

- train：1,484,825 条；
- valid：15,175 条；
- EN tokenizer：BPE 8000；
- ZH tokenizer：unigram 8000；
- 四路任务标签：`asr_eng/asr_cmn/nar_s2tt_eng/nar_s2tt_cmn`；
- 25 Hz CTC path audit 无非法样本。

#### 与原文比较

相同点：

- 同时保留 source transcription 和 target translation；
- source/target CTC 使用语言文本 token；
- 使用 SentencePiece；
- 在训练前检查 CTC input length 是否足够。

不同点：

- 原文 source/target 都是 6000 unigram；当前 EN BPE、ZH unigram，均为 8000；
- 当前尚未把 `target_bicodec` 构造成原文 NAR T2U 的 unit CTC target；
- 当前没有原文统一的 S2ST multitask batch 格式，而是通过 sidecar 在后续阶段 join。

判断：**数据思想部分对应，但只完成了文本侧。**

### 5.2 Stage02：冻结 encoder 的 CTC probe

#### Motivation

先用最便宜的实验回答：已有 causal latent 是否已经包含足够的 ASR 和 target alignment 信息？如果只训练四个线性头就能通过，则无需破坏性地微调 Emformer。

#### Objective

\[
L=4L_{ASR}+4L_{NAR-S2TT}.
\]

encoder、Qwen、BiCodec 全部冻结，只训练四个 linear CTC heads。

#### 实际结果

- English ASR WER：81.57%；
- Chinese ASR CER：84.44%；
- NAR-S2TT English unigram recall：4.73%；
- NAR-S2TT Chinese unigram recall：5.59%。

预设 gate 明显失败，因此证明原 Student latent 不能直接线性承担 StreamSpeech endpoint tasks。

#### 与原文比较

原文没有“冻结 encoder 只训线性 probe”这一阶段。它是合理的可行性诊断，但不能称为 StreamSpeech 正式训练。

判断：**原文 loss 的诊断 ablation。**

### 5.3 Stage03：CTC-only Emformer 联合微调

#### Motivation

Stage02 失败说明必须让 encoder 本身适应 source/target text alignment。Stage03 从 Stage-B-v3 Emformer 初始化，训练整个声学前端、Emformer 和四个新 CTC heads。

#### Objective

\[
L=4L_{ASR}+4L_{NAR-S2TT}.
\]

- encoder LR：`2e-5`；
- head LR：`2e-4`；
- 8 GPU；
- 10,000 steps；
- 固定 Emformer 160 ms segment + 80 ms right context。

#### 实际结果

- English ASR WER：41.25%；
- Chinese ASR CER：34.93%；
- target English recall：17.99%；
- target Chinese recall：18.56%。

ASR 明显改善，但 NAR-S2TT 仍远低于预设 40% gate。

#### 与原文比较

对应了原文 `L_ASR` 和 `L_NAR-S2TT`，并且这两个 loss 会更新共享 encoder。缺失：

- AR-S2TT；
- S2UT unit CTC；
- multi-chunk；
- CTC-guided AR mask；
- 四任务同次联合训练。

判断：**原文四任务 objective 的两任务 ablation。**

### 5.4 Stage03b：加入 AR-S2TT

#### Motivation

论文明确说明 NAR-S2TT 适合做 alignment，但不适合独立负责流畅翻译；因此加入 AR translation decoder。

#### Objective

\[
L=4L_{ASR}+4L_{NAR-S2TT}+8L_{AR-S2TT}.
\]

权重和原文文本侧完全相同。当前 AR decoder 为共享四层 causal Transformer decoder，按目标语言使用不同 embedding/output projection。

#### 实际结果

- English ASR WER：40.55%；
- Chinese ASR CER：34.42%；
- NAR target English recall：18.35%；
- NAR target Chinese recall：19.28%；
- AR English token accuracy：37.11%；
- AR Chinese token accuracy：30.31%。

#### 最关键的不一致

当前 `EndpointCTCARStudent.forward()` 中，AR decoder 的 memory 是整条 utterance 的全部 Emformer hidden，只使用普通 memory padding mask。它没有实现原文 `g(i)` 对每个 target position 限制可见 source prefix。

因此当前 Stage03b 实际训练的是：

\[
-\log p(y_i\mid X_{full},Y_{<i}),
\]

而原文训练的是：

\[
-\log p(y_i\mid X_{\le g(i)},Y_{<i}).
\]

这不是小差异。它意味着训练时 AR decoder 可以看整句，Stage09/10 推理时却只能看当前 prefix，直接造成 early WRITE 下的翻译不完整、重复和语义漂移。

判断：**损失名称和权重最接近原文，但训练语义只完成了一半。**

### 5.5 Stage04：B2 discrete Phase3 bridge

#### Motivation

原文 AR decoder 和 T2U decoder属于同一模型；UniSS 已有预训练 Phase3 Qwen 和 GLM speech-token 接口。Stage04 的目标是尽量不改 Phase3，把 Emformer hidden 映射回 Phase3 能读取的 speech embedding。

#### Objective

```text
Emformer hidden
  → 2:1 pooling
  → 768→1280 projection
  → frozen WhisperVQ codebook nearest neighbor
  → frozen Phase3 embedding
  → Phase3 target NLL
```

只训练 bridge projection：

\[
L=L_{Phase3-NLL}+10^{-3}L_{entropy}.
\]

#### 与原文比较

这一步没有原文直接对应。它解决的是 UniSS 预训练接口兼容性，而不是原文 target text → target unit alignment。

判断：**必要的 UniSS bridge，但不是 StreamSpeech objective。**

### 5.6 Stage05：CTC-count policy

#### Motivation

把 Stage03b 的 source ASR CTC 和 target NAR-S2TT CTC 转化为无需再训练的 READ/WRITE policy。

#### 当前规则

- source stable count 增加；
- target stable count 大于已提交 count + `lagging_k`；
- 两个连续 chunk 观察一致；
- committed token 永不回滚；
- 英文不提交未完成 SentencePiece word；
- final 可 flush。

#### 与原文比较

原文主体条件是：

```text
source CTC count 新增
AND target CTC count 能支持新的 target token
→ WRITE
```

当前实现与这个主体一致。额外的“两次确认、whole-word、append-only、conflict 统计”是为了语音不可逆输出增加的安全措施，不改变核心 motivation。

实际 256 条 balanced valid：

- WRITE coverage：96.48%；
- first WRITE p50：1320 ms；
- first WRITE p95：4240 ms；
- committed unigram recall：15.96%；
- rollback：0。

这说明 policy 工程结构成立，但 target CTC 内容质量较低。

判断：**当前对原文推理 policy 最忠实的阶段。它不是训练阶段。**

### 5.7 Stage06：B1 continuous residual

#### Motivation

Stage04 hard codebook projection有量化上限，因此保持 B2 baseline 不变，只训练一个有界连续 residual，提高 Phase3 输入表示。

#### Objective

\[
E_{Qwen}=E_{B2-hard}+0.05\tanh(W h),
\]

\[
L=L_{Phase3-NLL}+10^{-4}L_{residual-MSE}.
\]

#### 需要特别澄清

目录名是 `stage06_b1_nar`，但当前 Stage06：

- 没有 target text hidden → unit 序列；
- 没有 CTC upsampling；
- 没有 target unit CTC loss；
- 没有 NAR BiCodec semantic decoder。

所以它**不是 StreamSpeech 原文的 NAR Text-to-Unit**。它只是一个连续 Phase3 bridge residual。

Stage07 固定 probe 中，Stage06 iter600 达到 EN→ZH 18.85、ZH→EN 19.07 BLEU，优于 Stage04 B2，但仍未通过 gate。

判断：**UniSS 自研表示修复。**

### 5.8 Stage08 Step1：当前最接近 joint multi-task 的训练

#### Motivation

Stage03b 只优化 endpoint tasks，Stage04/06 只优化 Phase3 endpoint，梯度彼此断开。Step1 让一次共享 Emformer forward 同时收到文本对齐、翻译和 Phase3 下游梯度。

#### Objective

\[
L=
4L_{ASR}
+4L_{NAR-S2TT}
+8L_{AR-S2TT}
+0.5L_{Phase3-NLL}
+10^{-4}L_{residual}.
\]

训练参数：

- 最后四层 Emformer；
- output norm；
- 四个 CTC heads；
- AR decoder；
- B1 residual。

冻结：

- 前十二层 Emformer；
- B2 bridge；
- Qwen；
- BiCodec。

#### 与原文相同的部分

- 一个 shared streaming encoder；
- ASR CTC、NAR-S2TT CTC、AR-S2TT 三项同时训练；
- 三项权重 `4/4/8` 与原文一致；
- downstream translation loss 能回传到部分 encoder；
- 使用同一个样本的 source speech、source text、target text 和 target semantic supervision。

#### 与原文不同的部分

1. 没有 `L_S2UT` NAR unit CTC；
2. AR decoder仍能看完整 utterance hidden，没有 `g(i)` policy mask；
3. 固定 160/80，没有 multi-chunk；
4. 只解冻最后四层，不是 all modules joint optimization；
5. Phase3 NLL 是完整 source embedding prompt 的 offline-style teacher forcing，不是逐 prefix Micro-WRITE training；
6. BiCodec 不参与训练；
7. 多阶段 checkpoint 拼接，不是从同一模型端到端训练。

#### 实际结果

Step1 最佳 iter800：

- EN→ZH BLEU：21.99；
- ZH→EN BLEU：17.15；
- mean：19.57；
- 两方向 gate 均未通过。

判断：**当前最接近原文 joint-training principle 的阶段，但不是完整 StreamSpeech objective。**

### 5.9 Stage08 Step1-R：方向平衡修复

#### Motivation

原 Step1 偏向 EN→ZH，故使用严格 50:50 sampling、提高 Phase3 loss、提高 ZH→EN 权重。

#### 改动

- Phase3 weight：`0.5 → 2.0`；
- ZH→EN sample weight：`1.0 → 1.25`；
- 400 iterations，低 LR；
- 仍使用 `4/4/8` endpoint losses。

#### 结果

iter350：EN→ZH 21.20、ZH→EN 20.19、mean 20.70。方向平衡改善，但仍未通过 22.95/22.46 gate。

#### 与原文比较

方向平衡 sampling 和下游权重是 UniSS 数据/模型特定修复，原文没有对应设计。

判断：**工程修复，不增加原文复现度。**

### 5.10 Stage08 Step2：Qwen LoRA + offline replay

#### Motivation

让冻结的 Phase3 Qwen 适应 predicted streaming Emformer/B1 embeddings，同时用原 offline `source_glm` replay 防止遗忘。

#### Objective

\[
L=0.70L_{streaming-Phase3}+0.30L_{offline-replay}.
\]

只训练 Qwen `q_proj/v_proj` rank-8 LoRA。Step1-R 未过 gate，因此该实验明确标记 research-only。

#### 结果

iter100：EN→ZH 21.85、ZH→EN 20.07、mean 20.96，只比 Step1-R mean 增加 0.26。

#### 与原文比较

原文没有 LLM、LoRA、offline replay，也不把 AR translation 和 unit generation拆成这种适配过程。

判断：**UniSS 自研后端适配。**

### 5.11 Stage09–13：运行时、音频和 Demo

这些阶段没有新的模型训练：

- Stage09：真实 `Emformer.infer` cache + CTC-count policy；
- Stage10：Qwen KV-cache Micro-WRITE；
- Stage11：BiCodec incremental audio；
- Stage12：两样本双向研究评估；
- Stage13：公网 Gradio。

Stage09 的 policy 原理与原文最接近；Stage10/11 是将原文 AR text + NAR unit 后端替换为 UniSS Phase3 + BiCodec 后的工程实现。

当前 Stage12 结果：

| Direction | BLEU | First WRITE | First audio NCA | First audio CA | 结论 |
|---|---:|---:|---:|---:|---|
| EN→ZH | 2.26 | 560 ms | 880 ms | 5.16 s | 很早但文本重复，质量失败 |
| ZH→EN | 14.35 | 2160 ms | 10.64 s | 79.63 s | 在线 semantic 全拒绝，使用 final fallback |

这证明 runtime 链路接通，但不证明原文训练方法已经成功复现。

## 6. 原文可移植部分的逐项矩阵

| 原文可移植要素 | 当前状态 | 当前对应 Stage | 结论 |
|---|---|---|---|
| `(source speech, source text, target text, target units)` 四类监督 | 部分完成 | Stage01 | target BiCodec 已存在，但未作为 NAR unit CTC target |
| Source ASR CTC | 已实现 | Stage02/03/03b/08 | objective 对应 |
| Target NAR-S2TT CTC | 已实现 | Stage02/03/03b/08 | objective 对应，但质量仍低 |
| AR-S2TT CE | 已实现 | Stage03b/08 | loss/权重对应 |
| CTC 对齐限制 AR source prefix `g(i)` | 未实现 | 无 | 当前最大训练缺口之一 |
| Target-text → unit NAR CTC | 未实现 | 无 | 当前最大训练缺口之一 |
| T2U upsampling | 未实现 | 无 | `target_bicodec/text length` 可用于重新估计 rate |
| 冻结 vocoder | 类似实现 | Stage11 BiCodec | 只有推理，没有对应 unit CTC 训练 |
| 四任务同次端到端 joint training | 未完整实现 | Stage08 Step1 最接近 | 缺 S2UT，且只部分解冻 |
| Multi-chunk random training | 未实现 | 无 | 当前固定 160/80 |
| 一个 checkpoint 支持多 latency | 未证明 | 无 | 当前没有训练时 chunk 随机化 |
| CTC-count READ/WRITE | 已实现并扩展 | Stage05/09 | policy 主体对应 |
| AR 增量翻译 | 工程替换 | Stage10 | Qwen KV-cache，不是原文 decoder |
| NAR 增量 unit generation | 未实现 | 无 | Stage11 是 AR semantic span 的 codec 解码 |
| ASR-BLEU | 未用于当前 Stage12 正式集 | 后续待补 | 当前主要是 Text-BLEU/chrF |
| AL/AP/DAL/LAAL/ATD | 未完整用于 Stage12 | 后续待补 | 目前主要报告 First WRITE、NCA/CA、RTF |
| NumChunks/Discontinuity | 未完整报告 | 后续待补 | 原文 streaming degree 指标缺失 |

## 7. 为什么当前 Demo 早，但质量很差

当前失败并不能说明 StreamSpeech 的 CTC policy 无效，更准确的原因是训练闭环没有按原文完成。

### 7.1 Policy 触发早，但 target CTC 质量低

Stage05 balanced validation 的 target committed unigram recall 只有约 15.96%。Policy 可以稳定、零回滚地提交错误或不完整 target count。

### 7.2 AR/Qwen 没有在相同 prefix policy 下训练

Stage03b AR decoder和 Stage08 Phase3 NLL都主要在完整 utterance source 表示上训练；Stage10 推理却在 560/720/880 ms source prefix 上执行 Micro-WRITE。

这相当于：

```text
训练：听完整句子后翻译
推理：只听到半句话就要求不可逆输出
```

原文 `g(i)` 的作用正是消除这个 mismatch。当前没有移植这一训练机制。

### 7.3 没有原文 NAR T2U，目标 semantic 仍是长序列自回归

原文把长 unit 序列交给 NAR CTC，因此在每次新 target text 出现时可以快速同步生成对应 units。当前 Phase3 需要自回归生成 text/semantic structure，早期结构经常不完整或 repetitive，Stage11 只能大量 reject。

### 7.4 固定 160 ms 训练不等于 multi-latency robustness

固定小 chunk 会提高时间分辨率，但不自动保证早期翻译质量。原文通过 multi-chunk 同时看到小 chunk、中 chunk 和 offline context，维持 latency-quality Pareto。当前没有这种 regularization。

### 7.5 未通过 gate 的 checkpoint 被用于下游流水线

Stage08 Step1、Step1-R 都没有通过正式双向 BLEU gate。Step2是在用户要求下继续的 research-only pipeline validation。Stage09–13 因此应解释为“验证整条系统是否能运行”，而不是“质量合格的 StreamSpeech 移植模型”。

## 8. 如果要真正移植原文可移植方案，应该补什么

建议新建完全隔离的 `streamspeech_faithful_objective_v1/`，不覆盖当前 Stage00–13。

### 8.1 Faithful-A：补 multi-chunk Emformer training

训练时对同一个 Emformer checkpoint随机选择：

```text
160 / 320 / 640 / 960 / 1280 ms / offline
```

Emformer 本身 segment geometry可以保留，但需要控制训练时可见 source 范围和/或组合多个 segment形成不同 policy chunk。目标是让同一个模型推理时支持多 latency，而不是只在固定 160 ms 上过拟合。

### 8.2 Faithful-B：实现 CTC-guided `g(i)` AR training mask

对每个 batch：

1. 从 ASR CTC posterior 计算 source token expected count；
2. 从 NAR-S2TT CTC posterior 计算 target token expected count；
3. 为每个 target position `i` 计算 source boundary `g(i)`；
4. 构造 AR decoder cross-attention mask；
5. 只允许 `y_i` 看 `X≤g(i)`；
6. 保留 offline replay/multi-chunk 中的 full-context样本作为质量锚点。

这一步比继续调 Stage05 threshold 更重要，因为它让训练分布与 Micro-WRITE 推理分布一致。

### 8.3 Faithful-C：实现 BiCodec NAR Text-to-Unit CTC

不需要使用 mHuBERT km1000，可以把原文 `U` 替换为当前 `target_bicodec` semantic units：

```text
AR target text hidden
  → T2U encoder
  → learned/deterministic upsampling
  → causal NAR semantic decoder
  → CTC over BiCodec semantic vocabulary
  → existing BiCodec decoder
```

`bicodec_global` 作为固定/预测 speaker condition，不进入普通 CTC path。上采样率不应直接照搬 `r=25`，应根据 UniST 中：

```text
target_bicodec_length / target_text_token_length
```

的 p50/p95 重新选择，并检查 CTC path feasibility。

### 8.4 Faithful-D：一次真正的四任务联合训练

第一版建议严格使用原文相对权重：

\[
L_{faithful}=
1L_{BiCodec-CTC}
+8L_{AR-S2TT-policy}
+4L_{ASR-CTC}
+4L_{NAR-S2TT-CTC}.
\]

为了保护已有 Phase3，可增加小权重兼容项，但必须明确标记为 UniSS extension：

\[
L=L_{faithful}+\lambda_{P3}L_{Phase3-replay}+\lambda_{distill}L_{offline-distill}.
\]

`lambda` 应显著小于原文四个主任务，避免再次由 Phase3 NLL 主导而破坏对齐任务。

### 8.5 Faithful-E：按原文方式评估

同一个 checkpoint 扫：

```text
chunk = 160 / 320 / 640 / 960 / 1280 ms / offline
```

每个点报告：

- speech quality：ASR-BLEU、BLASER/AutoPCP（可用时）；
- text quality：BLEU/chrF；
- latency：AL、AP、DAL、LAAL、ATD、StartOffset、EndOffset；
- computation-aware：上述 `_CA` 指标；
- streaming degree：NumChunks、Discontinuity Sum/Ave/Num；
- RTF；
- UniSS 特有：speaker similarity、semantic reject rate、fallback rate、rollback/conflict。

只有这样才能与原文 Figure 4/6 的 latency-quality Pareto 公平比较。

## 9. 推荐的最小验证顺序

如果目标是先验证原文方法是否能解决当前重复和低质量问题，建议按以下顺序，而不是继续扩展 Demo：

1. **只补 `g(i)` AR mask**，保持 Stage03b 其余参数不变；这是最直接的因果对照。
2. 加入 multi-chunk，比较固定 160 ms 与 multi-chunk 的 dev BLEU/first WRITE Pareto。
3. 在同一文本侧 checkpoint 上训练 BiCodec NAR unit CTC head。
4. 最后进行四任务联合微调，再接 Phase3 compatibility/replay。
5. 通过固定双向 quality gate 后，才重新运行 Stage09–13。

建议最关键的消融表：

| 实验 | ASR CTC | NAR-S2TT CTC | AR `g(i)` | Multi-chunk | BiCodec unit CTC | 目的 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Current Stage03b | ✓ | ✓ | ✗ | ✗ | ✗ | 当前基线 |
| F1 | ✓ | ✓ | ✓ | ✗ | ✗ | 验证 policy-conditioned AR |
| F2 | ✓ | ✓ | ✓ | ✓ | ✗ | 验证多 latency 鲁棒性 |
| F3 | ✓ | ✓ | ✓ | ✓ | ✓ | 最接近原文完整可移植 objective |
| F4 | ✓ | ✓ | ✓ | ✓ | ✓ | 加 UniSS Phase3 replay，验证兼容性 |

## 10. 最终判断

### 哪个阶段“真正按照原文 objective”训练？

- 如果只看 loss 名称和权重：**Stage03b** 最接近原文文本侧 `4/4/8`。
- 如果看 shared encoder joint optimization：**Stage08 Step1** 最接近原文多任务联合训练原则。
- 如果看 simultaneous policy：**Stage05/Stage09** 最接近原文 CTC-count READ/WRITE。
- 如果要求完整 objective、policy-conditioned AR、multi-chunk 和 NAR T2U 同时满足：**当前没有任何阶段满足。**

### 当前路线是否仍有价值？

有。当前路线已经完成了：

- 真 cache Emformer；
- 双语 ASR/NAR CTC heads；
- AR 辅助翻译；
- Phase3 bridge；
- CTC-count policy；
- Qwen KV-cache；
- BiCodec 增量音频；
- 公网 Demo。

它把所有工程接口跑通，并暴露了最关键的科学问题：**早期 policy 可以工作，但生成模型没有在相同 policy/prefix 分布下被正确训练。**

所以接下来最值得做的不是继续缩短 Stage05 threshold，而是补齐原文真正可移植的训练机制：

> `g(i)` policy-conditioned AR training + multi-chunk + BiCodec NAR unit CTC + 四任务联合优化。

## 11. 审计证据索引

### 当前 UniSS 实现

- 总入口：`experiments/uniss_streamspeech_ctc_v1/README.md`
- Stage01 数据：`experiments/uniss_streamspeech_ctc_v1/stage01_data/`
- Stage03 CTC：`experiments/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/`
- Stage03b AR：`experiments/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/ar_s2tt_v1/`
- Stage05 policy：`experiments/uniss_streamspeech_ctc_v1/stage05_ctc_policy/`
- Stage06 residual：`experiments/uniss_streamspeech_ctc_v1/stage06_b1_nar/`
- Stage08 joint：`experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/`
- Stage09 runtime：`experiments/uniss_streamspeech_ctc_v1/stage09_online_runtime/`
- Stage12 report：`reports/uniss_streamspeech_ctc_v1/stage12_stage09_11_bilingual_eval_v1.md`

### StreamSpeech 原始来源

- 论文：[https://arxiv.org/pdf/2406.03049](https://arxiv.org/pdf/2406.03049)
- GitHub：[https://github.com/ictnlp/StreamSpeech](https://github.com/ictnlp/StreamSpeech)
- 原始 model：`researches/ctc_unity/models/streamspeech_model.py`
- 原始 criterion：`researches/ctc_unity/criterions/speech_to_speech_ctc_asr_st_criterion.py`
- 原始 multitask config：`configs/fr-en/config_mtl_asr_st_ctcst.yaml`
- 原始训练脚本：`researches/ctc_unity/train_scripts/train.simul-s2st.sh`
- 原始 S2ST agent：`agent/speech_to_speech.streamspeech.agent.py`
- 原始数据处理：`preprocess_scripts/README.md`
