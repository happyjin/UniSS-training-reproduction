# 分布式训练：大数据量分层预热、加载与训练的主流方案（汇总）

> 本文档汇总了一次关于“大规模分布式训练中，数据如何从原始资产走到 GPU”的系统性讨论。
> 内容聚焦**行业主流实践**，不涉及任何具体项目的实现细节。
>
> **文档组织**：分为三个部分——
> - **第一部分 · 概览**：一张图看懂全链路与贯穿原则；
> - **第二部分 · 数据流水线各环节（按执行顺序）**：采集 → 汇聚 → 清洗 → 标注 → 采样配比 → 分片打包 → 预热 → 加载 → 训练，每节对应一个**具体的、可执行的顺序环节**，并在各环节内说明它如何支撑**资产复用与数据飞轮**；
> - **第三部分 · 支撑系统与工具**。

---

## 目录

**第一部分 · 概览**
1. [整体链路概览](#1-整体链路概览)

**第二部分 · 数据流水线各环节（按执行顺序）**
2. [数据生产/采集（Data Production/Collection）](#2-数据生产采集data-productioncollection)
3. [汇聚：多源落地数据湖 + 每样本元数据列（Ingestion / Aggregation）](#3-汇聚多源落地数据湖--每样本元数据列ingestion--aggregation)
4. [数据清洗（Data Cleaning）](#4-数据清洗data-cleaning)
5. [数据标注（Data Annotation）](#5-数据标注data-annotation)
6. [数据采样与配比（Sampling & Mixing）](#6-数据采样与配比sampling--mixing)
7. [分片与打包（Sharding & Packaging）](#7-分片与打包sharding--packaging)
8. [分层预热（Tiered Pre-warm）](#8-分层预热tiered-pre-warm)
   - 8.6 [案例：ms-swift 下 rank 数据顺序的可预测性与预热调度（本仓库）](#86-案例ms-swift-下-rank-数据顺序的可预测性与预热调度本仓库)
9. [数据加载（Loading）](#9-数据加载loading)
10. [分布式训练（Training）](#10-分布式训练training)

**第三部分 · 支撑系统与工具**
11. [支撑上述方案的主流系统/工具](#11-支撑上述方案的主流系统工具)

---

# 第一部分 · 概览

## 1. 整体链路概览

大数据量训练的数据生命周期是一条**贯穿始终的流水线**，以 `shard`（数据分片）为统一抽象：

```
[数据采集：爬虫/日志/合成/授权语料/用户生成/多模态]
        ↓
[汇聚：多源按统一 schema 落地数据湖 + 每样本元数据列]
        ↓
[数据清洗：解析提取 → 编码/语言识别 → 基础去噪 → 质量过滤 → 去重 → 毒性/合规 → 质量打分]
        ↓
[数据标注：指令-回答对 / 偏好对 / 质量·领域标签 / 纠错精标]
        ↓
[采样与配比：确定本次训练子集及各源比例]
        ↓
[分片与打包：切分样本 + 编码/格式/压缩，写入分片文件]
        ↓
[本次训练数据预热：对象存储 → 共享FS → 节点本地NVMe缓存]
        ↓
[流式分片加载 + 计算/IO 重叠]
        ↓
[分布式训练 + checkpoint 容错]
```

核心思想：**预热只是“把已成型、已选型、已切好、已打包的本次训练数据搬到热层”**；
它前面的采集、汇聚、清洗、标注、采样配比、分片打包才是决定训练数据质量与效率的上游工序，且其产物（manifest + shard 布局 + 元数据）直接约束了预热该搬什么、搬到哪、怎么与 rank 对齐。

### 1.1 贯穿全链路的核心原则：资产层 vs 任务层解耦

- **资产层（一次性、重活只做一次）**：上游的汇聚/清洗/标注/分片打包尽量做成**带元数据与版本的持久数据资产**；过滤/打分/去重/标注的结果以**元数据列 / 新分区**沉淀，且**非销毁性**（不删原始样本），因此同一份资产可被任意模型、任意训练任务反复消费。
- **任务层（轻量、按需组装）**：每个训练任务只做**声明式轻量组装**——选哪些源、配比多少、引用哪个资产版本、过滤用什么阈值——**不重写 PB 级数据**。
- **数据飞轮**：数据源动态更新以**增量**追加进资产；人工精标以**新分区/新标记**写回资产；下一轮训练只需引用新版本即可，从而把“几百 PB 重做”变成“分钟级挑选”，实现快速迭代。
- 第 3~7 节会在各自环节具体说明如何支撑这一复用与飞轮；第 7.4 节专门讨论分片打包粒度对预热的约束。

---

# 第二部分 · 数据流水线各环节（按执行顺序）

## 2. 数据生产/采集（Data Production/Collection）

决定“原始数据从哪来”，是整个链路最上游、也最容易被忽略的一环。一份数据资产往往由多种来源汇聚而成。

- **来源类型**：
  - **网页爬虫**：大规模爬取公开网页（如 Common Crawl 类语料），是 LLM 预训练语料的主体；
  - **业务/产品日志**：搜索日志、对话记录、点击流、App 内行为等第一方数据；
  - **合成数据（Synthetic）**：用强模型生成（蒸馏、自我博弈、指令演化）、数据增广，弥补真实数据稀缺（尤其推理、对齐、长尾领域）；
  - **授权/采购语料**：书籍、新闻、论文、代码仓库、百科等带版权的合规语料；
  - **用户生成内容（UGC）/社区贡献**：论坛、问答、开源项目等；
  - **多模态来源**：音频（录音/播客/通话）、图像、视频及其转写/描述。
- **采集方式**：
  - **批式采集**（一次性全量抓取） vs **流式采集**（持续增量追加，接数据飞轮）；
  - **去重种子/URL 级**初步去重，避免重复抓取；
  - **合规约束**：遵守 robots.txt、license、版权与隐私法规，采集即打来源/授权标记；
  - **采集即带元数据**：来源、采集时间、原始 URL/ID、语种猜测等元数据随原始样本一起保留，供下游清洗与溯源。

> 采集阶段产出的“原始资产”通常体量最大、最脏，后面的汇聚、清洗、标注与资产化就是把它变成可复用、可筛选的训练就绪数据。

---

## 3. 汇聚：多源落地数据湖 + 每样本元数据列（Ingestion / Aggregation）

> 严格说，“汇聚”是把**多源原始数据按统一 schema 落地到数据湖**这一步，发生在清洗之前，是 lakehouse「先汇聚（bronze）再清洗（silver）」的标准顺序。它和后面“清洗后按配比混合”不是一回事。

- **定义**：把采集到的多源异构原始数据，按**统一 schema** 落地到数据湖（对象存储 + 表格式，如 Iceberg/Delta/Hudi），并为每样本附带**元数据列**：来源、采集时间、原始 ID/URL、语种猜测、license/授权标记、采集批次等。
- **为什么在清洗之前**：
  - 各源格式/语义异构，先归一化落地，再在湖上统一做清洗/标注，便于**增量摄入、版本化、溯源**；
  - 清洗/标注/采样都变成“在湖上对该表/分区做操作”，无需反复搬动原始数据。
- **关键特性**：
  - **统一 schema / 元数据注册表（metadata registry）**：所有源共享一套字段定义，下游才能按列查询与下推过滤；
  - **幂等 + 增量摄入**：新数据追加为新分区，不重建全量；
  - **内容寻址 / 去重种子**：用 URL/哈希做摄入级去重，避免重复落盘；
  - **可溯源（lineage）**：每样本可知来自哪个源、哪次采集、哪版清洗；
  - **分层存储**：常见 bronze（原始）→ silver（清洗后）→ gold（精选/标注后）三态，至少区分 raw 与 curated。
- **资产复用视角（连接飞轮）**：
  - **版本化**：资产按内容寻址（hash）做不可变快照，每个版本是一个 snapshot；训练任务**引用某版本**即可启动，新增数据只产生 delta 版本、不碰旧数据（重活绝不会每次重做）。
  - **增量摄入（接飞轮）**：数据源动态更新 → 新增分区进入管道，只跑“过滤 + 对已有去重索引做成员判定 + 打分”的增量步骤（不重算全量）→ 作为资产新 delta 版本。管道可常驻，使“训练就绪数据”永远是最新的；启动训练只是对当前资产**快照**。
- **地位**：这是“资产层”的物理底座；第 4~5 节的清洗、标注，以及第 7 节的分片打包，全部建立在这张带元数据的、可版本化/可增量的湖表之上。

---

## 4. 数据清洗（Data Cleaning）

把“原始脏数据”变成“高质量、结构化、带元数据”的资产。这是最细粒度、也是最耗工程的环节，主流按以下子环节逐级推进（各环节输出尽量以**元数据标记**持久化，而非物理删除，便于不同任务复用）：

- **格式解析 / 提取（Parsing & Extraction）**：
  - 文本：HTML→正文（Boilerplate 去除）、PDF/DOCX→纯文本、Markdown 规整；
  - 语音：ASR 转写、说话人分离、语种判定；
  - 图像/视频：OCR 提取文字、视频抽帧+描述、跨模态对齐；
  - 目标：把异构原始格式统一为可训练的“样本 + 结构化字段”。
- **编码与语言识别**：
  - 检测并统一编码（UTF-8 等），修复乱码；
  - 语言 ID（fastText langid 等）做语种路由——不同语种走不同后续阈值/流程；
  - 语种/脚本标记的元数据用于后续按语种配比。
- **基础清洗 / 去噪**：
  - 去除控制字符、乱码、重复行、截断残片；
  - 规范化空白、标点、Unicode 全半角；
  - 去除/脱敏明显个人信息（PII）占位。
- **质量过滤（Quality Filtering，含自动打标）**：
  - **规则法**：长度下限、符号/字母比、停用词比、重复 n-gram 比、困惑度（perplexity）阈值；
  - **模型法**：质量分类器（fastText/神经网络）、领域分类器，对每样本打质量分/领域标签；
  - 输出作为“质量分”元数据列，不同任务改阈值即可复用，不必重过滤。
  - *注：此处的质量/领域标记属于“自动打标”（模型产元数据），与第 5 节更 deliberate 的**指令数据标注**不同，两者共同构成“打标”这一大类。*
- **去重（Dedup）**：
  - **精确去重**：归一化（去空白/小写/标点）后做 hash，去除完全重复；
  - **近似去重**：MinHash + LSH、SimHash、后缀数组等，去除近重复文档/片段；
  - 文档级 + 片段级两道；结果打 `is_dup` 标记（而非物理删除），供任务选严格/宽松去重。
- **毒性 / 安全过滤（Toxicity & Safety）**：
  - 毒性、偏见、暴力/NSFW 分类器过滤或降权；
  - 安全红线内容拦截，是合规与上线前的必过环节。
- **隐私 / 合规（Privacy & Compliance）**：
  - PII 检测与脱敏（redaction）、密钥/手机号/邮箱掩码；
  - license / 版权校验、合规审查，标注可商用性。
- **质量打分（Scoring）**：
  - 综合质量、领域、难度、语种等维度给每样本打分，作为后续**采样与配比**的核心元数据；
  - 这一步把“清洗结果”直接喂给下游的配比决策（见第 6 节）。

> 关键原则：**清洗各环节“非销毁性”**——原始样本保留，过滤/打分/去重结果以元数据列持久化。这正是资产可被多任务复用的基础：不同任务对同一份清洗后资产只需**改阈值/换分类器**（如“本轮只要质量分>0.8 的”），不必重清洗；也是第 1.1 节飞轮能成立的前提。

---

## 5. 数据标注（Data Annotation）

把“原始/清洗后语料”提升为“可训练监督信号”的关键环节，尤其对**指令数据（instruction data）质量**起决定性作用。它贯穿但不等同于清洗——清洗期的是“自动打标”（模型产元数据，见 §4），本节是更 deliberate 的**生产性 / 精炼标注**。本节先明确它在流水线中的定位与边界，再展开标注类型、标注流水线、方法、质量管控与资产化沉淀，最后落到数据飞轮闭环与语音场景。

### 5.1 在流水线中的定位与边界

- **输入**：已完成 §4 清洗、带每样本元数据列的资产（质量分、领域、语种、来源等）。
- **输出**：带**标注信号**的样本——以新样本（指令-回答对、偏好对）或新元数据列（质量/难度/安全/领域标签）形式写回资产。
- **与 §4 自动打标的边界**（这是单列本节的根本原因）：
  - §4 的标记是**模型自动产元数据**（质量分、语言 ID、毒性、去重指纹等），低成本、规模化、无需人工判断，服务于“过滤/筛选”；
  - 本节是**生产性 / 精炼标注**：把语料“变成”可用的监督目标或偏好信号，常含人工判断或强模型生成后的人类校验，价值高、常是瓶颈，必须作为独立阶段管理（产能、成本、质检、版本都单独跟踪）。

### 5.2 标注任务类型矩阵

按“标注产出的是什么信号”分类（LLM/多模态语境）：

- **指令-回答对（SFT 数据）**：把文本/知识/对话转化为 `(instruction, response)`，可含 CoT、多轮对话、工具调用、代码执行。这是“**打标 = 生产目标文本**”，是监督训练主粮。
- **偏好标注（RLHF / DPO / RM 训练）**：对同指令的多个候选回答标 `chosen / rejected`，产偏好对；或打** pointwise 质量分**训练奖励模型。
- **质量 / 难度 / 安全标注**：对指令或回答打分级或标签（如“指令清晰度”“回答正确性”“不安全”）。与 §4 自动质量分**互补**：自动分是粗筛，此处更精细、常人工，直接喂给 §6 配比与 §7 分片（按质量 tier 内聚）。
- **领域 / 任务 / 语种标注**：标领域、任务类型、语种、模态，供 §6 分层抽样与 §7 语义内聚分片。
- **纠错 / 精标（飞轮反馈）**：对模型错误样本人工修正答案、重写指令、补偏好，沉淀为新标签/新分区（见 5.8 飞轮闭环）。
- **对齐专用标注**：有用性、无害性、真实性（factuality）、指令遵循度等多维偏好，常一个样本对应多维标签。

### 5.3 标注流水线（从样本到可用资产）

一套可落地的标注作业通常分 5 步，与数据湖协同：

1. **样本抽取**：从 §4 资产按策略抽样（常结合主动学习挑难例，见 5.4）；
2. **任务设计**：写标注规范（guideline）、设计标注模板、定义标签体系与边界 case；
3. **标注执行**：人工/众包/LLM-as-annotator（见 5.4）产标注；
4. **质检与验收**：golden set 抽检、多人一致度（IAA）、争议仲裁；
5. **沉淀入库**：以新分区/新样本/新列写回资产（见 5.7），更新 manifest 与索引。

### 5.4 标注方法：人工、LLM-as-annotator 与主动学习

- **纯人工 / 专业标注团队**：质量最高、成本最高，用于核心 SFT、偏好、安全等“命脉数据”。
- **众包**：扩产能，但需强质检与一致性约束。
- **LLM-as-annotator（human-in-the-loop）**：用强模型先生成标注（写回答、写偏好理由、打标签），人工只做**校验/修正**，标注产能可放大 10× 以上；关键风险是“模型自我偏好”偏差，需用更强的裁判模型或分层抽样人工复核来抑制。
- **主动学习（Active Learning）**：用**当前模型的不确定性 / 错误分析**挑高价值难例给人标，而非随机抽——把有限人工预算花在边际信息最大的样本上。这与 5.8 飞轮闭环同源：要标什么，由模型当前弱点决定。
- **偏好数据专门收集**：RLHF/DPO 偏好对需专门收集流程（同 prompt 多模型/多采样生成候选 → 人/强模型判 chosen/rejected），并控制标注者偏差与 prompt 泄露。

### 5.5 质量管控（让标注可信）

- **标注规范 + 培训**：统一 guideline，减少主观分歧。
- **Golden set / 暗集抽检**：插入已知答案的样本测标注者准确率。
- **多人一致度（Inter-Annotator Agreement, IAA）**：如 Cohen's κ / Krippendorff's α，低于阈值重标或仲裁。
- **争议仲裁与回流**：不一致样本进仲裁队列；反复出错的标注者降权。
- **可追溯**：每条标注带标注者/模型、时间、版本、方法（人工/LLM/混合），便于 §6 配比时按“标注可信度”分层或打折。

### 5.6 成本、产能与配比杠杆

- 人工标注是整条流水线最贵的环节之一，因此**“该标的才标、能用模型标的绝不用人”**是核心原则：用 5.4 的 LLM-as-annotator + 主动学习把人工集中在高价值难例。
- 标注数据的**单位 token 训练价值远高于爬取数据**，故在 §6 配比时通过温度采样（6.4）/ 上采样（6.8）显著放大其曝光——标注环节直接决定“下一轮训练有多好”。
- 标注产能也决定飞轮转速（5.8）：标注越快、越准，模型迭代越短。

### 5.7 标注产物的资产化沉淀（连接 §1.1 复用 / §6 / §7）

- 以**新元数据列 / 新分区 / 新样本**形式写回数据湖，**不破坏原始资产**（非销毁性，呼应 §1.1）。
- 指令-回答对作为独立 **SFT 资产分区**；偏好对作为独立 **RLHF/DPO 分区**；质量/难度标签作为**元数据列**；纠错结果写入**新标注分区**。
- 不同训练任务（SFT / RLHF / DPO）按 §6 recipe **引用不同标注分区/列**即可，无需重建数据——这是“一份资产服务多种任务”的关键体现。
- 标注的**领域/质量/语种标签**直接被 §7 用于“语义内聚分片”（同质量 tier / 同域同 shard），提升 §8 预热时的整 shard 命中率。

### 5.8 数据飞轮闭环（标注是下游反馈的入口）

- 模型在下游训练（§10）并评测后，错误分析挑出的难例回到本环节做**人工 / LLM 精准标注**：产出标签/纠错/指令-回答对/偏好对，用来**更新质量分类器（主动学习）**和/或直接给样本打新标记列、写入新标注分区——这些都不需要重处理原始数据。
- 新标注只需“**追加 + 索引更新 + 声明式选择**”，从标注完成到下一轮训练可用常可压到小时级（配合 §6 视图化 recipe 即时纳入）。
- 闭环：`当前资产训练模型 → 评测/错误分析 → 人工标注难例（本环节）→ 作为定向数据（过采样）进入下一轮`。标注因此既是**最上游的指令数据生产**，也是**下游迭代时把人工判断反馈回资产的通道**，驱动数据飞轮越转越快。

### 5.9 语音 / 多模态场景的特殊标注（连接 §9.6）

对 Qwen-OmniSpeak 这类语音/同传模型（见 §9.6），标注还要覆盖模态对齐信号：

- **语音转写 / 翻译对齐标注**：音频 ↔ 文本、源语 ↔ 目标语的对齐转写与翻译对，是 ASR / 翻译训练的核心监督；
- **同传策略标注**：流式/同传场景下的“何时输出、输出多少、是否打断”等策略信号，常需专家标注或强模型生成后人工修正；
- **口语化 / 副语言标注**：语气、停顿、情感等副语言标签，用于提升语音自然度；
- **多模态指令标注**：以语音为指令、以语音/文本为回答的指令对，标注规范需额外约束音频质量、对齐准确性与跨模态一致性。
- 这些都以独立标注分区/列沉淀，纳入 §6 recipe 与 §7 语义内聚分片，并通过 5.8 飞轮随新精标快速迭代。

---

## 6. 数据采样与配比（Sampling & Mixing）

建立在**已清洗、已标注资产**之上，决定“本次训练集由哪些源、各占多少、按什么节奏喂入”。它位于 §5 标注之后、§7 分片打包之前——**先有“采样配方（recipe）”，才有“按配方选哪些 shard 去预热/分片”**。

> 与 §1.1 复用视角的关系：本节产出的 recipe 是一份**声明式配置 + 一份 manifest / 视图**，而非一份物理上重新按比例写好的数据集。这正是“任务层不重写几百 PB 数据”的核心落点：换配比只需改配置，下游 §7/§8 据此重新选取 shard，无需全量 repack。

### 6.1 输入、输出与目标

- **输入**：清洗+标注后、带**每样本元数据列**的资产数据湖（质量分、领域、语种、来源、标注类型、难度、长度等，见 §4/§5）。这些元数据是采样可下推查询、可分层、可复现的前提。
- **输出**：一份 **sampling recipe**，明确：
  - 参与本次训练的**源/域集合**及其**权重**；
  - 每个源的**采样基数（epoch 数 / 抽多少）**；
  - **分层 / 课程 / 退火**的调度曲线；
  - 以及一份指向资产 shard 的 **manifest / 视图**（带采样比例，供 §7/§8 消费）。
- **目标**：在预算（token 数 / 步数 / 墙钟时间）约束下，最大化下游模型在**目标评测集**上的表现——本质是“多目标语料配比优化”，不是简单按比例拼。

### 6.2 Sample-level 与 Token-level 配比（先分清维度）

配比有**两个常被混淆的维度**，必须先确定口径，否则“比例”毫无意义：

- **Sample-level（按样本数配比）**：每个源贡献相等数量的样本。适合各源平均长度接近、或样本数本身是稀缺资源（如人工标注对）的场景。
- **Token-level（按 token 数配比）**：每个源贡献相等数量的 token。大模型训练几乎都按 token 算“训练量”，因为 loss 是在 token 上定义的。当不同源平均长度差异大（如长文档 vs 短指令 vs 语音片段）时，**sample-level 会严重偏袒长样本源**，必须用 token-level。
- **实操建议**：先在 token-level 上定总预算与比例，再换算成各源样本数（样本数 = 预算 token × 权重 / 该源平均长度）。平均长度来自 §4 统计，随资产更新而更新。

### 6.3 静态配比（经验 / 人工设定）

不依赖训练反馈、一次性定好的配比，仍是工业界主流基线：

- **经验配方**：人工按领域知识设定（如 The Pile 的 22 子集比例、GPT-3 的多源 mix、LLaMA 的 Web/Book/Code/... 权重）。优点：可控、可解释、易复现；缺点：依赖专家经验、对新任务迁移差。
- **两种落地**：
  - **逻辑混合（推荐）**：不物理挪动数据，只生成一份指向各源 shard 的 manifest + 采样比例（见 6.10），由 loader 按权重抽取。换配方零成本。
  - **物理混合**：真正把各源按比例 shuffle 到一起落盘。好处是 locality 好、loader 简单；坏处是换配方就要重洗重落，回到“重做”问题——**仅当配比长期稳定且追求极致读取性能时才考虑**。

### 6.4 温度 / 幂次采样（Temperature / Power sampling）

当“高质量小源”与“低质量大源”并存时，单纯按 token 等比采样会让小源被淹没。**温度采样（也称幂次 / α-sampling）**是业界最常用的一剂解药：

- **做法**：对源 i，按 `p_i ∝ w_i^{1/T}`（或 `n_i ∝ N_i^{α}`，α<1 为幂次）重算采样量，再归一化。
  - `T=1`（或 α=1）：严格按比例；
  - `T→∞`（α→0）：所有源等样本（抹平体量差，偏向小源/高质量小源）；
  - `T→0`（α→1）：基本只取最大源（接近体量主导）。
- **直觉**：通过 T/α 把“体量主导”拉向“等权覆盖”，让高质量但体量小的源（人工标注、难例、领域精料）获得更高相对曝光。常用于低资源语种、人工标注对、难例集的上采样。
- **注意**：T 不是拍脑袋，常配合 6.5 的混合定律或小规模消融来定。

### 6.5 动态配比：数据混合定律（Data-mixing laws / DoReMi）

配比不是定一次就完，主流用**小规模代理实验反推大模型的“最优配比”**：

- **DoReMi 思路**：先训一个**小 proxy 模型**（同架构、少步数），用**分布鲁棒优化（group DRO）**让它在各源上“最差源也不差”，得到一组对各源的参考权重；再用这组权重去训大模型。本质是“用便宜的实验估计每源边际收益”。
- **Data-mixing laws**：拟合“配比 → 目标 loss/评测分”的缩放规律，预测换配比后大模型表现，从而选权重，避免每次都烧全量训练来验证。
- **在线 / 课程自适应**：训练中发现某域 loss 降不动或评测掉点，就动态调高该源权重（见 6.6）。这要求 recipe 是**可热更新**的（声明式 + manifest 的优势再次体现）。

### 6.6 课程学习与退火调度（Curriculum / Annealing）

配比不是常量的，常随训练进度演化：

- **课程式（Curriculum）**：早期偏通用/大规模易学数据打底，中后期逐步加入难例、领域、人工标注等“硬料”。用一条调度曲线（如线性/余弦插值两套权重）随 step 切换。
- **退火（Annealing / Cooldown）**：训练末期切换到“高质量精选小集”（高质量 web、教材、指令、人工标注）做少量 step 退火，显著提升下游评测——这是 Llama3、DeepSeek 等都采用的收尾手法。
- **实施要点**：课程/退火曲线要写进 recipe（带 step 区间与对应权重），loader 按当前 step 取对应权重分桶；若用物理混合则需提前切好“阶段数据集”。

### 6.7 领域 / 质量 / 语种 分层抽样与覆盖约束

单纯按权重抽容易在某些长尾维度上“抽空”，需要**硬约束**兜底：

- **分层抽样（Stratified sampling）**：按元数据维度（领域、语种、质量 tier、标注类型）分桶，保证每个桶都有**最小曝光量**（floor），避免小域在随机采样下被淹没。
- **覆盖约束 / 黑名单**：关键语种、关键能力域设置“至少 N token”的保底；敏感/失效源设置黑名单不参与。
- **可下推查询**：这些都基于 §4/§5 留下的元数据列做谓词下推（如 `lang=='zh' AND quality>=0.8 AND domain IN (...)`），**极快**，是“从几百 PB 大池子里秒级抽出本次子集”的关键。

### 6.8 每源 Epoch 控制与人工标注数据的上采样（连接飞轮）

不同源天然要“过不同遍数”：

- **Epoch 数 = 采样基数 / 源总量**。体量巨大的 web 源可能 <1 epoch（只抽一部分），而稀有源、人工标注对可能多 epoch 上采样。
- **人工标注数据上采样（飞轮关键）**：§5 标注产出的精标数据体量永远小于爬取数据，但其单位 token 的“信息密度/训练价值”极高。配比上常通过 6.4 的温度采样或显式 epoch 数把它们**显著上采样**，让模型充分吸收人类偏好/指令/纠错信号。
- **飞轮连接**：§3/§5 讲到“数据源动态更新 + 人工精准打标快速反馈”。本节的 recipe 是飞轮的**消费端开关**——新增精标、新源一进来，只要在资产资产上更新元数据，改一份 recipe（上采样新标注、降权失效源）即可让下一轮训练立刻吸收，无需重洗全量（呼应 §1.1 与 6.10 视图化）。

### 6.9 确定性、可复现与去偏

- **确定性**：采样必须带固定 seed + 确定性 shuffle（如基于 shard 内 offset 的可复现排序），保证同一 recipe 两次抽出完全一致的数据集，方便复现与消融。
- **去偏**：监控各维度实际曝光量与目标权重的偏差（运行中 drifting），必要时做重要性采样修正权重（importance sampling / reweight），保证稳态配比贴合 recipe。
- **预算封顶**：recipe 要声明总 token 预算，超出即当停（或循环到 epoch 上限），避免某源因体量动态膨胀而挤占其他源。

### 6.10 视图化 / 声明式采样（manifest / view，连接复用视角 §1.1）

这是把上面所有手段“低成本落地”的载体，也是消除“几百 PB 重做”的根本：

- **不物理混合**：recipe 只生成一份 **manifest / 视图**——一行指向资产某 shard，并带“该 shard 在本任务中的采样比例 / 起始 epoch / 分层桶”。loader（见 §9）或 §7 的轻量组装器据此在采样时实现混合。
- **换配方 = 改配置**：调整权重、课程曲线、上采样新标注，都只需改 recipe + 重新生成 manifest，**不碰资产数据**。
- **物化只做 delta**：仅当性能确需（如要极致 locality 做 §7 语义内聚分片、或 §8 预热要整 shard hit）时，才对**筛选后的子集**（通常远小于全量）做 repack，且只对新增/变更部分物化，避免全量重写。
- **与 §7/§8 的关系**：这份 manifest 同时是 §7 “按 recipe 选哪些 shard 去分片/语义内聚打包”的输入，也是 §8 “manifest 驱动预热、整 shard 选取”的输入——**一份 recipe 贯通采样→分片→预热**。

### 6.11 采样产物的下游衔接（→ §7 分片 / §8 预热）

- 采样输出 **recipe + manifest** 后：
  1. §7 据此**选取/生成语义内聚的 shard**（按 recipe 的源/域/语种分桶，使任务能整 shard 选取，见 §7.4.2）；
  2. §8 据此做**manifest 驱动的滚动预热**（只把 recipe 选中的 shard 逐级搬上热层，零浪费）；
  3. §9 loader 按 recipe 的权重/课程曲线在采样时实现最终混合。
- 一句话：**采样决定“喂什么”，分片决定“怎么切”，预热决定“放哪”，加载决定“怎么读”——四者由同一份 recipe / manifest 串联，避免任何环节重复物化全量数据。**

### 6.12 语音 / 多模态场景的特殊考量（连接 §9.6）

对 Qwen-OmniSpeak 这类语音/同传模型（见 §9.6），采样配比还有额外维度：

- **模态配比**：文本、ASR 语音、翻译对齐语音、同传策略数据等多路输入要定 token/样本混合比例；语音侧因 codec 编码（12Hz）长度特征与文本不同，务必用 **token-level** 配比。
- **课程式模态进阶**：常见“先单语 ASR 打底 → 再双语翻译对齐 → 末期加入同传策略/实时决策数据”的模态课程，对应 6.6 的调度曲线。
- **对齐数据稀缺性**：翻译/同传对齐对（平行语料、人工精标对）体量最小却最关键，靠 6.4 温度采样 + 6.8 上采样保其高曝光；新精标进来走 6.10 视图化即时纳入下一轮，正是数据飞轮在语音模型上的体现。

---

## 7. 分片与打包（Sharding & Packaging）

> **关于顺序的说明**：先“分片”还是先“打包”？物理上你理解得对——**先把样本按分片策略分配到各个 shard，再对每个 shard 做编码写入（即“分好片、再打包每个 shard”）**。本节把两者放在一节，是因为它们本就是**同一个“物化”步骤的两面**：
> - **打包方式** = “每个样本/每个 shard 怎么编码”的设计决策（格式、是否预 tokenize、是否 packing、是否压缩），这是在物化前就要定好的；
> - **分片策略** = “样本怎么切、切多大”的设计决策（shard 大小、分配方式、索引、rank 映射）；
> - **实际执行**：遍历样本 → 按分片策略归属到某 shard → 对该 shard 内样本编码并写入文件（分片+打包合一）。
>
> 所以“打包方式”作为设计先于执行，但物理落盘是“先分片、再打包每个 shard”。

### 7.1 分片策略（先决定怎么切）

- **Shard 大小权衡**：太小→元数据爆炸、随机性差、打开开销大；太大→单卡加载慢、容错粒度粗（一个 shard 损坏影响多）。主流在 **100MB ~ 几 GB / shard**，且常对齐到存储/网络的最优传输单元。
- **分片方式**：
  - **均匀随机分片**：无偏，混合比例交由 loader 控制；
  - **按源/域分片**：同源同 shard，便于按比例抽样和按源统计（也利于预热时整 shard 选取，见 7.4 节）；
  - **按 hash/key 分片**：同一样本稳定落在同一 shard，有利于本地缓存命中。
- **全局 Manifest / 索引**：记录每个 shard 的路径、样本数、字节大小、checksum、所属源，供调度与预热使用。
- **与 rank 的映射约定**：round-robin、范围分配或一致性哈希，决定“哪张卡/哪个节点读哪些 shard”，是预热把 shard 铺到对应节点本地缓存的依据。
- **资产复用视角**：资产层维护**规范化分片布局**（固定大小、按 hash 稳定落 shard），**多任务复用同一布局**；任务层尽量**复用现有 shard**，或用 manifest（虚拟数据集）直接引用资产 shard，做到**零拷贝**——这是第 1.1 节“任务层不重写数据”在分片层面的体现。

### 7.2 打包方式（再决定每个 shard 怎么编码）

- **序列化格式选型**：
  - `TFRecord` / `Parquet`（列式，支持谓词下推、按需列读取）；
  - `WebDataset`（tar 内按 key 分组，顺序流式、天然适合对象存储）；
  - `MDS`（MosaicML StreamingDataset，分片索引+数据分片，支持断点续训与节点间分发）；
  - `Arrow / Lance`、或预 tokenize 后的纯二进制（如 `.bin` token id 文件）。
- **是否预 tokenize**：提前转成 token id/特征落盘（训练时直接读张量），砍掉训练期大量 CPU 开销，但牺牲灵活性。
- **样本打包（Sequence Packing）**：把多个短样本拼接成固定长度序列，消除 padding 浪费——对短序列、语音片段类数据收益尤其大。
- **压缩**：shard 内 inline 压缩（zstd/gzip），降低存储与跨网络传输量，代价是读时解压。

### 7.3 物化执行（分片 + 打包合一）

- 典型循环：取 N 个样本 → 按 7.1 归属到某 shard → 按 7.2 编码（打包）后写入该 shard 文件 → 更新 manifest。
- 输出：**不可变分片文件 + manifest + 每样本元数据**，三者共同构成“训练就绪的资产”，供下游预热/加载/训练（第 8~10 节）消费。

### 7.4 分片打包对数据预热的影响（核心张力）

> 本小节承接“分片打包”决策，讨论它如何直接约束下游“预热（第 8 节）”的效率。核心问题是：**一个 shard 里是否只有部分样本是某任务需要的？** 这决定了预热整 shard 时是 100% 有效还是含大量无效数据。

#### 7.4.1 两种极端及其代价

- **A. 资产层规范分片（按原始/规范化数据打包，与任务无关）**
  - shard 在资产层切好（常按 hash 桶或按源/域），跨任务复用。
  - 代价：**一个 shard 里只有部分样本是某任务需要的**。预热时整 shard 搬上热层 → 带宽/热存储浪费，可能搬了 70% 用不上的数据。
  - 优点：shard 布局稳定、一次分片多次复用、不重复 repack（呼应 7.1 资产复用视角）。
- **B. 任务层定制分片（按任务过滤后打包）**
  - 过滤/配比/采样之后才打包，shard 内 100% 有效，预热零浪费。
  - 代价：**每个任务都要重新 repack**（回到“几百 PB 重做”的老问题），且丧失复用。

纯 A 浪费预热，纯 B 浪费打包。所以真实做法是折中。

#### 7.4.2 主流做法：分层 + 语义内聚分片

- **资产层（稳定、可复用、粗粒度内聚）**：不按“单任务过滤”切 shard，但**分片按语义内聚**：同源 / 同域 / 同质量层 / 同语种放在同一组 shard（或按 hash 桶使同一样本稳定落位）。这样任务可以**“整 shard 选取”**——例如“本轮只用中文+高质量”，就只预热中文高质量那批 shard，hit rate 很高，而不是从一个大杂烩 shard 里抠少量样本。这是复用与预热效率之间的关键妥协：**shard 不是为某个任务定制的，但足够“同质”，让任务能按 shard 粒度选取而非按样本抠**。
- **任务层（轻量、按需、两种落地）**：
  - 方式一：**in-loader 过滤**。直接预热资产 shard，loader 流式读、丢弃不匹配样本（reject sampling）。简单、复用最大化，代价是热层含少量无效数据（被丢弃的那部分）。适合任务间重叠度高、热层充裕的场景。
  - 方式二：**生成任务专属打包（materialized view）**。按 recipe 过滤后生成一份“100% 有效”的任务 shard，只对这些 shard 预热。代价是 repack，但 repack 的是**已筛选后的子集**（通常远小于全量），且可只对 delta 做。

#### 7.4.3 对预热策略的影响（关键）

- **资产粗分片 + in-loader 过滤**：预热以 shard 为单位，可能含无效样本。预热策略要接受“过量预热 / 容忍浪费”，但胜在 shard 布局稳定、可跨任务复用、预热一次多次用。
- **任务专属打包**：预热目标 = 任务 shard，100% 有效、零浪费，但每次任务先 repack（成本从预热转移到打包）。
- **更优主流实践**：**预热的目标不是“原始资产 shard”，而是“本次任务的 manifest（虚拟数据集）”**。预热系统读 manifest，只把任务真正需要的样本/shard 子集拉到热层。manifest 本身是元数据操作（便宜），无需先全量 repack——既避免整 shard 浪费，又避免全量重写。
- **样本级预热**：用支持随机访问/行组索引的格式（如 MDS、Arrow 按 row group），预热可以精确到“这些 N 个样本”而非“这些整个 shard”，从根本上消除“暖一个 shard 只有部分有效”的浪费。

#### 7.4.4 决策矩阵

| 维度 | 偏资产粗分片 + in-loader 过滤 | 偏任务专属打包 + 样本级预热 |
|---|---|---|
| 任务间数据重叠 | 高（共享大部分语料）✅ | 低（任务差异大）✅ |
| 热层成本/容量 | 充裕 ✅ | 紧张/昂贵 ✅ |
| repack 成本敏感度 | 高（不想重切）✅ | 低（愿为效率 repack）✅ |
| 预热浪费容忍 | 可接受 ✅ | 不可接受 ✅ |

> 现实里大厂常是**混合**：公共基础语料用资产粗分片复用 + in-loader 过滤；任务专属/高价值/小量的精标数据用任务专属打包、100% 有效预热。

---

## 8. 分层预热（Tiered Pre-warm）

### 8.1 存储层级

```
对象存储 (S3/COS/OSS，容量无限、便宜)
   ↓ 预热导入
共享高性能文件系统 (FSx for Lustre / JuiceFS / CephFS / GPFS，几百 GB/s 吞吐)
   ↓ 本地缓存/拷贝
本地 NVMe SSD (单卡数 GB/s，最低延迟)
```

### 8.2 主流预热实践

- **全量预热（Bulk import）**：训练启动前，用 HSM/`lfs hsm_import`、FSx 的 `import` 任务或自定义脚本，把训练集从对象存储整批导入共享 FS。优点是最简、训练时零抖动；缺点是需要等导入完成、占用预热存储容量。
- **分层/渐进预热**：先把元数据、索引和部分 shard 拉到热层，训练进程按 epoch 顺序后台异步拉取后续 shard（prefetch warm）。适合数据量远超热层容量、且访问有局部性的场景。
- **本地缓存预热**：每个计算节点把常用 shard 缓存到本地 NVMe，配合一致性哈希或 shard→node 映射，使同节点多卡复用本地副本，显著降低共享 FS 压力。
- **校验与去重**：预热阶段做文件级 checksum（md5/etag）和 shard 完整性校验，避免脏数据进入训练。

> **与上游的衔接**：预热搬多大数据、以什么粒度搬，由第 7.4 节的分片策略与任务 manifest 决定——主流以“本次任务的 manifest（必要时样本级）”为目标，只拉有效子集，避免整 shard 含无效数据的浪费。

### 8.3 预热如何“得知”训练的数据切分情况

预热系统要正确地把 shard 搬到对应节点的热层，必须和训练作业**共享同一份“数据放置契约”**，并能感知训练进度：

- **共享的放置契约（谁读哪些 shard）**：
  - **任务 manifest**（第 7 节产出）：本次训练需要哪些 shard、各 shard 的路径/大小/校验和、所属 `task`/`lang`（用于语义内聚选取）。
  - **sampler / 分片分配函数**：决定 shard 如何在 rank 间分配（round-robin、按 hash、或 MDS 的 point-to-point 分配）。训练侧用同一个函数，预热侧也用同一个函数 → 两侧算出的 `rank→shard` 完全一致。
  - **rank→node 映射**：多卡多节点下，`rank` 被固定绑到某台节点（如 `torchrun`/`Ray` 的 placement group）。预热侧据此把“某个 rank 的 shard”落到“该 rank 所在节点的本地 NVMe”。
- **如何感知训练进度（已消费到哪）**：
  - **checkpoint 游标**：checkpoint 存 `epoch / global_step / 已消费样本数`（第 10 节容错），预热侧据此从断点继续，而不是从头暖。
  - **进度回调 / 心跳**：训练循环周期上报 `(global_step, 当前 shard 下标)`，预热控制器更新“前瞻窗口”。
  - **可纯计算**：因 sampler 是确定性的、batch size 已知，控制器**无需被通知**也能由 `step` 反推“下一步将消费哪些 shard”，从而独立驱动滚动预热。
- **优雅降级（cache miss 不阻塞）**：若预热滞后，DataLoader 在本地 NVMe 未命中时，从下一层（共享 FS / 对象存储）**懒拉**该 shard（呼应 9.2 “直读对象存储” + 8.2 “本地缓存预热”），训练不卡死，只是那一步略慢。

### 8.4 不同框架下的预热驱动方式

预热逻辑本身与训练框架解耦，但“如何拿到 rank→node 与 shard 分配”各框架略有不同：

- **PyTorch（DDP / FSDP）**：
  - 数据切分由 `DistributedSampler`（或自定义 sampler）按 `world_size` 把样本/shard 分给各 rank；用 WebDataset/MDS 时，每个 rank 绑定一组**互不相交的 shard**。
  - 驱动方式：启动器（`torchrun`）在拉起训练前，先以“每节点”为单位跑预热——因为 rank 与节点绑定、shard 分配确定，节点 N 只需暖“本节点 ranks 的 shard”到本地 NVMe；同节点多卡（同组 shard）共享一份本地副本。
  - 滚动：训练上报 `step`，控制器按 `shard大小 × batch` 推算未来 N 步要用的 shard，提前暖到该节点 NVMe。

- **Megatron-LM（3D 并行）**：
  - 数据按 **DP rank** 切分；同一 DP 组内的 TP/PP rank 消费**完全相同**的数据。因此“数据消费者”只有 `world_size/(TP×PP)` 个 DP rank，预热只需按 DP rank 暖，TP/PP rank 复用同节点本地 shard 缓存。
  - 数据混合由 `--data-path` / `--data-blend`（各源权重）配置；预热侧需拿到同一份 blend 规格 + DP 组布局，才能算出每个 DP rank 的 shard 子集。
  - Megatron 自带 `--consumed_train_samples` 用于续训，预热可据此定位“下一个该暖的 shard”；其 `build_train_valid_test_data_loaders` 的 index 数据集也便于预热侧对齐顺序。

- **Ray（Ray Train / Ray Data）**：
  - 本仓库基于 Anyscale/Ray 编排，预热可做成与 trainer **同 placement group** 的 Ray Actor/Task，天然获得节点亲和，shard 放置与训练 worker 同节点。
  - **Ray Train**（`TorchTrainer`/`MegatronTrainer`）通过 `session`/`worker` 上下文暴露 `world_rank`/`node_id`，预热 Actor 读同一 manifest + 该布局即可按节点暖。
  - **Ray Data** 的 `streaming_split` / `split` 本身做 block 级分布式分发，并带 prefetch；其流式执行器会把后续 block **预取到对应节点**，相当于内建的滚动预热；`prefetch_batches` 控制前瞻量，block 可 pinned 到节点 object store，等价于“热层”。
  - 优势：Ray 把“编排 + 数据局部性 + 流式预取”统一在一个运行时里，预热控制器与训练共享调度视图，rank→node 不需额外对账。

### 8.5 分层预热 + 滚动预热：可实操的主流方案

> 本节把 8.1 的“三层存储”与 8.3 的“放置契约/进度感知”落到**可执行的工程方案**。业界对“如何预热”没有唯一解，下面列出五种主流实操方案（可由简到繁组合使用），每种都说明：触发时机、具体机制、如何得知切分、如何按进度滚动、适用框架。

#### 方案 A：加载器懒拉 + 节点 LRU 缓存（pull-on-miss，零独立预热）

- **机制**：不单独跑预热作业，靠 DataLoader 的**缓存文件系统**在首次读取时自动把 shard 拉到本地 NVMe 并缓存；后续命中本地。分层由“缓存命中=热、远端对象存储=冷”两档构成（温层可省）。
- **具体落地（按格式）**：
  - **WebDataset**：`wds.WebDataset(urls, cache_dir="/local/nvme/cache").cache_thresholds((1e9, 1e10))`——大于阈值的 shard 自动下载并常驻本地缓存。
  - **MDS**：`StreamingDataset(..., local_cache_path="/local/nvme/mds_cache")`——shard 首次访问后落到本地缓存。
  - **fsspec**：用 `CachingFileSystem`/`SimpleCacheFileSystem` 包装 `s3://`，`cache_storage="/local/nvme"`、`check` 策略控制校验；上层 Dataset 无感知。
- **如何得知切分**：无需预热侧计算——缓存是**消费驱动的**，训练按自己的 sampler 读到哪个 shard 就拉哪个；8.3 的 `rank→shard` 由训练侧 sampler 自然决定。
- **如何滚动**：滚动 = 消费本身；LRU 自动淘汰最久未用 shard，腾出 NVMe。无前瞻，故首个 epoch 有 cache-miss 抖动。
- **适用**：快速起步、中小数据集、NVMe 容量能容纳单任务全集；可与方案 B 的 Wave 0 组合消除首 epoch 抖动。
- **优缺点**：实现成本最低、零协调；缺点是首 epoch 有冷启动延迟，且 NVMe 命中率依赖访问局部性。

#### 方案 B：启动器前置预热 + 节点后台滚动（launcher pre-step，最通用）

- **触发时机**：在训练进程拉起**之前**，由启动器（`torchrun` 的 `train_script` 前置、`Ray` 的 `setup` hook、k8s `initContainer`、或 Anyscale 的 `prewarm.py`/`worker.sh`）在每个节点先跑一段预热脚本；训练启动后由一个**节点内后台进程**继续滚动。
- **如何得知切分（关键实操）**：
  - 启动器已知 `WORLD_SIZE`/`NODE_RANK`/`LOCAL_RANK`（`torchrun` 注入的环境变量）或 Ray 的 `node_id`；
  - 预热脚本复用与训练**同一个 sampler 的 shard 分配函数**，算出“本节点 ranks 负责的 shard 列表”。例如 WebDataset 是 `shards[rank::world_size]`；MDS 直接读其 `rank/world_size` 分片结果；Megatron 则只对 `dp_rank = rank//(tp*pp)` 算一份。
  - 每个节点只暖**自己 ranks 的 shard** → 落到本节点 NVMe，同节点多卡共享一份。
- **具体命令（按存储）**：
  - 对象存储→本地：`aws s3 cp --recursive s3://bucket/shards/<node_shard_list> /local/nvme/shards/`（或 `juicefs`/`gos` 等）；
  - 共享 FS→本地：`cp -r /mnt/fsx/shards/<list> /local/nvme/shards/`。
- **如何滚动（后台进程）**：
  - Wave 0：启动前把首批 `K` 个 shard 暖到 NVMe，训练零等待启动；
  - 后台 watcher 读训练写的进度文件（如 `/shared/progress.json` 的 `global_step`，训练每 N 步 flush 一次），按 `目标shard下标 = step × samples_per_step ÷ shard_size` 推算“接下来 `K` 个”shard，并行 `aws s3 cp` 暖到 NVMe；
  - 淘汰：NVMe 使用率超阈值（如 85%）时，删掉“已消费且最旧”的 shard（维护一个 consumed 集合），腾空间给前瞻窗口。
- **适用**：PyTorch DDP/FSDP、Megatron、Ray 都可用；是多卡多节点下**最主流的可控方案**。
- **优缺点**：可控、NVMe 命中率高、省容量（只暖前瞻窗口）；缺点是需写 watcher + 进度文件，且与训练共享 sampler 实现（两处必须一致）。

#### 方案 C：文件系统原生预热（Lustre HSM / FSx for Lustre import）

- **机制**：不靠应用层拷贝，而是让**共享 FS 自身**把数据从 S3 后端提升到 Lustre 层；FS 即“温层+热层”的统一承载。
- **具体命令**：
  - Lustre HSM：`lfs hsm_restore /mnt/lustre/shard_0001.tar`（对每个 manifest shard 发 restore，FS 后台把数据从 HSM/S3 拉到 Lustre）；
  - FSx for Lustre：`aws fsx ... create-data-repository-task --type EXPORT_TO_REPOSITORY` 的反向即 `import` task，列出 manifest 的 shard 路径做批量导入；
  - 节点 NVMe 再按需从 Lustre 读（可作为方案 A 的本地缓存，进一步降延迟）。
- **如何得知切分/滚动**：预热以 **shard 文件粒度**操作，无需知道 rank；按 manifest 列出路径即可 import。滚动 = **分批发 `lfs hsm_restore`**：先 restore Wave 0，训练启动后按进度（或由 FS 命中率触发）继续 restore 后续 shard。
- **适用**：已用 Lustre/FSx 的集群（本仓库 anyscale 即有 FSx 相关 Dockerfile/配置）；天然跨节点共享、无需每节点各拷一份。
- **优缺点**：运维简单、零应用改造、跨节点一致；缺点是热层粒度是“整个 FS/文件”而非“每节点 NVMe 私有缓存”，且滚动节奏受 FS 后台传输速率限制，需配合前瞻窗口。

#### 方案 D：Ray Data 流式预取（Ray 原生，本仓库最契合）

- **机制**：用 Ray Data 的**流式执行器**做数据分发与预取，滚动预热由运行时内建，无需自写 watcher。
- **具体落地**：
  - `ds = ray.data.read_xxx(...)` → `ds = ds.streaming_split(world_size, equal=True)`（或 `split`）把 block 分给各 trainer；
  - trainer 端 `it = ds.iter_datasets(prefetch_batches=K, local_cache_dir="/local/nvme")`；
  - Ray 的流式 executor 会把**下一个 K 个 block 预取到消费它的节点 object store/本地盘**（locality），等价于“热层滚动预热”；`prefetch_batches` 即前瞻窗口。
- **如何得知切分**：由 Ray Data 的 `streaming_split` 按 placement group 自动把 block 映射到训练 worker 所在节点；训练与预热共享同一调度视图，`rank→node` 不需额外对账（呼应 8.4 Ray 段）。
- **如何滚动**：执行器消费当前 K 时自动预取后续 K；block 在节点 object store 命中即“热”，未命中从 S3 拉（冷）。可 spill 到本地 SSD 形成持久缓存。
- **适用**：本仓库基于 Anyscale/Ray，Ray Train / Ray Data 是**首选**；`TorchTrainer`/`MegatronTrainer` 均套用此路径。
- **优缺点**：与编排一体、零独立预热代码、自动 locality；缺点是要走 Ray Data 管线（与现 ms-swift 直接读 JSONL 的 9.6 形态需适配），且 block 级而非 shard 级语义。

#### 方案 E：独立预热控制器 / sidecar（进度感知 + 淘汰，完整形态）

- **机制**：把方案 B 的“前置脚本 + 后台 watcher”升级为一个**有状态的独立服务**（Ray Actor / k8s sidecar / 常驻进程），统一拥有 8.3 的放置契约与进度订阅，做精细的分层+滚动+淘汰。
- **操作流程（实操）**：
  1. **作业提交时**：控制器收到 `manifest + sampler 配置 + rank→node 映射`（来自 Ray placement group 或 torchrun env）。它**离线算出全量 `rank→shard` 计划**与每节点 NVMe 预算。
  2. **启动前（Wave 0）**：对每个节点并发下发“暖首批 K 个 shard 到本地 NVMe”指令（调 `aws s3 cp` 或 FS import）。
  3. **训练中（滚动）**：订阅训练进度——训练把 `global_step` 写 Redis pub/sub 或周期更新共享进度文件（也可由控制器用 8.3 “纯计算”从 checkpoint 游标推）；控制器算 `next_shards = sampler_order[step×spu÷shard : +lookahead]`，下发对应节点预热。
  4. **淘汰**：监控每节点 NVMe 占用，把“已消费且超出保留窗口”的 shard 删除以释放空间；续训时读 checkpoint 游标 reposition 窗口，不从头暖。
  5. **降级**：若某 shard 未及时暖好，DataLoader 走方案 A 的懒拉兜底，训练不阻塞。
- **如何得知切分**：控制器与训练**共用同一份 manifest + sampler 函数 + rank→node**（8.3 契约），因此即使不通信也能算出一致计划；进度通过 Redis/文件/checkpoint 任一通道获取。
- **适用**：数据量极大、任务频繁切换、需精细成本控制的产线；是方案 B 的工程化完整版。
- **优缺点**：最灵活、命中率与成本最优、支持续训/弹性；代价是需维护一个服务、并保证 controller 与训练两处 sampler 语义一致。

#### 五种方案对比

| 方案 | 触发时机 | 如何得知切分 | 如何滚动 | 适用框架 | 主要优缺点 |
|---|---|---|---|---|---|
| A 懒拉+LRU | 训练读取时 | 消费驱动，无需预热侧算 | 消费即滚动，LRU 淘汰 | 所有（WebDataset/MDS/fsspec） | 零成本；首 epoch 冷启动抖动 |
| B 启动器前置+后台 | 启动前 + 后台进程 | 复用训练 sampler 算 `rank→shard` | 读进度文件推 lookahead，后台 cp | PyTorch/Megatron/Ray | 可控、命中率高；需写 watcher+进度文件 |
| C FS 原生（HSM/FSx） | 启动前 + 分批 restore | 按 manifest 路径，不需 rank | 分批 `lfs hsm_restore` | Lustre/FSx 集群 | 零应用改造、跨节点一致；粒度粗、受 FS 速率限 |
| D Ray Data 流式 | 训练启动时 | `streaming_split` 自动 locality | 执行器 prefetch_batches 内建 | Ray / Anyscale | 与编排一体、零预热代码；需走 Ray Data 管线 |
| E 独立控制器 | 提交时 + 常驻 | 共用契约离线算全量计划 | 订阅进度/checkpoint，精细淘汰 | 大规模产线 | 最灵活、成本最优；需维护服务、保证 sampler 一致 |

> **实操建议（落地路径）**：先用 **A**（懒拉缓存）快速跑通；叠加 **B** 的 Wave 0 消除首 epoch 抖动；若已在 Ray 上则直接上 **D**；用 Lustre/FSx 时以 **C** 作温层；当数据量与任务切换频率上来后，演进到 **E** 做精细分层滚动。多卡多节点下，所有方案的共性前提是：**预热侧与训练侧必须对同一份“manifest + sampler 分配 + rank→node”达成一致**（8.3），否则会出现“暖错节点/暖了不用”的浪费。

---

### 8.6 案例：ms-swift 下 rank 数据顺序的可预测性与预热调度（本仓库）

> 本节回答一个具体工程问题：**能否领先于训练 DataLoader 提前得知“后面一步或多步要用哪些数据”，从而把它们从 S3/EFS 预热到训练节点本地 NVMe？**
>
> 结论先行：**能，而且是纯计算、零通信的精确推算**——但“提前几步取数”这个朴素形态在随机 shuffle 下几乎无效，真正可行的是 8.6.7 / 8.6.8 两类**顺序—容量协同设计**。
>
> **前提与现状**：本仓库当前是 §8.2 的「全量预热（Bulk import）」形态——WDS shards 与 manifest 在训练前由 sync Job 整批落到节点 NVMe（`pytorch_job/pytorchjob.yaml:58-61,93-94`，hostPath `/opt/dlami/nvme` → 容器 `/nvme`，`/curated_audio` 软链到 `/nvme/data`）。本节讨论的是**当数据量增长到单节点 NVMe 装不下时**的演进路径。

#### 8.6.0 命名说明（避免与 8.5 混淆）

| | 讨论的维度 | 取值 |
|---|---|---|
| **§8.5 方案 A~E** | **预热的执行形态**（谁来搬、什么时候搬） | 懒拉 / 启动器前置 / FS 原生 / Ray 流式 / 独立控制器 |
| **§8.6 方案 A、B** | **训练顺序的局部性策略**（决定“搬什么划不划算”） | A = 块序化（Block-ordered）；B = 保持全随机 + 计划式预热（Permutation-planned） |

两者**正交**：§8.6 的 A/B 都需要搭配 §8.5 的某种执行形态落地（组合建议见 8.6.11）。

---

#### 8.6.1 核心结论

ms-swift 中每个 rank 消费的样本顺序，是 `(N, data_seed, epoch, rank, world_size, batch_size)` 的**纯确定性函数**，不依赖任何运行时状态：

1. **可提前推算任意未来 step 的样本 index**，无需与 DataLoader 通信、无需 hook worker 进程；
2. **任一进程可算出全部其他 rank 的序列**（所有 rank 共享同一个全局 permutation），这是“节点级去重预热”的前提；
3. **唯一不可预测的例外**是 `LazyLLMDataset` 的 encode 失败 fallback（占比极低，可降级处理）与 `--streaming` 模式（见 8.6.6 注）。

---

#### 8.6.2 顺序在哪里产生（源码证据）

##### (1) Swift / HF Trainer 路径 —— 自研 `BatchSamplerShard`

ms-swift **不使用** accelerate 的 `prepare()`，也**不使用** `DistributedSampler`，而是自研 batch_sampler（注释即写明 “Higher efficiency”）：

```1306:1318:swift/trainers/mixin.py
            if hasattr(train_dataset, '__len__'):
                if args.group_by_length:
                    batch_sampler_params['group_by_length'] = args.group_by_length
                    batch_sampler_params['lengths'] = train_dataset['lengths']
                batch_sampler = BatchSamplerShard(
                    len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_params)
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)
                dataloader_params['batch_sampler'] = batch_sampler
                dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)
```

顺序生成的全部逻辑只有 13 行：

```45:57:swift/dataloader/shard.py
    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            if self.group_by_length:
                from transformers.trainer_pt_utils import get_length_grouped_indices
                total_idx = get_length_grouped_indices(
                    self.lengths, self.batch_size * self.world_size, generator=generator)
            else:
                total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[self.rank::self.world_size]
        else:
            total_idx = range(self.rank, self.total_samples * self.world_size, self.world_size)
```

关键语义（`swift/dataloader/shard.py`）：

| 项 | 值 / 行为 | 行号 |
|---|---|---|
| `total_samples` | `N // world_size`，**先整除、尾部截断**；randperm 规模是 `total_samples × world_size` 而非 `N` | `:24` |
| seed | `curr_seed = data_seed + epoch`（`set_epoch` 设置），`data_seed` 默认 **42** | `:28-29`, `:70-71`；默认值见 `swift/arguments/base_args/data_args.py:81` |
| `rank` / `world_size` | `dist.get_rank() // tp_size` / `dist.get_world_size() // tp_size`（DeepSpeed AutoTP 时 TP 组内共享数据） | `:37-43`；`tp_size` 来源 `swift/trainers/mixin.py:1301-1303` |
| 切分方式 | **所有 rank 用同一 seed 生成同一全局 permutation，再 `[rank::world_size]` 跨步切** | `:55` |
| `shuffle=False` | 退化为 `range(rank, total, world_size)`，**仍是跨步而非连续块**（这一点对方案 A 至关重要，见 8.6.7-R0） | `:57` |
| 尾批 | `drop_last=False` 时保留不完整 batch | `:66-67` |

##### (2) Megatron 路径（本仓库 `pytorchjob.yaml` 实际使用的路径）

`pytorch_job/pytorchjob.yaml:79` 的 `TRAIN_SCRIPT` 指向 `train_speech_to_speech_translation_megatron.sh`，走的是 Megatron 分支：

```131:142:swift/megatron/trainers/batch_sampler.py
                full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
                full_bucket_offset = current_epoch_samples
                g = torch.Generator()
                g.manual_seed(self.epoch)
                if self.group_by_length:
                    from transformers.trainer_pt_utils import get_length_grouped_indices
                    idx_range_total = get_length_grouped_indices(
                        self.lengths, self.micro_batch_times_data_parallel_size, generator=g)
                else:
                    idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
                idx_range_active = idx_range_total[full_bucket_offset:]
                idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]
```

调用点：`swift/megatron/trainers/base.py:978-989`（`MegatronPretrainingRandomSampler`，`data_sharding` 默认 `False`，见 `swift/megatron/arguments/megatron_args.py:520`）。

##### (3) 两条路径的差异（预热控制器必须区分）

| 维度 | Swift/HF (`BatchSamplerShard`) | Megatron (`MegatronPretrainingRandomSampler`) |
|---|---|---|
| seed | `data_seed + epoch`（含 `data_seed`） | **仅 `epoch`**，不含 `data_seed`（`batch_sampler.py:127,134`）|
| epoch 来源 | `set_epoch()` 由 `DataLoaderShard` 传递（`shard.py:86-93`） | `consumed_samples // active_total_samples` **在 `__iter__` 内推导**（`batch_sampler.py:115`）|
| 断点续训 | `SkipBatchSampler(skip_batches)`（`mixin.py:1314-1316`） | `full_bucket_offset = consumed_samples % active_total`，在 permutation 上偏移（`batch_sampler.py:116,141`）|
| 消费者数量 | `world_size / tp_size` | **`data_parallel_size`**，同 DP 组内的 TP/PP rank 消费完全相同的数据 → 预热只按 DP rank 算（呼应 §8.4 Megatron 段）|
| 有状态性 | 无（每次 `__iter__` 重算） | **有**：`consumed_samples` 在 yield 时自增（`batch_sampler.py:153`）|
| `data_sharding=True` 分支 | 无此选项 | 每 DP rank 独占一个连续 bucket 内部随机（`batch_sampler.py:121-129`）——**天然具备 rank 级局部性，是方案 A 的现成开关**|

> ⚠️ `data_sharding=True` 这一分支值得单独注意：它让每个 DP rank 只在 `[rank×bucket_size, (rank+1)×bucket_size)` 这个**连续区间**内随机，rank 之间数据完全不交叉。这正是 8.6.9 追求的 “node-disjoint” 效果的**官方内置版本**，代价是牺牲了跨 rank 的样本混合（每个 rank 每个 epoch 只见到固定的 1/DP 数据）。

---

#### 8.6.3 从 index 到物理 shard 的映射链

本仓库是多模态模型，`lazy_tokenize` 会被自动置为 `True`（`swift/arguments/base_args/base_args.py:130-132`：多模态 + 非 streaming + 非 packing + 非 group_by_length），因此 `train_dataset` 是 `LazyLLMDataset`：

```85:94:swift/dataset/utils.py
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, str):
            return self.dataset[idx]
        for i in range(self.n_try_fetch):
            if i > 0:
                idx = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[idx]
```

完整链条：

```
batch_sampler → idx
  → LazyLLMDataset.__getitem__(idx) → self.dataset[idx]        # HF Arrow 行，内存/mmap，廉价
  → template.encode(row) → resolve_audio_reference(row['audios'][k])   # custom/wds_audio.py:232
  → read_wds_audio_bytes(uri)                                   # custom/wds_audio.py:200
  → os.open(shard.tar) + os.pread(offset, length)               # custom/wds_audio.py:213-221
```

物理路径由 URI 的 netloc + path 直接决定，无隐式映射：

```98:104:custom/wds_audio.py
def _resolve_shard_path(netloc: str, path_parts: Tuple[str, ...]) -> Path:
    """Map wds://<netloc>/<path> to an absolute filesystem path.

    The netloc becomes the first path component, matching the container's
    volumeMount layout (hostPath → /curated_audio).
    """
    return Path('/').joinpath(netloc, *path_parts)
```

**两条关键推论**：

1. **预热的原子单位是 shard tar 文件，不是单条音频。** URI 里虽然带 `offset/length`（理论上支持 HTTP Range 到 S3 的样本级预热），但 S3 的 per-request 开销使得“为一条 200KB 音频发一次 Range 请求”极不经济；而 `custom/wds_audio.py:209-211` 要求 `shard_path.is_file()`，当前实现只认本地文件。
2. **`index → shard_id` 的映射可以离线一次性建好**：读 manifest 的 `audios` 列（Arrow 列式读取，不触碰音频），解析出 `shard_path`，得到 `np.int32[N]` 的 `shard_of_sample` 数组 + `shard_id → (path, bytes)` 表。1 亿样本约 400MB，可 mmap 常驻。

---

#### 8.6.4 可复算的顺序（与源码逐位一致）

```python
import torch

def rank_indices_swift(n_total, batch_size, data_seed, epoch, rank, world_size,
                       shuffle=True):
    """复刻 swift/dataloader/shard.py:45-57。返回该 rank 本 epoch 的 flat index 序列。
    第 s 个 micro-batch = out[s*batch_size : (s+1)*batch_size]。"""
    total_samples = n_total // world_size            # 注意：先整除
    if shuffle:
        g = torch.Generator(); g.manual_seed(data_seed + epoch)
        idx = torch.randperm(total_samples * world_size, generator=g).tolist()
        return idx[rank::world_size]
    return list(range(rank, total_samples * world_size, world_size))


def rank_indices_megatron(n_total, micro_bs, epoch, consumed_samples,
                          dp_rank, dp_size):
    """复刻 swift/megatron/trainers/batch_sampler.py:113-146（data_sharding=False）。"""
    mb_dp = micro_bs * dp_size
    last = n_total % mb_dp
    active = n_total - last
    cur = consumed_samples % active
    full_bucket = (n_total // micro_bs) * micro_bs
    g = torch.Generator(); g.manual_seed(epoch)      # 注意：不含 data_seed
    idx = torch.randperm(full_bucket, generator=g).tolist()[cur:]
    return idx[dp_rank::dp_size]
```

参数取值来源：

- `n_total = len(trainer.train_dataset)`
- `batch_size = trainer._train_batch_size`（= `per_device_train_batch_size`，**不含** `gradient_accumulation_steps`）；一个 optimizer step 消耗 `grad_accum` 个 micro-batch
- `data_seed`：`--data_seed`，默认 42
- resume 时 Swift 路径还要跳过 `skip_batches × batch_size`

---

#### 8.6.5 挂载点：`callbacks_map` + monkey-patch

ms-swift 的 callback 注册表是一个**可变 dict**，`--external_plugins` 加载的插件可直接写入：

```9:17:swift/callbacks/mapping.py
callbacks_map = {
    'activation_cpu_offload': ActivationCpuOffloadCallBack,
    'adalora': AdaloraCallback,
    'deepspeed_elastic': DeepspeedElasticCallback,
    'early_stop': EarlyStopCallback,
    'graceful_exit': GracefulExitCallback,
    'lisa': LISACallback,
    'perf_log': PerfMetricsLogCallback
}
```

实例化于 `swift/trainers/mixin.py:155-158`，签名是 `callbacks_map[name](args, trainer)` —— **第二个参数就是 trainer**，可拿到 `train_dataset` / `state` / `args`。Megatron 侧对应 `swift/megatron/callbacks/mapping.py:8-12`，实例化于 `swift/megatron/trainers/base.py:98-100`。

本仓库插件入口 `custom/plugin.py:88-89` 已有 monkey-patch 先例（`install_qwen_omnispeak_accuracy_patch()` / `install_wandb_extra_config_patch()`），因此：

- **只读型**（方案 B 的预热控制器）→ 注册 callback：`callbacks_map['wds_prewarm'] = WdsPrewarmCallback`，启动加 `--callbacks wds_prewarm`；
- **改顺序型**（方案 A / B-2 的自定义 sampler）→ monkey-patch `swift.dataloader.shard.BatchSamplerShard` 或 `swift.megatron.trainers.base.MegatronPretrainingRandomSampler`，与现有 patch 同一处安装。

---

#### 8.6.6 核心矛盾：全局 randperm 摧毁 shard 局部性

这是决定 A/B 取舍的**唯一关键事实**，必须先量化。

设：总样本 `N`，shard 数 `S`，每 shard 样本数 `m = N/S`。若从全局 permutation 中取 `n` 个样本（随机、无放回），某个特定 shard **一次都不被碰到**的概率约为

```
P(miss) = (1 - n/N)^m ≈ exp(-m·n/N)
```

于是**工作集覆盖率** `coverage = 1 - exp(-m·n/N)`：

| 每 shard 样本数 `m` | 消费比例 `n/N` | 被触碰的 shard 占比 |
|---|---|---|
| 1000 | 0.1% | **63.2%** |
| 1000 | 1% | **99.995%** |
| 1000 | 10% | ~100% |
| 3000 | 0.1% | **95.0%** |
| 200 | 1% | **86.5%** |
| 200 | 0.1% | 18.1% |

**两条毁灭性推论**：

- **推论 1（“提前 K 步预取”无效）**：一个前瞻窗口 `K` 步只覆盖 `n = K × global_batch` 个样本。以 `global_batch=128`、`K=64` 计，`n=8192`；若 `N=1e8`、`m=3000`，则 `coverage = 1-exp(-0.246) ≈ 22%` 的 shard——绝对数量是 **7000+ 个 shard、7TB+**。也就是说“下 64 步要用的数据”几乎等于“全量数据的一大片”，朴素前瞻预取在物理上不可能完成。
- **推论 2（“只训练部分 epoch”不省预热量）**：即使 `max_steps` 只让训练消费 1% 的样本，需要触碰的 shard 仍是 99.99%。**减少训练量并不能按比例减少预热量**——这是 8.6.8 B-1 的核心结论。

> **注（streaming 例外）**：若开 `--streaming`，走 `swift/trainers/mixin.py:1319-1324` 的 IterableDataset 分支 + `DataLoaderDispatcher`，顺序由 HF `datasets` 的 shuffle buffer 在运行时决定，**不可提前推算**；同时 `--streaming` 与 `lazy_tokenize` 互斥（`base_args.py:139-140`）。本仓库训练脚本均未开启 streaming（`pipelines/bridge_alignment/PERF_TUNING.md:137`），此路径暂不适用。

---

#### 8.6.7 方案 A：块序化（Block-ordered）——改变顺序以制造局部性

**核心思想**：既然全局 randperm 摧毁了局部性，就**主动约束顺序**——让训练在一段时间内只访问少数几个 shard，从而使前瞻窗口的工作集变得可预热。

##### A-0 前置发现（务必先读）：`shuffle=false` 不是“顺序读”

最容易犯的错是以为 `--train_dataloader_shuffle false` 就能获得连续局部性。**并不会**：

```57:57:swift/dataloader/shard.py
            total_idx = range(self.rank, self.total_samples * self.world_size, self.world_size)
```

这是**跨步（strided）**切分，不是连续块切分。由此产生两个后果：

- ✅ **节点内局部性极好**：`world_size=64` 时，rank 0..7（同一节点）在第 s 个 micro-batch 分别取 index `s·bs·64 + {0..7}` 附近，落在 manifest 的**相邻区域**，大概率同 shard。
- ❌ **节点间零隔离**：所有 64 个 rank 在同一时刻都在扫描 manifest 的**同一段**，只是取不同余数。因此 **8 个节点在同一时刻需要同一批 shard**，冗余度 = 节点数，跨网络流量放大 8×，且**无法做 node-disjoint 分区**（这一点直接决定了 A 无法单独解决 8.6.8-B2 的容量问题）。

> 若要 node-disjoint，必须自定义 sampler 把 `[rank::world_size]` 改为“先按 node 分区、再在分区内跨步”。此时 A 已经与 B-2 的实现合流（见 8.6.9）。

##### A-1 细则：块的定义与分层打包（离线）

1. **建索引**：从 manifest 的 `audios` 列（Arrow 列式读，不触碰音频）解析 `wds://` URI（`custom/wds_audio.py:107` 的 `parse_wds_uri`），产出
   - `shard_of_sample: np.int32[N]`
   - `shard_table: shard_id → (path, bytes, n_samples, source, lang, task)`
2. **块粒度**：`block = k 个 shard 的交错组`，`k` 取 **8~32**（不是 1，原因见 A-4）。块字节数 `≈ k × shard_bytes`，应满足 `k × shard_bytes ≤ 0.3 × C_node`（留出双缓冲与其他占用）。
3. **分层打包（强制）**：块内必须**保持全局配比**。按 `(source, task, lang)` 分层，每层按其全局占比向块内投放 shard，使任意块的分布 ≈ 全局分布。否则会触发风险 R7（配比失真）。
4. **按字节 bin-packing**：shard 大小不均时用贪心 first-fit-decreasing，保证块字节数方差小，避免某块超 NVMe 预算。

##### A-2 细则：顺序生成（写 order 索引，不重写 manifest）

**不要物理重排 jsonl**（违反 §1.1“任务层不重写数据”，且破坏资产版本一致性）。改为产出一个 **order 索引文件**：

```
order[e] : np.int32[N]   # epoch e 的样本访问顺序
```

生成算法（seed 确定性、可在训练进程内在线复算，无需落盘）：

```python
def block_order(shard_of_sample, blocks, epoch, seed=42):
    """blocks: List[List[shard_id]]，A-1 产出的分层块划分。"""
    g = np.random.default_rng(seed + epoch)
    order = []
    for b in g.permutation(len(blocks)):            # 块间随机
        rows = np.concatenate([rows_of_shard[s] for s in blocks[b]])
        order.append(g.permutation(rows))           # 块内全随机（跨 k 个 shard 交错）
    return np.concatenate(order)
```

落地两种形态：
- **形态 1（改 manifest 顺序 + 关 shuffle）**：按 `order[0]` 重排 manifest 一次，训练加 `--train_dataloader_shuffle false`。**仅适用于单 epoch**，且受 A-0 的节点间冗余问题限制。
- **形态 2（自定义 sampler，推荐）**：monkey-patch `BatchSamplerShard`，在 `__iter__` 中用 `block_order(..., epoch=self.curr_seed - self.base_seed)` 替换 `torch.randperm`，其余（`[rank::world_size]` 切分、`set_epoch`、`drop_last`）**完全保持不变**。侵入面最小，且自动兼容 resume 的 `SkipBatchSampler`。

##### A-3 细则：预热控制器（与 §8.5 方案 B 组合）

块序化之后，前瞻变得平凡：

- 当前块下标 `b = global_step × global_batch // block_samples`
- 预热窗口 = 块 `b+1`（双缓冲）或 `b+1, b+2`（三缓冲，若带宽充裕）
- 驱逐：块 `b-1` 整体删除（**Belady 最优**，因为访问模式完全已知，不需要 LRU）
- NVMe 预算需求 = `2~3 × block_bytes`，与数据总量 `D` **无关**——这是方案 A 相对方案 B 的根本优势

##### A-4 细则：交错（interleave）是必需的，不是可选的

若 `k=1`（块 = 1 个 shard），一个 batch 的 `8 × micro_bs` 个样本全部来自**同一个 tar shard**。本仓库的 shard 按 `dataset=common_voice_scripted_speech / process_version=...` 组织（`pytorch_job/pytorchjob.yaml:93-94`），同一 shard 内极可能是**同一批说话人、同一语向、同一录制批次** → batch 内样本高度同质，梯度估计有偏。

因此块必须由 `k≥8` 个**来自不同分层桶的 shard 交错**构成，并在块内做样本级全排列（A-2 的 `g.permutation(rows)` 已实现）。这正是 WebDataset 的经典做法：`shard-level shuffle + 样本级 shuffle buffer`。

##### A-5 风险清单

| # | 风险 | 说明与证据 | 缓解 |
|---|---|---|---|
| **R0** | **节点间冗余 8×** | `shuffle=false` 是跨步切分（`shard.py:57`），所有节点同时需要同一批 shard | 自定义 sampler 做 node 分区 → 即 8.6.9 统一方案 |
| **R1** | **梯度估计有偏 / loss 震荡** | 块内样本同质（同 speaker / 同 lang），破坏 i.i.d. 假设 | A-4 交错 `k≥8`；分层打包；监控 loss 方差与 grad-norm 抖动 |
| **R2** | **多 epoch 记忆化** | 若 order 不随 epoch 变，模型每轮看到完全相同的 batch 组合 | `block_order` 必须以 `seed+epoch` 为种子（A-2 已含） |
| **R3** | **`group_by_length` 不兼容** | `shard.py:31-32` 明确要求 `group_by_length=True` 时 `shuffle=True`，否则抛 `ValueError` | 本仓库未开启；若要开需在自定义 sampler 内自行实现长度分组 |
| **R4** | **resume 数据错位** | order 若在两次运行间用不同 seed 重新生成，`SkipBatchSampler`（`mixin.py:1314-1316`）跳过的 batch 数虽对，但对应的样本已变 | **把 block 划分的 seed 与 hash 写入 checkpoint**，恢复时校验；块划分算法版本号也要落盘 |
| **R5** | **重写 manifest 的资产污染** | 物理重排 jsonl 会产生一份任务专属副本，违反 §1.1 与 §7.4 | 用 order 索引 / 在线 sampler（A-2 形态 2），不动资产 |
| **R6** | **显存峰值波动 / OOM** | 若 shard 按录制批次组织，音频时长有系统性差异；块序化下某块可能集中长音频，per-step token 数方差大于随机顺序 | 块内按 `duration` 分层投放；监控 per-step token 数 P99；`max_length` 截断兜底 |
| **R7** | **配比失真 → 灾难性遗忘** | 多任务联合训练（本仓库 `joint_stage1`，`custom/plugin.py:102-103`）若块未分层，某段时间只训一个 task，等价于把“混合配比”变成“顺序课程” | A-1 强制分层打包；在线校验每块的 `(task, lang)` 分布与全局的 KL 散度 |
| **R8** | **弹性伸缩失效** | 块划分与 `world_size` 无关（好），但 A-3 的预热计划与 `rank→node` 绑定；节点数变化需重算 | 块划分独立于拓扑；只重算 A-3 的节点映射 |
| **R9** | **eval 不受影响（非风险，备注）** | eval 走 HF 原生（`mixin.py:1344-1345`）并临时关闭 `group_by_length`（`mixin.py:1327-1334`） | 无需改动 |

##### A-6 适用边界

方案 A 的**唯一不可替代场景**：`C_node`（单节点热层预算）小到连一个“合理大小的时间窗”都装不下，被迫把块压到 GB 级。此时随机性损失是**为容量付出的必要代价**。若 `C_node` 有几 TB（如 p5e 类节点），应直接走 8.6.9 的大块方案，随机性损失可忽略。

---

#### 8.6.8 方案 B：保持样本级全随机 + 计划式预热（Permutation-planned）

**核心思想**：不动训练顺序（零收敛风险），在 `on_epoch_begin` 用 8.6.4 的公式算出**本节点全部 rank 的完整访问序列**，映射到 shard 的 `(首次触碰 step, 末次触碰 step)`，据此驱动后台预热与驱逐。

##### B-0 基本形态

```python
class WdsPrewarmCallback(TrainerCallback):
    def __init__(self, args, trainer):          # swift/trainers/mixin.py:155-158 传入
        self.args, self.trainer = args, trainer
        self.pool = ThreadPoolExecutor(max_workers=32)

    def on_train_begin(self, args, state, control, **kw):
        ds = self.trainer.train_dataset.dataset          # LazyLLMDataset -> HfDataset
        self.shard_of_sample, self.shard_table = build_shard_index(ds['audios'])

    def on_epoch_begin(self, args, state, control, **kw):
        if args.local_process_index != 0:                # 节点内只由 local_rank0 负责
            return
        n_local = args.world_size // args.local_world_size   # 本节点 rank 起始
        base = (args.process_index // args.local_world_size) * args.local_world_size
        seqs = [rank_indices_swift(len(...), args.per_device_train_batch_size,
                                   args.data_seed, int(state.epoch), r, args.world_size)
                for r in range(base, base + args.local_world_size)]
        self.plan = build_shard_timeline(seqs, self.shard_of_sample)  # shard -> (first, last)

    def on_step_begin(self, args, state, control, **kw):
        self.submit_window(state.global_step)            # 拉 first_touch 落在前瞻窗口内的 shard
        self.evict(state.global_step)                    # 删 last_touch 已过的 shard
```

三个必备特性：

- **节点内去重**：8 个 rank 共享同一块 NVMe，且所有 rank 共享同一 permutation（`shard.py:54-55`），因此 `local_rank=0` 可独立算出本节点 8 个 rank 的并集，避免 8 倍重复下载。用 `/nvme` 上的 `.lock` + `.done` 标记文件做跨进程协调。
- **优雅降级**：预热未命中时 `read_wds_audio_bytes` 直接失败（`custom/wds_audio.py:209-211` 要求文件存在）——需要增强为“本地缺失则回源 S3 Range 读”，否则 miss 即崩（呼应 §8.3 “cache miss 不阻塞”）。
- **与 DataLoader 预取解耦**：现有 `num_workers=16 / prefetch_factor=8`（`pipelines/bridge_alignment/common_args_megatron.sh:191-192`）提供 128 micro-batch 的**进程内**预取，覆盖的是 CPU codec 编码延迟（秒级）；冷存储预热覆盖的是 S3 首字节延迟与带宽（分钟级）。两者必须是独立的后台线程/进程，**不要混在 dataloader worker 里做**。

---

##### B-1 细分讨论一：一个 epoch 是全量数据 vs 一个 epoch 是部分数据

先明确 ms-swift 中“epoch”的四种实际含义，它们对预热计划的影响完全不同：

###### (a) epoch = 遍历完整 dataset（默认，`num_train_epochs` 驱动）

工作集分析（设每 shard `m` 个样本，均匀散布在整个 epoch）：

- shard 的 `m` 次访问时刻近似 `[0,1]` 上的 `m` 个均匀随机点；
- **首次触碰**时刻的期望 `= 1/(m+1)`；对 `m=3000` 即 epoch 的 **0.03%** 处；
- **末次触碰**时刻的期望 `= m/(m+1)`；对 `m=3000` 即 epoch 的 **99.97%** 处。

> **结论（关键）**：**每一个 shard 的驻留区间几乎等于整个 epoch。**几乎所有 shard 在 epoch 开始的前 0.1% 就被首次触碰，且直到 epoch 结束前还会被再次触碰。
>
> 这意味着：**没有任何 shard 可以被驱逐**，Belady 最优与 LRU 的表现都等于“全部留下”。方案 B 在此情形下退化为 **“必须把全量数据常驻热层”**，即 §8.2 的 Bulk import——**滚动预热完全失效**。
>
> 方案 B 在 (a) 情形下成立的**唯一前提**是：`D ≤ J × C_node`（集群总热层装得下全量）。这正是本仓库当前的状态。

此时 B 的价值不在“省容量”，而在这三点（仍然值得做）：
1. **消除首 epoch 冷启动抖动**：按 `first_touch` 排序做 Wave 0 排程，让最早需要的 shard 最先到位；
2. **把 bulk import 的“阻塞式等待”改为“与训练重叠”**：训练在只暖了前 X% 时即可启动；
3. **节点级去重**：确认 8 rank 的并集，避免重复拉取。

###### (b) epoch = 部分数据（`max_steps` / Megatron `train_iters` 截断）

朴素预期是“只训 1% 就只需要 1% 的数据”。**这是错的**：由 8.6.6 推论 2，消费 1% 样本仍会触碰 99.99% 的 shard（`m=1000`）；即便消费 0.1%，也要触碰 63%。

覆盖率公式 `coverage = 1 - exp(-m · f)`（`f` 为消费比例）：

| `f`（消费比例） | `m=200` | `m=1000` | `m=3000` |
|---|---|---|---|
| 0.01% | 2.0% | 9.5% | 25.9% |
| 0.1% | 18.1% | 63.2% | 95.0% |
| 1% | 86.5% | ~100% | ~100% |

> **结论**：在样本级全随机下，**“训练量减半”≠“预热量减半”**。只有当 `m·f ≪ 1`（即消费样本数远小于 shard 数）时，截断才真正省预热量——而这通常意味着训练步数少到没有实际意义。
>
> **降低 `m`（缩小 shard）能改善这一点**，但 §7.1 已指出 shard 过小会导致元数据爆炸与 S3 请求数暴涨，`m` 不可能降到 O(1)。

###### (c) epoch 已在 manifest 层被抽样（§6.8 每源 epoch 控制）

这是**唯一能真正按比例减少预热量的方式**：抽样发生在 **shard 选择层**而非样本层。若 recipe 只选中 30% 的 shard（如“只要 zh + 高质量”），则 `D_effective = 0.3 D`，工作集直接减少 70%。

- 前提：shard 必须**语义内聚**（§7.4.2），即同 `task`/`lang`/`quality tier` 聚在同一批 shard 里；
- 本仓库现状具备该条件：shard 按 `dataset=... / process_version=...` 分目录（`pytorchjob.yaml:93-94`），`ARTIFACT_AUDIO_SHARDS` 已是“按 process_version 选取”的粒度；
- **行动项**：把“选哪些数据训”的决策**从 manifest 行级过滤上移到 shard 级选取**，是投入产出比最高的一步——它同时改善预热量、S3 成本与 sync Job 时长，且**不影响训练顺序、零收敛风险**。

###### (d) 多 epoch（`num_train_epochs > 1`）

- Swift 路径每 epoch seed 变化（`data_seed + epoch`，`shard.py:71`），Megatron 路径 seed = `epoch`（`batch_sampler.py:127,134`）→ **每个 epoch 是一个全新的 permutation**，工作集与上一 epoch 完全无关（但由 (a) 的分析，两者都 ≈ 全量）。
- 控制器要点：**epoch 边界不能停顿**。应在 epoch `e` 走到 ~90% 时就开始为 `e+1` 排程；由于工作集相同（都是全量），实际上无需重新拉取，只需重排“若已启用驱逐，则恢复被驱逐的部分”。
- Megatron 的 `epoch` 由 `consumed_samples // active_total_samples` 推导（`batch_sampler.py:115`），控制器必须跟踪 `consumed_train_samples` 而非 `state.epoch`。

###### B-1 汇总

| 情形 | 工作集大小 | 可驱逐性 | 滚动预热是否有效 | 结论 |
|---|---|---|---|---|
| (a) 全量 epoch | = 全量 `D` | 几乎为零 | ❌ 无效 | 退化为 Bulk import；要求 `D ≤ J·C_node` |
| (b) `max_steps` 截断 | `≈ D`（见覆盖率表） | 几乎为零 | ❌ 无效 | **减训练量不减预热量** |
| (c) manifest/shard 层抽样 | `= 选中 shard 的字节和` | — | ✅ 按比例有效 | **唯一真正省量的手段，优先做** |
| (d) 多 epoch | 每 epoch 都 `≈ D` | 跨 epoch 无复用退化 | ❌ 无效但无额外损失 | 需提前跨 epoch 排程，避免边界停顿 |

---

##### B-2 细分讨论二：单节点单 epoch 工作集仍远超 NVMe —— 需要更细的切分

这是 B-1 (a)/(b) 的直接后果：当 `D > J × C_node` 时，**方案 B 的纯净形态（不改顺序）在数学上无解**。必须引入切分维度。以下是完整设计。

###### 判据 0（先验算，不满足则一切调度无效）：带宽

```
R_node = 平均样本字节 × GPU数 × micro_bs / step_time        # 节点数据消费速率
要求：  R_node ≤ B_node                                      # 单节点可用冷存储读带宽
```

算例：平均音频 320KB、8 卡、`micro_bs=2`、`step_time=0.8s` → `R_node ≈ 6.4 MB/s`。相对 S3/EFS 的 GB/s 级带宽有 100× 余量 → **本仓库属于带宽充裕型**，问题纯粹是**容量与延迟**，不是吞吐。

> 反之，若 `R_node > B_node`（如高分辨率视频训练），**唯一出路是扩热层容量或降低训练速度**，任何 A/B 调度都无济于事。这一步必须最先算。

###### 判据 1 / 2：是否需要时间维切分

```
判据 1： D ≤ J × C_node   → 只需 L1（node 空间分区），一次性 bulk 预热，训练期零压力
判据 2： D >  J × C_node   → 必须叠加 L2（时间分窗），且必然牺牲一部分全局随机性
```

###### 三级切分设计（L1 / L2 / L3）

**L1 · 空间切分：shard → node 静态分区**

```
node_of_shard(s) = rendezvous_hash(s, nodes)        # 而非 hash(s) % J
```

- 有效热层容量从 `C_node` 放大到 **`J × C_node`**——这是解决容量问题的**第一杠杆**，优先于任何时间维调度；
- 用 **rendezvous / consistent hashing** 而非取模：节点数从 `J` 变到 `J±1` 时只迁移 `1/J` 的 shard（弹性训练必需，呼应 §10“弹性 launcher 重新分配 rank→shard”）；
- **代价**：每个 node 每个 epoch 只见到 `1/J` 的数据。注意这**不等于**模型只见到 `1/J`——DDP 下梯度 AllReduce，全局 batch 仍横跨全部 `J` 个分区，每一步模型都在看跨分区的混合数据。真正损失的是“任一样本只会被固定 node 处理”这一层的随机性；
- **消除该损失**：每 epoch 轮换分区偏移 `node_of_shard(s, e) = rendezvous_hash(s ⊕ e, nodes)`。代价是 epoch 边界要迁移全部 shard；折中做法是**每 K 个 epoch 才轮换一次**，或只轮换 `1/J` 的 shard（部分轮换）。
- **ms-swift 已内置的近似实现**：Megatron 的 `--data_sharding true`（`swift/megatron/trainers/batch_sampler.py:121-129`）让每个 DP rank 在独占的连续 bucket 内随机，等价于 L1 的 rank 级版本。若 `DP rank ↔ node` 有整齐映射，可**直接开这个开关**拿到 L1 效果，零代码。

**L2 · 时间切分：node 内 shard 集合 → W 个 window**

```
window 划分：对本 node 的 shard 集合做按字节的贪心 bin-packing
约束：      window_bytes ≤ α × C_node,  α ≈ 0.4（双缓冲）或 0.3（三缓冲）
epoch 顺序：window 序列随机排列 → window 内所有样本全排列 → [local_rank::8] 切给 8 卡
```

- **这一步等价于“块大小 = NVMe 容量的方案 A”**。区别在于块粒度：方案 A 的块是 `k≈8~32` 个 shard（GB 级、数万样本），L2 的 window 是**数百至数千个 shard**（数百 GB、数百万样本）。
- **随机性损失与块大小成反比**：window 内样本级完全随机、跨 `O(1000)` 个 shard 交错，分布几乎与全局随机不可区分；唯一损失是“跨 window 的样本不会出现在同一 batch”。相比方案 A 的 `k=8`，随机性损失小 2~3 个数量级。
- **`window_bytes = α × C_node` 是硬件决定的最优切点**：既然必须牺牲随机性，就应该只牺牲到 NVMe 装不下为止，一分不多。

**L3 · 样本级：window 内全排列 + rank 跨步切**

复用 ms-swift 原生的 `[rank::world_size]`（`shard.py:55`），仅把 `torch.randperm(N)` 替换为 `window_order(epoch)`，其余语义（`set_epoch` / `drop_last` / `SkipBatchSampler`）**完全不变**。

###### 时间与流水线约束（双缓冲验算）

```
window 持续时间：  T_w = window_samples_per_node / (R_node / 平均样本字节)
双缓冲要求：       window_bytes / T_w ≤ B_node        （与判据 0 等价，自动满足）
预热提前量：       在 window k 消费到 (1 - β) 处启动 window k+1 的拉取，β ≈ 0.3
NVMe 占用：        window_bytes × 2  ≤ C_node × 0.8
```

若 `T_w` 过短（window 太小）→ 预热线程频繁启停、S3 连接复用率低；若 `T_w` 过长（window 太大）→ 超 NVMe 预算。实践上取 `T_w ∈ [20min, 2h]` 较合适，反推出 `window_bytes`，再与 `α × C_node` 取较小值。

###### 驱逐策略：Belady 最优，不用 LRU

访问模式完全已知 ⇒ 直接实现最优置换：

- **window 边界整体驱逐**：window `k-1` 的 shard 在 window `k` 开始时全部可删（保留最近一个作为回退缓冲）；
- **window 内细粒度驱逐**（可选，用于 `window_bytes` 逼近 `C_node` 的情形）：window 内部同样可算出每个 shard 的 `last_touch`，已过 `last_touch` 的 shard 立即删。这能把 `α` 从 0.4 提到 0.6~0.7；
- 绝不要用 LRU——它在“均匀随机访问”下命中率等同随机置换。

###### 容错与续训

- checkpoint 额外落 `(epoch, window_idx, offset_in_window, partition_seed, window_plan_hash)`；
- 恢复时**只需重暖当前 window**（而非从 epoch 头开始），恢复时间从“暖全量”降到“暖一个 window”，这是 window 化的额外重要收益；
- `window_plan_hash` 用于检测 shard 集合/节点数变化导致的计划失效，失效则重算并告警（对应 A-5 的 R4）。

###### 什么时候该改用“样本级 gather”而不是 window

若工作集稀疏（`m · f ≪ 1`，即 8.6.6 覆盖率表左上角区域，例如小规模微调只用 0.01% 数据），按 shard 预热会拉进大量无用字节。此时应改为 **§7.4.3 的样本级预热**：用 `wds://` URI 已携带的 `offset/length`（`custom/wds_audio.py:107-182`）对 S3 发 **HTTP Range 请求**，把散落的样本 gather 成本地紧凑文件。

- **收益**：预热字节数从 `coverage × D` 降到 `f × D`；
- **代价**：请求数从 `S × coverage` 涨到 `n`（每样本一次），S3 per-request 开销与限流成为瓶颈；
- **判据**：当 `平均样本字节 / shard字节 × 请求开销 < 浪费的字节成本` 时划算，实践上即 `m · f < 0.3` 左右；
- **实现前提**：需要给 `read_wds_audio_bytes` 增加 S3 Range 后端（当前只支持本地 `os.pread`，`custom/wds_audio.py:213-221`），这个改造同时也是 B-0 “优雅降级”所需要的。

---

#### 8.6.9 A 与 B 的收敛：它们是同一个方案的两个刻度

把 A-1 的“块”与 B-2 的“window”并列看，会发现两者是**同构**的：

| | 方案 A（块序化） | 方案 B-2（L2 时间分窗） |
|---|---|---|
| 块的物理含义 | `k ≈ 8~32` 个 shard | `α × C_node` 字节的 shard 集合 |
| 典型块大小 | ~10 GB / ~3 万样本 | ~1 TB / ~300 万样本 |
| 块内随机性 | 跨 8~32 个 shard 交错 | 跨 O(1000) 个 shard 交错 |
| 块间随机性 | 随机排列 | 随机排列 |
| 与全局随机的差距 | 明显（需 A-4/A-5 一系列缓解） | 可忽略 |
| 决定块大小的因素 | 人为设定 | **硬件（NVMe 容量）** |
| 是否需要 node 分区 | 无法做（A-0/R0） | **L1 内建** |

> **统一结论**：A 与 B-2 是同一个“块随机（block shuffle）”方案在不同块粒度上的取值。既然随机性损失与块大小成反比，**正确做法是把块开到硬件允许的最大**，即 B-2。方案 A（小块）只在 `C_node` 极小、被迫压缩块大小时才使用。
>
> **推荐落地形态 = L1 + L2 + L3 三级切分**：
> 1. `shard → node` rendezvous 分区（把有效热层放大 `J` 倍，可先用 Megatron `--data_sharding true` 零成本试水）；
> 2. node 内按 `α × C_node` 做 window bin-packing，window 间随机、window 内样本全随机；
> 3. 复用 ms-swift 原生 `[rank::world_size]`，仅 monkey-patch `torch.randperm` 那一行。

---

#### 8.6.10 分阶段实施路线（按投入产出比排序）

| 阶段 | 动作 | 前置条件 | 收益 | 风险 |
|---|---|---|---|---|
| **0** | 验算判据 0（`R_node` vs `B_node`）与 `D` vs `J × C_node` | — | 确定是否需要做任何事 | 无 |
| **1** | **shard 级数据选取**（B-1 情形 c）：把 recipe 过滤从行级上移到 shard 级 | shard 语义内聚（已具备） | 按比例减少 `D`，同时降 S3 成本与 sync 时长 | 无（不改训练顺序） |
| **2** | 建 `shard_of_sample` 索引 + 只读预热 callback（B-0），仅做 **Wave 0 排程 + 节点内去重** | 阶段 1 | 消除首 epoch 冷启动，训练与预热重叠启动 | 低 |
| **3** | 给 `read_wds_audio_bytes` 加 **S3 Range 回源后端** | — | 预热 miss 不再崩溃（优雅降级）；同时解锁样本级 gather | 低 |
| **4** | 开 Megatron `--data_sharding true`，验证 L1 效果 | Megatron 路径 | 有效热层 ×`J`，零代码 | 中（每 rank 只见 1/DP 数据，需评估收敛） |
| **5** | 自定义 sampler 实现 L1+L2+L3（8.6.9） | 阶段 2、3 | 彻底解除 `D ≤ J·C_node` 约束 | 中（需 A-5 全部风险项的监控） |
| **6** | 若仍不足：迁移到 §9.6.1 的 MDS 预 tokenize 分片 | — | 数据体积大幅下降（codec token ≪ 原始音频），可能直接让判据 1 成立 | 高（大改造，但收益最大） |

> **重要提示**：阶段 6（§9.6.1 的离线 codec 化）会把训练数据从「原始音频 wav」变成「离散 codec token」，**体积下降 1~2 个数量级**。这很可能使 `D ≤ J × C_node` 重新成立，从而让整个 8.6.7/8.6.8 的复杂调度**变得不必要**。因此在投入阶段 5 之前，应先评估阶段 6 的可行性。

---

#### 8.6.11 与 §8.5 五种执行形态的组合矩阵

| §8.6 顺序策略 ＼ §8.5 执行形态 | A 懒拉+LRU | B 启动器前置+后台 | C FS 原生 HSM | D Ray Data 流式 | E 独立控制器 |
|---|---|---|---|---|---|
| **全随机 + 全量常驻**（当前状态） | ⚠️ LRU 在随机访问下无效 | ✅ **当前形态**（sync Job = Wave 0） | ✅ 适合 Lustre/FSx | ➖ 需改走 Ray Data | ➖ 过度设计 |
| **方案 B-0（计划式，不改顺序）** | ⚠️ 同上 | ✅ **推荐**（阶段 2） | ✅ 分批 `hsm_restore` 按 first_touch 排序 | ➖ | ✅ 完整形态 |
| **方案 A（小块序化）** | ✅ LRU 此时有效（有局部性） | ✅ 双缓冲简单 | ⚠️ FS 粒度粗，难跟块节奏 | ✅ block 语义天然契合 | ✅ |
| **B-2 / 8.6.9（L1+L2+L3）** | ✅ 可作 miss 兜底 | ✅ window 边界触发 | ⚠️ 同上 | ✅ | ✅ **推荐**（阶段 5） |

**关键一致性要求**（呼应 §8.3）：无论哪种组合，预热侧与训练侧必须对**同一份 `(manifest, 块/window 划分算法 + seed, rank→node 映射)`** 达成一致。本仓库应把这三者收敛到 `custom/` 下的**单一模块**，由训练 sampler 与预热控制器共同 import——**绝不允许两处各写一份**，否则会出现“暖错节点 / 暖了不用”的静默浪费（且这类 bug 只表现为性能下降，极难发现）。

---

## 9. 数据加载（Loading）

目标是**让 GPU 永远“喂不饱”变成“喂得饱”**，关键在分片 + 流式 + 计算/IO 重叠。下面从“DataLoader 在训练 step 中的位置”“加载什么格式”“加载后到计算前做什么”三个角度展开。

### 9.1 DataLoader 在训练 step 中的位置

- **是的，取数（DataLoader 产出 batch）是每一个训练 step 的第一个逻辑环节**。一个典型训练迭代顺序是：
  1. `for batch in dataloader:` —— **拉取并准备一个 batch**（fetch + decode + collate + 搬 GPU）；
  2. `optimizer.zero_grad()` —— 清空梯度；
  3. `outputs = model(batch)` —— **前向计算**（真正的训练计算从此开始）；
  4. `loss = criterion(outputs, labels)` —— 算损失；
  5. `loss.backward()` —— 反向传播；
  6. `optimizer.step()` —— 更新参数。
- **但物理上它并不“阻塞”计算**：DataLoader 用多个 worker 进程后台预取（prefetch），在 GPU 算上一个 step 的同时就把下一个 batch 准备好，实现“计算/IO 重叠”。所以准确说是**逻辑第一、物理重叠**。
- **在 DataLoader 之前**是框架的“准备工作”（不属于 per-step）：分布式进程组初始化、`DistributedSampler` / rank→shard 分配、模型与优化器构建、checkpoint 恢复等。这些决定“每个 rank 读哪些 shard”，但具体每个 step 的第一个动作仍是取数。

### 9.2 一般加载什么格式的数据

DataLoader 消费的就是第 7.2 节选定的**分片打包格式**，训练时按模态常见几类：

- **预 tokenize 的张量分片（LLM 主流）**：token id / 特征已编码好，常见 `.bin`（纯 token id + 长度索引）、MDS、Arrow/Lance、Parquet 的 tensor 列。加载后几乎零 CPU 解码、直接进 GPU——这是大模型训练最常见的形态。
- **列式表格式**：Parquet / Arrow，按列读取、支持谓词下推（只取需要的列/行），适合带多字段元数据的样本。
- **流式分片（tar / WebDataset / MDS）**：训练时顺序或随机流式读 shard 文件，不必全量落本地；WebDataset 按 key 分组、随机性靠打乱 shard 顺序+包内缓冲，MDS 支持点对点 shard 分发与节点间缓存共享、可断点续训。
- **原始媒体文件（CV/语音）**：图片 JPEG/PNG、音频 wav/flac 等，加载后还需解码（见 9.3）；也可预先存为解码后的张量以避免训练期解码压力。
- **TFRecord**：TensorFlow `tf.data` 管道原生格式。

> 一句话：**越上游越“生”（原始文本/媒体），越下游训练越“熟”（预 tokenize 张量）**。规模化 LLM 训练通常把“解码/tokenize”放到资产层（第 4~7 节）完成，训练加载的就是张量，省 CPU。

### 9.3 加载之后、计算之前做了什么

从 DataLoader 吐出一个 batch，到 `model(batch)` 正式开始前向，中间通常还有这些操作：

- **拼接/对齐（collate）**：把若干样本堆叠成批张量；变长序列做 **padding 到同长度**（或依赖第 7.2 节的 Sequence Packing 已拼接好、无需 pad），并生成 `attention_mask` 等附属张量。
- **搬设备（Host→Device）**：batch 张量从 CPU（常是 `pin_memory` 锁页内存）用非阻塞拷贝 `.to('cuda', non_blocking=True)` 搬到 GPU——这是“加载”与“计算”之间的桥梁。
- **精度转换**：混合精度（AMP/BF16）下把输入 cast 到计算精度（或进入 `autocast` 上下文由前向自动处理）；必要时做 dtype / shape 规整。
- **在线增强/变换（CV/语音常见，LLM 少见）**：随机裁剪、翻转、色彩抖动；语音的速度扰动、SpecAugment 等。LLM 因已预 tokenize，通常只做最少变换（随机截断/样本拼接已在打包阶段完成）。
- **梯度累积/微批切分**：若用梯度累积，会在此把一个大 batch 切成 micro-batch 逐步前向。

> 这些“加载后处理”大部分可在 DataLoader worker 内（dataset `__getitem__` / `collate_fn`）完成，只有**搬设备**必须在主进程/训练循环里做；合理分工能让 GPU 一拿到 batch 就直接计算。

### 9.4 DataLoader 工程优化（让 GPU 不饿）

- **多进程 `num_workers` + `prefetch_factor`**：worker 进程并行取数/解码，并提前预取若干 batch，掩盖 IO 延迟。
- **`pin_memory` + 非阻塞拷贝**：锁页内存使 Host→Device 拷贝可异步，减少传输阻塞。
- **计算与 IO 重叠**：GPU 算当前 batch 时 CPU/worker 已备好下一批（即 9.1 的重叠）。
- **解码下沉**：图像/音频/视频解码尽量用 GPU（`DALI`）或专用解码库，避免 CPU 成为瓶颈。

### 9.5 直读对象存储与缓存

- **直读对象存储**：通过 `fsspec`/`s3fs` 让训练直接流式读对象存储，配合本地缓存层，省去全量预热；但单连接带宽受限，通常需要分片并行 + 缓存兜底。
- **内存/磁盘缓存与 mmap**：小数据集直接 mmap 或驻留内存；大数据集用本地 NVMe 缓存层 + LRU 淘汰。

### 9.6 本仓库语音模型（Qwen-OmniSpeak）的 DataLoader 现状

> 对应实现：`custom/dataset.py`（数据集注册与行预处理）、`custom/template.py`（音频加载、codec 编码、collator）、`custom/codec_tokenizer.py`（音频读取与 codec token 化）。训练框架基于 **ms-swift**（`swift.dataset` / `swift.template`），底层是 HuggingFace `datasets` 的 `DataLoader`。

**数据格式（与 9.2 的“预 tokenize 张量分片”不同）**
- 数据以 **JSONL** 存储，每行是结构化字段：`messages`（多轮对话）、`audios`（**源音频文件路径**）、`target_audio`/`target_audios`（**目标音频路径**）、`alignment`（分块时间戳）、以及 `task`/`target_lang`/`speaker_profile` 等 `EXTRA_KEYS` 元数据。
- 音频是**原始 wav 路径引用**，**不是**训练前预 tokenize 好的张量分片；音频在训练时（dataloader worker 内）才被读取并编码。这属于 9.2 中的“原始媒体文件，加载后需再解码”这一类，但这里的“解码”是把音频送进**神经 codec 模型**编码为离散 token。

**取数链路（一个 batch 是如何被准备好的）**
1. **行预处理（RowPreprocessor.preprocess，`dataset.py`）**：按 `task` 过滤行、组装 `messages`（必要时追加 assistant 文本）、收集 `audios`、保留 `bridge_task` 与 `EXTRA_KEYS` 元数据。纯 Python 结构化，**不读音频、不编码**，在 `datasets.map` 阶段完成。
2. **模板编码（Template._encode，`template.py`，发生在 dataloader worker 内）**：
   - **源音频**：交给 Qwen-OmniSpeak 多模态编码器（父类的 `_encode`），读取 `audios` 的 wav 并产出音频特征/音频 token，供 thinker（语音理解）使用。
   - **目标音频 → codec token**：解析 `target_audio` 路径（`_resolve_audio_path`：支持 `s3://` URI 映射到本地镜像目录，见 `codec_tokenizer.py`），用 `soundfile` 读 wav，再调用 **Qwen3-TTS codec tokenizer**（`get_codec_tokenizer().encode_target_audio`）编码为离散 codec：
     - `codec_input_ids`（含 control 前缀：think/nothink + 语种 id + bos）、`codec_labels`（主码，EOS 结尾）、`codec_attention_mask`、`codec_residual_labels`（多 codebook 残差标签）。
     - codec 输出按文件指纹做**磁盘缓存**（`.pt`），避免重复编码。
   - **流式任务的真实时间线**：对同传类任务，`encode_target_audios_concat` 按 `alignment` 的 `target_start_ms` 把各分块 codec 帧铺到真实时间线，分块间/开头 listen-only 区间填**静音 codec 帧**（显式监督“何时该静听”），并基于时间戳生成单调非递减的 `codec_to_sem_visible` **读写调度**张量（决定每个目标 codec 帧在流式时只能看到已到达的源音频前缀）。
3. **批内整理（collator，`_data_collator` / `data_collator`，主进程）**：
   - **变长 padding**：`pad_1d` 对齐 `codec_input_ids/labels/attention_mask`；`pad_2d_time` 对齐 `codec_residual_labels`（时间 × codebook 二维 pad）。
   - **标签构造**：把 `codec_labels` 右移一位并填 `-100` 作为 `labels`，与 swift 标准 next-token 移位对齐，得到主码 token acc。
   - **混合 batch（continue-pretrain）**：无 codec 监督的行用占位 `[BOS]` + `attention_mask=0` + `-100` 标签，并以 `has_codec_supervision` 掩码让 forward 跳过其 codec loss；文本 `labels` 保留用于 thinker 文本 CE。
   - **清理**：丢弃原始 JSONL 字段（`_DROP_EXTRA_KEYS`），不送入 `model.forward`。

**与 9.2 / 9.3 的对照（本仓库的特点）**
- **不是“预 tokenize 张量直接进 GPU”**，而是“dataloader 内加载原始音频 → 音频塔特征提取 + 神经 codec 编码为离散 token”。因此**昂贵的编解码发生在 dataloader worker 里**（codec tokenizer 本身是 GPU 模型，按 `LOCAL_RANK` 上卡），而非训练前向；这正好用 9.1 的 worker 预取/计算重叠来掩盖延迟。
- 源音频走**音频特征提取**路径、目标音频走**codec 离散 token**路径——同一 batch 同时携带连续特征与离散 token 两套监督，collator 的定制 padding/掩码比纯文本 LLM 复杂得多（变长 codec 流 + 多维残差 codebook + 流式读写调度）。
- 数据未使用 Parquet/MDS 等列式/流式分片格式，而是 **JSONL + swift `datasets.map` + 磁盘 codec 缓存**；若后续要接第 7 节的分片/打包与第 8 节预热，则需把 codec token 结果物化为分片（呼应 7.2、9.2 中“预 tokenize 张量分片”的主流做法，把 worker 内的重编码前移为资产层一次性产出）。

#### 9.6.1 改造方案：迁移到 9.2 格式 + 第 7 章分片 + 多级预热（仅方案，待实施）

> 目标：把 9.6 中“训练时 worker 内做重编码”改为“资产层一次性编码、产出 9.2 格式的预 tokenize 张量分片”，从而自然接上第 7 章的分片/打包与第 8 章的多级预热，并让分布式切分更简单。

**核心思路**：将 `codec_tokenizer.py` 中已在 worker 内运行的编码逻辑（`encode_target_audio` / `encode_target_audios_concat` / `_build_training_tensors`，以及 `template.py` 的 `_build_streaming_schedule`）**前移到资产层离线跑一次**，结果按第 7 章策略写成 9.2 格式的预 tokenize 分片。训练侧 dataloader 此后只读分片、不再跑 codec GPU 模型。

**需要做的改变（按环节）**

1. **资产层离线编码（新增 prepare 作业）**
   - 新增一个离线编码作业（落在 `dataset_maker/` 或 `anyscale/data/prepare_data/`，与现有 `run_prepare.py`/`prewarm.py` 同族），遍历 JSONL：
     - 对每行调用 codec 编码，得到 `codec_input_ids / codec_labels / codec_attention_mask / codec_residual_labels`；
     - 将 `codec_to_sem_visible` 读写调度也离线算好存入分片（现由 `template.py` 在线算，可前移）；
     - 源音频顺手预处理：把 wav 解码为固定采样率波形张量（或 mel 特征）写入分片，使 worker 连音频解码都不做（音频塔输入特征与模型权重解耦，可安全预存；不建议预存音频塔输出隐藏态，那会耦合模型版本）。
   - 现有 `encode_codes` 的**逐文件 `.pt` 磁盘缓存**退役——它属“惰性、按文件、非分片”的半成品，替换为显式分片存储。

2. **选用 9.2 格式（预 tokenize 张量分片）**
   - 每行变为一列结构化张量：`input_ids`（文本预 tokenize）、`codec_input_ids`、`codec_labels`、`codec_residual_labels`（2D `[T,C]`）、`codec_to_sem_visible`、`source_audio`（波形/特征），以及 `task / target_lang / speaker / duration` 等**元数据列**。
   - **推荐 MDS**（或 Arrow/Parquet）：MDS 原生支持流式、shard 级分发到各 rank、断点续训、节点间按需拉取；列存 + 谓词下推，训练时按 `task`/`lang` 过滤只取相关列/行。

3. **落地第 7 章分片策略**
   - **语义内聚分片（7.4.2）**：按 `task` / `target_lang` / `speaker` 分 shard（而非随机 hash），使一次训练（如“仅 zh 的 s2s”）只需读对应 shard 子集。
   - **shard 大小（7.1）**：100MB~几 GB；生成 **manifest**（路径/样本数/字节/checksum/所属 task+lang）与 **rank→shard 映射**（round-robin 或一致性哈希）。
   - **打包（7.2）**：张量直接落 MDS；可选 Sequence Packing 拼 codec 流；shard 内 zstd 压缩。

4. **重写训练侧 dataloader（9.6 → 9.2 形态）**
   - `dataset.py`：把 `RowPreprocessor` 从“JSONL + 懒编码”改为**读 MDS/Arrow 分片**；`preprocess` 只剩廉价过滤（按元数据列选 task/lang，谓词下推），不碰音频、不跑 codec；保留 JSONL preprocessor 作 ad-hoc 回退。
   - `template.py` 的 `_encode`：**变轻**——直接读分行内预 tokenize 张量，仅做 dtype 对齐/组装；**移除 worker 内 GPU codec 模型**（`get_codec_tokenizer` 的 `device_map='cuda:LOCAL_RANK'` 及配套 DeepSpeed Zero-3 加载 hack 可删）；`_build_streaming_schedule` 不再在线算。
   - `collator`（`pad_1d`/`pad_2d_time`/标签构造）**逻辑保留**，但输入已是分片张量，pad 成本极低、无编码开销。

5. **分布式切分（接 7.1）**
   - MDS/Arrow 天然支持“每 rank 读互不相交的 shard 子集”：DistributedSampler 按 manifest 分配 shard，或 MDS 直接 point-to-point 拉本 rank 的 shard；本地 NVMe 缓存按一致性哈希复用，跨节点零冲突。

6. **多级存储预热（接 8 章）**
   - shard 语义内聚且 manifest 可枚举，预热系统读**本次任务的 manifest**（9.2 “更优主流实践”：按 manifest 而非整资产搬），只把相关 shard 从对象存储 → 共享 FS → 节点 NVMe 逐级预热；任务切换时只换 manifest、只暖差异 shard，避免“暖一个 shard 只有部分有效”的浪费（呼应 7.4）。

7. **增量与飞轮（接 1.1 / 第 5 / 7.4）**
   - 新数据/新标注只对新行跑**增量编码**、追加新 shard（MDS/Arrow 支持 append）；资产按内容寻址版本化，训练引用新版本即可——昂贵 codec 编码不再每次训练重跑。

**各文件改动点对照**

| 文件 | 现状 | 改造后 |
|---|---|---|
| `codec_tokenizer.py` | worker 内 `encode_codes`+`.pt` 懒缓存、`device_map` 上 GPU | 新增批量写分片函数；`.pt` 缓存退役；编码逻辑被离线作业调用 |
| `dataset.py` | `RowPreprocessor` 读 JSONL、组装 messages/audios、不编码 | 新增 shard(MDS/Arrow) 数据集 + 元数据过滤 preprocessor；保留 JSONL 作回退 |
| `template.py` | `_encode` 读音频+跑 codec+算 schedule；collator pad | `_encode` 变轻（读预 tokenize 张量）；删 GPU codec 与 schedule 在线计算；collator 不变 |
| 新增 prepare 作业 | — | 离线编码 JSONL→分片，落在 `dataset_maker/` 或 `anyscale/data/prepare_data/` |

**预期收益**：训练侧零重编码（worker 变纯张量加载，`num_workers`+prefetch 可拉满且不抢卡）；分布式 shard↔rank 天然映射；预热按语义内聚 shard + manifest 精准流动；契合“资产层一次性编码（第 4~7 章）→ 任务层轻量组装（9.2）→ 按 manifest 预热（第 8 章）→ 飞轮增量（1.1）”的全文主线。

---

## 10. 分布式训练（Training）

数据量大时，训练框架侧的主流做法：

- **数据并行（DDP）**：每个 rank 持有完整模型副本，消费不同数据分片，梯度 AllReduce。最简单、最常用，配合上面的分片加载即可横向扩到上千卡。
- **显存/通信优化**：
  - **FSDP / DeepSpeed ZeRO**：把优化器状态、梯度、参数分片到各卡，用通信换显存，支持训练远超单卡显存的大模型；
  - **梯度累加 + 混合精度（AMP/BF16）**：减少显存与通信量。
- **流水并行 / 张量并行**：当单模型过大（上百 B 参数）时，与数据并行组成 3D 混合并行，但数据加载层仍按 DDP 思路分片。
- **容错与弹性**：
  - 基于 **checkpoint**（模型 + 数据读取游标/epoch 状态）实现断点续训；
  - 节点故障后从最近 checkpoint + 当前 shard 偏移恢复，避免重跑整个 epoch；
  - 弹性训练（Elastic / 容错 launcher）可在节点增减时重新分配 rank→shard 映射。
- **存储后端配合**：训练进程统一挂载共享 FS（或对象存储 + 本地缓存），权重 checkpoint 写共享存储，日志/指标写对象存储，避免各节点写本地导致难以聚合。
- **数据飞轮闭环（连接标注）**：训练产出的**评测结果与错误分析**，是回到第 5 节「数据标注」环节做人工精标、更新分类器的输入——由此驱动下一轮数据迭代（见第 5 节飞轮闭环）。训练本身不重写数据，只消费资产引用。

---

# 第三部分 · 支撑系统与工具

## 11. 支撑上述方案的主流系统/工具

- **数据湖表格式（ACID + 版本 + 时间旅行）**：Apache Iceberg / Delta Lake / Hudi —— 汇聚落地、资产版本化/增量摄入、可复用资产、schema 演进的底座。
- **采集/标注规模化**：分布式爬虫、标注平台（Label Studio / Prodigy 类）、human-in-the-loop 工作流、LLM-as-annotator 管线。
- **清洗/打分规模化**：Spark/Flink 批流一体、GPU 集群做 ASR/OCR/质量分类。
- **去重/打分规模化**：MinHash+LSH、Spark/GPU 集群、ANN 索引（支持增量插入）。
- **数据版本与内容寻址**：LakeFS、DVC、Quilt，或自研 content-addressed store（支撑第 3 节版本化与第 6 节视图化）。
- **流式/可增量格式**：WebDataset、MDS 都支持“追加新 shard”而不破坏已有布局（支撑增量摄入与零拷贝复用）。
- **数据策划平台**：各大实验室内部的“data engine / data flywheel”，把上述环节串成常驻、可观测、可回滚的流水线。

---

## 一句话总结

把“采集 → 汇聚 → 清洗 → 标注 → 质量打分 → 去重”做成**一次性的、带元数据与版本的持久数据资产**（清洗/标注非销毁性、结果以元数据列/新分区沉淀）；把“采样/配比/分片/打包”做成**任务层的声明式轻量组装（视图/recipe）**。数据源更新和人工标注通过**增量管道 + 元数据列**沉淀进资产（汇聚增量摄入、标注新分区写回），下一轮训练只需引用新版本并调整 recipe，从而把“几百 PB 重做”变成“分钟级挑选”，实现快速迭代的数据飞轮；而分片以**语义内聚**切（第 7.4 节）、预热以**任务 manifest（必要时样本级）**为粒度，既保留跨任务复用，又把预热效率拉满。
