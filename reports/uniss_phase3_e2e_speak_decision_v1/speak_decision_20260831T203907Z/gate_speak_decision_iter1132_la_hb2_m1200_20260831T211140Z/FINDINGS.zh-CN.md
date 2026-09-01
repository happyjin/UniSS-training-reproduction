# S2 开口决策训练:门禁失败与真实瓶颈定位

运行 `speak_decision_20260831T203907Z`(1132 步 / 1 个 coverage epoch,自
`endmargin_epoch23` 的 `iter_0002264` 续训),门禁
`speak_decision_iter1132_la_hb2_m1200_20260831T211140Z`,与
`stage2_paced_m1200_iter0002264_20260831T180448Z` 逐项同配置对照
(local agreement holdback 2、pacing margin 1200 ms、semantic cap 384、
`SELECTION.json` 逐字节复制)。

## 1. 结论

**门禁失败,主目标反向。** `WRITE_MT`/事件 0.168 → 0.147。

新增的两项 loss(`speak_decision` 权重 0.5 / margin 1.0,`repetition_penalty`
权重 0.1 / window 8)在训练中确实持续开火(637/1132 个迭代上有监督行,
`n_write` ≈ 11,600、`n_wait` ≈ 5,090),但**监督的不是推理时真正卡住的那个
决策**。

## 2. 门禁对照

| 指标 | iter_0002264 | iter_0001132 | 门线 | |
|---|---:|---:|---:|:--|
| `WRITE_MT` / 事件 | 0.168 | 0.147 | ≥ 0.50 | ✗ |
| `e_s2s_free` 语义覆盖 | 0.6656 | 0.6182 | 不降 | ✗ |
| malformed 片段 | 13 | 14 | 不升 | ✗ |
| `natural_eos` | 0.500 | 0.500 | — | ✗(第 4 个 epoch 不变) |
| ASR 错误率 | 0.2326 | 0.2449 | — | ✗ |
| gold MT 覆盖 | 0.4952 | 0.4924 | — | ≈ |
| free-source MT 覆盖 | 0.2922 | 0.3096 | 不降 | ✓ +6% |
| LAAL | 633 ms | 253 ms | — | ✓ −60% |
| 首次发音 | 1137 ms | 987 ms | — | ✓ |
| 可听起始 | 1214 ms | 1371 ms | ≤ 1500 ms | ✓ |

## 3. 事件决策序列:没有一个事件走完 ASR → MT → TTS

| 序列 | iter_2264 | iter_1132 |
|---|---:|---:|
| `WRITE_ASR → WAIT` | 78 | 77 |
| `WRITE_MT → WRITE_SEMANTIC → READ_NEXT` | 9 | 8 |
| `WRITE_MT → WRITE_SEMANTIC → WAIT` | 4 | 5 |
| `WRITE_MT → WRITE_SEMANTIC → EOS` | 2 | 1 |
| `EOS` / `WAIT` / `WRITE_MT → EOS` | 2 | 4 |

事件要么只识别,要么跳过识别直接翻译+发声。

## 4. logit 探针:两个瓶颈,都不是我训练的那个

`diagnostics/family_logit_probe.py` 包住 `PersistentInterleavedSession._choice`
(纯 argmax,无采样),记录每次决策的候选 logit,返回值不变,因此会话逐位与
门禁一致。按**事件内第几次决策**切分后:

| 事件内位置 | 候选 | iter_2264 中位差 | iter_1132 中位差 | 选中分布(iter_2264) |
|---|---|---:|---:|---|
| 第 1 次 | WRITE vs WAIT | **+28.58** | +21.94 | WRITE 93 / 95 |
| 第 1 次 family | MT vs ASR | **−6.75** | −6.88 | ASR 77 / MT 16 |
| 第 2 次 | WRITE vs WAIT | **−2.88** | **−3.75** | WAIT 77 / WRITE 16 |
| 第 3 次 | (无 WRITE 候选) | — | — | WAIT 10 / READ_NEXT 4 / EOS 2 |

* **第 1 次 WRITE/WAIT 领先 28.6 个 logit,根本不是瓶颈。** 我的 loss 把它压到
  21.9,行为上毫无变化(仍然饱和),纯属空耗。
* **瓶颈 A:family 三选一。** `logit[TASK_MT] − logit[TASK_ASR]` 中位 −6.75,
  MT 仅在 17% 的决策上领先 —— 与实测 16% 的 MT 事件完全吻合。
* **瓶颈 B:写完一个 fragment 后的续写决策。** 中位 −2.88,WAIT 赢 83%。
  **我的 loss 把它推到了 −3.75,更差。**

### 为什么会推反

`speak_decision_masks` 把一个序列里**所有** `WRITE_GENERATE` 行等同看待。gold
数据里绝大多数 `WRITE_GENERATE` 是事件的**第一次**(那里差距已经 +28,
softplus 早已饱和、梯度接近 0),而两个类各占一半权重时,`WAIT` 类只有
5,090 行却分到与 11,616 行的 `WRITE` 类相同的权重 —— 每行权重高 2.28 倍。
于是实际学到东西的是 wait 分支(`speak_decision_wait` 0.306 → 0.222,
而 `speak_decision_write` 0.529 → 0.487 基本持平),把 `wait_logit` 全局抬高,
正好打在唯一重要的那个位置(第 2 次决策)上。

## 5. 与 gold 数据的距离

`train_interleaved_e2e_s2st.jsonl` 前 200 条 packed record 的控制 token 统计:

| | ASR : MT : TTS | WRITE : WAIT |
|---|---|---|
| gold 训练数据 | **1.95** : 1.00 : 1.00 | 2.70 : 1 |
| 自由运行推理 | **5.52** : 1.00 : 1.00 | 1.26 : 1 |

gold 的 `WRITE_MT`/事件约 **0.68**(36,714 MT / 53,770 WAIT),所以门线 0.50
本身是与 gold 相容的;当前 0.147 需要提升约 4.6 倍。

`TOKEN_TASK_ASR` / `TOKEN_TASK_S2T_TRANSLATION` / `TOKEN_TASK_TTS` 在
`task_samples.py:246-260` 的 `_mark_fragment` 里被一律标成 `LOSS_BOUNDARY` ——
与 WRITE/WAIT 训练前完全相同的处境:混在同一个无差别 CE 桶里、权重 0.10、
无 margin、无类别平衡、且与 `END_CONTENT`/`END_SEMANTIC`/语言/速度 token 共享。

## 6. 下一步:先做零训练的分级偏置扫描

在再花 6.4 小时训练之前,必须先回答一个问题:**模型是"会翻译但排序排错了",
还是"排序是对的,它确实还翻不出来"?**

S0.1 已经证伪了硬门控(强制开口 → 文本长度比 1.70 → 15.40 的重复循环),但硬
门控是全有全无的。**分级 logit 偏置**不同:在推理时给 `TASK_MT` 和第 2 次及
以后的 `WRITE_GENERATE` 各加一个 δ,扫 δ ∈ {2, 4, 7},观察是否存在一个 δ 能把
family 比例推向 gold 的 1.95:1 而不触发重复退化。

* 若存在这样的 δ → 能力在,只是排序错,第 6 节的 loss 修正必然有效,再开训练。
* 若任何能提升 MT 比例的 δ 都带来重复/垃圾译文 → 排序是**对的**,墙在增量 MT
  能力本身,应转向 S3(英文流式 ASR)与 MT 侧的定向修复,而不是决策 loss。

单次约 8 分钟(8 卡、16 样本),三点扫描约 30 分钟。

## 7. 若扫描通过,修正后的 loss

两项都**必须按事件内位置切分**,这是本次失败的直接教训:

1. **family margin**:在 family token 行上,gold family 对其余两个的 softplus
   margin。目标是把 MT−ASR 中位从 −6.75 推向 0。
2. **continue-after-fragment margin**:只在**非事件首个** `WRITE_GENERATE` /
   `WAIT_READ` 行上施加。目标是把第 2 次决策的中位从 −2.88 推向正值。
   事件边界可从 `labels` 里的 `WAIT_READ` / `START_GLM` / `EOS` 扫描得到,
   不需要重建数据。

**不要**再对事件首个 WRITE/WAIT 行施加任何监督:那里已经领先 28 个 logit。
