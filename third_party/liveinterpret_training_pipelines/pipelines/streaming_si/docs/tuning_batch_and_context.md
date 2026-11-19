# 调参依据：batch / 样本长度 / 推理窗口

batch size、梯度累积、`SI_MAX_AUDIO_SEC`、packing、推理窗口与滑窗延迟的取舍依据。
**只记「为什么 / 约束」，结论来自代码推理与既有日志，未经实测的项已标注。**

配套阅读：`../README.md` 第 5 节（各阶段关键参数默认值）。

## 1. 全局 batch 与梯度累积

全局 batch = `PER_DEVICE_TRAIN_BATCH_SIZE` × 卡数 × `GRADIENT_ACCUMULATION_STEPS`。

- **学习率缩放**：AdamW 下按 **平方根缩放**（Malladi et al., 2022），batch 放大 k 倍则 lr 放大 √k。线性缩放（k 倍）在 Adam 上偏激进，容易训崩。**为什么必须放大、以及两个缩放规则的来由，见第 7 节。**
- **临界 batch**：看每步参与 loss 的 token 数（S2 只有译文部分计 loss，约 50–150 token/条）。S2 在 2 节点 16 卡、每卡 24、全局 384 时约 5 万步 / epoch，属中等规模，收益递减不明显。**是否接近临界点必须实测**（对比相同「已见样本数」下的 loss / token_acc：如 768 的第 1.25 万步 vs 384 的第 2.5 万步）。
- **GA 只在显存放不下目标 batch 时才用**。它只能撑大全局 batch，不提高吞吐；每卡 24+GA=2 的速度等于每卡 24+GA=1，但更新次数减半。**H200 141GB 下应先实测每卡 batch 的显存上限，能直接加每卡 batch 就不要用 GA。**
- 用 GA 的其他场景：序列变长后每卡 batch 被迫降低但要保住全局 batch；卡数减少时复现原配置。另见第 6 节。
- 全局 batch 改变后必须同步调整学习率，并加 warmup；`SAVE_STEPS` / `LOGGING_STEPS` 按**参数更新次数**计数，GA 翻倍后对应两倍样本数。

## 2. packing vs 长度分组

**packing 在当前代码里不可用**，开了直接报错（见 `register_streaming_si.StreamingSITemplate._data_collator` 中对 `padding_free` / `packing` 的断言）。要支持需同时解决三件事：

1. `audio_pad` 注入是按样本对齐的，多段音频要能注入到序列中对应位置；
2. 4B 的 Qwen3.5 有线性注意力层，打包后需要 `cu_seqlens` 在样本之间重置递归状态，否则前一条样本会泄漏到后一条；
3. 与源项目逐 token 对齐的校验会失效。

**结论：用长度分组（`GROUP_BY_LENGTH=true`）。** P2 数据随机组 batch 时约 60% 的计算花在 padding 上，batch 越大收益越明显。代价是同 batch 内样本长度相近，loss 曲线会略抖。

## 3. `SI_MAX_AUDIO_SEC`：为什么是 30s

- **S2/S4 早期口径 20s 是「截断」**：音频超过 20s 截掉尾部但**保留完整译文**，即译文后半段在音频里找不到对应内容。这在教模型「没听到的内容也翻出来」，与 S5 要学的读写时机（没听到就该 WAIT）相反。
- **改成 30s + 丢弃超长**：`SI_DROP_OVERLONG=1` 在预处理阶段就按元数据列过滤，编码时再兜底一次。
- **30s vs 60s**：60s 只有在推理窗口也放大到 40s 以上时才用得上。窗口不变时，60s 多出的算力、显存（每卡 batch 要降到 8–12）、以及长样本权重偏移和脏数据混入，都换不来收益。60s 还需把 `SI_MAX_TARGET_LEN` 提到约 384、`MAX_LENGTH` 提到约 1280。
- Voxtral 编码器是 RoPE + `sliding_window=750`（15s），相对距离不会超窗；流式推理时位置持续增长，所以 `max_position_embeddings=1500`（30s）**不是硬性上限**。

## 4. 推理窗口（K_max / K_min）与训练的一致性

无限流推理用双阈值滑窗：`thinker_window_turns`（K_max）与 `thinker_window_min_turns`（K_min），turn 数达 K_max 时裁到 K_min 并重建 KV。引擎强制 `thinker_window_min_turns == prefix_context_turns`（源 `streaming_engine.py` 检查，不等即报错）。

按 `CHUNK_MS=640` 换算（1 turn ≈ 0.64s）：

| 推理窗口 | K_max / K_min（turn） | 对应时长 | S5 `SI_MAX_AUDIO_SEC` | S5 `SI_MAX_LENGTH`（粗估） |
|---|---|---|---|---|
| 20s（源默认，长文档正式口径 32/16） | 32 / 16 | 20.5s / 10.2s | 20 | 640 |
| 30s | 46 / 23 | 29.4s / 14.7s | 30 | 约 1024，以实测为准 |
| 60s | 93 / 46 | 59.5s / 29.4s | 60 | 约 1792 |

- **S5 必须与推理窗口一致**：S5 的一条样本就是一整段源音频按 640ms 切成的 turn，样本多长，训练时上下文就有多长。上限 20s（约 31 turn）时，31–46 turn 这段训练从未覆盖，而推理每次滑窗前都会经过，读写时机在长上下文下可能不稳。
- 改 30s 要连带改：`SI_S4_PREFIX_CONTEXT_TURNS=23`、`SI_MAX_LENGTH` 约 1024、`SI_MAX_SPEECH_LEN` 约 900（否则 30s 样本会被 `too_long` / speech 超限静默跳过，改了上限也没用）、每卡 batch 预计降到 24–32。
- **前提**：S5 数据里要有足够多 >20s 的样本，以及至少 25 个 turn 的样本（prefix-drop 裁剪要求 `n_step >= prefix_context_turns + 2`，见源 `si_sampler.py`）。否则要先把长文档样本做出来。

## 5. 滑窗导致的延迟尖峰

**成因**：turn 数达 K_max 时，`_maybe_replay_window()` 同步做三件事——丢掉旧 turn 只留 K_min；清空 Thinker KV 后把「指令 + K_min 个 turn（audio embedding / 动作 token / 已生成译文）」一次 forward 重新 prefill；Talker KV 同样整段重建（重放保留的 WRITE turn）。

不能只删最旧 KV 的原因：RoPE 位置需要重排；Qwen3.5 的线性注意力递归状态无法撤销旧 token，只能重建。该过程**同步执行且发生在当前 chunk 的决策之前**（源 `feed_audio_features` 的 `window_replay_check`）。按 (32,16) 每 16 个 turn（约 10.24s）触发一次。

**实测（0.6B S5 checkpoint，chunk 640ms，7 个 WebUI 会话，源仓库 `logs/s4_sessions/`）**：

| 项 | 实测值 |
|---|---|
| 触发位置 | 第 33、49、65、81… 步，每 16 步一次 |
| 单次重建耗时（43 次） | 77ms / 177ms / 195ms（min / 中位 / max） |
| 普通步引擎耗时 | 中位约 180ms；p90 约 540–930ms（WRITE 步含 Talker + 声码器） |
| 含滑窗的步 | 多数 350–390ms，个别 >1.4s（同时有 Talker 生成） |
| 客户端播放间隙 >50ms | 103 次，中位约 1s、最长约 8s |

**结论**：滑窗让单步多约 0.18s，仍在 640ms 实时预算内；**秒级间隙主要来自模型持续 WAIT 与说话人停顿，不是滑窗**。滑窗可能带来的影响是：周期性轻微延后（约每 10s 晚 0.1–0.2s）；与 WRITE 步叠加且 `preroll_sec`（默认 0.25）耗尽时多出约 0.2s 静音；整体 RTF 接近 1 时无法追回，表现为 `queue_delay` 逐步上升。此外滑窗瞬间上下文从 20s 骤降到 10s，术语 / 语气衔接可能跳变，这往往比延迟更影响体感。

注意：上表为 **0.6B** 实测，**4B 需另行实测**（`engine_window_replay_check` 的计时已埋点）。按 token 数线性外推：30s 窗口（K_min=23）约 0.25s，60s 窗口（K_min=46）约 0.5s，后者在 4B 上可能单独超出 640ms 预算。

缓解（需要时再做）：在空闲 WAIT 步提前重建或放到另一个 CUDA stream 异步执行，避免与 WRITE 步叠加；适当增大 `preroll_sec`；4B 上线前用 `infer_longform_s2st.py` 统计滑窗步与 WRITE 步叠加的总耗时。

## 6. 什么时候增大 `GRADIENT_ACCUMULATION_STEPS` / `NUM_TRAIN_EPOCHS`

**GA 该调大**：目标全局 batch 放不进显存；序列变长后每卡 batch 被迫降低但要保住全局 batch；卡数减少时复现原配置；梯度噪声导致 loss / `grad_norm` 尖峰且每卡 batch 已到显存上限（先排除数据问题、lr 过高、缺 warmup）。多节点通信瓶颈时 GA 能摊薄 all-reduce，但提升有限，不应作主要理由。

**不该调大**：显存还有余量（直接加每卡 batch 更快）；只是想提速（GA 不提速，只减少更新次数）；已接近临界 batch；加了 GA 却没调学习率。

**epoch 该调大**（需同时满足「仍欠拟合」且「无新数据可补」）：验证集 loss / 指标到训练末尾仍在明显改善；lr 已降到下限而训练 loss 仍稳定下降；数据量相对模型容量偏小（如 S1）；下游指标随训练量稳定提升未饱和。

**不该调大**：验证 loss 停降或回升且与训练 loss 差距拉大（已过拟合）；数据量大（S2 的 1929 万条，1 epoch 通常够）；数据噪声大（重复看会背下噪声，更易幻译）；全参微调在意保留原有能力；根因是 lr 太低或改了 batch 却没调 lr（先修正 lr）。

细节：

- **训练完再「补一个 epoch」≠ 一开始就设 2 epoch**。cosine 会按总步数拉伸，1 epoch 跑完时 lr 已到下限，直接 resume 效果很弱。要加量应重训并设更大的 epoch 或 `MAX_STEPS`；或在最后 ckpt 上小 lr 重新 warmup 续训（不如前者稳定）。
- **用 `MAX_STEPS` 更精确**（可设 1.3 epoch 这类非整数），设了之后 swift 忽略 `NUM_TRAIN_EPOCHS`。
- **改 batch 后一般不需要加 epoch**：epoch 衡量数据看了几遍而非更新次数，只要 lr 按 √k 同步放大，1 epoch 的训练量大致等价。
- 判断充分性需要验证集：S2/S4 当前 `split_dataset_ratio=0`、`EVAL_STRATEGY=no`，建议加约 0.1%（约 2 万条）验证集，每 2000 步评估，并在 0.5 / 0.75 / 1.0 epoch 处保存 ckpt 测 BLEU/COMET。

## 7. 为什么放大 batch 必须放大学习率

**一句话**：放大 batch 让每步梯度更准、噪声更小，但每步走的距离不变，而总步数变少了。不放大学率，模型在同样的数据上走的总距离就变短，会训练不足。

### 7.1 梯度噪声随 batch 衰减

设真实梯度为 \(G\)，单样本梯度的噪声标准差为 \(\sigma\)。batch 大小为 \(B\) 时，batch 梯度

\[
g_B = G + \frac{\sigma}{\sqrt{B}}\,\varepsilon
\]

噪声项按 \(1/\sqrt{B}\) 衰减 —— 这就是「大 batch 梯度更准」的由来。

### 7.2 SGD：线性缩放

每步更新 \(\Delta\theta = -\alpha\, g_B\)，其中噪声部分的步长为

\[
\alpha \cdot \frac{\sigma}{\sqrt{B}}
\]

要让噪声步长在 \(B\) 放大 \(k\) 倍后保持不变，就要让 \(\alpha\) 也放大 \(k\) 倍。即 **\(\alpha \propto B\)**，这是 Goyal et al. 2017 的线性缩放规则；ResNet、BERT 这类用 SGD / momentum 的训练都用它。

### 7.3 AdamW：平方根缩放

Adam 的更新近似为

\[
\Delta\theta \approx -\alpha \cdot \mathrm{sign}(g)
\]

即每步更新大小**几乎就等于 \(\alpha\)**，与梯度模长基本无关（这正是 Adam 对梯度尺度不敏感的原因）。此时噪声影响的只是梯度的**方向**：\(B\) 越大方向越准，每步越接近真实的 \(\mathrm{sign}(G)\)。

严格推导要用到 Malladi et al. 2022 的 SDE 分析，结论是：在 Adam 下要保持收敛行为不变，应保持

\[
\alpha\sqrt{B} = \text{常数}
\]

即 batch 放大 \(k\) 倍时 **\(\alpha\) 放大 \(\sqrt{k}\)**。这是 LLM 微调的通行做法。

### 7.4 不放大就会训练不足（S2 实例）

1 个 epoch、1929 万条样本：

| 全局 batch | 步数 / epoch | 每步距离 | 走过的总距离（相对） |
|---|---|---|---|
| 384，lr 1.7e-4 | 约 5.0 万 | 1.7e-4 | 8.5 |
| 768，lr 仍 1.7e-4 | 约 2.5 万 | 1.7e-4 | **4.25（只有一半）** |
| 768，lr 2.4e-4 | 约 2.5 万 | 2.4e-4 | 约 6.0 |

中间那行就是问题：步数减半、步长不变，模型只走完一半路程。这也解释了第 6 节的结论 —— **batch 变大后一般不需要加 epoch，前提是 lr 已按 √k 同步放大**；没放大的话，才需要靠加训练量来补。

### 7.5 三个注意点

1. **√k 保守、k 激进**：线性缩放在 Adam 上容易训崩。全局 384 → 768（k=2）应取 √2 ≈ 1.41 倍，即 1.7e-4 → 2.4e-4。2026-10 那轮实际用 **2e-4**，比 2.4e-4 更保守，偏安全方向，可接受。
2. **放大 lr 后必须加 warmup**：大 batch + 高峰值 lr + 首步即峰值，前期 `grad_norm` 容易出尖峰。那轮设了 `warmup_ratio=0.03`（约 750 步）；更早 384 那次是 `warmup_ratio=0`，风险更高。
3. **`max_grad_norm=1.0` 会限制实际步长**：若梯度经常超过 1.0 被裁剪，更新不再是 \(\alpha\cdot\mathrm{sign}(g)\)，放大 lr 的效果被削弱。开训后留意日志 `grad_norm` 是否长期接近或超过 1.0，若是则需单独处理。

### 7.6 边界：临界 batch

以上推导假设 batch 尚未超过临界 batch。超过后梯度已足够准，再加 batch 收益很小（见第 1 节）。判断方法：对比相同「已见样本数」下的 loss / token_acc（768 的第 1.25 万步 vs 384 的第 2.5 万步）。若 768 明显更差，说明已超临界点，应退回小 batch，**而不是继续调大 lr**。
