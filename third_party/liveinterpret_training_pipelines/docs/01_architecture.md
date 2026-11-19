# 01 · 架构对比：Bridge（拼接）vs Adapter（叠加）

本文说明本分支相对 public 原版在 **thinker → talker 连接方式**上的架构改动，
以及为什么改用 adapter。

---

## 1. 整体结构

模型由三部分组成（`QwenOmniSpeakForConditionalGeneration`）：

```
Qwen Omni Thinker  ──(连接层)──>  Qwen3-TTS Talker  ──>  code_predictor
(理解源音频 + 生成译文文本)                (生成 codec 声学 token)     (残差 codec 组)
```

- **Thinker**：Qwen2.5/Qwen3-Omni，多模态理解源音频 + 产出译文文本 hidden。
- **Talker**：Qwen3-TTS，自回归产出 codec token（主码 + 残差码）。
- **连接层**：把 thinker 的语义信息送进 talker —— **这里就是本分支改动的核心**。

---

## 2. 两种连接方式（并列共存）

同一份 `modeling_qwen_omnispeak.py` 里同时保留了两条连接路径，由配置
`use_upstream_talker_input` 选择。**两者都在 `__init__` 里被构建**（见下方代码位置），
只是 forward 时二选一走：

| | **Bridge 路径**（public 原版默认） | **Adapter / upstream 路径**（本分支默认） |
|---|---|---|
| 模块 | `self.bridge = build_bridge(...)`（DualRoute Bridge，MLP 或 Q-Former） | `self.thinker_adapter = Qwen3TTSThinkerAdapter(...)` |
| 连接方式 | thinker 语义块 **拼接（concat）** 到 codec 序列**前面**，形成更长序列 | thinker hidden 经 adapter 投影后 **逐帧相加（add）** 到 codec 序列上，序列长度不变 |
| 开关 | `use_upstream_talker_input=False` | `use_upstream_talker_input=True`（variant A） |
| 可训练模块 | `model.bridge`（+ talker 前若干层） | `model.thinker_adapter`（+ 可选 talker 层 / MTP / codec_head / thinker） |

代码位置（`modeling_qwen_omnispeak.py`，`QwenOmniSpeakModel.__init__`）：

```python
# bridge 总是被构建（兼容旧 checkpoint 加载），upstream 模式下 forward 不使用它
self.bridge = build_bridge(config.bridge_config)

# adapter 仅在 upstream 模式构建
self.thinker_adapter = None
if getattr(config, 'use_upstream_talker_input', False):
    self.thinker_adapter = Qwen3TTSThinkerAdapter(
        thinker_hidden_size=thinker_h,
        text_hidden_size=text_h,
        ...)
```

> 注意：`bridge` 即使在本分支也会被构建（为了让含 bridge 权重的旧 checkpoint 能干净加载），
> 但 **reply-add / full-add 的 forward 路径完全不使用它**。真正参与计算并被训练的是
> `thinker_adapter`。

---

## 3. 为什么改用 Adapter（叠加）而不是 Bridge（拼接）

1. **对齐 Qwen3-TTS 原生输入范式**：Qwen3-TTS 训练/推理时，文本信息是**逐帧叠加**在
   codec 通道上的（text channel + codec channel 相加），而不是拼接。用 adapter 叠加与其
   原生行为一致，能更好复用其预训练权重、更贴近其推理路径。
2. **序列不变长，显存/算力可控**：拼接会把 thinker 语义块加到 codec 前面，显著拉长序列
   （注意力是平方复杂度）。叠加不改变 codec 序列长度，长序列下更省显存。
3. **流式（streaming）友好**：reply-add 布局天然对应 Qwen3-TTS 的流式对齐方式
   （详见 [`02_reply_add_dataflow.md`](02_reply_add_dataflow.md)）。
4. **训练更轻**：只需训练一个小 adapter（`thinker_adapter`），thinker 与 talker 主体冻结，
   参数量小、显存低、收敛快。

---

## 4. `Qwen3TTSThinkerAdapter` 是什么

一个轻量投影模块（`modeling_qwen_omnispeak.py`）：

```
输入:  thinker_hidden_size (thinker 输出维度)
       └─ fc1 → act → dropout → fc2 → norm
输出:  text_hidden_size (talker text 通道维度)
```

它把 thinker 的隐状态投影到 talker 的 text 通道维度，随后再经 talker 自带的
`text_projection` 投到 talker hidden，最终逐帧叠加到 codec。

配套还有 `Qwen3TTSTalkerResizeMLP`（talker 维度对齐 MLP，处理 thinker/talker 隐维不一致）。

---

## 5. 可训练参数（本分支默认）

入口脚本默认（`train_speech_to_speech_translation_upstream.sh`）：

```
--freeze_parameters_ratio 1.0             # 先全冻结
--trainable_parameters model.thinker_adapter   # 只解冻 adapter
```

可选扩大训练范围（环境变量，见 [`04_env_and_training.md`](04_env_and_training.md)）：

- `UNFREEZE_TALKER_LAYERS=0|2|8|28` — 额外解冻 talker 若干层
- `UNFREEZE_MTP=1` — 解冻残差 code predictor
- `UNFREEZE_CODEC_HEAD=1` — 解冻主 codec 头
- `UNFREEZE_THINKER=1` — 解冻 thinker（阶段 2 用）

> 阶段 1（当前 batch4.5）：adapter-only + 冻结 thinker。
