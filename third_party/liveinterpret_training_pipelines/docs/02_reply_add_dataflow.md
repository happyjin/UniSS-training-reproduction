# 02 · Reply-Add 完整数据流

本文详细说明 upstream 路径（reply-add / full-add）在 `forward` 中的完整计算流程，
以及两种布局的本质区别。对应代码：`modeling_qwen_omnispeak.py` 的
`_build_upstream_text_channel(...)` 与 `forward(...)`。

---

## 1. 核心思想：文本信息"叠加"到 codec 序列

Qwen3-TTS talker 的输入是**两条同长度通道逐帧相加**：

```
talker 输入 = codec_channel(声学) + text_channel(文本)
```

- **codec_channel**：`[control_prefix..., codec_bos, frame_0, frame_1, ...]`，即声学 token 的 embedding。
- **text_channel**：与 codec 通道**等长**，每个位置放一个文本 embedding（或 `tts_pad`）。

本分支要做的，就是把 **thinker 产出的译文信息**，经过 adapter 投影后，
按正确的位置**填进 text_channel**，再与 codec_channel 相加送入 talker。

---

## 2. 完整数据流（reply-add）

以下步骤对应 `_build_upstream_text_channel`（编号与代码注释一致）：

```
① thinker 前向（run_thinker）
      源音频 + 译文 prompt → thinker
      产出: thinker_hidden_states [B, T_thinker, H_thinker]
            thinker_text_embeds   [B, T_thinker, H_thinker]   (token embedding)
            text_logits           (用于 thinker 文本 CE loss)
      并抽取 reply 段(译文对应位置): reply_hidden / reply_embed

② 构造 thinker_signal（信号 = hidden + embed）
      thinker_signal = reply_hidden + reply_embed        # signal_type=hidden_plus_embed

③ adapter → text_projection
      adapted   = thinker_adapter(thinker_signal)         # H_thinker → H_text
      projected = talker.text_projection(adapted)         # H_text → H_talker
      # projected: [B, T_reply, H_talker]

④ 计算每条样本的 reply 有效长度 text_valid_lens（要 scatter 多少个文本 token）

⑤ 构造 tts_pad_embed（text_channel 的默认填充值）
      pad_emb = talker.text_projection(talker.model.text_embedding(tts_pad_id))

⑥ 把 reply 文本 scatter 进长度为 codec_len 的 text_channel
      text_channel = pad_emb.expand(B, codec_len, H)      # 全部先填 tts_pad
      for b in batch:
          start = codec_bos_index[b]                       # codec_bos 的位置
          span  = text_valid_lens[b]
          text_channel[b, start:start+span] = projected[b, :span]
      # ⇒ codec_bos 之前全是 tts_pad，只有 reply 从 codec_bos 开始逐帧填入

⑦ (仅 full_add) 再把 thinker prompt 段 + text_bos 填入前缀位置（见下节）

最终:
      full_emb = codec_channel + text_channel   → 送入 talker
      talker 产出 last_hidden → codec_head(主码) + code_predictor(残差码)
```

关键点：**reply-add 下 `codec_bos` 之前的位置全部是 `tts_pad`**，只有译文（reply）从
`codec_bos` 开始逐帧叠加到 codec 上。

---

## 3. reply-add vs full-add 布局对比

二者共用上面的流程，唯一区别在**第 ⑦ 步是否填充 prompt 前缀**，由环境变量
`QWEN_OMNISPEAK_FULL_ADD_LAYOUT` 控制：

### reply-add（默认，`full_add_prompt_len=None`）

```
text_channel: [ tts_pad , tts_pad , ... , tts_pad | reply_0 , reply_1 , ... , pad ]
                └──────── codec_bos 之前 ────────┘ └────── 从 codec_bos 起 ──────┘
codec_channel:[ control_prefix............ , codec_bos , frame_0 , frame_1 , ...    ]
```

- 前缀位置只有 `tts_pad`，不含 prompt 语义。
- **更贴近流式推理**：解码第 t 帧时，text 通道只依赖到目前为止已产出的 reply，
  无需完整 prompt 上下文一次性对齐。

### full-add（`QWEN_OMNISPEAK_FULL_ADD_LAYOUT=true`）

```
text_channel: [ prompt_0 , ... , prompt_{P-1} , text_bos | reply_0 , reply_1 , ... ]
                └──── thinker prompt 段(hidden+embed) ──┘ └──── reply 段 ────────┘
codec_channel:[ codec_mask × P .......... , codec_pad , codec_bos , frame_0 , ...  ]
```

- 前缀位置 `[0:P]` 填入 **thinker prompt 段的 (hidden+embed)** 经 adapter 投影的结果；
- `codec_bos - 1` 位置放 `text_bos`；
- 与 `Qwen2.5-Omni` 的 `talker_inputs_embeds = cat([prompt_part, text_bos, reply_part])`
  行为一致（只是这里是叠加而非拼接）。

代码（`_build_upstream_text_channel` 第 ⑦ 步）：
```python
if full_add_prompt_len is not None:      # full-add
    for b in batch:
        P = full_add_prompt_len[b]
        ph = thinker_hidden_states[b, :P]
        if signal_type == 'hidden_plus_embed':
            ph = ph + thinker_text_embeds[b, :P]
        text_channel[b, :P] = talker.text_projection(thinker_adapter(ph))
        text_channel[b, codec_bos_index[b]-1] = text_bos_emb
```

---

## 4. 为什么"非流式训练"却能支持"流式推理"

- 训练时用**完整序列**（teacher forcing，一次算完），属于非流式；
- 但 reply-add 的**布局设计**保证：第 t 个 codec 帧的 text_channel 只依赖 reply 中
  第 t 个及之前的 token（前缀是 tts_pad，不含未来信息）；
- 因此推理时可以**边产出 reply 边解码 codec**，逐帧对齐 —— 即流式。

这也是本分支默认选 reply-add 的原因之一。

---

## 5. Loss 组成

`forward` 计算三路 loss（权重可配，见 [`04_env_and_training.md`](04_env_and_training.md)）：

| loss | 来源 | 默认 |
|---|---|---|
| 主 codec CE | `codec_head` 预测主码 vs `codec_labels` | 启用 |
| 残差 codec CE | `code_predictor` 预测 16 组残差码 | 权重 `0.3` |
| thinker 文本 CE | thinker `text_logits` vs 译文 labels | 权重 `1.0`；thinker 冻结时无梯度，仅作监控 |

> collator 会把主 `labels` 设为 codec-aligned（`acc_labels`），并将 thinker 文本 labels
> 通过单独的 `thinker_text_labels` 字段传入 forward，使文本 acc 与 codec loss 互不干扰。
> 详见 [`03_dataloader.md`](03_dataloader.md)。
