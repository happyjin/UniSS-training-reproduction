# 03 · 数据管线改动（Dataloader / Preprocessor / Collator）

## 先回答一个常见问题：DataLoader 本身改了吗？

**没有。** 我们**没有**自定义 PyTorch `DataLoader`、`Sampler` 或重写 swift 的
`get_train_dataloader`。批采样、shuffle、分布式切分全部沿用 swift/PyTorch 原生实现。

我们改动的是**数据处理的三个环节**（都在 `custom/` 里）：

```
原始 parquet 行
   │  ① dataset.py   BridgeAlignmentPreprocessor  (行级预处理 / 字段透传)
   ▼
样本 dict (messages / audios / codec 字段 ...)
   │  ② template.py  encode()                     (单样本 → 张量, 构造 upstream 专属字段)
   ▼
单样本编码结果
   │  ③ template.py  _data_collator()             (batch 打包 / padding / 构造对齐标签)
   ▼
喂给 model.forward 的 batch
```

DataLoader 只是把 ③ 的输出按 batch 吐出来，本身没动。以下逐环节说明改了什么。

---

## ① dataset.py — 行级预处理（`BridgeAlignmentPreprocessor`）

相对 public 版的改动：

1. **新增任务类型** `text_to_speech_translation`，并纳入 `SUPPORTED_TASKS`。
2. **接受预编码 codec**：原版要求每行必须有 `target_audio`；现在
   `target_audio` **或** `target_codec_codes` 有其一即可（支持离线预编码 parquet）。
   ```python
   target_audio = row.get('target_audio')
   target_codec_codes = row.get('target_codec_codes')
   if not target_audio and not target_codec_codes:
       raise ValueError('requires target_audio or target_codec_codes.')
   ```
3. **透传新字段**（加入 `EXTRA_KEYS`，使其能穿过预处理到达 template）：
   - `target_codec_codes`（目标音频离线预编码 `[T,16]`）
   - `source_codec_codes` / `source_codec_len` / `source_text_ids`（ICL 模式用）

> 预处理仍是逐行 `map`（swift `RowPreprocessor`），未改变并行方式（`DATASET_NUM_PROC`）。

---

## ② template.py — 单样本编码（`encode`）

这里构造 **upstream (reply/full-add) 路径专属的输入字段**（public 版没有）：

| 新增字段 | 含义 | 用途 |
|---|---|---|
| `target_text_ids` | 译文 token ids（thinker tokenizer 编码） | 填入 text_channel 的 reply 段 |
| `codec_bos_index` | `codec_bos` 在 codec 序列中的位置（= `len(control_prefix)-1`） | reply 从这里开始逐帧叠加 |
| `full_add_prompt_len` | （仅 full-add）thinker prompt 段长度 | full-add 时填 prompt 前缀 |
| `codec_input_ids` / `codec_labels` / `codec_attention_mask` / `codec_residual_labels` | codec 声学序列及其监督标签 | talker codec loss |
| `source_codec_codes` / `source_text_ids` | （仅 ICL）参考音频 codec + 源文本 | ICL 前缀 |

关键逻辑：
- 若样本带 `target_codec_codes`（离线预编码），直接构造 codec 字段，**不再实时编码音频**
  （省去训练时的 codec encoder 前向）。
- `codec_bos_index = len(control_prefix) - 1`，告诉 forward 从哪一帧开始叠 reply。
- ICL 字段仅在 `QWEN_OMNISPEAK_USE_ICL=true` 时透传。

---

## ③ template.py — 批打包（`_data_collator`）

在 swift 父类 collator 基础上，追加 upstream 专属的 padding 与标签构造：

1. **codec 张量 padding**：`codec_input_ids`(pad_id) / `codec_labels`(-100) /
   `codec_attention_mask` / `codec_residual_labels`（2D 时间维 pad）。

2. **构造 codec-aligned 主标签 `acc_labels`**（关键）：
   ```python
   acc_labels = codec_labels.new_full(codec_labels.shape, -100)
   acc_labels[:, 1:] = codec_labels[:, :-1]     # 右移一位
   res['thinker_text_labels'] = res['labels']   # 原文本 labels 另存
   res['labels'] = acc_labels                   # 主 labels 改为 codec-aligned
   ```
   - swift 用 `preds[:-1]` vs `labels[1:]` 算 acc，故主码标签要右移。
   - **文本 labels 单独放到 `thinker_text_labels`**，让文本 CE / `token_text_acc`
     与 codec loss 互不串扰（主 `labels` 流保持 codec-aligned，和原版 bridge-concat 一致）。

3. **batch 化 upstream 字段**：`target_text_ids`(pad=tts_pad) / `codec_bos_index` /
   `full_add_prompt_len`，以及（ICL 时）`source_codec_codes`(2D 时间维 pad) + mask /
   `source_text_ids`。

---

## 小结：改了什么、没改什么

| 环节 | 是否改动 | 说明 |
|---|---|---|
| PyTorch DataLoader / Sampler | ❌ 未改 | 沿用 swift/PyTorch 原生 |
| shuffle / 分布式切分 | ❌ 未改 | swift 默认 |
| dataset 行预处理 | ✅ 改 | 新任务类型 + 接受预编码 codec + 透传新字段 |
| template 单样本编码 | ✅ 改 | 构造 `target_text_ids`/`codec_bos_index`/`full_add_prompt_len`/codec 字段 |
| collator 批打包 | ✅ 改 | codec padding + `acc_labels` 右移 + `thinker_text_labels` 分流 + upstream 字段 batch 化 |

> DataLoader 相关的**运行参数**（`--dataloader_num_workers` / `persistent_workers` /
> `group_by_length` 等）可通过入口脚本环境变量调整，见 [`04_env_and_training.md`](04_env_and_training.md)，
> 但这些是配置项，不是代码层面的 DataLoader 改动。
