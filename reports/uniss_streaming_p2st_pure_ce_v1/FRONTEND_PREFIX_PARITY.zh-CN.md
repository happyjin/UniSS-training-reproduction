# 前端前缀一致性验证 —— C 方案上 GPU 前的必验项

**结论:方案可行,但按我最初的写法训不起来,需要一处明确的修正。**

命令与产物:
- `experiments/uniss_streaming_p2st_pure_ce_v1/evaluation/frontend_prefix_parity.py`
- `reports/uniss_streaming_p2st_pure_ce_v1/FRONTEND_PREFIX_PARITY.json`
- 12 条真实 valid 轨迹,201 个事件边界,V1 `iter_0000381` + `glm4_tokenizer`

## 为什么必须验这一条

C 的 prefix-to-prefix ASR 任务只把源 GLM 位置 `0 .. source_glm_end` 放进 prompt,
而既有的 `build_streaming_asr_task` 一直传整条轨迹。训练数据集读的是**整条**波形
(`waveform_length = waveform.numel()`),只用 `glm_lengths` 在 GLM 侧截断 ——
也就是说前端仍然看到了真实会话此刻还没听到的音频。
这只有在前端**真的是块因果**时才安全。

而 `run_cached_frontend` 按 160 ms 分块推送并设 `is_final = end == len(waveform)`,
截断调用会把前缀最后一块标成 final,整条调用不会。所以这不是可以假设的性质。

## 测到了什么

| 判据 | 结果 | |
|---|---|---|
| **因果性**:前缀的 token 与整条运行的对应位置逐位相同 | **201 / 201** | **通过** |
| **永不短缺**:前端返回的 token 数从不少于 `source_glm_end` | **201 / 201** | **通过** |
| 数量恰好相等 | 171 / 201(85.1%) | 报告,不判定 |
| 偏移分布 | **`+0` × 171,`+2` × 30,从不为负** | |

**因果性 201/201 是这次验证的核心结论:块因果是真的。** 把音频切在
`source_pcm_end` 再跑前端,得到的就是整条运行的那个前缀,一位不差。

## 发现的硬阻塞点

`StageAObjective._inject_causal_glm` 有一处硬校验:

```python
length = int(batch["glm_lengths"][row].item())
causal_length = int(pooled_lengths[row].item())
if causal_length == length:
    pass
elif terminal_codec_extension_deficit_samples(...) is not None:
    hidden = torch.cat((hidden, hidden[-1:]), dim=0)   # 只容忍 1 个末尾槽位
else:
    raise ValueError("causal WhisperVQ token count differs from packed GLM coverage")
```

我最初的 builder 设 `source_glm_length = event.source_glm_end`(截断值,例如 40)
而 `source_audio` 指向整个文件 → `causal_length` = 87 ≠ 40 → **直接抛异常**。
**这是真阻塞,不是软性不一致。**

## 修法(由测量结果唯一确定)

**必须连音频一起截,并且用前端自己的计数,而不是信任 `source_glm_end`。**

C 的 pool 打包器需要:

1. 每条轨迹跑一遍前端,记录每个 `source_pcm_end` 处的 token 数;
2. 令 `source_glm_length = len(prefix_tokens)` 而不是 `event.source_glm_end`;
3. 在打包行里带上 `source_pcm_end`,由 C 自己的 dataset 切波形。

这样 `causal_length == length` 是**构造上成立**的,校验永远不会触发。

两条测量结果保证了这个修法安全:

- **偏移从不为负**(201/201),所以总存在一个前端计数可用;
- **因果性 201/201**,所以多出的 ≤2 个 token 仍然只来自 `waveform[:source_pcm_end]`,
  即会话已经听到的音频 —— 不是未来信息。

**注意 dataset 必须是 C 自己的**:基础实验的 `runtime_dataset.py` 处于逐位冻结审计之下,不能改。

## 一个顺带查清的既有事实(不是缺陷)

轨迹里记录的 `source_glm_delta` 与前端产出的码**一致率只有 0.002**。
但这无害,因为 `glm_ids` **不是模型输入**:

```python
codes = self._nearest_codes(hidden)                 # 因果前端的码
base  = F.embedding(codes + glm_semantic_offset, W)
corrected.index_copy_(..., base + residual)          # ← 模型真正看到的
teacher_ids.append(batch["glm_ids"][row, :length])   # 记录的离线 GLM-4 码
agreement = (causal == teacher).float().mean()       # ← 仅诊断
```

它在 trainer 里的注册名就是 `"diagnostic/causal_glm_agreement"`,
不进任何 loss,而 B′ 整个训练日志里它一直是 **~0.001**。
两个 tokenizer 速率相同(12.5 tok/s)但码本分配不同 ——
Stage-A 那 381 步适配的正是这个差异。

**所以 C 的 `source_glm_ids` 字段只需要长度和取值范围正确,内容不影响学习。**
