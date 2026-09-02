# 三家族 GPU canary —— C 方案的数据路径验收

**结论:三个家族全部通过。C 现在可以训练。**

八卡、一个 optimizer update、GBS 128 / MBS 2、载入 B′ 的 `iter_0001132`,
每个家族各跑一次(smoke 上限 2 步且一个家族独占一个 global batch,
所以家族必须显式指定 —— 这同时也是更强的检查:它证明的是**各家族自己的 loss 真的开火了**,
而不只是"这一步没崩")。

## 结果

| 家族 | 内容 loss | 分母 | `boundary_ce` | `eos_ce` | `weighted/boundary_eos` | 其余 loss | 步时 |
|---|---:|---:|---:|---:|---:|---|---:|
| `p2st_streaming_asr` | `asr_ce` **4.664** | 4,849 | 0.646 | 2.196 | 1.421 | **全为 0** | 62.0 s |
| `p2st_incremental_mt` | `mt_ce` **3.447** | 23,502 | 0.712 | 3.583 | 2.148 | **全为 0** | 39.0 s |
| `p2st_streaming_tts` | `semantic_ce` **4.598** | 55,814 | 1.798 | 0.439 | 1.119 | **全为 0** | 27.4 s |

三次都是 `nan 0 / skipped 0`,都走到 `[after training is done]`。

## 三条验收判据

**① `StageAObjective._inject_causal_glm` 没有抛异常。**

这是唯一无法靠 CPU 测试确立的一条。它要求前端在该行波形上的 token 数
**等于** `glm_lengths`,只容忍一个末尾槽位。而 ASR 家族的诊断给出:

```
diagnostic/acoustic_rows:                  3922.75
diagnostic/causal_glm_terminal_extensions:    0.000   ← 一个槽位都没补
diagnostic/causal_glm_agreement:           1.098e-03
diagnostic/bridge_residual_rms:            0.152
```

**`terminal_extensions = 0` 意味着闭式 `ceil(samples/1280)` 在全部 3,923 个声学行上精确命中**,
trainer 一次都没有动用它的容错分支。音频切点这条路是对的。

`causal_glm_agreement` 是 1.098e-3,与既有血脉一直记的 ~0.001 一致 ——
这是离线 GLM-4 码本与因果前端码本的差异,是诊断项而非 loss,已在
`FRONTEND_PREFIX_PARITY.zh-CN.md` 里查清。

**② `validate_family_denominators` 通过。**

三个家族的必需分母都为正,否则 trainer 会在 backward **之前**抛异常:

```
p2st_streaming_asr  -> asr_ce, boundary_ce, eos_ce
p2st_incremental_mt -> mt_ce,  boundary_ce, eos_ce
p2st_streaming_tts  -> semantic_ce, boundary_ce, eos_ce
```

**③ 纯 CE 成立 —— 不该开火的一个都没开火。**

`replay_ce` / `v1_asr_kl` / `phase3_kl` / `commit_consistency` 与全部
margin、roll-in、binary 项在三次运行里**都精确为 0**,分母也为 0。
`diagnostic/acoustic_rows` 在 MT 与 TTS 家族为 0(纯文本),只有 ASR 家族为正。

## 一个顺带的设计确认

ASR 家族里 `content_end_ce` = **0.6460545**,`boundary_ce` = **0.6460545** —— 逐位相同。

因为在 C 的 ASR 序列里,**每一个 boundary token 就是那个 `END_CONTENT`**。
这正是 C 相对交织池的结构差别:交织池的 boundary 桶混着
WAIT/WRITE/TASK/语言/速度,让 32.8% 的 token 份额塌到 4.2% 的子类上;
这里桶内只有一个无歧义的类。

## 已知的下一步风险(非阻塞)

- **步时与家族强相关**(62 / 39 / 27 s),而这三步都含首步编译开销,不能直接外推。
- **`eos_ce` 起点偏高**(ASR 2.196、MT 3.583):模型从未被训练过在 `END_CONTENT` 后发 EOS。
  这是预期的冷启动,不是缺陷 —— 但要盯它是否下降。
- **TTS 的 `boundary_ce` 是三者最高**(1.798):`END_SEMANTIC` 是 B′ 里退化最严重的那一类
  (监控项涨 32.8%),从这里开始训练正好是要修的东西。
