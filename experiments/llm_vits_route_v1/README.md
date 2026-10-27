# llm_vits_route_v1 — LLM × VITS 流式路线（手册 S6 + T2 + T3）

本目录实现《LLM × VITS: Component Replacement and Verifiable Theory》手册中推荐的主路线，包含数据、训练、推理和评测。所有数值都来自本目录的实际运行，未测量的指标不填写。

## 需求
- 新建独立目录，实现手册路线的数据、训练、推理全流程。
- 先给 S1–S6、T1–T5 排序，固定 baseline 后按顺序执行，与 baseline 对比并分析。
- 先用简单数据集（LJSpeech，手册 P0）验证路线是否可行。
- 尽量用满全部 GPU；多任务时多卡并行。每段代码验证通过后 commit，最后推送到当前分支。

## 排序（可控性 × 理论价值 × 本机可验证性）
1. **S6** 有状态、未来条件化的随机时长（future-r + rho）——最可控的流式组件（手册 §4.1 / S6.5）
2. **T3** 轨迹级校准的不可逆提交——证书与“全上下文计划”的偏差有保证
3. **T2** 声学几何而非词汇几何：声学影响门控（E6）对比词汇熵门控（E5）
4. **T5** 计算感知停止——扩展，本轮不做
5. **T1** 未来影响 Γ——需要 CF-Prefix 反事实数据，本轮只用 E1/E2 前瞻曲线做诊断
6. S5 WavLM 判别器（辅助项）、S1 扩散、S3+T4、S2、S4——各自是独立项目，本轮不做

## Baseline
- **E0** 离线 VITS（全文，官方目标，字符输入），作为离线参考
- **E1（主 baseline）** 同一冻结 backbone，文本编码器和 SDP 在前缀上微调；固定前瞻 k 个词后提交
- **E2** E1 + Qwen3-1.7B 前缀隐状态门控适配器（G2）——“只是语义特征吗？”这一更强的 baseline
- **E5** 词汇熵等待策略（Qwen 下一 token 熵门控）——策略 baseline

## 方法
- **S6** = E2 + future-r（LLM 采样 M=8 条未来续写 → 想象前瞻编码 → r 的先验；后验看真实块韵律）+ rho（已提交单元的实际时长 → 节奏状态）。新条件在 SDP detach 之后注入（S6.2），所有新分支零初始化。
- **E6 / T2** 声学影响门控：E1 探针在 M 条想象未来下规划当前单元，计划离散度大则等待。
- **T3** 风险头 + 轨迹级 split-conformal 半径 Q_conf；证书 B ≤ ε 才提交，到 deadline 时为未认证兜底提交并标记。

## 协议
- 数据：LJSpeech 1.1，官方 VITS train/val/test 划分；用 crc32 从 train 中留出 390 句作为 calib（T3），任何系统都不在 calib 上训练。
- 单元：带标点的空格分词；单元拥有前导空格（与 LLM token 流 `" word"` 一致），因此看到单元 j 时无法得知句子是否结束。
- 提交：单元 i 在单元 j ≥ i 到达后提交；已提交的时长和潜变量永不修改；波形逐单元增量渲染（flow 只看已提交帧，解码器带左上下文）。
- 公平性：所有系统共享冻结的 backbone 和相同的 d* 时长标签；随机数按 (utterance, 单元, 绝对帧) 生成（common random numbers）。
- 指标：词时长 log-MAE、边界停顿 log1p-MAE 与 F1（≥10 帧）、语速误差（对照真实录音的 MAS 对齐）；平均前瞻（词数）；Whisper-large-v3 WER、UTMOS22、词级 log-F0 误差；T3 的认证率与轨迹违约率。

## 流水线
`prepare_ljspeech.py` → `units_v2.py` → `train_backbone.py`（E0）→ `extract_qwen_prefix.py` → `sample_future_candidates.py`（vLLM）→ `align_gt.py` → `train_stream.py`（各变体）→ `run_stream_eval.py`（策略）/ `offline_plans.py` → `fit_t3.py` → `eval_metrics.py` / `eval_audio.py`；多卡调度用 `make_jobs.py` + `run_jobs.py`。
