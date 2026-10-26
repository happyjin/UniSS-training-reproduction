# 多说话人语音翻译完整方案 v7(执行版)

**Nemotron-3 说话人分离 + 非贝叶斯同传 baseline,G-STAR 式端到端,先离线后同传**

- 日期:2026-10-26(v1–v6 保留不动;本版是自包含的执行版,内容以本版为准)
- 已定的决定:
  - 只用 IHM-mix;只用审查通过的数据;
  - 离线和同传都用之前的非贝叶斯 baseline,不用 ABC、TTB、SDR,也不用贝叶斯音色记忆;
  - 不用 SCOPE 的贝叶斯后处理;
  - 输出用 G-STAR 同款特殊 token;
  - 不做 C0;
  - **不再使用 E0**(已于 20:36 停止,checkpoint 保留,不再投入资源)。

---

## 1. 数据现状:能不能训练

| 项 | 状态 | 位置 |
|---|---|---|
| IHM-mix 整场音频(171 场,100 h,逐个校验) | ✅ | `data/ami_multispk_si_v1/session_audio/ihm_mix/` |
| 译文 + 译音分片 + 逐条审查结果 | ✅ | `.../shards/`、`.../index/`、`.../reports/` |
| AMI 词级时间标注 | ✅ | `.../annotations/ami_public_manual_1.6.2.zip` |
| 已审查的段落(train / dev / test = 25,676 / 3,566 / 2,815) | ✅ | `.../e2e_v0/pieces_*.jsonl`(段落和审查可复用;v0 的输出格式不用) |
| Nemotron-3 后验 + RTTM(离线、流式两种配置) | 🔄 正在 8 卡上导出(20:37 开始) | `.../diar/nemotron_{offline,lowlat}/` |
| 新格式的训练目标和训练行 | ⏳ 后验导出完就能构造 | `.../e2e_gstar_v1/` |

**结论**:音频、译文、审查和时间标注都已就绪。还差 Nemotron 后验(正在导出)和新格式目标(导出后约半小时构造完),之后就能训练。

---

## 2. 系统:G-STAR-SI baseline

```
会议单通道音频(IHM-mix 16 kHz)
 ├─ 说话人分离支:Nemotron-3-Diarization(冻结,8 槽位,整场槽位一致)
 │     离线配置 / 流式配置(1.04 s 前瞻)→ 后验 [T, 8](80 ms)+ RTTM(谁在什么时候说话)
 │     → 后验插值到音频 token 网格 → SD 投影器(Conv1d k5 s5 + MLP,约 6.3M)→ 说话人 token s_j ∈ R^2048
 ├─ 同传支:原版 Qwen3-Omni-30B-A3B Thinker 的 AuT 编码器(冻结)→ 音频 token a_i(约 13/s,2048 维)
 └─ 融合:<|audio_start|> a0..a4 s0 a5..a9 s1 … <|audio_end|>(K:1 交错,同 G-STAR)
        → 同一个 Thinker LLM(原版 + 注意力 LoRA r32,48 层)
        → 输出 <|spk k|><|t起|>译文<|t止|> …(新增 251 个时间 token + 8 个说话人 token,放在词表空闲行)
        → UniSS Talker T1c(冻结):每个槽位一条流,音色 = 该槽位"独自说话"的源音频(≤6 s)
        → 译音 + 会议记录(XML / JSON,分支同款)
```

- **端到端**:单一模型图。分离支是冻结的子网络;谁说的、何时、说什么由大模型一次输出。没有"先转写再翻译"的文字接口,没有贝叶斯后处理。
- **输出格式**:说话人 token 放在段首(同传开口前就要知道是谁)。导出会议记录时转回 G-STAR 的 `<|t起|>文字<|t止|><|spk k|>` 格式。

---

## 3. Pipeline

### 3.1 数据(D 步骤)

| 步骤 | 内容 | 状态 |
|---|---|---|
| D1 | 专用环境 `venvs/nemo_diar`(NeMo 3.1.0,torch / torchaudio 2.11.0)+ Nemotron-3 模型 | ✅ |
| D2 | 171 场 × {离线, 流式}:后验 [T80, 8] + RTTM(`data_clean/dump_nemotron_posteriors.py`、`scripts/run_nemotron_dump.sh`) | 🔄 |
| D3 | 参考说话人 0/1 活动(AMI 词级时间,80 ms),用于上限实验 | ⏳ |
| D4 | 每场:参考说话人 → Nemotron 槽位的 Hungarian 映射(按帧重叠),离线、流式各一份 | ⏳ |
| D5 | 窗口:①参考句组(已审查);②按 Nemotron 活动切窗(1 s 无活动关窗,≤20 s);窗口里有任何一句没通过审查就整窗剔除 | ⏳ |
| D6 | 目标:窗口内每句 → `<\|spk 槽位\|><\|t起\|>译文<\|t止\|>`,按源开始时间排序;纯犹豫词不出片段 | ⏳ |
| D7 | 训练行:每窗口一行(离线),后验离线和流式配置各一半;另混入 15% 单人整句翻译行 | ⏳ |

D3–D7 由一个脚本完成:`data_clean/build_ami_gstar_si_rows.py`。

### 3.2 训练(只训练一次)

| 项 | 设置 |
|---|---|
| 入口 | `training/thinker_gstar_model.py`(交错输入、新 token 行、SD 投影器、分层损失、arange 位置)+ `training/pretrain_thinker_gstar.py` + `scripts/run_thinker_gstar.sh` |
| 可训练 | SD 投影器;259 个新 token 的嵌入和输出头行;Thinker 注意力 LoRA r32(全部 48 层) |
| 损失 | 交叉熵;时间 token ×1.5,说话人 token ×2,其余 ×1 |
| 扰动 | 后验以 0.2 的概率加噪或平移 ±1 帧 |
| 配置 | Megatron 8 卡;GBS 64,MBS 8;全量 shuffle;LoRA lr 1e-4,投影器和新 token 5e-4;warm-up 100,cosine;约 2 遍数据(约 1.5 h);`setsid nohup` 运行 |
| 合并 | `training/merge_thinker_gstar.py`:LoRA 合进原版 Thinker;新 token 行写入空闲行;分词器加 token;投影器另存 |

### 3.3 推理

| 模式 | 流程 |
|---|---|
| 离线 | 整场 → Nemotron(离线配置)→ 按活动切窗 → 每窗一次生成全部片段 → 整场时间 → XML / JSON;Talker 按源开始时间排布(配音式) |
| 同传 | 音频流 → Nemotron(流式配置)→ 因果切窗 → 每 Δ 秒读一次,生成续写 → **LA2** 提交(本次与上次续写的最长公共前缀,以 token 计,包含说话人和时间 token)→ 窗口结束时提交剩余 → 按说话人 token 进对应 Talker 流 → 单通道(换人时停 0.2 s) |
| 引擎 | `evaluation/gstar_si_stream.py`,HF 批量生成,离线和同传共用;同一轮读取的所有窗口拼成一个 batch |

---

## 4. 实验

### 4.1 必做(按执行顺序)

| # | 编号 | 内容 | 回答的问题 |
|---|---|---|---|
| 1 | **O-G1** | 离线主系统 | 多说话人离线语音翻译能做到多好(质量上限) |
| 2 | **S-G1** | 同一模型 + LA2,Δ = 1 / 2 / 3 s | 同传的质量和延迟曲线;相对离线损失多少 |
| 3 | **B-LA2** | 原版 Thinker + LA2,不接说话人;同样的窗口和 Δ | 不做多说话人的同传参照 |
| 4 | O-nosd | O-G1 的说话人 token 置零(另训一次,格式不变) | 说话人分离支的贡献 |
| 5 | O-oracle / S-oracle | 推理时用参考 0/1 后验(不重训) | Nemotron 分离误差造成的损失上限 |
| 6 | S-single | S-G1 在 RealSI 英→中、ACL60 英→中 | 单人同传是否退化 |

### 4.2 选做(看必做结果再定)

- O-enc:加 AuT LoRA;
- O-add:说话人 token 相加,不交错;
- O-local:段内局部编号;
- 同传专门训练(读取行训练 wait 标记,对比 LA2);
- SCOPE 的 BSSC 和条件链。

### 4.3 判据

| 对比 | 要求 |
|---|---|
| O-G1 对 O-nosd | cpBLEU、DER、说话人数正确率显著更好 |
| S-G1 对 B-LA2 | 同一 Δ 下 cpBLEU 显著更高;不分说话人的 BLEU 不降超过 1 |
| S-G1 对 O-G1 | 损失量化清楚;Δ = 1 s 时 StreamLAAL ≤ 3 s(含 1.04 s 前瞻) |
| S-single 对 B-LA2 单人 | 差在 ±0.5 BLEU 以内 |

---

## 5. 指标

| 类别 | 指标 |
|---|---|
| 带说话人的译文 | cpBLEU / cpchrF(整场口径,槽位按整场 Hungarian 匹配参考说话人;也报窗口口径) |
| 谁在什么时候说话 | 从输出片段算 DER(collar 0);并列报告 Nemotron 自身的 DER;说话人数正确率;片段归属准确率 |
| 不分说话人 | 整场 BLEU(mwerSegmenter 重切分) |
| 语音 | cp-ASR-BLEU;按人音色相似度(WavLM,与该人 IHM 原声);音色串的比例 |
| 延迟(同传) | LAAL 和 StreamLAAL(含前瞻);fluency 提交包 |
| 分组 | 单人 / 多人 / 重叠窗口;严格审查子集 |
| 显著性 | 按会话的配对 bootstrap 1000 次 |

---

## 6. 时间线(布拉格时间)

| 时间 | 内容 |
|---|---|
| 10-26 20:37– | D2 导出(8 卡,进行中);写 D3–D7 构造脚本,完成后马上构造数据 |
| 10-26 夜 | 写模型类、训练入口、合并脚本、HF 推理(离线 + LA2);2 卡冒烟。冒烟通过后 O-G1 用 `setsid nohup` 在 8 卡上开训(约 1.5 h) |
| 10-27 上午 | O-G1 离线测试 → S-G1(Δ = 1 / 2 / 3)、B-LA2 同传测试;O-nosd 训练 |
| 10-27 下午 | O-nosd、oracle、单人回归测试;结果表 |
| 10-28 | 视结果做选做实验;报告 |
