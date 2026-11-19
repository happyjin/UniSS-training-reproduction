# E10：LiveInterpret S5（Voxtral + Qwen3.5-4B Thinker–Talker）接入多说话人同传 baseline 的方案

- 撰写：2026-11-19（布拉格时间）。本文是方案，不是预登记。按惯例，正式预登记 `e10_liveinterpret_multispk_prereg.zh-CN.md` 要在看到任何 dev/test 输出之前另写（§7 给了草稿要点）。
- 目标：用和 B 相同的数据、标签、diarizer 输入和评测链，把同事的流式 Thinker–Talker 模型（下称 **LI-S5**）训练成多说话人同传，作为 Table 1 中"同数据训练"的新 baseline。
- 本文提到的路径都已在本机核对存在，"待确认"的除外。

---

## 0. 结论速览

1. **资产可以访问，已落盘并校验。**
   - 推理/训练代码：`third_party/liveinterpret_training_pipelines/`（feature/streaming-si 分支）。
   - checkpoint：`third_party/liveinterpret_training_pipelines/checkpoints/p4mix_2n_20261117/checkpoint-29773/`，sha256 与 `_MODEL_READY.json` 一致。
   - 这个 checkpoint 只是 LoRA adapter（r64，330 MB），必须配合 S4 底座和声码器使用。这两样都在 S3 上，可读，**尚未下载**（见 §1）。
2. **LI-S5 和 B 的差别。** LI-S5 是单说话人、句级（≤30 s）的模型：640 ms 一块，每块后做一次 `<|wait|>`/`<|W|>` 决策，写出的增量立刻交给 Talker 出声。它没有说话人通道，音色取自"最近 2 s 源音频"。B 则是在 AMI 多说话人窗口上训练的：输入有 diarizer 后验，输出带 `<|spk k|>`，读写由 deadline 规则控制。
3. **主方案（E10）：B 形式，同数据微调。**
   - 把 B 用的 AMI 窗口和 MT 目标文本按同样的 deadline 规则转成 LI-S5 的 `si-rw-m1-v0` 训练记录。
   - 在增量文本里加 `<|spk k|>`。不新增词表：复用 tokenizer 里已有、但 S5 没用到的 `<|dur:*|>` 行。
   - 把 Nemotron lowlat 后验因果地加到 audio_pad 嵌入上。两者都是 80 ms 一帧，可以逐帧对齐。
   - 从 checkpoint-29773 出发，按 joint scope 训练 LoRA。
   - 推理时按说话人分路，每个说话人用自己的参考音合成，评测链与 B、E7、E8、E9 完全相同。
4. **E10-0（零样本，外部系统式）兼做可行性冒烟。** 不训练，直接把 checkpoint-29773 当外部系统跑在线窗口，说话人用因果重标（同 E7/E8）。它的数据可以放附录；主表放不放，需要你决定（§9）。
5. **需要你拍板的事**集中在 §9：
   - 新建 venv；
   - 向同事确认训练数据是否和 AMI/AliMeeting 有重叠；
   - joint 还是 thinker-only；
   - E10-0 是否进主表。

---

## 1. 已完成：下载、解压、校验

| 项 | 位置 | 状态 |
|---|---|---|
| 代码 zip（正确分支） | `third_party/_liveinterpret_zip/liveinterpret_training_pipelines-feature_streaming-si.zip` | 已解压到 `third_party/liveinterpret_training_pipelines/` |
| 先前误解压的 master | `third_party/liveinterpret_training_pipelines_master/` | 改名保留，未删除；里面没有 `custom/streaming_si/` |
| 推理入口 | `third_party/liveinterpret_training_pipelines/custom/streaming_si/inference/infer_s5_streaming_dynchunk.py` | 已读（745 行） |
| S5 adapter | `third_party/liveinterpret_training_pipelines/checkpoints/p4mix_2n_20261117/checkpoint-29773/`（来源 `s3://stevehguo-ap-southeast-3/speech_model_backup/zezhongjin/exps/p4mix_2n_20261117_eval_assets/checkpoint-29773/`，7 个文件，330 MB） | 已下载；`adapter_model.safetensors`、`args.json` 的 sha256 与清单一致 |
| S4 底座 | `s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift/trained_streaming_si_4b_enzhjako/S4_1031v2_ckpt93767/`（3 个分片，共 13.0 GB） | 可读，未下载 |
| 声码器 | `s3://.../streaming_si_swift/base_streaming_si_4b/speech_decoder/`（`flow.pt`、`hift.pt`、`cosyvoice3.yaml`、`campplus.onnx`、`manifest.json`，1.4 GB） | 可读，未下载。底座目录里没有这个子目录，推理时要用 `--cosyvoice_dir` 指过去 |
| CosyVoice 官方仓库 | 推理需要 `--cosyvoice_repo`（`--recursive` 克隆） | 待准备 |

**checkpoint 的来历**（`args.json`、`trainer_state.json`）：
- ms-swift 4.4.0，`tuner_type=streaming_si_s4`，`train_scope=joint`；
- 从 `s5_enzhjako_off_mchunk_ft_lora_r64 …/checkpoint-10000` 继续训练；
- LoRA r64、α128，`modules_to_save = talker.thinker_proj, talker.fusion`；
- lr 1.2e-5，cosine 退到 2e-6；24 卡 × BS 16，共 29,773 步（1 epoch，约 1,140 万条记录）；
- 选的是 eval_loss 最低的最后一步（dec_macro_f1 最高的是 16000 步）。
- 末步 eval：`dec_macro_f1` 0.848，`write_f1` 0.767，`text_acc` 0.749，`speech_acc10` 0.760。

---

## 2. LI-S5 推理代码要点（读码结论）

**结构**
- 冻结的 Voxtral 流式编码器（12.5 Hz，每帧 80 ms）
- → Thinker：Qwen3.5-4B，24 层 Gated DeltaNet + 8 层全注意力
- → GatedBridge + Fusion
- → Talker：Qwen3-0.6B 结构，自回归生成 CosyVoice3 FSQ 25 Hz 单码本 speech token
- → CosyVoice3 flow/HiFT，输出 24 kHz。

**序列布局**（训练和推理逐 token 一致，见 `s4_streaming_dataset.py` 的 docstring）：
```
[instr: "\nTranslate the streaming speech into {Chinese|English} simultaneously:\n"]
<|chunk|> <|audio_pad|>×n_0 [<|src_end|>] <|wait|>
<|chunk|> <|audio_pad|>×n_1                <|W|> 译文增量_1
...
<|W|> 译文增量_K <eos>
```

**驱动循环**（`stream_translate` / `S5Session`）
- 每步喂一块音频，默认 640 ms，可选 1280/1920/2560 ms。
- 然后只在 `{wait, W}` 里做决策（源读完后再加 eos）。
- 如果是 W，就逐 token 采样增量，直到遇到 `<|chunk|>`、`<|wait|>` 或 eos。
- 用本段增量在 Thinker 中的末层 hidden 作为 Talker 条件（`cond_mode=step`），立刻生成这一段的 speech token。
- 可选 `write_threshold`：当 P(W)/(P(W)+P(wait)) 超过阈值就写。

**音色**
- 不由 Talker LM 负责，而是 flow 解码时的 `spk_emb`（CAM++）加上 flow 声学 prompt。
- `S5Session` 用"最近 `prompt_sec` = 2 s 的源音频"做参考，每 2 s 滚动刷新一次（`reference_policy=rolling`）。
- 后果：在多说话人场景里，换人以后的前几秒会用上一个说话人的音色。这是 E10 必须改的地方（§5.4）。

**长度与分段**
- 训练口径：`SI_MAX_AUDIO_SEC=30`，`max_length=1024`，`max_speech_len=750`（30 s）。
- 长流靠 Silero VAD 外部分段（`s5_segmenter.py`）。
- 我们的在线窗口最长 20.5 s（AMI Zh→En 3973 窗、Ali 4065 窗、AMI En→Zh 5126 窗，没有超过 30 s 的），**一个窗口就是一个会话，不需要 VAD**。这和 E7/E8 每个窗口 reset 的做法一致。

**时间线**
- `results.jsonl` 里每一步记录 `source_end_ms`（字段名虽然叫 ms，单位其实是秒）、`decision`、`text_delta`、`speech_tokens`、`elapsed_sec`。
- `streaming` 模式下另有 PCM 事件（`--event_jsonl`）。
- 我们的延迟按"不计计算时间"口径（paper-writing-style）：写出时刻 = 该步已读入的源时长。

**流式模式的限制**
- `--synthesis_mode streaming` 要求 `--retry 0` 且 `cond_mode=step`，并且需要 `campplus.onnx`。
- 不允许在整句级别重试，这和同传的定义一致，我们也照此执行。

---

## 3. 我们的多说话人 baseline B 是怎么训和怎么评的（要照搬的"形式"）

这一节来自仓库核对。引用文件：`paper/gstar_si_v17/sections/appendix_v17.tex`、`docs/multi_speaker_si_plan.zh-CN.md`、`reports/mist_w3po_v1/e9_cascade_multispk_results.zh-CN.md`。

| 维度 | B 的做法 |
|---|---|
| 训练语料 | 只用 AMI IHM-mix，官方切分。En→Zh：137 场、21,581 窗、36.8 h。Zh→En（中文重配音 AMI）：127 场、27,953 窗、48.5 h。AliMeeting 只做测试 |
| 目标文本 | 3 个 MT 的候选，用 XCOMET-XL + XLM-R-XL QE + chrF 选一条，再用 VoxCPM2 做可懂度检查（`speech_chain_all_local_en_zh_v1`） |
| 窗口 | 重叠或相接的句子合成一组，每组 ≤20 s（`build_ami_gstar_si_rows.py`、`build_ami_ms_rows.py`） |
| 说话人标签 | 不用 oracle：Nemotron 3 Diarization lowlat slot，整场会议做 Hungarian 映射（`data/ami_multispk_si_v1/e2e_gstar_v1/slot_map.json`） |
| 说话人输入 | diarizer 真实后验（每窗 `post/lowlat/*.npy`，形状 `(帧, 8)`，80 ms 一帧）。训练时扰动：p=0.2，高斯噪声 0.1，平移 ≤1 帧。`post_until_ms = t_ms − 1040` |
| 读写监督 | 每 1 s 读一次。译文按标点切块，块对应的源词结束后再等 1 s 就到期（Zh→En 按比例估计）。没有新到期内容的读以 0.5 概率保留。另混 20% 离线整窗行和 15% 单说话人行（`build_ami_gstar_stream_rows*.py`） |
| 输出格式 | `<|spk k|><|t a|>译文<|t b|>`，以 `<|box_end|>`（wait）或 `<|im_end|>`（结束）收尾 |
| 训练 | Megatron + Qwen3-Omni Thinker LoRA r32/α64，加 `sd_proj`、新 token 行。lr 1e-4→1e-5，GBS 64，8×H200，6000 iters。**取最后一步，不选 checkpoint** |
| Talker | T1c：单说话人同传 Talker，**不在多说话人数据上训练**。每个说话人的音色取其首次提交前、单独说话 ≥0.4 s 的片段，最长 8 s（`gstar_talker_v6 prep --causal --blocks --preset lowlat`） |
| 评测 | c2g06 在线窗口，1 s 读，`--final-read-post close_detect --policy wait --lag-ms 2000`。两个 TTS 种子（默认种子和 20261109）。ASR：Whisper-large-v3 / Paraformer-zh。`score_run --asr` 算 cp/tcpASR-BLEU；`table1_lat.sh` 在 chunk 协议下算 LAAL；`table1_analysis` 做配对检验和 Holm |
| 外部系统适配 | E7/E8：每个窗口独立会话，输出 `segs`，再由 `build_e*.py` 转成 `hyp_<sys>.jsonl` 和 `talker/results.jsonl`。说话人用 `overlap_score.relabel_causal`，提交时刻按在线协议钳位到 open/close_detect |

---

## 4. 差距对照：要接上 B 形式，LI-S5 缺什么

| 需要 | LI-S5 现状 | E10 的处理 |
|---|---|---|
| 多说话人训练数据 | 只有单说话人句级 records（`si-rw-m1-v0`），而且是内部数据 | 把 B 的 AMI 窗口转成 `si-rw-m1-v0`（§5.1）。只用开源数据训练，符合 open-data 规则 |
| diarizer 输入通道 | 无 | 新增 `sd_proj`：Linear(8→2560)，因果地加到 audio_pad 嵌入上（§5.2b） |
| 说话人输出 | 无 | 增量里加 `<|spk k|>`，复用未使用的 `<|dur:*|>` 行（§5.2a） |
| 时间 token `<|t a|>` | 无，时间由 step 结构隐含 | **不加**。每一段的写出时刻就是它所在 step 的 `source_end`，源区间按 E7/E8 的规则取 [上一段写出时刻, 本段写出时刻]（§5.4） |
| 按说话人分别出音色 | 只有一路滚动的源参考 | 按 slot 维护参考音，按说话人分路做 flow 解码（§5.4） |
| 读写粒度 | 640 ms | 训练和推理都用 640 ms（LI-S5 的原生粒度）。B 的 deadline 映射到第一个 `source_end ≥ deadline` 的 640 ms 步 |

---

## 5. 主方案 E10：B 形式同数据微调

### 5.1 数据：AMI 窗口 → `si-rw-m1-v0` records

新脚本：`experiments/s2st_baselines_v1/liveinterpret_multispk/build_li_records.py`（只用 CPU）。它不改 B 的任何数据文件，只读取以下内容：
- `data/ami_multispk_si_v1/e2e_gstar_v1/windows_{train,validation}_ref.jsonl`（En→Zh）；
- `e2e_gstar_zh2en_v1/` 下对应的 Zh→En 文件；
- `slot_map.json`；
- `post/lowlat/*`。

**每个窗口生成一条 record**，字段按 `make_smoke_fixture.py:_s4_record` 的生产字段集：
```json
{"schema_version": "si-rw-m1-v0", "record_id": "IS1006b_zh_0005", "source_audio": "zh2en/train/IS1006b_zh_0005.wav",
 "source_duration_ms": 6330, "source_transcript": "...", "target_lang": "en", "target_text": "...",
 "target_codec_codes": [...25Hz FSQ...],
 "trajectory": {"base_chunk_ms": 640, "codec": {"spec": "cosy3_fsq_25hz", "frame_ms": 40.0},
   "raw_target_num_frames": N,
   "steps": [{"source_start_ms": 0, "source_end_ms": 640, "action": "wait", "text_delta": "",
              "source_codec_spans": [], "commit_ms": 640, "source_stream_done": false, "translation_eos": false}, ...]},
 "spk_post": "post/lowlat/train/IS1006b_zh_0005.npy", "spk_slots": [1, 1, 2]}
```
最后两个字段是 E10 新增的，原 loader 会忽略它们。

**步骤**

1. **译文块和 deadline。** 直接调用 B 的 `build_ami_gstar_stream_rows.py`（Zh→En 用 `_zh2en` 版）里的分块和 deadline 函数（标点切块，源词结束后加 1 s；Zh→En 按比例估计）。只 import，不复制。这样保证标签和 B 逐块相同。
2. **映射到 640 ms 步。** 块 c 落在第一个满足 `source_end_ms ≥ deadline_c` 的步 i。该步 `action=write`，`text_delta` 是这些块按顺序拼接，每块前加 `<|spk k|>`（k 来自 slot_map）。
   - **一步只放一个说话人**：如果同一步到期的块来自两个说话人，较晚开口的那位推迟到下一步。
   - 最后一步无法再推迟时允许混合，推理侧对此有兜底（§5.4）。
   - 推迟和混合各发生多少次，记入数据审计。
3. **结束标记。** `translation_eos` 标在最后一个 W 步；`source_stream_done` 标在最后一个源步；尾部 WAIT 按 `steps_until_thinker_eos` 的约定处理。
4. **目标语音**（Talker 监督，joint scope 必需）
   - 用 Fun-CosyVoice3-0.5B 零样本克隆，逐块合成。这就是 LI-S5 声码器所在的模型族，CosyVoice 仓库现成可用。
   - 参考音：该说话人在本场会议训练集里最长的单独发声片段，≤8 s，筛选规则同 B 的 voice rule。训练标签允许非因果；推理侧仍然严格因果。
   - 再用 LI-S5 自带的 `dataset_maker/streaming_si/tts_stage1_fsq/fsq_encoder.py` 把合成音重新编码成 v3 FSQ 25 Hz。README §4.1 警告旧码本"训错不报错"，所以必须用同一个编码器。
   - 按块顺序拼接，得到 `target_codec_codes`。每个 W 步的 `source_codec_spans` 是它所含块的帧区间，按要求严格连续平铺 [0, N)。
   - 质量门：每块 ASR 回识。WER/CER 超过阈值就换一个种子重合成一次，仍失败则丢弃这个窗口，并计数。阈值沿用 B 的 VoxCPM2 可懂度检查。
   - 规模：约 85 h 源音频，对应约 49.5k 个窗口的目标语音。
5. **长度过滤。** `N > 750`（30 s）或序列超过 1024 的，丢弃并计数，不截断，与 LI-S5 口径一致。
6. **混合比例照搬 B**
   - **单说话人行（15%）**：按 `build_ami_gstar_stream_rows.py` 里 `inst: sd` 行的定义选同样的窗口，生成后验置零的副本（记为 `nosd`）。实现前再读一遍该脚本确认定义。
   - **离线行（20%）不照搬**。理由：S5 的布局要求一步只有一次 W，而离线整窗行在多说话人场景下必然一步多人；LI-S5 底座本身就是离线 S4 模型，S5 也有 m 合并机制可以承担这个角色。这一偏离写进预登记。
7. **val 划分和索引**
   - val 用 AMI 官方 validation 窗口。不用 `build_s4_val_split.py` 的随机抽样：按会议划分，避免泄漏，与 B 一致。
   - 再跑 `build_record_index.py`，生成分层采样需要的 NIR × 句长索引。
   - 布局：`data/ami_multispk_si_v1/li_s5_records_v1/records/{en2zh,zh2en}/chunk_640ms/{train,val}/records/part-*.jsonl` 加 `_manifest.json`；音频根是 `data/ami_multispk_si_v1/`（直接复用已有 wav，不复制）。
8. **审计**（写成 `data_audit.json`）
   - 窗口数、丢弃原因分布；
   - 推迟和混合的次数；
   - W 步占比，对比 LI-S5 原数据：wait_recall 0.96、write 少数类；
   - 文本与 B 的 target 逐窗比对，去掉 spk/t token 后必须一致；
   - 抽 20 窗人工试听目标语音。

### 5.2 模型改动（在 `custom/streaming_si` 外加补丁，不改同事的原文件）

补丁放在 `experiments/s2st_baselines_v1/liveinterpret_multispk/li_ms_patch.py`，通过 swift 的 `--external_plugins` 注入（checkpoint 本身就是这样训出来的，见 `args.json:external_plugins`）。

**(a) 说话人 token**
- 用 `<|dur:n|>` 的前 8 行（id 248080 起，S5 训练和推理都没用到）作为 `<|spk1|>`…`<|spk8|>`。映射写进 `li_ms_tokens.json`。
- 不改 tokenizer 和 embedding 的大小，`s4_stream_token_ids` 的校验也不受影响。
- 把这 8 个 id 加进 `trainable_token_indices`，embedding 和 lm_head 共享，沿用 s4_peft 的绑定硬校验。
- 推理时这 8 个 id 加进 `_pick` 的 `allow`：原来 `ban_above=tokenizer.vocab_size` 会屏蔽它们。
- 训练前用 S5 的 `reinit_stream_token_rows` 同法重置这 8 行。

**(b) 后验注入 `sd_proj`**
- 第 t 个 audio_pad 帧（时刻 (t+1)·80 ms）只能用 chunk 协议下已经发布的后验帧：f* = max{f : 9⌊f/9⌋+13 ≤ t+1}。
- 注入方式：`e_t ← e_t + W_sd · p_{f*}`；还没有可用帧时加零。
- 这与 B 的"lowlat，1.04 s 后可用"同口径，并直接采用 2026-11-12 起的 chunk 协议。
- `W_sd` 用零初始化，训练开始时模型与 checkpoint-29773 逐位等价。lr 单独设为 1e-4，与 B 的 `sd_proj` 相同。
- 训练时的扰动照搬 B：p=0.2，σ=0.1，平移 ≤1 帧。
- 实现上挂在 `inject_audio_features` 之后，训练和推理共用一个函数。

**(c) Talker 条件里排除 spk token**
- 训练时 `answer_mask` 不标 spk token 的位置；推理时 `gather_answer_hidden` 同样跳过。
- 这样 Talker 看到的条件序列和原来"纯译文 token"同分布。

**(d) checkpoint 契约**
- `s4_metadata.json` 增加 `li_ms: {spk_token_ids, sd_proj: true}`。
- `sd_proj` 作为 `modules_to_save` 保存在 adapter 里。
- 加载时校验 id。

**(e) 单测**（CPU，用 LI-S5 自带的 tiny fixture，`make_smoke_fixture.py --only model_4b`）
- `W_sd = 0` 时，输出 logits 与无补丁版本逐位相同；
- spk token 能被采到；
- answer_mask 跳过了 spk token；
- 帧对齐检查：`sum(audio_seg_lens)` 等于后验可用帧的数量。

### 5.3 训练配置（只用一组设置，不扫超参）

| 项 | 值 | 理由 |
|---|---|---|
| 入口 | `pipelines/streaming_si/train_simul.sh`，加 `--adapters <checkpoint-29773> --external_plugins <li_ms_patch.py>` | 从 0 步开始，只加载 adapter，与同事的续训方式相同 |
| MODEL_PATH | S4 底座 `S4_1031v2_ckpt93767` | adapter 的 `base_model_name_or_path` |
| scope | `SI_S4_TRAIN_SCOPE=joint`；损失权重 text 1 / speech 1 / decision 1 | Talker 的条件就是 Thinker hidden，只训 Thinker 会让 Talker 看到的分布漂移（§9-c） |
| LoRA | r64、α128、dropout 0，目标正则与 checkpoint 相同 | 与 adapter 结构一致才能加载 |
| lr | 1.2e-5，cosine 退到 2e-6，warmup 3%；`sd_proj` 和 8 个 spk 行 1e-4 | 前者同 checkpoint；后者同 B 的新模块 |
| batch | 8×H200，每卡 8，GBS 64 | 同 B 的 GBS |
| 时长 | 2 epoch，约 49.5k 窗 × 2 / 64 ≈ 1,550 步 | B 共看了 6000×64 = 38.4 万个"读行"；按每窗约 4–6 次读折算，相当于 6.4–9.6 万个窗口，和这里同一量级 |
| 采样 | `SI_S4_STRATIFIED=1`、`LATENCY_MULTIPLIER=1`、`SI_S4_SEED=20261026`（同 B） | S5 默认 |
| 选 checkpoint | **取最后一步，不做选择** | 同 B 和 E9 |
| val | 每 250 步算一次 `dec_macro_f1` 和 text/speech loss，只用于监控 | 不参与选择 |

**训练时间估算**：S5 首训在 24 卡上约 865 条/s。按 8 卡和 r64 折算，2 epoch 不到 1 小时。整个流程的瓶颈在 §5.1 第 4 步的 TTS 数据生成。

**GPU 规则**：训练占满 8 卡。按 memory，训练卡上不叠别的训练；数据生成和 dev 解码可以在训练开始前后穿插。

### 5.4 推理与评测适配（与 E7/E8/E9 相同的接口）

新增 3 个脚本，放在 `experiments/s2st_baselines_v1/liveinterpret_multispk/`：

**`run_e10.py`**（GPU，每卡可以叠 3–4 个进程，每个进程约 20 GB）
- 每个在线窗口是一个 `S5Session`：`segmentation=none`，`input_pacing=accelerated`，`chunk_ms=640`，`write_threshold=None`（argmax），`retry=0`，talker top-k 25 / top-p 0.8，ode_steps 5。
- 后验按 §5.2b 的协议在线注入，读取窗口的 `post.lowlat`。
- **按说话人分路发声**：把每步增量按 spk token 切成子段，每个子段单独调用一次 `talker.stream_generate_s4`；Talker 状态连续，等价于连续两个 W 步。speech token 记到该说话人的流上。
- **每个说话人的参考音**：
  - 规则同 B 的 voice rule：该 slot 单独发声（P>0.5 且其他 slot <0.5，连续 ≥0.4 s）；只用首次提交之前、已经发布的音频；最多 8 s，flow prompt 取最后 2 s。
  - 没有可用片段时，回退到"窗口开头到首次提交之间的音频"，同 E9 的 `unk` 回退。
  - flow 按 (窗口 × 说话人) 分别解码。
- 原始输出：每窗一行 `part_*.jsonl`，`segs: [{text, spk, text_ms(=该步 source_end), write_ms, dur_ms, wav}]`；没说出来的尾部记 `unspoken`。
- 两个 TTS 种子：0 和 20261109。Thinker 是贪心解码，不读 speech token，所以两个种子只在语音上不同。

**`build_e10.py`**（CPU）：复制 `build_e8.py`，只改段规则。输出：
- `decode/<name>/hyp_li.jsonl`，格式 `{id, segments:[{spk, start_s, end_s, text, write_ms}]}`；
- `speech/<name>/talker/results.jsonl`，格式 `blocks:[{write_ms, start_ms=max(write, 本路上一块结束), dur_ms, text}]`，含 `speech_path`、`audio_path`；
- `_1spk` 下界变体；
- 在线钳位：提交时刻 ≥ `open_detect_ms`；读完整窗后写出的段 ≥ `close_detect_ms`。
- 说话人：用模型自己输出的 spk，所以 Table 1 的"own speaker attribution"列标 ✓。另外用 `relabel_causal` 生成 `_rel` 变体，只进附录，用来区分"说话人通道带来的增益"和"重标带来的增益"。

**打分**（全部原样复用）：
`asr_merge → asr_hyp → score_run --asr`（cp/tcpASR-BLEU）
→ `table1_lat.sh <set> <run>`（chunk 协议下的 LAAL_block）
→ `table1_analysis --extra "li=LI-S5 (E10):ami=…;ali=…;en=…"`（配对检验 + Holm）。

**测试集**：AMI Zh→En（`WT_AMI`）、AliMeeting（`WT_ALI`，DiariST 人工英文参考）、AMI En→Zh（`WT_EN`），每个集合只解码一次。

**计算量估算**：三个测试集合计约 18 h 源音频 × 2 个种子 × RTF 约 0.45 ≈ 16 GPU·h，8 卡叠跑约 2–3 h。dev 冒烟加 dev 全量约 1 h。

### 5.5 dev 检查（不选 checkpoint，只做健全性门槛）

在 c2g06 dev（`WZ_DEV` / `WE_DEV`）上：
- 塌缩率（`collapse_ok`）≤ 2%；
- spk token 输出率 ≥ 95%；
- 和 E10-0 比，tcpASR-BLEU 不下降。

任一项不达标，**停下来汇报**，不调参重训（no-hparam-sweeps 规则）。

---

## 6. E10-0：零样本外部系统式（可行性冒烟兼附录对照）

- 不训练，直接用 checkpoint-29773：滚动参考音，没有 spk 输出，没有后验。在每个在线窗口上跑 `infer_s5_streaming_dynchunk.py --synthesis_mode streaming --segmentation none`。
- 说话人和源区间完全照 E7/E8：用 `relabel_causal` 重标，源区间取 [上一段写出时刻, 本段写出时刻]。own attribution 标 "voice only" 或 ×。
- 先做 3 个 dev 窗口冒烟，并且 ≤1 小时内完成可行性判定。这同时验证环境、底座加载、声码器，以及 `build_e10.py` 的格式能被打分器读取。

---

## 7. 预登记草稿要点（正式文件看结果前另写）

**文件**：`reports/mist_w3po_v1/e10_liveinterpret_multispk_prereg.zh-CN.md`。

**内容**
- 系统定义：E10，以及（可选的）E10-0。
- 数据：§5.1，包括"不混离线行"这一偏离。
- 训练：§5.3，取最后一步。
- 解码参数：§5.4，640 ms、argmax、两个种子。
- 指标：cp/tcpASR-BLEU、chunk 协议下的 LAAL_block。EndOffset 只进附录。
- Holm 族：E10 并入 Table 1 现有的族。
- 测试集只解码一次。

**规则**
- 修订必须带时间戳，并且写在受影响的结果出来之前。
- MIST 不因 E10 的结果做任何改动。
- 每个决定都记进 `reports/mist_w3po_v1/LOG.zh-CN.md`。

---

## 8. 时间线（布拉格时间，按 8×H200 估算）

| 阶段 | 内容 | 时长 | 资源 |
|---|---|---|---|
| T0 | 新建 venv（ms-swift 4.4.0 / transformers 4.57.3 / peft 0.15.2 / CosyVoice 依赖 / silero-vad / onnxruntime），下载底座（13 GB）和声码器（1.4 GB） | 1–2 h | CPU/网络 |
| T1 | E10-0 冒烟（3 个 dev 窗口）和 `build_e10.py` 的格式验证 | 1 h | 1 卡 |
| T2 | `build_li_records.py`：块、deadline、步、spk、审计（不含 TTS） | 0.5 天 | CPU |
| T3 | 目标语音：CosyVoice3 克隆、FSQ 编码、ASR 质检，约 49.5k 窗 | 4–8 h | 8 卡（无 checkpoint 的负载，可随时按 PID 停） |
| T4 | `li_ms_patch.py` 加单测，再用 tiny 模型跑 3 步冒烟 | 0.5 天 | CPU 和 1 卡 |
| T5 | E10 训练，约 1,550 步 | ≤1 h | 8 卡 |
| T6 | dev 门槛（§5.5），然后写正式预登记 | 1 h | 8 卡 |
| T7 | 3 个测试集 × 2 个种子，ASR，打分，Table 1 合表 | 3–4 h | 8 卡 |

合计约 2 个工作日，主要的不确定性在 T3。

---

## 9. 风险与需要你决定的事项

a. **新 venv。** 按 memory，只能用你允许的 venv。LI-S5 需要 ms-swift 4.4.0 和 transformers 4.57.3，和现有的 uniss-train、vllm_omni_030 都不兼容。建议新建 `venvs/liveinterpret_s5`。请确认。

b. **数据重叠（必须向同事确认）。** checkpoint 的训练数据（p4mix、enzhjako 等内部数据）里有没有 AMI、AliMeeting 或 DiariST 的音频或文本？按 realsi-acl-test-only 规则，测试集必须没有被训练见过。如果有重叠，AliMeeting 这一列要加脚注或者剔除。

c. **joint 还是 thinker-only。**
- 推荐 joint：Talker 的条件就是 Thinker hidden，Thinker 的 LoRA 一变，Talker 看到的分布就会漂移；而且 S5 本来就是这样联合训练的。
- 代价是要做 T3 的目标语音生成。
- 如果你认为应该严格照 B（Talker 不在多说话人数据上训练），可以改用 `SI_S4_TRAIN_SCOPE=thinker`、`SI_SPEECH_LOSS_WEIGHT=0`，T3 就可以省掉。但 records 仍然需要 `target_codec_codes` 来通过 span 校验，那就要给 loader 打一个 text-only 补丁，并且在 dev 上单独检查塌缩率。

d. **E10-0 是否进 Table 1。** 建议只放附录：它不在同一份数据上训练，也没有自己的说话人归属。主表的 LI-S5 行用 E10。

e. **读写粒度。** 推荐 640 ms，这是 LI-S5 训练的原生粒度。B 是 1 s 读。同事的公开报告用的是 m3（1920 ms）加 `write_threshold`。但按 BLEU-before-LAAL 和"一种设置"的规则，不在测试集上挑粒度。

f. **开源数据规则。** E10 的微调只用 AMI（CC BY 4.0）和 CosyVoice3 合成的语音。checkpoint 本身的预训练数据不受我们控制，在论文里按"external pretrained model, fine-tuned on the same open data"披露。

---

## 10. 附：关键文件索引

- LI-S5 推理：`custom/streaming_si/inference/{infer_s5_streaming_dynchunk.py, s5_session.py, s5_segmenter.py, infer_s2st.py}`
- LI-S5 数据和训练：`custom/streaming_si/data/{s4_streaming_dataset.py, s4_pipeline.py, si_traj_ops.py}`、`pipelines/streaming_si/{train_simul.sh, common_args.sh, README.md}`、`dataset_maker/streaming_si/{build_s4_val_split.py, build_record_index.py, tts_stage1_fsq/fsq_encoder.py}`、`custom/streaming_si/scripts/make_smoke_fixture.py`（record 字段集）
- B 的数据：`experiments/uniss_qwen3_omni_v1/data_clean/build_ami_gstar_stream_rows{,_zh2en}.py`、`data/ami_multispk_si_v1/{e2e_gstar_v1, e2e_gstar_zh2en_v1, e2e_gstar_stream_bidir_v2}`
- B 的训练：`experiments/mist_w3po_v1/scripts/joint_stage1.sh`、`experiments/uniss_qwen3_omni_v1/training/pretrain_thinker_gstar.py`
- 评测：`experiments/mist_w3po_v1/scripts/{joint_lib.sh, joint_test.sh, table1_lat.sh}`、`experiments.mist_w3po_v1.evaluation.table1_analysis`
- 外部系统适配范例：`experiments/s2st_baselines_v1/{seamless_multispk/build_e7.py, xtranslator_multispk/build_e8.py}`、`reports/mist_w3po_v1/{e7_seamless_multispk_prereg, e9_cascade_multispk_results, multispk_s2st_baselines_prereg_v2}.zh-CN.md`
