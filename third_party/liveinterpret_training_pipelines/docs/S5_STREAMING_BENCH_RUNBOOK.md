# S5 实际 PCM 跑测操作

本入口属于自研 S5，不属于 Repro-Omni。当前实现和待办见 [优化说明](S5_STREAMING_INFERENCE_OPTIMIZATION.md)。命令本身不代表验收通过。

修订前版本已完成的L40S质量、RTF与1倍速首包对照见 [完整报告](S5_L40S_CHUNK_BENCH_REPORT.md)（[HTML版](S5_L40S_CHUNK_BENCH_REPORT.html)）；原始音频、日志和集群监控工具保留在本地。该报告不能替代本轮640/1280/1920/2560与滚动源参考的验收。

本轮声学滑窗修订版的四个640倍数chunk、换人对照与长流失败记录见 [修订版报告](S5_ACOUSTIC_WINDOW_V2_REPORT.md)。

2026-11-17当前无保护候选（独立测试接口，未替换原线上）：在下方streaming文件命令中使用
`--context_window_sec 10 --segmentation none`；不创建VAD或30秒时长分段器。
当前候选和上一阶段结果见优化说明最新章节。

上一阶段的保护配置：在streaming文件命令中使用
`--context_window_sec 10 --segmentation interval --hard_sec 30`，关闭VAD并保留独立30秒保护；
`--context_window_sec 0`是同一时长保护下不跨段保留语义的对照。
这不是每10秒清空模型：过期时只保留近期完整交互并重放，不重复输出。

可用 `--context_window_32_16` 替代 `--context_window_sec 10`，CLI 的流式模式与 Web 后端均支持。每个 chunk 的 WAIT 或 WRITE 决策算一轮，累计 32 轮后重放最近 16 轮。Talker 在下次生成前按 Thinker 保留历史的起始轮裁剪。两种策略都为新输入及生成预留缓存，容量不足时可以提前裁剪。做窗口对照时使用固定 640ms、`--segmentation none` 和同一份长音频。
窗口、容量和重放限制见优化说明最新章节；验收通过前不要把该配置当作已完成长流或质量验证。

## 固定资产与协议

- S4 底座：`S4_1031v2_ckpt93767`，对应报告中的13.9GB权重。
- S5 adapter：`s5_enzhjako_off_mchunk_ft_lora_r64_20261106/v0-20261106-061157/checkpoint-10000`。
- 源清单：1193句，RealSI en→zh346、zh→en431、ACL en→zh416；使用新入口时不添加伪造预热句到质量样本。
- 正式质量沿用原报告库：`custom/streaming_si/eval/report_protocol/run.py`及同目录`README.md`，中文Qwen3-ASR、英文Whisper、OpenCC t2s、原COMET。原文件从Git恢复并校验SHA。原先另一套Whisper双向诊断协议已移到仓库外，不作为正式入口。目标ja/ko同样用Qwen3-ASR，并分别使用ja-mecab/ko-mecab；不允许字符tokenizer静默回退。

## 直接启动文件推理与Web

在已安装S5模型依赖、能访问CUDA和CosyVoice3的环境运行。以下路径均需替换成实际本地资产；`BASE`为S4底座，`ADAPTER`为S5 adapter，`MANIFEST`每行至少包含`id/audio/tgt_lang`。

```bash
export PYTHONPATH="$PWD/custom"
python -m streaming_si.inference.infer_s5_streaming_dynchunk \
  --model_dir "$BASE" --adapter "$ADAPTER" --input_jsonl "$MANIFEST" \
  --out_dir "$NEW_OUTPUT_DIR" --event_jsonl "$NEW_EVENTS_JSONL" \
  --synthesis_mode streaming --segmentation none \
  --chunk_ms 640 --retry 0 --max_audio_sec 0 --ode_steps 5 \
  --cosyvoice_repo "$COSYVOICE_REPO" --cosyvoice_dir "$DECODER" \
  --prompt_tokenizer "$PROMPT_TOKENIZER" --prompt_sec 2
```

两阶段chunk增加`--chunk_ms_after_write 1920`并设置首段`--chunk_ms 640`；固定chunk则省略after参数。CLI和Web主选项统一为640/1280/1920/2560ms。默认使用`--reference_policy rolling --reference_refresh_sec 2`，`--stream_prompt_sec 2`（Web为`--prompt_sec 2`）控制源参考长度，`--reference_window_sec 5`控制源音频缓冲上限；`--reference_policy fixed`用于冻结初始参考的A/B。`--input_pacing realtime`可用同一CLI按1倍速推进文件；它在推理线程内按时钟等待，不能替代独立输入生产者/真实麦克风的积压与网络测试。

```bash
python -m pip install -r custom/streaming_si/inference/requirements-web-s5.txt
python -m streaming_si.inference.web_demo_s5 \
  --model_dir "$BASE" --adapter "$ADAPTER" \
  --cosyvoice_repo "$COSYVOICE_REPO" --cosyvoice_dir "$DECODER" \
  --prompt_tokenizer "$PROMPT_TOKENIZER" \
  --segmentation none --chunk_ms 640 \
  --host 127.0.0.1 --port 7865
```

页面为`/web_demo_s5.html`，上传与麦克风共用同一会话实现。浏览器麦克风要求localhost或HTTPS；远程访问请配置自己的代理与TLS。每GPU进程当前只允许一个活跃会话。

## 短样本和全量

先运行3句每数据集的真实GPU冒烟，确认有有效PCM、窗口状态无异常，再运行40句。下列工具自动按数据集均衡抽40句，并覆盖各组源时长的短端和长端；不要把前40行当成双向覆盖。

```bash
export PYTHONPATH="$PWD/custom"
python -m streaming_si.eval.s5_streaming_bench run \
  --manifest /path/all.jsonl --model_dir /path/S4_1031v2_ckpt93767 \
  --adapter /path/checkpoint-10000 --out /work/s5-probe \
  --stage probe --pacing accelerated --repeats 2 \
  -- --cosyvoice_repo /opt/CosyVoice --cosyvoice_dir /path/speech_decoder \
  --prompt_tokenizer /path/speech_tokenizer_v3.onnx
```

默认遍历640/1280/1920/2560ms固定档；可用`--configuration 640`先只跑一档。输出目录必须全新，避免旧events混入。

CLI/Web默认不启用VAD断段重置，共用持续S5Session。`--segmentation vad`仅显式启用旧版分段/清状态诊断，不作为源声学滑窗的默认策略。

检查`actual_pcm_report.json`和音频后，使用同一命令改为`--stage full --configuration 640 --probe_report /work/s5-probe/640-r1/actual_pcm_report.json`与新的输出目录，清单必须恰好1193句。`--pacing realtime`单列实验，不与加速输入墙钟混合。

质量评测通过`--quality_commands /path/quality.json`显式串接已有ASR、UTMOS、para、`score_eval.py`等工具，JSON为argv数组的数组，允许`{run_dir}`、`{manifest}`占位。工具不自动猜测ASR模型和评分资产；缺少配置会写`quality_status.json: ran=false`，不能宣称质量通过。COMET评分按目标语言拆分manifest；不要将中英混合manifest直接传给`score_eval.py`。评分计时与推理计时分开。

## 长流与暂停

`longrun`默认`--segmentation none`，沿用持续语义路径与滚动声学参考。`--segmentation vad`只用于旧版断段/重置诊断；不能用它每30秒清状态的结果证明连续语义流已稳定。未分段历史上曾出现Talker重复/拉长，修订后必须重测并显式记录失败。

长流工具使用同一`S5Session`，先在独立会话上预热fixture（不写PCM/事件/评分，单列warmup耗时），再对测量会话构造后重设seed。重复短音频只用于压力/状态测试，不代表长文翻译质量。输入不会整段拼成40分钟波形，也不允许30秒裁剪。

```bash
python -m streaming_si.eval.s5_streaming_bench longrun \
  --audio /path/source.wav --tgt_lang zh \
  --model_dir /path/S4_1031v2_ckpt93767 --adapter /path/checkpoint-10000 \
  --cosyvoice_repo /opt/CosyVoice --cosyvoice_dir /path/speech_decoder \
  --prompt_tokenizer /path/speech_tokenizer_v3.onnx \
  --segmentation none --reference_policy rolling --duration_sec 2400 --out /work/s5-long40
```

- 10分钟连续：`--duration_sec 600`；40分钟：`2400`。
- 10分钟后无包暂停30/60秒再继续10分钟：`--duration_sec 1200 --pause_kind no_packets --pause_at_sec 600 --pause_sec 30`（或60）。无包暂停实际等待墙钟，不推进源位置、不重置会话。
- 静音对照：把`no_packets`改为`silence`；此时实际喂零值音频，源位置正常推进。`duration_sec`是有声fixture总时长，静音额外加入源时长。
- `--pacing realtime`采用1倍速输入；加速文件模式与1倍速结果必须分开。
- 默认目标声学窗口60秒；可用较短`--acoustic_max_sec`强制更多切换，但应满足合成hop/lookahead要求。Thinker容量用`--context_max_tokens`控制，必须为单WRITE留足空间，不可盲目减到很小。

`events.jsonl`记录真实PCM墙钟、样本offset、源消费位置；step/context/end还记录GPUallocated/reserved显存。`output.wav`由实际输出PCM顺序写入，不重建等待静音。

## Web 实际播放验证

安装`custom/streaming_si/inference/requirements-web-s5.txt`后，通过`python -m streaming_si.inference.web_demo_s5`传入相同的`--model_dir/--adapter/--cosyvoice_repo/--cosyvoice_dir/--prompt_tokenizer`启动。默认单GPU、单Gunicorn worker，模型在worker内加载，HTTP线程共用模型锁；不要打开preload或增加worker复制模型。`--server development`仅用于诊断。

Web默认`--segmentation none`，支持上传文件加速/1倍速及麦克风。浏览器使用localhost或HTTPS；远端服务可通过本地端口转发访问。麦克风完成指标到达后由客户端以正常关闭确认收尾。测试`check_s5_web_browser.cjs ... zh microphone`使用真实文件模拟Chromium采集设备，仍不等同于人工麦克风、听感或生产网络验收。

## 计时口径

`first_pcm_wall_sec`为服务端实际产出，`pcm_interarrival_sec`为输出间隔；`source_step_compute_sec`是源块处理耗时，三者不能互换。`server_playback_proxy_gaps`仅按服务端到达和PCM时长估计播放断粮，不包含网络/浏览器，不叫真实用户断流；实际浏览器播放请使用Web demo的客户端指标。

当前`compute_sec/inference_rtf`计时包围整个feed，含同步PCM/事件回调，不能称为严格排除I/O的纯模型或GPU kernel计时；需要该口径时另行插桩。CLI的summary另报inference RTF、扣除主动输入等待的warm I/O RTF、含加载及输入节奏的cold总墙钟；实时模式的cold RTF不能解读成吞吐能力。质量未通过、缺失PCM或计时模式不同的结果不得合并为最终优化结论。最终L40S至少复跑两次；A10功能结果单独标注，不换算成L40S实测。


## Web demo

后端 `custom/streaming_si/inference/web_demo_s5.py`，页面为同目录 `web_demo_s5.html`，保留原附件页面布局；目标语言en/zh/ja/ko由服务端配置提供。页面可选首段chunk及首WRITE后chunk，服务端再次校验参数。线上服务可能运行冻结的历史快照；以部署时记录的版本为准，页面能打开不代表已经切换到本轮候选版。

Web仅保留当前真流式。旧网页A/B的legacy桥接已归档；历史伪流式对照仍可用原dynchunk CLI的 `--synthesis_mode offline` 复现。每GPU进程目前一次仅一个活跃会话，多人调度尚未实施。浏览器可用上传及麦克风；浏览器时间、服务端PCM产出墙钟、源消费位置分别报告。

## 局部profile

使用 `custom/streaming_si/eval/profile_s5_streaming.py`，仅CUDA。先完整独立会话预热并丢弃输出，再在选定源step打开profiler。示例以真实资产路径替换变量：

```bash
PYTHONPATH=custom CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m streaming_si.eval.profile_s5_streaming \
  --model_dir "$BASE" --adapter "$ADAPTER" --audio "$AUDIO" --tgt_lang zh \
  --cosyvoice_repo "$COSYVOICE_REPO" --cosyvoice_dir "$DECODER" \
  --prompt_tokenizer "$PROMPT_TOKENIZER" --segmentation none \
  --chunk_ms 640 --seed 0 --profiler torch --start_step 4 --steps 6 --out "$OUT"
```

nsys使用相同命令改成 `--profiler nsys`，外层加 `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output "$NSYS_PREFIX"`。OUT应尚不存在，NSYS_PREFIX父目录应存在。fixture必须含WRITE，才能覆盖Talker/Flow/HiFT。

scope只累计CPU user_annotation；嵌套范围不能相加。kernel归属需通过runtime correlation关联发射范围。普通路径与强制窗口路径分开报告，带profiler时间不能作为RTF。

## 回归工具与归档

- `test_script/streaming_si/check_s5_chunks_gpu.py`：真实权重小chunk、尾块及PCM记账。
- `test_script/streaming_si/check_voxtral_bounded_cache_cuda.py`：裁剪与原滑窗可见性对拍。
- `test_script/streaming_si/check_s5_web_browser.cjs`：浏览器上传/模拟麦克风集成。
- `tests/test_s5_session.py`、`test_script/streaming_si/test_s5_pcm_decoder.py`、`custom/streaming_si/eval/tests/test_s5_streaming_bench.py`、`test_script/demo/test_web_demo_s5.py`：状态/协议回归，不能替代真实模型质量验证。

集群监控、任务进度、下载凭据和本次实验产物由本地工作目录管理，不提交到推理分支。
