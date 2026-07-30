# UniSS R2 Streaming S2ST 在线试听与麦克风同传 Web Demo 实施计划

> 计划日期：2026-07-30
> 新目录：`web_demo/streaming_s2st_r2_v1/`
> 主模型：Stage7A Reward-v2 `R2 explicit-latency`，step 300
> 回退模型：Stage7A Reward-v2 `R3 bilingual + adaptive KL`，step 900
> 公网要求：默认使用 Gradio `share=True` 自动生成无需注册、无需登录的 HTTPS 公网 URL
> 本文只制定实施计划；当前尚未创建服务代码、启动新服务或占用额外 GPU。

## 1. 目标

在不修改、不覆盖现有 `web_demo/web_demo.py` 和
`web_demo/offline_s2st_phase3_v1/` 的前提下，新建一个独立的
simultaneous/streaming speech-to-speech 网站，支持两种主要交互。

### 1.1 上传音频同步试听

用户上传中文或英文音频后，页面左右分栏：

- 左侧以正常 1× 速度播放源音频；
- 右侧按照模型实际 WAIT/WRITE 时间线播放翻译语音；
- 页面同步显示源音频时间轴、WAIT/WRITE、增量翻译文本、目标语音 chunk；
- 支持暂停、重新播放、拖动时间轴、单独下载两路音频；
- 支持导出“源音频左声道、同传音频右声道”的对齐立体声 WAV，便于直接戴耳机比较。

### 1.2 麦克风边录边译

用户点击开始后：

- 浏览器持续采集麦克风 PCM；
- 服务端按 chunk 处理不断增长的源语音；
- 模型执行 WAIT/WRITE；
- WRITE 后立即生成增量翻译文字和 BiCodec semantic token；
- 翻译语音 chunk 到达浏览器后立即进入播放队列；
- 用户停止录音或 VAD 判定结束时执行 final flush；
- 页面显示实时状态、累计延迟、输出 chunk 数和错误恢复信息。

## 2. 必须如实标注的技术边界

当前 R2/R3 正式训练和评估使用预计算 `source_glm`，不是运行时原始波形
增量 Whisper encoder。当前已验证的是：

```text
预计算 source_glm chunk
→ R2/R3 WAIT/WRITE
→ 增量目标文本/semantic token
→ Streaming BiCodec waveform
```

因此网站必须区分三种模式，不能都标成“真正端到端流式”。

| 模式 | 源端处理 | 定位 | 默认用途 |
|---|---|---|---|
| Evaluation-compatible replay | 完整音频先生成 GLM/BiCodec token，再按 640 ms 时间线回放 | 与现有评估最接近的伪流式 | 上传试听默认模式，效果最稳定 |
| Live prefix re-encode | 每 640 ms 对累计音频重新运行 WhisperVQ，稳定前缀才提交 | 在线 Whisper 伪流式 | 麦克风和上传实时实验模式 |
| True causal frontend | causal/chunk-causal audio student + encoder cache | 真正原始音频流式 | 后续阶段，本次 v1 不冒充已完成 |

页面必须持续显示当前模式，例如：

```text
当前模式：Online pseudo-streaming
源前端：WhisperVQ cumulative-prefix re-encoding
目标生成：R2 WAIT/WRITE + Streaming BiCodec
```

禁止把上传音频完整预编码后再切 token 的模式写成“真实麦克风端到端延迟”。

## 3. 模型选择

### 3.1 默认主模型：R2 explicit-latency

路径：

```text
checkpoints/exported_hf/
  simul_uniss_stage7a_reward_v2_15shard_v1/
  r2_explicit_latency_best_hf/
```

冻结信息：

```text
action checkpoint step = 300
model = Qwen2ForCausalLM, 0.5B
weights ≈ 1.3 GB
max model context = 32768
training context boundary = 18000
write logit bias = 0.0
decode = greedy
repetition penalty = 1.1
```

选择原因：

- Reward-v2 中 ZH→EN Speech-BLEU 最高：test `17.1067`；
- EN→ZH Text-BLEU 最高：test `40.623`；
- dev First WRITE 最低：`4111.2 ms`；
- dev ATD 最低：`1861.4 ms`；
- 是当前质量与策略延迟综合最优点。

### 3.2 回退模型：R3 bilingual + adaptive KL

路径：

```text
checkpoints/exported_hf/
  simul_uniss_stage7a_reward_v2_15shard_v1/
  r3_bilingual_adaptive_best_hf/
```

R3 已完成完整 test、AutoPCP 和独立 batch-one 延迟审计：

```text
First WRITE NCA mean = 4697.6 ms
StartOffset CA mean = 5293.4 ms
StartOffset CA p95 = 8826.6 ms
source-audio RTF mean = 0.1647
```

R2 在正式公开前必须先通过相同的单样本真实音频 smoke。若 R2 出现结构恢复、
codec 连续性或在线前缀适配回归，服务自动回退到 R3，并在 UI 中明确显示实际模型。

## 4. 固定推理参数

初始版本严格复用已验证参数：

```text
source chunk = 640 ms
max write tokens = 700
max model len = 32768
training context warning = 18000
repetition penalty = 1.1
action decode = greedy
write decode = greedy
BiCodec left context = 50 semantic tokens
BiCodec holdback = 5 semantic tokens
BiCodec overlap = 80 ms
sample rate = 16000 Hz
```

禁止为了让网页“看起来更快”而静默改变 WAIT/WRITE bias、chunk size、holdback 或
sampling。后续任何 operating point 调整必须生成新的配置名和独立验证报告。

## 5. 独立目录设计

实施后目录规划如下：

```text
web_demo/streaming_s2st_r2_v1/
├── streaming_s2st_web_demo_implementation_plan.md
├── README.md
├── VALIDATION.md
├── __init__.py
├── config.py
├── app_gradio.py
├── audio_io.py
├── session_manager.py
├── engine/
│   ├── __init__.py
│   ├── model_runtime.py
│   ├── qwen_live_adapter.py
│   ├── upload_replay.py
│   ├── prefix_frontend.py
│   ├── streaming_pipeline.py
│   └── codec_runtime.py
├── frontend/                 # Gradio 中注入的可选 HTML/CSS/JS
│   ├── styles.css
│   ├── audio_sync.js
│   └── timeline.js
├── scripts/
│   ├── setup_environment.sh
│   ├── run_local.sh
│   ├── launch_public_tmux.sh
│   ├── share_watchdog.sh
│   ├── status.sh
│   └── stop.sh
├── tests/
│   ├── test_audio_io.py
│   ├── test_stable_prefix.py
│   ├── test_live_adapter.py
│   ├── test_codec_streaming.py
│   ├── test_session_protocol.py
│   └── test_upload_replay.py
├── runtime_outputs/          # gitignored, TTL 清理
├── runtime_logs/             # gitignored
├── public_url.txt            # gitignored
└── access_info.json          # gitignored, mode 0600
```

现有以下路径只读复用，禁止改动：

```text
web_demo/offline_s2st_phase3_v1/
web_demo/web_demo.py
evaluation/simultaneous_streaming/
uniss/streaming/
checkpoints/
eval_outputs/
```

如果需要抽取共用逻辑，优先在新目录写适配器调用已有函数；不移动、不重命名旧文件。

## 6. Web 技术选型

### 6.1 默认 v1：Gradio 5.49.1

v1 默认使用 Gradio 实现页面、流式回调和公网分享。它符合“不注册账号、访问者不登录、
启动后自动得到公网地址”的要求，启动方式固定为：

```python
demo.queue(default_concurrency_limit=1, max_size=4).launch(
    share=True,
    auth=None,
)
```

启动成功后终端会自动返回类似下面的临时 HTTPS 地址：

```text
https://<random>.gradio.live
```

无需注册 Gradio/Hugging Face 账号，也不要求网页访问者输入用户名或密码。链接是完全公开的，
因此必须同时限制并发、队列长度、音频大小、音频时长和请求频率。

上传页使用：

- 左侧 `gr.Audio` 显示和播放源音频；
- 右侧 `gr.Audio(streaming=True, autoplay=True)` 接收生成器持续 yield 的目标 PCM；
- `gr.HTML`/自定义 JavaScript 展示 WAIT/WRITE 时间线并尽量同步双播放器；
- 推理完成后另外提供完整目标 WAV、aligned stereo 和 JSON 下载。

麦克风页使用：

```python
microphone = gr.Audio(
    sources=["microphone"],
    streaming=True,
    type="numpy",
)
```

通过 `.stream(...)` 回调接收增量音频，以 `gr.State` 保存每个浏览器会话的 controller、稳定
前缀、目标 token 和 codec 状态；生成器持续输出目标 PCM chunk、增量文本、WAIT/WRITE 和延迟。
浏览器通常禁止未经过用户操作的自动播放，所以第一次开始录音/翻译时必须由用户点击按钮解锁
AudioContext；这不是服务端错误。

### 6.2 公网访问与地址生命周期

公网访问是硬性功能。`launch_public_tmux.sh` 必须：

1. 在独立 tmux 会话中运行 `app_gradio.py`；
2. 以 `share=True, auth=None` 启动；
3. 从启动日志解析实际 `https://*.gradio.live` 地址并原子写入 `public_url.txt`；
4. 将 URL、`auth_mode: public_no_login`、模型、模式和启动时间写入权限为0600的
   `access_info.json`，文件中不得保存用户名或密码；
5. `status.sh` 检查进程、URL 和 HTTP 可用性；
6. `share_watchdog.sh` 在进程或 share link 失效时重启应用并更新地址。

Gradio share 地址是临时地址，服务重启后通常会改变，也可能受 Gradio share 服务期限影响；
它不能被承诺为永久固定网址。若以后需要固定域名，再使用 FastAPI/WebSocket + Cloudflare
named tunnel 或自有域名部署。

### 6.3 备用：FastAPI/WebSocket + Cloudflare Tunnel

只有在实测发现 Gradio 对目标音频 chunk 缓冲过多、双播放器时钟无法满足要求、麦克风长会话
重连不可靠，或者需要固定公网域名时，才启用备用实现：

- FastAPI/Uvicorn 提供 HTTP 和会话 API；
- 原生 WebSocket 传输麦克风 PCM、WAIT/WRITE、增量文本和目标 PCM；
- AudioWorklet/Web Audio API 精确调度左右播放器；
- Cloudflare quick/named tunnel 提供 HTTPS 公网访问。

备用方案仍保持无登录公开访问，但沿用同样的并发、时长、速率和队列限制。v1 验收优先验证
Gradio；本计划第11节协议仅作为这一备用后端的接口约定。

## 7. 后端核心架构

```text
Browser / Gradio
  ├─ gr.Audio source player / streaming microphone
  ├─ gr.Audio(streaming=True) translated output
  └─ gr.HTML WAIT/WRITE + text timeline
          │ Gradio queue + stream callback
          ▼
SessionManager
  ├─ AudioIngress
  ├─ SourceFrontend
  ├─ LiveQwenAdapter(R2/R3)
  ├─ StreamingController
  ├─ StreamingBiCodecDecoder
  └─ ArtifactWriter
          │
          ▼
GPU Runtime
  ├─ WhisperVQ / UniSSTokenizer
  ├─ vLLM Qwen R2 or R3
  └─ BiCodec
```

### 7.1 SessionManager

每个浏览器会话独立保存：

```text
session_id
mode
direction
audio buffer
candidate GLM history
committed GLM tokens
speaker tokens
Qwen prompt IDs
generated text IDs
semantic history
BiCodec emitted sample count
WAIT/WRITE event trace
source/target playback clocks
cancel/final/error state
```

初版只允许一个活动 GPU 推理会话，其他用户进入有界队列。不能让两个会话共享 prompt、
speaker token 或 codec state。

### 7.2 LiveQwenAdapter

从 `evaluation.simultaneous_streaming.stage4_streaming_generate` 抽取等价逻辑，但在新目录
实现在线 adapter：

```python
append_source(new_glm)
choose_action(is_final)
commit_wait()
generate_write(is_final)
reset()
```

adapter 必须：

- 使用 R2/R3 对应 tokenizer；
- 初始化 streaming task、target language 和固定 speaker prompt；
- source token 只追加，不回滚；
- action 只允许 WAIT/WRITE；
- final WAIT 强制转换成 WRITE flush；
- WRITE 结构异常时执行与评估一致的 normalization；
- 记录 forced action、structural recovery、TTFT 和 ACT；
- prompt 超过 18000 token 时显示警告，超过 32768 前安全终止。

### 7.3 Streaming BiCodec

直接复用：

```text
uniss.streaming.bicodec_streamer.StreamingBiCodecDecoder
```

固定：

```text
left_context_tokens=50
holdback_tokens=5
overlap_ms=80
sample_rate=16000
semantic_rate=50
```

每次 WRITE 立即调用 `push()`，得到稳定 PCM 后由 Gradio 生成器 yield 给流式音频组件，
不等待整句结束。备用 FastAPI 后端才通过 WebSocket 发送 PCM。

## 8. 上传音频模式

### 8.1 输入约束

初版建议：

```text
格式：wav/flac/ogg/mp3/m4a/aac/webm
最大文件：100 MB
时长：0.5--60 秒
内部格式：16 kHz、mono、float32/PCM16
```

复用旧 demo 的解码经验，但在新目录保留独立实现和测试，不 import 旧 demo 私有模块，
避免两个应用形成隐式耦合。

### 8.2 Evaluation-compatible replay

这是“当前最好效果”的默认上传模式。

流程：

1. 校验并标准化上传音频；
2. 完整运行一次 UniSSTokenizer，得到 `source_glm`、source BiCodec 和 speaker tokens；
3. 使用 `training.simul_uniss.schedule.tokens_per_chunk()` 的同一换算切成 640 ms source chunk；
4. 不使用 reference translation、oracle action 或 target token；
5. R2 自由运行预测 WAIT/WRITE 和目标 semantic；
6. Streaming BiCodec 生成每个目标 PCM chunk；
7. 保存每个 event 的 `source_end_ms`、action、text、semantic、codec wall time；
8. 推理完成后浏览器从 t=0 播放左侧源音频；
9. 右侧按 NCA 或 CA 时间线调度目标 chunk；
10. UI 明确标记“完整音频预编码后的评估兼容回放”。

该模式能最大程度复现当前 R2 测试效果，但源 token 可能包含约4秒 block 内未来上下文，
所以只用于在线试听和评估复现，不作为真正实时麦克风结论。

### 8.3 Upload live-inference

可选实验模式：上传后不提前编码完整音频，而是服务端按真实 1× 时间推进：

```text
每 640 ms 追加一段 waveform
→ WhisperVQ prefix re-encode
→ stable prefix commit
→ R2 WAIT/WRITE
→ 目标 PCM
```

该模式能够测量真实服务器处理速度，但效果可能低于 evaluation-compatible replay，必须单独
显示 prefix revision、first stable token 和 computation-aware latency。

## 9. 麦克风模式

### 9.1 浏览器采集

- Gradio `gr.Audio(..., streaming=True, type="numpy")` 持续提交录音块；
- `.stream()` 回调接收 `(sample_rate, numpy_audio)`，并将其归一化为单声道 float32；
- 服务端统一重采样到 16 kHz；
- 服务端每累计 640 ms 触发一次 frontend step；
- 浏览器本地显示源 waveform，但默认不把麦克风回放到扬声器，避免啸叫；
- 页面提示佩戴耳机后开启“监听原声”选项。

### 9.2 WhisperVQ 累计前缀前端

初版采用当前已实现的 cumulative-prefix baseline：

```text
640 ms  → encode audio[0:640]
1280 ms → encode audio[0:1280]
1920 ms → encode audio[0:1920]
...
```

使用：

```text
StablePrefixCommitter(holdback_tokens=2)
```

只向 R2 追加稳定且尚未提交的 GLM token。页面显示：

- candidate token 数；
- committed token 数；
- revision event 数；
- first stable token；
- frontend RTF。

已知 baseline 首个稳定 token 约 `4.22 s`，因此麦克风 v1 的首次目标音频不能承诺等于
R2 的 `4.11 s` policy latency；实际可能达到约 `6--10 s`，必须实测后报告。

### 9.3 Speaker token 策略

实时情况下完整 speaker token 不能提前获得。提供两种策略：

1. 默认固定目标音色：启动时加载经过许可的固定32个 speaker tokens，延迟最稳定；
2. 实验性 source-voice：收集前2--4秒后提取并冻结 speaker tokens，冻结后不允许变化。

不能每640 ms更新 speaker tokens，否则目标音色会漂移。页面必须显示当前音色策略。

### 9.4 结束与 flush

以下任一事件触发 final：

- 用户点击停止；
- 浏览器连接正常关闭并发送 `end`；
- VAD 检测连续1.2--1.8秒静音；
- 达到最大会话时长；
- 管理员取消。

final 时：

1. 提交全部剩余稳定前缀；
2. 强制最终 WRITE；
3. BiCodec `is_final=True` flush holdback；
4. 输出最终 WAV、event trace 和 JSON；
5. 浏览器播放完队列后显示完成。

## 10. Gradio 双播放器设计

### 10.1 页面布局

```text
┌─────────────────────────────┬─────────────────────────────┐
│ 源语音 / Source            │ 同声传译 / Translation      │
│ 原音频播放器或麦克风波形    │ 目标音频队列/播放器          │
│ source transcript（可选）   │ 增量翻译文本                  │
│ source timeline             │ WAIT/WRITE + chunk timeline  │
└─────────────────────────────┴─────────────────────────────┘
```

底部显示：

```text
模型、模式、First WRITE、StartOffset CA、ATD、RTF、forced actions、连接状态
```

### 10.2 音频调度

Gradio 流式音频组件负责增量播放，注入的 JavaScript 维护必要的目标音频状态和时间线：

- 初始 buffer 250--500 ms；
- 每个 PCM chunk 带 `event_index`、source timestamp 和 server-ready timestamp；
- Evaluation replay 按冻结的 NCA/CA 时间线调度；
- Live 模式按 chunk 实际到达时间调度；
- 自定义同步模式以 AudioContext 时钟作为播放参考；
- 浏览器暂停后恢复时重新计算目标队列偏移；
- 不对已播放目标语音执行回滚。

若 Gradio 浏览器端实际把多个 chunk 合并后才播放，或无法可靠恢复暂停后的时间轴，则该项
记录为 Gradio v1 限制并触发 FastAPI/WebSocket 备用实现，不能为了通过验收伪造时间戳。

### 10.3 同步导出

完成后生成：

```text
source.wav
translation.wav
aligned_stereo.wav       # 左=source，右=translation
event_trace.json
session_summary.json
```

`aligned_stereo.wav` 中目标音频按 StartOffset/WRITE 时间戳插入静音，使用户能直接听到实际
同传延迟，而不是把两段音频都从0秒开始。

## 11. 可选 FastAPI/WebSocket 备用协议

本节不属于 Gradio v1 的必需路径。Gradio v1 使用组件事件、`.stream()`、生成器和
`gr.State` 传递同等状态；只有第6.3节备用后端被启用时才实现下列协议。

### 11.1 Client → Server

```json
{"type":"start","mode":"microphone_prefix","direction":"cmn_to_eng","model":"r2"}
{"type":"audio","sequence":1,"sample_rate":48000,"encoding":"pcm16"}
{"type":"end"}
{"type":"cancel"}
```

音频 payload 使用二进制 frame；JSON 只传控制消息，避免 base64 增加带宽和 CPU。

### 11.2 Server → Client

```json
{"type":"ready","session_id":"...","actual_model":"r2"}
{"type":"frontend","candidate":42,"committed":8,"revision_events":1}
{"type":"action","event_index":4,"action":"wait","source_end_ms":3200}
{"type":"text_delta","event_index":5,"text":"Good morning"}
{"type":"audio_meta","event_index":5,"samples":6400,"sample_rate":16000}
{"type":"final","summary_url":"...","audio_url":"..."}
{"type":"error","recoverable":false,"message":"..."}
```

目标 PCM 紧跟 `audio_meta` 使用二进制 frame 发送。

## 12. GPU 与运行时设计

- 新服务默认使用独立 GPU，例如 GPU1；
- 启动脚本检测 GPU 是否被训练/评估任务占用；
- 不停止现有 `uniss_offline_phase3_demo`；
- R2 vLLM、WhisperVQ 和 BiCodec 共卡前先做显存 smoke；
- H200 显存足够，但 vLLM KV cache 必须设置有界利用率；
- 初版 `concurrency=1`，队列最大4；
- 每个请求完成或异常后 reset controller、codec 和 CUDA 临时缓存；
- 服务启动只加载一个主模型，不能为每个请求重新加载1.3GB权重；
- R3 fallback 不能与 R2 同时常驻，除非显存和初始化时间验证通过。

## 13. 结果保存与隔离

所有运行时数据仅写入：

```text
web_demo/streaming_s2st_r2_v1/runtime_outputs/YYYYMMDD/<session_id>/
```

禁止写入或覆盖：

```text
eval_outputs/
checkpoints/
data/
web_demo/offline_s2st_phase3_v1/runtime_outputs/
```

默认24小时清理；用户主动删除立即清理。结果 JSON 必须保存完整配置、模型 hash、模式和
“pseudo-streaming”标记，确保试听结果可追溯。

## 14. 安全与公网约束

- 按用户要求采用公开无登录模式：`share=True`、`auth=None`；
- `access_info.json` 权限设为0600并 gitignore，记录 `auth_mode: public_no_login`，不得保存密码；
- 页面明确提示“持有链接的任何人均可访问”，不得展示私有数据路径或历史会话；
- 上传文件名不能直接用作服务器路径；
- 校验扩展名、实际解码、文件大小、时长和非有限采样；
- 初版并发固定为1、队列最大4，并限制每个会话时长和请求频率；
- Gradio事件必须配置取消、空闲超时和最大输入长度；备用WebSocket必须有心跳和最大消息长度；
- 日志不保存鉴权密码、浏览器 token 或完整用户路径；
- `allowed_paths` 只允许当前 session 输出，`blocked_paths` 显式覆盖仓库、数据和checkpoint根目录；
- 明确阻止 checkpoint、data、pretrained_models 被静态服务器暴露；
- 麦克风权限仅在用户按钮触发后申请，页面说明数据保留时间。

## 15. 测试计划

### 15.1 纯CPU单元测试

- 音频格式、重采样、时长和大小校验；
- WebSocket消息顺序与非法状态；
- stable-prefix commit不回滚；
- session隔离和reset；
- 双播放器时间戳计算；
- aligned stereo 导出；
- TTL清理不越过新目录；
- 断连/cancel/final幂等。

### 15.2 GPU smoke

1. R2模型、WhisperVQ、BiCodec同卡加载；
2. 固定3条中英样本；
3. evaluation-compatible replay与现有generation结果对比；
4. 验证WAIT/WRITE序列、目标文本和semantic结构；
5. 生成PCM非空且无NaN；
6. 最大prompt不超过18000；
7. 无CUDA OOM；
8. reset后第二个会话不继承第一个会话状态。

### 15.3 上传端到端测试

- 10秒、30秒、60秒音频；
- 中文→英文、英文→中文；
- 左侧源音频1×播放；
- 右侧目标音频按事件时间到达；
- 暂停/恢复后同步误差不超过100ms；
- 导出文件时长和时间偏移正确；
- 刷新页面不破坏服务器session清理。

### 15.4 麦克风浏览器测试

- Chrome/Edge桌面端；
- 浏览器48kHz输入重采样；
- 连续说话、自然停顿、长静音；
- 中途停止、取消、网络断连、重连；
- 佩戴耳机时无明显回声；
- 翻译音频分块连续，无严重click；
- 首个稳定token、First WRITE和首个可播放PCM都被记录。

### 15.5 Gradio 公网测试

- `share=True, auth=None` 能自动生成 `https://*.gradio.live`；
- 未注册、未登录的外部浏览器可以打开页面；
- `public_url.txt` 和 `access_info.json` 记录的是当前有效地址；
- 外部网络完成一次上传和一次麦克风流式 smoke；
- 首次用户点击后目标流式音频可以自动播放；
- share失效时 watchdog 不继续报告旧地址，并可更新为新地址；
- 公网队列超过4时明确拒绝，不让请求无限占用GPU。

### 15.5 性能 gate

R2公开前至少满足：

```text
GPU OOM = 0
decode failure = 0（固定smoke）
RTF < 1
forced actions/sample <= 现有R3 audit水平
boundary click rate <= 0.01
prompt >18000 的样本必须显式告警
浏览器目标队列 underrun 有计数且可见
```

## 16. 实施阶段

### P0：隔离骨架

- 创建新目录、配置、gitignore、环境和测试骨架；
- 冻结R2/R3路径和hash；
- 不启动公网。

### P1：R2上传回放版

- 实现上传、完整tokenize、R2自由运行、Streaming BiCodec；
- 实现左右播放器和event timeline；
- 与现有评估样本做确定性回归；
- 这是最快可交付的“最佳效果在线试听”版本。

### P2：上传实时推进版

- 源音频按1×速度送入；
- cumulative-prefix WhisperVQ；
- stable prefix + R2在线状态；
- 记录真实服务器墙钟。

### P3：麦克风版

- Gradio `gr.Audio.stream()` + `gr.State`；
- numpy音频增量上传和翻译PCM生成器回传；
- stop/VAD/final flush；
- 浏览器实时文本和音频队列。

### P4：公网与稳定性

- Gradio `share=True, auth=None` 自动生成无需注册登录的 HTTPS 公网 URL；
- 开放访问下的并发、队列、速率、时长、TTL、日志和健康检查；
- tmux启动/停止/status和share watchdog；
- 外部浏览器真实请求smoke。

### P4b：仅在需要时启用的公网备用方案

- Gradio缓冲或同步达不到验收要求时，新增FastAPI/WebSocket适配层；
- Cloudflare quick tunnel自动生成临时URL，named tunnel提供固定域名；
- 不替换或覆盖已经可用的Gradio实现。

### P5：真正流式前端（后续研究）

- full-data causal audio student；
- encoder cache；
- 有限右上下文；
- 替换WhisperVQ累计前缀；
- 重新测真实raw-audio端到端latency。

## 17. 验收标准

### 上传模式

- 用户能上传、选择方向并看到左右双播放器；
- 左侧原音频正常1×播放；
- 右侧按照模型WAIT/WRITE时间开始播放；
- 页面显示增量翻译和每次WRITE位置；
- 能下载source、translation、aligned stereo和JSON；
- 使用R2并显示模型step/hash；
- 不修改或停止旧offline demo。

### 麦克风模式

- 用户点击开始后无需先录完整句子；
- 源PCM持续送入服务器；
- WRITE后目标PCM立即回传并进入播放队列；
- 用户停止后完整flush；
- 页面明确显示伪流式前端；
- 实测First stable token、First WRITE、StartOffset CA和RTF；
- 网络或模型错误不会导致GPU状态污染。

### 公网访问

- `launch_public_tmux.sh` 一次命令启动Gradio应用和share；
- `public_url.txt` 中必须是可用的 `https://` 地址；
- 外部网络无需账号、注册或登录即可打开首页；
- Chrome/Edge 在公网地址上能够申请麦克风权限；
- 公网完成至少一次上传双播放器和一次麦克风增量 smoke；
- 超出并发、队列、时长或输入限制的请求被安全拒绝；
- share进程或链接失效后watchdog自动恢复并原子更新URL；
- `status.sh` 不得继续报告已经失效的旧地址。

## 18. 主要风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| WhisperVQ前缀token高revision | 已播放目标无法撤回 | stable prefix、holdback、显示revision、限制会话长度 |
| 首包延迟高 | 用户误以为服务卡死 | 实时状态条、显示WAIT原因、预估延迟范围 |
| 完整预编码有未来信息 | 不能声称真流式 | 单独命名evaluation replay，UI永久标记 |
| speaker token实时不稳定 | 输出音色漂移 | 默认固定音色；实验模式2--4秒后冻结 |
| BiCodec分块边界click | 听感不连续 | 复用50/5/80ms参数，边界指标和试听gate |
| 浏览器禁止自动播放 | 目标音频不响 | 首次用户手势解锁AudioContext，显示播放状态 |
| 麦克风扬声器回授 | 啸叫/ASR污染 | 默认不回放源麦克风，提示戴耳机 |
| R2缺独立batch-one完整审计 | 公网稳定性不确定 | 上线前R2 smoke；失败自动回退R3 |
| 多用户状态串线 | 隐私和结果错误 | 每session独立controller/codec，初版单并发 |
| Gradio流式音频缓冲或双播放器同步不足 | 首包播放变慢或时间线偏移 | 实测gate；失败时启用FastAPI/WebSocket备用层 |
| 临时公网URL重启后变化 | 用户保存的地址失效 | `status.sh`显示实时URL；正式使用named tunnel固定域名 |
| Gradio share进程/链接失效 | 网页突然无法访问 | tmux+30秒健康检查+退避重启，并更新而非复用旧URL |
| 无登录公网链接被滥用 | GPU队列拥塞或恶意长音频 | concurrency=1、max queue=4、限时长/大小/频率和TTL |

## 19. 最终建议

按以下顺序实现：

1. 先完成 **R2上传回放版**，最快得到可公开试听、效果最接近当前正式评估的页面；
2. 再实现 **上传音频1×实时推进**，验证在线前缀重编码的真实退化；
3. 通过稳定性gate后开放 **麦克风边录边译**；
4. 用Gradio `share=True, auth=None`自动生成无需注册和登录的临时公网HTTPS网址；
5. 公网默认模型用R2，发生在线回归时明确回退R3；
6. 页面始终把当前版本标为 `pseudo-streaming source frontend`；
7. 只有Gradio实测无法满足音频增量播放或双路同步时，才启用FastAPI/WebSocket + Cloudflare；
8. 真正低延迟麦克风同传作为causal audio student接入后的独立v2，不在v1中夸大结论。

该方案既能满足“左侧播放原音频、右侧按同传时间播放目标音频”和“边录边译边播放”，
也能保持当前所有训练、评估和offline Gradio完全可复现、互不污染。
