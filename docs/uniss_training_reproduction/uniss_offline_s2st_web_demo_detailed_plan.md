# UniSS full198 Offline Speech-to-Speech Web Demo 详细设计与实施计划

> 日期：2026-07-27
> 目标：只使用当前效果最好的 UniSS full198 Phase3 checkpoint，建立独立的、非 simultaneous 的 speech-to-speech Gradio Web Demo，并提供一个可从公网打开的实际网址。
> 约束：不修改或覆盖现有 `web_demo/web_demo.py`、训练脚本、checkpoint、评估结果和 historical demo；所有新代码、日志和输出使用独立目录。

## 1. 结论摘要

可以实现以下页面能力：

1. 上传 WAV/MP3/M4A/FLAC 等音频；
2. 浏览器直接录音；
3. 自动判断或手动选择中英翻译方向；
4. 固定使用最佳 Phase3 checkpoint，不向用户暴露模型切换；
5. 固定使用 Phase3 `Quality` 模式，不提供模式切换；
6. 同时显示用户输入音频、模型自身ASR得到的源语音转写、目标翻译文本和模型生成音频；
7. 下载生成音频及本轮 JSON 元数据；
8. 用聊天式页面保存多轮上传/录音记录；
9. 显示 tokenization、LLM generation、BiCodec decode 和总耗时；
10. 第一版启动后必须返回一个实际可访问的公网 URL；没有固定域名时使用 Gradio share URL。

这个 demo 是 **offline/non-simultaneous S2ST**：模型必须收到一整段音频，或至少收到 VAD 判定结束的一段音频，才开始生成翻译。聊天式页面只是交互形式，不会把 offline 模型包装成真正的同声传译。

如果希望用户持续对着麦克风讲话、模型在讲话尚未结束时就逐块输出翻译语音，需要另外为 Stage4/Stage6 建立 simultaneous/streaming demo，不能与本文的 offline demo 混用指标或产品名称。

## 2. 当前仓库审计

### 2.1 现有 `web_demo.py` 能借鉴什么

当前文件：

```text
web_demo/web_demo.py
```

它实现了：

- Flask HTTP 服务；
- SSE 文本/PCM 推送；
- Flask-Sock WebSocket；
- VAD 后的音频段队列；
- 聊天历史展示；
- 模型锁和串行推理；
- 音频录制、音色选择、音色克隆等接口。

这些 Web 交互思想可以借鉴，但不能把当前文件直接换成 UniSS checkpoint 后使用，原因是：

1. 它加载的是 `MiniMindOmni`、`Mimi`、`SenseVoice` 和 CAM++，不是本项目 Phase3 的 UniSS Qwen + GLM speech tokenizer + BiCodec 路径；
2. 它引用的 `model.model_omni`、`trainer.trainer_utils` 等接口不是当前 UniSS offline 推理主链路；
3. 当前目录中没有它路由所需的 `web_demo.html`；
4. 它的语音输出是 Mimi code，当前 UniSS 输出是 BiCodec semantic token；
5. 它允许 WebSocket 实时切段，但这并不会让 offline checkpoint 具备因果 simultaneous 能力。

因此，实施时只参考其页面交互、队列、状态推送和模型锁设计，不修改该文件。

### 2.2 已验证的模型资产

唯一使用的 Phase3 HF 导出：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

该导出已经通过：

- tokenizer size：180,407；
- model vocab size：180,480；
- 73 个 Megatron padding token row；
- source checkpoint 和 safetensors SHA256 审计。

Speech tokenizer/BiCodec 资产：

```text
pretrained_models/UniSS
```

Web Demo 固定 checkpoint：

```text
Phase3 full198 iter_0009075
```

Demo 不加载 Phase2，不提供 Phase2/Phase3 A/B，不接受前端传入 checkpoint 路径，也不提供管理员在线切换模型功能。页面只显示当前固定模型身份：`Phase3 full198 iter_0009075`。

### 2.3 可以复用的 UniSS 推理实现

单文件官方示例：

```text
infer.py
```

批量 vLLM 示例：

```text
vllm_example.py
```

已经在 full198 评估中验证的组件：

```text
uniss/tokenizer.py
uniss/cli/prompt.py
training/generate_unist_eval_audio.py
training/sample_builders.py
evaluation/uniss_outputs.py
evaluation/vllm_generate.py
evaluation/decode_audio.py
```

Web Demo 后端应复用这些组件，不再自行发明特殊 token 格式。

## 3. Gradio 与公网 URL 决策

### 3.1 技术上不必须，但本项目第一版确定使用 Gradio

现有 `web_demo.py` 使用 Flask，说明从技术上不依赖 Gradio 也能完成录音、上传、聊天历史和音频播放。

但用户要求第一版必须给出一个实际公网网址，当前没有确认可用的固定域名、DNS和反向代理入口。因此第一版明确选择 Gradio，并使用其公网 share tunnel 生成可访问地址。

### 3.2 Gradio 的优势

Gradio 很适合第一版 MVP：

- `Audio` 组件原生支持麦克风和文件上传；
- `Audio` 输出组件可直接播放和下载生成音频；
- `Chatbot` 可显示每轮源文本和翻译文本；
- `Queue` 可以限制并发并展示排队状态；
- Python generator 可以持续更新“正在分词/正在生成/正在解码”等进度；
- 不需要先开发完整前端。

当前 `uniss-train` 环境没有检测到 Gradio。实施时需要额外安装，但应安装在独立 demo 环境中，不能直接改动训练环境。

建议环境位置：

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo
```

所有缓存继续放在：

```text
/opt/dlami/nvme/jasonleeeli/pip_cache
/opt/dlami/nvme/jasonleeeli/conda_pkgs
```

### 3.3 公网网址要求

第一版启动方式固定为：

```python
demo.queue(default_concurrency_limit=1, max_size=8).launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=True,
)
```

成功启动后，控制台会返回类似以下格式的实际网址：

```text
https://<gradio-generated-id>.gradio.live
```

实施完成时必须把控制台真实生成的完整 URL 告诉用户，并从服务器之外实际打开一次进行验证。文档中的 `<gradio-generated-id>` 只是格式说明，不算完成交付。

Gradio share URL 通常是临时地址，服务重启后可能变化。如果用户需要固定、长期不变的网站地址，则第二阶段必须准备域名和DNS，例如：

```text
https://uniss-demo.<user-domain>
```

然后使用 Nginx/Caddy 反向代理到本地 Gradio 服务。没有用户持有的域名时，第一版以真实生成的 `gradio.live` URL 作为公网交付网址。

## 4. 产品能力边界

### 4.1 支持的输入

- 浏览器麦克风录音；
- 上传 WAV、FLAC、MP3、M4A、OGG；
- 单声道或多声道输入；
- 任意采样率，后端统一转成 16 kHz mono float32；
- 中文或英文源语音；
- 自动方向或手动选择 `cmn→eng`、`eng→cmn`。

### 4.2 支持的输出

- 源音频播放器；
- Phase3 Quality 模式自身生成的源语音转写；
- 目标翻译文本；
- 16 kHz 目标翻译语音；
- WAV 下载；
- 本轮 JSON 下载；
- 阶段耗时、输出时长、semantic token 数、重复告警。

### 4.3 聊天式交互的正确含义

页面可以像聊天软件一样显示：

```text
用户输入：上传/录制的源音频
模型识别：源语音 transcription（用户刚才说的句子）
模型翻译：目标语言 translation
模型语音：可在线播放和下载的目标翻译音频
```

每一轮默认独立推理。历史记录只用于页面展示，不应自动拼入下一次 UniSS prompt，原因是 Phase3 不是按多轮对话式 S2ST 训练的，强行拼接历史可能降低翻译质量或越过上下文长度。

可支持“清空会话”“下载本次会话”和“重新生成本轮”，但不默认支持语义对话记忆。

### 4.4 与同声传译的区别

可提供两种用户体验：

1. 单段 offline：用户停止录音后处理整段；
2. VAD 分段 offline：持续录音，VAD 检测一句结束后把该句送入模型，下一句继续排队。

第二种看起来接近实时，但每句仍是 offline 推理，不能使用 AL/LAAL/ATD 等 simultaneous 结论来宣传。真正 simultaneous 页面应以后单独连接 Stage4/Stage6 streaming controller。

## 5. 推荐的独立目录结构

新实现建议放在：

```text
web_demo/offline_s2st_phase3_v1/
├── README.md
├── app_gradio.py
├── config.py
├── inference_engine.py
├── audio_io.py
├── session_store.py
├── launch_local.sh
├── launch_public.sh
├── requirements-demo.txt
├── assets/
│   └── optional_examples/
└── tests/
    ├── test_audio_io.py
    ├── test_prompt_and_parse.py
    ├── test_output_isolation.py
    └── test_security_limits.py
```

运行时输出使用独立且默认被 Git 忽略的目录：

```text
web_demo/offline_s2st_phase3_v1/runtime_outputs/<date>/<request_uuid>/
```

每个请求保存：

```text
input_original.*
input_16k.wav
chunks/chunk_*.wav
generated/chunk_*.wav
output_translation.wav
result.json
timing.json
```

禁止复用或写入：

```text
eval_outputs/
checkpoints/
pretrained_models/
web_demo/
```

## 6. 后端推理架构

### 6.1 模型初始化

服务启动时一次性加载：

1. `AutoTokenizer`：Phase3 HF export；
2. `AutoModelForCausalLM`：Phase3 HF export，bf16，eval；
3. `UniSSTokenizer`：`pretrained_models/UniSS`；
4. 可选VAD模型；
5. 模型和codec warmup。

禁止每个请求重新加载模型。

默认模型：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

模型路径只能来自服务端只读配置，客户端完全不能选择或覆盖。

### 6.2 单段推理流程

```text
上传/录音
  ↓
格式验证、大小/时长限制
  ↓
解码为16 kHz mono float32
  ↓
可选VAD切段
  ↓
UniSSTokenizer.tokenize()
  ├─ GLM linguistic tokens
  └─ BiCodec global/speaker tokens
  ↓
process_input()或与训练一致的sample builder构造prompt
  ↓
Phase3 HF model.generate()
  ↓
按EOS截断并屏蔽73个padding vocabulary row
  ↓
parse_with_tokenizer()解析文本和BiCodec semantic token
  ↓
UniSSTokenizer.decode()
  ↓
拼接多段音频、保存WAV和JSON
  ↓
页面返回输入音频、模型自身ASR转写、翻译文本、翻译音频和耗时
```

### 6.3 必须保留的生成保护

Megatron导出模型有73个padding vocabulary row。Demo 必须和正式评估一样屏蔽：

```text
[logical_vocab_size, model.config.vocab_size)
```

不能直接照抄旧 `infer.py` 而忽略 dummy token masking，否则可能生成不存在的特殊 token。

生成时还需：

- 设置 UniSS `pad_token_id` 和 `eos_token_id`；
- 遇到 EOS 立即截断；
- `repetition_penalty=1.1`；
- 限制 `max_new_tokens`；
- 记录 missing translation、missing semantic、missing EOS；
- 检测 semantic 最大连续重复；
- 无 semantic token 时返回文本和明确错误，不生成空 WAV。

### 6.4 固定使用 Phase3 Quality 模式

用户要求同时看到模型自身的源语音转写、目标翻译文本，并播放最终生成语音，因此第一版只使用 `Quality`：

```text
输入源语音
→ Phase3自身ASR transcription
→ Phase3目标translation
→ Phase3 BiCodec semantic/audio
```

`Performance` 只直接生成翻译和语音，不提供模型自身的源转写。若选择 Performance，就必须额外加载 Whisper/SenseVoice 等外部ASR，页面显示的 transcription 将不再是这次 UniSS Quality 推理自身的结果，因此不采用。

页面不显示模式下拉框，只读显示：

```text
Model: Phase3 full198 iter_0009075
Mode: Quality
```

Quality 输出解析必须验证两个完整 text block：

1. source transcription；
2. target translation。

如果模型没有生成完整 transcription/translation delimiter，页面应显示结构告警并保留原始生成诊断，不能用外部ASR结果静默替换。

Phase3没有正式训练 `direct_s2st` 主任务，因此第一版也不暴露 direct S2ST。

### 6.5 Transformers 与 vLLM 选择

第一版建议使用 Transformers：

- 单用户/低并发更简单；
- 不需要为交互请求预留超大KV cache；
- 和 BiCodec、Python音频处理集成直接；
- 更容易逐步报告进度和捕获错误。

访问量增加后，可将文本/semantic generation 替换为已经验证的 vLLM runner，但 speech tokenize 和 BiCodec decode 仍需独立GPU执行。不能简单启用高 tensor parallel；0.5B模型已经验证更适合单卡实例加数据并行worker。

## 7. 页面设计

### 7.1 主页面

左侧输入区：

- 录音/上传音频；
- 源语言：自动、中文、英文；
- 目标语言：自动、中文、英文；
- 模式：固定Quality，只读显示，无切换控件；
- 模型信息：只读显示 `Phase3 full198 iter_0009075`，无下拉切换；
- VAD切段开关；
- “开始翻译”“停止/取消排队”“清空会话”。

右侧输出区：

- 输入音频；
- `gr.Textbox(label="源语音转写 / Source transcription")`，显示用户说的句子；
- `gr.Textbox(label="翻译文本 / Translation")`，显示目标语言句子；
- `gr.Audio(label="翻译语音 / Generated speech")`，页面直接播放生成音频；
- 下载WAV/JSON；
- tokenization、generation、decode、total timing；
- 警告：无EOS、无semantic、结构恢复、长重复。

### 7.2 聊天记录区

每一轮显示：

- 输入音频文件名和时长；
- 源语言→目标语言；
- 源转写；
- 翻译文本；
- 输出音频播放器；
- 固定Phase3模型、Quality模式和生成参数；
- 本轮失败时的可读错误。

浏览器刷新后默认不永久保存历史。公网若要保存，需要用户明确同意，并设置过期时间。

### 7.3 长音频策略

第一版建议：

- 单文件硬限制60秒；
- 推荐输入30秒以内；
- 超过阈值自动VAD切段；
- 每段串行推理，保持顺序；
- 目标音频按输出顺序拼接，中间加入可配置短静音或cross-fade；
- 默认输出独立翻译音频，不把译文强行覆盖到原视频/原音频时间轴。

“保持原时间轴配音”可以作为后续实验选项，但与自然的目标语速可能冲突，不能作为第一版默认行为。

## 8. 并发和GPU资源

第一版只允许一个生成任务占用模型：

```text
model concurrency = 1
queue length = 4～8
```

原因：

- GLM tokenizer、Qwen和BiCodec共享同一GPU时峰值显存需要实测；
- 两个用户同时调用同一BiCodec实例需要避免状态污染；
- 公网无限并发容易造成OOM和磁盘堆积。

推荐使用一张独立GPU，不与训练、正式evaluation或TensorBoard服务抢卡。启动器应先检查指定GPU是否空闲，不能自动抢占已有进程。

模型锁的粒度建议覆盖：

```text
speech tokenize
→ Qwen generate
→ BiCodec decode
```

第一版优先正确性。后续若拆成多worker，需要每个worker独立加载完整模型和codec。

## 9. 公网部署、实际网址与安全要求

### 9.1 第一版公网方式：Gradio share URL

第一版为了满足“启动后立即给出公网网址”的要求，允许使用：

```bash
python app_gradio.py --share
```

启动器必须：

1. 开启 `share=True`；
2. 捕获并打印真实 `https://*.gradio.live` 地址；
3. 将URL写入独立运行日志；
4. 把真实URL告诉用户；
5. 使用外部网络访问页面并完成一次健康检查；
6. 如果share tunnel建立失败，明确报错，不能只给本地 `127.0.0.1` 地址冒充公网网址。

Gradio share 适合当前演示需求，但不是永久域名，服务停止或重启后URL可能变化。

### 9.2 长期固定网址

若需要长期固定网站，正式部署建议：

```text
Internet
  ↓ HTTPS
Nginx/Caddy
  ↓ authentication + rate limit + body limit
Gunicorn/Uvicorn/Gradio service
  ↓ local-only backend
UniSS GPU worker
```

固定网址必须由用户提供或注册域名并配置DNS。没有域名时不能承诺固定字符串，只能提供每次启动时真实生成的 Gradio URL。

### 9.3 必须增加的保护

1. HTTPS；
2. 登录认证或至少访问token；
3. IP/用户限流；
4. 请求队列上限；
5. 上传文件大小和音频时长上限；
6. MIME、扩展名和真实音频解码联合校验；
7. UUID文件名，禁止使用用户文件名拼接服务器路径；
8. 禁止任意路径、URL下载和模型路径参数；
9. 超时和任务取消；
10. 临时音频自动删除；
11. 磁盘配额和低空间熔断；
12. 不使用 `CORS(*)`；
13. 日志不记录完整语音内容和用户隐私文本；
14. 对外说明上传音频处理和删除策略；
15. 服务进程使用非root用户；
16. checkpoint和代码目录只读挂载；
17. 仅 demo output/temp 目录可写；
18. `/metrics`、调试栈和管理接口不能匿名暴露。

### 9.4 Gradio公网配置

Gradio 第一版应：

- `share=True`，生成临时公网网址；
- 本地同时监听固定端口，便于SSH转发和故障排查；
- 配置Gradio登录认证或一次性高强度用户名/密码；
- 开启Gradio queue；
- 限制并发为1；
- 关闭不需要的API端点；
- 固定allowed paths，禁止暴露整个项目目录。

进入长期固定域名阶段后，改为 `share=False`，仅监听本机/内网端口，并由Nginx提供域名、TLS、认证和限流。

## 10. 测试方案

### 10.1 单元测试

- 多格式音频转16 kHz mono；
- 空文件、损坏文件、超长文件拒绝；
- 中英目标语言映射；
- Quality prompt与训练格式一致；
- 73个dummy token始终被屏蔽；
- EOS截断；
- text/semantic解析；
- semantic重复检测；
- 输出目录UUID隔离；
- 路径穿越测试；
- 任务取消和临时文件清理。

### 10.2 推理回归测试

从既有评估试听集中选择固定样本：

```text
experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_dev_listen_50.jsonl
```

至少覆盖：

- 中文→英文；
- 英文→中文；
- Quality源转写、翻译文本和生成语音三项完整；
- 短句；
- 长句；
- 历史上出现semantic重复的样本。

使用相同 checkpoint、seed、temperature、top-p、repetition penalty 时，Demo 后端应与正式 HF evaluation 产生相同或可解释的一致输出。

### 10.3 Web测试

- 麦克风权限拒绝；
- 上传后重复点击；
- 两个浏览器同时提交；
- 排队和取消；
- 页面刷新；
- 输出音频播放与下载；
- 推理异常后GPU显存恢复；
- Nginx超时和大文件拒绝；
- 临时文件过期删除。

### 10.4 性能验收

记录：

- 冷启动时间；
- warmup后首token时间；
- speech tokenize时间；
- Qwen generation时间；
- BiCodec decode时间；
- 总RTF；
- GPU峰值显存；
- 队列等待时间。

Demo性能数据不能替代正式dev/test benchmark，但可以用于公网容量规划。

## 11. 分阶段实施顺序

### Phase A：纯后端单文件验证

1. 新建独立demo目录；
2. 固定Phase3 HF checkpoint和speech tokenizer；
3. 实现单WAV输入；
4. 固定实现Phase3 Quality；
5. 输出模型自身ASR transcription、翻译文本和WAV；
6. 对固定listen样本做回归；
7. 记录耗时和错误。

通过条件：已知中英样本均能显示非空模型ASR transcription、非空翻译文本和可播放翻译音频。

### Phase B：Gradio本地MVP

1. 增加上传和麦克风；
2. 增加聊天式历史；
3. 增加模型状态和进度；
4. 增加下载和清空；
5. 增加队列和并发限制；
6. 在localhost完成浏览器测试。

### Phase C：长音频和VAD

1. 音频限制；
2. VAD切段；
3. 分段串行生成；
4. 文本和音频拼接；
5. 长输出重复保护；
6. 单请求超时和取消。

### Phase D：Gradio公网交付

1. 用 `share=True` 启动；
2. 获取真实 `gradio.live` URL；
3. 从外部网络打开；
4. 上传一条短音频验证完整Phase3 S2ST；
5. 把真实URL和有效期说明告诉用户；
6. 配置登录认证；
7. 限制并发、文件大小和音频时长；
8. 启用临时文件清理；
9. 将进程放入tmux/systemd，避免SSH断开导致服务退出。

### Phase E：固定域名生产化或自定义前端

访问量和页面需求明确后，再决定：

- 保留Gradio；或
- 抽出FastAPI/Flask API；
- 开发与现有 `web_demo.py` 风格接近的独立HTML/JavaScript页面；
- 增加用户、任务和会话数据库。

## 12. 预期启动方式

第一版启动命令最终应类似：

```bash
CUDA_VISIBLE_DEVICES=0 \
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  web_demo/offline_s2st_phase3_v1/app_gradio.py \
  --model checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf \
  --speech-tokenizer pretrained_models/UniSS \
  --host 127.0.0.1 \
  --port 7861 \
  --share
```

程序启动成功后应同时显示：

```text
Local URL:  http://127.0.0.1:7861
Public URL: https://<real-generated-id>.gradio.live
```

其中第二行必须是真实生成且验证可访问的地址。交付时需要把它直接告诉用户。

内网访问可通过SSH转发：

```bash
ssh -L 7861:127.0.0.1:7861 <server>
```

浏览器访问：

```text
http://127.0.0.1:7861
```

如果 Gradio share 建立失败，则保留本地/SSH访问用于诊断，但任务不能标记为“公网完成”。必须修复出网、Gradio tunnel或改用域名反向代理后再提供网址。

## 13. 验收标准

第一版完成应同时满足：

1. 不修改 `web_demo/web_demo.py`；
2. 不修改训练、evaluation和historical脚本默认行为；
3. 只加载Phase3 iter 9075 HF export，不加载Phase2；
4. 页面不提供模型切换，checkpoint hash和source checkpoint写入每次结果；
5. 支持浏览器录音和文件上传；
6. 中→英和英→中均可显示模型自身ASR transcription、翻译文本并播放生成音频；
7. Quality的transcription、translation和semantic/audio三段结构正确；
8. dummy vocabulary row被屏蔽；
9. 每次输出使用UUID独立目录；
10. 页面可连续完成多轮独立翻译；
11. 并发限制不会导致OOM或请求交叉；
12. 失败请求不会留下模型锁或无限临时文件；
13. 本地smoke通过后使用Gradio share建立公网URL；
14. 提供真实完整的 `https://*.gradio.live` 地址；
15. 从外部网络完成页面访问和一次短音频翻译；
16. 公网入口具备认证、限流和自动清理；
17. 如果需要固定长期网址，必须完成域名、DNS和反向代理部署。

## 14. 最终建议

最合理的第一版是：

```text
Gradio聊天式页面
+ 麦克风/上传
+ 固定Phase3 full198 iter 9075 offline Quality
+ 显示模型自身ASR transcription
+ 显示翻译文本
+ 独立输出音频播放器
+ 单GPU、单并发队列
+ Gradio share公网URL
```

完成本地稳定性验证后，立即启动Gradio share并把真实公网URL告诉用户。需要长期固定URL时，再迁移到Nginx/Caddy和用户域名。

不要在第一版中加入：

- 真正continuous simultaneous声称；
- 匿名无限上传；
- 任意checkpoint路径；
- Phase2或在线模型切换；
- 多GPU动态抢占；
- 无限制长音频；
- 用户上传音频永久保存；
- Stage4/Stage6 streaming与Phase3 offline混合在同一个指标口径中。

未来若需要真正同声传译，应新建第二个独立应用，例如：

```text
web_demo/streaming_s2st_v1/
```

它连接Stage4/Stage6 streaming controller、WebSocket audio chunks和computation-aware latency，与本文的offline demo并列而不是替代。
