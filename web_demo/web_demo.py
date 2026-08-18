import argparse, os, sys, re, json, time, math, torch, threading, queue, base64, io, logging, contextlib
import numpy as np
import torchaudio
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image
from pydub import AudioSegment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel
from model.model_omni import MiniMindOmni, RealtimeSession, OmniConfig
from trainer.trainer_utils import log_model_params
logging.getLogger().setLevel(logging.ERROR)
with contextlib.redirect_stdout(io.StringIO()):
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

app = Flask(__name__, static_folder='.')
CORS(app)
sock = Sock(app)
M = {}  # model / tokenizer / device / mimi / asr / cfg
V = {}  # voice_name -> {ref_codes, spk_emb}
V_builtin, V_unseen, V_manual = [], [], []
MODEL_LOCK = threading.Lock()
SAMPLES_PER_FRAME = 1920
REF_FRAMES = 300
CLONE_VOICE = 'voice_clone'
CLONE_FILE = 'voice_clone.pt'

# 翻译目标语言名称（与 dataset/convert_s2s_translate.py 训练格式一致）
_ZH_NAME = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文", "yue": "中文"}
_EN_NAME = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}
_LANG_TAGS = ('zh', 'en', 'yue', 'ja', 'ko')

# -------- helpers --------
def sse(d): return f"data: {json.dumps(d)}\n\n"

def scan_hf_models(base_dir):
    models = {}
    base_dir = os.path.abspath(base_dir)
    for d in sorted(os.listdir(base_dir), reverse=True):
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path) or d.startswith('.') or d.startswith('_'):
            continue
        files = set(os.listdir(full_path))
        has_model = bool(files & {'pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json', 'model.safetensors.index.json'})
        if has_model:
            models[d] = full_path
    return models

def asr_run(samples):
    r = M['asr'].generate(input=samples, cache={}, language='auto', use_itn=True)
    return rich_transcription_postprocess(r[0]['text']).strip() if r else ''

def asr_run_with_lang(samples):
    """一次调用 SenseVoice，同时返回 (语种, 文本)。语种从原始输出的 <|xx|> 标签解析。"""
    r = M['asr'].generate(input=samples, cache={}, language='auto', use_itn=True)
    if not r:
        return 'zh', ''
    raw = r[0]['text']
    m = re.search(r'<\|([a-z]{2,3})\|>', raw)
    lang = m.group(1) if (m and m.group(1) in _LANG_TAGS) else 'zh'
    text = rich_transcription_postprocess(raw).strip()
    return lang, text

def build_translate_instruction(src_lang, src_text):
    """根据 ASR 识别的源语种构造翻译指令，返回 (指令, 目标语种)。
    与训练格式一致：中文源用中文指令译成英文；其余源用英文指令译成中文。"""
    if src_lang in ('zh', 'yue'):
        tgt = 'en'
        return f'将"{src_text}"翻译成{_ZH_NAME[tgt]}', tgt
    tgt = 'zh'
    return f'Translate "{src_text}" into {_EN_NAME[tgt]}', tgt

def prep_audio(samples):
    m = M['model']
    proc = m.audio_processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel = proc.input_features.squeeze(0).unsqueeze(0).to(M['device'])
    vlen = proc.attention_mask.sum().item()
    prompt = m.config.audio_special_token * (vlen or 1)
    return mel, torch.tensor([vlen], device=M['device']), prompt

def prep_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return {k: v.to(M['device']) for k, v in M['model'].vision_processor(images=img, return_tensors="pt").items()}

def build_ids(prompt, history):
    tok, dev, n = M['tokenizer'], M['device'], M['cfg'].max_history_turns
    hist = history[-n:] if n > 0 else []
    msgs = hist + [{"role": "user", "content": prompt}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tok(t).data['input_ids'], dtype=torch.long, device=dev)[None, ...]

def _mimi_decode(frames):
    codes = [f for f in frames if f and len(f) == 8]
    if not codes or not M['mimi']: return None
    mc = torch.tensor(codes, dtype=torch.long, device=M['device']).T.unsqueeze(0)
    mc = torch.where(mc >= 2049, torch.zeros_like(mc), mc)
    with torch.no_grad():
        au = M['mimi'].decode(mc).audio_values.squeeze().cpu().numpy()
    return au, mc.shape[-1]

def pcm_bytes(frames, ov):
    r = _mimi_decode(frames)
    if r is None: return None
    au, T = r
    if ov > 0: au = au[int(ov * len(au) / T):]
    return (au * 32767).astype('int16').tobytes()

def stream_pcm(frames, flush=False):
    """yield (pcm_bytes,) on chunk boundaries or on final flush."""
    if not M['mimi']: return
    cf, ov_max, n = M['cfg'].audio_chunk_frames, M['cfg'].audio_overlap, len(frames)
    if not flush and n >= cf and n % cf == 0:
        ov = min(ov_max, n - cf)
        p = pcm_bytes(frames[-(cf + ov):], ov)
        if p: yield p
    elif flush:
        rem = n % cf
        if rem:
            ov = min(ov_max, n - rem)
            p = pcm_bytes(frames[-(rem + ov):], ov)
            if p: yield p

def voice_args(name):
    if name and name != 'default' and name in V:
        v = V[name]
        dev = M['device']
        rc = v['ref_codes'].unsqueeze(0).to(dev)
        se = v['spk_emb'].half().unsqueeze(0).to(dev) if 'spk_emb' in v else None
        return {'ref_codes': rc, 'spk_emb': se}
    return {}

def register_voice(name, value, group='manual'):
    V[name] = value
    groups = {'builtin': V_builtin, 'unseen': V_unseen, 'manual': V_manual}
    dst = groups[group]
    if name not in dst:
        dst.append(name)
    for k, lst in groups.items():
        if k != group and name in lst:
            lst.remove(name)

def clone_voice_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'speaker', CLONE_FILE)

def delete_manual_voice(name):
    if name not in V_manual:
        raise RuntimeError('只能删除手动克隆的音色')
    out_path = clone_voice_path()
    saved = torch.load(out_path, map_location='cpu') if os.path.exists(out_path) else {}
    if name in saved:
        saved.pop(name)
        torch.save(saved, out_path)
    V.pop(name, None)
    if name in V_manual:
        V_manual.remove(name)

def normalize_voice_name(name):
    name = ' '.join(str(name or '').split())
    if not name:
        name = CLONE_VOICE
    if len(name) > 24:
        raise RuntimeError('音色名太长，建议控制在 24 个字以内')
    if name.lower() == 'default':
        raise RuntimeError('default 是保留名称，请换一个')
    if name in V_builtin or name in V_unseen:
        raise RuntimeError('该名称已被现有音色占用，请换一个')
    return name

def validate_clone_audio(w16):
    if w16.numel() < int(16000 * 1.8):
        raise RuntimeError('录音太短，请把整句话读完')
    peak = w16.abs().max().item()
    frame, hop = 800, 400
    if w16.numel() >= frame:
        rms = w16.unfold(0, frame, hop).pow(2).mean(dim=1).sqrt().cpu().numpy()
    else:
        rms = np.array([w16.pow(2).mean().sqrt().item()])
    hi = float(np.quantile(rms, 0.95))
    lo = float(np.quantile(rms, 0.2))
    if hi < 0.008:
        raise RuntimeError('录音太轻，请靠近麦克风一点')
    if hi > 0 and lo / hi > 0.45:
        raise RuntimeError('环境噪声太大，请换安静一点的环境')
    if peak > 0.995:
        raise RuntimeError('录音有爆音，请离麦克风远一点')

def build_clone_voice(audio_b64):
    if M.get('mimi') is None or M.get('campplus') is None or M.get('mel_fn') is None:
        raise RuntimeError('Mimi 或 CAM++ 未加载')
    seg = AudioSegment.from_file(io.BytesIO(base64.b64decode(audio_b64))).set_channels(1).set_sample_width(2)
    if len(seg) < 1000:
        raise RuntimeError('录音太短，至少读 1 秒')
    try:
        seg = seg.speedup(playback_speed=1.5, chunk_size=150, crossfade=25)
    except Exception:
        seg = seg.speedup(playback_speed=1.5)
    seg24 = seg.set_frame_rate(24000)
    seg16 = seg.set_frame_rate(16000)
    w24 = torch.tensor(np.frombuffer(seg24.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)
    w16 = torch.tensor(np.frombuffer(seg16.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)
    validate_clone_audio(w16)
    mimi_dev = next(M['mimi'].parameters()).device
    mimi_dtype = torch.float16 if mimi_dev.type != 'cpu' else torch.float32
    with torch.inference_mode():
        t = w24.unsqueeze(0).unsqueeze(0).to(device=mimi_dev, dtype=mimi_dtype)
        codes = M['mimi'].encode(t).audio_codes
        nf = math.ceil(w24.shape[-1] / SAMPLES_PER_FRAME)
        ref_codes = codes[0, :8, :nf].cpu()[:, :min(nf, REF_FRAMES)]
    with torch.no_grad():
        mel = M['mel_fn'](w16.unsqueeze(0).to(M['device']))
        feat = mel.clamp(min=1e-10).log().transpose(1, 2)
        feat = feat - feat.mean(dim=1, keepdim=True)
        spk_emb = M['campplus'](feat).squeeze(0).cpu()
    return {'ref_codes': ref_codes, 'spk_emb': spk_emb}

def voice_from_samples(samples):
    """音色克隆：直接用输入音频本身作为音色参考，返回 generate 所需的 {ref_codes[, spk_emb]}。
    samples: np.float32 @16kHz。ref_codes 走 Mimi(24k)，spk_emb 走 CAM++(16k)。"""
    if samples is None or M.get('mimi') is None or len(samples) == 0:
        return {}
    dev = M['device']
    w16 = torch.as_tensor(np.asarray(samples, dtype=np.float32))
    w24 = torchaudio.functional.resample(w16, 16000, 24000)
    mimi_dev = next(M['mimi'].parameters()).device
    mimi_dtype = torch.float16 if mimi_dev.type != 'cpu' else torch.float32
    with torch.inference_mode():
        t = w24.unsqueeze(0).unsqueeze(0).to(device=mimi_dev, dtype=mimi_dtype)
        codes = M['mimi'].encode(t).audio_codes
        nf = max(1, math.ceil(w24.shape[-1] / SAMPLES_PER_FRAME))
        ref_codes = codes[0, :8, :nf][:, :min(nf, REF_FRAMES)].long().unsqueeze(0).to(dev)
    out = {'ref_codes': ref_codes}
    if M.get('campplus') is not None and M.get('mel_fn') is not None:
        with torch.no_grad():
            mel = M['mel_fn'](w16.unsqueeze(0).to(dev))
            feat = mel.clamp(min=1e-10).log().transpose(1, 2)
            feat = feat - feat.mean(dim=1, keepdim=True)
            out['spk_emb'] = M['campplus'](feat).squeeze(0).half().unsqueeze(0).to(dev)
    return out

def run_generate(x, audio_inputs, audio_lens, pixel_values, **kw):
    with MODEL_LOCK, torch.no_grad():
        yield from M['model'].generate(
            x, M['tokenizer'].eos_token_id, stream=True, return_audio_codes=True,
            audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, **kw)

def load_main_model(ckpt_path, model_name):
    with MODEL_LOCK:
        M.pop('model', None); M.pop('tokenizer', None)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        cfg = M['cfg']
        tok = AutoTokenizer.from_pretrained('../model', trust_remote_code=True)
        m = MiniMindOmni(
            OmniConfig(hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers, use_moe=bool(cfg.use_moe)),
            audio_encoder_path='../model/SenseVoiceSmall',
            vision_model_path=None,  # 翻译服务无需视觉
        )
        m.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        m = m.half().eval().to(M['device'])
        if m.audio_encoder is not None: m.audio_encoder.to(M['device'])
        M['tokenizer'], M['model'], M['model_name'] = tok, m, model_name
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f'Loaded translate model: {model_name} ({params:.2f}M)')
        return round(params, 2)

def prepare_translate_turn(text, samples):
    """构造一轮翻译输入：语音→SenseVoice(语种+文本)→翻译指令；纯文本按字符集判定语种。
    返回 (audio_inputs, audio_lens, prompt, src_text, src_lang, tgt_lang)。"""
    audio_inputs = audio_lens = None
    if samples is not None and len(samples) > 0:
        src_lang, src_text = asr_run_with_lang(samples)
        audio_inputs, audio_lens, audio_prompt = prep_audio(samples)
        instruction, tgt_lang = build_translate_instruction(src_lang, src_text)
        prompt = audio_prompt + '\n\n' + instruction
    else:
        src_text = (text or '').strip()
        src_lang = 'zh' if re.search(r'[\u4e00-\u9fff]', src_text) else 'en'
        instruction, tgt_lang = build_translate_instruction(src_lang, src_text)
        prompt = instruction
    return audio_inputs, audio_lens, prompt, src_text, src_lang, tgt_lang

# -------- routes --------
@app.route('/')
def index(): return send_from_directory('.', 'web_demo.html')
@app.route('/call')
def call_page(): return send_from_directory('.', 'web_demo.html')

@app.route('/voices')
def get_voices():
    return json.dumps({'builtin': sorted(V_builtin), 'unseen': sorted(V_unseen), 'manual': sorted(V_manual)})

@app.route('/models')
def get_models():
    return json.dumps({'models': list(M.get('models', {}).keys()), 'current': M.get('model_name')})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    name = (request.json or {}).get('name')
    if name not in M.get('models', {}):
        return Response(json.dumps({'ok': False, 'error': 'unknown model'}), status=400, mimetype='application/json')
    try:
        params = load_main_model(M['models'][name], name)
        return Response(json.dumps({'ok': True, 'model': name, 'params': params}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    d = request.json or {}
    if not d.get('audio'):
        return Response(json.dumps({'ok': False, 'error': 'missing audio'}), status=400, mimetype='application/json')
    try:
        name = normalize_voice_name(d.get('name'))
        value = build_clone_voice(d['audio'])
        out_path = clone_voice_path()
        saved = torch.load(out_path, map_location='cpu') if os.path.exists(out_path) else {}
        saved[name] = value
        torch.save(saved, out_path)
        register_voice(name, value, group='manual')
        return Response(json.dumps({'ok': True, 'voice': name, 'path': './model/speaker/' + CLONE_FILE}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/delete_voice', methods=['POST'])
def delete_voice():
    d = request.json or {}
    name = ' '.join(str(d.get('name') or '').split())
    if not name:
        return Response(json.dumps({'ok': False, 'error': 'missing name'}), status=400, mimetype='application/json')
    try:
        delete_manual_voice(name)
        return Response(json.dumps({'ok': True, 'voice': name}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/chat', methods=['POST'])
def chat():
    d = request.json
    history = d.get('history', [])
    samples = None
    if d.get('audio'):
        seg = AudioSegment.from_file(io.BytesIO(base64.b64decode(d['audio']))).set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    def gen():
        audio_inputs, audio_lens, prompt, src_text, src_lang, tgt_lang = prepare_translate_turn(d.get('text', ''), samples)
        print(f'[CHAT] lang={src_lang}->{tgt_lang} src="{src_text}"', flush=True)
        # 音色克隆：默认用输入音频本身；无音频时回退到所选音色
        va = voice_from_samples(samples) if samples is not None else voice_args(d.get('voice', 'default'))
        x = build_ids(prompt, history)
        if src_text:
            yield sse({'type': 'user_prompt', 'content': src_text})
        frames, text_ttft, audio_ttft = [], None, None
        t0 = time.time(); hi = 0; full = ''
        for y, af in run_generate(x, audio_inputs, audio_lens, None,
                                   max_new_tokens=d.get('max_tokens', 512),
                                   temperature=d.get('temperature', 0.3), top_p=0.85, **va):
            if y is not None:
                if text_ttft is None:
                    text_ttft = (time.time() - t0) * 1000
                    yield sse({'type': 'ttft', 'text_ttft': round(text_ttft, 1)})
                ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                if ans and ans[-1] != '\ufffd' and len(ans) > hi:
                    yield sse({'type': 'text', 'content': ans[hi:]}); hi = len(ans); full = ans
            if af:
                if audio_ttft is None:
                    audio_ttft = (time.time() - t0) * 1000
                    yield sse({'type': 'ttft', 'audio_ttft': round(audio_ttft, 1)})
                frames.append(af)
                for pcm in stream_pcm(frames):
                    b64 = base64.b64encode(pcm).decode()
                    for i in range(0, len(b64), 2000):
                        yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
        for pcm in stream_pcm(frames, flush=True):
            b64 = base64.b64encode(pcm).decode()
            for i in range(0, len(b64), 2000):
                yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
        print(f'[CHAT] gen="{full}" audio_frames={len(frames)}', flush=True)
        yield sse({'type': 'done'})

    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@sock.route('/ws/realtime')
def realtime(ws):
    session = RealtimeSession(M['vad_path'], min_silence_ms=M['cfg'].vad_silence_ms)
    q = queue.Queue()            # 原始 ws 消息
    seg_queue = queue.Queue()    # VAD 切出的完整语音段(np.float32)，待翻译
    alive = [True]
    state = {'history': [], 'voice': 'default'}
    n_hist = M['cfg'].max_history_turns
    send_lock = threading.Lock()

    def safe_send(obj):
        try:
            with send_lock:
                ws.send(json.dumps(obj))
        except Exception:
            alive[0] = False

    def push_audio(data):
        return session.push_chunk(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

    def set_ctx(msg):
        h = msg.get('history') or []
        state['history'] = h[-n_hist:] if n_hist > 0 else []
        if 'voice' in msg: state['voice'] = msg.get('voice', 'default')

    def recv_loop():
        while alive[0]:
            try:
                data = ws.receive(timeout=1)
                if data is None: alive[0] = False; break
                q.put(data)
            except: alive[0] = False; break

    def worker_loop():
        """消费者：顺序翻译 seg_queue 中的每一段语音，边翻边流式输出；不打断。
        单段处理若抛异常(如显存OOM)不会杀死线程，会清缓存、通知前端并继续下一段。"""
        while alive[0]:
            try: audio = seg_queue.get(timeout=0.1)
            except queue.Empty: continue
            if audio is None: break
            try:
                audio_inputs, audio_lens, prompt, src_text, src_lang, tgt_lang = prepare_translate_turn('', audio)
                print(f'[RT] dur={len(audio)/16000:.2f}s lang={src_lang}->{tgt_lang} src="{src_text}"', flush=True)
                x = build_ids(prompt, state['history'])
                va_rt = voice_from_samples(audio)  # 音色克隆：默认用输入音频本身
                safe_send({'type': 'generating'})
                if src_text: safe_send({'type': 'user_prompt', 'content': src_text})
                frames, full_text = [], ''
                for y, af in run_generate(x, audio_inputs, audio_lens, None,
                                           max_new_tokens=512, temperature=0.3, **va_rt):
                    if not alive[0]: break
                    if y is not None:
                        ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                        if ans and ans[-1] != '\ufffd' and len(ans) > len(full_text):
                            safe_send({'type': 'text', 'content': ans[len(full_text):]}); full_text = ans
                    if af:
                        frames.append(af)
                        for pcm in stream_pcm(frames):
                            safe_send({'type': 'pcm', 'data': base64.b64encode(pcm).decode()})
                for pcm in stream_pcm(frames, flush=True):
                    safe_send({'type': 'pcm', 'data': base64.b64encode(pcm).decode()})
                print(f'[RT] gen="{full_text}" audio_frames={len(frames)}', flush=True)
                if n_hist > 0:
                    if src_text: state['history'].append({'role': 'user', 'content': src_text})
                    if full_text: state['history'].append({'role': 'assistant', 'content': full_text})
                    state['history'] = state['history'][-n_hist:]
            except Exception as ex:
                print(f'[RT][ERROR] 翻译该段失败: {repr(ex)}', flush=True)
                try:
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception: pass
                safe_send({'type': 'text', 'content': f'⚠ 翻译失败: {ex}'})
            safe_send({'type': 'done', 'pending': seg_queue.qsize()})

    threading.Thread(target=recv_loop, daemon=True).start()
    threading.Thread(target=worker_loop, daemon=True).start()
    try:
        while alive[0]:
            try: data = q.get(timeout=0.1)
            except queue.Empty: continue
            if isinstance(data, str):
                m = json.loads(data)
                t = m.get('type')
                if t == 'context': set_ctx(m)
                elif t == 'end': break
                # 'stop' 不再打断，忽略
                continue
            # 音频：持续 VAD 切分（翻译进行时也照常切分入队，不丢音频）
            status = push_audio(data)
            safe_send({'type': 'vad', 'speaking': session.speaking})
            if status == 'speech_end':
                seg = session.get_audio()
                if len(seg) > 0:
                    seg_queue.put(seg)  # 入队等待顺序翻译，队列不丢
    finally:
        alive[0] = False
        seg_queue.put(None)


def init_model(args):
    M['cfg'] = args; M['device'] = args.device
    with contextlib.redirect_stdout(io.StringIO()):
        M['asr'] = AutoModel(model='../model/SenseVoiceSmall', trust_remote_code=True, device=args.device, disable_update=True)
    model_name = os.path.basename(args.ckpt)
    M['models'] = {model_name: args.ckpt}
    load_main_model(args.ckpt, model_name)
    try:
        from transformers import MimiModel
        M['mimi'] = MimiModel.from_pretrained('../model/mimi').eval().to(args.device)
        if args.device != 'cpu': M['mimi'] = M['mimi'].half()
        print('Mimi model loaded')
    except: M['mimi'] = None
    try:
        from modelscope.models.audio.sv.DTDNN import CAMPPlus
        M['campplus'] = CAMPPlus(feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4,
                                 init_channels=128, config_str='batchnorm-relu', memory_efficient=True)
        sd = torch.load('../model/campplus/campplus_cn_common.pt', map_location='cpu')
        M['campplus'].load_state_dict({k: v.float() for k, v in sd.items()})
        M['campplus'] = M['campplus'].eval().to(args.device)
        M['mel_fn'] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            n_mels=80, f_min=20, f_max=7600, norm='slaney', mel_scale='slaney',
        ).to(args.device)
        print('CAM++ loaded')
    except Exception as e:
        M['campplus'], M['mel_fn'] = None, None
        print(f'CAM++ load failed: {e}')
    M['vad_path'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'vad', 'silero_vad.onnx')
    spk_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'speaker')
    for fn, group in [('voices.pt', 'builtin'), ('voices_unseen.pt', 'unseen'), (CLONE_FILE, 'manual')]:
        fp = os.path.join(spk_dir, fn)
        if os.path.exists(fp):
            for speaker, v in torch.load(fp, map_location=args.device).items():
                if speaker not in V or fn == CLONE_FILE:
                    register_voice(speaker, v, group=group)
    if V: print(f'Loaded {len(V)} voices: builtin={sorted(V_builtin)}, unseen={sorted(V_unseen)}, manual={sorted(V_manual)}')
    log_model_params(M['model'])
    print('Warmup...')
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]], device=args.device)
        au = torch.full((1, 8, 3), 2049, dtype=torch.long, device=args.device)
        M['model'].forward(torch.cat((au, ids.unsqueeze(1)), dim=1))
        if M['model'].audio_encoder: M['model'].audio_encoder(torch.zeros(1, 100, 560, device=args.device), torch.tensor([100], device=args.device))
        if M['mimi']: M['mimi'].decode(torch.zeros(1, 8, 1, dtype=torch.long, device=args.device))
    print('Warmup done!')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='../out/sft_omni_768_translate_expriment2_5.pth', help='翻译模型的原生 torch 权重路径（相对 webui/ 目录）。')
    p.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度，需与权重一致。')
    p.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量，需与权重一致。')
    p.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用 MoE 架构，需与权重一致。')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='推理设备；CUDA 可用时默认 cuda。显存不足或排查环境问题时可改为 cpu。')
    p.add_argument('--port', default=7860, type=int, help='WebUI 服务端口；端口被占用或需要同时启动多个实例时调整。')
    p.add_argument('--audio_chunk_frames', default=4, type=int, help='流式播放每次解码的 Mimi frame 数；默认 4 约 320ms。WebUI 播放卡顿时可调大到 8/12，低延迟优先时保持 4。')
    p.add_argument('--audio_overlap', default=2, type=int, help='分块 Mimi 解码的重叠帧数；默认 2 用于缓解块边界断裂。一般不需要调整，边界杂音明显时可适当增大。')
    p.add_argument('--max_history_turns', default=0, type=int, help='对话历史轮数；默认 0 不带历史以降低延迟和显存。翻译通常单轮即可。')
    p.add_argument('--vad_silence_ms', default=800, type=int, help='VAD 判定一段语音结束所需的静音时长(ms)；越小切段越快、片段越短。默认 800。')
    args = p.parse_args()
    init_model(args)
    app.run(host='0.0.0.0', port=args.port, threaded=True)
