"""实时句级同传服务：S4 离线整句 S2ST 常驻进程（`inference/infer_s2st.S2STPipeline` 的薄包装）。

输入约定：**每个请求是一段已经切好的语音**（切句由上游负责），服务不做 VAD、不改推理逻辑。

接口：
    GET  /health          就绪状态、设备、模型目录
    POST /v1/translate    请求体 = wav 文件字节；query `tgt_lang`（zh / en）
                          -> JSON：译文、逐段耗时、24kHz 语音（base64 PCM16）
    WS   /v1/stream       先发 JSON {"tgt_lang": "zh"}，之后每条二进制消息 = 一段 wav；
                          每段回两条消息：JSON 头（seq / 译文 / 耗时）+ 二进制 24kHz PCM16

GPU 侧只有一个执行线程：`S2STPipeline` 的 talker CUDA Graph 与静态缓存绑定在模型实例上，
同一时刻只能有一个状态在用（见 `model_si.TalkerStreamState` 注释），所以所有请求串行排队。

用法（容器内）：
    PYTHONPATH=custom python -m streaming_si.serve.server \
        --model_dir /fsx/.../s4_ck --cosyvoice_repo /opt/CosyVoice --port 8000
本地无 GPU 时可用 --fake_pipeline 只验证协议与队列（返回固定译文 + 1s 正弦波）。
"""
import argparse
import asyncio
import base64
import io
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from typing import Any, Optional, Tuple

import numpy as np

SR = 16000
OUT_SR = 24000
SPEECH_TOKENS_PER_SEC = 25.0          # talker 语音 token 帧率（与 infer_s5_streaming 的 speech_sec 口径一致）
ZH_LANGS = ('zh', 'cmn', 'zho', 'yue')


# ---------------------------------------------------------------------------
# 推理引擎：模型只加载一次，GPU 调用全部在单线程里串行执行
# ---------------------------------------------------------------------------
class Engine:
    """把 `S2STPipeline.translate` 包成「提交 -> Future」的单线程执行器。"""

    def __init__(self, pipeline, *, device='cuda:0', max_audio_sec=30.0,
                 retry=1, seed=0, log=print):
        self.pipe = pipeline
        self.device = device
        self.max_audio_sec = max_audio_sec
        self.retry = int(retry)
        self.seed = int(seed)
        self.log = log
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name='s2st-engine', daemon=True)
        self._thread.start()
        self.stats = dict(done=0, failed=0, total_queue_ms=0.0, total_infer_ms=0.0)

    # -- 对外 ---------------------------------------------------------------
    def submit(self, wav16: np.ndarray, tgt_lang: str) -> Tuple[Future, float]:
        """提交一段 16kHz 单声道 float32，返回 (Future, 入队时刻)。"""
        fut: Future = Future()
        t_enq = time.time()
        self._q.put((fut, wav16, tgt_lang, t_enq))
        return fut, t_enq

    def queue_depth(self) -> int:
        return self._q.qsize()

    def shutdown(self):
        self._q.put(None)

    # -- 线程体 -------------------------------------------------------------
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, wav16, tgt_lang, t_enq = item
            if fut.set_running_or_notify_cancel() is False:
                continue
            t_start = time.time()
            try:
                res = self._translate(wav16, tgt_lang)
            except Exception as exc:                       # 单段失败不影响服务
                self.stats['failed'] += 1
                fut.set_exception(exc)
                continue
            t_end = time.time()
            self.stats['done'] += 1
            self.stats['total_queue_ms'] += (t_start - t_enq) * 1000
            self.stats['total_infer_ms'] += (t_end - t_start) * 1000
            fut.set_result(dict(res=res, t_enq=t_enq, t_start=t_start, t_end=t_end))

    def _translate(self, wav16: np.ndarray, tgt_lang: str) -> dict:
        """调用 S2STPipeline；语音塌缩（talker 复读/静音）时换 seed 重试。

        判据与 `infer_s5_streaming.collapse_ok` 相同：码字种类占比低 且 语音明显长于源音频。
        """
        import torch
        best = None
        for attempt in range(self.retry + 1):
            torch.manual_seed(self.seed + attempt)        # talker 是采样生成，逐段固定 seed
            np.random.seed(self.seed + attempt)
            res = self.pipe.translate(wav16, tgt_lang)
            res['speech_sec'] = len(res.get('speech_tokens') or []) / SPEECH_TOKENS_PER_SEC
            best = res
            if not _collapse_ok(res):                      # 空语音或塌缩才重试（_collapse_ok=True 表示正常）
                if attempt < self.retry:
                    self.log(f'[engine] 语音塌缩/为空，换 seed 重试（{attempt + 1}/{self.retry}）')
                    continue
            break
        return best


def _collapse_ok(res: dict) -> bool:
    """True = 生成正常。判据取自 docs/talker_collapse.md（与 infer_s5_streaming 一致）。"""
    toks = res.get('speech_tokens') or []
    if not toks:
        return False
    uniq = len(set(toks)) / len(toks)
    long = res.get('speech_sec', 0.0) > 1.6 * res.get('source_sec', 0.0) + 1.0
    return not (uniq < 0.3 and long)


class FakeEngine:
    """无 GPU 时验证协议用：固定译文 + 1s 正弦波，按 RTF 0.3 模拟耗时。"""

    def __init__(self, *, rtf=0.3, log=print):
        self.rtf, self.log = rtf, log
        self.stats = dict(done=0, failed=0, total_queue_ms=0.0, total_infer_ms=0.0)

    def submit(self, wav16, tgt_lang):
        fut: Future = Future()
        t_enq = time.time()

        def _job():
            t_start = time.time()                     # 与 Engine 一致：t_start 记在「开始算」的时刻
            time.sleep(max(0.05, len(wav16) / SR * self.rtf))
            t = np.arange(int(OUT_SR * 1.0)) / OUT_SR
            wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            fut.set_result(dict(res=dict(text=f'[fake {tgt_lang}] {len(wav16) / SR:.1f}s',
                                         speech_tokens=[1, 2, 3], speech_sec=1.0,
                                         source_sec=len(wav16) / SR, wav=wav),
                                 t_enq=t_enq, t_start=t_start, t_end=time.time()))

        threading.Thread(target=_job, daemon=True).start()
        return fut, t_enq

    def queue_depth(self):
        return 0

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# 音频编解码
# ---------------------------------------------------------------------------
def wav_bytes_to_wav16(data: bytes) -> np.ndarray:
    """wav 文件字节 -> 16kHz 单声道 float32（任意采样率 / 声道数）。"""
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def wav_to_pcm16(wav: np.ndarray, sr: int = OUT_SR) -> bytes:
    return (np.clip(wav.reshape(-1), -1.0, 1.0) * 32767.0).astype('<i2').tobytes()


# ---------------------------------------------------------------------------
# HTTP / WebSocket
# ---------------------------------------------------------------------------
def create_app(holder, *, model_dir: str = '', device: str = ''):
    """`holder` 是 `/health` 与各接口共用的可变状态：

        engine      推理引擎（模型加载完成前为 None）
        ready       True = 可以接请求
        error       加载失败时的异常字符串
        load_sec    模型加载耗时
        warmup_sec  预热耗时

    模型在后台线程里加载，HTTP 端口先起来：调用方在加载期间拿到 503 而不是连接被拒，
    port-forward / readiness 探针都不会因为「端口还没开」而失败。
    """
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    app = FastAPI(title='streaming_si realtime s2st')
    state: dict[str, Any] = dict(model_dir=model_dir, device=device, started_at=time.time())

    def engine_or_503():
        eng = holder.get('engine')
        if eng is None:
            why = holder.get('error')
            msg = f'模型加载失败: {why}' if why else '模型加载中，请稍候'
            return None, JSONResponse({'error': msg, 'ready': False}, status_code=503)
        return eng, None

    def _result_payload(item, seq: int) -> dict:
        res, t_enq, t_start, t_end = item['res'], item['t_enq'], item['t_start'], item['t_end']
        return dict(
            seq=seq, text=res.get('text', ''), status=res.get('status'),
            source_sec=round(float(res.get('source_sec') or 0.0), 3),
            speech_sec=round(float(res.get('speech_sec') or 0.0), 3),
            speech_stop=res.get('speech_stop'), text_stop=res.get('text_stop'),
            truncated=bool(res.get('audio_truncated')),
            queue_ms=round((t_start - t_enq) * 1000, 1),
            infer_ms=round((t_end - t_start) * 1000, 1),
            sample_rate=OUT_SR, audio_sec=None)

    @app.get('/')
    def index():
        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html'))

    @app.get('/health')
    def health():
        eng = holder.get('engine')
        return dict(ready=bool(holder.get('ready')), model_dir=state['model_dir'],
                    device=state['device'], error=holder.get('error'),
                    uptime_sec=round(time.time() - state['started_at'], 1),
                    load_sec=holder.get('load_sec'), warmup_sec=holder.get('warmup_sec'),
                    queue_depth=eng.queue_depth() if eng is not None else None,
                    stats=dict(eng.stats) if eng is not None else None)

    @app.post('/v1/translate')
    async def translate(request: Request, tgt_lang: str = 'zh'):
        engine, err = engine_or_503()
        if err is not None:
            return err
        data = await request.body()
        if not data:
            return JSONResponse({'error': 'empty body; send wav file bytes'}, status_code=400)
        try:
            wav16 = wav_bytes_to_wav16(data)
        except Exception as exc:
            return JSONResponse({'error': f'could not decode wav: {exc}'}, status_code=400)
        fut, t_enq = engine.submit(wav16, tgt_lang)
        item = await asyncio.wrap_future(fut)
        t_done = time.time()
        payload = _result_payload(item, 0)
        wav = item['res'].get('wav')
        if wav is not None:
            payload['audio_b64'] = base64.b64encode(wav_to_pcm16(wav)).decode('ascii')
            payload['audio_sec'] = round(float(len(wav)) / OUT_SR, 3)
        payload['e2e_ms'] = round((t_done - t_enq) * 1000, 1)
        return JSONResponse(payload)

    @app.websocket('/v1/stream')
    async def stream(ws: WebSocket):
        """按到达顺序收段、按 seq 顺序回结果。

        每条二进制消息是一段已切好的 wav。服务端回：
          * `accepted`（收到即回，带 recv_ms / queue_depth）
          * `result`（JSON 头）+ 一条二进制 PCM16（24kHz），按 seq 顺序
        文本消息：`{"tgt_lang": "zh"}` 配置；`{"type": "done"}` 收尾——等在途结果全部发完再回 `done`。
        """
        import json
        await ws.accept()
        engine = holder.get('engine')
        if engine is None:
            await ws.send_json(dict(type='error', error=holder.get('error') or '模型加载中，请稍候'))
            await ws.close()
            return
        st = dict(tgt_lang='zh', seq=0, submitted=0, sent=0)
        out: asyncio.Queue = asyncio.Queue()

        async def drain_forever():
            """按 seq 顺序取结果并发出去（串行引擎天然有序）。"""
            while True:
                fut, i = await out.get()
                try:
                    item = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=1800)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await ws.send_json(dict(seq=i, error=f'{type(exc).__name__}: {exc}'))
                else:
                    payload = _result_payload(item, i)
                    wav = item['res'].get('wav')
                    if wav is not None:
                        payload['audio_sec'] = round(float(len(wav)) / OUT_SR, 3)
                    await ws.send_json(payload)
                    if wav is not None:
                        await ws.send_bytes(wav_to_pcm16(wav))
                finally:
                    st['sent'] += 1

        await ws.send_json(dict(type='ready', tgt_lang=st['tgt_lang'],
                                sample_rate=OUT_SR, in_rate=SR))
        writer = asyncio.create_task(drain_forever())

        async def wait_all(timeout: float = 1800.0):
            deadline = time.time() + timeout
            while st['sent'] < st['submitted']:
                if time.time() > deadline:
                    return False
                await asyncio.sleep(0.02)
            return True

        try:
            while True:
                msg = await ws.receive()
                if msg['type'] == 'websocket.disconnect':
                    break
                if msg.get('text') is not None:
                    cfg = json.loads(msg['text'])
                    kind = cfg.get('type')
                    if kind in (None, 'config', 'start'):
                        st['tgt_lang'] = cfg.get('tgt_lang', st['tgt_lang'])
                        await ws.send_json(dict(type='ready', tgt_lang=st['tgt_lang'],
                                                sample_rate=OUT_SR, in_rate=SR))
                    elif kind == 'done':
                        ok = await wait_all()
                        await ws.send_json(dict(type='done', sent=st['sent'],
                                                submitted=st['submitted'], complete=ok))
                        break
                    continue
                t_recv = time.time()
                data = msg.get('bytes') or b''
                try:
                    wav16 = wav_bytes_to_wav16(data)
                except Exception as exc:
                    await ws.send_json(dict(seq=st['seq'], error=f'could not decode wav: {exc}'))
                    st['seq'] += 1
                    continue
                fut, _ = engine.submit(wav16, st['tgt_lang'])
                await out.put((fut, st['seq']))
                st['submitted'] += 1
                await ws.send_json(dict(type='accepted', seq=st['seq'],
                                        recv_ms=round((time.time() - t_recv) * 1000, 1),
                                        source_sec=round(len(wav16) / SR, 3),
                                        queue_depth=engine.queue_depth()))
                st['seq'] += 1
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json(dict(seq=-1, error=f'{type(exc).__name__}: {exc}'))
            except Exception:
                pass
        finally:
            writer.cancel()

    return app


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def build_pipeline(args):
    from streaming_si.inference.infer_s2st import S2STPipeline
    return S2STPipeline(
        args.model_dir, device=args.device, dtype=args.dtype, vocoder=not args.no_vocoder,
        cosyvoice_dir=args.cosyvoice_dir, cosyvoice_repo=args.cosyvoice_repo,
        cosyvoice_config_yaml=args.cosyvoice_config_yaml, speaker=not args.no_speaker,
        max_length=args.max_length, max_new_tokens=args.max_new_tokens,
        max_speech_tokens=args.max_speech_tokens, max_audio_sec=args.max_audio_sec,
        temperature=args.temperature, top_p=args.top_p, talker_top_k=args.talker_top_k,
        talker_top_p=args.talker_top_p, ode_steps=args.ode_steps, text_only=args.text_only,
        source_prompt=args.source_prompt, prompt_sec=args.prompt_sec,
        prompt_position=args.prompt_position, prompt_tokenizer=args.prompt_tokenizer)


def warmup(engine, seconds: float = 3.0, tgt_lang: str = 'zh', log=print):
    """跑一遍完整链路，把 thinker / talker 的 CUDA Graph 录好，避免首个真实请求承担预热。"""
    t0 = time.time()
    for _ in range(2):
        fut, _ = engine.submit(np.zeros(int(SR * seconds), dtype=np.float32), tgt_lang)
        fut.result(timeout=600)
    log(f'[serve] warmup 完成，用时 {time.time() - t0:.1f}s')


def parse_args(argv=None):
    from streaming_si.inference.infer_s2st import add_source_prompt_args

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', default='', help='S4 checkpoint 目录（S1-S4 完整模型目录）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no_vocoder', action='store_true')
    ap.add_argument('--text_only', action='store_true')
    ap.add_argument('--no_speaker', action='store_true')
    ap.add_argument('--cosyvoice_dir', default=None)
    ap.add_argument('--cosyvoice_repo', default=None)
    ap.add_argument('--cosyvoice_config_yaml', default=None)
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--max_speech_tokens', type=int, default=750)
    ap.add_argument('--max_audio_sec', type=float, default=30.0)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.0)
    ap.add_argument('--talker_top_k', type=int, default=25)
    ap.add_argument('--talker_top_p', type=float, default=0.8)
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--retry', type=int, default=1, help='语音塌缩时的重试次数（换 seed）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_warmup', action='store_true')
    ap.add_argument('--warmup_sec', type=float, default=3.0)
    ap.add_argument('--fake_pipeline', action='store_true', help='无 GPU 时验证协议用')
    ap.add_argument('--fake_load_sec', type=float, default=0.0,
                    help='配合 --fake_pipeline：模拟模型加载耗时，用来验证加载期间 /health 返回 503')
    add_source_prompt_args(ap)
    return ap.parse_args(argv)


def load_into(holder, args):
    """后台线程：加载模型 -> 建引擎 -> 预热，最后把 holder 置为 ready。"""
    try:
        if args.fake_pipeline:
            if args.fake_load_sec > 0:
                print(f'[serve] fake：模拟加载 {args.fake_load_sec}s', flush=True)
                time.sleep(args.fake_load_sec)
            holder['engine'] = FakeEngine()
            holder['load_sec'] = float(args.fake_load_sec)
            holder['warmup_sec'] = 0.0
            holder['ready'] = True
            print('[serve] fake pipeline 就绪', flush=True)
            return
        t0 = time.time()
        pipe = build_pipeline(args)
        holder['load_sec'] = round(time.time() - t0, 1)
        print(f'[serve] 模型加载完成，用时 {holder["load_sec"]}s', flush=True)
        eng = Engine(pipe, device=args.device, max_audio_sec=args.max_audio_sec,
                     retry=args.retry, seed=args.seed)
        holder['engine'] = eng                     # 先挂上：/health 能看到进度与统计
        if not args.no_warmup:
            t0 = time.time()
            warmup(eng, seconds=args.warmup_sec)
            holder['warmup_sec'] = round(time.time() - t0, 1)
            print(f'[serve] 预热完成（{holder["warmup_sec"]}s）', flush=True)
        holder['ready'] = True
        print('[serve] 可以接请求了', flush=True)
    except BaseException as exc:                   # noqa: BLE001 - 失败要暴露给 /health
        import traceback
        holder['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        print(f'[serve][FATAL] 模型加载失败：{holder["error"]}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    if not args.fake_pipeline and not args.model_dir:
        raise SystemExit('需要 --model_dir（或用 --fake_pipeline 只验证协议）')

    holder: dict[str, Any] = dict(engine=None, ready=False, error=None,
                                 load_sec=None, warmup_sec=None)
    app = create_app(holder, model_dir=args.model_dir, device=args.device)

    import threading
    threading.Thread(target=load_into, args=(holder, args), name='s2st-load', daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', ws='websockets')


if __name__ == '__main__':
    main()
