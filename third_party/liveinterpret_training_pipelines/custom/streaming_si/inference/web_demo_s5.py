"""S5 Web transport: true streaming S5Session.

Run with PYTHONPATH=custom python -m streaming_si.inference.web_demo_s5 --help.
Microphone access requires localhost or an HTTPS reverse proxy (with WS support).
Dependencies: Flask, flask-sock, soundfile, scipy, plus the S5 model environment.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

SR = 16000
CHUNK_CHOICES_MS = (640, 1280, 1920, 2560)
TARGET_LANGUAGES = {'en': 'English', 'zh': '中文', 'ja': '日本語', 'ko': '한국어'}
WIRE_TYPES = {'session', 'text', 'pcm', 'decision', 'context', 'end', 'metrics', 'error', 'seg_gate', 'segment_end', 'decoder_conditions'}


def request_chunks(fields, config):
    """Validate per-request settings without mutating model/server defaults."""
    result = {}
    for key in ('chunk_ms', 'chunk_ms_after_write'):
        value = fields.get(key, config.get(key, 640 if key == 'chunk_ms' else None))
        if key == 'chunk_ms_after_write' and value in (None, '', 0, '0') and not isinstance(value, bool):
            result[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f'{key} must be one of {CHUNK_CHOICES_MS}')
        try:
            size = int(value)
        except ValueError:
            raise ValueError(f'{key} must be one of {CHUNK_CHOICES_MS}') from None
        if size not in CHUNK_CHOICES_MS:
            raise ValueError(f'{key} must be one of {CHUNK_CHOICES_MS}')
        result[key] = size
    return result


def serialize_event(event):
    """Transmit PCM when it is produced, without synthetic token-based timing."""
    payload = dict(event)
    if payload.get('type') == 'pcm':
        pcm = np.asarray(payload.pop('pcm'), dtype=np.float32).reshape(-1)
        payload['data'] = base64.b64encode((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes()).decode('ascii')
        payload['encoding'] = 's16le'
    return payload


def decode_upload(raw):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        factor = gcd(rate, SR)
        audio = resample_poly(audio, SR // factor, rate // factor).astype(np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('input must contain finite audio samples')
    return audio


class OriginalPageEvents:
    """Translate actual model events to the unmodified original page protocol."""
    def __init__(self):
        self.text = ''
        self.first_text = self.first_pcm = None

    def convert(self, event):
        kind = event['type']
        # The original client starts capture on WS open, before session/Graph
        # preparation. Use the request clock so that queued input and warmup
        # are included instead of starting at the first later ws.receive().
        elapsed = event.get('server_request_sec', event.get('elapsed_sec', 0))
        if kind == 'decision':
            action = event.get('action', event.get('decision'))
            if action in ('write', 'wait'):
                return [dict(type='decision', action=action,
                             source_step=event.get('step', event.get('source_step', 0)),
                             source_sec=event.get('source_sec'),
                             model_source_sec=event.get('model_source_sec'),
                             n_audio_frames=event.get('n_audio_frames'))]
        elif kind == 'text':
            self.text += event['text']
            output = [dict(type='text', text=event['text'])]
            if self.first_text is None:
                self.first_text = elapsed
                output.append(dict(type='stat', text_ttft=round(elapsed * 1000, 1)))
            return output
        elif kind == 'pcm':
            output = [dict(type='audio', pcm=event['data'], sr=event['sample_rate'],
                           sample_offset=event.get('sample_offset'), source_sec=event.get('source_sec'))]
            if self.first_pcm is None:
                self.first_pcm = elapsed
                output.append(dict(type='stat', audio_ttft=round(elapsed * 1000, 1)))
            return output
        elif kind == 'metrics':
            return [dict(type='done', text=self.text, source_sec=event.get('source_sec'),
                         output_sec=event.get('output_sec'), rtf=event.get('inference_rtf'),
                         metrics=event)]
        elif kind == 'error':
            return [dict(type='error', message=event.get('error', 'inference failed'))]
        return []


def create_app(session_factory, config, *, enable_websocket=True):
    """factory(target_lang, on_event) must create a fresh, exclusively owned session."""
    from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024
    model_lock = threading.Lock()

    def default_version():
        return 'streaming'

    def version_available(version):
        return version == 'streaming'

    @app.get('/')
    @app.get('/web_demo_s5.html')
    def index():
        return send_from_directory(Path(__file__).parent, 'web_demo_s5.html')

    @app.get('/config')
    def get_config():
        return jsonify(dict(config, default_version=default_version(), src_sr=SR, out_sr=24000,
                            chunk_choices_ms=list(CHUNK_CHOICES_MS),
                            langs=[dict(code=code, label=label) for code, label in TARGET_LANGUAGES.items()]))

    @app.post('/translate')
    def translate():
        original_page = request.is_json
        fields = request.get_json() if original_page else request.form
        if not hasattr(fields, 'get'):
            return jsonify(error='request must be an object'), 400

        def reject(message, code=400):
            if original_page:
                payload = json.dumps(dict(type='error', message=message), ensure_ascii=False)
                return Response('data: ' + payload + '\n\n', status=code, mimetype='text/event-stream')
            return jsonify(error=message), code

        target = fields.get('target_lang' if original_page else 'tgt_lang', 'zh')
        mode = fields.get('mode', 'accelerated')
        version = fields.get('version', default_version())
        if target not in TARGET_LANGUAGES or mode not in ('accelerated', 'realtime'):
            return reject('invalid target language or mode')
        if not version_available(version):
            return reject('invalid or unavailable version')
        try:
            chunks = request_chunks(fields, config)
        except ValueError as exc:
            return reject(str(exc))
        upload = None if original_page else request.files.get('audio')
        if (not original_page and upload is None) or (original_page and not fields.get('audio')):
            return reject('audio file required')
        request_started = time.perf_counter()
        try:
            raw = base64.b64decode(fields['audio'], validate=True) if original_page else upload.read()
            audio = decode_upload(raw)
        except Exception as exc:
            return reject(f'cannot decode audio: {exc}')
        if not model_lock.acquire(blocking=False):
            return reject('GPU busy: another session owns the model', 409)
        events = queue.Queue(maxsize=32)
        cancelled = threading.Event()
        original_events = OriginalPageEvents() if original_page else None

        def emit(event):
            if event['type'] not in WIRE_TYPES:
                return
            item = serialize_event(event)
            item['server_request_sec'] = time.perf_counter() - request_started
            while not cancelled.is_set():
                try:
                    events.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass
            raise ConnectionAbortedError('client disconnected')

        def worker():
            try:
                session = session_factory(target, emit, chunk_options=chunks) if config.get('request_chunk_config') else session_factory(target, emit)
                started = time.perf_counter()
                # Transport granularity is independent of the model's input chunk policy.
                for start in range(0, audio.size, 1280):
                    if cancelled.is_set():
                        return
                    block = audio[start:start + 1280]
                    if mode == 'realtime':
                        deadline = started + (start + block.size) / SR
                        if cancelled.wait(max(0, deadline - time.perf_counter())):
                            return
                    session.feed_audio(block)
                session.feed_audio(np.zeros(0, np.float32), finalize=True)
                emit(dict(session.metrics(), type='metrics', **chunks))
            except ConnectionAbortedError:
                pass
            except Exception as exc:
                try:
                    emit(dict(type='error', error=f'{type(exc).__name__}: {exc}'))
                except ConnectionAbortedError:
                    pass
            finally:
                model_lock.release()
                while not cancelled.is_set():
                    try:
                        events.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        pass

        threading.Thread(target=worker, daemon=True).start()

        @stream_with_context
        def stream():
            try:
                while True:
                    event = events.get()
                    if event is None:
                        return
                    if original_page:
                        for payload in original_events.convert(event):
                            yield 'data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n'
                    else:
                        yield json.dumps(event, ensure_ascii=False) + '\n'
            finally:
                cancelled.set()
        return Response(stream(), mimetype='text/event-stream' if original_page else 'application/x-ndjson',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    if enable_websocket:
        try:
            from flask_sock import Sock
        except ImportError as exc:
            raise RuntimeError('microphone mode requires pip install flask-sock') from exc
        sock = Sock(app)

        @sock.route('/stream')
        @sock.route('/ws/simul')
        def microphone(ws):
            original_page = request.path == '/ws/simul'
            original_events = OriginalPageEvents() if original_page else None
            def send(payload):
                messages = original_events.convert(payload) if original_page else [payload]
                for message in messages:
                    ws.send(json.dumps(message, ensure_ascii=False))
            if not model_lock.acquire(blocking=False):
                send(dict(type='error', error='GPU busy: another session owns the model'))
                return
            request_started = time.perf_counter()
            try:
                setup = json.loads(ws.receive(timeout=15))
                target = setup.get('target_lang' if original_page else 'tgt_lang', 'zh')
                version = setup.get('version', default_version())
                if target not in TARGET_LANGUAGES or (not original_page and setup.get('sample_rate') != SR):
                    raise ValueError('microphone input must be mono s16le 16000Hz, tgt_lang en/zh/ja/ko')
                if not version_available(version):
                    raise ValueError('invalid or unavailable version')
                chunks = request_chunks(setup, config)
                first_input = None
                def emit(event):
                    if event['type'] in WIRE_TYPES:
                        payload = serialize_event(event)
                        payload['server_request_sec'] = time.perf_counter() - request_started
                        payload['server_since_first_input_sec'] = None if first_input is None else time.perf_counter() - first_input
                        send(payload)
                session = session_factory(target, emit, chunk_options=chunks) if config.get('request_chunk_config') else session_factory(target, emit)
                if not original_page:
                    ws.send(json.dumps(dict(type='ready', config=config, version=version)))
                while True:
                    message = ws.receive()
                    if message is None:
                        return
                    if isinstance(message, str):
                        command = json.loads(message)
                        if original_page and command.get('type') == 'end':
                            return
                        if command.get('type') != ('stop' if original_page else 'finish'):
                            raise ValueError('unknown microphone command')
                        session.feed_audio(np.zeros(0, np.float32), finalize=True)
                        send(dict(session.metrics(), type='metrics', **chunks))
                        # Let the browser acknowledge the last application
                        # message before the WSGI handler closes its socket.
                        # A bounded wait also releases the model on disconnect.
                        from simple_websocket import ConnectionClosed
                        try:
                            acknowledgment = ws.receive(timeout=5)
                        except ConnectionClosed as closed:
                            if closed.reason == 1000:
                                return
                            raise
                        if acknowledgment is not None and (
                                not isinstance(acknowledgment, str) or
                                json.loads(acknowledgment).get('type') != ('end' if original_page else 'ack')):
                            raise ValueError('expected completion acknowledgment')
                        return
                    if len(message) % 2 or len(message) > 2 * SR:
                        raise ValueError('PCM packet must be s16le and at most one second')
                    if first_input is None:
                        first_input = time.perf_counter()
                    session.feed_audio(np.frombuffer(message, '<i2').astype(np.float32) / 32768)
            except Exception as exc:
                app.logger.exception('S5 microphone session failed')
                try:
                    send(dict(type='error', error=f'{type(exc).__name__}: {exc}'))
                except Exception:
                    pass
            finally:
                model_lock.release()
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--cosyvoice_repo', default=os.environ.get('COSYVOICE_REPO'), required=not bool(os.environ.get('COSYVOICE_REPO')))
    p.add_argument('--cosyvoice_dir')
    p.add_argument('--prompt_tokenizer')
    p.add_argument('--no_source_prompt', action='store_true')
    p.add_argument('--no_speaker', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float16', 'float32'))
    p.add_argument('--chunk_ms', type=int, choices=CHUNK_CHOICES_MS, default=640)
    p.add_argument('--chunk_ms_after_write', type=int, choices=CHUNK_CHOICES_MS)
    p.add_argument('--context_max_tokens', type=int, default=2048)
    p.add_argument('--context_window_sec', type=float, default=0)
    p.add_argument('--context_window_32_16', action='store_true',
                   help='at 32 chunk decisions replay the latest 16 turns; WAIT counts')
    p.add_argument('--acoustic_max_sec', type=float, default=60)
    p.add_argument('--reference_window_sec', type=float, default=5)
    p.add_argument('--reference_policy', choices=('rolling', 'fixed'), default='rolling')
    p.add_argument('--reference_refresh_sec', type=float, default=2)
    p.add_argument('--prompt_sec', type=float, default=2)
    p.add_argument('--write_threshold', type=float)
    p.add_argument('--ode_steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_speech_tokens', type=int, default=750)
    p.add_argument('--max_text_tokens', type=int, default=256)
    p.add_argument('--max_tail_steps', type=int, default=8)
    p.add_argument('--talker_top_k', type=int, default=25)
    p.add_argument('--talker_top_p', type=float, default=.8)
    p.add_argument('--no_thinker_cache', action='store_true')
    p.add_argument('--no_thinker_graph', action='store_true')
    p.add_argument('--no_talker_graph', action='store_true')
    from streaming_si.inference.s5_segmenter import add_segmentation_args
    add_segmentation_args(p)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=7865)
    p.add_argument('--certfile', help='TLS certificate for browser microphone access over HTTPS')
    p.add_argument('--keyfile', help='TLS private key')
    p.add_argument('--server', choices=('gunicorn', 'development'), default='gunicorn',
                   help='one GPU-owning Gunicorn worker; development is diagnostic only')
    args = p.parse_args(argv)
    if args.context_window_32_16 and args.context_window_sec:
        p.error('--context_window_32_16 is exclusive with --context_window_sec')
    if not np.isfinite(args.reference_refresh_sec) or args.reference_refresh_sec <= 0:
        p.error('reference_refresh_sec must be finite and positive')
    if not 0 < args.prompt_sec <= args.reference_window_sec:
        p.error('require 0 < prompt_sec <= reference_window_sec')
    if bool(args.certfile) != bool(args.keyfile):
        p.error('--certfile and --keyfile must be supplied together')
    from streaming_si.inference.s5_segmenter import segmentation_options
    try:
        segmentation_options(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def build_app(args):
    import torch
    from streaming_si.inference.infer_s5_streaming_dynchunk import load_model, VoxtralAudioProcessor, VoxtralFeatureExtractor
    from streaming_si.inference.infer_s2st import CampplusSpeaker, build_source_prompt
    from streaming_si.model.speech_decoder import resolve_speech_decoder
    from streaming_si.inference.s5_session import S5Session
    from streaming_si.inference.s5_segmenter import SegmentedSession, SileroFrameVad, segmentation_options
    startup = time.perf_counter()
    torch.manual_seed(args.seed)
    model, tokenizer, ids = load_model(args.model_dir, args.adapter, dtype=args.dtype)
    model.to(args.device)
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model_dir))
    directory, yaml = resolve_speech_decoder(args.model_dir, args.cosyvoice_dir, None)
    model.config.talker_config_yaml = yaml
    model.attach_speech_decoder(directory, device=args.device, cosyvoice_repo=args.cosyvoice_repo, strict=True)
    speaker = None if args.no_speaker else CampplusSpeaker(os.path.join(directory, 'campplus.onnx'))
    prompter = build_source_prompt(not args.no_source_prompt, directory, args.prompt_tokenizer)
    settings = {key: getattr(args, key) for key in ('device', 'chunk_ms', 'chunk_ms_after_write', 'context_max_tokens', 'context_window_sec', 'acoustic_max_sec', 'reference_window_sec', 'reference_policy', 'reference_refresh_sec', 'prompt_sec', 'write_threshold', 'ode_steps', 'max_speech_tokens', 'max_text_tokens', 'max_tail_steps', 'talker_top_k', 'talker_top_p')}
    settings['context_window_32_16'] = args.context_window_32_16
    settings.update(thinker_cache=not args.no_thinker_cache, thinker_graph=not args.no_thinker_graph, talker_graph=not args.no_talker_graph)
    vad = SileroFrameVad(args.vad_threshold) if args.segmentation == 'vad' else None
    segment_settings = segmentation_options(args)
    session_count = 0
    def factory(target, on_event, *, chunk_options=None):
        nonlocal session_count
        # The transport holds the model lock before entering this factory.
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        session = S5Session(model, tokenizer, ids, mel_fn, target, speaker=speaker, prompter=prompter,
                            on_event=None if vad is not None else on_event, collect_audio=False,
                            **dict(settings, **(chunk_options or {})))
        if args.segmentation != 'none':
            if vad is not None:
                vad.reset()
            session = SegmentedSession(session, vad, on_event=on_event, **segment_settings)
        # Graph preparation must not consume the generation RNG differently on
        # the first request versus a subsequent request using captured graphs.
        torch.manual_seed(args.seed)
        session_count += 1
        on_event(dict(type='session', version='streaming', session_index=session_count, seed=args.seed,
                      session_init_sec=time.perf_counter() - started,
                      graph_warmup_guaranteed=False))
        return session
    config = dict(model_dir=args.model_dir, adapter=args.adapter, seed=args.seed, dtype=args.dtype,
                  runtime_host=os.environ.get('HOSTNAME'), gpu_name=torch.cuda.get_device_name(args.device),
                  default_version='streaming', enabled_versions=['streaming'], request_chunk_config=True,
                  source_prompt_enabled=prompter is not None, speaker_enabled=speaker is not None,
                  synthesis_mode='streaming', retry=0,
                  segmentation=args.segmentation,
                  vad_threshold=args.vad_threshold, segmentation_settings=segment_settings,
                  startup_sec=time.perf_counter() - startup, **settings)
    config['timing'] = {
        'elapsed_sec': 'session wall time, including session construction/graph preparation; excludes model loading',
        'server_request_sec': 'HTTP/WS handler wall time, including upload decoding/session construction; excludes model loading and upload bytes sent before handler start',
        'source_sec': 'VAD mode: observed original source position including skipped silence; not wall TTFT',
        'model_source_sec': 'compressed model input counter, mapped separately to model_source_origin_sec',
        'inference_rtf': 'active input pipeline including VAD / full original input duration; excludes input pacing waits',
        'browser': 'click-to-text/PCM receipt and scheduled playback; includes transport waits',
        'warmup': 'session_index=1 may include graph preparation; later requests are not guaranteed fully warm for all shapes',
    }
    return create_app(factory, config)


def main(argv=None):
    args = parse_args(argv)
    if args.server == 'development':
        build_app(args).run(host=args.host, port=args.port, threaded=True, debug=False)
        return
    from gunicorn.app.base import BaseApplication

    class Application(BaseApplication):
        def load_config(self):
            # Build the CUDA model in the worker, after fork. Never preload it
            # in the master or create a replica for each HTTP thread.
            options = dict(bind=f'{args.host}:{args.port}', workers=1,
                                   worker_class='gthread', threads=4, timeout=0,
                                   preload_app=False, accesslog='-')
            if args.certfile:
                options.update(certfile=args.certfile, keyfile=args.keyfile)
            for key, value in options.items():
                self.cfg.set(key, value)

        def load(self):
            return build_app(args)

    Application().run()


if __name__ == '__main__':
    main()
