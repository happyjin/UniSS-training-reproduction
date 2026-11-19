"""Transport contracts without loading model checkpoints or a CUDA device."""
import importlib.util
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

pytest.importorskip('flask')
PATH = Path(__file__).resolve().parents[2] / 'custom/streaming_si/inference/web_demo_s5.py'
spec = importlib.util.spec_from_file_location('s5_web_under_test', PATH)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class Session:
    def __init__(self, language, callback):
        self.callback = callback
        self.total = 0
        self.closed = False

    def feed_audio(self, block, finalize=False):
        self.total += len(block)
        if len(block):
            self.callback(dict(type='pcm', pcm=block, sample_rate=24000,
                               sample_offset=self.total - len(block), elapsed_sec=.2, source_sec=self.total / 16000))
            self.callback(dict(type='text', text='译文'))
        self.closed = finalize

    def metrics(self):
        return dict(source_sec=self.total / 16000, finalized=self.closed)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(web, 'decode_upload', lambda raw: np.ones(1600, np.float32) * .25)
    return web.create_app(Session, dict(model_dir='base93767', adapter='s5-10000'), enable_websocket=False)


def upload(client, **fields):
    return client.post('/translate', data=dict(audio=(io.BytesIO(b'fake'), 'audio.wav'), **fields))


def test_upload_pcm_and_metrics(app):
    with app.test_client() as client:
        response = upload(client, tgt_lang='zh')
        assert response.status_code == 200
        events = [json.loads(line) for line in response.data.splitlines()]
        pcm = [e for e in events if e['type'] == 'pcm']
        assert len(pcm) == 2
        import base64
        actual = b''.join(base64.b64decode(e['data']) for e in pcm)
        assert len(actual) == 3200
        assert events[-1]['type'] == 'metrics' and events[-1]['finalized']
        assert all(e['server_request_sec'] >= 0 for e in events)
        assert client.get('/config').json['adapter'] == 's5-10000'
        assert client.get('/').status_code == 200


def test_validation_and_decoder_error(app, monkeypatch):
    with app.test_client() as client:
        assert client.post('/translate').status_code == 400
        assert upload(client, mode='bogus').status_code == 400
        def fail(raw):
            raise ValueError('unsupported input')
        monkeypatch.setattr(web, 'decode_upload', fail)
        assert upload(client).status_code == 400


def test_inference_error_is_reported_and_lock_released(monkeypatch):
    monkeypatch.setattr(web, 'decode_upload', lambda raw: np.ones(1600, np.float32))
    class Broken(Session):
        def feed_audio(self, block, finalize=False):
            raise RuntimeError('decode failed')
    app = web.create_app(Broken, {}, enable_websocket=False)
    with app.test_client() as client:
        for _ in range(2):
            response = upload(client)
            events = [json.loads(line) for line in response.data.splitlines()]
            assert events[-1]['type'] == 'error'
            assert 'decode failed' in events[-1]['error']


def test_pcm_visible_before_feed_finishes_and_busy_rejected(monkeypatch):
    monkeypatch.setattr(web, 'decode_upload', lambda raw: np.ones(1280, np.float32))
    release = threading.Event()
    class Slow(Session):
        def feed_audio(self, block, finalize=False):
            super().feed_audio(block, finalize)
            if not finalize:
                release.wait(2)
    app = web.create_app(Slow, {}, enable_websocket=False)
    with app.test_client() as client:
        response = client.post('/translate', data={'audio': (io.BytesIO(b'x'), 'a.wav')}, buffered=False)
        try:
            first = json.loads(next(iter(response.response)))
            assert first['type'] == 'pcm'
            with app.test_client() as second:
                assert upload(second).status_code == 409
        finally:
            release.set()
            response.close()


def test_wire_encoding_clips_and_preserves_sample_position():
    import base64
    event = dict(type='pcm', pcm=np.array([-2., 0., 2.]), sample_offset=45, elapsed_sec=.5)
    wire = web.serialize_event(event)
    assert np.frombuffer(base64.b64decode(wire['data']), '<i2').tolist() == [-32767, 0, 32767]
    assert wire['sample_offset'] == 45 and wire['elapsed_sec'] == .5
    assert 'pcm' in event and 'pcm' not in wire


def test_original_page_json_upload_and_sse(app):
    import base64
    with app.test_client() as client:
        response = client.post('/translate', json=dict(
            audio=base64.b64encode(b'fake').decode(), target_lang='zh'))
        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'
        events = [json.loads(part[6:]) for part in response.data.decode().split('\n\n') if part]
        assert [e['type'] for e in events].count('audio') == 2
        assert events[-1]['type'] == 'done'
        assert events[-1]['text'] == '译文译文'
        assert events[-1]['metrics']['finalized']
        assert len([e for e in events if e['type'] == 'stat' and 'audio_ttft' in e]) == 1
        assert client.get('/config').json['src_sr'] == 16000
        assert client.get('/web_demo_s5.html').status_code == 200


def test_original_protocol_uses_actual_wall_time_and_filters_eos():
    converter = web.OriginalPageEvents()
    assert converter.convert(dict(type='decision', decision='eos')) == []
    events = converter.convert(dict(type='text', text='test', source_sec=100,
                                   server_request_sec=20, server_since_first_input_sec=3.5))
    assert events[-1] == dict(type='stat', text_ttft=20000)
    events = converter.convert(dict(type='pcm', data='AAAA', sample_rate=24000,
                                   sample_offset=0, source_sec=100, server_request_sec=4.2))
    assert events[-1] == dict(type='stat', audio_ttft=4200)
    assert events[0]['pcm'] == 'AAAA'


def test_original_page_errors_are_readable_as_sse(app):
    with app.test_client() as client:
        response = client.post('/translate', json=dict(audio='not base64!', target_lang='zh'))
        assert response.status_code == 400
        assert response.mimetype == 'text/event-stream'
        event = json.loads(response.data.decode().strip()[6:])
        assert event['type'] == 'error' and 'cannot decode audio' in event['message']


def test_per_request_chunk_selection_is_isolated(monkeypatch):
    monkeypatch.setattr(web, 'decode_upload', lambda raw: np.ones(1600, np.float32))
    received = []
    defaults = dict(request_chunk_config=True, chunk_ms=640, chunk_ms_after_write=None)
    def factory(target, callback, *, chunk_options):
        received.append(dict(chunk_options))
        return Session(target, callback)
    app = web.create_app(factory, defaults, enable_websocket=False)
    with app.test_client() as client:
        for fields, expected in [
            (dict(chunk_ms=640, chunk_ms_after_write=1280), dict(chunk_ms=640, chunk_ms_after_write=1280)),
            (dict(chunk_ms=2560, chunk_ms_after_write=None), dict(chunk_ms=2560, chunk_ms_after_write=None)),
            ({}, dict(chunk_ms=640, chunk_ms_after_write=None)),
        ]:
            import base64
            response = client.post('/translate', json=dict(audio=base64.b64encode(b'fake').decode(), **fields))
            events = [json.loads(part[6:]) for part in response.data.decode().split('\n\n') if part]
            assert response.status_code == 200
            assert received[-1] == expected
            assert {key: events[-1]['metrics'][key] for key in expected} == expected
        for fields in [dict(chunk_ms=161), dict(chunk_ms=160.5), dict(chunk_ms=True), dict(chunk_ms_after_write=123)]:
            assert client.post('/translate', json=dict(audio='ZmFrZQ==', **fields)).status_code == 400
    assert defaults == dict(request_chunk_config=True, chunk_ms=640, chunk_ms_after_write=None)


def test_all_four_target_languages_reach_the_session(monkeypatch):
    import base64
    monkeypatch.setattr(web, 'decode_upload', lambda raw: np.ones(1600, np.float32))
    targets = []
    def factory(target, callback):
        targets.append(target)
        return Session(target, callback)
    app = web.create_app(factory, {}, enable_websocket=False)
    with app.test_client() as client:
        assert {r['code'] for r in client.get('/config').json['langs']} == {'en', 'zh', 'ja', 'ko'}
        for target in ['en', 'zh', 'ja', 'ko']:
            response = client.post('/translate', json=dict(audio=base64.b64encode(b'fake').decode(), target_lang=target))
            assert response.status_code == 200
            events = [json.loads(part[6:]) for part in response.data.decode().split('\n\n') if part]
            assert events[-1]['type'] == 'done'
        assert targets == ['en', 'zh', 'ja', 'ko']


def test_primary_chunks_and_default_bounded_segments():
    for chunk in (640, 1280, 1920, 2560):
        assert web.request_chunks({'chunk_ms': chunk}, {})['chunk_ms'] == chunk
    for chunk in (160, 320, 5120):
        with pytest.raises(ValueError):
            web.request_chunks({'chunk_ms': chunk}, {})
    args = web.parse_args(['--model_dir', 'base', '--adapter', 'adapter',
                           '--cosyvoice_repo', '/unused'])
    assert args.segmentation == 'vad'
    assert args.hard_sec == 30 and args.soft_sec == 20 and args.pause_end_sec == 1.2
    assert args.reference_policy == 'rolling' and args.reference_refresh_sec > 0
