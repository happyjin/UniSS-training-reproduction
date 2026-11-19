"""CPU state-machine regression tests; these do not certify checkpoint audio quality."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class Tokenizer:
    eos_token_id = 99
    vocab_size = 100
    def encode(self, *args, **kwargs):
        return [1, 2]
    def decode(self, tokens, **kwargs):
        return ''.join(chr(97 + int(x) % 26) for x in tokens)


class Talker:
    def __init__(self):
        self.speech_embedding = torch.nn.Embedding(120, 4)
        self.inputs = []
        self.backbone = self.forward
    def sos_embed(self, dtype, device):
        return torch.zeros(1, 1, 4, dtype=dtype, device=device)
    def forward(self, inputs_embeds, past_key_values=None, use_cache=True):
        self.inputs.append(inputs_embeds.clone())
        return SimpleNamespace(past_key_values=object(), last_hidden_state=inputs_embeds)
    def stream_generate_s4(self, state, cond_chunk=None, finalize=False, input_observer=None, **kw):
        x = state.pending
        state.pending = None
        if cond_chunk is not None:
            x = cond_chunk if x is None else torch.cat([x, cond_chunk], 1)
            state.cond_consumed += cond_chunk.size(1)
        if x is not None:
            input_observer(x)
        if finalize:
            state.finished = True
            return
        state.n_seg += 1
        state.n_fill += 1
        state.last_stop_reason = 'fill'
        state.decoded.append(11)
        yield 11


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(audio_ids=[98], talker_cond_positions='answer')
        self.thinker = SimpleNamespace(embed_tokens=torch.nn.Embedding(120, 4))
        self.talker = Talker()
        self.seen_mel = []
    def encode_audio_chunk(self, mel, cache=None):
        self.seen_mel.append(mel.clone())
        n = mel.shape[-1] // 8
        return torch.zeros(1, n, 4), SimpleNamespace(mel_remainder=None)
    def inject_audio_features(self, pad, embed, feats):
        return feats[0].unsqueeze(0)
    def pick_bridge_hidden(self, out):
        return out.last_hidden_state
    def build_stream_talker_condition(self, hidden, text):
        return hidden


class FakeWaveStream:
    feature_size = 128
    hop_length = 160
    n_fft = 400

    def __init__(self, _):
        self._finalized = False
        self.num_samples = 0
        self.emitted_frames = 0
    def feed(self, part, finalize=False):
        self._finalized |= finalize
        self.num_samples += len(part)
        # Timing-only frontend double: frames require observed right context;
        # EOF releases the remaining frames without adding source samples.
        frames = (self.num_samples // self.hop_length if self._finalized else
                  max(0, (self.num_samples - self.n_fft // 2) // self.hop_length + 1))
        count = max(0, frames - self.emitted_frames)
        self.emitted_frames += count
        return torch.ones(self.feature_size, count)


class FakeRunner:
    def __init__(self, model, embeds, *args, **kwargs):
        self.embeds = embeds
        self.n_forward = 0
        self.use_cache = True
        self.static = None
    def __call__(self):
        self.n_forward += 1
        logits = torch.zeros(120)
        logits[11] = 10  # WRITE for every call.
        return logits, SimpleNamespace(last_hidden_state=torch.cat(self.embeds, 1))


@pytest.fixture
def runtime(monkeypatch):
    from streaming_si.inference import s5_session as mod
    monkeypatch.setattr(mod, 'VoxtralWaveformStream', FakeWaveStream)
    monkeypatch.setattr(mod, 'ThinkerRunner', FakeRunner)
    def sample(embeds, run, tok, *args, **kwargs):
        embeds.append(tok(5))
        return [5], run()[1], 12
    monkeypatch.setattr(mod, 'sample_delta', sample)
    return mod


def session(runtime, **kwargs):
    return runtime.S5Session(FakeModel(), Tokenizer(), [10, 11, 12, 13], object(), 'en',
        device='cpu', decode=False, thinker_graph=False, talker_graph=False,
        max_text_tokens=8, max_speech_tokens=8, **kwargs)


def test_small_and_transition_chunks_consume_every_sample(runtime):
    s = session(runtime, chunk_ms=160, chunk_ms_after_write=640, collect_audio=True)
    x = np.ones(16000 + 137, np.float32)
    for pos in range(0, len(x), 97):
        s.feed_audio(x[pos:pos + 97])
    s.feed_audio(np.zeros(0, np.float32), finalize=True)
    assert s.source_samples == len(x) == s.consumed_samples
    assert s.waveform_stream.num_samples == len(x)
    assert len(s.pending_audio) == 0
    assert s.source_padding_mel_frames < 8
    assert sum(m.shape[-1] for m in s.model.seen_mel) % 8 == 0
    assert s.closed
    assert s.feed_audio(np.zeros(0, np.float32), finalize=True) == []
    with pytest.raises(RuntimeError):
        s.feed_audio(np.zeros(1, np.float32))


def test_window_replay_never_resamples_or_reemits(runtime):
    s = session(runtime, chunk_ms=160, context_max_tokens=128)
    text, codec = [], []
    s.on_event = lambda e: text.append(e['text']) if e['type'] == 'text' else codec.extend(e['tokens']) if e['type'] == 'speech_tokens' else None
    for _ in range(160):
        s.feed_audio(np.ones(2560, np.float32))
    s.feed_audio(np.ones(40, np.float32))  # Release the last chunk's STFT right context.
    assert s.context_replays > 0
    assert s.talker_replays > 0
    assert len(text) == len(codec) == 160
    assert sum(x.size(1) for x in s.embeds) <= s.capacity
    assert s.talker_fed <= s.capacity
    assert len(s.state.decoded) <= 10
    assert s.reference_audio.size <= s.reference_samples


def test_pcm_is_delivered_during_write_with_contiguous_offsets(runtime, monkeypatch):
    from streaming_si.inference import s5_pcm_decoder as decoder_mod
    class Decoder:
        session_rollovers = 0
        def __init__(self, *args, **kwargs):
            pass
        def feed(self, tokens, finalize=False, on_pcm_chunk=None):
            assert not finalize
            on_pcm_chunk(torch.ones(80))
        def finish(self, on_pcm_chunk=None):
            on_pcm_chunk(torch.ones(20))
    monkeypatch.setattr(decoder_mod, 'PersistentSpeechDecoder', Decoder)
    s = session(runtime, chunk_ms=160, collect_audio=True)
    s.decode = True
    observed = []
    s.on_event = lambda event: observed.append(event.copy())
    s.feed_audio(np.ones(2560, np.float32))
    assert s.first_pcm_sec is None  # The last mel frame still needs 40 source samples.
    s.feed_audio(np.ones(40, np.float32))
    assert s.first_pcm_sec is not None and not s.closed
    s.feed_audio(np.ones(2520, np.float32), finalize=True)
    pcm = [e for e in observed if e['type'] == 'pcm']
    assert all(a['sample_offset'] + a['pcm'].size == b['sample_offset'] for a, b in zip(pcm, pcm[1:]))
    np.testing.assert_array_equal(s.result()['wav'], np.concatenate([e['pcm'] for e in pcm]))
    assert observed.index(next(e for e in observed if e['type'] == 'pcm')) < next(i for i,e in enumerate(observed) if e['type']=='step')


@pytest.mark.parametrize('chunk,after,threshold', [(0,None,None),(81,None,None),(160,-80,None),(160,None,1.1)])
def test_invalid_chunks_are_rejected(runtime, chunk, after, threshold):
    with pytest.raises(ValueError):
        session(runtime, chunk_ms=chunk, chunk_ms_after_write=after, write_threshold=threshold)


def test_source_reference_contains_only_consumed_audio(runtime):
    s = session(runtime, chunk_ms=160)
    def speech(_tokens):
        assert np.all(s.reference_audio == 1)
        assert s.consumed_samples == 2560
    s._speech = speech
    s.feed_audio(np.concatenate([np.ones(2560), np.full(100, 2)]).astype(np.float32))
    assert s.pending_audio.size == 100


def test_source_reference_follows_consumed_audio_with_a_refresh_budget(runtime):
    s = session(runtime, chunk_ms=640, reference_refresh_sec=1.28)
    spans = []
    def prompt(audio, **kwargs):
        spans.append(audio.copy())
        return [1, 2], torch.zeros(1, 4, 80)
    s.prompter = prompt
    for value in (1., 2., 3., 4., 5., 6., 7.):
        s.feed_audio(np.full(10240, value, np.float32))
        condition = s._source_reference()
        if condition:
            assert condition['source_end_sec'] == s.consumed_samples / 16000
            assert condition['source_start_sec'] >= 0
            assert condition['source_end_sec'] <= s.source_samples / 16000
            assert len(condition['prompt_speech_tokens']) * 2 == condition['prompt_feat'].size(1)
            # Last fully consumed chunk precedes the currently received right context.
            assert spans[-1][-1] <= value
    versions = s.reference_version
    assert versions >= 2 and len(spans) == versions
    assert s._source_reference() is None  # No new source -> no re-encode.
    assert s.reference_version == versions
    s.reference_policy = 'fixed'
    s.feed_audio(np.ones(40960, np.float32))
    assert s._source_reference() is None
    assert s.reference_version == versions


def test_silent_reference_does_not_replace_last_voice(runtime):
    s = session(runtime, chunk_ms=640, reference_refresh_sec=.64)
    s.prompter = lambda audio, **kw: ([1], torch.zeros(1, 2, 80))
    s.feed_audio(np.ones(40960, np.float32))
    assert s._source_reference() is not None
    version = s.reference_version
    s.feed_audio(np.zeros(64000, np.float32))
    assert s._source_reference() is None and s.reference_version == version
    s.feed_audio(np.ones(40960, np.float32))
    assert s._source_reference() is not None and s.reference_version == version + 1


def test_cli_defaults_restore_segmented_pcm_without_truncating_source():
    from streaming_si.inference.infer_s5_streaming_dynchunk import parse_args
    common = ['--model_dir', 'base', '--adapter', 'adapter', '--input_jsonl', 'input.jsonl', '--out_dir', 'out']
    streaming = parse_args(common + ['--synthesis_mode', 'streaming'])
    assert streaming.segmentation == 'vad'
    assert streaming.hard_sec == 30 and streaming.soft_sec == 20 and streaming.pause_end_sec == 1.2
    assert streaming.max_audio_sec == 0 and streaming.retry == 0
    offline = parse_args(common)
    assert offline.segmentation == 'none' and offline.max_audio_sec == 30
    diagnostic = parse_args(common + ['--synthesis_mode', 'streaming', '--segmentation', 'none'])
    assert diagnostic.segmentation == 'none' and diagnostic.max_audio_sec == 0


def test_timed_context_survives_segment_end_without_reemitting(runtime):
    s = session(runtime, chunk_ms=640, context_window_sec=10, collect_audio=True)
    s.feed_audio(np.ones(6 * 16000, np.float32), segment_end=True)
    before = (s.text_count, s.speech_count, s.consumed_samples)
    assert s.records and s.talker_records
    assert s.run.embeds is s.embeds
    assert s.reference_audio.size and s.state.pending is not None
    assert s.state.cond_consumed == 0 and not s.state.finished
    assert not any(e.get('boundary') and e.get('replayed') and e['type'] != 'context'
                   for e in s.events)
    s.feed_audio(np.ones(6 * 16000, np.float32), segment_end=True)
    assert s.consumed_samples == before[2] + 6 * 16000
    assert s.text_count > before[0] and s.speech_count > before[1]
    # Whole interactions crossing the left boundary survive, expired ones do not.
    cutoff = s.consumed_samples / 16000 - 10
    assert all(r.end_sec > cutoff for r in s.records)
    assert all(r.end_sec > cutoff for r in s.talker_records)
    assert s.records[0].start_sec <= cutoff + .64
    assert s.run.n_forward == 1  # Replay prefill only; no sampled/output tokens.


def test_idle_does_not_age_timed_context(runtime):
    s = session(runtime, context_window_sec=10)
    s.feed_audio(np.ones(4 * 16000, np.float32))
    before = (list(s.records), s.context_replays, s.consumed_samples)
    s.feed_audio(np.zeros(0, np.float32))
    assert list(s.records) == before[0]
    assert (s.context_replays, s.consumed_samples) == before[1:]


@pytest.mark.parametrize('wait_only', [False, True])
def test_turn_context_replays_latest_16_at_32_turns(runtime, monkeypatch, wait_only):
    if wait_only:
        original = FakeRunner.__call__

        def wait(self):
            logits, out = original(self)
            logits[10] = 20
            return logits, out

        monkeypatch.setattr(FakeRunner, '__call__', wait)
    s = session(runtime, context_window_32_16=True, collect_audio=True)
    s.feed_audio(np.ones(49 * 10240 + 400, np.float32))
    thinker = [e for e in s.events if e['type'] == 'context' and e['component'] == 'thinker']
    assert [e['retained_turns'] for e in thinker] == [16, 16]
    assert all(e['turn_limit'] and not e['capacity_limit'] for e in thinker)
    assert [r.turn for r in s.records] == list(range(32, 49))
    assert s.text_count == s.speech_count == (0 if wait_only else 49)
    assert all(r.turn >= 32 for r in s.talker_records)
    assert s.talker_replays == (0 if wait_only else 2)


def test_duration_guard_has_no_vad_and_preserves_silence_and_tail(runtime, monkeypatch):
    from streaming_si.inference import s5_segmenter as segmenter
    def forbidden():
        raise AssertionError('interval mode must not instantiate VAD')
    monkeypatch.setattr(segmenter, 'SileroFrameVad', forbidden)
    core = session(runtime, context_window_sec=10, collect_audio=True)
    stream = segmenter.SegmentedSession(core, interval_sec=3)
    source = np.zeros(7 * 16000 + 137, np.float32)
    for pos in range(0, len(source), 777):
        stream.feed_audio(source[pos:pos + 777])
    stream.close()
    assert core.consumed_samples == core.source_samples == source.size
    assert [s['reason'] for s in stream.feeder.segments] == ['hard', 'hard', 'eof']
    assert [s['source_start_sample'] for s in stream.feeder.segments] == [0, 48000, 96000]
    assert stream.feeder.segments[-1]['source_end_sample'] == source.size
    assert core.records and core.context_replays >= 3


def test_fixed_guard_cli_is_independent_of_vad():
    from streaming_si.inference.infer_s5_streaming_dynchunk import parse_args
    from streaming_si.inference.s5_segmenter import segmentation_options
    args = parse_args(['--model_dir', 'base', '--adapter', 'adapter', '--input_jsonl', 'in',
        '--out_dir', 'out', '--synthesis_mode', 'streaming', '--segmentation', 'interval',
        '--context_window_sec', '10'])
    assert args.context_window_sec == 10
    assert segmentation_options(args) == {'interval_sec': 30}


def test_duration_guard_merges_sub_stft_tail_instead_of_dropping_source(runtime):
    from streaming_si.inference.s5_segmenter import SegmentedSession
    core = session(runtime, context_window_sec=10)
    stream = SegmentedSession(core, interval_sec=3)
    source = np.ones(6 * 16000 + 137, np.float32)
    for pos in range(0, len(source), 1280):
        stream.feed_audio(source[pos:pos + 1280])
    stream.close()
    assert core.consumed_samples == source.size
    assert len(stream.feeder.segments) == 2
    assert stream.feeder.segments[-1]['source_end_sample'] == source.size
    assert stream.feeder.segments[-1]['dur_sec'] < 3.025
