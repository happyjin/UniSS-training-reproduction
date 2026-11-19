"""Device-independent contract tests for persistent output and Flow rollover.

The fake Flow exposes per-token values so lost/repeated lookahead is observable.
These tests do not establish real Flow/HiFT audio quality at window seams.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

MODULE = Path(__file__).resolve().parents[2] / 'custom/streaming_si/inference/s5_pcm_decoder.py'
sys.path.insert(0, str(MODULE.parents[2]))
spec = importlib.util.spec_from_file_location('s5_pcm_decoder_under_test', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PersistentSpeechDecoder = module.PersistentSpeechDecoder


class Flow(torch.nn.Module):
    token_mel_ratio = 2
    pre_lookahead_len = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.decoder = SimpleNamespace(rand_noise=torch.zeros(1, 80, 15000))


class HiFT:
    def inference(self, speech_feat, finalize):
        stable = speech_feat if finalize else speech_feat[..., :-1]
        return stable[:, 0].repeat_interleave(480, dim=-1), None


class HiFTContractStub:
    """Sample bookkeeping only; real phase/crop checks live in GPU probes."""
    def __init__(self, hift, **kwargs):
        self.hift = hift
        self.mel = None
        self.total_samples = self.crops = 0
    @property
    def retained_frames(self):
        return 0 if self.mel is None else self.mel.size(-1)
    def feed(self, mel, finalize=False):
        self.mel = mel if self.mel is None else torch.cat([self.mel, mel], -1)
        wav, _ = self.hift.inference(self.mel, finalize=finalize)
        out = wav[..., self.total_samples:]
        self.total_samples = wav.size(-1)
        return out


@pytest.fixture(autouse=True)
def fake_hift_stream(monkeypatch):
    monkeypatch.setattr(module, 'BoundedCausalHiFT', HiFTContractStub)


class FakeModel:
    STREAM_TOKEN_HOP_LEN = 25
    STREAM_TOKEN_MAX_HOP_LEN = 100
    STREAM_SCALE_FACTOR = 2
    STREAM_SAMPLES_PER_MEL = 480

    def __init__(self):
        self.speech_decoder = (Flow(), HiFT())
        self.calls = []
        self.embeddings = []

    def _set_ode_steps(self, flow, steps):
        self.steps = steps

    def _prep_decode_inputs(self, emb, tokens, feat, device, ratio):
        tokens = torch.as_tensor(tokens if tokens is not None else [], dtype=torch.long).reshape(1, -1)
        feat = torch.zeros(1, 0, 80) if feat is None else feat
        if tokens.size(1) * ratio != feat.size(1):
            raise ValueError('unaligned prompt')
        return tokens, feat, torch.zeros(1, 192) if emb is None else emb

    def _flow_infer(self, flow, tokens, length, prompt, prompt_len, feat, feat_len,
                    embedding, streaming, finalize):
        self.calls.append((tokens.clone(), prompt.clone(), feat.clone(), finalize))
        self.embeddings.append(embedding.clone())
        # Include lookahead deliberately; decoder must only commit a stable hop.
        return tokens.float().repeat_interleave(2, dim=1).unsqueeze(1).expand(-1, 80, -1)

    def _hift_infer(self, hift, mel, finalize):
        # Real causal HiFT withholds an unreliable tail until finalize.
        stable = mel if finalize else mel[..., :-1]
        return stable[:, 0].repeat_interleave(480, dim=-1), None


def expected(tokens):
    return torch.tensor(tokens, dtype=torch.float32).repeat_interleave(960).reshape(1, -1)


def test_write_does_not_finalize_or_flush_an_incomplete_hop():
    model = FakeModel()
    decoder = PersistentSpeechDecoder(model)
    assert decoder.feed(range(1, 11)) == []
    assert decoder.feed([]) == []
    assert not model.calls
    chunks = decoder.feed(range(11, 29))
    assert len(chunks) == 1
    assert model.calls[-1][-1] is False
    assert model.calls[-1][0].numel() == 28
    chunks += decoder.finish()
    assert torch.equal(torch.cat(chunks, -1), expected(list(range(1, 29))))
    assert [c[-1] for c in model.calls] == [False, True]


@pytest.mark.parametrize('partition', [1, 7, 113, 401])
def test_rollover_and_arbitrary_write_partition_emit_each_sample_once(partition):
    model = FakeModel()
    calls = []
    condition_calls = []
    def condition():
        condition_calls.append(1)
        return {'spk_emb': torch.ones(1, 192)}
    decoder = PersistentSpeechDecoder(model, max_session_sec=4,
                                      prompt_tail_sec=2, condition_provider=condition)
    tokens = list(range(1, 402))
    chunks = []
    for i in range(0, len(tokens), partition):
        chunks += decoder.feed(tokens[i:i+partition], on_pcm_chunk=calls.append)
        assert decoder.stats['retained_tokens'] <= 100 + decoder.lookahead
        assert decoder.stats['retained_mel_frames'] <= (100 + decoder.lookahead) * 2
    chunks += decoder.finish(on_pcm_chunk=calls.append)
    assert torch.equal(torch.cat(chunks, -1), expected(tokens))
    assert len(chunks) == len(calls)
    assert all(a is b for a, b in zip(chunks, calls))
    assert len(condition_calls) == decoder.session_rollovers == 3
    # Capacity rollover must use genuine right context, not signal sentence end.
    assert sum(finalize for _, _, _, finalize in model.calls) == 1
    assert decoder.total_samples == len(tokens) * 960
    assert all(prompt.numel() * 2 == feat.size(1) for _, prompt, feat, _ in model.calls)
    assert max(tokens.numel() for tokens, _, _, _ in model.calls) <= 100 + decoder.lookahead
    assert max(prompt.numel() for _, prompt, _, _ in model.calls) <= 50
    assert torch.equal(model.embeddings[-1], torch.ones(1, 192))


def test_exact_cap_finishes_once_and_empty_stream_has_no_audio():
    model = FakeModel()
    decoder = PersistentSpeechDecoder(model, max_session_sec=4)
    chunks = decoder.feed(range(100)) + decoder.finish()
    assert decoder.session_rollovers == 0
    assert sum(c[-1] for c in model.calls) == 1
    assert torch.equal(torch.cat(chunks, -1), expected(list(range(100))))
    assert PersistentSpeechDecoder(FakeModel()).finish() == []
    with pytest.raises(RuntimeError, match='closed'):
        decoder.feed([1])


def test_failed_output_callback_is_terminal():
    decoder = PersistentSpeechDecoder(FakeModel())
    def fail(chunk):
        raise OSError('socket closed')
    with pytest.raises(OSError, match='socket closed'):
        decoder.feed(range(28), on_pcm_chunk=fail)
    with pytest.raises(RuntimeError, match='failed'):
        decoder.finish()


def test_prompt_alignment_padding_and_embedding_freeze():
    model = FakeModel()
    decoder = PersistentSpeechDecoder(model, prompt_speech_tokens=[4] * 13,
                                      prompt_feat=torch.zeros(1, 26, 80), max_session_sec=4)
    decoder.set_next_embedding(torch.ones(1, 192))
    assert decoder.feed(range(39)) == []  # hop25 + prompt pad12 + lookahead3 =40
    assert decoder.feed([39])
    assert torch.equal(model.embeddings[-1], torch.zeros(1, 192))
    chunks = decoder.feed(range(40, 103)) + decoder.finish()
    assert decoder.session_rollovers == 1
    assert torch.equal(model.embeddings[-1], torch.ones(1, 192))


def test_invalid_windows_and_noncausal_hift_fail_explicitly():
    for seconds in (0, float('nan'), 0.5):
        with pytest.raises(ValueError):
            PersistentSpeechDecoder(FakeModel(), max_session_sec=seconds)
    model = FakeModel()
    class Legacy:
        def inference(self, speech_feat, cache_source):
            pass
    model.speech_decoder = model.speech_decoder[0], Legacy()
    with pytest.raises(ValueError, match='causal HiFT'):
        PersistentSpeechDecoder(model)


def test_source_refresh_preserves_target_history_and_all_pcm():
    model = FakeModel()
    update = {}
    decoder = PersistentSpeechDecoder(model, max_session_sec=4,
        prompt_tail_sec=2, reference_provider=lambda: update.pop('next', None))
    chunks = decoder.feed(range(1, 79))
    committed = decoder.offset
    assert committed > 0
    update['next'] = dict(version=1, spk_emb=torch.ones(1, 192),
        prompt_speech_tokens=[700, 701], prompt_feat=torch.full((1, 4, 80), 7.),
        source_start_sec=2., source_end_sec=4.)
    chunks += decoder.feed(range(79, 190))
    assert decoder.source_condition_updates == 1
    assert decoder.reference_start_sec == 2. and decoder.reference_end_sec == 4.
    assert decoder.source_prompt.tolist() == [[700, 701]]
    assert decoder.continuity_prompt.size(1) > 0
    assert decoder.prompt_feat.size(1) == decoder.prompt.size(1) * 2
    assert decoder.source_prompt.size(1) == 2  # Source never replaced by target tail.
    chunks += decoder.finish()
    assert torch.equal(torch.cat(chunks, -1), expected(list(range(1, 190))))
    assert decoder.total_samples == 189 * 960
    assert sum(c[-1] for c in model.calls) == 1  # Only real EOF finalizes Flow.


def test_bad_source_pair_fails_before_rebasing_or_emitting():
    model = FakeModel()
    decoder = PersistentSpeechDecoder(model,
        reference_provider=lambda: dict(version=1, prompt_speech_tokens=[1, 2],
                                        prompt_feat=torch.zeros(1, 2, 80)))
    with pytest.raises(ValueError, match='unaligned'):
        decoder.feed([1])
    assert decoder.failed and decoder.total_samples == 0 and decoder.session_rollovers == 0


def test_new_write_lookahead_cannot_reassign_old_tokens_to_new_voice():
    model = FakeModel()
    model.STREAM_TOKEN_MAX_HOP_LEN = 25
    condition = dict(version='A', spk_emb=torch.zeros(1, 192))
    decoder = PersistentSpeechDecoder(model, prompt_tail_sec=0,
        reference_metadata=condition, reference_provider=lambda: condition,
        bind_reference_to_tokens=True)
    chunks = decoder.feed(range(28))  # 25 A tokens committed, 3 A lookahead pending.
    assert decoder.offset == 25
    condition = dict(version='B', spk_emb=torch.ones(1, 192))
    chunks += decoder.feed(range(28, 53))
    # The next hop starts with three old A tokens; B lookahead cannot recolor it.
    assert decoder.reference_version == 'A'
    assert all(torch.count_nonzero(e) == 0 for e in model.embeddings)
    chunks += decoder.feed(range(53, 103))
    assert decoder.reference_version == 'B'
    assert torch.equal(model.embeddings[-1], torch.ones(1, 192))
    chunks += decoder.finish()
    assert torch.equal(torch.cat(chunks, -1), expected(list(range(103))))
    assert len(decoder.token_references) <= decoder.max_tokens + decoder.lookahead
