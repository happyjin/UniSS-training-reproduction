"""Regression tests for independently switchable inference optimizations."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from custom.qwen_omnispeak.configuration_qwen_omnispeak import Qwen3TtsTalkerConfig
from custom.qwen_omnispeak.modeling_qwen_omnispeak import (
    Qwen3TtsCodePredictor,
    Qwen3TtsTalkerForConditionalGeneration,
)
from inference.code_predictor_runtime import CodePredictorRunner
from inference.codec_buffer import CodecFrameBuffer
from inference.common import (
    InferenceConfig,
    InferenceSample,
    build_e2e_latency_metrics,
    generate_text,
    new_optimization_metrics,
    thinker_sampling_params,
    timed_phase,
)
from inference.streaming_decoder import CompiledStreamingDecoder
from inference.benchmark_streaming_optimizations import parse_args as parse_benchmark_args
from inference.s2s_infer import parse_args as parse_s2s_args


def _tiny_talker_config() -> Qwen3TtsTalkerConfig:
    return Qwen3TtsTalkerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        codec_vocab_size=64,
        num_code_groups=4,
        code_predictor_hidden_size=32,
        code_predictor_intermediate_size=64,
        code_predictor_num_layers=1,
        code_predictor_num_attention_heads=4,
        code_predictor_num_key_value_heads=2,
        code_predictor_head_dim=8,
        code_predictor_vocab_size=32,
        max_position_embeddings=64,
    )


def _tiny_code_predictor() -> Qwen3TtsCodePredictor:
    return Qwen3TtsCodePredictor(_tiny_talker_config()).float().eval()


def _tiny_talker() -> Qwen3TtsTalkerForConditionalGeneration:
    return Qwen3TtsTalkerForConditionalGeneration(
        _tiny_talker_config()
    ).float().eval()


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeNativeThinker:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.generate_calls = 0
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.last_kwargs = kwargs
        streamer = kwargs["streamer"]
        prompt = kwargs["input_ids"]
        streamer.put(prompt.cpu())
        generated = []
        for token in self.tokens:
            token_tensor = torch.tensor([token], device=prompt.device)
            streamer.put(token_tensor.cpu())
            generated.append(token_tensor.reshape(1, 1))
        streamer.end()
        return SimpleNamespace(
            sequences=torch.cat([prompt, *generated], dim=1)
        )


class _FakeOmniSpeakInner:
    def __init__(self, tokens):
        self.thinker = _FakeNativeThinker(tokens)
        self.run_thinker_calls = 0

    def run_thinker(self, **kwargs):
        del kwargs
        self.run_thinker_calls += 1
        raise AssertionError("native generate route must not call run_thinker")


def _fake_native_generate_stack(tokens):
    inner = _FakeOmniSpeakInner(tokens)
    model = SimpleNamespace(model=inner)
    processor = SimpleNamespace(tokenizer=_FakeTokenizer())
    inputs = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    return model, processor, inputs


class _FakeCausalDecoder(torch.nn.Module):
    def __init__(self, quantizers=2, upsample=3):
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=quantizers)
        self.upsample = upsample
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, codes):
        main_codes = codes[:, :1, :].to(torch.float32) + self.anchor
        return main_codes.repeat_interleave(self.upsample, dim=-1)


def _fake_12hz_tokenizer(upsample=3):
    decoder = _FakeCausalDecoder(upsample=upsample)
    model = SimpleNamespace(
        decoder=decoder,
        get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
        get_decode_upsample_rate=lambda: upsample,
        get_output_sample_rate=lambda: 24000,
    )
    return SimpleNamespace(tokenizer=SimpleNamespace(model=model))


def _fake_25hz_tokenizer():
    tokenizer = _fake_12hz_tokenizer()
    tokenizer.tokenizer.model.get_model_type = (
        lambda: "qwen3_tts_tokenizer_25hz"
    )
    return tokenizer


def test_optimization_flags_default_off():
    """Changing defaults must not silently alter existing inference calls."""
    config = InferenceConfig(model="checkpoint")

    assert config.use_code_predictor_kv_cache is False
    assert config.compile_code_predictor is False
    assert config.code_predictor_compile_mode == "default"
    assert config.use_thinker_kv_cache is False
    assert config.use_gpu_codec_buffer is False
    assert config.compile_streaming_decoder is False
    assert config.online_thinker_talker is False
    assert config.online_talker_start_tokens == 1
    assert config.compile_talker is False
    assert config.predictor_frame_batch == 1
    assert config.latency_profiling == "sync"


def test_thinker_sampling_defaults_preserve_shared_legacy_behavior():
    """Old callers must keep using temperature/top-k for Thinker decoding."""
    config = InferenceConfig(
        model="checkpoint", temperature=0.9, top_k=50, top_p=0.8
    )

    assert thinker_sampling_params(config) == (0.9, 50, 1.0)


def test_thinker_sampling_can_be_overridden_without_changing_talker_settings():
    """Production S2S needs greedy text and sampled codec decoding."""
    config = InferenceConfig(
        model="checkpoint",
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        thinker_temperature=0.0,
        thinker_top_k=0,
        thinker_top_p=1.0,
    )

    assert thinker_sampling_params(config) == (0.0, 0, 1.0)
    assert (config.temperature, config.top_k, config.top_p) == (0.9, 50, 1.0)


def test_e2e_latency_metrics_use_complete_wav_as_offline_first_pcm():
    """A non-streaming response is not playable before the full WAV exists."""
    metrics = build_e2e_latency_metrics(
        complete_wav_ms=4321.0,
        first_text_ms=123.0,
        first_pcm_ms=None,
        compile_warmup_ms=0.0,
    )

    assert metrics == {
        "input_ready_to_first_text_ms": 123.0,
        "input_ready_to_first_pcm_ms": 4321.0,
        "input_ready_to_complete_wav_ms": 4321.0,
        "compile_warmup_ms": 0.0,
    }


def test_e2e_latency_metrics_preserve_observed_online_first_pcm():
    metrics = build_e2e_latency_metrics(
        complete_wav_ms=2100.0,
        first_text_ms=90.0,
        first_pcm_ms=320.0,
        compile_warmup_ms=12.0,
    )

    assert metrics["input_ready_to_first_pcm_ms"] == 320.0
    assert metrics["input_ready_to_complete_wav_ms"] == 2100.0


def test_s2s_cli_accepts_split_thinker_and_talker_decoding(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "s2s_infer.py",
            "--model-type",
            "model_type",
            "--checkpoint",
            "checkpoint",
            "--audio",
            "source.wav",
            "--temperature",
            "0.9",
            "--top-k",
            "50",
            "--thinker-temperature",
            "0",
            "--thinker-top-k",
            "0",
            "--thinker-top-p",
            "1",
            "--seed",
            "1234",
        ],
    )

    args = parse_s2s_args()

    assert (args.temperature, args.top_k, args.top_p) == (0.9, 50, 1.0)
    assert (
        args.thinker_temperature,
        args.thinker_top_k,
        args.thinker_top_p,
    ) == (0.0, 0, 1.0)
    assert args.seed == 1234


def test_benchmark_cli_accepts_split_thinker_decoding(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--checkpoint", "checkpoint",
            "--model-type", "model_type",
            "--val-jsonl", "val.jsonl",
            "--output-dir", "outputs",
            "--temperature", "0.9",
            "--top-k", "50",
            "--thinker-temperature", "0",
            "--thinker-top-k", "0",
            "--thinker-top-p", "1",
        ],
    )

    args = parse_benchmark_args()

    assert (args.thinker_temperature, args.thinker_top_k, args.thinker_top_p) == (
        0.0, 0, 1.0
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("online_talker_start_tokens", 3),
        ("predictor_frame_batch", 3),
        ("latency_profiling", "event"),
        ("latency_profiling", "trace"),
    ],
)
def test_online_optimization_config_rejects_unsupported_values(field, value):
    """Unsupported scheduler/profiling values must fail before model load."""
    config = InferenceConfig(model="checkpoint", **{field: value})

    with pytest.raises(ValueError, match=field):
        config.validate_static_optimizations()


def test_timed_phase_off_does_not_synchronize_cuda(monkeypatch):
    """Production profiling-off mode must not serialize every CUDA phase."""
    synchronize_calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: synchronize_calls.append(device)
    )
    metrics = new_optimization_metrics()

    with timed_phase(
        metrics, "talker_step_ms", torch.device("cuda:0"), mode="off"
    ):
        pass

    assert synchronize_calls == []
    assert metrics["talker_step_ms"] == 0.0


def test_invalid_predictor_compile_mode_is_rejected():
    """An unsupported compile mode must fail before loading model weights."""
    config = InferenceConfig(
        model="checkpoint", code_predictor_compile_mode="max-autotune"
    )

    with pytest.raises(ValueError, match="code_predictor_compile_mode"):
        config.validate_static_optimizations()


def test_compiled_decoder_requires_streaming():
    """Compiled callback decoding is invalid without streaming callbacks."""
    config = InferenceConfig(
        model="checkpoint", compile_streaming_decoder=True, streaming=False
    )

    with pytest.raises(ValueError, match="requires streaming"):
        config.validate_static_optimizations()


def test_code_predictor_forward_cached_matches_full_prefixes():
    """Cached group decoding must equal causal full-prefix recomputation."""
    torch.manual_seed(7)
    predictor = _tiny_code_predictor()
    projected = torch.randn(
        2, 6, predictor.parent_config.code_predictor_hidden_size
    )
    full = predictor.model(inputs_embeds=projected, causal=True)

    cache = None
    pieces = []
    for start, end in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6)):
        output = predictor.model.forward_cached(
            projected[:, start:end], past_key_values=cache
        )
        cache = output.past_key_values
        pieces.append(output.last_hidden_state)

    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
    assert cache.get_seq_length() == projected.size(1)


def test_code_predictor_forward_cached_supports_aot_autograd():
    """The inference entry point must not nest inference_mode in AOT capture."""
    predictor = _tiny_code_predictor()
    projected = torch.randn(
        1, 2, predictor.parent_config.code_predictor_hidden_size
    )
    compiled = torch.compile(
        predictor.model.forward_cached,
        backend="aot_eager",
        fullgraph=False,
    )

    with torch.inference_mode():
        output = compiled(inputs_embeds=projected)

    assert output.last_hidden_state.shape == projected.shape
    assert output.past_key_values.get_seq_length() == 2


def test_cached_predictor_runner_matches_baseline_greedy():
    """KV-cache rollout must preserve the existing greedy residual codes."""
    torch.manual_seed(11)
    talker = _tiny_talker()
    hidden = torch.randn(2, 1, talker.config.hidden_size)
    main_codes = torch.tensor([3, 5])

    baseline = CodePredictorRunner(talker).predict(hidden, main_codes)
    cached = CodePredictorRunner(talker, use_kv_cache=True).predict(
        hidden, main_codes
    )

    assert torch.equal(cached, baseline)


def test_predictor_runner_rejects_unknown_compile_mode():
    """Compile configuration errors must fail before a first frame is run."""
    with pytest.raises(ValueError, match="compile_mode"):
        CodePredictorRunner(
            _tiny_talker(), compile_model=True, compile_mode="max-autotune"
        )


def test_predictor_compile_receives_module_instead_of_bound_method():
    """Inductor must see predictor buffers as registered module state."""
    compiled_targets = []

    def fake_compile(target, **kwargs):
        del kwargs
        compiled_targets.append(target)
        return target

    talker = _tiny_talker()
    runner = CodePredictorRunner(
        talker,
        use_kv_cache=True,
        compile_model=True,
        compile_fn=fake_compile,
    )

    assert len(compiled_targets) == 1
    assert isinstance(compiled_targets[0], torch.nn.Module)
    assert compiled_targets[0].model is talker.code_predictor.model
    runner._warmup_complete = True
    hidden = torch.randn(1, 1, talker.config.hidden_size)
    assert runner.predict(hidden, torch.tensor([3])).shape == (1, 4)


def test_generate_text_uses_native_thinker_generate_when_enabled():
    """The opt-in route must use native caching and trim EOS from the reply."""
    model, processor, inputs = _fake_native_generate_stack(tokens=[21, 22, 2])
    config = InferenceConfig(
        model="checkpoint", use_thinker_kv_cache=True, max_new_tokens=8
    )

    result = generate_text(
        model,
        processor,
        inputs,
        InferenceSample(sample_id="sample", prompt="translate"),
        config,
    )

    assert result.token_ids == [21, 22]
    assert model.model.thinker.generate_calls == 1
    assert model.model.run_thinker_calls == 0
    assert model.model.thinker.last_kwargs["use_cache"] is True
    assert result.metrics.first_token_latency_ms is not None


def test_native_thinker_generate_honors_thinker_only_greedy_override():
    model, processor, inputs = _fake_native_generate_stack(tokens=[21, 2])
    config = InferenceConfig(
        model="checkpoint",
        use_thinker_kv_cache=True,
        temperature=0.9,
        top_k=50,
        thinker_temperature=0.0,
        thinker_top_k=0,
    )

    generate_text(
        model,
        processor,
        inputs,
        InferenceSample(sample_id="sample", prompt="translate"),
        config,
    )

    kwargs = model.model.thinker.last_kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs


@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_codec_frame_buffer_roundtrip(use_gpu):
    """Both storage modes must expose the same frame ordering and history."""
    device = torch.device("cuda:0" if use_gpu else "cpu")
    buffer = CodecFrameBuffer(4, 3, device=device, use_gpu=use_gpu)

    buffer.append(
        torch.tensor([7], device=device),
        torch.tensor([[1, 2, 3]], device=device),
    )
    buffer.append(
        torch.tensor([8], device=device),
        torch.tensor([[4, 5, 6]], device=device),
    )

    assert buffer.to_numpy().tolist() == [[7, 1, 2, 3], [8, 4, 5, 6]]
    assert buffer.main_history().tolist() == [[7, 8]]


def test_gpu_codec_buffer_rejects_cpu_device():
    """The GPU optimization must never silently degrade to host storage."""
    with pytest.raises(ValueError, match="CUDA"):
        CodecFrameBuffer(
            4, 3, device=torch.device("cpu"), use_gpu=True
        )


def test_fixed_window_decoder_right_pads_and_trims():
    """Future right-padding must not alter a causal decoder's real prefix."""
    adapter = CompiledStreamingDecoder(
        _fake_12hz_tokenizer(upsample=3),
        window_frames=8,
        compile_fn=lambda function, **kwargs: function,
    )
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])

    waveform, sample_rate = adapter.decode(codes)

    assert sample_rate == 24000
    assert waveform.shape == (9,)
    np.testing.assert_array_equal(
        waveform, np.repeat([1, 3, 5], 3).astype(np.float32)
    )


def test_compiled_decoder_rejects_non_12hz_tokenizer():
    """The fixed-window assumptions must not run against a 25Hz tokenizer."""
    with pytest.raises(ValueError, match="12Hz"):
        CompiledStreamingDecoder(_fake_25hz_tokenizer(), window_frames=8)


def test_optimization_metrics_schema_is_complete():
    """Every run must expose the same phase keys, including baseline runs."""
    metrics = new_optimization_metrics()

    assert set(metrics) == {
        "thinker_generate_ms",
        "thinker_finalize_ms",
        "talker_prefill_ms",
        "talker_step_ms",
        "code_predictor_ms",
        "codec_transfer_ms",
        "stream_decoder_ms",
        "compile_warmup_ms",
    }
    assert all(value == 0.0 for value in metrics.values())
