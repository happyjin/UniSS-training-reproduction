from pathlib import Path
import threading
import time
from types import SimpleNamespace


def _args(tmp_path: Path):
    return SimpleNamespace(
        model_type="qwen2_5_3b_omnispeak_a",
        checkpoint=str(tmp_path / "checkpoint"),
        device="cuda:3",
        dtype="bf16",
        attn_impl="flash_attention_2",
        talker_layout="reply_add",
        codec_tokenizer=None,
        codec_tokenizer_device=None,
        speaker_encoder=None,
        speaker_prompt=True,
        max_new_tokens=256,
        max_codec_frames=384,
    )


def test_from_args_enables_the_verified_online_single_gpu_stack(tmp_path):
    """A missing flag would silently turn the latency demo into another inference path."""
    from demo.streaming_runtime import StreamingDemoRuntime

    runtime = StreamingDemoRuntime.from_args(_args(tmp_path))
    config = runtime.config

    assert config.device == "cuda:3"
    assert config.codec_tokenizer_device == "cuda:3"
    assert config.streaming is True
    assert config.stream_emit_every == 4
    assert config.stream_decode_window == 80
    assert config.use_kv_cache is True
    assert config.use_thinker_kv_cache is True
    assert config.use_code_predictor_kv_cache is True
    assert config.compile_code_predictor is True
    assert config.use_gpu_codec_buffer is True
    assert config.compile_streaming_decoder is True
    assert config.online_thinker_talker is True
    assert config.online_talker_start_tokens == 1
    assert config.compile_talker is True
    assert config.predictor_frame_batch == 4
    assert config.latency_profiling == "off"
    assert config.thinker_temperature == 0.0
    assert config.thinker_top_k == 0
    assert config.temperature == 0.9
    assert config.top_k == 50
    assert config.repetition_penalty == 1.05


def test_iter_events_forwards_worker_events_in_order(tmp_path):
    """Losing queue order would attach PCM or metrics to the wrong UI state."""
    from demo.streaming_runtime import StreamingDemoRuntime

    runtime = StreamingDemoRuntime.from_args(_args(tmp_path))
    runtime.ready = True

    def run_request(audio_path, target_lang, emit):
        assert audio_path == "sample.wav"
        assert target_lang == "en"
        emit({"type": "translation", "text": "hello"})
        emit({"type": "done", "translation": "hello"})

    runtime._run_request = run_request

    assert list(runtime.iter_events("sample.wav", "en")) == [
        {"type": "translation", "text": "hello"},
        {"type": "done", "translation": "hello"},
    ]


def test_model_asset_falls_back_from_worktree_to_main_checkout(tmp_path):
    """Worktree demos must not point speaker/codec loaders at a missing local models dir."""
    from demo.streaming_runtime import resolve_model_asset

    main_repo = tmp_path / "repo"
    worktree = main_repo / ".worktrees" / "feature"
    asset = main_repo / "models" / "codec"
    worktree.mkdir(parents=True)
    asset.mkdir(parents=True)

    assert resolve_model_asset("codec", repo=worktree) == asset


def test_closing_event_stream_waits_for_the_gpu_worker(tmp_path):
    """Early HTTP disconnect must not delete the input WAV while inference still uses it."""
    from demo.streaming_runtime import StreamingDemoRuntime

    runtime = StreamingDemoRuntime.from_args(_args(tmp_path))
    runtime.ready = True
    finished = threading.Event()

    def run_request(audio_path, target_lang, emit):
        del audio_path, target_lang
        emit({"type": "started"})
        time.sleep(0.05)
        finished.set()

    runtime._run_request = run_request
    events = runtime.iter_events("sample.wav", "en")

    assert next(events) == {"type": "started"}
    events.close()

    assert finished.is_set()
