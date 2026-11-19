import base64
import io
import json
import wave
from pathlib import Path

import numpy as np


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.array([0, 1000, -1000, 0], dtype="<i2").tobytes())
    return buffer.getvalue()


def _events(response) -> list[dict]:
    body = response.get_data(as_text=True)
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


class _FakeRuntime:
    ready = True

    def iter_events(self, audio_path: str, target_lang: str):
        assert Path(audio_path).is_file()
        assert Path(audio_path).suffix == ".wav"
        assert target_lang == "en"
        yield {"type": "translation", "text": "hello", "server_elapsed_ms": 12.5}
        yield {
            "type": "pcm",
            "audio": np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32),
            "sample_rate": 24000,
            "server_elapsed_ms": 20.0,
        }
        yield {
            "type": "metrics",
            "metrics": {"input_ready_to_first_pcm_ms": 20.0},
        }
        yield {"type": "done", "translation": "hello"}


def test_infer_streams_translation_pcm_metrics_and_done_events():
    """Dropping an event would make the latency demo incomplete or unplayable."""
    from demo.web_demo import create_app

    client = create_app(_FakeRuntime()).test_client()
    response = client.post(
        "/api/infer",
        data={
            "target_lang": "en",
            "audio": (io.BytesIO(_wav_bytes()), "sample.wav"),
        },
        content_type="multipart/form-data",
        buffered=True,
    )

    assert response.status_code == 200
    events = _events(response)
    assert [event["type"] for event in events] == [
        "translation", "pcm", "metrics", "done"
    ]
    pcm = events[1]
    assert pcm["sample_rate"] == 24000
    assert "audio" not in pcm
    assert base64.b64decode(pcm["data"]) == np.array(
        [-32767, 0, 16383, 32767], dtype="<i2"
    ).tobytes()


def test_infer_rejects_missing_audio_and_unknown_target_language():
    """Invalid requests must fail before they enter the single-GPU queue."""
    from demo.web_demo import create_app

    client = create_app(_FakeRuntime()).test_client()

    missing = client.post("/api/infer", data={"target_lang": "en"})
    invalid = client.post(
        "/api/infer",
        data={
            "target_lang": "fr",
            "audio": (io.BytesIO(_wav_bytes()), "sample.wav"),
        },
        content_type="multipart/form-data",
    )

    assert missing.status_code == 400
    assert missing.get_json()["error"] == "audio is required"
    assert invalid.status_code == 400
    assert invalid.get_json()["error"] == "target_lang must be 'zh' or 'en'"
