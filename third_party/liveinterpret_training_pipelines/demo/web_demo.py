#!/usr/bin/env python3
"""Single-GPU Flask demo for online Qwen-OmniSpeak S2S inference."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from pydub import AudioSegment


DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
TARGET_LANGUAGES = {"zh", "en"}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _serialize_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    if payload.get("type") == "pcm":
        waveform = np.asarray(payload.pop("audio"), dtype=np.float32).reshape(-1)
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
        payload["data"] = base64.b64encode(pcm.tobytes()).decode("ascii")
    return payload


def _canonicalize_upload(upload, directory: Path) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".bin"
    source = directory / f"upload{suffix}"
    output = directory / "input.wav"
    upload.save(source)
    segment = (
        AudioSegment.from_file(source)
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )
    segment.export(output, format="wav")
    return output


def create_app(runtime) -> Flask:
    """Create the HTTP transport around a loaded streaming runtime."""
    app = Flask(__name__, static_folder=str(DEMO_DIR))

    @app.get("/")
    def index():
        return send_from_directory(DEMO_DIR, "web_demo.html")

    @app.get("/health")
    def health():
        return jsonify({"ready": bool(getattr(runtime, "ready", False))})

    @app.post("/api/infer")
    def infer():
        upload = request.files.get("audio")
        if upload is None or not upload.filename:
            return jsonify({"error": "audio is required"}), 400
        target_lang = (request.form.get("target_lang") or "").strip().lower()
        if target_lang not in TARGET_LANGUAGES:
            return jsonify({"error": "target_lang must be 'zh' or 'en'"}), 400
        if not bool(getattr(runtime, "ready", False)):
            return jsonify({"error": "model is not ready"}), 503

        temporary = tempfile.TemporaryDirectory(prefix="liveinterpret-demo-")
        try:
            audio_path = _canonicalize_upload(upload, Path(temporary.name))
        except Exception as exc:
            temporary.cleanup()
            return jsonify({"error": f"could not decode audio: {exc}"}), 400

        @stream_with_context
        def event_stream():
            try:
                for event in runtime.iter_events(str(audio_path), target_lang):
                    yield _sse(_serialize_event(event))
            except GeneratorExit:
                raise
            except Exception as exc:
                yield _sse({"type": "error", "error": str(exc)})
            finally:
                temporary.cleanup()

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--talker-layout", choices=("reply_add", "full_add"), default="reply_add")
    parser.add_argument("--codec-tokenizer")
    parser.add_argument("--codec-tokenizer-device")
    parser.add_argument("--speaker-encoder")
    parser.add_argument("--speaker-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-audio")
    parser.add_argument("--warmup-target-lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-codec-frames", type=int, default=384)
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    from demo.streaming_runtime import StreamingDemoRuntime

    args = parse_args()
    runtime = StreamingDemoRuntime.from_args(args)
    runtime.load()
    if args.warmup_audio:
        runtime.warmup(args.warmup_audio, args.warmup_target_lang)
    else:
        print("Warning: --warmup-audio was not provided; the first request includes compile warmup.")
    create_app(runtime).run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
