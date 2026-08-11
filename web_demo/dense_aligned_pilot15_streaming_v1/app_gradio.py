#!/usr/bin/env python3
"""Public Gradio launcher for dense-aligned pilot15 iter_0000500."""

from __future__ import annotations

import argparse
import sys

from web_demo.streaming_s2st_r2_v1.audio_io import write_json
from web_demo.true_subsecond_pilot15_streaming_v1 import app_gradio as base_app
from web_demo.true_subsecond_pilot15_streaming_v1.engine import (
    TrueSubsecondStreamingEngine,
)

from .config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7874)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = load_config()
    config.validate_assets(require_export=True)
    engine = TrueSubsecondStreamingEngine(config)
    engine.load()
    demo = base_app.build_demo(config, engine).queue(
        default_concurrency_limit=1, max_size=2
    )
    launched = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=None,
        prevent_thread_lock=True,
        allowed_paths=[str(config.output_root.resolve())],
        blocked_paths=[
            str((config.repo_root / "checkpoints").resolve()),
            str((config.repo_root / "data").resolve()),
            str((config.repo_root / "pretrained_models").resolve()),
        ],
        show_api=False,
        quiet=False,
        max_file_size="160mb",
    )
    local_url = str(launched[1])
    public_url = str(launched[2]) if launched[2] else None
    (config.demo_root / "public_url.txt").write_text(
        (public_url or "") + "\n", encoding="utf-8"
    )
    write_json(
        config.demo_root / "access_info.json",
        {
            "local_url": local_url,
            "public_url": public_url,
            "auth_mode": "public_no_login",
            "model": "dense-aligned pilot15 validation-best iter_0000500",
            "maximum_audio_seconds": config.max_audio_seconds,
            "decision_chunks_ms": [320, 480, 640],
            "stereo": "left=source,right=translation-at-emission-time",
            "true_input_streaming": True,
            "browser_webrtc_live": False,
        },
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={public_url or ''}", flush=True)
    if args.share and not public_url:
        raise RuntimeError("Gradio did not return a public share URL")
    demo.block_thread()
    return local_url, public_url


if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"FATAL={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

