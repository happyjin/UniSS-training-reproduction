"""Verify the authenticated public Gradio tunnel with one real Phase3 request."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from gradio_client import Client, handle_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--direction",
        choices=("中文 → 英文", "英文 → 中文"),
        default="中文 → 英文",
    )
    parser.add_argument("--expected-transcription", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audio = Path(args.audio)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    client = Client(args.url, auth=(args.username, args.password))
    result = client.predict(
        handle_file(str(audio)),
        args.direction,
        True,
        api_name="/translate_phase3_quality",
    )
    transcription, translation, output_audio, result_json, status, chat = result
    if not str(transcription).strip():
        raise RuntimeError("Public demo returned an empty transcription")
    if args.expected_transcription and transcription != args.expected_transcription:
        raise RuntimeError(
            f"Unexpected transcription: expected={args.expected_transcription!r} actual={transcription!r}"
        )
    if not str(translation).strip():
        raise RuntimeError("Public demo returned an empty translation")
    if not Path(output_audio).is_file():
        raise RuntimeError(f"Public demo audio was not downloaded: {output_audio}")
    if not Path(result_json).is_file():
        raise RuntimeError(f"Public demo JSON was not downloaded: {result_json}")
    print(
        json.dumps(
            {
                "url": args.url,
                "direction": args.direction,
                "transcription": transcription,
                "translation": translation,
                "output_audio": output_audio,
                "result_json": result_json,
                "status": status,
                "chat_messages": len(chat),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
