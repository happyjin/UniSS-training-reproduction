#!/usr/bin/env python3
"""Fail-fast contract checks for the Batch 4.6 adapter+talker speed test."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom"))

from dataset import BridgeAlignmentPreprocessor, TASK_S2S_TRANSLATION  # noqa: E402
from task_prompts import s2s_task_prompt  # noqa: E402
from wds_audio import read_wds_audio_bytes  # noqa: E402


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def expect(name: str, expected: str) -> None:
    actual = env(name)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def validate_launch() -> None:
    expected_checkpoint = (
        REPO_ROOT
        / "output/qwen_omnispeak_output/bidirectionalS2S"
        / "upstream_reply_add-thinker-talker-3b-lr1e-5-bs64x8-s2s"
        / "v1-20260726-021439/checkpoint-13500"
    ).resolve()
    model_path = Path(env("MODEL_PATH")).resolve()
    if model_path != expected_checkpoint:
        raise AssertionError(f"unexpected model source: {model_path}")
    for filename in ("args.json", "config.json", "model.safetensors.index.json"):
        if not (model_path / filename).is_file():
            raise FileNotFoundError(model_path / filename)

    expected = {
        "MODEL_TYPE": "qwen2_5_3b_omnispeak_a",
        "TALKER_METHOD": "reply_add",
        "TALKER_TRAINING_SCOPE": "whole",
        "UNFREEZE_THINKER": "0",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "NPROC_PER_NODE": "8",
        "SPEED_TEST_MAX_STEPS": "1000",
        "MAX_LENGTH": "4096",
        "MAX_CODEC_FRAMES": "384",
        "CODEC_EOS_LOSS_WEIGHT": "10",
        "CODEC_HISTORY_CORRUPTION_PROB": "0.05",
        "RESIDUAL_LOSS_WEIGHT": "0.3",
        "BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT": "true",
        "WDS_AUDIO_MODE": "required",
        "WDS_AUDIO_STRICT": "true",
        "EVAL_STRATEGY": "no",
        "AR_EVAL_ENABLED": "false",
    }
    for name, value in expected.items():
        expect(name, value)
    if os.environ.get("RESUME_FROM"):
        raise AssertionError("speed test must warm-start weights, not resume optimizer state")


def validate_data() -> None:
    manifest_dir = Path(env("WDS_MANIFEST_DIR")).resolve()
    expected_dir = (
        WORKSPACE_ROOT
        / "data/manifests/common_voice_bidirectional_ms_swift_codec_20260727_v1"
    ).resolve()
    if manifest_dir != expected_dir:
        raise AssertionError(f"unexpected manifest directory: {manifest_dir}")

    committed = json.loads((manifest_dir / "_COMMITTED.json").read_text())
    if committed.get("status") != "committed":
        raise AssertionError(f"manifest is not committed: {committed}")
    if int(committed.get("record_count", -1)) != 2_553_191:
        raise AssertionError(f"unexpected record count: {committed.get('record_count')}")

    manifests = sorted(glob.glob(env("BRIDGE_ALIGNMENT_DATA")))
    if len(manifests) != 25:
        raise AssertionError(f"expected 25 manifest shards, found {len(manifests)}")

    preprocessor = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION)
    seen: dict[str, str] = {}
    for path in (manifests[0], manifests[-1]):
        with open(path, "rb") as handle:
            first = handle.readline()
            handle.seek(0, os.SEEK_END)
            pos = max(0, handle.tell() - 2)
            while pos > 0:
                handle.seek(pos)
                if handle.read(1) == b"\n":
                    break
                pos -= 1
            handle.seek(pos + 1 if pos > 0 else 0)
            candidates = (first, handle.readline())
        for raw in candidates:
            if not raw.strip():
                continue
            row = json.loads(raw)
            lang = str(row.get("target_lang", "")).lower()
            if lang not in {"en", "zh"} or lang in seen:
                continue
            mapped = preprocessor.preprocess(row)
            prompt = s2s_task_prompt(lang)
            if mapped["messages"][0] != {"role": "system", "content": prompt}:
                raise AssertionError(f"canonical prompt mismatch for target_lang={lang}")
            for uri in (mapped["audios"][0], mapped["target_audio"]):
                if not read_wds_audio_bytes(uri):
                    raise AssertionError(f"empty indexed WDS payload: {uri}")
            seen[lang] = row["record_id"]

    if set(seen) != {"en", "zh"}:
        raise AssertionError(f"both directions not found in boundary rows: {seen}")
    print(
        f"[batch4.6-speed-preflight] manifests={len(manifests)} "
        f"rows={committed['record_count']} directions={seen}"
    )


def main() -> None:
    validate_launch()
    validate_data()
    print(
        "[batch4.6-speed-preflight] checkpoint-13500 weight warm-start; "
        "adapter+whole-talker trainable; thinker frozen; 64x8; 1000 steps: OK"
    )


if __name__ == "__main__":
    main()
