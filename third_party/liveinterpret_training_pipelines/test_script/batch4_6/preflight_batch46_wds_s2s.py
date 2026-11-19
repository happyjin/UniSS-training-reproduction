#!/usr/bin/env python3
"""Fail-fast checks for the batch 4.6 WDS S2S launch."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom"))

from dataset import BridgeAlignmentPreprocessor, TASK_S2S_TRANSLATION  # noqa: E402
from task_prompts import s2s_task_prompt  # noqa: E402


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def assert_close(name: str, actual: object, expected: float) -> None:
    if abs(float(actual) - expected) > 1e-12:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def validate_checkpoint() -> None:
    checkpoint = Path(require_env("MODEL_PATH"))
    args_path = checkpoint / "args.json"
    if not checkpoint.is_dir() or not args_path.is_file():
        raise FileNotFoundError(f"invalid checkpoint: {checkpoint}")
    args = json.loads(args_path.read_text())

    expected = {
        "model_type": "qwen2_5_3b_omnispeak_a",
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "num_train_epochs": 5.0,
        "warmup_ratio": 0.03,
        "max_length": 4096,
        "dataset_num_proc": 16,
        "dataloader_num_workers": 16,
        "dataloader_persistent_workers": True,
        "group_by_length": False,
        "gradient_checkpointing": True,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise AssertionError(
                f"checkpoint-13500 {key}: expected {value!r}, got {args.get(key)!r}")

    trainable = set(args.get("trainable_parameters") or [])
    expected_trainable = {"model.thinker_adapter", "model.talker", "model.thinker"}
    if trainable != expected_trainable:
        raise AssertionError(
            f"checkpoint-13500 trainable scope changed: {sorted(trainable)}")


def validate_launch_settings() -> None:
    assert_close("CODEC_EOS_LOSS_WEIGHT", require_env("CODEC_EOS_LOSS_WEIGHT"), 10.0)
    assert_close(
        "CODEC_HISTORY_CORRUPTION_PROB",
        require_env("CODEC_HISTORY_CORRUPTION_PROB"),
        0.05,
    )
    if require_env("BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT").lower() != "true":
        raise AssertionError("BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT must be true")
    if require_env("TALKER_TRAINING_SCOPE") != "whole":
        raise AssertionError("TALKER_TRAINING_SCOPE must be whole")
    if require_env("UNFREEZE_THINKER").lower() not in {"1", "true"}:
        raise AssertionError("UNFREEZE_THINKER must preserve the checkpoint-13500 scope")
    if os.environ.get("RESUME_FROM"):
        raise AssertionError(
            "batch 4.6 must warm-start through MODEL_PATH, not resume optimizer state")


def validate_real_prompts() -> None:
    paths = sorted(glob.glob(require_env("BRIDGE_ALIGNMENT_DATA")))
    if not paths:
        raise FileNotFoundError("BRIDGE_ALIGNMENT_DATA did not match any manifest")

    preprocessor = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION)
    seen: dict[str, str] = {}
    for path in paths:
        # Rows are direction-grouped, so a sequential search for the second
        # direction can traverse millions of records. The first and last row
        # of each committed shard give a constant-time real-data check.
        with open(path, "rb") as handle:
            first = handle.readline()
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            pos = max(0, end - 2)
            while pos > 0:
                handle.seek(pos)
                if handle.read(1) == b"\n":
                    break
                pos -= 1
            handle.seek(pos + 1 if pos > 0 else 0)
            last = handle.readline()
        for raw_line in (first, last):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            target_lang = str(row.get("target_lang") or "").lower()
            if target_lang in seen:
                continue
            mapped = preprocessor.preprocess(row)
            if mapped is None:
                continue
            expected = s2s_task_prompt(target_lang)
            actual = mapped["messages"][0]
            if actual != {"role": "system", "content": expected}:
                raise AssertionError(
                    f"wrong {target_lang} prompt: expected {expected!r}, got {actual!r}")
            if mapped.get("task_prompt") != "":
                raise AssertionError("legacy task_prompt survived normalization")
            seen[target_lang] = expected
            if set(seen) == {"en", "zh"}:
                break
        if set(seen) == {"en", "zh"}:
            break

    if set(seen) != {"en", "zh"}:
        raise AssertionError(f"did not find both translation directions; found {sorted(seen)}")
    for language in ("en", "zh"):
        print(f"[batch4.6-preflight] target={language} prompt={seen[language]}")


def main() -> None:
    validate_checkpoint()
    validate_launch_settings()
    validate_real_prompts()
    print("[batch4.6-preflight] checkpoint/hyperparameters/loss/prompt: OK")


if __name__ == "__main__":
    main()
