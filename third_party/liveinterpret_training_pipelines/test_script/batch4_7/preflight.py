#!/usr/bin/env python3
"""Fail-fast checks for the batch 4.7 clean-data S2S launch."""
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
from inference.s2s_infer import _default_prompt  # noqa: E402
from task_prompts import s2s_task_prompt  # noqa: E402


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def assert_close(name: str, actual: object, expected: float) -> None:
    if abs(float(actual) - expected) > 1e-12:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def validate_model_source() -> None:
    model_path = Path(require_env("MODEL_PATH"))
    config_path = model_path / "config.json"
    if not model_path.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"invalid model source: {model_path}")

    stage = os.environ.get("BATCH4_7_STAGE", "single")
    args_path = model_path / "args.json"
    if stage in {"stage1_adapter", "single"}:
        expected_base = (REPO_ROOT / "models" / "qwen2_5_3b_omnispeak_a").resolve()
        if model_path.resolve() != expected_base:
            raise AssertionError(
                f"batch 4.7 stage 1 must start from base model {expected_base}, "
                f"got {model_path.resolve()}")
        if args_path.exists():
            raise AssertionError("batch 4.7 stage 1 must not load a training checkpoint")
        config = json.loads(config_path.read_text())
        expected_config = {
            "model_type": "qwen_omnispeak",
            "variant": "A",
            "use_upstream_talker_input": True,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                raise AssertionError(
                    f"base model {key}: expected {value!r}, got {config.get(key)!r}")
        if not (model_path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("base model weight index is missing")
        print(f"[batch4.7-preflight] initial_model={model_path} (no S2S checkpoint)")
        return

    if not args_path.is_file():
        raise FileNotFoundError(f"invalid previous-stage checkpoint: {model_path}")
    args = json.loads(args_path.read_text())
    expected = {
        "model_type": "qwen2_5_3b_omnispeak_a",
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
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
                f"previous-stage checkpoint {key}: expected {value!r}, "
                f"got {args.get(key)!r}")
    expected_source = os.environ.get("BATCH4_7_EXPECTED_SOURCE_TRAINABLE")
    if expected_source:
        trainable = set(args.get("trainable_parameters") or [])
        expected_trainable = set(expected_source.split(","))
        if trainable != expected_trainable:
            raise AssertionError(
                f"source checkpoint trainable scope: expected "
                f"{sorted(expected_trainable)}, got {sorted(trainable)}")
    expected_source_epochs = os.environ.get("BATCH4_7_EXPECTED_SOURCE_EPOCHS")
    if expected_source_epochs:
        assert_close(
            "source checkpoint num_train_epochs",
            args.get("num_train_epochs"), float(expected_source_epochs))


def validate_launch_settings() -> None:
    if require_env("QWEN_OMNISPEAK_TEXT_LOSS").lower() != "true":
        raise AssertionError("Thinker text CE must be enabled")
    assert_close(
        "QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT",
        require_env("QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT"), 1.0)
    assert_close("CODEC_EOS_LOSS_WEIGHT", require_env("CODEC_EOS_LOSS_WEIGHT"), 1.0)
    assert_close(
        "CODEC_HISTORY_CORRUPTION_PROB",
        require_env("CODEC_HISTORY_CORRUPTION_PROB"), 0.0)
    assert_close("RESIDUAL_LOSS_WEIGHT", require_env("RESIDUAL_LOSS_WEIGHT"), 0.3)
    if require_env("RESIDUAL_LOSS").lower() != "true":
        raise AssertionError("RESIDUAL_LOSS must be true")
    if require_env("BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT").lower() != "true":
        raise AssertionError("BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT must be true")
    if require_env("LR_SCHEDULER_TYPE") != "constant_with_warmup":
        raise AssertionError("batch 4.7 must keep LR constant after warmup")
    scope = require_env("TALKER_TRAINING_SCOPE")
    if scope not in {"layers", "whole"}:
        raise AssertionError(f"invalid TALKER_TRAINING_SCOPE={scope!r}")
    if scope == "layers" and require_env("UNFREEZE_TALKER_LAYERS") != "0":
        raise AssertionError("adapter stage must not unfreeze talker layers")
    if require_env("UNFREEZE_MTP").lower() not in {"0", "false"}:
        raise AssertionError("standalone MTP unfreezing is not part of batch 4.7")
    if require_env("UNFREEZE_CODEC_HEAD").lower() not in {"0", "false"}:
        raise AssertionError("standalone codec-head unfreezing is not part of batch 4.7")
    stage = os.environ.get("BATCH4_7_STAGE", "single")
    expected = {
        "stage1_adapter": ("layers", False),
        "stage2_adapter_talker": ("whole", False),
        "stage3_thinker_adapter_talker": ("whole", True),
        "single": ("whole", True),
    }
    if stage not in expected:
        raise AssertionError(f"unknown BATCH4_7_STAGE={stage!r}")
    expected_scope, expected_thinker = expected[stage]
    actual_thinker = require_env("UNFREEZE_THINKER").lower() in {"1", "true"}
    if (scope, actual_thinker) != (expected_scope, expected_thinker):
        raise AssertionError(
            f"{stage} expected scope/thinker={(expected_scope, expected_thinker)}, "
            f"got {(scope, actual_thinker)}")
    resume_from = os.environ.get("RESUME_FROM")
    allow_stage1_resume = (
        os.environ.get("BATCH4_7_ALLOW_STAGE1_RESUME", "false").lower() == "true")
    if resume_from:
        if not allow_stage1_resume or stage != "stage1_adapter":
            raise AssertionError("only an explicit Stage 1 resume is allowed")
        checkpoint = Path(resume_from)
        state_path = checkpoint / "trainer_state.json"
        args_path = checkpoint / "args.json"
        if not state_path.is_file() or not args_path.is_file():
            raise FileNotFoundError(f"incomplete resume checkpoint: {checkpoint}")
        state = json.loads(state_path.read_text())
        checkpoint_args = json.loads(args_path.read_text())
        start_step = int(state.get("global_step", -1))
        target_step = int(require_env("BATCH4_7_RESUME_TARGET_STEP"))
        if start_step < 0 or target_step <= start_step:
            raise AssertionError(
                f"invalid resume step range: {start_step} -> {target_step}")
        if set(checkpoint_args.get("trainable_parameters") or []) != {
                "model.thinker_adapter"}:
            raise AssertionError("resume checkpoint is not adapter-only Stage 1")
        expected_output = checkpoint.parent.resolve()
        actual_output = Path(require_env("OUTPUT_DIR")).resolve()
        if actual_output != expected_output:
            raise AssertionError(
                f"resume must reuse output directory {expected_output}, got {actual_output}")
        print(
            f"[batch4.7-preflight] resume={checkpoint} "
            f"steps={start_step}->{target_step} output={actual_output}")
    elif allow_stage1_resume:
        raise AssertionError("BATCH4_7_ALLOW_STAGE1_RESUME=true but RESUME_FROM is unset")
    if os.environ.get("BATCH4_7_MULTISTAGE_CONTRACT", "false").lower() == "true":
        assert_close("STAGE1_EPOCHS", require_env("STAGE1_EPOCHS"), 2.0)
        assert_close("STAGE2_EPOCHS", require_env("STAGE2_EPOCHS"), 5.0)
        if float(require_env("STAGE3_EPOCHS")) <= 0:
            raise AssertionError("STAGE3_EPOCHS must be positive")


def validate_clean_manifest_and_prompts() -> None:
    manifest_dir = Path(require_env("WDS_MANIFEST_DIR"))
    commit = json.loads((manifest_dir / "_COMMITTED.json").read_text())
    if commit.get("record_count") != 1_437_701:
        raise AssertionError(f"unexpected clean manifest count: {commit.get('record_count')}")
    paths = sorted(glob.glob(require_env("BRIDGE_ALIGNMENT_DATA")))
    if len(paths) != 25:
        raise AssertionError(f"expected 25 clean manifest parts, found {len(paths)}")

    preprocessor = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION)
    seen: dict[str, str] = {}
    for path in (paths[0], paths[-1]):
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
            last = handle.readline()
        for raw_line in (first, last):
            row = json.loads(raw_line)
            target_lang = str(row["target_lang"]).lower()
            mapped = preprocessor.preprocess(row)
            expected = s2s_task_prompt(target_lang)
            if mapped["messages"][0] != {"role": "system", "content": expected}:
                raise AssertionError(f"training prompt mismatch for {target_lang}")
            if mapped.get("task_prompt") != "":
                raise AssertionError("legacy task_prompt survived training normalization")
            if _default_prompt(target_lang) != expected:
                raise AssertionError(f"inference prompt mismatch for {target_lang}")
            if not mapped.get("target_text") or not mapped.get("target_codec_codes"):
                raise AssertionError("text/main/residual codec supervision is incomplete")
            seen[target_lang] = expected
    if set(seen) != {"en", "zh"}:
        raise AssertionError(f"both directions not found: {sorted(seen)}")
    for language in ("en", "zh"):
        print(f"[batch4.7-preflight] target={language} prompt={seen[language]}")


def main() -> None:
    validate_model_source()
    validate_launch_settings()
    validate_clean_manifest_and_prompts()
    if os.environ.get("BATCH4_7_MULTISTAGE_CONTRACT", "false").lower() == "true":
        print(
            "[batch4.7-preflight] multistage=adapter(2) -> "
            "adapter+talker(5) -> thinker+adapter+talker"
            f"({os.environ['STAGE3_EPOCHS']})")
    print("[batch4.7-preflight] clean-data/text+main+residual/prompt: OK")


if __name__ == "__main__":
    main()
