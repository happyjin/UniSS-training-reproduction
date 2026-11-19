#!/usr/bin/env python3
"""Fail-fast contract checks for Batch 4.7 Stage 2.5 on UniST V2."""
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
from wds_audio import parse_wds_uri, read_wds_audio_bytes  # noqa: E402


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def expect(name: str, expected: str) -> None:
    actual = env(name)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def validate_model_and_launch() -> None:
    expected_checkpoint = (
        REPO_ROOT
        / "output/qwen_omnispeak_output/batch4.7/batch4.7-20260728-083603"
        / "stage2_adapter_talker/v0-20260728-124644/checkpoint-13900"
    ).resolve()
    model_path = Path(env("MODEL_PATH")).resolve()
    if model_path != expected_checkpoint:
        raise AssertionError(f"unexpected model source: {model_path}")
    for filename in ("config.json", "args.json", "model.safetensors.index.json"):
        if not (model_path / filename).is_file():
            raise FileNotFoundError(model_path / filename)

    source_args = json.loads((model_path / "args.json").read_text())
    if set(source_args.get("trainable_parameters") or []) != {
        "model.thinker_adapter", "model.talker"
    }:
        raise AssertionError("checkpoint-13900 is not the expected Stage 2 checkpoint")
    if float(source_args.get("num_train_epochs", -1)) != 5.0:
        raise AssertionError("checkpoint-13900 source stage was not the 5-epoch Stage 2")

    expected = {
        "BATCH4_7_STAGE": "stage2_5_unist",
        "MODEL_TYPE": "qwen2_5_3b_omnispeak_a",
        "TALKER_METHOD": "reply_add",
        "TALKER_TRAINING_SCOPE": "whole",
        "UNFREEZE_THINKER": "0",
        "NUM_TRAIN_EPOCHS": "1",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "CODEC_EOS_LOSS_WEIGHT": "1.0",
        "CODEC_HISTORY_CORRUPTION_PROB": "0.0",
        "RESIDUAL_LOSS_WEIGHT": "0.3",
        "QWEN3_TTS_MINOR_RANGE_WARNING_THRESHOLD": "1.05",
        "MAX_CODEC_FRAMES": "737",
        "AR_EVAL_MAX_CODEC_FRAMES": "737",
        "AR_EVAL_PATHOLOGICAL_FRAMES": "737",
        "GROUP_BY_LENGTH": "false",
        "SPLIT_DATASET_RATIO": "0.0005",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NPROC_PER_NODE": "8",
        "DATA_SEED": "42",
        "AR_EVAL_SEED": "42",
    }
    for name, value in expected.items():
        expect(name, value)
    if os.environ.get("RESUME_FROM"):
        raise AssertionError("Stage 2.5 must not restore optimizer/scheduler state")


def validate_unist() -> None:
    dataset_root = Path(env("WDS_MANIFEST_DIR")).resolve()
    expected_root = (
        WORKSPACE_ROOT / "data/unist/converted/unist_qwen3_s2s_v2"
    ).resolve()
    if dataset_root != expected_root:
        raise AssertionError(f"unexpected UniST root: {dataset_root}")
    committed = json.loads((dataset_root / "_COMMITTED.json").read_text())
    if int(committed.get("rows", -1)) != 19_227_252:
        raise AssertionError(f"unexpected UniST row count: {committed.get('rows')}")
    quality_filter = committed.get("quality_filter") or {}
    if quality_filter.get("version") != "unist_quality_v1":
        raise AssertionError("UniST quality filter marker is missing")
    if int(quality_filter.get("filtered_rows", -1)) != 98_372:
        raise AssertionError(f"unexpected quality reject count: {quality_filter}")

    manifests = sorted(glob.glob(env("BRIDGE_ALIGNMENT_DATA")))
    if len(manifests) != 208:
        raise AssertionError(f"expected 208 manifest shards, found {len(manifests)}")

    preprocessor = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION)
    seen: dict[str, str] = {}
    for manifest in manifests:
        with open(manifest, encoding="utf-8") as handle:
            row = json.loads(handle.readline())
        lang = str(row.get("target_lang", "")).lower()
        if lang not in {"en", "zh"} or lang in seen:
            continue
        expected_prompt = s2s_task_prompt(lang)
        mapped = preprocessor.preprocess(row)
        if mapped["messages"][0] != {"role": "system", "content": expected_prompt}:
            raise AssertionError(f"training prompt mismatch for target_lang={lang}")
        codec = mapped.get("target_codec_codes")
        if not codec or len(codec) != int(row["codec_len"]):
            raise AssertionError(f"invalid codec length for {row.get('record_id')}")
        if any(not isinstance(frame, list) or len(frame) != 16 for frame in codec):
            raise AssertionError(f"target codec is not [T,16] for {row.get('record_id')}")
        audio_uri = mapped["audios"][0]
        parse_wds_uri(audio_uri)
        payload = read_wds_audio_bytes(audio_uri)
        if not payload:
            raise AssertionError(f"empty WDS audio for {row.get('record_id')}")
        seen[lang] = row["record_id"]
        if len(seen) == 2:
            break
    if set(seen) != {"en", "zh"}:
        raise AssertionError(f"both translation directions not found: {sorted(seen)}")

    val_rows = round(int(committed["rows"]) * float(env("SPLIT_DATASET_RATIO")))
    print(f"[batch4.7-stage2.5-preflight] manifests={len(manifests)} rows={committed['rows']}")
    print(f"[batch4.7-stage2.5-preflight] random_val≈{val_rows} data_seed={env('DATA_SEED')}")
    print(f"[batch4.7-stage2.5-preflight] directions={seen}")


def main() -> None:
    validate_model_and_launch()
    validate_unist()
    print(
        "[batch4.7-stage2.5-preflight] checkpoint weights only; fresh optimizer/scheduler; "
        "adapter+whole-talker; thinker frozen: OK"
    )


if __name__ == "__main__":
    main()
