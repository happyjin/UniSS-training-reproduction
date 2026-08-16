#!/usr/bin/env python3
"""Freeze and verify the canonical Phase3/WhisperVQ artifact identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--whispervq-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    native = Path(args.native_checkpoint).resolve()
    hf = Path(args.hf_checkpoint).resolve()
    whisper = Path(args.whispervq_checkpoint).resolve()
    output = Path(args.output_json).resolve()
    required = {
        "native_metadata": native / "metadata.json",
        "native_dist_metadata": native / ".metadata",
        "hf_config": hf / "config.json",
        "hf_export_manifest": hf / "export_manifest.json",
        "hf_model": hf / "model.safetensors",
        "whisper_config": whisper / "config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical artifacts: {missing}")
    native_shards = sorted(native.glob("__*_0.distcp"))
    if len(native_shards) != 8:
        raise ValueError(f"expected eight native distcp shards, found {len(native_shards)}")

    config = _json(required["hf_config"])
    export = _json(required["hf_export_manifest"])
    whisper_config = _json(required["whisper_config"])
    checks = {
        "native_iteration_is_9075": native.name == "iter_0009075",
        "native_has_eight_distcp_shards": len(native_shards) == 8,
        "export_points_to_native_9075": Path(export["source_checkpoint"]).resolve() == native,
        "hf_model_sha_matches_manifest": _sha256(required["hf_model"])
        == export["weight_files"][0]["sha256"],
        "hf_layers_24": config.get("num_hidden_layers") == 24,
        "hf_hidden_896": config.get("hidden_size") == 896,
        "hf_heads_14_kv2": config.get("num_attention_heads") == 14
        and config.get("num_key_value_heads") == 2,
        "hf_vocab_180480": config.get("vocab_size") == 180480,
        "hf_cache_enabled": config.get("use_cache") is True,
        "whisper_pool_and_quantize_at_16": whisper_config.get("pooling_position") == 16
        and whisper_config.get("quantize_position") == 16,
        "whisper_causal_convolution": whisper_config.get("encoder_causal_convolution")
        is True,
    }
    result = {
        "schema_version": "uniss_stage00_canonical_artifact_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "native_checkpoint": str(native),
        "native_shards": [
            {"name": path.name, "size": path.stat().st_size} for path in native_shards
        ],
        "native_metadata_sha256": _sha256(required["native_metadata"]),
        "native_dist_metadata_sha256": _sha256(required["native_dist_metadata"]),
        "hf_checkpoint": str(hf),
        "hf_config_sha256": _sha256(required["hf_config"]),
        "hf_model_sha256": _sha256(required["hf_model"]),
        "hf_export_manifest": export,
        "whispervq_checkpoint": str(whisper),
        "whisper_config_sha256": _sha256(required["whisper_config"]),
        "whisper_effective_pre_vq_layers": whisper_config.get("quantize_position"),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

