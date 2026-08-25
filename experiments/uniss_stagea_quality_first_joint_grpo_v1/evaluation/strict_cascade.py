#!/usr/bin/env python3
"""Adapter-routed strict-causal cascade for auditable listening examples.

The historical Stage-A cascade remains immutable.  This module imports it as a
read-only runtime and routes its three generation families as required by the
comparison protocol: ASR adapter off, incremental MT and semantic TTS adapter
on.  A missing adapter checkpoint gives the paired immutable Stage-A baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.hf_routed_lora import (
    RoutedHFLoRA,
    load_model_and_adapter,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer


CHUNKS_MS = (160, 320, 640, 1280)


def _load_runtime(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "uniss_stage_a_strict_cascade_readonly", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strict cascade runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_for_prompt(prompt_ids: Sequence[int]) -> bool:
    """Return whether the policy adapter is active for a Phase3 task prompt."""
    values = {int(value) for value in prompt_ids}
    if c.TOKEN_TASK_ASR in values:
        return False
    if (
        c.TOKEN_TASK_S2T_TRANSLATION in values
        or c.TOKEN_TASK_T2T_TRANSLATION in values
        or c.TOKEN_TASK_TTS in values
    ):
        return True
    raise ValueError("strict cascade prompt has no recognized ASR/MT/TTS family")


def _indexed_record(path: Path, record_index: int) -> dict[str, object]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing JSONL offset index: {path}")
    if not 0 <= int(record_index) < len(offsets):
        raise IndexError(f"record {record_index} is outside {path}")
    with path.open("rb") as handle:
        handle.seek(int(offsets[int(record_index)]))
        return json.loads(handle.readline())


def _selection_rows(
    selection_path: Path,
    manifest_path: Path,
    sample_ids: Sequence[str],
) -> list[dict[str, object]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = list(selection["records"])
    requested = set(sample_ids)
    if requested:
        selected = [row for row in selected if str(row["sample_id"]) in requested]
        observed = {str(row["sample_id"]) for row in selected}
        if observed != requested:
            raise ValueError(f"selection is missing requested IDs: {sorted(requested-observed)}")
    rows: list[dict[str, object]] = []
    for selected_row in selected:
        row = _indexed_record(manifest_path, int(selected_row["record_index"]))
        if str(row["id"]) != str(selected_row["sample_id"]):
            raise RuntimeError("selection and indexed validation manifest differ")
        rows.append(row)
    return rows


def _legacy_rows(
    results_path: Path,
    fixed_speaker: Sequence[int],
    sample_ids: Sequence[str],
) -> list[dict[str, object]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    requested = set(sample_ids)
    rows: list[dict[str, object]] = []
    for value in payload["results"]:
        sample_id = str(value["sample_id"])
        if requested and sample_id not in requested:
            continue
        source = results_path.parent / sample_id / "source.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "id": sample_id,
                "src_lang": str(value["src_lang"]),
                "tgt_lang": str(value["tgt_lang"]),
                "source_audio": str(source.resolve()),
                "transcription": str(value["reference_transcription"]),
                "translation": str(value["reference_translation"]),
                "bicodec_global": [int(token) for token in fixed_speaker],
            }
        )
    observed = {str(row["id"]) for row in rows}
    if requested and observed != requested:
        raise ValueError(f"legacy results are missing requested IDs: {sorted(requested-observed)}")
    return rows


def _external_rows(
    protocol_path: Path,
    fixed_speaker: Sequence[int],
    sample_ids: Sequence[str],
) -> list[dict[str, object]]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    requested = set(sample_ids)
    rows: list[dict[str, object]] = []
    for value in payload["records"]:
        sample_id = str(value["sample_id"])
        if requested and sample_id not in requested:
            continue
        source = Path(str(value["source_audio"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "id": sample_id,
                "src_lang": str(value["src_lang"]),
                "tgt_lang": str(value["tgt_lang"]),
                "source_audio": str(source.resolve()),
                "transcription": "unreferenced external audio",
                "translation": "unreferenced external audio",
                "bicodec_global": [int(token) for token in fixed_speaker],
                "_unreferenced_external": True,
            }
        )
    observed = {str(row["id"]) for row in rows}
    if requested and observed != requested:
        raise ValueError(f"external protocol is missing requested IDs: {sorted(requested-observed)}")
    return rows


def _load_models(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("strict routed cascade requires CUDA")
    if args.adapter_checkpoint is None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_hf, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_hf,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval().requires_grad_(False)
        controller = RoutedHFLoRA(model, {}, scale=2.0)
        adapter_manifest: dict[str, object] = {
            "enabled": False,
            "route_semantics": "immutable_stage_a_baseline",
            "base_hf": str(args.base_hf.resolve()),
        }
    else:
        model, tokenizer, controller, adapter_manifest = load_model_and_adapter(
            args.base_hf, args.adapter_checkpoint, device=device
        )
        adapter_manifest["enabled"] = True
    objective = stage_a_eval.load_objective(
        args.v1_checkpoint, args.whispervq_model, device
    ).eval().requires_grad_(False)
    codec = BiCodecTokenizer(model_dir=args.bicodec_model, device=device)
    codec.model.eval().requires_grad_(False)
    return model, tokenizer, controller, adapter_manifest, objective, codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decision-chunk-ms", type=int, choices=CHUNKS_MS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--strict-runtime", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--legacy-results", type=Path)
    parser.add_argument("--external-audio-protocol", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--validation-sample-id", action="append", default=[])
    parser.add_argument("--legacy-sample-id", action="append", default=[])
    parser.add_argument("--external-sample-id", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if bool(args.selection) != bool(args.validation_manifest):
        raise ValueError("selection and validation manifest must be provided together")
    if (
        args.selection is None
        and args.legacy_results is None
        and args.external_audio_protocol is None
    ):
        raise ValueError(
            "provide validation selection, legacy results and/or external audio protocol"
        )
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [
        int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]
    ]
    if len(fixed_speaker) != 32:
        raise ValueError("Stage-A fixed speaker must contain 32 tokens")
    rows: list[dict[str, object]] = []
    requested = list(dict.fromkeys(str(value) for value in args.sample_id))
    explicit_sources = any(
        (args.validation_sample_id, args.legacy_sample_id, args.external_sample_id)
    )
    if requested and explicit_sources:
        raise ValueError("do not mix --sample-id with source-specific sample IDs")
    active_sources = sum(
        value is not None
        for value in (args.selection, args.legacy_results, args.external_audio_protocol)
    )
    if requested and active_sources > 1:
        raise ValueError(
            "use source-specific sample IDs when combining multiple input sources"
        )
    if args.selection is not None:
        rows.extend(
            _selection_rows(
                args.selection,
                args.validation_manifest,  # type: ignore[arg-type]
                args.validation_sample_id if explicit_sources else requested,
            )
        )
    if args.legacy_results is not None:
        rows.extend(
            _legacy_rows(
                args.legacy_results,
                fixed_speaker,
                args.legacy_sample_id if explicit_sources else requested,
            )
        )
    if args.external_audio_protocol is not None:
        rows.extend(
            _external_rows(
                args.external_audio_protocol,
                fixed_speaker,
                args.external_sample_id if explicit_sources else requested,
            )
        )
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique.setdefault(str(row["id"]), row)
    rows = list(unique.values())
    if not rows:
        raise ValueError("strict cascade sample selection is empty")
    for row in rows:
        row["_stage_a_fixed_speaker_global"] = fixed_speaker

    runtime = _load_runtime(args.strict_runtime)
    model, tokenizer, controller, adapter_manifest, objective, codec = _load_models(args)
    original_generate = runtime.generate

    def routed_generate(*call_args, **call_kwargs):
        prompt = call_kwargs.get("prompt_ids")
        if prompt is None:
            raise ValueError("routed strict cascade requires explicit prompt_ids")
        enabled = args.adapter_checkpoint is not None and _route_for_prompt(prompt)
        with controller.route(enabled):
            return original_generate(*call_args, **call_kwargs)

    runtime.generate = routed_generate
    args.output.mkdir(parents=True)
    results: list[Mapping[str, object]] = []
    try:
        for index, row in enumerate(rows):
            print(
                f"run={args.run_id} chunk={args.decision_chunk_ms} sample={row['id']}",
                flush=True,
            )
            value = runtime.evaluate_sample(
                row,
                decision_chunk_ms=args.decision_chunk_ms,
                model=model,
                tokenizer=tokenizer,
                objective=objective,
                bicodec=codec,
                output=args.output,
                seed=20260825 + args.decision_chunk_ms * 100 + index * 1_000_000,
            )
            if bool(row.get("_unreferenced_external")):
                value.update(
                    {
                        "reference_transcription": None,
                        "reference_translation": None,
                        "asr_metric": None,
                        "asr_errors": None,
                        "asr_reference_units": None,
                        "asr_error_rate": None,
                        "external_audio_unreferenced": True,
                    }
                )
            results.append(value)
            print(
                json.dumps(
                    {
                        key: value[key]
                        for key in (
                            "sample_id",
                            "first_audio_source_ms",
                            "audio_writes",
                            "asr_error_rate",
                            "strict_streaming_runtime_passed",
                        )
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        runtime.generate = original_generate
        controller.close()
    payload = {
        "schema_version": "uniss_stagea_joint_grpo_strict_cascade_v1",
        "status": "complete",
        "run_id": args.run_id,
        "decision_chunk_ms": args.decision_chunk_ms,
        "physical_acoustic_block_ms": runtime.PHYSICAL_BLOCK_MS,
        "adapter_manifest": adapter_manifest,
        "route_semantics": "adapter_off_asr_on_mt_tts; baseline_off_all_families",
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
