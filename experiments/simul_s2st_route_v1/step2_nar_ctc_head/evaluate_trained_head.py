#!/usr/bin/env python3
"""Unit-decode probe for a Megatron-trained duration-anchored NAR CTC head.

Loads only ``head.*`` weights from a torch.distributed checkpoint (Qwen stays the
frozen Phase3 HF export used in training). Best-case setting: teacher-forced
reference translation + duration frame budget. Compared against Step 2b's
all-blank V6 head.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "third_party" / "Megatron-LM")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.evaluate_existing_head import (
    blank_suppressed_decode,
    ctc_greedy_decode,
    edit_distance,
    summarise,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.teacher_forced import (
    target_text_hidden,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder

SCHEMA_VERSION = "simul_s2st_route_v1_step2_trained_nar_head_v1"
DIRECTIONS = ("zh2en", "en2zh")


def load_head_from_distcp(
    checkpoint_dir: Path,
    *,
    device: torch.device,
) -> DurationAnchoredCausalNARCTC:
    head = DurationAnchoredCausalNARCTC(
        qwen_hidden_size=896,
        model_size=512,
        semantic_vocab_size=c.BICODEC_SEMANTIC_SIZE,
        frames_per_second=75.0,
        max_frames=1500,
    )
    state = {f"head.{key}": value for key, value in head.state_dict().items()}
    reader = FileSystemReader(str(checkpoint_dir))
    _load_state_dict(state_dict=state, storage_reader=reader, no_dist=True)
    loaded = {key[len("head.") :]: value for key, value in state.items()}
    missing, unexpected = head.load_state_dict(loaded, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"head load mismatch missing={missing} unexpected={unexpected}")
    return head.to(device).eval()


def _lang_family(code: str) -> str:
    value = code.lower().replace("_", "-")
    if value.startswith(("zh", "cmn", "yue")):
        return "zh"
    if value.startswith(("en", "eng")):
        return "en"
    return value


def direction_of(record: dict[str, object]) -> str:
    src = _lang_family(str(record["src_lang"]))
    tgt = _lang_family(str(record["tgt_lang"]))
    if src == "zh" and tgt == "en":
        return "zh2en"
    if src == "en" and tgt == "zh":
        return "en2zh"
    raise ValueError(f"unsupported direction {record['src_lang']}->{record['tgt_lang']}")


def select_records(
    manifest: Path,
    *,
    samples_per_direction: int,
    min_audio_seconds: float,
    max_audio_seconds: float,
    max_unit_tokens: int,
) -> dict[str, list[dict[str, object]]]:
    buckets = {name: [] for name in DIRECTIONS}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if all(len(buckets[name]) >= samples_per_direction for name in DIRECTIONS):
                break
            record = json.loads(line)
            seconds = float(record.get("source_duration_ms", 0)) / 1000.0
            units = record.get("target_bicodec") or []
            if not (min_audio_seconds <= seconds <= max_audio_seconds):
                continue
            if not units or len(units) > max_unit_tokens:
                continue
            if not record.get("translation") or not record.get("source_glm"):
                continue
            name = direction_of(record)
            if len(buckets[name]) >= samples_per_direction:
                continue
            buckets[name].append(record)
    for name, rows in buckets.items():
        if len(rows) < samples_per_direction:
            raise RuntimeError(
                f"only {len(rows)} usable {name} rows in {manifest} "
                f"(need {samples_per_direction})"
            )
    return buckets


@torch.inference_mode()
def score_record(
    qwen,
    text_encoder,
    head: DurationAnchoredCausalNARCTC,
    record: dict[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    text_hidden, text_lengths, _ = target_text_hidden(
        qwen,
        text_encoder,
        source_glm=[int(value) for value in record["source_glm"]],
        bicodec_global=[int(value) for value in record["bicodec_global"]],
        tgt_lang=str(record["tgt_lang"]),
        translation=str(record["translation"]),
        target_bicodec=[int(value) for value in record["target_bicodec"]],
        source_id=str(record["id"]),
        device=device,
    )
    duration = torch.tensor(
        [int(record["source_duration_ms"])], dtype=torch.long, device=device
    )
    reference = [int(value) for value in record["target_bicodec"]]
    unit_lengths = torch.tensor([len(reference)], dtype=torch.long, device=device)
    unit_repeats = torch.tensor(
        [sum(1 for left, right in zip(reference, reference[1:]) if left == right)],
        dtype=torch.long,
        device=device,
    )
    speaker = torch.tensor(
        [[int(value) for value in record["bicodec_global"]]],
        dtype=torch.long,
        device=device,
    )
    speaker_lengths = torch.tensor([speaker.shape[1]], dtype=torch.long, device=device)
    source_glm = torch.tensor(
        [[int(value) for value in record["source_glm"]]],
        dtype=torch.long,
        device=device,
    )
    source_glm_lengths = torch.tensor(
        [source_glm.shape[1]], dtype=torch.long, device=device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, frame_lengths = head(
            text_hidden,
            text_lengths,
            duration,
            unit_lengths=unit_lengths,
            unit_repeats=unit_repeats,
            speaker_ids=speaker,
            speaker_lengths=speaker_lengths,
            source_glm=source_glm,
            source_glm_lengths=source_glm_lengths,
        )
    frames = int(frame_lengths[0])
    active = logits[0, :frames].float()
    probabilities = active.softmax(dim=-1)
    predicted = ctc_greedy_decode(active, head.blank_id)
    suppressed = blank_suppressed_decode(active, head.blank_id)
    blank_fraction = float((active.argmax(dim=-1) == head.blank_id).float().mean())
    distance = edit_distance(predicted, reference)
    suppressed_distance = edit_distance(suppressed, reference)
    required = len(reference) + int(unit_repeats[0])
    return {
        "id": str(record["id"]),
        "direction": direction_of(record),
        "reference_units": len(reference),
        "predicted_units": len(predicted),
        "blank_suppressed_units": len(suppressed),
        "edit_distance": distance,
        "blank_suppressed_edit_distance": suppressed_distance,
        "length_ratio": len(predicted) / max(1, len(reference)),
        "distinct_predicted": len(set(predicted)),
        "distinct_reference": len(set(reference)),
        "blank_suppressed_distinct": len(set(suppressed)),
        "blank_fraction": blank_fraction,
        "lattice_occupancy": required / max(1, frames),
        "ctc_infeasible": required > frames,
        "frames": frames,
        "mean_blank_probability": float(probabilities[:, head.blank_id].mean()),
        "mean_best_nonblank_probability": float(
            probabilities[:, : head.blank_id].max(dim=-1).values.mean()
        ),
        "predicted_preview": predicted[:32],
        "reference_preview": reference[:32],
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Step 2 — trained duration-anchored NAR CTC head decode probe",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']}",
        "",
        "Teacher-forced Phase3 hidden + duration frame budget. Compared with Step 2b "
        "(V6 head was all-blank / ~100% UER).",
        "",
        "| Checkpoint | Dir | Samples | UER | Pred units | Ref units | Len ratio | "
        "Empty | Blank frames | Blank-sup UER | Distinct pred |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["results"]:
        for direction in DIRECTIONS:
            block = entry["scores"].get(direction)
            if not block:
                continue
            lines.append(
                f"| `{entry['label']}` | {direction} | {block['samples']} | "
                f"{block['unit_error_rate'] * 100:.1f}% | "
                f"{block['mean_predicted_units']:.0f} | {block['mean_reference_units']:.0f} | "
                f"{block['mean_length_ratio']:.3f} | {block['empty_predictions']} | "
                f"{block['mean_blank_fraction'] * 100:.1f}% | "
                f"{block['blank_suppressed_unit_error_rate'] * 100:.1f}% | "
                f"{block['mean_distinct_predicted']:.1f} |"
            )
    lines += [
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(payload["config"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_valid.jsonl",
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
    )
    parser.add_argument("--samples-per-direction", type=int, default=16)
    parser.add_argument("--min-audio-seconds", type=float, default=2.0)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--max-unit-tokens", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise SystemExit(f"refusing to overwrite {args.output_json} / {args.output_md}")
    if not args.checkpoint:
        raise SystemExit("pass at least one --checkpoint LABEL=PATH")

    device = torch.device(args.device)
    buckets = select_records(
        args.manifest,
        samples_per_direction=args.samples_per_direction,
        min_audio_seconds=args.min_audio_seconds,
        max_audio_seconds=args.max_audio_seconds,
        max_unit_tokens=args.max_unit_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    qwen = (
        AutoModelForCausalLM.from_pretrained(
            args.phase3_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    qwen.requires_grad_(False)

    results = []
    started = time.time()
    for item in args.checkpoint:
        label, path_text = item.split("=", 1)
        checkpoint_dir = Path(path_text)
        head = load_head_from_distcp(checkpoint_dir, device=device)
        scores = {}
        rows_by_dir = {}
        for direction, records in buckets.items():
            rows = [
                score_record(qwen, text_encoder, head, record, device=device)
                for record in records
            ]
            rows_by_dir[direction] = rows
            scores[direction] = summarise(rows)
        pooled = summarise([row for rows in rows_by_dir.values() for row in rows])
        results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint_dir),
                "scores": scores,
                "pooled": pooled,
                "rows": rows_by_dir,
            }
        )
        del head
        torch.cuda.empty_cache()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config": {
            "manifest": str(args.manifest),
            "phase3_model": str(args.phase3_model),
            "samples_per_direction": args.samples_per_direction,
            "min_audio_seconds": args.min_audio_seconds,
            "max_audio_seconds": args.max_audio_seconds,
            "checkpoints": args.checkpoint,
        },
        "results": [
            {
                "label": entry["label"],
                "checkpoint": entry["checkpoint"],
                "scores": entry["scores"],
                "pooled": entry["pooled"],
                "examples": {
                    direction: entry["rows"][direction][:2]
                    for direction in DIRECTIONS
                },
            }
            for entry in results
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"wrote": str(args.output_json), "pooled": results[-1]["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
