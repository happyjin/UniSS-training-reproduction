#!/usr/bin/env python3
"""Step 2b - is the NAR CTC head already inside joint V6 worth keeping?

The plan treats the NAR BiCodec CTC head as something to build, but joint V6 already trains
one (`NARBiCodecCTC`, hanging off the Qwen target-text hidden states) through Stage A and
Stage B. Before spending a training cycle building a replacement, measure what the existing
head produces.

This gives the head its best case on purpose: the frozen Phase3 backbone, teacher GLM source
tokens and the reference translation, decoded under ordinary causal attention rather than the
policy-conditioned mask used in training. If it cannot emit a sensible BiCodec stream here, it
will not do better under streaming constraints.

Reported against two references that make the numbers readable: a blank-only decoder (the
degenerate solution CTC falls into when it has learnt nothing) and the reference stream's own
token statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.evaluate import (  # noqa: E402
    DIRECTIONS,
    read_records,
)
from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.loader import (  # noqa: E402
    load_joint_checkpoint,
)
from training import constants_uniss as c  # noqa: E402
from training.generate_unist_eval_audio import load_hf_text_encoder  # noqa: E402
from training.phase3_whisper_streamspeech_joint.config import MultiChunkConfig  # noqa: E402
from training.phase3_whisper_streamspeech_joint.model import (  # noqa: E402
    Phase3WhisperStreamSpeechJointModel,
)
from training.phase3_whisper_streamspeech_joint.phase3_batch import (  # noqa: E402
    gather_target_hidden,
)
from training.sample_builders import build_performance_sample  # noqa: E402

SCHEMA_VERSION = "simul_s2st_route_v1_step2b_existing_nar_head_v1"


def edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Levenshtein distance, one vectorised pass per row.

    Deletion and substitution only read the previous row, so they vectorise directly.
    Insertion is the sequential term, but because it always costs exactly one,
    ``row[j] = min(base[j], row[j-1] + 1)`` unrolls to ``min over k<=j of base[k] + (j - k)``,
    which is a running minimum of ``base[k] - k``. That keeps whole 3000-token streams in
    numpy instead of a Python loop over every cell.
    """

    if not left:
        return len(right)
    if not right:
        return len(left)
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    offsets = np.arange(len(right_array) + 1, dtype=np.int64)
    previous = offsets.copy()
    for index, token in enumerate(left_array):
        base = np.empty_like(previous)
        base[0] = index + 1
        np.minimum(
            previous[1:] + 1,
            previous[:-1] + (token != right_array),
            out=base[1:],
        )
        previous = np.minimum.accumulate(base - offsets) + offsets
    return int(previous[-1])


def collapse(ids: Sequence[int], blank_id: int | None) -> list[int]:
    output: list[int] = []
    previous = None
    for value in ids:
        if value != previous and value != blank_id:
            output.append(int(value))
        previous = value
    return output


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int) -> list[int]:
    return collapse(logits.argmax(dim=-1).tolist(), blank_id)


def blank_suppressed_decode(logits: torch.Tensor, blank_id: int) -> list[int]:
    """Decode ignoring the blank column entirely.

    An all-blank greedy output can mean two different things: the head learnt nothing, or it
    learnt the content but sits behind an overwhelming blank prior (the usual CTC collapse
    when most of the lattice is padding). Scoring the best non-blank token per frame
    separates the two, because only the second one produces a stream that resembles the
    reference.
    """

    masked = logits.clone()
    masked[:, blank_id] = float("-inf")
    return collapse(masked.argmax(dim=-1).tolist(), None)


@torch.inference_mode()
def predict_units(
    phase3,
    head,
    text_encoder,
    record,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Teacher-forced hidden states -> NAR head -> greedy CTC units."""

    sample = build_performance_sample(
        source_glm=record.teacher_glm,
        bicodec_global=record.bicodec_global,
        tgt_lang=record.tgt_lang,
        target_bicodec=record.target_bicodec,
        translation=record.translation,
        text_encoder=text_encoder,
        source_id=record.sample_id,
    )
    start, end = sample.segment_spans["performance_translation_text"]
    translation_ids = sample.target_ids[start:end]
    input_ids = torch.tensor(
        [[*sample.prompt_ids, *translation_ids]], dtype=torch.long, device=device
    )
    hidden = phase3(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    positions = torch.arange(
        sample.prompt_length,
        sample.prompt_length + len(translation_ids),
        device=device,
    ).unsqueeze(0)
    target_hidden, text_lengths = gather_target_hidden(hidden, positions)
    logits, output_lengths = head(target_hidden.to(head.output.weight.dtype), text_lengths)
    frames = int(output_lengths[0])
    active = logits[0, :frames].float()
    probabilities = active.softmax(dim=-1)
    return {
        "units": ctc_greedy_decode(active, head.blank_id),
        "blank_suppressed_units": blank_suppressed_decode(active, head.blank_id),
        "frames": frames,
        "text_length": len(translation_ids),
        "mean_blank_probability": float(probabilities[:, head.blank_id].mean()),
        "mean_best_nonblank_probability": float(
            probabilities[:, : head.blank_id].max(dim=-1).values.mean()
        ),
    }


def summarise(rows: Sequence[dict]) -> dict[str, object]:
    if not rows:
        return {}
    reference_units = sum(int(row["reference_units"]) for row in rows)
    return {
        "samples": len(rows),
        "unit_error_rate": sum(int(row["edit_distance"]) for row in rows)
        / max(1, reference_units),
        "mean_predicted_units": float(np.mean([row["predicted_units"] for row in rows])),
        "mean_reference_units": float(np.mean([row["reference_units"] for row in rows])),
        "mean_length_ratio": float(np.mean([row["length_ratio"] for row in rows])),
        "empty_predictions": sum(1 for row in rows if not row["predicted_units"]),
        "mean_distinct_predicted": float(np.mean([row["distinct_predicted"] for row in rows])),
        "mean_distinct_reference": float(np.mean([row["distinct_reference"] for row in rows])),
        "mean_blank_fraction": float(np.mean([row["blank_fraction"] for row in rows])),
        "mean_lattice_occupancy": float(np.mean([row["lattice_occupancy"] for row in rows])),
        "ctc_infeasible": sum(1 for row in rows if row["ctc_infeasible"]),
        "mean_blank_probability": float(np.mean([row["mean_blank_probability"] for row in rows])),
        "mean_best_nonblank_probability": float(
            np.mean([row["mean_best_nonblank_probability"] for row in rows])
        ),
        "blank_suppressed_unit_error_rate": sum(
            int(row["blank_suppressed_edit_distance"]) for row in rows
        )
        / max(1, reference_units),
        "mean_blank_suppressed_units": float(
            np.mean([row["blank_suppressed_units"] for row in rows])
        ),
        "mean_blank_suppressed_distinct": float(
            np.mean([row["blank_suppressed_distinct"] for row in rows])
        ),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Step 2b — the NAR CTC head already trained inside joint V6",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']} · research only.",
        "",
        "Best case for the head: frozen Phase3 backbone, teacher GLM source tokens, reference "
        "translation, ordinary causal attention. "
        f"{payload['config']['samples_per_direction']} samples per direction.",
        "",
        "## 1. Does the head emit anything?",
        "",
        "| Checkpoint | Dir | Samples | Unit error rate | Predicted units | Reference units | "
        "Length ratio | Empty | Blank frames | CTC infeasible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{block['mean_blank_fraction'] * 100:.1f}% | {block['ctc_infeasible']} |"
            )
    lines += [
        "",
        "A unit error rate at or above 100% with a near-empty prediction is the degenerate "
        "all-blank CTC solution — the head has not learnt to emit units at all. Values "
        "meaningfully below 100% with a length ratio near 1.0 mean it has.",
        "",
        "## 2. Is there any signal under the blank prior?",
        "",
        "`blank suppressed` decodes the best non-blank token per frame. If the head learnt the "
        "content but mis-calibrated its blank prior, this stream resembles the reference and "
        "the collapse is a loss-balance problem. If it stays at ~100% error with a handful of "
        "distinct tokens, the head learnt nothing and has to be retrained.",
        "",
        "| Checkpoint | Dir | Mean blank prob | Mean best non-blank prob | "
        "Blank-suppressed UER | Blank-suppressed units | Distinct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["results"]:
        for direction in DIRECTIONS:
            block = entry["scores"].get(direction)
            if not block:
                continue
            lines.append(
                f"| `{entry['label']}` | {direction} | "
                f"{block['mean_blank_probability']:.4f} | "
                f"{block['mean_best_nonblank_probability']:.4f} | "
                f"{block['blank_suppressed_unit_error_rate'] * 100:.1f}% | "
                f"{block['mean_blank_suppressed_units']:.0f} | "
                f"{block['mean_blank_suppressed_distinct']:.1f} |"
            )
    lines += [
        "",
        "## 3. Vocabulary use",
        "",
        "| Checkpoint | Dir | Distinct predicted | Distinct reference | Lattice occupancy |",
        "|---|---|---:|---:|---:|",
    ]
    for entry in payload["results"]:
        for direction in DIRECTIONS:
            block = entry["scores"].get(direction)
            if not block:
                continue
            lines.append(
                f"| `{entry['label']}` | {direction} | "
                f"{block['mean_distinct_predicted']:.1f} | "
                f"{block['mean_distinct_reference']:.1f} | "
                f"{block['mean_lattice_occupancy'] * 100:.1f}% |"
            )
    lines += [
        "",
        "## 4. Configuration",
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
        / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl",
    )
    parser.add_argument(
        "--whisper-model", type=Path, default=ROOT / "pretrained_models/UniSS/glm4_tokenizer"
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
    )
    parser.add_argument(
        "--tokenizer-map-dir",
        type=Path,
        default=ROOT
        / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/tokenizer_maps",
    )
    parser.add_argument("--samples-per-direction", type=int, default=16)
    parser.add_argument("--min-audio-seconds", type=float, default=2.0)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output_json, args.output_md):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite Step 2b report: {output}")
    checkpoints = []
    for entry in args.checkpoint:
        label, separator, path = entry.partition("=")
        if not separator:
            raise ValueError(f"--checkpoint expects LABEL=PATH, got: {entry}")
        checkpoints.append((label, Path(path)))
    if not checkpoints:
        raise ValueError("at least one --checkpoint is required")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = read_records(
        args.manifest,
        per_direction=args.samples_per_direction,
        max_audio_seconds=args.max_audio_seconds,
        min_audio_seconds=args.min_audio_seconds,
    )
    print(json.dumps({"stage": "selected", "samples": len(records)}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.phase3_model), local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    phase3 = (
        AutoModelForCausalLM.from_pretrained(
            str(args.phase3_model), local_files_only=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    phase3.requires_grad_(False)

    model = Phase3WhisperStreamSpeechJointModel.from_pretrained(
        whisper_path=args.whisper_model,
        phase3_model=args.phase3_model,
        tokenizer_map_dir=args.tokenizer_map_dir,
        chunk_config=MultiChunkConfig(chunk_ms=(320, 640, 960, 1280, None), right_context_ms=80),
        upsample_ratio=48,
        gradient_checkpointing=False,
        bridge_surrogate="topk_soft",
        bridge_topk=8,
        bridge_temperature=0.1,
        teacher_temperature=0.1,
        freeze_whisper_codebook=True,
        freeze_whisper_post_vq=True,
        trainable_whisper_pre_vq_layers=1,
    ).eval()
    model.requires_grad_(False)
    model.to(device)

    results = []
    checkpoint_reports = []
    for label, path in checkpoints:
        report = load_joint_checkpoint(model, path)
        checkpoint_reports.append({"label": label, **report.to_dict()})
        rows = []
        for record in records:
            prediction = predict_units(
                phase3, model.unit_ctc, text_encoder, record, device=device
            )
            units = prediction["units"]
            suppressed = prediction["blank_suppressed_units"]
            frames = int(prediction["frames"])
            reference = [int(value) for value in record.target_bicodec]
            repeats = sum(1 for a, b in zip(reference, reference[1:]) if a == b)
            rows.append(
                {
                    "id": record.sample_id,
                    "direction": record.direction,
                    "predicted_units": len(units),
                    "reference_units": len(reference),
                    "edit_distance": edit_distance(units, reference),
                    "blank_suppressed_units": len(suppressed),
                    "blank_suppressed_edit_distance": edit_distance(suppressed, reference),
                    "blank_suppressed_distinct": len(set(suppressed)),
                    "length_ratio": len(units) / max(1, len(reference)),
                    "distinct_predicted": len(set(units)),
                    "distinct_reference": len(set(reference)),
                    "frames": frames,
                    "text_length": int(prediction["text_length"]),
                    "mean_blank_probability": float(prediction["mean_blank_probability"]),
                    "mean_best_nonblank_probability": float(
                        prediction["mean_best_nonblank_probability"]
                    ),
                    "blank_fraction": 1.0 - len(units) / max(1, frames),
                    "lattice_occupancy": (len(reference) + repeats) / max(1, frames),
                    "ctc_infeasible": bool(len(reference) + repeats > frames),
                }
            )
        scores = {
            direction: summarise([row for row in rows if row["direction"] == direction])
            for direction in DIRECTIONS
        }
        results.append({"label": label, "scores": scores, "samples": rows})
        print(
            json.dumps(
                {
                    "stage": "checkpoint_done",
                    "label": label,
                    "scores": {
                        direction: {
                            "uer": round(float(block["unit_error_rate"]), 4),
                            "predicted": round(float(block["mean_predicted_units"]), 1),
                            "reference": round(float(block["mean_reference_units"]), 1),
                        }
                        for direction, block in scores.items()
                        if block
                    },
                }
            ),
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "run_name": args.run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "manifest": str(args.manifest),
            "phase3_model": str(args.phase3_model),
            "samples_per_direction": args.samples_per_direction,
            "total_samples": len(records),
            "min_audio_seconds": args.min_audio_seconds,
            "max_audio_seconds": args.max_audio_seconds,
            "upsample_ratio": int(model.unit_ctc.upsample_ratio),
            "blank_id": int(model.unit_ctc.blank_id),
            "device": str(device),
        },
        "checkpoints": checkpoint_reports,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"stage": "done", "report": str(args.output_md)}))


if __name__ == "__main__":
    main()
