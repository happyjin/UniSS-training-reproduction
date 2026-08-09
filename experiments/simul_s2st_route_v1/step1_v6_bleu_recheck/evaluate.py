#!/usr/bin/env python3
"""Step 1 (D1) - does joint-V6 lose downstream BLEU where it loses teacher agreement?

Stage B was stopped by a safety gate that watches teacher GLM agreement, and the failure
analysis argued that agreement is both unreachable and a poor proxy for downstream quality.
That argument is only actionable if it is measured: for each checkpoint this runs the V6
frontend (WhisperVQ + STE bridge) to a source GLM stream, feeds that stream to the *frozen*
Phase3 export, and reports bidirectional Text-BLEU next to the agreement of the same stream.

The Phase3 backend is held fixed across all checkpoints on purpose. That is what makes the
numbers comparable to the existing sensitivity table (released 33.45 / 26.61, exact
prefix-causal 80 ms 31.22 / 25.21, Student v2 21.13 / 15.32).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sacrebleu  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from evaluation.uniss_outputs import parse_with_tokenizer  # noqa: E402
from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.loader import (  # noqa: E402
    backbone_drift,
    load_joint_checkpoint,
)
from training import constants_uniss as c  # noqa: E402
from training.generate_unist_eval_audio import load_hf_text_encoder  # noqa: E402
from training.phase3_whisper_streamspeech_joint.config import MultiChunkConfig  # noqa: E402
from training.phase3_whisper_streamspeech_joint.model import (  # noqa: E402
    Phase3WhisperStreamSpeechJointModel,
)
from training.sample_builders import build_performance_sample  # noqa: E402

SCHEMA_VERSION = "simul_s2st_route_v1_step1_v6_bleu_recheck_v1"
SAMPLE_RATE = 16_000
DIRECTIONS = ("eng->cmn", "cmn->eng")
TEACHER_STREAM = "manifest_teacher_glm"
PRETRAINED_STREAM = "pretrained_frontend"


@dataclass
class Record:
    index: int
    sample_id: str
    direction: str
    source_audio: str
    translation: str
    bicodec_global: list[int]
    target_bicodec: list[int]
    tgt_lang: str
    teacher_glm: list[int]
    duration_ms: float
    waveform: torch.Tensor = field(repr=False, default=None)


def read_records(
    manifest: Path,
    *,
    per_direction: int,
    max_audio_seconds: float,
    min_audio_seconds: float,
) -> list[Record]:
    by_direction: dict[str, list[dict]] = {direction: [] for direction in DIRECTIONS}
    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            direction = f"{row['src_lang']}->{row['tgt_lang']}"
            if direction not in by_direction:
                continue
            seconds = float(row.get("source_duration_ms", 0.0)) / 1000.0
            if not min_audio_seconds <= seconds <= max_audio_seconds:
                continue
            row["_index"] = index
            by_direction[direction].append(row)

    chosen: list[Record] = []
    for direction, rows in by_direction.items():
        if len(rows) < per_direction:
            raise RuntimeError(
                f"{direction} has only {len(rows)} rows inside the duration window, "
                f"need {per_direction}"
            )
        stride = max(1, len(rows) // per_direction)
        for row in rows[::stride][:per_direction]:
            chosen.append(
                Record(
                    index=int(row["_index"]),
                    sample_id=str(row["id"]),
                    direction=direction,
                    source_audio=str(row["source_audio"]),
                    translation=str(row["translation"]),
                    bicodec_global=[int(value) for value in row["bicodec_global"]],
                    target_bicodec=[int(value) for value in row["target_bicodec"]],
                    tgt_lang=str(row["tgt_lang"]),
                    teacher_glm=[int(value) for value in row["source_glm"]],
                    duration_ms=float(row.get("source_duration_ms", 0.0)),
                )
            )
    chosen.sort(key=lambda item: (item.direction, item.index))
    return chosen


def load_waveforms(records: Sequence[Record]) -> None:
    for record in records:
        waveform, sample_rate = torchaudio.load(record.source_audio)
        waveform = waveform[:1]
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        record.waveform = waveform.squeeze(0).contiguous()


def agreement(stream: Sequence[int], teacher: Sequence[int]) -> dict[str, float]:
    overlap = min(len(stream), len(teacher))
    if overlap == 0:
        return {"position_agreement": 0.0, "length_ratio": 0.0, "compared_positions": 0}
    matches = sum(1 for left, right in zip(stream[:overlap], teacher[:overlap]) if left == right)
    return {
        "position_agreement": matches / overlap,
        "length_ratio": len(stream) / len(teacher),
        "compared_positions": overlap,
    }


@torch.inference_mode()
def frontend_stream(
    model: Phase3WhisperStreamSpeechJointModel,
    record: Record,
    *,
    chunk_ms: int | None,
    device: torch.device,
) -> list[int]:
    waveform = record.waveform.to(device).unsqueeze(0)
    lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        whisper = model.whisper(waveform, lengths, chunk_ms=chunk_ms)
        bridge = model.bridge(whisper.pre_vq_hidden, whisper.token_lengths)
    length = int(whisper.token_lengths[0])
    return [int(value) for value in bridge.hard_code_ids[0, :length].tolist()]


@torch.inference_mode()
def generate_translation(
    phase3,
    tokenizer,
    text_encoder,
    record: Record,
    source_glm: Sequence[int],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    if not source_glm:
        return ""
    sample = build_performance_sample(
        source_glm=list(source_glm),
        bicodec_global=record.bicodec_global,
        tgt_lang=record.tgt_lang,
        target_bicodec=record.target_bicodec,
        translation=record.translation,
        text_encoder=text_encoder,
        source_id=record.sample_id,
    )
    prompt = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
    suppressed = list(range(c.VOCAB_SIZE, int(phase3.config.vocab_size)))
    generated = phase3.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=c.TOKEN_PAD,
        eos_token_id=c.TOKEN_EOS,
        suppress_tokens=suppressed or None,
    )
    tail = generated[0, prompt.shape[1] :].tolist()
    parsed = parse_with_tokenizer(tail, mode="performance", tokenizer=tokenizer)
    return str(parsed.get("generated_translation") or "").strip()


def score_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scores: dict[str, object] = {}
    for direction in DIRECTIONS:
        subset = [row for row in rows if row["direction"] == direction]
        if not subset:
            continue
        hypotheses = [str(row["hypothesis"]) for row in subset]
        references = [[str(row["reference"]) for row in subset]]
        tokenize = "zh" if direction == "eng->cmn" else "13a"
        scores[direction] = {
            "samples": len(subset),
            "text_bleu": sacrebleu.corpus_bleu(hypotheses, references, tokenize=tokenize).score,
            "chrf": sacrebleu.corpus_chrf(hypotheses, references).score,
            "empty_hypotheses": sum(1 for value in hypotheses if not value),
            "position_agreement": statistics.fmean(
                float(row["position_agreement"]) for row in subset
            ),
            "length_ratio": statistics.fmean(float(row["length_ratio"]) for row in subset),
        }
    return scores


def parse_chunk(value: str) -> int | None:
    return None if value.lower() in {"offline", "none", "full"} else int(value)


def chunk_label(value: int | None) -> str:
    return "offline" if value is None else f"{value}ms"


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Step 1 (D1) — joint-V6 checkpoints under a frozen Phase3 BLEU probe",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']} · research only.",
        "",
        f"Backend held fixed at `{payload['config']['phase3_model']}` for every row. "
        f"{payload['config']['samples_per_direction']} samples per direction "
        f"({payload['config']['total_samples']} total) from "
        f"`{Path(str(payload['config']['manifest'])).name}`.",
        "",
        "## 1. Agreement against downstream BLEU",
        "",
        "| Stream | Chunk | Agreement EN→ZH | BLEU EN→ZH | Agreement ZH→EN | BLEU ZH→EN |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in payload["results"]:
        scores = entry["scores"]
        eng = scores.get("eng->cmn", {})
        cmn = scores.get("cmn->eng", {})
        lines.append(
            f"| `{entry['label']}` | {entry['chunk']} | "
            f"{_pct(eng.get('position_agreement'))} | {_num(eng.get('text_bleu'))} | "
            f"{_pct(cmn.get('position_agreement'))} | {_num(cmn.get('text_bleu'))} |"
        )
    lines += [
        "",
        "## 2. Detail",
        "",
        "| Stream | Chunk | Dir | Samples | BLEU | chrF | Agreement | Length ratio | Empty hyp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["results"]:
        for direction in DIRECTIONS:
            block = entry["scores"].get(direction)
            if not block:
                continue
            lines.append(
                f"| `{entry['label']}` | {entry['chunk']} | {direction} | {block['samples']} | "
                f"{block['text_bleu']:.2f} | {block['chrf']:.2f} | "
                f"{block['position_agreement'] * 100:.2f}% | {block['length_ratio']:.3f} | "
                f"{block['empty_hypotheses']} |"
            )
    drifts = [entry for entry in payload["checkpoints"] if entry.get("backbone_drift")]
    if drifts:
        lines += [
            "",
            "## 3. Did the checkpoint's own Qwen move?",
            "",
            "| Checkpoint | Iteration | Loaded tensors | Changed Qwen tensors | Max abs delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for entry in drifts:
            drift = entry["backbone_drift"]
            lines.append(
                f"| `{entry['label']}` | {entry['iteration']} | {entry['loaded_tensors']} | "
                f"{drift['changed_tensors']}/{drift['compared_tensors']} | "
                f"{drift['max_abs_delta']:.4g} |"
            )
    lines += ["", "## 4. Configuration", "", "```json", json.dumps(payload["config"], indent=2), "```", ""]
    return "\n".join(lines)


def _progress(entry: Mapping[str, object]) -> dict[str, object]:
    scores = entry["scores"]
    return {
        "stage": "stream_done",
        "label": entry["label"],
        "chunk": entry["chunk"],
        "scores": {
            direction: {
                "text_bleu": round(float(block["text_bleu"]), 2),
                "agreement": round(float(block["position_agreement"]), 4),
            }
            for direction, block in scores.items()  # type: ignore[union-attr]
        },
    }


def _num(value: object) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _pct(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="joint-V6 iter_XXXXXXXX directory to probe; repeatable",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl",
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
    parser.add_argument("--chunks", nargs="+", default=["320", "offline"])
    parser.add_argument("--samples-per-direction", type=int, default=16)
    parser.add_argument("--min-audio-seconds", type=float, default=2.0)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-pretrained-baseline",
        action="store_true",
        help="omit the untrained-frontend control row",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output_json, args.output_md):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite Step 1 report: {output}")
    checkpoints: list[tuple[str, Path]] = []
    for entry in args.checkpoint:
        label, separator, path = entry.partition("=")
        if not separator:
            raise ValueError(f"--checkpoint expects LABEL=PATH, got: {entry}")
        checkpoints.append((label, Path(path)))

    chunks = [parse_chunk(value) for value in args.chunks]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    records = read_records(
        args.manifest,
        per_direction=args.samples_per_direction,
        max_audio_seconds=args.max_audio_seconds,
        min_audio_seconds=args.min_audio_seconds,
    )
    load_waveforms(records)
    print(
        json.dumps(
            {
                "stage": "selected",
                "samples": len(records),
                "per_direction": args.samples_per_direction,
                "checkpoints": [label for label, _ in checkpoints],
                "chunks": [chunk_label(value) for value in chunks],
            }
        ),
        flush=True,
    )

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

    def run_stream(label: str, chunk: int | None, streams: dict[str, list[int]]) -> dict[str, object]:
        rows = []
        for record in records:
            stream = streams[record.sample_id]
            hypothesis = generate_translation(
                phase3,
                tokenizer,
                text_encoder,
                record,
                stream,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append(
                {
                    "id": record.sample_id,
                    "direction": record.direction,
                    "reference": record.translation,
                    "hypothesis": hypothesis,
                    "stream_tokens": len(stream),
                    "teacher_tokens": len(record.teacher_glm),
                    **agreement(stream, record.teacher_glm),
                }
            )
        return {
            "label": label,
            "chunk": chunk_label(chunk),
            "scores": score_rows(rows),
            "samples": rows,
        }

    results: list[dict[str, object]] = []
    teacher_streams = {record.sample_id: record.teacher_glm for record in records}
    results.append(run_stream(TEACHER_STREAM, None, teacher_streams))
    print(json.dumps(_progress(results[-1])), flush=True)

    checkpoint_reports: list[dict[str, object]] = []
    if checkpoints:
        # Mirrors experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_8gpu.sh
        # so the module tree matches the saved tensor names exactly; the topk_soft surrogate in
        # particular is what keeps the bridge free of a `continuous_projection` submodule.
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

        # Control: the same frontend before any V6 training. At the offline chunk this should
        # reproduce the manifest teacher stream almost exactly, which validates the whole probe;
        # at 320 ms it isolates the cost of chunking alone from the cost of V6 training.
        if not args.skip_pretrained_baseline:
            for chunk in chunks:
                streams = {
                    record.sample_id: frontend_stream(model, record, chunk_ms=chunk, device=device)
                    for record in records
                }
                results.append(run_stream(PRETRAINED_STREAM, chunk, streams))
                print(json.dumps(_progress(results[-1])), flush=True)

        for label, path in checkpoints:
            report = load_joint_checkpoint(model, path)
            drift = backbone_drift(model.qwen, phase3)
            checkpoint_reports.append(
                {"label": label, "backbone_drift": drift, **report.to_dict()}
            )
            print(
                json.dumps({"stage": "checkpoint_loaded", "label": label, **report.to_dict(),
                            "backbone_drift": drift}),
                flush=True,
            )
            for chunk in chunks:
                streams = {
                    record.sample_id: frontend_stream(
                        model, record, chunk_ms=chunk, device=device
                    )
                    for record in records
                }
                results.append(run_stream(label, chunk, streams))
                print(json.dumps(_progress(results[-1])), flush=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "run_name": args.run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "manifest": str(args.manifest),
            "phase3_model": str(args.phase3_model),
            "whisper_model": str(args.whisper_model),
            "tokenizer_map_dir": str(args.tokenizer_map_dir),
            "samples_per_direction": args.samples_per_direction,
            "total_samples": len(records),
            "min_audio_seconds": args.min_audio_seconds,
            "max_audio_seconds": args.max_audio_seconds,
            "max_new_tokens": args.max_new_tokens,
            "chunks": [chunk_label(value) for value in chunks],
            "device": str(device),
        },
        "selection": [
            {
                "id": record.sample_id,
                "direction": record.direction,
                "manifest_index": record.index,
                "duration_ms": record.duration_ms,
                "teacher_glm_tokens": len(record.teacher_glm),
            }
            for record in records
        ],
        "checkpoints": checkpoint_reports,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "done",
                "rows": len(results),
                "report": str(args.output_md),
                "summary": [
                    {
                        "label": entry["label"],
                        "chunk": entry["chunk"],
                        **{
                            direction: {
                                "bleu": round(float(block["text_bleu"]), 2),
                                "agreement": round(float(block["position_agreement"]), 4),
                            }
                            for direction, block in entry["scores"].items()
                        },
                    }
                    for entry in results
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
